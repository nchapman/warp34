//! Native ARC-1.0 hybrid local/cloud router for `warp34 serve`.
//!
//! Combines:
//! - In-process `nchapman/ARC-1.0-W4A16-HGPTQ` 12-layer encoder (`~300 MiB` VRAM, ~1–4 ms)
//! - Exact Stage 1 (`P(pass)` + 4-leg token usage), Stage 2 (effort selection), and
//!   `signals.pii.payload` (14 PII/credential classes) heads
//! - Native KV-cache boundary awareness (stays local when local prefix cache is warm)
//! - Zero-config OpenRouter / Anthropic / OpenAI / custom HTTP spillover proxy

pub mod encoder;
pub mod heads;
pub mod input;
pub mod policy;
pub mod proxy;
pub mod registry;

use std::sync::Arc;

use axum::http::{HeaderMap, HeaderName, HeaderValue};
use encoder::ArcEncoderWorker;
use heads::ArcHeads;
use input::{ApiProtocol, serialize_request};
use policy::{
    CloudCandidate, DecideInput, RouteTarget, RoutingBasis, RoutingDecision, RoutingPolicy, decide,
    infer_local_slot, resolve_cloud_candidates_from_registry,
};
use proxy::UpstreamProxy;
use registry::ModelsDevRegistry;
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::HashSet,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

pub const DEFAULT_ARC_ROUTER_MODEL: &str = "nchapman/ARC-1.0-W4A16-HGPTQ";

#[derive(Debug, Clone)]
pub struct ArcRouterConfig {
    pub model_dir: PathBuf,
    pub device_id: i32,
    pub local_model_name: String,
    /// Optional forced upstream provider (e.g. `openrouter`, `google`, `anthropic`, `deepseek`,
    /// `https://...`). When `None` or `"auto"`, each cloud model auto-detects its provider and
    /// API key from `models.dev` + environment variables.
    pub upstream: Option<String>,
    pub upstream_api_key: Option<String>,
    pub cloud_model_filter: Vec<String>,
    pub policy: RoutingPolicy,
    pub pin_pii_classes: HashSet<String>,
}

#[derive(Debug, Default)]
pub struct RouterMetrics {
    pub total_requests: AtomicU64,
    pub local_requests: AtomicU64,
    pub cloud_requests: AtomicU64,
    pub pii_pinned_requests: AtomicU64,
    pub warm_kv_stayed_local: AtomicU64,
    pub effort_none: AtomicU64,
    pub effort_low: AtomicU64,
    pub effort_medium: AtomicU64,
    pub effort_high: AtomicU64,
    pub total_router_latency_us: AtomicU64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouterMetricsSnapshot {
    pub total_requests: u64,
    pub local_requests: u64,
    pub cloud_requests: u64,
    pub pii_pinned_requests: u64,
    pub warm_kv_stayed_local: u64,
    pub effort_none: u64,
    pub effort_low: u64,
    pub effort_medium: u64,
    pub effort_high: u64,
    pub mean_router_latency_ms: f64,
}

impl RouterMetrics {
    pub fn record(&self, d: &RoutingDecision) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        match d.target {
            RouteTarget::Local => {
                self.local_requests.fetch_add(1, Ordering::Relaxed);
            }
            RouteTarget::Cloud { .. } => {
                self.cloud_requests.fetch_add(1, Ordering::Relaxed);
            }
        }
        match d.basis {
            RoutingBasis::PiiPinnedLocal => {
                self.pii_pinned_requests.fetch_add(1, Ordering::Relaxed);
            }
            RoutingBasis::WarmKvStayLocal => {
                self.warm_kv_stayed_local.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        match d.reasoning_effort.as_str() {
            "none" => {
                self.effort_none.fetch_add(1, Ordering::Relaxed);
            }
            "low" => {
                self.effort_low.fetch_add(1, Ordering::Relaxed);
            }
            "high" => {
                self.effort_high.fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                self.effort_medium.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.total_router_latency_us
            .fetch_add((d.latency_ms * 1000.0) as u64, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> RouterMetricsSnapshot {
        let total = self.total_requests.load(Ordering::Relaxed);
        let total_us = self.total_router_latency_us.load(Ordering::Relaxed);
        RouterMetricsSnapshot {
            total_requests: total,
            local_requests: self.local_requests.load(Ordering::Relaxed),
            cloud_requests: self.cloud_requests.load(Ordering::Relaxed),
            pii_pinned_requests: self.pii_pinned_requests.load(Ordering::Relaxed),
            warm_kv_stayed_local: self.warm_kv_stayed_local.load(Ordering::Relaxed),
            effort_none: self.effort_none.load(Ordering::Relaxed),
            effort_low: self.effort_low.load(Ordering::Relaxed),
            effort_medium: self.effort_medium.load(Ordering::Relaxed),
            effort_high: self.effort_high.load(Ordering::Relaxed),
            mean_router_latency_ms: if total > 0 {
                (total_us as f64 / total as f64) / 1000.0
            } else {
                0.0
            },
        }
    }
}

pub struct ArcRouter {
    encoder: ArcEncoderWorker,
    heads: Arc<ArcHeads>,
    local_slot: usize,
    cloud_candidates: Vec<CloudCandidate>,
    policy: RoutingPolicy,
    pin_pii_classes: HashSet<String>,
    proxy: Option<UpstreamProxy>,
    metrics: RouterMetrics,
}

impl ArcRouter {
    pub fn load(cfg: ArcRouterConfig) -> Result<Self, String> {
        let heads = ArcHeads::load(&cfg.model_dir)?;
        let local_slot = infer_local_slot(&cfg.local_model_name, &heads.catalog);
        let registry = ModelsDevRegistry::load_cached_or_fetch();
        let wants_cloud = cfg.upstream.is_some() || !cfg.cloud_model_filter.is_empty();
        let cloud_candidates = if wants_cloud {
            resolve_cloud_candidates_from_registry(
                &registry,
                &heads.catalog,
                cfg.upstream.as_deref(),
                cfg.upstream_api_key.as_deref(),
                &cfg.cloud_model_filter,
                |var| std::env::var(var).ok(),
            )?
        } else {
            // Even with bare `--router`, auto-enable cloud spillover if a known provider key exists.
            resolve_cloud_candidates_from_registry(
                &registry,
                &heads.catalog,
                None,
                cfg.upstream_api_key.as_deref(),
                &[],
                |var| std::env::var(var).ok(),
            )
            .unwrap_or_default()
        };
        let proxy = if cloud_candidates.is_empty() {
            None
        } else {
            Some(UpstreamProxy::new()?)
        };
        let encoder = ArcEncoderWorker::spawn(&cfg.model_dir, cfg.device_id)?;
        Ok(Self {
            encoder,
            heads: Arc::new(heads),
            local_slot,
            cloud_candidates,
            policy: cfg.policy,
            pin_pii_classes: cfg.pin_pii_classes,
            proxy,
            metrics: RouterMetrics::default(),
        })
    }

    pub fn local_slot_name(&self) -> &str {
        &self.heads.catalog.models[self.local_slot].name
    }

    pub fn cloud_model_names(&self) -> Vec<String> {
        self.cloud_candidates
            .iter()
            .map(|c| c.wire.clone())
            .collect()
    }

    pub fn cloud_summary(&self) -> String {
        if self.cloud_candidates.is_empty() {
            return "none (local adaptive effort)".to_string();
        }
        let items: Vec<String> = self
            .cloud_candidates
            .iter()
            .map(|c| {
                let prov = c
                    .endpoint
                    .as_ref()
                    .map_or("cloud", |e| e.provider_id.as_str());
                let price = c
                    .price_override
                    .or(self.heads.catalog.model_prices[c.slot])
                    .map(|p| format!(" ${:.2}/${:.2}", p[0], p[3]))
                    .unwrap_or_default();
                format!("{prov}:{}{price}", c.wire)
            })
            .collect();
        format!("[{}]", items.join(", "))
    }

    pub fn metrics(&self) -> &RouterMetrics {
        &self.metrics
    }

    pub fn proxy(&self) -> Option<&UpstreamProxy> {
        self.proxy.as_ref()
    }

    /// Run ARC-1.0 encoding + Stage 1/2/PII heads and return a `RoutingDecision`.
    pub async fn route(
        &self,
        protocol: ApiProtocol,
        body: &Value,
        warm_local_kv: bool,
    ) -> Result<RoutingDecision, String> {
        let t0 = std::time::Instant::now();
        let body_clone = body.clone();
        let canonical =
            tokio::task::spawn_blocking(move || serialize_request(protocol, &body_clone))
                .await
                .map_err(|e| format!("ARC serialize join error: {e}"))?;
        let enc = self.encoder.encode_blocks(canonical.blocks).await?;
        let heads = Arc::clone(&self.heads);
        let local_slot = self.local_slot;
        let cloud_candidates: Vec<CloudCandidate> = if self.proxy.is_some() {
            self.cloud_candidates
                .iter()
                .filter(|c| {
                    c.endpoint
                        .as_ref()
                        .is_some_and(|ep| UpstreamProxy::supports_protocol(ep, protocol))
                })
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        let policy = self.policy;
        let pin_pii_classes = self.pin_pii_classes.clone();
        let has_tools = canonical.has_tools;
        let mut decision = tokio::task::spawn_blocking(move || {
            decide(
                &heads,
                &DecideInput {
                    block_readouts: &enc.block_readouts,
                    block_roles: &enc.block_roles,
                    prompt_tokens: enc.prompt_tokens,
                    has_tools,
                    warm_local_kv,
                    local_slot,
                    cloud_candidates: &cloud_candidates,
                    policy,
                    pin_pii_classes: &pin_pii_classes,
                },
            )
        })
        .await
        .map_err(|e| format!("ARC decide join error: {e}"))??;
        decision.latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.metrics.record(&decision);
        Ok(decision)
    }
}

/// Attach standardized `x-warp34-route*` transparency headers to any HTTP response.
pub fn apply_routing_headers(headers: &mut HeaderMap, decision: &RoutingDecision) {
    let target_str = match &decision.target {
        RouteTarget::Local => "local".to_string(),
        RouteTarget::Cloud { wire, .. } => format!("cloud:{wire}"),
    };
    let pairs = [
        ("x-warp34-route", target_str),
        (
            "x-warp34-route-basis",
            decision.basis.as_header_str().to_string(),
        ),
        ("x-warp34-route-p-pass", format!("{:.4}", decision.p_pass)),
        (
            "x-warp34-route-local-p-pass",
            format!("{:.4}", decision.local_p_pass),
        ),
        ("x-warp34-route-effort", decision.reasoning_effort.clone()),
        (
            "x-warp34-route-latency-ms",
            format!("{:.2}", decision.latency_ms),
        ),
    ];
    for (k, v) in pairs {
        if let (Ok(name), Ok(val)) = (
            HeaderName::from_bytes(k.as_bytes()),
            HeaderValue::from_str(&v),
        ) {
            headers.insert(name, val);
        }
    }
    if !decision.pii_triggered.is_empty()
        && let Ok(val) = HeaderValue::from_str(&decision.pii_triggered.join(","))
    {
        headers.insert(HeaderName::from_static("x-warp34-route-pii"), val);
    }
}
