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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
    pub cloud_model_filter: Vec<String>,
    pub policy: RoutingPolicy,
    pub pin_pii_classes: HashSet<String>,
}

#[derive(Debug, Default, Clone)]
struct ModelRouteStats {
    is_local: bool,
    requests: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    cached_tokens: u64,
    spend_usd: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouterModelStatsSnapshot {
    pub model: String,
    pub is_local: bool,
    pub requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: u64,
    pub spend_usd: f64,
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
    pub router_fallbacks: AtomicU64,
    pub cloud_upstream_errors: AtomicU64,
    pub total_router_latency_us: AtomicU64,
    per_model: Mutex<HashMap<String, ModelRouteStats>>,
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
    pub router_fallbacks: u64,
    pub cloud_upstream_errors: u64,
    pub mean_router_latency_ms: f64,
    pub total_spend_usd: f64,
    pub models: Vec<RouterModelStatsSnapshot>,
}

impl RouterMetrics {
    pub fn record(&self, d: &RoutingDecision) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        let (model_key, is_local) = match &d.target {
            RouteTarget::Local => {
                self.local_requests.fetch_add(1, Ordering::Relaxed);
                ("local".to_string(), true)
            }
            RouteTarget::Cloud { name, .. } => {
                self.cloud_requests.fetch_add(1, Ordering::Relaxed);
                (name.clone(), false)
            }
        };
        if let Ok(mut map) = self.per_model.lock() {
            let entry = map.entry(model_key).or_insert_with(|| ModelRouteStats {
                is_local,
                ..Default::default()
            });
            entry.requests = entry.requests.saturating_add(1);
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

    /// Records actual token usage & USD spend reported by an upstream cloud response.
    /// `prices` is `[input, cache_read, cache_write, output]` in USD per million tokens.
    pub fn record_cloud_usage(
        &self,
        model_name: &str,
        prices: [f64; 4],
        prompt_tokens: u64,
        cached_tokens: u64,
        cache_write_tokens: u64,
        completion_tokens: u64,
    ) {
        let non_cached_in = prompt_tokens.saturating_sub(cached_tokens + cache_write_tokens);
        let spend = (non_cached_in as f64 * prices[0]
            + cached_tokens as f64 * prices[1]
            + cache_write_tokens as f64 * prices[2]
            + completion_tokens as f64 * prices[3])
            * 1e-6;
        if let Ok(mut map) = self.per_model.lock() {
            let entry = map
                .entry(model_name.to_string())
                .or_insert_with(|| ModelRouteStats {
                    is_local: false,
                    ..Default::default()
                });
            entry.prompt_tokens = entry.prompt_tokens.saturating_add(prompt_tokens);
            entry.completion_tokens = entry.completion_tokens.saturating_add(completion_tokens);
            entry.cached_tokens = entry.cached_tokens.saturating_add(cached_tokens);
            if spend.is_finite() && spend > 0.0 {
                entry.spend_usd += spend;
            }
        }
    }

    pub fn record_fallback(&self) {
        self.router_fallbacks.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cloud_error(&self) {
        self.cloud_upstream_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> RouterMetricsSnapshot {
        let total = self.total_requests.load(Ordering::Relaxed);
        let total_us = self.total_router_latency_us.load(Ordering::Relaxed);
        let mut models: Vec<RouterModelStatsSnapshot> = self
            .per_model
            .lock()
            .map(|map| {
                map.iter()
                    .map(|(model, st)| RouterModelStatsSnapshot {
                        model: model.clone(),
                        is_local: st.is_local,
                        requests: st.requests,
                        prompt_tokens: st.prompt_tokens,
                        completion_tokens: st.completion_tokens,
                        cached_tokens: st.cached_tokens,
                        spend_usd: st.spend_usd,
                    })
                    .collect()
            })
            .unwrap_or_default();
        models.sort_by(|a, b| {
            b.requests
                .cmp(&a.requests)
                .then_with(|| a.model.cmp(&b.model))
        });
        let total_spend_usd = models.iter().map(|m| m.spend_usd).sum();
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
            router_fallbacks: self.router_fallbacks.load(Ordering::Relaxed),
            cloud_upstream_errors: self.cloud_upstream_errors.load(Ordering::Relaxed),
            mean_router_latency_ms: if total > 0 {
                (total_us as f64 / total as f64) / 1000.0
            } else {
                0.0
            },
            total_spend_usd,
            models,
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
    metrics: Arc<RouterMetrics>,
}

impl ArcRouter {
    pub fn load(cfg: ArcRouterConfig) -> Result<Self, String> {
        let heads = ArcHeads::load(&cfg.model_dir)?;
        let local_slot = infer_local_slot(&cfg.local_model_name, &heads.catalog);
        let registry = ModelsDevRegistry::load_cached_or_fetch();
        let cloud_candidates = resolve_cloud_candidates_from_registry(
            &registry,
            &heads.catalog,
            &cfg.cloud_model_filter,
            |var| std::env::var(var).ok(),
        )?;
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
            metrics: Arc::new(RouterMetrics::default()),
        })
    }

    pub fn metrics_arc(&self) -> Arc<RouterMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Returns the 4-tuple USD/Mtok pricing `[input, cache_read, cache_write, output]` for a
    /// routed cloud candidate (`models.dev` override first, then ARC catalog fallback).
    pub fn prices_for_cloud_model(&self, name: &str, slot: usize) -> [f64; 4] {
        if let Some(cand) = self.cloud_candidates.iter().find(|c| c.name == name)
            && let Some(p) = cand.price_override
        {
            return p;
        }
        self.heads
            .catalog
            .model_prices
            .get(slot)
            .copied()
            .flatten()
            .unwrap_or([0.0; 4])
    }

    /// Returns the `models.dev` capability descriptor for a routed cloud candidate, if present.
    pub fn capabilities_for_cloud_model(&self, name: &str) -> Option<&registry::ModelCapabilities> {
        self.cloud_candidates
            .iter()
            .find(|c| c.name == name)
            .map(|c| &c.capabilities)
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
                let prov = c.endpoint.provider_id.as_str();
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
                .filter(|c| UpstreamProxy::supports_protocol(&c.endpoint, protocol))
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        let policy = self.policy;
        let pin_pii_classes = self.pin_pii_classes.clone();
        let has_tools = canonical.has_tools;
        let has_images = canonical.has_images;
        let mut decision = tokio::task::spawn_blocking(move || {
            decide(
                &heads,
                &DecideInput {
                    block_readouts: &enc.block_readouts,
                    block_roles: &enc.block_roles,
                    prompt_tokens: enc.prompt_tokens,
                    has_tools,
                    has_images,
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
