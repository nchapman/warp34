//! Routing policy, local model slot auto-mapping, upstream presets, and decision logic.

use super::heads::{ArcHeads, ArcRuntimeCatalog, LEGS, PiiSignalPrediction, StageOnePrediction};
use std::collections::HashSet;

/// Built-in cloud upstream provider presets (zero config files required).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpstreamTarget {
    /// OpenRouter (`https://openrouter.ai/api/v1`, `OPENROUTER_API_KEY`), all 11 live ARC slots `[0..10]`.
    OpenRouter,
    /// Anthropic Messages API (`https://api.anthropic.com`, `ANTHROPIC_API_KEY`), Claude slots `[0..=2]`.
    Anthropic,
    /// OpenAI API (`https://api.openai.com/v1`, `OPENAI_API_KEY`), GPT slots `[6..=9]`.
    OpenAi,
    /// Custom OpenAI-compatible base URL (e.g. `https://gateway.internal/v1`).
    Custom { base_url: String },
}

impl UpstreamTarget {
    pub fn parse(spec: &str) -> Result<Self, String> {
        let trimmed = spec.trim();
        if trimmed.eq_ignore_ascii_case("openrouter") {
            Ok(Self::OpenRouter)
        } else if trimmed.eq_ignore_ascii_case("anthropic") {
            Ok(Self::Anthropic)
        } else if trimmed.eq_ignore_ascii_case("openai") {
            Ok(Self::OpenAi)
        } else if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
            Ok(Self::Custom {
                base_url: trimmed.trim_end_matches('/').to_string(),
            })
        } else {
            Err(format!(
                "invalid --router-upstream `{spec}` (expected openrouter, anthropic, openai, or http(s)://...)"
            ))
        }
    }

    pub fn base_url(&self) -> &str {
        match self {
            Self::OpenRouter => "https://openrouter.ai/api/v1",
            Self::Anthropic => "https://api.anthropic.com",
            Self::OpenAi => "https://api.openai.com/v1",
            Self::Custom { base_url } => base_url.as_str(),
        }
    }

    pub fn default_api_key_env(&self) -> &'static str {
        match self {
            Self::OpenRouter => "OPENROUTER_API_KEY",
            Self::Anthropic => "ANTHROPIC_API_KEY",
            Self::OpenAi => "OPENAI_API_KEY",
            Self::Custom { .. } => "WARP34_ROUTER_API_KEY",
        }
    }

    /// Which ARC cloud model slots are eligible by default for this upstream.
    pub fn default_cloud_slots(&self, catalog: &ArcRuntimeCatalog) -> Vec<usize> {
        catalog
            .models
            .iter()
            .enumerate()
            .filter_map(|(idx, m)| {
                let wire = m.wire.as_deref()?;
                if wire.starts_with("qwen/") || m.name.starts_with("withheld-") {
                    return None;
                }
                let ok = match self {
                    Self::OpenRouter | Self::Custom { .. } => catalog.model_prices[idx].is_some(),
                    Self::Anthropic => wire.starts_with("anthropic/"),
                    Self::OpenAi => wire.starts_with("openai/"),
                };
                ok.then_some(idx)
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoutingPolicy {
    /// Stay local whenever the local model clears `min_p_pass`; otherwise pick the best
    /// cloud model by `value_usd * P(pass) - expected_cost`.
    PreferLocal { min_p_pass: f64, value_usd: f64 },
    /// Pure utility maximization across local ($0) + cloud models with `value_usd`.
    Utility { value_usd: f64 },
}

impl RoutingPolicy {
    pub fn from_args(
        policy: &str,
        min_p_pass: f64,
        value_usd_override: Option<f64>,
    ) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&min_p_pass) {
            return Err(format!(
                "--router-min-p-pass must be in 0.0..=1.0, got {min_p_pass}"
            ));
        }
        match policy.trim().to_ascii_lowercase().as_str() {
            "prefer-local" | "prefer_local" => Ok(Self::PreferLocal {
                min_p_pass,
                value_usd: value_usd_override.unwrap_or(4.5),
            }),
            "sage" => Ok(Self::Utility {
                value_usd: value_usd_override.unwrap_or(4.5),
            }),
            "swift" => Ok(Self::Utility {
                value_usd: value_usd_override.unwrap_or(0.1),
            }),
            other => Err(format!(
                "invalid --router-policy `{other}` (expected prefer-local, sage, or swift)"
            )),
        }
    }
}

/// Automatically maps a served local `warp34` model ID/path to the best unmasked ARC-1.0 slot:
/// - `[53]` (`qwen3.8-flash-next`): `Flash-Next` / `35B-A3B` / `80B`
/// - `[55]` (`qwen3.8-27b`): `27B` / `32B`
/// - `[54]` (`qwen3.5-9b`): `9B` / `4B` / `2B` / `0.8B` (fallback)
pub fn infer_local_slot(model_name: &str, catalog: &ArcRuntimeCatalog) -> usize {
    let lower = model_name.to_ascii_lowercase();
    let preferred = if lower.contains("flash-next")
        || lower.contains("flash_next")
        || lower.contains("35b")
        || lower.contains("80b")
    {
        53
    } else if lower.contains("27b") || lower.contains("32b") {
        55
    } else {
        54
    };
    if preferred < catalog.models.len() {
        preferred
    } else {
        catalog.models.len().saturating_sub(1)
    }
}

/// Filter cloud candidate slots by optional user-supplied model names or wire IDs.
pub fn resolve_cloud_slots(
    upstream: &UpstreamTarget,
    catalog: &ArcRuntimeCatalog,
    filter: &[String],
) -> Result<Vec<usize>, String> {
    let base = upstream.default_cloud_slots(catalog);
    if filter.is_empty() {
        return Ok(base);
    }
    let wanted: HashSet<String> = filter
        .iter()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    let mut matched = Vec::new();
    for idx in base {
        let m = &catalog.models[idx];
        let name_l = m.name.to_ascii_lowercase();
        let wire_l = m.wire.as_deref().unwrap_or("").to_ascii_lowercase();
        if wanted.contains(&name_l) || wanted.contains(&wire_l) {
            matched.push(idx);
        }
    }
    if matched.is_empty() {
        return Err(format!(
            "--router-cloud-models {filter:?} matched no eligible models for upstream {}",
            upstream.base_url()
        ));
    }
    Ok(matched)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingBasis {
    /// Local-only mode (no cloud upstream configured); Stage 2 selects reasoning effort.
    LocalOnlyEffort,
    /// Sensitive PII/credential class triggered `signals.pii.payload`; pinned locally.
    PiiPinnedLocal,
    /// Local radix KV prefix cache is warm on a continuing session; stayed local to preserve KV.
    WarmKvStayLocal,
    /// Local model cleared `prefer-local` `min_p_pass` threshold.
    PreferLocal,
    /// Local model won utility maximization `V * P(pass) - expected_cost`.
    UtilityLocal,
    /// Spilled over to best cloud model under `V * P(pass) - expected_cost`.
    CloudSpillover,
}

impl RoutingBasis {
    pub fn as_header_str(self) -> &'static str {
        match self {
            Self::LocalOnlyEffort => "local_only_effort",
            Self::PiiPinnedLocal => "pii_pinned_local",
            Self::WarmKvStayLocal => "warm_kv_stay_local",
            Self::PreferLocal => "prefer_local",
            Self::UtilityLocal => "utility_local",
            Self::CloudSpillover => "cloud_spillover",
        }
    }
}

#[derive(Debug, Clone)]
pub enum RouteTarget {
    Local,
    Cloud {
        slot: usize,
        name: String,
        wire: String,
    },
}

#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub target: RouteTarget,
    pub p_pass: f64,
    pub local_p_pass: f64,
    pub reasoning_effort: String,
    pub basis: RoutingBasis,
    pub pii_triggered: Vec<String>,
    pub latency_ms: f64,
}

fn expected_cost_usd(usage: &[f64; LEGS], rates: &[f64; LEGS]) -> f64 {
    usage
        .iter()
        .zip(rates)
        .map(|(toks, rate)| if toks.is_finite() { toks * rate } else { 0.0 })
        .sum::<f64>()
        / 1_000_000.0
}

/// Map ARC Stage-2 `(effort, level)` action fields to `warp34`'s canonical reasoning effort.
fn map_arc_effort_to_warp34(effort: &str, level: Option<&str>) -> &'static str {
    match (effort, level) {
        ("none", _) => "none",
        ("high", Some("up")) => "high",
        ("high", Some("down")) => "low",
        ("high", _) => "high",
        ("low", _) => "low",
        ("medium", _) => "medium",
        ("default", Some("down")) => "low",
        ("default", Some("up")) => "high",
        _ => "medium",
    }
}

/// Select the best Stage-2 reasoning effort for `model_slot`.
fn choose_effort_for_slot(
    heads: &ArcHeads,
    readout: &[f32],
    last_role: &str,
    model_slot: usize,
    value_usd: f64,
) -> String {
    let actions: Vec<&super::heads::ArcActionEntry> = heads
        .catalog
        .actions
        .iter()
        .filter(|a| a.model == model_slot && !a.arms.is_empty())
        .collect();
    if actions.is_empty() {
        return "medium".to_string();
    }
    let all_arms: Vec<usize> = actions
        .iter()
        .flat_map(|a| a.arms.iter().copied())
        .collect();
    let Ok(scored) = heads.stage_two.predict_arms(readout, last_role, &all_arms) else {
        return "medium".to_string();
    };
    let score_map: std::collections::HashMap<usize, super::heads::ArmScore> =
        scored.into_iter().collect();
    let rates = heads.catalog.model_prices[model_slot].unwrap_or([0.0; LEGS]);
    let cost_weight = (1.0 / value_usd.max(0.05)).clamp(0.03, 10.0) as f32;

    let mut best_effort = "medium";
    let mut best_score = f32::NEG_INFINITY;
    for action in &actions {
        let mut sum = 0.0f32;
        let mut cnt = 0usize;
        for &arm in &action.arms {
            if let Some(s) = score_map.get(&arm) {
                let arm_cost: f32 = s
                    .usage
                    .iter()
                    .zip(&rates)
                    .map(|(toks, r)| toks * (*r as f32))
                    .sum::<f32>()
                    / 1_000_000.0;
                sum += s.quality - cost_weight * arm_cost;
                cnt += 1;
            }
        }
        if cnt > 0 {
            let mean = sum / cnt as f32;
            if mean > best_score {
                best_score = mean;
                best_effort = map_arc_effort_to_warp34(&action.effort, action.level.as_deref());
            }
        }
    }
    best_effort.to_string()
}

pub struct DecideInput<'a> {
    pub readout: &'a [f32],
    pub last_role: &'a str,
    pub prompt_tokens: usize,
    pub has_tools: bool,
    pub warm_local_kv: bool,
    pub local_slot: usize,
    pub cloud_slots: &'a [usize],
    pub policy: RoutingPolicy,
    pub pin_pii_classes: &'a HashSet<String>,
}

pub fn decide(heads: &ArcHeads, input: &DecideInput<'_>) -> Result<RoutingDecision, String> {
    let t0 = std::time::Instant::now();
    // Regime 0 = tool-bearing coding/agentic, 1 = tool-free.
    let regime = usize::from(!input.has_tools);
    let preds: Vec<StageOnePrediction> = heads.stage_one.predict(input.readout, regime)?;
    let local_p = preds
        .get(input.local_slot)
        .map(|p| p.p_pass)
        .filter(|p| p.is_finite())
        .unwrap_or(0.5);

    let pii_thresh = heads.catalog.pii_threshold_for_tokens(input.prompt_tokens);
    let pii: PiiSignalPrediction =
        heads
            .pii
            .predict(input.readout, pii_thresh, heads.catalog.pii_type_threshold)?;
    let pii_pin = !input.pin_pii_classes.is_empty()
        && pii
            .triggered_classes
            .iter()
            .any(|c| input.pin_pii_classes.contains(c));

    let value_usd = match input.policy {
        RoutingPolicy::PreferLocal { value_usd, .. } | RoutingPolicy::Utility { value_usd } => {
            value_usd
        }
    };

    // 1. Local-only mode (no cloud upstream configured).
    if input.cloud_slots.is_empty() {
        let effort = choose_effort_for_slot(
            heads,
            input.readout,
            input.last_role,
            input.local_slot,
            value_usd,
        );
        return Ok(RoutingDecision {
            target: RouteTarget::Local,
            p_pass: local_p,
            local_p_pass: local_p,
            reasoning_effort: effort,
            basis: RoutingBasis::LocalOnlyEffort,
            pii_triggered: pii.triggered_classes,
            latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
        });
    }

    // 2. Automatic PII / credential local pinning.
    if pii_pin {
        let effort = choose_effort_for_slot(
            heads,
            input.readout,
            input.last_role,
            input.local_slot,
            value_usd,
        );
        return Ok(RoutingDecision {
            target: RouteTarget::Local,
            p_pass: local_p,
            local_p_pass: local_p,
            reasoning_effort: effort,
            basis: RoutingBasis::PiiPinnedLocal,
            pii_triggered: pii.triggered_classes,
            latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
        });
    }

    // 3. KV-cache boundary awareness: stay local when prefix cache is warm on a continuing session.
    if input.warm_local_kv {
        let effort = choose_effort_for_slot(
            heads,
            input.readout,
            input.last_role,
            input.local_slot,
            value_usd,
        );
        return Ok(RoutingDecision {
            target: RouteTarget::Local,
            p_pass: local_p,
            local_p_pass: local_p,
            reasoning_effort: effort,
            basis: RoutingBasis::WarmKvStayLocal,
            pii_triggered: pii.triggered_classes,
            latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
        });
    }

    // 4. Policy decision: PreferLocal vs Utility.
    if let RoutingPolicy::PreferLocal { min_p_pass, .. } = input.policy
        && local_p >= min_p_pass
    {
        let effort = choose_effort_for_slot(
            heads,
            input.readout,
            input.last_role,
            input.local_slot,
            value_usd,
        );
        return Ok(RoutingDecision {
            target: RouteTarget::Local,
            p_pass: local_p,
            local_p_pass: local_p,
            reasoning_effort: effort,
            basis: RoutingBasis::PreferLocal,
            pii_triggered: pii.triggered_classes,
            latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
        });
    }

    // Score cloud candidates by `V * P(pass) - expected_cost` (local cost = $0).
    let local_utility = value_usd * local_p;
    let mut best_cloud: Option<(usize, f64, f64)> = None;
    for &slot in input.cloud_slots {
        let Some(pred) = preds.get(slot) else {
            continue;
        };
        if !pred.p_pass.is_finite() {
            continue;
        }
        let rates = heads.catalog.model_prices[slot].unwrap_or([0.0; LEGS]);
        let cost = expected_cost_usd(&pred.usage, &rates);
        let util = value_usd * pred.p_pass - cost;
        if best_cloud.is_none_or(|(_, best_u, _)| util > best_u) {
            best_cloud = Some((slot, util, pred.p_pass));
        }
    }

    let stay_local = match best_cloud {
        None => true,
        Some((_, best_u, _)) => local_utility >= best_u,
    };

    if stay_local {
        let effort = choose_effort_for_slot(
            heads,
            input.readout,
            input.last_role,
            input.local_slot,
            value_usd,
        );
        return Ok(RoutingDecision {
            target: RouteTarget::Local,
            p_pass: local_p,
            local_p_pass: local_p,
            reasoning_effort: effort,
            basis: RoutingBasis::UtilityLocal,
            pii_triggered: pii.triggered_classes,
            latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
        });
    }

    let (slot, _, cloud_p) = best_cloud.unwrap();
    let entry = &heads.catalog.models[slot];
    let wire = entry.wire.clone().unwrap_or_else(|| entry.name.clone());
    let effort = choose_effort_for_slot(heads, input.readout, input.last_role, slot, value_usd);
    Ok(RoutingDecision {
        target: RouteTarget::Cloud {
            slot,
            name: entry.name.clone(),
            wire,
        },
        p_pass: cloud_p,
        local_p_pass: local_p,
        reasoning_effort: effort,
        basis: RoutingBasis::CloudSpillover,
        pii_triggered: pii.triggered_classes,
        latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}
