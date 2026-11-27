//! Routing policy, local model slot auto-mapping, and ARC-1.0 Stage 1/2/PII decision logic.

use super::{
    heads::{ArcHeads, ArcRuntimeCatalog, LEGS, PiiSignalPrediction, StageOnePrediction},
    registry::{ModelCapabilities, ModelsDevRegistry, ProviderEndpoint, infer_tier_from_prices},
};
use std::collections::HashSet;

/// Default `prefer-local` threshold on the local model's `P(pass)` (`rayline-arc-decide` parity).
pub const DEFAULT_PREFER_LOCAL_MIN_P_PASS: f64 = 0.65;

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
                value_usd: value_usd_override.unwrap_or(1.5),
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

#[derive(Debug, Clone)]
pub struct CloudCandidate {
    pub slot: usize,
    pub name: String,
    pub wire: String,
    /// Per-candidate upstream endpoint resolved via `models.dev`.
    pub endpoint: ProviderEndpoint,
    /// Live USD/Mtok `[input, cache_read, cache_write, output]` from `models.dev` (overrides slot price).
    pub price_override: Option<[f64; LEGS]>,
    /// Model capabilities & valid reasoning effort levels from `models.dev`.
    pub capabilities: ModelCapabilities,
}

pub fn find_native_arc_slot(catalog: &ArcRuntimeCatalog, id: &str) -> Option<usize> {
    let wanted = id.trim().to_ascii_lowercase();
    let wanted_canon = wanted.replace('.', "-");
    catalog.models.iter().enumerate().find_map(|(idx, m)| {
        let name_lc = m.name.to_ascii_lowercase();
        let name_canon = name_lc.replace('.', "-");
        let wire_lc = m.wire.as_deref().map(str::to_ascii_lowercase);
        let matches = name_lc == wanted
            || name_canon == wanted_canon
            || wire_lc
                .as_deref()
                .is_some_and(|w| w == wanted || w.replace('.', "-") == wanted_canon);
        matches.then_some(idx)
    })
}

/// Returns true when `--router-cloud-models` requests local-only routing (`[]`, `none`, or `local`),
/// allowing startup to skip loading or refreshing `models.dev`.
pub fn is_local_only_cloud_filter(filter: &[String]) -> bool {
    let non_empty: Vec<&str> = filter
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    non_empty.is_empty()
        || (non_empty.len() == 1
            && matches!(non_empty[0].to_ascii_lowercase().as_str(), "none" | "local"))
}

/// Resolves cloud candidates through the `models.dev` registry, auto-detecting provider base URLs,
/// API keys from environment variables, live USD/Mtok pricing, and model capabilities.
pub fn resolve_cloud_candidates_from_registry<F>(
    registry: &ModelsDevRegistry,
    catalog: &ArcRuntimeCatalog,
    filter: &[String],
    env_lookup: F,
) -> Result<Vec<CloudCandidate>, String>
where
    F: Fn(&str) -> Option<String>,
{
    if is_local_only_cloud_filter(filter) {
        return Ok(Vec::new());
    }
    let specs: Vec<String> = if filter.len() == 1 && filter[0].trim().eq_ignore_ascii_case("auto") {
        registry.default_models_for_env(&env_lookup)
    } else {
        filter
            .iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };
    if specs.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = Vec::with_capacity(specs.len());
    for raw in &specs {
        let resolved = registry.resolve_spec(raw, &env_lookup)?;
        let is_custom_url = resolved.endpoint.provider_id.starts_with("http://")
            || resolved.endpoint.provider_id.starts_with("https://");
        if resolved.endpoint.api_key.is_none() && !is_custom_url {
            return Err(format!(
                "cloud model `{raw}` resolved to provider `{}` ({}) which requires `{}` (or OPENROUTER_API_KEY) to be set",
                resolved.endpoint.provider_id,
                resolved.endpoint.base_url,
                resolved.endpoint.env_hint,
            ));
        }

        let slot = if let Some(tier) = resolved.explicit_tier.as_deref() {
            resolve_tier_slot(tier, catalog)
                .ok_or_else(|| format!("unknown tier `{tier}` in --router-cloud-models `{raw}`"))?
        } else if let Some(native) = find_native_arc_slot(catalog, &resolved.canonical_id)
            .or_else(|| find_native_arc_slot(catalog, &resolved.wire_model))
        {
            native
        } else if let Some(prices) = resolved.prices {
            let inferred = infer_tier_from_prices(prices);
            resolve_tier_slot(inferred, catalog).unwrap_or(2)
        } else {
            resolve_tier_slot("medium", catalog).unwrap_or(2)
        };

        out.push(CloudCandidate {
            slot,
            name: resolved.canonical_id,
            wire: resolved.wire_model,
            endpoint: resolved.endpoint,
            price_override: resolved.prices,
            capabilities: resolved.capabilities,
        });
    }
    Ok(out)
}

fn resolve_tier_slot(token: &str, catalog: &ArcRuntimeCatalog) -> Option<usize> {
    let idx = match token {
        "fast" | "cheap" | "small" => 3, // deepseek/deepseek-v4.1-flash
        "medium" | "mid" => 2,           // anthropic/claude-sonnet-5.5
        "strong" | "big" => 10,          // moonshotai/kimi-k3 ($3/$15 frontier reasoning)
        "frontier" => 1,                 // anthropic/claude-opus-5.5
        _ => return None,
    };
    (idx < catalog.models.len()).then_some(idx)
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
pub struct CloudRouteTarget {
    pub slot: usize,
    pub name: String,
    pub wire: String,
    pub endpoint: ProviderEndpoint,
    pub prices: [f64; LEGS],
    pub capabilities: ModelCapabilities,
}

#[derive(Debug, Clone)]
pub enum RouteTarget {
    Local,
    Cloud(Box<CloudRouteTarget>),
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
///
/// In `ARC-1.0` (`rayline-arc-decide` `release.rs` / `rayline-arc-routes` `shape.rs`), every
/// Stage-2 action runs with `thinking.mode = "on"`: `effort` is the base model effort
/// (`"default"`, `"medium"`, `"high"`) and `level` (`Thinking::steering_level`) is a
/// `-1 / 0 / +1` steering delta (`"down"`, `"none"` / `None`, `"up"`). `level == Some("none")`
/// means *unsteered* baseline thinking, not disabled thinking.
pub(crate) fn map_arc_effort_to_warp34(effort: &str, level: Option<&str>) -> &'static str {
    match (effort, level) {
        ("none" | "off", _) => "none",
        ("minimal", Some("up")) => "low",
        ("minimal", _) => "minimal",
        ("low", Some("up")) => "medium",
        ("low", _) => "low",
        ("default" | "medium", Some("down")) => "low",
        ("default" | "medium", Some("up")) => "high",
        ("default" | "medium", _) => "medium",
        ("high", Some("down")) => "medium",
        ("high", Some("up")) => "xhigh",
        ("high", _) => "high",
        ("xhigh" | "max", Some("down")) => "high",
        ("xhigh" | "max", _) => "xhigh",
        _ => "medium",
    }
}

/// Select the best Stage-2 reasoning effort for `model_slot` using multi-block pooled history,
/// filtering to supported arms (`quality_support && usage_support`), using `catalog.stage_two_cost_weight`,
/// effective `rates_override` (from `models.dev` when present), and scoring each action by its
/// best (`max`) supported arm (`rayline-arc-decide` parity).
fn choose_effort_for_slot(
    heads: &ArcHeads,
    block_readouts: &[Vec<f32>],
    block_roles: &[String],
    model_slot: usize,
    rates_override: Option<[f64; LEGS]>,
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
        .filter(|&arm| {
            heads
                .stage_two
                .quality_support
                .get(arm)
                .copied()
                .unwrap_or(false)
                && heads
                    .stage_two
                    .usage_support
                    .get(arm)
                    .copied()
                    .unwrap_or(false)
        })
        .collect();
    if all_arms.is_empty() {
        return "medium".to_string();
    }
    let Ok(scored) = heads
        .stage_two
        .predict_arms(block_readouts, block_roles, &[], &all_arms)
    else {
        return "medium".to_string();
    };
    let score_map: std::collections::HashMap<usize, super::heads::ArmScore> =
        scored.into_iter().collect();
    let rates = rates_override
        .or(heads.catalog.model_prices[model_slot])
        .unwrap_or([0.0; LEGS]);
    let cost_weight = heads.catalog.stage_two_cost_weight as f32;

    let mut best_effort = "medium";
    let mut best_score = f32::NEG_INFINITY;
    for action in &actions {
        let mut action_best = f32::NEG_INFINITY;
        for &arm in &action.arms {
            if let Some(s) = score_map.get(&arm) {
                let arm_cost: f32 = s
                    .usage
                    .iter()
                    .zip(&rates)
                    .map(|(toks, r)| toks * (*r as f32))
                    .sum::<f32>()
                    / 1_000_000.0;
                let score = s.quality - cost_weight * arm_cost;
                if score.is_finite() && score > action_best {
                    action_best = score;
                }
            }
        }
        if action_best > best_score {
            best_score = action_best;
            best_effort = map_arc_effort_to_warp34(&action.effort, action.level.as_deref());
        }
    }
    best_effort.to_string()
}

pub struct DecideInput<'a> {
    pub block_readouts: &'a [Vec<f32>],
    pub block_roles: &'a [String],
    pub prompt_tokens: usize,
    pub has_tools: bool,
    pub has_images: bool,
    pub warm_local_kv: bool,
    pub local_slot: usize,
    pub cloud_candidates: &'a [CloudCandidate],
    pub policy: RoutingPolicy,
    pub pin_pii_classes: &'a HashSet<String>,
}

pub fn decide(heads: &ArcHeads, input: &DecideInput<'_>) -> Result<RoutingDecision, String> {
    let t0 = std::time::Instant::now();
    let decision_readout = input
        .block_readouts
        .last()
        .map(Vec::as_slice)
        .ok_or_else(|| "ARC decide requires at least one block readout".to_string())?;
    // Collect only the Stage-1 candidate indices for active local + cloud models
    // (models[] indices != stage_one.candidates[] indices in arc-runtime.json!).
    let local_cand = heads
        .catalog
        .stage_one_candidate_for_model(input.local_slot);
    let mut active_cands = Vec::with_capacity(1 + input.cloud_candidates.len());
    if let Some(c) = local_cand {
        active_cands.push(c);
    }
    for cand in input.cloud_candidates {
        if let Some(c) = heads.catalog.stage_one_candidate_for_model(cand.slot) {
            active_cands.push(c);
        }
    }
    let preds: Vec<StageOnePrediction> = heads.stage_one.predict_selected(
        decision_readout,
        heads.catalog.stage_one_regime,
        &active_cands,
    )?;
    let local_p = local_cand
        .and_then(|c| preds.get(c))
        .map(|p| p.p_pass)
        .filter(|p| p.is_finite())
        .unwrap_or(0.5);

    let pii_thresh = heads.catalog.pii_threshold_for_tokens(input.prompt_tokens);
    let pii: PiiSignalPrediction = heads.pii.predict(
        decision_readout,
        pii_thresh,
        heads.catalog.pii_type_threshold,
    )?;
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

    // 1–3. Local pinning guards in semantic priority order:
    //      (a) PII / credential guard (`pii_pinned_local`),
    //      (b) live warm radix + GDN KV continuation (`warm_kv_stay_local`),
    //      (c) local-only mode / stateful local-pinned request (`local_only_effort`).
    if pii_pin || input.warm_local_kv || input.cloud_candidates.is_empty() {
        let basis = if pii_pin {
            RoutingBasis::PiiPinnedLocal
        } else if input.warm_local_kv {
            RoutingBasis::WarmKvStayLocal
        } else {
            RoutingBasis::LocalOnlyEffort
        };
        let effort = choose_effort_for_slot(
            heads,
            input.block_readouts,
            input.block_roles,
            input.local_slot,
            None,
        );
        return Ok(RoutingDecision {
            target: RouteTarget::Local,
            p_pass: local_p,
            local_p_pass: local_p,
            reasoning_effort: effort,
            basis,
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
            input.block_readouts,
            input.block_roles,
            input.local_slot,
            None,
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

    // Score cloud candidates by `V * P(pass) - expected_cost` (local cost = $0),
    // indexing Stage-1 predictions through `stage_one_candidate_for_model`, filtering by
    // `models.dev` capabilities (`tool_call`, `supports_image`, `context_limit`), and using
    // live prices if present. Under `PreferLocal`, only consider cloud models that strictly
    // beat `local_p` so a weaker cheap model cannot block a stronger frontier model.
    let local_utility = value_usd * local_p;
    let require_better_p_pass = matches!(input.policy, RoutingPolicy::PreferLocal { .. });
    let mut best_cloud: Option<(&CloudCandidate, f64, f64)> = None;
    for cand in input.cloud_candidates {
        if input.has_tools && !cand.capabilities.tool_call {
            continue;
        }
        if input.has_images && !cand.capabilities.supports_image {
            continue;
        }
        if cand
            .capabilities
            .context_limit
            .is_some_and(|lim| input.prompt_tokens > lim)
        {
            continue;
        }
        let Some(cand_idx) = heads.catalog.stage_one_candidate_for_model(cand.slot) else {
            continue;
        };
        let Some(pred) = preds.get(cand_idx) else {
            continue;
        };
        if !pred.p_pass.is_finite() {
            continue;
        }
        if require_better_p_pass && pred.p_pass <= local_p {
            continue;
        }
        let rates = cand
            .price_override
            .or(heads.catalog.model_prices[cand.slot])
            .unwrap_or([0.0; LEGS]);
        let cost = expected_cost_usd(&pred.usage, &rates);
        let util = value_usd * pred.p_pass - cost;
        if best_cloud.is_none_or(|(_, best_u, _)| util > best_u) {
            best_cloud = Some((cand, util, pred.p_pass));
        }
    }

    let Some((cand, _, cloud_p)) = best_cloud.filter(|&(_, best_u, _)| match input.policy {
        RoutingPolicy::Utility { .. } => best_u > local_utility,
        RoutingPolicy::PreferLocal { .. } => true,
    }) else {
        let effort = choose_effort_for_slot(
            heads,
            input.block_readouts,
            input.block_roles,
            input.local_slot,
            None,
        );
        let basis = match input.policy {
            RoutingPolicy::PreferLocal { .. } => RoutingBasis::PreferLocal,
            RoutingPolicy::Utility { .. } => RoutingBasis::UtilityLocal,
        };
        return Ok(RoutingDecision {
            target: RouteTarget::Local,
            p_pass: local_p,
            local_p_pass: local_p,
            reasoning_effort: effort,
            basis,
            pii_triggered: pii.triggered_classes,
            latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
        });
    };

    let raw_effort = choose_effort_for_slot(
        heads,
        input.block_readouts,
        input.block_roles,
        cand.slot,
        cand.price_override,
    );
    let effort = cand.capabilities.clamp_effort(&raw_effort);
    let prices = cand
        .price_override
        .or_else(|| heads.catalog.model_prices.get(cand.slot).copied().flatten())
        .unwrap_or([0.0; LEGS]);
    Ok(RoutingDecision {
        target: RouteTarget::Cloud(Box::new(CloudRouteTarget {
            slot: cand.slot,
            name: cand.name.clone(),
            wire: cand.wire.clone(),
            endpoint: cand.endpoint.clone(),
            prices,
            capabilities: cand.capabilities.clone(),
        })),
        p_pass: cloud_p,
        local_p_pass: local_p,
        reasoning_effort: effort,
        basis: RoutingBasis::CloudSpillover,
        pii_triggered: pii.triggered_classes,
        latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_two_steering_levels_preserve_thinking_mode() {
        // In ARC-1.0 (`rayline-arc-decide` `release.rs` / `rayline-arc-routes` `shape.rs`),
        // `level: Some("none")` means unsteered baseline thinking (`thinking.mode = "on"`),
        // NOT disabled thinking.
        assert_eq!(map_arc_effort_to_warp34("default", Some("none")), "medium");
        assert_eq!(map_arc_effort_to_warp34("default", None), "medium");
        assert_eq!(map_arc_effort_to_warp34("default", Some("down")), "low");
        assert_eq!(map_arc_effort_to_warp34("default", Some("up")), "high");
        assert_eq!(map_arc_effort_to_warp34("high", Some("none")), "high");
        assert_eq!(map_arc_effort_to_warp34("high", None), "high");
        assert_eq!(map_arc_effort_to_warp34("high", Some("down")), "medium");
        assert_eq!(map_arc_effort_to_warp34("high", Some("up")), "xhigh");
        assert_eq!(map_arc_effort_to_warp34("low", Some("up")), "medium");
        assert_eq!(map_arc_effort_to_warp34("low", None), "low");
        assert_eq!(map_arc_effort_to_warp34("medium", None), "medium");
        assert_eq!(map_arc_effort_to_warp34("none", None), "none");
    }

    #[test]
    fn find_native_arc_slot_matches_mixed_dash_and_dot_version_ids() {
        use super::super::heads::{ArcModelEntry, ArcRuntimeCatalog};
        let catalog = ArcRuntimeCatalog {
            models: vec![
                ArcModelEntry {
                    name: "claude-haiku-5.5".to_string(),
                    wire: Some("anthropic/claude-haiku-5.5".to_string()),
                },
                ArcModelEntry {
                    name: "claude-opus-5.5".to_string(),
                    wire: Some("anthropic/claude-opus-5.5".to_string()),
                },
                ArcModelEntry {
                    name: "claude-sonnet-5.5".to_string(),
                    wire: Some("anthropic/claude-sonnet-5.5".to_string()),
                },
                ArcModelEntry {
                    name: "deepseek-v4.1-flash".to_string(),
                    wire: Some("deepseek/deepseek-v4.1-flash".to_string()),
                },
            ],
            actions: Vec::new(),
            fallback_action: 0,
            model_to_candidate: vec![Some(0), Some(1), Some(2), Some(3)],
            candidate_latency_factor: vec![None; 4],
            stage_one_regime: 0,
            stage_one_cost_weight: 0.3,
            stage_one_latency_weight: 0.0,
            stage_two_cost_weight: 0.3,
            stage_two_support_columns: Vec::new(),
            model_prices: vec![None; 4],
            pii_classes: Vec::new(),
            pii_presence_threshold: 0.5,
            pii_type_threshold: 0.5,
            pii_length_buckets: Vec::new(),
        };
        assert_eq!(find_native_arc_slot(&catalog, "claude-haiku-5-5"), Some(0));
        assert_eq!(find_native_arc_slot(&catalog, "claude-opus-5-5"), Some(1));
        assert_eq!(find_native_arc_slot(&catalog, "claude-sonnet-5-5"), Some(2));
        assert_eq!(
            find_native_arc_slot(&catalog, "anthropic/claude-sonnet-5-5"),
            Some(2)
        );
        assert_eq!(
            find_native_arc_slot(&catalog, "deepseek-v4-1-flash"),
            Some(3)
        );
        assert_eq!(find_native_arc_slot(&catalog, "Claude-Opus-5-5"), Some(1));
        assert_eq!(find_native_arc_slot(&catalog, "unknown-model-99"), None);
        assert!((DEFAULT_PREFER_LOCAL_MIN_P_PASS - 0.65).abs() < 1e-12);
    }

    #[test]
    fn local_only_cloud_filter_detects_empty_and_none_aliases() {
        assert!(is_local_only_cloud_filter(&[]));
        assert!(is_local_only_cloud_filter(&["  ".to_string()]));
        assert!(is_local_only_cloud_filter(&["none".to_string()]));
        assert!(is_local_only_cloud_filter(&["LOCAL".to_string()]));
        assert!(!is_local_only_cloud_filter(&["auto".to_string()]));
        assert!(!is_local_only_cloud_filter(&[
            "deepseek/deepseek-v4-pro".to_string()
        ]));
    }
}
