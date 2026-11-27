//! Routing policy, local model slot auto-mapping, and ARC-1.0 Stage 1/2/PII decision logic.

use super::{
    heads::{ArcHeads, ArcRuntimeCatalog, LEGS, PiiSignalPrediction, StageOnePrediction},
    input::TurnEvent,
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

/// Trained ARC-1.0 local anchor for 80B-A3B MoE (`qwen3.8-flash-next`, 2 Stage-2 actions).
pub const LOCAL_FLASH_NEXT_SLOT: usize = 53;
/// Trained ARC-1.0 local anchor for small/9B-class models (`qwen3.5-9b`, 4 Stage-2 actions).
pub const LOCAL_9B_SLOT: usize = 54;
/// Canonical ARC-1.0 reference slot for the ~27B local model target (`qwen3.8-27b`, 4 Stage-2
/// actions: `down` / `none` / `up` / unsteered).
pub const DEFAULT_LOCAL_SLOT: usize = 55;
/// Canonical ARC-1.0 reference slot for the mid/workhorse remote tier (`claude-sonnet-5.5`).
pub const REMOTE_SONNET_SLOT: usize = 2;
/// Canonical ARC-1.0 reference slot for the frontier/heavy remote tier (`claude-opus-5.5`).
pub const REMOTE_OPUS_SLOT: usize = 1;

/// Calibrated local routing profile pairing one of ARC-1.0's three trained local anchors
/// (`[54] qwen3.5-9b`, `[55] qwen3.8-27b`, `[53] qwen3.8-flash-next`) with a monotone
/// logit-space ability offset (`local_p' = sigmoid(logit(local_p) + logit_bias)`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalProfile {
    pub slot: usize,
    pub logit_bias: f64,
}

fn has_ascii_token(haystack: &str, needle: &str) -> bool {
    haystack
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '.')
        .any(|tok| tok.eq_ignore_ascii_case(needle))
}

/// Maps a served local `warp34` model ID/path onto its empirically calibrated ARC-1.0 anchor
/// slot and logit bias:
/// - `Flash-Next` (`flash-next` / `flash_next`): `[53]` `qwen3.8-flash-next`, `bias = 0.0`
/// - `Ornith-1.5-35B`: `[54]` `qwen3.5-9b`, `bias = +0.45` (`82.5%` vs `75.0%` screen)
/// - `Apex-35B` (`apex-35b` / `apex_35b`): `[54]` `qwen3.5-9b`, `bias = +0.20` (`78.5%` vs `75.0%` screen)
/// - `Ornith-1.5-9B`: `[54]` `qwen3.5-9b`, `bias = +0.20` (`78.7%` vs `75.0%` screen)
/// - `Ternary-Bonsai-2.5-27B`: `[55]` `qwen3.8-27b`, `bias = +0.40` (`93.5%` vs `90.7%` screen)
/// - Small `<= 14B` models (`0.5b`..`14b`): `[54]` `qwen3.5-9b`, `bias = 0.0`
/// - Default (including `27B` and arbitrary/third-party local models): `[55]` `qwen3.8-27b`, `bias = 0.0`
pub fn infer_local_profile(model_name: &str, catalog: &ArcRuntimeCatalog) -> LocalProfile {
    // Match only the model basename/leaf segment so parent directory names never skew matching.
    let trimmed = model_name.trim().trim_end_matches(['/', '\\']);
    let leaf = trimmed.rsplit(['/', '\\']).next().unwrap_or(trimmed);
    let lower = leaf.to_ascii_lowercase();
    let is_small = [
        "0.5b", "0.8b", "1b", "1.5b", "1.7b", "2b", "3b", "4b", "7b", "8b", "9b", "12b", "14b",
    ]
    .iter()
    .any(|sz| has_ascii_token(&lower, sz));
    let (raw_slot, logit_bias) = if lower.contains("flash-next") || lower.contains("flash_next") {
        (LOCAL_FLASH_NEXT_SLOT, 0.0)
    } else if lower.contains("ornith") && has_ascii_token(&lower, "35b") {
        (LOCAL_9B_SLOT, 0.45)
    } else if (lower.contains("ornith") && has_ascii_token(&lower, "9b"))
        || lower.contains("apex-35b")
        || lower.contains("apex_35b")
    {
        (LOCAL_9B_SLOT, 0.20)
    } else if lower.contains("bonsai") && has_ascii_token(&lower, "27b") {
        (DEFAULT_LOCAL_SLOT, 0.40)
    } else if is_small {
        (LOCAL_9B_SLOT, 0.0)
    } else {
        (DEFAULT_LOCAL_SLOT, 0.0)
    };
    LocalProfile {
        slot: raw_slot.min(catalog.models.len().saturating_sub(1)),
        logit_bias,
    }
}

/// Convenience wrapper returning just the resolved local anchor slot.
pub fn infer_local_slot(model_name: &str, catalog: &ArcRuntimeCatalog) -> usize {
    infer_local_profile(model_name, catalog).slot
}

/// Applies a monotone logit-space calibration offset (`sigmoid(logit(p) + bias)`) to a Stage-1
/// pass probability `p`. Returns `p` clamped to `[0, 1]` when `bias == 0.0` or non-finite.
pub fn apply_logit_bias(p: f64, bias: f64) -> f64 {
    let clamped = p.clamp(0.0, 1.0);
    if bias == 0.0 || !bias.is_finite() {
        return clamped;
    }
    let eps = 1e-6;
    let p_safe = clamped.clamp(eps, 1.0 - eps);
    let logit = (p_safe / (1.0 - p_safe)).ln();
    1.0 / (1.0 + (-(logit + bias)).exp())
}

/// Human-readable archetype label for a resolved remote cloud slot.
pub fn remote_tier_label(slot: usize) -> &'static str {
    if slot == REMOTE_OPUS_SLOT {
        "frontier"
    } else {
        "standard"
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

/// Resolves an explicit remote tier prefix (`standard=`/`sonnet=` or `frontier=`/`opus=`) to one
/// of the two canonical remote ARC-1.0 archetypes (`REMOTE_SONNET_SLOT` or `REMOTE_OPUS_SLOT`).
pub fn resolve_remote_tier_slot(token: &str, catalog: &ArcRuntimeCatalog) -> Option<usize> {
    let idx = match token.trim().to_ascii_lowercase().as_str() {
        "standard" | "sonnet" => REMOTE_SONNET_SLOT,
        "frontier" | "opus" => REMOTE_OPUS_SLOT,
        _ => return None,
    };
    (idx < catalog.models.len()).then_some(idx)
}

/// Maps any swappable cloud model to the generalized two-tier remote ARC-1.0 archetypes:
/// `REMOTE_SONNET_SLOT` (`claude-sonnet-5.5`, `standard`) or `REMOTE_OPUS_SLOT`
/// (`claude-opus-5.5`, `frontier`).
///
/// Precedence:
/// 1. Explicit tier prefix (`standard=`/`sonnet=` or `frontier=`/`opus=`).
/// 2. Explicit archetype token in the model ID (`"opus"` vs `"sonnet"`).
/// 3. Live `models.dev` output pricing (`> $15/M` -> `frontier`, `<= $15/M` -> `standard`).
/// 4. Default fallback: `standard`.
pub fn infer_remote_slot(
    explicit_tier: Option<&str>,
    canonical_id: &str,
    wire_model: &str,
    prices: Option<[f64; LEGS]>,
    catalog: &ArcRuntimeCatalog,
) -> Result<usize, String> {
    if let Some(tier) = explicit_tier {
        return resolve_remote_tier_slot(tier, catalog)
            .ok_or_else(|| format!("unknown remote tier `{tier}`"));
    }
    let has_token = |s: &str, needle: &str| {
        s.split(|c: char| !c.is_ascii_alphanumeric())
            .any(|tok| tok.eq_ignore_ascii_case(needle))
    };
    let tier = if has_token(canonical_id, "opus") || has_token(wire_model, "opus") {
        "frontier"
    } else if has_token(canonical_id, "sonnet") || has_token(wire_model, "sonnet") {
        "standard"
    } else if let Some(p) = prices {
        infer_tier_from_prices(p)
    } else {
        "standard"
    };
    Ok(resolve_remote_tier_slot(tier, catalog)
        .unwrap_or_else(|| REMOTE_SONNET_SLOT.min(catalog.models.len().saturating_sub(1))))
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

        let slot = infer_remote_slot(
            resolved.explicit_tier.as_deref(),
            &resolved.canonical_id,
            &resolved.wire_model,
            resolved.prices,
            catalog,
        )
        .map_err(|e| format!("{e} in --router-cloud-models `{raw}`"))?;

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

/// Last served routing target for a session, held symmetrically across `TurnEvent::ToolResult`
/// loops to match `rayline-arc-decide` (`Local` or a specific cloud model's `wire` ID).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeldTarget {
    Local,
    Cloud(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingBasis {
    /// Local-only mode (no cloud upstream configured); Stage 2 selects reasoning effort.
    LocalOnlyEffort,
    /// Sensitive PII/credential class triggered `signals.pii.payload`; pinned locally.
    PiiPinnedLocal,
    /// Local radix KV prefix cache is warm on a continuing session; stayed local to preserve KV.
    WarmKvStayLocal,
    /// Held the session's current model (local or cloud) across a `tool_result` loop.
    SessionHeld,
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
            Self::SessionHeld => "session_held",
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
    cost_weight_override: Option<f32>,
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
    let cost_weight = cost_weight_override
        .filter(|w| w.is_finite() && *w >= 0.0)
        .unwrap_or(heads.catalog.stage_two_cost_weight as f32);

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
    pub event: TurnEvent,
    pub held_target: Option<&'a HeldTarget>,
    pub local_slot: usize,
    pub local_logit_bias: f64,
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
    let raw_local_p = local_cand
        .and_then(|c| preds.get(c))
        .map(|p| p.p_pass)
        .filter(|p| p.is_finite())
        .unwrap_or(0.5);
    let local_p = apply_logit_bias(raw_local_p, input.local_logit_bias);

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
    let cloud_stage_two_cost_weight =
        (value_usd.is_finite() && value_usd > 0.0).then_some((1.0 / value_usd) as f32);

    // 1–2. Unconditional local guards:
    //      (a) PII / credential guard (`pii_pinned_local`),
    //      (b) local-only mode / stateful local-pinned request (`local_only_effort`).
    if pii_pin || input.cloud_candidates.is_empty() {
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

    let cloud_eligible = |cand: &CloudCandidate| -> bool {
        (!input.has_tools || cand.capabilities.tool_call)
            && (!input.has_images || cand.capabilities.supports_image)
            && cand
                .capabilities
                .context_limit
                .is_none_or(|lim| input.prompt_tokens <= lim)
    };

    // 3. Symmetric tool-loop hold (matching `rayline-arc-decide`): on `TurnEvent::ToolResult`,
    //    hold whichever model (local OR cloud) was last served for this session and run Stage 2
    //    for effort steering, preserving prompt/KV caches and avoiding mid-task split-brain flips.
    //    If the in-memory hold was evicted but local radix KV is warm, preserve local KV.
    if input.event == TurnEvent::ToolResult {
        if let Some(held) = input.held_target {
            match held {
                HeldTarget::Local => {
                    let effort = choose_effort_for_slot(
                        heads,
                        input.block_readouts,
                        input.block_roles,
                        input.local_slot,
                        None,
                        None,
                    );
                    return Ok(RoutingDecision {
                        target: RouteTarget::Local,
                        p_pass: local_p,
                        local_p_pass: local_p,
                        reasoning_effort: effort,
                        basis: if input.warm_local_kv {
                            RoutingBasis::WarmKvStayLocal
                        } else {
                            RoutingBasis::SessionHeld
                        },
                        pii_triggered: pii.triggered_classes,
                        latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
                    });
                }
                HeldTarget::Cloud(held_wire) => {
                    if let Some(cand) = input
                        .cloud_candidates
                        .iter()
                        .find(|c| c.wire == *held_wire && cloud_eligible(c))
                    {
                        let prices = cand
                            .price_override
                            .or(heads.catalog.model_prices[cand.slot])
                            .unwrap_or([0.0; LEGS]);
                        let cloud_p = heads
                            .catalog
                            .stage_one_candidate_for_model(cand.slot)
                            .and_then(|idx| preds.get(idx))
                            .map(|p| p.p_pass)
                            .filter(|p| p.is_finite())
                            .unwrap_or(local_p);
                        let raw_effort = choose_effort_for_slot(
                            heads,
                            input.block_readouts,
                            input.block_roles,
                            cand.slot,
                            Some(prices),
                            cloud_stage_two_cost_weight,
                        );
                        let effort = cand.capabilities.clamp_effort(&raw_effort);
                        return Ok(RoutingDecision {
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
                            basis: RoutingBasis::SessionHeld,
                            pii_triggered: pii.triggered_classes,
                            latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
                        });
                    }
                }
            }
        } else if input.warm_local_kv {
            let effort = choose_effort_for_slot(
                heads,
                input.block_readouts,
                input.block_roles,
                input.local_slot,
                None,
                None,
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
    }

    // 4. Warm local KV continuation on a new `UserMessage` (cache-boundary hysteresis):
    //    only applies when the session is NOT currently held on cloud (preventing stale Turn-0
    //    local KV from hijacking an escalated cloud session). Under `PreferLocal`, stay local
    //    while `local_p >= (min_p_pass - 0.15).clamp(0.0, min_p_pass)`. Under `Utility`, warm KV
    //    is modeled in Step 5 as a small prefill-savings utility bonus rather than a value-blind
    //    probability floor.
    let eligible_warm_local = input.warm_local_kv
        && input.event == TurnEvent::UserMessage
        && !matches!(input.held_target, Some(HeldTarget::Cloud(_)));
    if eligible_warm_local && let RoutingPolicy::PreferLocal { min_p_pass, .. } = input.policy {
        let warm_floor = (min_p_pass - 0.15).clamp(0.0, min_p_pass);
        if local_p >= warm_floor {
            let effort = choose_effort_for_slot(
                heads,
                input.block_readouts,
                input.block_roles,
                input.local_slot,
                None,
                None,
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
    }

    // 5. Boundary Stage-1 policy decision: PreferLocal vs Utility.
    if let RoutingPolicy::PreferLocal { min_p_pass, .. } = input.policy
        && local_p >= min_p_pass
    {
        let effort = choose_effort_for_slot(
            heads,
            input.block_readouts,
            input.block_roles,
            input.local_slot,
            None,
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
    const WARM_LOCAL_KV_BONUS_USD: f64 = 0.01;
    let local_utility = value_usd * local_p
        + if eligible_warm_local {
            WARM_LOCAL_KV_BONUS_USD
        } else {
            0.0
        };
    let require_better_p_pass = matches!(input.policy, RoutingPolicy::PreferLocal { .. });
    let mut best_cloud: Option<(&CloudCandidate, f64, f64)> = None;
    for cand in input.cloud_candidates {
        if !cloud_eligible(cand) {
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
            None,
        );
        let basis = match input.policy {
            RoutingPolicy::PreferLocal { .. } => RoutingBasis::PreferLocal,
            RoutingPolicy::Utility { .. } => {
                if eligible_warm_local
                    && best_cloud.is_some_and(|(_, best_u, _)| best_u > value_usd * local_p)
                {
                    RoutingBasis::WarmKvStayLocal
                } else {
                    RoutingBasis::UtilityLocal
                }
            }
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
        cloud_stage_two_cost_weight,
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

    fn synthetic_catalog() -> super::super::heads::ArcRuntimeCatalog {
        use super::super::heads::{ArcModelEntry, ArcRuntimeCatalog};
        let models: Vec<ArcModelEntry> = (0..56)
            .map(|i| ArcModelEntry {
                name: format!("slot-{i}"),
                wire: None,
            })
            .collect();
        ArcRuntimeCatalog {
            models,
            actions: Vec::new(),
            fallback_action: 0,
            model_to_candidate: (0..56).map(Some).collect(),
            candidate_latency_factor: vec![None; 56],
            stage_one_regime: 0,
            stage_one_cost_weight: 0.3,
            stage_one_latency_weight: 0.0,
            stage_two_cost_weight: 0.3,
            stage_two_support_columns: Vec::new(),
            model_prices: vec![None; 56],
            pii_classes: Vec::new(),
            pii_presence_threshold: 0.5,
            pii_type_threshold: 0.5,
            pii_length_buckets: Vec::new(),
        }
    }

    #[test]
    fn local_profile_calibration_and_logit_bias_are_monotone() {
        let catalog = synthetic_catalog();
        assert_eq!(
            infer_local_profile("nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ", &catalog),
            LocalProfile {
                slot: LOCAL_FLASH_NEXT_SLOT,
                logit_bias: 0.0,
            }
        );
        assert_eq!(
            infer_local_profile("nchapman/Ornith-1.5-35B-A3B-W4A16-HGPTQ", &catalog),
            LocalProfile {
                slot: LOCAL_9B_SLOT,
                logit_bias: 0.45,
            }
        );
        assert_eq!(
            infer_local_profile("nchapman/Ornith-1.5-9B-W4A16-HGPTQ", &catalog),
            LocalProfile {
                slot: LOCAL_9B_SLOT,
                logit_bias: 0.20,
            }
        );
        assert_eq!(
            infer_local_profile("nchapman/Apex-35B-V1.0-W4A16-HGPTQ", &catalog),
            LocalProfile {
                slot: LOCAL_9B_SLOT,
                logit_bias: 0.20,
            }
        );
        assert_eq!(
            infer_local_profile("nchapman/Ternary-Bonsai-2.5-27B-warp34", &catalog),
            LocalProfile {
                slot: DEFAULT_LOCAL_SLOT,
                logit_bias: 0.40,
            }
        );
        assert_eq!(
            infer_local_profile("nchapman/Qwen3.5-9B-W4A16-HGPTQ", &catalog),
            LocalProfile {
                slot: LOCAL_9B_SLOT,
                logit_bias: 0.0,
            }
        );
        assert_eq!(
            infer_local_profile("nchapman/Qwen3.5-4B-W4A16-HGPTQ", &catalog),
            LocalProfile {
                slot: LOCAL_9B_SLOT,
                logit_bias: 0.0,
            }
        );
        assert_eq!(
            infer_local_profile("nchapman/Qwen3.8-27B-W4A16-HGPTQ", &catalog),
            LocalProfile {
                slot: DEFAULT_LOCAL_SLOT,
                logit_bias: 0.0,
            }
        );
        assert_eq!(
            infer_local_profile("custom-local-model", &catalog),
            LocalProfile {
                slot: DEFAULT_LOCAL_SLOT,
                logit_bias: 0.0,
            }
        );

        // Logit-space bias is identity at 0 and strictly monotone.
        assert!((apply_logit_bias(0.75, 0.0) - 0.75).abs() < 1e-12);
        // Parent directory names must not skew basename matching; 7B/14B map to 9B anchor.
        assert_eq!(
            infer_local_profile("/models/9b-experiments/Qwen3.8-27B-W4A16", &catalog),
            LocalProfile {
                slot: DEFAULT_LOCAL_SLOT,
                logit_bias: 0.0,
            }
        );
        assert_eq!(
            infer_local_profile("Qwen/Qwen2.5-14B-Instruct", &catalog),
            LocalProfile {
                slot: LOCAL_9B_SLOT,
                logit_bias: 0.0,
            }
        );

        let p_up = apply_logit_bias(0.75, 0.20);
        let p_down = apply_logit_bias(0.75, -0.50);
        assert!(p_up > 0.75 && p_up < 1.0);
        assert!(p_down < 0.75 && p_down > 0.0);
        assert!((p_up - 0.7856).abs() < 0.002);
    }

    #[test]
    fn remote_archetype_slot_mapping_covers_standard_and_frontier() {
        let catalog = synthetic_catalog();
        // Explicit tier prefixes: vendor-neutral (`standard`/`frontier`) + Claude (`sonnet`/`opus`).
        for tok in ["standard", "sonnet"] {
            assert_eq!(
                infer_remote_slot(Some(tok), "x/y", "y", None, &catalog).unwrap(),
                REMOTE_SONNET_SLOT
            );
        }
        for tok in ["frontier", "opus"] {
            assert_eq!(
                infer_remote_slot(Some(tok), "x/y", "y", None, &catalog).unwrap(),
                REMOTE_OPUS_SLOT
            );
        }
        for removed in ["fast", "medium", "mid", "strong", "bogus"] {
            assert!(infer_remote_slot(Some(removed), "x/y", "y", None, &catalog).is_err());
        }

        // Name cues for swappable Sonnet/Opus models (including dash/dot variants).
        assert_eq!(
            infer_remote_slot(
                None,
                "opencode/claude-sonnet-5-5",
                "claude-sonnet-5-5",
                Some([2.0, 0.2, 2.5, 10.0]),
                &catalog
            )
            .unwrap(),
            REMOTE_SONNET_SLOT
        );
        assert_eq!(
            infer_remote_slot(
                None,
                "opencode/claude-opus-5-5",
                "claude-opus-5-5",
                Some([4.0, 0.4, 5.0, 20.0]),
                &catalog
            )
            .unwrap(),
            REMOTE_OPUS_SLOT
        );
        // Price-based tiering for non-Claude swappable cloud models.
        assert_eq!(
            infer_remote_slot(
                None,
                "openai/gpt-5.4",
                "gpt-5.4",
                Some([2.5, 0.25, 0.0, 15.0]),
                &catalog
            )
            .unwrap(),
            REMOTE_SONNET_SLOT
        );
        assert_eq!(
            infer_remote_slot(
                None,
                "openai/gpt-5.4-pro",
                "gpt-5.4-pro",
                Some([30.0, 3.0, 0.0, 180.0]),
                &catalog
            )
            .unwrap(),
            REMOTE_OPUS_SLOT
        );
        assert_eq!(
            infer_remote_slot(None, "custom/unknown", "unknown", None, &catalog).unwrap(),
            REMOTE_SONNET_SLOT
        );
        assert_eq!(remote_tier_label(REMOTE_SONNET_SLOT), "standard");
        assert_eq!(remote_tier_label(REMOTE_OPUS_SLOT), "frontier");
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
