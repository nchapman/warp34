//! Routing policy, local model slot auto-mapping, upstream presets, and decision logic.

use super::{
    heads::{ArcHeads, ArcRuntimeCatalog, LEGS, PiiSignalPrediction, StageOnePrediction},
    registry::{ModelCapabilities, ModelsDevRegistry, ProviderEndpoint, infer_tier_from_prices},
};
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
    /// Per-candidate upstream endpoint resolved via `models.dev` (None in pure unit tests).
    pub endpoint: Option<ProviderEndpoint>,
    /// Live USD/Mtok `[input, cache_read, cache_write, output]` from `models.dev` (overrides slot price).
    pub price_override: Option<[f64; LEGS]>,
    /// Model capabilities & valid reasoning effort levels from `models.dev`.
    pub capabilities: ModelCapabilities,
}

fn find_native_arc_slot(catalog: &ArcRuntimeCatalog, id: &str) -> Option<usize> {
    let wanted = id.trim().to_ascii_lowercase();
    let dotted = wanted.replace('-', ".");
    let dashed = wanted.replace('.', "-");
    catalog.models.iter().enumerate().find_map(|(idx, m)| {
        let name_lc = m.name.to_ascii_lowercase();
        let wire_lc = m.wire.as_deref().map(str::to_ascii_lowercase);
        let matches = name_lc == wanted
            || name_lc == dotted
            || name_lc == dashed
            || wire_lc
                .as_deref()
                .is_some_and(|w| w == wanted || w == dotted || w == dashed);
        matches.then_some(idx)
    })
}

/// Resolves cloud candidates through the `models.dev` registry, auto-detecting provider base URLs,
/// API keys from environment variables, live USD/Mtok pricing, and model capabilities.
pub fn resolve_cloud_candidates_from_registry<F>(
    registry: &ModelsDevRegistry,
    catalog: &ArcRuntimeCatalog,
    cli_api_key: Option<&str>,
    filter: &[String],
    env_lookup: F,
) -> Result<Vec<CloudCandidate>, String>
where
    F: Fn(&str) -> Option<String>,
{
    if filter.is_empty()
        || (filter.len() == 1
            && matches!(
                filter[0].trim().to_ascii_lowercase().as_str(),
                "none" | "local"
            ))
    {
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
        let resolved = registry.resolve_spec(raw, cli_api_key, &env_lookup)?;
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
            endpoint: Some(resolved.endpoint),
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

/// Filter cloud candidate slots by optional user-supplied model names, wire IDs, or `tier=wire`
/// overrides (e.g. `medium=meta/muse-spark-1.3-contributor,strong=google/gemini-4-neon`).
pub fn resolve_cloud_candidates(
    upstream: &UpstreamTarget,
    catalog: &ArcRuntimeCatalog,
    filter: &[String],
) -> Result<Vec<CloudCandidate>, String> {
    let base = upstream.default_cloud_slots(catalog);
    if filter.is_empty() {
        return Ok(base
            .into_iter()
            .map(|slot| {
                let m = &catalog.models[slot];
                CloudCandidate {
                    slot,
                    name: m.name.clone(),
                    wire: m.wire.clone().unwrap_or_else(|| m.name.clone()),
                    endpoint: None,
                    price_override: None,
                    capabilities: ModelCapabilities::default(),
                }
            })
            .collect());
    }
    let mut matched = Vec::new();
    for raw in filter {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((lhs, rhs)) = trimmed.split_once('=') {
            let lhs_l = lhs.trim().to_ascii_lowercase();
            let wire = rhs.trim().to_string();
            if wire.is_empty() {
                return Err(format!(
                    "invalid --router-cloud-models override `{trimmed}`"
                ));
            }
            let slot = resolve_tier_slot(&lhs_l, catalog)
                .or_else(|| {
                    base.iter().copied().find(|&idx| {
                        let m = &catalog.models[idx];
                        m.name.eq_ignore_ascii_case(&lhs_l)
                            || m.wire
                                .as_deref()
                                .is_some_and(|w| w.eq_ignore_ascii_case(&lhs_l))
                    })
                })
                .ok_or_else(|| {
                    format!(
                        "unknown tier/slot `{lhs}` in --router-cloud-models `{trimmed}` (use fast, medium, strong, or a built-in ARC model name)"
                    )
                })?;
            matched.push(CloudCandidate {
                slot,
                name: wire.clone(),
                wire,
                endpoint: None,
                price_override: None,
                capabilities: ModelCapabilities::default(),
            });
        } else {
            let wanted = trimmed.to_ascii_lowercase();
            let slot = base
                .iter()
                .copied()
                .find(|&idx| {
                    let m = &catalog.models[idx];
                    m.name.eq_ignore_ascii_case(&wanted)
                        || m.wire
                            .as_deref()
                            .is_some_and(|w| w.eq_ignore_ascii_case(&wanted))
                })
                .ok_or_else(|| {
                    format!(
                        "--router-cloud-models `{trimmed}` matched no eligible models for upstream {}",
                        upstream.base_url()
                    )
                })?;
            let m = &catalog.models[slot];
            matched.push(CloudCandidate {
                slot,
                name: m.name.clone(),
                wire: m.wire.clone().unwrap_or_else(|| m.name.clone()),
                endpoint: None,
                price_override: None,
                capabilities: ModelCapabilities::default(),
            });
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
        endpoint: Option<ProviderEndpoint>,
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
/// ARC-1.0 encodes no-thinking as `level == Some("none")` (e.g. `("default", Some("none"))`).
pub(crate) fn map_arc_effort_to_warp34(effort: &str, level: Option<&str>) -> &'static str {
    match (effort, level) {
        ("none", _) | (_, Some("none")) => "none",
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

    // 1. Local-only mode (no cloud upstream configured).
    if input.cloud_candidates.is_empty() {
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
            basis: RoutingBasis::LocalOnlyEffort,
            pii_triggered: pii.triggered_classes,
            latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
        });
    }

    // 2. Automatic PII / credential local pinning.
    if pii_pin {
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
            basis: RoutingBasis::PiiPinnedLocal,
            pii_triggered: pii.triggered_classes,
            latency_ms: t0.elapsed().as_secs_f64() * 1000.0,
        });
    }

    // 3. KV-cache boundary awareness: stay local when prefix cache is warm on a continuing session.
    if input.warm_local_kv {
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

    let stay_local = match (input.policy, best_cloud) {
        (_, None) => true,
        (RoutingPolicy::Utility { .. }, Some((_, best_u, _))) => local_utility >= best_u,
        (RoutingPolicy::PreferLocal { .. }, Some(_)) => false,
    };

    if stay_local {
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
    }

    let (cand, _, cloud_p) = best_cloud.unwrap();
    let raw_effort = choose_effort_for_slot(
        heads,
        input.block_readouts,
        input.block_roles,
        cand.slot,
        cand.price_override,
    );
    let effort = cand.capabilities.clamp_effort(&raw_effort);
    Ok(RoutingDecision {
        target: RouteTarget::Cloud {
            slot: cand.slot,
            name: cand.name.clone(),
            wire: cand.wire.clone(),
            endpoint: cand.endpoint.clone(),
        },
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
    fn stage_two_level_none_maps_to_none_effort() {
        assert_eq!(map_arc_effort_to_warp34("default", Some("none")), "none");
        assert_eq!(map_arc_effort_to_warp34("high", Some("none")), "none");
        assert_eq!(map_arc_effort_to_warp34("none", None), "none");
        assert_eq!(map_arc_effort_to_warp34("default", Some("down")), "low");
        assert_eq!(map_arc_effort_to_warp34("default", Some("up")), "high");
        assert_eq!(map_arc_effort_to_warp34("high", Some("down")), "low");
        assert_eq!(map_arc_effort_to_warp34("medium", None), "medium");
    }
}
