//! `models.dev` provider & model registry for zero-ceremony ARC-1.0 cloud routing.
//!
//! Fetches and caches `https://models.dev/api.json` at `~/.cache/warp34/models.dev.json`
//! (24h TTL, offline-safe) so users can write `--router-cloud-models provider/model` (e.g.
//! `anthropic/claude-sonnet-4-6`, `openrouter/google/gemini-2.5-pro`, `https://host/v1#model`)
//! and automatically resolve:
//! - Provider base URL and wire dialect (`OpenAiChatCompat`, `AnthropicMessages`, `OpenAiResponses`)
//! - API key from the provider's `models.dev` `env` list (with automatic fallback to `openrouter`
//!   when `OPENROUTER_API_KEY` is set and the direct provider key is not)
//! - Live 4-tuple pricing `[input, cache_read, cache_write, output]` (USD/Mtok) for ARC scoring
//! - Capabilities (`tool_call`, vision `image`, `limit.context`) and valid `reasoning_options` efforts

use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

const MODELS_DEV_URL: &str = "https://models.dev/api.json";
const CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);
const FETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// Upstream HTTP wire dialect used to talk to a resolved cloud provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireDialect {
    /// OpenAI-compatible `/chat/completions` (covers ~300 `models.dev` providers, including
    /// OpenRouter, Google Gemini `/v1beta/openai`, DeepSeek, xAI, Fireworks, Together, Groq).
    OpenAiChatCompat,
    /// Native OpenAI (`/chat/completions` + `/responses`).
    OpenAiNative,
    /// Native Anthropic (`/v1/messages`).
    AnthropicNative,
    /// OpenRouter (`/chat/completions` + `/responses`, with OpenRouter-specific headers & guards).
    OpenRouter,
}

/// Resolved upstream endpoint for a single cloud candidate.
#[derive(Debug, Clone)]
pub struct ProviderEndpoint {
    pub provider_id: String,
    pub base_url: String,
    pub dialect: WireDialect,
    pub api_key: Option<String>,
    pub env_hint: String,
}

/// Model capabilities & reasoning constraints hydrated from `models.dev`.
#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    pub tool_call: bool,
    pub supports_image: bool,
    pub context_limit: Option<usize>,
    /// Allowed `reasoning.effort` values (empty means no explicit effort list / unrestricted).
    pub effort_values: Vec<String>,
    /// Whether the model exposes a `toggle` reasoning option (can disable thinking explicitly).
    pub supports_toggle: bool,
    pub supports_reasoning: bool,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            tool_call: true,
            supports_image: true,
            context_limit: None,
            effort_values: Vec::new(),
            supports_toggle: false,
            supports_reasoning: true,
        }
    }
}

impl ModelCapabilities {
    /// Clamps an ARC-selected effort (`none`, `low`, `medium`, `high`, `xhigh`, `max`) to the
    /// nearest supported effort value declared in `models.dev`, or returns `effort` unchanged when
    /// the model does not publish an explicit effort list.
    ///
    /// Toggle-capable models (`supports_toggle == true`) preserve `"none"` so `proxy.rs` can
    /// explicitly disable thinking (`{"enabled": false}`), whereas non-toggle always-reasoning
    /// models (e.g. `gpt-5.4-pro`, `gemini-4-argon`) clamp `"none"` to their minimum supported
    /// effort rather than sending an unsupported `"none"` (HTTP 400) or omitting `reasoning`
    /// (which falls back to a higher provider default like `"medium"`/`"high"`).
    pub fn clamp_effort(&self, effort: &str) -> String {
        if effort == "none" && self.supports_toggle {
            return "none".to_string();
        }
        if self.effort_values.is_empty() {
            return match effort {
                "xhigh" | "max" => "high".to_string(),
                _ => effort.to_string(),
            };
        }
        if self.effort_values.iter().any(|v| v == effort) {
            return effort.to_string();
        }
        const ORDER: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];
        let want_idx = ORDER.iter().position(|&v| v == effort).unwrap_or(3);
        self.effort_values
            .iter()
            .min_by_key(|cand| {
                let c_idx = ORDER.iter().position(|&v| v == cand.as_str()).unwrap_or(3);
                want_idx.abs_diff(c_idx)
            })
            .cloned()
            .unwrap_or_else(|| effort.to_string())
    }
}

/// Full resolution result for one `--router-cloud-models` entry.
#[derive(Debug, Clone)]
pub struct ResolvedCloudSpec {
    /// Optional explicit tier override (`fast`, `medium`, `strong`, `frontier`).
    pub explicit_tier: Option<String>,
    /// Wire model ID sent in the upstream JSON `"model"` field.
    pub wire_model: String,
    /// Canonical `provider/model` display & catalog-lookup ID.
    pub canonical_id: String,
    /// Resolved provider endpoint + API key.
    pub endpoint: ProviderEndpoint,
    /// Live 4-tuple pricing `[input, cache_read, cache_write, output]` (USD/Mtok) if known.
    pub prices: Option<[f64; 4]>,
    /// Model capabilities from `models.dev` (or permissive defaults if unknown).
    pub capabilities: ModelCapabilities,
}

#[derive(Debug, Clone, Deserialize)]
struct RawProvider {
    #[serde(default)]
    id: String,
    #[serde(default)]
    api: Option<String>,
    #[serde(default)]
    npm: Option<String>,
    #[serde(default)]
    env: Vec<String>,
    #[serde(default)]
    models: HashMap<String, RawModel>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawModel {
    #[serde(default)]
    id: String,
    #[serde(default = "default_true")]
    tool_call: bool,
    #[serde(default)]
    reasoning: bool,
    #[serde(default)]
    reasoning_options: Vec<RawReasoningOption>,
    #[serde(default)]
    modalities: Option<RawModalities>,
    #[serde(default)]
    cost: Option<RawCost>,
    #[serde(default)]
    limit: Option<RawLimit>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
struct RawReasoningOption {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    values: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawModalities {
    #[serde(default)]
    input: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawCost {
    #[serde(default)]
    input: f64,
    #[serde(default)]
    output: f64,
    #[serde(default)]
    cache_read: f64,
    #[serde(default)]
    cache_write: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct RawLimit {
    #[serde(default)]
    context: Option<usize>,
}

/// Loaded `models.dev` registry (300+ providers, offline-safe).
#[derive(Debug, Clone)]
pub struct ModelsDevRegistry {
    providers: HashMap<String, RawProvider>,
}

impl ModelsDevRegistry {
    /// Loads `models.dev` from `~/.cache/warp34/models.dev.json`, refreshing from
    /// `https://models.dev/api.json` if missing or older than 24h (unless `offline` is set).
    /// Never fails hard: falls back to stale cache or built-in defaults if offline.
    pub fn load_cached_or_fetch() -> Self {
        Self::load_cached_or_fetch_with_options(false)
    }

    /// Loads `models.dev` honoring an explicit `offline` flag (never performs network I/O when
    /// `offline == true`).
    pub fn load_cached_or_fetch_with_options(offline: bool) -> Self {
        let cache_path = cache_file_path();
        let cached_bytes = cache_path.as_ref().and_then(|p| std::fs::read(p).ok());
        let is_fresh = cache_path
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .and_then(|m| m.modified().ok())
            .and_then(|t| SystemTime::now().duration_since(t).ok())
            .is_some_and(|age| age < CACHE_TTL);

        if (is_fresh || offline)
            && let Some(bytes) = &cached_bytes
            && let Ok(reg) = Self::from_json_bytes(bytes)
        {
            return reg;
        }

        if !offline
            && let Some(bytes) = fetch_models_dev_blocking()
            && let Ok(reg) = Self::from_json_bytes(&bytes)
        {
            if let Some(path) = &cache_path {
                write_cache_atomic(path, &bytes);
            }
            return reg;
        }

        if let Some(bytes) = &cached_bytes
            && let Ok(reg) = Self::from_json_bytes(bytes)
        {
            return reg;
        }

        Self::builtin_fallback()
    }

    /// Parses a `models.dev/api.json` payload.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut providers: HashMap<String, RawProvider> =
            serde_json::from_slice(bytes).map_err(|e| format!("parse models.dev json: {e}"))?;
        for (k, v) in &mut providers {
            if v.id.is_empty() {
                v.id.clone_from(k);
            }
        }
        Ok(Self { providers })
    }

    /// Minimal offline fallback covering major providers when `models.dev` has never been cached.
    pub fn builtin_fallback() -> Self {
        let mut providers = HashMap::new();
        for (id, api, npm, env) in [
            (
                "openrouter",
                Some("https://openrouter.ai/api/v1"),
                Some("@openrouter/ai-sdk-provider"),
                vec!["OPENROUTER_API_KEY"],
            ),
            (
                "openai",
                Some("https://api.openai.com/v1"),
                Some("@ai-sdk/openai"),
                vec!["OPENAI_API_KEY"],
            ),
            (
                "anthropic",
                Some("https://api.anthropic.com"),
                Some("@ai-sdk/anthropic"),
                vec!["ANTHROPIC_API_KEY"],
            ),
            (
                "google",
                Some("https://generativelanguage.googleapis.com/v1beta/openai"),
                Some("@ai-sdk/google"),
                vec![
                    "GOOGLE_API_KEY",
                    "GOOGLE_GENERATIVE_AI_API_KEY",
                    "GEMINI_API_KEY",
                ],
            ),
            (
                "deepseek",
                Some("https://api.deepseek.com/v1"),
                Some("@ai-sdk/openai-compatible"),
                vec!["DEEPSEEK_API_KEY"],
            ),
            (
                "xai",
                Some("https://api.x.ai/v1"),
                Some("@ai-sdk/xai"),
                vec!["XAI_API_KEY"],
            ),
            (
                "groq",
                Some("https://api.groq.com/openai/v1"),
                Some("@ai-sdk/groq"),
                vec!["GROQ_API_KEY"],
            ),
            (
                "togetherai",
                Some("https://api.together.xyz/v1"),
                Some("@ai-sdk/togetherai"),
                vec!["TOGETHER_API_KEY"],
            ),
            (
                "fireworks-ai",
                Some("https://api.fireworks.ai/inference/v1"),
                Some("@ai-sdk/openai-compatible"),
                vec!["FIREWORKS_API_KEY"],
            ),
        ] {
            providers.insert(
                id.to_string(),
                RawProvider {
                    id: id.to_string(),
                    api: api.map(ToString::to_string),
                    npm: npm.map(ToString::to_string),
                    env: env.into_iter().map(ToString::to_string).collect(),
                    models: HashMap::new(),
                },
            );
        }
        Self { providers }
    }

    /// Returns true if `provider_id` is known in the registry.
    pub fn has_provider(&self, provider_id: &str) -> bool {
        self.providers.contains_key(provider_id)
    }

    /// Resolves a provider's `(base_url, dialect, env_vars)` from `models.dev` + well-known defaults.
    pub fn resolve_provider_meta(
        &self,
        provider_id: &str,
    ) -> Option<(String, WireDialect, Vec<String>)> {
        // Handle custom URLs directly (`http://...` or `https://...`).
        if provider_id.starts_with("http://") || provider_id.starts_with("https://") {
            return Some((
                provider_id.trim_end_matches('/').to_string(),
                WireDialect::OpenAiChatCompat,
                vec!["WARP34_ROUTER_API_KEY".to_string()],
            ));
        }
        let prov = self.providers.get(provider_id)?;
        let (default_url, dialect) = provider_defaults(provider_id, prov.npm.as_deref());
        let base_url = prov
            .api
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(|s| s.trim_end_matches('/').to_string())
            .or_else(|| default_url.map(ToString::to_string))?;
        Some((base_url, dialect, prov.env.clone()))
    }

    /// Resolves a single `--router-cloud-models` spec (e.g. `google/gemini-4-hydrogen`,
    /// `openrouter/anthropic/claude-sonnet-4-6`, `standard=deepseek/deepseek-chat`,
    /// `frontier=openai/gpt-5.4-pro`, or `https://gateway.example.com/v1#my-model`) using
    /// `env_lookup` for API keys.
    pub fn resolve_spec<F>(
        &self,
        raw_spec: &str,
        env_lookup: F,
    ) -> Result<ResolvedCloudSpec, String>
    where
        F: Fn(&str) -> Option<String>,
    {
        let trimmed = raw_spec.trim();
        let (explicit_tier, rest) = if let Some((lhs, rhs)) = trimmed.split_once('=') {
            let tier = lhs.trim().to_ascii_lowercase();
            if matches!(tier.as_str(), "standard" | "sonnet" | "frontier" | "opus") {
                (Some(tier), rhs.trim())
            } else {
                (None, trimmed)
            }
        } else {
            (None, trimmed)
        };
        if rest.is_empty() {
            return Err(format!(
                "cloud model spec `{raw_spec}` has an empty model id"
            ));
        }

        let pick_key = |env_vars: &[String]| -> Option<String> {
            for var in env_vars {
                if let Some(val) = env_lookup(var).filter(|s| !s.is_empty()) {
                    return Some(val);
                }
            }
            None
        };

        // Support inline custom OpenAI-compatible gateways: `https://host/v1#model-id`
        // (preserving exact URL path casing).
        let (provider_id, wire_model, prefix_prov, suffix_model) =
            if rest.starts_with("http://") || rest.starts_with("https://") {
                let (url, model) = rest.split_once('#').ok_or_else(|| {
                format!(
                    "custom gateway `{raw_spec}` must specify a model as `https://host/v1#<model>`"
                )
            })?;
                let model = model.trim();
                if model.is_empty() {
                    return Err(format!("custom gateway `{raw_spec}` has an empty model id"));
                }
                (
                    url.trim_end_matches('/').to_string(),
                    model.to_string(),
                    None,
                    model,
                )
            } else {
                // Parse optional explicit `provider:model` or `provider/model` (case-insensitive provider prefix).
                let (prefix_prov, suffix_model): (Option<String>, &str) =
                    if let Some((p, m)) = rest.split_once(':') {
                        let p_lc = p.trim().to_ascii_lowercase();
                        if self.has_provider(&p_lc) {
                            (Some(p_lc), m.trim())
                        } else {
                            (None, rest)
                        }
                    } else if let Some((p, m)) = rest.split_once('/') {
                        let p_lc = p.trim().to_ascii_lowercase();
                        if self.has_provider(&p_lc) {
                            (Some(p_lc), m.trim())
                        } else {
                            (None, rest)
                        }
                    } else {
                        (None, rest)
                    };
                if prefix_prov.is_some() && suffix_model.is_empty() {
                    return Err(format!(
                        "cloud model spec `{raw_spec}` has an empty model id"
                    ));
                }

                // Prefer direct provider when its env key is present, otherwise fall back to
                // OpenRouter if OPENROUTER_API_KEY is present. Explicit `openrouter/<vendor>/<model>`
                // always stays on OpenRouter.
                let openrouter_key = self
                    .resolve_provider_meta("openrouter")
                    .and_then(|(_, _, envs)| pick_key(&envs));

                let (prov, wire) = if let Some(ref p) = prefix_prov {
                    if p == "openrouter" {
                        ("openrouter".to_string(), suffix_model.to_string())
                    } else {
                        let direct_key = self
                            .resolve_provider_meta(p)
                            .and_then(|(_, _, envs)| pick_key(&envs));
                        if direct_key.is_some() {
                            (p.clone(), suffix_model.to_string())
                        } else if openrouter_key.is_some() {
                            ("openrouter".to_string(), format!("{p}/{suffix_model}"))
                        } else {
                            (p.clone(), suffix_model.to_string())
                        }
                    }
                } else if openrouter_key.is_some() {
                    ("openrouter".to_string(), rest.to_string())
                } else if let Some(p) = self.find_provider_for_bare_model(rest, &env_lookup) {
                    (p, rest.to_string())
                } else {
                    ("openrouter".to_string(), rest.to_string())
                };
                (prov, wire, prefix_prov, suffix_model)
            };

        let (base_url, dialect, env_vars) =
            self.resolve_provider_meta(&provider_id).ok_or_else(|| {
                format!("unknown router upstream provider `{provider_id}` (not in models.dev)")
            })?;
        let api_key = pick_key(&env_vars);
        let env_hint = env_vars
            .first()
            .cloned()
            .unwrap_or_else(|| "WARP34_ROUTER_API_KEY".to_string());

        let canonical_id = if provider_id == "openrouter" {
            wire_model.clone()
        } else {
            format!("{provider_id}/{wire_model}")
        };

        let raw_model = self.lookup_model(
            &provider_id,
            &wire_model,
            prefix_prov.as_deref(),
            suffix_model,
        );
        let prices = raw_model.and_then(extract_prices);
        let capabilities = raw_model.map(extract_capabilities).unwrap_or_default();

        Ok(ResolvedCloudSpec {
            explicit_tier,
            wire_model,
            canonical_id,
            endpoint: ProviderEndpoint {
                provider_id,
                base_url,
                dialect,
                api_key,
                env_hint,
            },
            prices,
            capabilities,
        })
    }

    /// Auto-detects default cloud models when `--router-cloud-models auto` is passed.
    pub fn default_models_for_env<F>(&self, env_lookup: F) -> Vec<String>
    where
        F: Fn(&str) -> Option<String>,
    {
        let has_key = |prov: &str| -> bool {
            self.resolve_provider_meta(prov)
                .is_some_and(|(_, _, envs)| {
                    envs.iter()
                        .any(|e| env_lookup(e).is_some_and(|v| !v.is_empty()))
                })
        };
        let mut specs = Vec::new();
        if has_key("openrouter") {
            specs.push("openrouter/anthropic/claude-sonnet-5.5".to_string());
            specs.push("openrouter/anthropic/claude-opus-5.5".to_string());
            return specs;
        }
        if has_key("anthropic") {
            specs.push("anthropic/claude-sonnet-5.5".to_string());
            specs.push("anthropic/claude-opus-5.5".to_string());
        }
        if has_key("google") {
            specs.push("standard=google/gemini-4-hydrogen".to_string());
            specs.push("frontier=google/gemini-4-argon".to_string());
        }
        if has_key("openai") {
            specs.push("standard=openai/gpt-5.6-terra".to_string());
        }
        if has_key("deepseek") {
            specs.push("standard=deepseek/deepseek-v4-pro".to_string());
        }
        specs
    }

    fn find_provider_for_bare_model<F>(&self, model_id: &str, env_lookup: &F) -> Option<String>
    where
        F: Fn(&str) -> Option<String>,
    {
        for (pid, prov) in &self.providers {
            if prov.models.contains_key(model_id)
                && prov
                    .env
                    .iter()
                    .any(|e| env_lookup(e).is_some_and(|v| !v.is_empty()))
            {
                return Some(pid.clone());
            }
        }
        None
    }

    fn lookup_model(
        &self,
        provider_id: &str,
        wire_model: &str,
        prefix_prov: Option<&str>,
        suffix_model: &str,
    ) -> Option<&RawModel> {
        let find_in = |pid: &str, mid: &str| -> Option<&RawModel> {
            let prov = self.providers.get(pid)?;
            if let Some(m) = prov.models.get(mid) {
                return Some(m);
            }
            // Handle minor dot/dash differences (e.g. `claude-sonnet-5.5` vs `claude-sonnet-5-5`).
            let dashed = mid.replace('.', "-");
            if let Some(m) = prov.models.get(&dashed) {
                return Some(m);
            }
            let dotted = mid.replace('-', ".");
            prov.models.get(&dotted).or_else(|| {
                prov.models
                    .values()
                    .find(|m| m.id.eq_ignore_ascii_case(mid))
            })
        };

        find_in(provider_id, wire_model)
            .or_else(|| prefix_prov.and_then(|p| find_in(p, suffix_model)))
            .or_else(|| find_in("openrouter", wire_model))
    }
}

/// Infers the generalized remote ARC-1.0 archetype (`"standard"` or `"frontier"`) from live
/// `models.dev` USD/Mtok output pricing when no explicit `standard=`/`sonnet=` or
/// `frontier=`/`opus=` prefix or model name token was supplied.
pub fn infer_tier_from_prices(prices: [f64; 4]) -> &'static str {
    if prices[3] > 15.0 {
        "frontier"
    } else {
        "standard"
    }
}

fn provider_defaults(provider_id: &str, npm: Option<&str>) -> (Option<&'static str>, WireDialect) {
    match provider_id {
        "openrouter" => (
            Some("https://openrouter.ai/api/v1"),
            WireDialect::OpenRouter,
        ),
        "openai" => (Some("https://api.openai.com/v1"), WireDialect::OpenAiNative),
        "anthropic" => (
            Some("https://api.anthropic.com"),
            WireDialect::AnthropicNative,
        ),
        "google" => (
            Some("https://generativelanguage.googleapis.com/v1beta/openai"),
            WireDialect::OpenAiChatCompat,
        ),
        "xai" => (Some("https://api.x.ai/v1"), WireDialect::OpenAiChatCompat),
        "groq" => (
            Some("https://api.groq.com/openai/v1"),
            WireDialect::OpenAiChatCompat,
        ),
        "togetherai" => (
            Some("https://api.together.xyz/v1"),
            WireDialect::OpenAiChatCompat,
        ),
        "cerebras" => (
            Some("https://api.cerebras.ai/v1"),
            WireDialect::OpenAiChatCompat,
        ),
        "mistral" => (
            Some("https://api.mistral.ai/v1"),
            WireDialect::OpenAiChatCompat,
        ),
        _ => match npm {
            Some("@openrouter/ai-sdk-provider") => (
                Some("https://openrouter.ai/api/v1"),
                WireDialect::OpenRouter,
            ),
            Some("@ai-sdk/anthropic") => (
                Some("https://api.anthropic.com"),
                WireDialect::AnthropicNative,
            ),
            Some("@ai-sdk/openai") => {
                (Some("https://api.openai.com/v1"), WireDialect::OpenAiNative)
            }
            Some("@ai-sdk/google") => (
                Some("https://generativelanguage.googleapis.com/v1beta/openai"),
                WireDialect::OpenAiChatCompat,
            ),
            _ => (None, WireDialect::OpenAiChatCompat),
        },
    }
}

fn extract_prices(m: &RawModel) -> Option<[f64; 4]> {
    let c = m.cost.as_ref()?;
    if c.input <= 0.0 && c.output <= 0.0 {
        return None;
    }
    Some([c.input, c.cache_read, c.cache_write, c.output])
}

fn extract_capabilities(m: &RawModel) -> ModelCapabilities {
    let supports_image = m
        .modalities
        .as_ref()
        .is_none_or(|modl| modl.input.iter().any(|x| x == "image"));
    let effort_values = m
        .reasoning_options
        .iter()
        .find(|o| o.kind == "effort")
        .map(|o| o.values.clone())
        .unwrap_or_default();
    let supports_toggle = m.reasoning_options.iter().any(|o| o.kind == "toggle");
    ModelCapabilities {
        tool_call: m.tool_call,
        supports_image,
        context_limit: m.limit.as_ref().and_then(|l| l.context),
        effort_values,
        supports_toggle,
        supports_reasoning: m.reasoning,
    }
}

fn cache_file_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("warp34").join("models.dev.json"))
}

fn fetch_models_dev_blocking() -> Option<Vec<u8>> {
    const MAX_REGISTRY_BYTES: usize = 32 * 1024 * 1024;
    // Run a tiny async reqwest fetch on an isolated current-thread runtime so calling from
    // either sync startup code or inside a tokio runtime never panics.
    std::thread::spawn(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .ok()?;
        rt.block_on(async {
            let client = reqwest::Client::builder()
                .timeout(FETCH_TIMEOUT)
                .build()
                .ok()?;
            let resp = client.get(MODELS_DEV_URL).send().await.ok()?;
            if !resp.status().is_success() {
                return None;
            }
            if resp
                .content_length()
                .is_some_and(|n| n as usize > MAX_REGISTRY_BYTES)
            {
                return None;
            }
            let bytes = resp.bytes().await.ok()?;
            (bytes.len() <= MAX_REGISTRY_BYTES).then(|| bytes.to_vec())
        })
    })
    .join()
    .ok()?
}

fn write_cache_atomic(path: &PathBuf, bytes: &[u8]) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
    if std::fs::write(&tmp, bytes).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_registry() -> ModelsDevRegistry {
        let json = r#"{
            "openrouter": {
                "id": "openrouter",
                "api": "https://openrouter.ai/api/v1",
                "npm": "@openrouter/ai-sdk-provider",
                "env": ["OPENROUTER_API_KEY"],
                "models": {
                    "google/gemini-4-hydrogen": {
                        "id": "google/gemini-4-hydrogen",
                        "tool_call": true,
                        "reasoning": true,
                        "reasoning_options": [{"type": "effort", "values": ["low", "medium", "high", "xhigh"]}],
                        "modalities": {"input": ["text", "image"]},
                        "cost": {"input": 1.5, "output": 7.5, "cache_read": 0.075, "cache_write": 0.083},
                        "limit": {"context": 1048576}
                    }
                }
            },
            "google": {
                "id": "google",
                "npm": "@ai-sdk/google",
                "env": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
                "models": {
                    "gemini-4-argon": {
                        "id": "gemini-4-argon",
                        "tool_call": true,
                        "reasoning": true,
                        "reasoning_options": [{"type": "effort", "values": ["low", "medium", "high"]}],
                        "modalities": {"input": ["text", "image"]},
                        "cost": {"input": 2.0, "output": 10.0, "cache_read": 0.1},
                        "limit": {"context": 1048576}
                    }
                }
            },
            "deepseek": {
                "id": "deepseek",
                "api": "https://api.deepseek.com",
                "npm": "@ai-sdk/openai-compatible",
                "env": ["DEEPSEEK_API_KEY"],
                "models": {
                    "deepseek-v4-pro": {
                        "id": "deepseek-v4-pro",
                        "tool_call": true,
                        "reasoning": true,
                        "reasoning_options": [{"type": "toggle"}, {"type": "effort", "values": ["low", "high", "max"]}],
                        "modalities": {"input": ["text"]},
                        "cost": {"input": 0.3, "output": 1.2, "cache_read": 0.03},
                        "limit": {"context": 1000000}
                    }
                }
            }
        }"#;
        ModelsDevRegistry::from_json_bytes(json.as_bytes()).expect("parse sample")
    }

    #[test]
    fn auto_detects_direct_provider_key_or_falls_back_to_openrouter() {
        let reg = sample_registry();

        // Direct GEMINI_API_KEY set -> hits Google OpenAI-compat directly!
        let direct = reg
            .resolve_spec("google/gemini-4-argon", |k| {
                (k == "GEMINI_API_KEY").then(|| "gm-key".to_string())
            })
            .expect("resolve direct google");
        assert_eq!(direct.endpoint.provider_id, "google");
        assert_eq!(
            direct.endpoint.base_url,
            "https://generativelanguage.googleapis.com/v1beta/openai"
        );
        assert_eq!(direct.wire_model, "gemini-4-argon");
        assert_eq!(direct.endpoint.api_key.as_deref(), Some("gm-key"));
        assert_eq!(direct.prices, Some([2.0, 0.1, 0.0, 10.0]));
        assert_eq!(direct.capabilities.clamp_effort("xhigh"), "high");

        // No GEMINI_API_KEY, but OPENROUTER_API_KEY set -> auto-falls back to OpenRouter!
        let via_or = reg
            .resolve_spec("google/gemini-4-hydrogen", |k| {
                (k == "OPENROUTER_API_KEY").then(|| "or-key".to_string())
            })
            .expect("resolve via openrouter");
        assert_eq!(via_or.endpoint.provider_id, "openrouter");
        assert_eq!(via_or.wire_model, "google/gemini-4-hydrogen");
        assert_eq!(via_or.endpoint.api_key.as_deref(), Some("or-key"));
        assert_eq!(via_or.prices, Some([1.5, 0.075, 0.083, 7.5]));

        // Explicit `openrouter/<vendor>/<model>` forces OpenRouter even if GEMINI_API_KEY is set.
        let forced = reg
            .resolve_spec("openrouter/google/gemini-4-argon", |k| {
                Some(format!("{k}-val"))
            })
            .expect("explicit openrouter prefix");
        assert_eq!(forced.endpoint.provider_id, "openrouter");
        assert_eq!(forced.wire_model, "google/gemini-4-argon");
        assert_eq!(
            forced.endpoint.api_key.as_deref(),
            Some("OPENROUTER_API_KEY-val")
        );
    }

    #[test]
    fn capabilities_and_tier_inference_match_models_dev_metadata() {
        let reg = sample_registry();
        let ds = reg
            .resolve_spec("deepseek/deepseek-v4-pro", |k| {
                (k == "DEEPSEEK_API_KEY").then(|| "ds-key".to_string())
            })
            .expect("resolve deepseek");
        assert_eq!(ds.endpoint.provider_id, "deepseek");
        assert_eq!(ds.endpoint.base_url, "https://api.deepseek.com");
        assert!(!ds.capabilities.supports_image);
        assert!(ds.capabilities.supports_toggle);
        assert_eq!(ds.capabilities.clamp_effort("medium"), "low");
        // Toggle-capable models preserve `"none"` instead of clamping to `"low"`.
        assert_eq!(ds.capabilities.clamp_effort("none"), "none");
        assert_eq!(infer_tier_from_prices(ds.prices.unwrap()), "standard");
        assert_eq!(infer_tier_from_prices([4.0, 0.2, 5.0, 20.0]), "frontier");

        // Non-toggle models without `"none"` in `effort_values` clamp `"none"` to `"low"`.
        let gm = reg
            .resolve_spec("Google/gemini-4-argon", |k| {
                (k == "GEMINI_API_KEY").then(|| "gm-key".to_string())
            })
            .expect("case-insensitive provider prefix");
        assert_eq!(gm.endpoint.provider_id, "google");
        assert!(!gm.capabilities.supports_toggle);
        assert_eq!(gm.capabilities.clamp_effort("none"), "low");
        // Unlisted cloud models (`effort_values.is_empty()`) clamp `"xhigh"`/`"max"` to `"high"`
        // so OpenAI/OpenRouter wire payloads never send an unlisted `"xhigh"` effort.
        let unlisted = ModelCapabilities::default();
        assert_eq!(unlisted.clamp_effort("xhigh"), "high");
        assert_eq!(unlisted.clamp_effort("max"), "high");
        assert_eq!(unlisted.clamp_effort("medium"), "medium");

        // Reject empty model suffix on explicit provider prefixes or tier= prefixes.
        assert!(reg.resolve_spec("google/", |_| None).is_err());
        assert!(reg.resolve_spec("deepseek:", |_| None).is_err());
        assert!(reg.resolve_spec("standard=", |_| None).is_err());

        assert_eq!(
            reg.default_models_for_env(|k| (k == "OPENROUTER_API_KEY").then(|| "k".to_string())),
            vec![
                "openrouter/anthropic/claude-sonnet-5.5",
                "openrouter/anthropic/claude-opus-5.5"
            ]
        );

        // Inline custom gateway URL preserves case-sensitive path segments.
        let custom = reg
            .resolve_spec("https://Gateway.Example.com/CustomPath/v1#my-model", |_| {
                None
            })
            .expect("custom url");
        assert_eq!(
            custom.endpoint.base_url,
            "https://Gateway.Example.com/CustomPath/v1"
        );
        assert_eq!(custom.wire_model, "my-model");
    }
}
