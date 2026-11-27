//! Async HTTP proxy for multi-provider cloud spillover requests (streaming SSE and JSON).

use std::sync::Arc;

use super::{
    RouterMetrics,
    input::ApiProtocol,
    policy::RoutingDecision,
    registry::{ProviderEndpoint, WireDialect},
};
use axum::{
    body::Body,
    http::{HeaderValue, StatusCode},
    response::Response,
};
use serde_json::Value;
use tokio_stream::StreamExt;

#[derive(Debug, Default, PartialEq, Eq)]
struct ParsedUsage {
    prompt_tokens: u64,
    cached_tokens: u64,
    cache_write_tokens: u64,
    completion_tokens: u64,
}

/// Extracts token usage from OpenAI Chat, OpenAI Responses, or Anthropic Messages JSON objects.
fn extract_usage_from_json(v: &Value) -> Option<ParsedUsage> {
    let usage = v
        .get("usage")
        .or_else(|| v.pointer("/response/usage"))
        .or_else(|| v.pointer("/message/usage"))?;
    if !usage.is_object() {
        return None;
    }
    let prompt = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached = usage
        .pointer("/prompt_tokens_details/cached_tokens")
        .or_else(|| usage.pointer("/input_tokens_details/cached_tokens"))
        .or_else(|| usage.get("cache_read_input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_write = usage
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if prompt == 0 && completion == 0 && cached == 0 && cache_write == 0 {
        return None;
    }
    Some(ParsedUsage {
        prompt_tokens: prompt,
        cached_tokens: cached,
        cache_write_tokens: cache_write,
        completion_tokens: completion,
    })
}

/// Bounded SSE/JSON usage tracker that merges incremental usage events (e.g. Anthropic
/// `message_start` + `message_delta`) and commits a single `record_cloud_usage` call on `Drop`
/// (when the response stream finishes or closes), with zero added TTFB and bounded memory.
struct UsageStreamTracker {
    metrics: Arc<RouterMetrics>,
    model: String,
    prices: [f64; 4],
    is_sse: bool,
    /// Incomplete SSE line carry-over across TCP chunk boundaries (bounded to 64 KiB).
    sse_line_buf: Vec<u8>,
    sse_line_overflow: bool,
    /// Bounded non-SSE JSON body capture (up to 2 MiB) so large bodies never OOM.
    json_buf: Vec<u8>,
    json_overflow: bool,
    /// Merged peak usage seen across the stream (prevents double-counting cumulative events).
    merged: ParsedUsage,
}

const MAX_SSE_LINE_BYTES: usize = 64 * 1024;
const MAX_JSON_USAGE_BYTES: usize = 2 * 1024 * 1024;

impl UsageStreamTracker {
    fn new(metrics: Arc<RouterMetrics>, model: String, prices: [f64; 4], is_sse: bool) -> Self {
        Self {
            metrics,
            model,
            prices,
            is_sse,
            sse_line_buf: Vec::new(),
            sse_line_overflow: false,
            json_buf: Vec::new(),
            json_overflow: false,
            merged: ParsedUsage::default(),
        }
    }

    fn merge(&mut self, u: ParsedUsage) {
        self.merged.prompt_tokens = self.merged.prompt_tokens.max(u.prompt_tokens);
        self.merged.cached_tokens = self.merged.cached_tokens.max(u.cached_tokens);
        self.merged.cache_write_tokens = self.merged.cache_write_tokens.max(u.cache_write_tokens);
        self.merged.completion_tokens = self.merged.completion_tokens.max(u.completion_tokens);
    }

    fn process_sse_line(&mut self, line: &[u8]) {
        let Ok(text) = std::str::from_utf8(line) else {
            return;
        };
        let Some(data) = text.trim_end_matches('\r').strip_prefix("data:") else {
            return;
        };
        let trimmed = data.trim();
        if trimmed.is_empty() || trimmed == "[DONE]" || !trimmed.contains("\"usage\"") {
            return;
        }
        if let Ok(val) = serde_json::from_str::<Value>(trimmed)
            && let Some(u) = extract_usage_from_json(&val)
        {
            self.merge(u);
        }
    }

    fn push_chunk(&mut self, chunk: &[u8]) {
        if self.is_sse {
            for &b in chunk {
                if b == b'\n' {
                    if !self.sse_line_overflow && !self.sse_line_buf.is_empty() {
                        let line = std::mem::take(&mut self.sse_line_buf);
                        self.process_sse_line(&line);
                    } else {
                        self.sse_line_buf.clear();
                    }
                    self.sse_line_overflow = false;
                } else if !self.sse_line_overflow {
                    if self.sse_line_buf.len() < MAX_SSE_LINE_BYTES {
                        self.sse_line_buf.push(b);
                    } else {
                        self.sse_line_buf.clear();
                        self.sse_line_overflow = true;
                    }
                }
            }
        } else if !self.json_overflow {
            if self.json_buf.len().saturating_add(chunk.len()) <= MAX_JSON_USAGE_BYTES {
                self.json_buf.extend_from_slice(chunk);
            } else {
                self.json_buf.clear();
                self.json_overflow = true;
            }
        }
    }
}

impl Drop for UsageStreamTracker {
    fn drop(&mut self) {
        if self.is_sse {
            if !self.sse_line_overflow && !self.sse_line_buf.is_empty() {
                let line = std::mem::take(&mut self.sse_line_buf);
                self.process_sse_line(&line);
            }
        } else if !self.json_overflow
            && !self.json_buf.is_empty()
            && let Ok(val) = serde_json::from_slice::<Value>(&self.json_buf)
            && let Some(u) = extract_usage_from_json(&val)
        {
            self.merge(u);
        }

        let u = &self.merged;
        if u.prompt_tokens > 0
            || u.completion_tokens > 0
            || u.cached_tokens > 0
            || u.cache_write_tokens > 0
        {
            self.metrics.record_cloud_usage(
                &self.model,
                self.prices,
                u.prompt_tokens,
                u.cached_tokens,
                u.cache_write_tokens,
                u.completion_tokens,
            );
        }
    }
}

pub struct UpstreamProxy {
    client: reqwest::Client,
}

impl UpstreamProxy {
    pub fn new() -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .read_timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| format!("build upstream HTTP client: {e}"))?;
        Ok(Self { client })
    }

    /// Returns true if `protocol` can be forwarded natively to `endpoint` without schema mismatch.
    pub fn supports_protocol(endpoint: &ProviderEndpoint, protocol: ApiProtocol) -> bool {
        match endpoint.dialect {
            WireDialect::AnthropicNative => protocol == ApiProtocol::AnthropicMessages,
            WireDialect::OpenRouter | WireDialect::OpenAiNative => {
                matches!(
                    protocol,
                    ApiProtocol::OpenAiChat | ApiProtocol::OpenAiResponses
                )
            }
            WireDialect::OpenAiChatCompat => protocol == ApiProtocol::OpenAiChat,
        }
    }

    /// Forward a request to `endpoint`, rewriting `"model"` to `wire_model` and streaming the
    /// response body + `x-warp34-route*` headers directly back to the client. Records actual cloud
    /// usage & USD spend in `metrics` when reported by the upstream.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward(
        &self,
        endpoint: &ProviderEndpoint,
        protocol: ApiProtocol,
        mut body: Value,
        wire_model: &str,
        model_display_name: &str,
        prices: [f64; 4],
        metrics: Arc<RouterMetrics>,
        decision: &RoutingDecision,
        anthropic_beta: Option<&str>,
    ) -> (Response, bool) {
        let base = endpoint.base_url.trim_end_matches('/');
        let url = match (endpoint.dialect, protocol) {
            (WireDialect::AnthropicNative, ApiProtocol::AnthropicMessages) => {
                if base.ends_with("/v1") {
                    format!("{base}/messages")
                } else {
                    format!("{base}/v1/messages")
                }
            }
            (WireDialect::OpenRouter | WireDialect::OpenAiNative, ApiProtocol::OpenAiResponses) => {
                format!("{base}/responses")
            }
            _ => {
                if base.ends_with("/chat/completions") {
                    base.to_string()
                } else {
                    format!("{base}/chat/completions")
                }
            }
        };

        if let Some(obj) = body.as_object_mut() {
            obj.insert("model".to_string(), Value::String(wire_model.to_string()));
            // Strip local-only extension fields that strict native OpenAI/Anthropic APIs reject.
            if matches!(
                endpoint.dialect,
                WireDialect::OpenAiNative | WireDialect::AnthropicNative
            ) {
                for local_ext in [
                    "chat_template_kwargs",
                    "thinking_budget",
                    "top_k",
                    "min_p",
                    "repetition_penalty",
                ] {
                    obj.remove(local_ext);
                }
            }
        }

        let mut req = self.client.post(&url).json(&body);
        if endpoint.dialect == WireDialect::AnthropicNative {
            req = req.header("anthropic-version", "2023-06-01");
            if let Some(beta) = anthropic_beta.filter(|s| !s.is_empty()) {
                req = req.header("anthropic-beta", beta);
            }
            if let Some(key) = &endpoint.api_key {
                req = req.header("x-api-key", key);
            }
        } else {
            if let Some(key) = &endpoint.api_key {
                req = req.bearer_auth(key);
            }
            if endpoint.dialect == WireDialect::OpenRouter {
                req = req
                    .header("HTTP-Referer", "https://github.com/warp34/warp34")
                    .header("X-Title", "warp34");
            }
        }

        let upstream_resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                let err_msg = format!(
                    "ARC cloud upstream ({}) request failed: {}",
                    endpoint.provider_id,
                    e.without_url()
                );
                let payload = match protocol {
                    ApiProtocol::AnthropicMessages => serde_json::json!({
                        "type": "error",
                        "error": { "type": "api_error", "message": err_msg }
                    }),
                    _ => serde_json::json!({
                        "error": { "message": err_msg, "type": "upstream_error" }
                    }),
                };
                let mut resp = Response::new(Body::from(payload.to_string()));
                *resp.status_mut() = StatusCode::BAD_GATEWAY;
                resp.headers_mut().insert(
                    axum::http::header::CONTENT_TYPE,
                    HeaderValue::from_static("application/json"),
                );
                super::apply_routing_headers(resp.headers_mut(), decision);
                return (resp, true);
            }
        };

        let status = StatusCode::from_u16(upstream_resp.status().as_u16())
            .unwrap_or(StatusCode::BAD_GATEWAY);
        let is_err = !status.is_success();
        let content_type = upstream_resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);

        let is_sse = content_type
            .as_deref()
            .is_some_and(|ct| ct.to_ascii_lowercase().contains("text/event-stream"));

        let mut tracker = if status.is_success() {
            Some(UsageStreamTracker::new(
                metrics,
                model_display_name.to_string(),
                prices,
                is_sse,
            ))
        } else {
            None
        };
        let stream = upstream_resp.bytes_stream().map(move |chunk| {
            if let (Some(t), Ok(bytes)) = (tracker.as_mut(), &chunk) {
                t.push_chunk(bytes);
            }
            chunk.map_err(|e| std::io::Error::other(format!("upstream stream read error: {e}")))
        });

        let mut resp = Response::new(Body::from_stream(stream));
        *resp.status_mut() = status;
        if let Some(ct) = content_type
            && let Ok(val) = HeaderValue::from_str(&ct)
        {
            resp.headers_mut()
                .insert(axum::http::header::CONTENT_TYPE, val);
        }
        if status.is_success() {
            resp.headers_mut().insert(
                axum::http::header::CACHE_CONTROL,
                HeaderValue::from_static("no-cache"),
            );
        }
        super::apply_routing_headers(resp.headers_mut(), decision);
        (resp, is_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_openai_anthropic_and_split_sse_usage_without_double_counting() {
        let openai = serde_json::json!({
            "usage": {
                "prompt_tokens": 5000,
                "completion_tokens": 400,
                "prompt_tokens_details": { "cached_tokens": 2000 }
            }
        });
        assert_eq!(
            extract_usage_from_json(&openai),
            Some(ParsedUsage {
                prompt_tokens: 5000,
                cached_tokens: 2000,
                cache_write_tokens: 0,
                completion_tokens: 400,
            })
        );

        // Anthropic SSE splits input usage across `message_start` and output across `message_delta`,
        // and TCP framing can split a `data:` line mid-JSON. Verify reassembly + single accounting.
        let metrics = Arc::new(RouterMetrics::default());
        {
            let mut tracker = UsageStreamTracker::new(
                Arc::clone(&metrics),
                "anthropic/claude-sonnet-4.6".to_string(),
                [2.0, 0.1, 2.5, 10.0],
                true,
            );
            tracker.push_chunk(
                b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":",
            );
            tracker.push_chunk(
                b"{\"input_tokens\":5000,\"cache_read_input_tokens\":2000,\"output_tokens\":1}}}\n\n",
            );
            tracker.push_chunk(
                b"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":400}}\n\n",
            );
        }
        let snap = metrics.snapshot();
        // 3000 non-cached in ($0.006) + 2000 cached ($0.0002) + 400 out ($0.004) = $0.0102
        assert!((snap.total_spend_usd - 0.0102).abs() < 1e-9);
        assert_eq!(snap.models.len(), 1);
        assert_eq!(snap.models[0].prompt_tokens, 5000);
        assert_eq!(snap.models[0].cached_tokens, 2000);
        assert_eq!(snap.models[0].completion_tokens, 400);

        // Non-SSE JSON stream commits once on Drop.
        let metrics_json = Arc::new(RouterMetrics::default());
        {
            let mut tracker = UsageStreamTracker::new(
                Arc::clone(&metrics_json),
                "openai/gpt-5.4".to_string(),
                [2.5, 0.25, 0.0, 15.0],
                false,
            );
            tracker.push_chunk(b"{\"id\":\"resp_1\",\"usage\":{\"input_tokens\":1000,");
            tracker.push_chunk(b"\"output_tokens\":200}}");
        }
        let snap_json = metrics_json.snapshot();
        assert_eq!(snap_json.models[0].prompt_tokens, 1000);
        assert_eq!(snap_json.models[0].completion_tokens, 200);
    }
}
