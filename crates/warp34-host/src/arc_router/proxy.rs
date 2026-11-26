//! Async HTTP proxy for multi-provider cloud spillover requests (streaming SSE and JSON).

use super::{
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
    /// response body + `x-warp34-route*` headers directly back to the client.
    pub async fn forward(
        &self,
        endpoint: &ProviderEndpoint,
        protocol: ApiProtocol,
        mut body: Value,
        wire_model: &str,
        decision: &RoutingDecision,
    ) -> Response {
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
        }

        // OpenRouter's Gemini `functionResponse` translator runs a broken `{...}` extractor on
        // non-JSON `role: "tool"` strings (turning Python `.format()` `{}` into `[{},{},{},{}]`).
        // Wrapping non-JSON-object tool strings in `{"result": s}` prevents output corruption.
        if endpoint.dialect == WireDialect::OpenRouter
            && protocol == ApiProtocol::OpenAiChat
            && wire_model.starts_with("google/")
        {
            sanitize_openrouter_gemini_tool_messages(&mut body);
        }

        let mut req = self.client.post(&url).json(&body);
        if endpoint.dialect == WireDialect::AnthropicNative {
            req = req.header("anthropic-version", "2023-06-01");
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
                let msg = format!(
                    "{{\"error\":{{\"message\":\"ARC cloud upstream ({}) request failed: {e}\",\"type\":\"upstream_error\"}}}}",
                    endpoint.provider_id
                );
                let mut resp = Response::new(Body::from(msg));
                *resp.status_mut() = StatusCode::BAD_GATEWAY;
                resp.headers_mut().insert(
                    axum::http::header::CONTENT_TYPE,
                    HeaderValue::from_static("application/json"),
                );
                super::apply_routing_headers(resp.headers_mut(), decision);
                return resp;
            }
        };

        let status = StatusCode::from_u16(upstream_resp.status().as_u16())
            .unwrap_or(StatusCode::BAD_GATEWAY);
        let content_type = upstream_resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);

        let stream = upstream_resp.bytes_stream().map(|chunk| {
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
        resp
    }
}

/// Wraps non-JSON-object `role: "tool"` string contents in `{"result": s}` so OpenRouter's
/// Gemini `functionResponse` adapter does not run its broken `{...}` JSON-salvage extractor
/// over raw Python/bash tool outputs.
pub fn sanitize_openrouter_gemini_tool_messages(body: &mut Value) {
    let Some(msgs) = body.get_mut("messages").and_then(|v| v.as_array_mut()) else {
        return;
    };
    for msg in msgs {
        if msg.get("role").and_then(|v| v.as_str()) != Some("tool") {
            continue;
        }
        let Some(content) = msg.get("content").and_then(|v| v.as_str()) else {
            continue;
        };
        let is_json_object = serde_json::from_str::<Value>(content)
            .ok()
            .is_some_and(|v| v.is_object());
        if !is_json_object {
            let wrapped = serde_json::json!({ "result": content }).to_string();
            if let Some(obj) = msg.as_object_mut() {
                obj.insert("content".to_string(), Value::String(wrapped));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openrouter_gemini_guard_wraps_plain_tool_output_and_preserves_json_objects() {
        let mut body = serde_json::json!({
            "model": "google/gemini-4-hydrogen",
            "messages": [
                {"role": "tool", "tool_call_id": "c1", "content": "n_inputs={}, n_outputs={}"},
                {"role": "tool", "tool_call_id": "c2", "content": "{\"ok\":true}"},
                {"role": "user", "content": "hi"}
            ]
        });
        sanitize_openrouter_gemini_tool_messages(&mut body);
        let c0 = body["messages"][0]["content"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(c0).expect("valid json");
        assert_eq!(parsed["result"], "n_inputs={}, n_outputs={}");
        assert_eq!(body["messages"][1]["content"], "{\"ok\":true}");
        assert_eq!(body["messages"][2]["content"], "hi");
    }
}
