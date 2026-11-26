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
        (resp, is_err)
    }
}
