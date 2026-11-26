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
