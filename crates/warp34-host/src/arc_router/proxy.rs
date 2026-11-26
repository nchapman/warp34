//! Async HTTP proxy for cloud spillover requests (streaming SSE and JSON).

use super::{
    input::ApiProtocol,
    policy::{RoutingDecision, UpstreamTarget},
};
use axum::{
    body::Body,
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::Response,
};
use serde_json::Value;
use tokio_stream::StreamExt;

pub struct UpstreamProxy {
    client: reqwest::Client,
    target: UpstreamTarget,
    api_key: Option<String>,
}

impl UpstreamProxy {
    pub fn new(target: UpstreamTarget, api_key: Option<String>) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .read_timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| format!("build upstream HTTP client: {e}"))?;
        Ok(Self {
            client,
            target,
            api_key,
        })
    }

    /// Returns true if `protocol` can be forwarded natively to this upstream without schema mismatch.
    pub fn supports_protocol(&self, protocol: ApiProtocol) -> bool {
        match self.target {
            UpstreamTarget::Anthropic => protocol == ApiProtocol::AnthropicMessages,
            UpstreamTarget::OpenRouter | UpstreamTarget::OpenAi | UpstreamTarget::Custom { .. } => {
                matches!(
                    protocol,
                    ApiProtocol::OpenAiChat | ApiProtocol::OpenAiResponses
                )
            }
        }
    }

    pub fn target(&self) -> &UpstreamTarget {
        &self.target
    }

    /// Forward a request to the configured cloud upstream, rewriting `"model"` to `wire_model`
    /// and streaming the response body + `x-warp34-route*` headers directly back to the client.
    pub async fn forward(
        &self,
        protocol: ApiProtocol,
        mut body: Value,
        wire_model: &str,
        decision: &RoutingDecision,
    ) -> Response {
        let (url, model_for_upstream) = match (&self.target, protocol) {
            (UpstreamTarget::Anthropic, ApiProtocol::AnthropicMessages) => {
                let stripped = wire_model.strip_prefix("anthropic/").unwrap_or(wire_model);
                (
                    format!("{}/v1/messages", self.target.base_url()),
                    stripped.to_string(),
                )
            }
            (UpstreamTarget::OpenAi, ApiProtocol::OpenAiResponses) => {
                let stripped = wire_model.strip_prefix("openai/").unwrap_or(wire_model);
                (
                    format!("{}/responses", self.target.base_url()),
                    stripped.to_string(),
                )
            }
            (UpstreamTarget::OpenAi, _) => {
                let stripped = wire_model.strip_prefix("openai/").unwrap_or(wire_model);
                (
                    format!("{}/chat/completions", self.target.base_url()),
                    stripped.to_string(),
                )
            }
            (
                UpstreamTarget::OpenRouter | UpstreamTarget::Custom { .. },
                ApiProtocol::OpenAiResponses,
            ) => (
                format!("{}/responses", self.target.base_url()),
                wire_model.to_string(),
            ),
            (UpstreamTarget::OpenRouter | UpstreamTarget::Custom { .. }, _) => (
                format!("{}/chat/completions", self.target.base_url()),
                wire_model.to_string(),
            ),
            (UpstreamTarget::Anthropic, _) => {
                let stripped = wire_model.strip_prefix("anthropic/").unwrap_or(wire_model);
                (
                    format!("{}/v1/messages", self.target.base_url()),
                    stripped.to_string(),
                )
            }
        };

        if let Some(obj) = body.as_object_mut() {
            obj.insert("model".to_string(), Value::String(model_for_upstream));
            // Strip warp34-only fields that strict upstream providers reject.
            obj.remove("chat_template_kwargs");
            obj.remove("reasoning_effort");
        }

        let mut req = self.client.post(&url).json(&body);
        if let Some(ref key) = self.api_key {
            match self.target {
                UpstreamTarget::Anthropic => {
                    req = req
                        .header("x-api-key", key)
                        .header("anthropic-version", "2023-06-01");
                }
                _ => {
                    req = req.bearer_auth(key);
                }
            }
        }
        if matches!(self.target, UpstreamTarget::OpenRouter) {
            req = req
                .header("HTTP-Referer", "https://github.com/nchapman/warp34")
                .header("X-Title", "warp34 ARC Router");
        }

        let upstream_resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                let msg = serde_json::json!({
                    "error": {
                        "message": format!("ARC cloud upstream error ({url}): {e}"),
                        "type": "upstream_error",
                    }
                });
                let mut resp = Response::builder()
                    .status(StatusCode::BAD_GATEWAY)
                    .header("content-type", "application/json")
                    .body(Body::from(msg.to_string()))
                    .unwrap();
                super::apply_routing_headers(resp.headers_mut(), decision);
                return resp;
            }
        };

        let status = StatusCode::from_u16(upstream_resp.status().as_u16())
            .unwrap_or(StatusCode::BAD_GATEWAY);
        let content_type = upstream_resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/json")
            .to_string();

        let stream = upstream_resp
            .bytes_stream()
            .map(|res| res.map_err(std::io::Error::other));
        let mut headers = HeaderMap::new();
        if let Ok(val) = HeaderValue::from_str(&content_type) {
            headers.insert(HeaderName::from_static("content-type"), val);
        }
        super::apply_routing_headers(&mut headers, decision);

        let mut builder = Response::builder().status(status);
        if let Some(h) = builder.headers_mut() {
            *h = headers;
        }
        builder.body(Body::from_stream(stream)).unwrap()
    }
}
