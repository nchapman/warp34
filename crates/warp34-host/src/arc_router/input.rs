//! Canonical `arc-role-blocks-v2` conversation serializer for ARC-1.0 routing.
//!
//! Converts OpenAI Chat Completions, OpenAI Responses API, or Anthropic Messages
//! request payloads into the `<|im_start|>{role}\n{body}<|im_end|>\n` token stream
//! expected by `ARC-1.0`'s Layer-11 encoder.

use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiProtocol {
    OpenAiChat,
    OpenAiResponses,
    AnthropicMessages,
}

#[derive(Debug, Clone)]
pub struct CanonicalTurn {
    pub text: String,
    pub last_role: &'static str,
    pub has_tools: bool,
    pub turn_count: usize,
}

fn push_block(out: &mut String, role: &str, body: &str) {
    let json_body = serde_json::to_string(body.trim()).unwrap_or_else(|_| "\"\"".to_string());
    out.push_str("<|im_start|>");
    out.push_str(role);
    out.push('\n');
    out.push_str(&json_body);
    out.push_str("<|im_end|>\n");
}

fn extract_text_content(val: &Value) -> String {
    match val {
        Value::String(s) => s.clone(),
        Value::Array(arr) => {
            let mut buf = String::new();
            for item in arr {
                let kind = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if matches!(kind, "text" | "input_text" | "output_text")
                    && let Some(t) = item.get("text").and_then(|v| v.as_str())
                {
                    if !buf.is_empty() {
                        buf.push('\n');
                    }
                    buf.push_str(t);
                } else if kind == "tool_use" {
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("tool");
                    let input = item.get("input").cloned().unwrap_or(Value::Null);
                    if !buf.is_empty() {
                        buf.push('\n');
                    }
                    buf.push_str(&format!("<tool_call name=\"{name}\">{input}</tool_call>"));
                } else if kind == "tool_result" {
                    let content = item
                        .get("content")
                        .map(extract_text_content)
                        .unwrap_or_default();
                    if !buf.is_empty() {
                        buf.push('\n');
                    }
                    buf.push_str(&content);
                } else if kind == "image_url" || kind == "image" || kind == "input_image" {
                    if !buf.is_empty() {
                        buf.push('\n');
                    }
                    buf.push_str("<image>");
                }
            }
            buf
        }
        _ => String::new(),
    }
}

fn normalize_role(role: &str) -> &'static str {
    match role {
        "system" | "developer" => "system",
        "assistant" => "assistant",
        "tool" | "function" => "tool",
        _ => "user",
    }
}

/// Serialize a request JSON body into ARC-1.0 canonical role blocks.
pub fn serialize_request(protocol: ApiProtocol, body: &Value) -> CanonicalTurn {
    let mut out = String::with_capacity(2048);
    let mut last_role: &'static str = "user";
    let mut turn_count = 0usize;
    let has_tools = body
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty());

    // Include compact tool definitions header when tools are declared (matches `include_recorded`).
    let tool_summary = if has_tools {
        let names: Vec<&str> = body["tools"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|t| {
                t.get("function")
                    .and_then(|f| f.get("name"))
                    .or_else(|| t.get("name"))
                    .and_then(|v| v.as_str())
            })
            .collect();
        if names.is_empty() {
            None
        } else {
            Some(format!("Available tools: {}", names.join(", ")))
        }
    } else {
        None
    };

    match protocol {
        ApiProtocol::OpenAiChat => {
            if let Some(ref ts) = tool_summary
                && !body
                    .get("messages")
                    .and_then(|v| v.as_array())
                    .is_some_and(|ms| {
                        ms.first()
                            .and_then(|m| m.get("role"))
                            .and_then(|r| r.as_str())
                            .is_some_and(|r| r == "system" || r == "developer")
                    })
            {
                push_block(&mut out, "system", ts);
            }
            if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
                for (i, msg) in msgs.iter().enumerate() {
                    let role =
                        normalize_role(msg.get("role").and_then(|v| v.as_str()).unwrap_or("user"));
                    let mut text = msg
                        .get("content")
                        .map(extract_text_content)
                        .unwrap_or_default();
                    if i == 0
                        && role == "system"
                        && let Some(ref ts) = tool_summary
                    {
                        if !text.is_empty() {
                            text.push_str("\n\n");
                        }
                        text.push_str(ts);
                    }
                    if let Some(tcs) = msg.get("tool_calls").and_then(|v| v.as_array()) {
                        for tc in tcs {
                            let name = tc
                                .pointer("/function/name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("tool");
                            let args = tc
                                .pointer("/function/arguments")
                                .and_then(|v| v.as_str())
                                .unwrap_or("{}");
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(&format!(
                                "<tool_call name=\"{name}\">{args}</tool_call>"
                            ));
                        }
                    }
                    push_block(&mut out, role, &text);
                    last_role = role;
                    if role != "system" {
                        turn_count += 1;
                    }
                }
            }
        }
        ApiProtocol::AnthropicMessages => {
            let mut sys = body
                .get("system")
                .map(extract_text_content)
                .unwrap_or_default();
            if let Some(ref ts) = tool_summary {
                if !sys.is_empty() {
                    sys.push_str("\n\n");
                }
                sys.push_str(ts);
            }
            if !sys.is_empty() {
                push_block(&mut out, "system", &sys);
            }
            if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
                for msg in msgs {
                    let role =
                        normalize_role(msg.get("role").and_then(|v| v.as_str()).unwrap_or("user"));
                    let text = msg
                        .get("content")
                        .map(extract_text_content)
                        .unwrap_or_default();
                    push_block(&mut out, role, &text);
                    last_role = role;
                    turn_count += 1;
                }
            }
        }
        ApiProtocol::OpenAiResponses => {
            let mut sys = body
                .get("instructions")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if let Some(ref ts) = tool_summary {
                if !sys.is_empty() {
                    sys.push_str("\n\n");
                }
                sys.push_str(ts);
            }
            if !sys.is_empty() {
                push_block(&mut out, "system", &sys);
            }
            match body.get("input") {
                Some(Value::String(s)) => {
                    push_block(&mut out, "user", s);
                    last_role = "user";
                    turn_count = 1;
                }
                Some(Value::Array(items)) => {
                    for item in items {
                        let role = normalize_role(
                            item.get("role").and_then(|v| v.as_str()).unwrap_or("user"),
                        );
                        let text = item
                            .get("content")
                            .map(extract_text_content)
                            .or_else(|| {
                                item.get("output")
                                    .and_then(|v| v.as_str())
                                    .map(String::from)
                            })
                            .unwrap_or_default();
                        push_block(&mut out, role, &text);
                        last_role = role;
                        if role != "system" {
                            turn_count += 1;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    if out.is_empty() {
        push_block(&mut out, "user", "");
    }

    CanonicalTurn {
        text: out,
        last_role,
        has_tools,
        turn_count,
    }
}
