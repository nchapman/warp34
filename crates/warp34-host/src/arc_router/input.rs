//! Canonical `arc-role-blocks-v2` request serializer for OpenAI Chat Completions,
//! OpenAI Responses, and Anthropic Messages payloads, aligned with `rayline-arc-input`.

use std::collections::{BTreeMap, HashMap};

use serde_json::{Map, Value, json};

/// Incoming HTTP API dialect for request normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiProtocol {
    OpenAiChat,
    OpenAiResponses,
    AnthropicMessages,
}

/// One canonical role block (`<|im_start|>{role}\n{canonical_json}<|im_end|>\n`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArcRoleBlock {
    pub role: String,
    pub content: String,
}

impl ArcRoleBlock {
    /// Returns the exact ChatML-framed role block text fed to the ARC encoder tokenizer.
    pub fn serialized(&self) -> String {
        format!("<|im_start|>{}\n{}<|im_end|>\n", self.role, self.content)
    }
}

/// Serialized ARC-1.0 encoder input plus lightweight metadata extracted from the request.
#[derive(Debug, Clone)]
pub struct SerializedArcInput {
    pub blocks: Vec<ArcRoleBlock>,
    pub last_role: String,
    pub has_tools: bool,
    pub turn_count: usize,
}

impl SerializedArcInput {
    /// Concatenated `arc-role-blocks-v2` text across all blocks.
    pub fn serialized_text(&self) -> String {
        let mut out = String::new();
        for b in &self.blocks {
            out.push_str(&b.serialized());
        }
        out
    }
}

/// Formats a `serde_json::Value` matching Python's `json.dumps(v, sort_keys=True, ensure_ascii=False)`
/// (sorted object keys, `", "` / `": "` separators, raw UTF-8, `-0.0` normalized to `0.0`).
pub fn canonical_json_dumps(value: &Value) -> String {
    let mut out = String::new();
    write_canonical_json(value, &mut out);
    out
}

fn write_canonical_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else if let Some(f) = n.as_f64() {
                let f = if f == 0.0 { 0.0 } else { f };
                let s = format!("{f}");
                out.push_str(&s);
                if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                    out.push_str(".0");
                }
            } else {
                out.push_str(&n.to_string());
            }
        }
        Value::String(s) => write_python_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (idx, item) in items.iter().enumerate() {
                if idx > 0 {
                    out.push_str(", ");
                }
                write_canonical_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let sorted: BTreeMap<&str, &Value> = map.iter().map(|(k, v)| (k.as_str(), v)).collect();
            out.push('{');
            for (idx, (k, v)) in sorted.into_iter().enumerate() {
                if idx > 0 {
                    out.push_str(", ");
                }
                write_python_string(k, out);
                out.push_str(": ");
                write_canonical_json(v, out);
            }
            out.push('}');
        }
    }
}

fn write_python_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn strip_billing_header(text: &str) -> String {
    let mut lines = text.lines();
    let Some(first) = lines.next() else {
        return String::new();
    };
    if first
        .trim_start()
        .to_ascii_lowercase()
        .starts_with("x-anthropic-billing-header:")
    {
        let rest: Vec<&str> = lines.collect();
        rest.join("\n").trim_start_matches('\n').to_string()
    } else {
        text.to_string()
    }
}

fn normalize_tool_def(tool: &Value) -> Option<Value> {
    let obj = tool.as_object()?;
    let inner = obj
        .get("function")
        .and_then(Value::as_object)
        .unwrap_or(obj);
    let name = inner
        .get("name")
        .or_else(|| obj.get("name"))
        .and_then(Value::as_str);
    let tool_type = obj
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("function");
    let Some(name) = name.filter(|s| !s.is_empty()) else {
        // Omit nameless server tools (e.g. web_search).
        return None;
    };
    let mut out = Map::new();
    out.insert("type".to_string(), Value::String(tool_type.to_string()));
    out.insert("name".to_string(), Value::String(name.to_string()));
    if let Some(desc) = inner
        .get("description")
        .or_else(|| obj.get("description"))
        .and_then(Value::as_str)
    {
        out.insert("description".to_string(), Value::String(desc.to_string()));
    }
    if let Some(params) = inner
        .get("parameters")
        .or_else(|| inner.get("input_schema"))
        .or_else(|| obj.get("parameters"))
        .or_else(|| obj.get("input_schema"))
    {
        out.insert("parameters".to_string(), params.clone());
    }
    Some(Value::Object(out))
}

fn clean_content_part(part: &Value, is_leading_system: bool) -> Option<Value> {
    if let Some(s) = part.as_str() {
        let text = if is_leading_system {
            strip_billing_header(s)
        } else {
            s.to_string()
        };
        return Some(json!({ "type": "text", "text": text }));
    }
    let obj = part.as_object()?;
    let ptype = obj.get("type").and_then(Value::as_str).unwrap_or("");
    match ptype {
        "text" | "input_text" | "output_text" => {
            let raw = obj.get("text").and_then(Value::as_str).unwrap_or("");
            let text = if is_leading_system {
                strip_billing_header(raw)
            } else {
                raw.to_string()
            };
            Some(json!({ "type": "text", "text": text }))
        }
        "thinking" => {
            let thinking = obj
                .get("thinking")
                .or_else(|| obj.get("text"))
                .and_then(Value::as_str)
                .unwrap_or("");
            Some(json!({ "type": "thinking", "thinking": thinking }))
        }
        "redacted_thinking" => {
            let data = obj.get("data").and_then(Value::as_str).unwrap_or("");
            Some(json!({
                "type": "redacted_thinking",
                "chars": data.chars().count(),
                "count": 1
            }))
        }
        "tool_use" | "server_tool_use" => {
            let mut out = Map::new();
            out.insert("type".to_string(), Value::String(ptype.to_string()));
            if let Some(id) = obj.get("id").and_then(Value::as_str) {
                out.insert("id".to_string(), Value::String(id.to_string()));
            }
            if let Some(name) = obj.get("name").and_then(Value::as_str) {
                out.insert("name".to_string(), Value::String(name.to_string()));
            }
            let input = obj
                .get("input")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new()));
            out.insert("input".to_string(), input);
            Some(Value::Object(out))
        }
        "tool_result" => {
            let mut out = Map::new();
            out.insert("type".to_string(), Value::String("tool_result".to_string()));
            if let Some(id) = obj.get("tool_use_id").and_then(Value::as_str) {
                out.insert("tool_use_id".to_string(), Value::String(id.to_string()));
            }
            if let Some(content) = obj.get("content") {
                if let Some(arr) = content.as_array() {
                    let cleaned: Vec<Value> = arr
                        .iter()
                        .filter_map(|p| clean_content_part(p, false))
                        .collect();
                    out.insert("content".to_string(), Value::Array(cleaned));
                } else {
                    out.insert("content".to_string(), content.clone());
                }
            }
            if obj.get("is_error").and_then(Value::as_bool) == Some(true) {
                out.insert("is_error".to_string(), Value::Bool(true));
            }
            Some(Value::Object(out))
        }
        "image_url" => {
            let url = obj
                .get("image_url")
                .and_then(|u| u.as_str().or_else(|| u.get("url").and_then(Value::as_str)))
                .unwrap_or("");
            Some(json!({
                "type": "image",
                "source": { "type": "url", "url": url }
            }))
        }
        _ => {
            let mut out = obj.clone();
            out.remove("cache_control");
            out.remove("signature");
            if out.get("is_error").and_then(Value::as_bool) == Some(false) {
                out.remove("is_error");
            }
            if out
                .get("citations")
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty)
            {
                out.remove("citations");
            }
            if out
                .get("annotations")
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty)
            {
                out.remove("annotations");
            }
            Some(Value::Object(out))
        }
    }
}

fn extract_parts_from_content(content: Option<&Value>, is_leading_system: bool) -> Vec<Value> {
    match content {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => {
            let text = if is_leading_system {
                strip_billing_header(s)
            } else {
                s.clone()
            };
            vec![json!({ "type": "text", "text": text })]
        }
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|p| clean_content_part(p, is_leading_system))
            .collect(),
        Some(other) => vec![json!({ "type": "text", "text": other.to_string() })],
    }
}

struct CanonicalMessage {
    role: String,
    parts: Vec<Value>,
}

fn push_or_merge(out: &mut Vec<CanonicalMessage>, role: &str, parts: Vec<Value>) {
    if parts.is_empty() {
        return;
    }
    if role != "system"
        && let Some(prev) = out.last_mut()
        && prev.role == role
    {
        prev.parts.extend(parts);
        return;
    }
    out.push(CanonicalMessage {
        role: role.to_string(),
        parts,
    });
}

fn renumber_tool_ids(messages: &mut [CanonicalMessage]) {
    let mut id_map: HashMap<String, String> = HashMap::new();
    let mut next_id = 1usize;
    let mut map_id = |raw: &str| -> String {
        if let Some(existing) = id_map.get(raw) {
            return existing.clone();
        }
        let assigned = format!("t{next_id}");
        next_id += 1;
        id_map.insert(raw.to_string(), assigned.clone());
        assigned
    };

    for msg in messages.iter_mut() {
        for part in &mut msg.parts {
            let Some(obj) = part.as_object_mut() else {
                continue;
            };
            let ptype = obj
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if (ptype == "tool_use" || ptype == "server_tool_use")
                && let Some(id) = obj.get("id").and_then(Value::as_str).map(str::to_string)
            {
                obj.insert("id".to_string(), Value::String(map_id(&id)));
            } else if ptype == "tool_result"
                && let Some(id) = obj
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            {
                obj.insert("tool_use_id".to_string(), Value::String(map_id(&id)));
            }
        }
    }
}

fn extract_tool_definitions(raw: &Value) -> Vec<Value> {
    raw.get("tools")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(normalize_tool_def).collect())
        .unwrap_or_default()
}

/// Serializes an incoming OpenAI / Responses / Anthropic JSON request into canonical
/// `arc-role-blocks-v2` role blocks matching `rayline-arc-input`.
pub fn serialize_request(protocol: ApiProtocol, raw: &Value) -> SerializedArcInput {
    let tool_defs = extract_tool_definitions(raw);
    let has_tools = !tool_defs.is_empty();
    let mut canonical: Vec<CanonicalMessage> = Vec::new();

    if has_tools {
        canonical.push(CanonicalMessage {
            role: "system".to_string(),
            parts: vec![json!({
                "type": "tool_definitions",
                "tools": tool_defs,
            })],
        });
    }

    match protocol {
        ApiProtocol::AnthropicMessages => {
            if let Some(sys) = raw.get("system") {
                let parts = extract_parts_from_content(Some(sys), true);
                push_or_merge(&mut canonical, "system", parts);
            }
            if let Some(messages) = raw.get("messages").and_then(Value::as_array) {
                for msg in messages {
                    let raw_role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
                    let role = match raw_role {
                        "system" | "developer" => "system",
                        "assistant" => "assistant",
                        _ => "user",
                    };
                    let parts = extract_parts_from_content(msg.get("content"), false);
                    push_or_merge(&mut canonical, role, parts);
                }
            }
        }
        ApiProtocol::OpenAiResponses => {
            if let Some(inst) = raw.get("instructions").and_then(Value::as_str) {
                let parts =
                    extract_parts_from_content(Some(&Value::String(inst.to_string())), true);
                push_or_merge(&mut canonical, "system", parts);
            }
            match raw.get("input") {
                Some(Value::String(s)) => {
                    push_or_merge(
                        &mut canonical,
                        "user",
                        vec![json!({ "type": "text", "text": s })],
                    );
                }
                Some(Value::Array(items)) => {
                    for item in items {
                        let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
                        if item_type == "function_call" {
                            let id = item
                                .get("call_id")
                                .or_else(|| item.get("id"))
                                .and_then(Value::as_str)
                                .unwrap_or("");
                            let name = item.get("name").and_then(Value::as_str).unwrap_or("");
                            let args = item
                                .get("arguments")
                                .and_then(Value::as_str)
                                .and_then(|s| serde_json::from_str::<Value>(s).ok())
                                .unwrap_or_else(|| {
                                    item.get("arguments")
                                        .cloned()
                                        .unwrap_or_else(|| Value::Object(Map::new()))
                                });
                            push_or_merge(
                                &mut canonical,
                                "assistant",
                                vec![json!({
                                    "type": "tool_use",
                                    "id": id,
                                    "name": name,
                                    "input": args,
                                })],
                            );
                        } else if item_type == "function_call_output" {
                            let id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                            let content = item
                                .get("output")
                                .cloned()
                                .unwrap_or_else(|| Value::String(String::new()));
                            push_or_merge(
                                &mut canonical,
                                "user",
                                vec![json!({
                                    "type": "tool_result",
                                    "tool_use_id": id,
                                    "content": content,
                                })],
                            );
                        } else {
                            let raw_role =
                                item.get("role").and_then(Value::as_str).unwrap_or("user");
                            let role = match raw_role {
                                "system" | "developer" => "system",
                                "assistant" => "assistant",
                                _ => "user",
                            };
                            let is_lead_sys =
                                role == "system" && canonical.iter().all(|m| m.role != "user");
                            let parts =
                                extract_parts_from_content(item.get("content"), is_lead_sys);
                            push_or_merge(&mut canonical, role, parts);
                        }
                    }
                }
                _ => {}
            }
        }
        ApiProtocol::OpenAiChat => {
            if let Some(messages) = raw.get("messages").and_then(Value::as_array) {
                let mut seen_non_system = false;
                for msg in messages {
                    let raw_role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
                    if raw_role == "tool" || raw_role == "function" {
                        seen_non_system = true;
                        let call_id = msg
                            .get("tool_call_id")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        let content = msg
                            .get("content")
                            .cloned()
                            .unwrap_or_else(|| Value::String(String::new()));
                        push_or_merge(
                            &mut canonical,
                            "user",
                            vec![json!({
                                "type": "tool_result",
                                "tool_use_id": call_id,
                                "content": content,
                            })],
                        );
                        continue;
                    }
                    let role = match raw_role {
                        "system" | "developer" => "system",
                        "assistant" => "assistant",
                        _ => "user",
                    };
                    if role != "system" {
                        seen_non_system = true;
                    }
                    let mut parts = Vec::new();
                    if let Some(reasoning) = msg
                        .get("reasoning_content")
                        .or_else(|| msg.get("reasoning"))
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                    {
                        parts.push(json!({ "type": "thinking", "thinking": reasoning }));
                    }
                    parts.extend(extract_parts_from_content(
                        msg.get("content"),
                        role == "system" && !seen_non_system,
                    ));
                    if let Some(refusal) = msg
                        .get("refusal")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                    {
                        parts.push(json!({ "type": "text", "text": refusal }));
                    }
                    if let Some(tool_calls) = msg.get("tool_calls").and_then(Value::as_array) {
                        for tc in tool_calls {
                            let id = tc.get("id").and_then(Value::as_str).unwrap_or("");
                            let func = tc.get("function").unwrap_or(tc);
                            let name = func.get("name").and_then(Value::as_str).unwrap_or("");
                            let input = match func.get("arguments") {
                                Some(Value::String(s)) => serde_json::from_str::<Value>(s)
                                    .unwrap_or_else(|_| Value::String(s.clone())),
                                Some(v) => v.clone(),
                                None => Value::Object(Map::new()),
                            };
                            parts.push(json!({
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": input,
                            }));
                        }
                    }
                    push_or_merge(&mut canonical, role, parts);
                }
            }
        }
    }

    renumber_tool_ids(&mut canonical);

    let mut blocks: Vec<ArcRoleBlock> = canonical
        .into_iter()
        .map(|m| ArcRoleBlock {
            role: m.role,
            content: canonical_json_dumps(&Value::Array(m.parts)),
        })
        .collect();

    if blocks.is_empty() {
        blocks.push(ArcRoleBlock {
            role: "user".to_string(),
            content: canonical_json_dumps(&json!([{ "type": "text", "text": "" }])),
        });
    }

    let turn_count = blocks.iter().filter(|b| b.role == "user").count();
    let last_role = blocks
        .last()
        .map_or_else(|| "user".to_string(), |b| b.role.clone());

    SerializedArcInput {
        blocks,
        last_role,
        has_tools,
        turn_count,
    }
}
