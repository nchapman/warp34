//! Golden parity and policy unit tests for the native ARC-1.0 router.

use std::{collections::HashSet, path::PathBuf};
use warp34_host::arc_router::{
    heads::ArcHeads,
    input::{ApiProtocol, TurnEvent, serialize_request},
    policy::{
        DecideInput, HeldTarget, LocalProfile, RouteTarget, RoutingBasis, RoutingPolicy, decide,
        infer_local_profile, resolve_cloud_candidates_from_registry,
    },
    registry::ModelsDevRegistry,
};

fn cached_arc_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snap_root = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--nchapman--ARC-1.0-W4A16-HGPTQ/snapshots");
    let entry = std::fs::read_dir(snap_root).ok()?.next()?.ok()?;
    Some(entry.path())
}

#[test]
fn canonical_serializer_formats_openai_anthropic_and_responses() {
    let openai = serde_json::json!({
        "messages": [
            {"role": "system", "content": "x-anthropic-billing-header: foo\nBe terse."},
            {"role": "user", "content": "Run ls"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "README.md"}
        ],
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]
    });
    let turn = serialize_request(ApiProtocol::OpenAiChat, &openai);
    assert!(turn.has_tools);
    assert_eq!(turn.last_role, "user");
    assert_eq!(turn.blocks.len(), 5);
    assert_eq!(turn.blocks[0].role, "system");
    assert!(
        turn.blocks[0]
            .content
            .contains("\"type\": \"tool_definitions\"")
    );
    assert_eq!(
        turn.blocks[1].content,
        "[{\"text\": \"Be terse.\", \"type\": \"text\"}]"
    );
    assert_eq!(
        turn.blocks[2].content,
        "[{\"text\": \"Run ls\", \"type\": \"text\"}]"
    );
    assert!(turn.blocks[3].content.contains("\"id\": \"t1\""));
    assert!(turn.blocks[3].content.contains("\"type\": \"tool_use\""));
    assert_eq!(turn.blocks[4].role, "user");
    assert!(turn.blocks[4].content.contains("\"tool_use_id\": \"t1\""));
    assert!(turn.blocks[4].content.contains("\"type\": \"tool_result\""));
    assert!(
        turn.blocks[4]
            .content
            .contains("\"content\": [{\"text\": \"README.md\", \"type\": \"text\"}]")
    );

    // Base64 media DoS guard: large inline image payloads must collapse to a tiny placeholder.
    let huge_b64 = "A".repeat(100_000);
    let media_req = serde_json::json!({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{huge_b64}")}}
            ]
        }]
    });
    let media_turn = serialize_request(ApiProtocol::OpenAiChat, &media_req);
    assert!(media_turn.has_images);
    assert!(
        media_turn.serialized_text().len() < 256,
        "base64 media payload was not bounded: {} bytes",
        media_turn.serialized_text().len()
    );
}

#[test]
fn responses_reasoning_item_maps_to_assistant() {
    let resp_reasoning = serde_json::json!({
        "input": [
            {"role": "user", "content": "Solve x+1=2"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "x must be 1"}]},
            {"role": "assistant", "content": "x = 1"}
        ]
    });
    let rturn = serialize_request(ApiProtocol::OpenAiResponses, &resp_reasoning);
    assert_eq!(rturn.blocks.len(), 2);
    assert_eq!(rturn.blocks[1].role, "assistant");
    assert!(rturn.blocks[1].content.contains("x must be 1"));
}

#[test]
fn canonical_serializer_handles_anthropic_and_responses() {
    let anth = serde_json::json!({
        "system": "Sys prompt",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    });
    let turn = serialize_request(ApiProtocol::AnthropicMessages, &anth);
    assert!(!turn.has_tools);
    assert_eq!(turn.blocks.len(), 2);
    assert_eq!(
        turn.blocks[0].serialized(),
        "<|im_start|>system\n[{\"text\": \"Sys prompt\", \"type\": \"text\"}]<|im_end|>\n"
    );
    assert_eq!(
        turn.blocks[1].serialized(),
        "<|im_start|>user\n[{\"text\": \"Hi\", \"type\": \"text\"}]<|im_end|>\n"
    );

    let resp = serde_json::json!({
        "instructions": "Code carefully",
        "input": "Fix bug"
    });
    let turn = serialize_request(ApiProtocol::OpenAiResponses, &resp);
    assert_eq!(turn.blocks.len(), 2);
    assert_eq!(
        turn.blocks[0].serialized(),
        "<|im_start|>system\n[{\"text\": \"Code carefully\", \"type\": \"text\"}]<|im_end|>\n"
    );
    assert_eq!(
        turn.blocks[1].serialized(),
        "<|im_start|>user\n[{\"text\": \"Fix bug\", \"type\": \"text\"}]<|im_end|>\n"
    );
}

#[test]
fn heads_catalog_slot_inference_and_policy_invariants() {
    let Some(dir) = cached_arc_dir() else {
        eprintln!("skipping: nchapman/ARC-1.0-W4A16-HGPTQ not cached");
        return;
    };
    let heads = ArcHeads::load(&dir).expect("load ARC heads");
    // Verify C1 fix: stage_one.candidates[] order != models[] order for cloud slots.
    assert_eq!(heads.catalog.stage_one_candidate_for_model(0), Some(3)); // claude-haiku-5.5
    assert_eq!(heads.catalog.stage_one_candidate_for_model(1), Some(10)); // claude-opus-5.5
    assert_eq!(heads.catalog.stage_one_candidate_for_model(2), Some(15)); // claude-sonnet-5.5
    assert_eq!(heads.catalog.stage_one_candidate_for_model(53), Some(53)); // qwen3.8-flash-next
    assert_eq!(heads.catalog.stage_one_candidate_for_model(54), Some(54)); // qwen3.5-9b
    assert_eq!(heads.catalog.stage_one_candidate_for_model(55), Some(55)); // qwen3.8-27b
    // Local anchor & empirical logit-bias calibration across all three trained ARC-1.0 anchors
    // (`[54] 9B`, `[55] 27B`, `[53] Flash-Next`), and remote cloud models map to `standard`/`sonnet`
    // (`[2]`) or `frontier`/`opus` (`[1]`).
    assert_eq!(
        infer_local_profile("Apex-35B-V1.0-W4A16-HGPTQ", &heads.catalog),
        LocalProfile {
            slot: 54,
            logit_bias: 0.20,
        }
    );
    assert_eq!(
        infer_local_profile("Qwen3.8-Flash-Next", &heads.catalog),
        LocalProfile {
            slot: 53,
            logit_bias: 0.0,
        }
    );
    assert_eq!(
        infer_local_profile("Ternary-Bonsai-2.5-27B", &heads.catalog),
        LocalProfile {
            slot: 55,
            logit_bias: 0.40,
        }
    );
    assert_eq!(
        infer_local_profile("Qwen3.5-9B", &heads.catalog),
        LocalProfile {
            slot: 54,
            logit_bias: 0.0,
        }
    );
    assert_eq!(
        infer_local_profile("Qwen3.5-4B", &heads.catalog),
        LocalProfile {
            slot: 54,
            logit_bias: 0.0,
        }
    );
    assert_eq!(
        infer_local_profile("custom-local-model", &heads.catalog),
        LocalProfile {
            slot: 55,
            logit_bias: 0.0,
        }
    );

    let registry = ModelsDevRegistry::builtin_fallback();
    // Default (empty filter) resolves to local-only (no surprise cloud spend).
    let default_slots =
        resolve_cloud_candidates_from_registry(&registry, &heads.catalog, &[], |k| {
            (k == "OPENROUTER_API_KEY").then(|| "test-key".to_string())
        })
        .expect("default local-only");
    assert!(default_slots.is_empty());

    // Explicit `auto` detects env keys and maps to Sonnet (`2`) + Opus (`1`).
    let auto_slots = resolve_cloud_candidates_from_registry(
        &registry,
        &heads.catalog,
        &["auto".to_string()],
        |k| (k == "OPENROUTER_API_KEY").then(|| "test-key".to_string()),
    )
    .expect("auto openrouter");
    assert_eq!(auto_slots.len(), 2);
    assert_eq!(auto_slots[0].slot, 2);
    assert_eq!(auto_slots[1].slot, 1);

    let filtered = resolve_cloud_candidates_from_registry(
        &registry,
        &heads.catalog,
        &[
            "standard=openrouter/meta/muse-spark-1.3-contributor".to_string(),
            "frontier=openrouter/google/gemini-4-neon".to_string(),
        ],
        |k| (k == "OPENROUTER_API_KEY").then(|| "test-key".to_string()),
    )
    .expect("filtered");
    assert_eq!(filtered.len(), 2);
    assert_eq!(filtered[0].wire, "meta/muse-spark-1.3-contributor");
    assert_eq!(filtered[0].slot, 2);
    assert_eq!(filtered[1].wire, "google/gemini-4-neon");
    assert_eq!(filtered[1].slot, 1);

    // Warm local KV on a continuing user message stays local under PreferLocal while
    // `local_p >= warm_floor` (`(min_p_pass - 0.15).clamp(0.0, min_p_pass)`).
    let zero_readouts = vec![vec![0.0f32; 1024]];
    let roles = vec!["user".to_string()];
    let d = decide(
        &heads,
        &DecideInput {
            block_readouts: &zero_readouts,
            block_roles: &roles,
            prompt_tokens: 128,
            has_tools: false,
            has_images: false,
            warm_local_kv: true,
            event: TurnEvent::UserMessage,
            held_target: Some(&HeldTarget::Local),
            local_slot: 54,
            local_logit_bias: 0.0,
            cloud_candidates: &auto_slots,
            policy: RoutingPolicy::PreferLocal {
                min_p_pass: 0.65,
                value_usd: 1.5,
            },
            pin_pii_classes: &HashSet::new(),
        },
    )
    .expect("decide");
    assert!(matches!(d.target, RouteTarget::Local));
    assert_eq!(d.basis, RoutingBasis::WarmKvStayLocal);
}

#[test]
fn symmetric_tool_loop_hold_and_warm_kv_escape() {
    let Some(dir) = cached_arc_dir() else {
        eprintln!("skipping symmetric_tool_loop_hold_and_warm_kv_escape: ARC-1.0 not cached");
        return;
    };
    let heads = ArcHeads::load(&dir).expect("load ARC heads");
    let registry = ModelsDevRegistry::builtin_fallback();
    let auto_slots = resolve_cloud_candidates_from_registry(
        &registry,
        &heads.catalog,
        &["auto".to_string()],
        |k| (k == "OPENROUTER_API_KEY").then(|| "test-key".to_string()),
    )
    .expect("auto openrouter");
    let zero_readouts = vec![vec![0.0f32; 1024]];
    let roles = vec!["user".to_string()];

    // Symmetric tool-loop hold: when the session's held target is a cloud model (e.g. Opus),
    // a ToolResult turn holds that exact cloud model instead of flipping mid-task to local.
    let held_opus = HeldTarget::Cloud(auto_slots[1].wire.clone());
    let d_held_cloud = decide(
        &heads,
        &DecideInput {
            block_readouts: &zero_readouts,
            block_roles: &roles,
            prompt_tokens: 128,
            has_tools: true,
            has_images: false,
            warm_local_kv: false,
            event: TurnEvent::ToolResult,
            held_target: Some(&held_opus),
            local_slot: 54,
            local_logit_bias: 4.0,
            cloud_candidates: &auto_slots,
            policy: RoutingPolicy::PreferLocal {
                min_p_pass: 0.65,
                value_usd: 1.5,
            },
            pin_pii_classes: &HashSet::new(),
        },
    )
    .expect("decide held cloud");
    assert_eq!(d_held_cloud.basis, RoutingBasis::SessionHeld);
    match d_held_cloud.target {
        RouteTarget::Cloud(c) => assert_eq!(c.wire, auto_slots[1].wire),
        RouteTarget::Local => panic!("expected ToolResult to hold active cloud model"),
    }

    // Warm local KV escape hatch: on a new UserMessage where local_p collapses well below
    // warm_floor (via negative bias), Stage 1 is allowed to escalate to cloud.
    let d_escape = decide(
        &heads,
        &DecideInput {
            block_readouts: &zero_readouts,
            block_roles: &roles,
            prompt_tokens: 128,
            has_tools: false,
            has_images: false,
            warm_local_kv: true,
            event: TurnEvent::UserMessage,
            held_target: Some(&HeldTarget::Local),
            local_slot: 54,
            local_logit_bias: -4.0,
            cloud_candidates: &auto_slots,
            policy: RoutingPolicy::PreferLocal {
                min_p_pass: 0.65,
                value_usd: 1.5,
            },
            pin_pii_classes: &HashSet::new(),
        },
    )
    .expect("decide escape");
    assert_eq!(d_escape.basis, RoutingBasis::CloudSpillover);
    assert!(matches!(d_escape.target, RouteTarget::Cloud(_)));

    // Stale Turn-0 warm_local_kv must NOT pull a cloud-held session back to local at warm_floor
    // on a subsequent UserMessage when local_p is below min_p_pass.
    let d_no_stale_pullback = decide(
        &heads,
        &DecideInput {
            block_readouts: &zero_readouts,
            block_roles: &roles,
            prompt_tokens: 128,
            has_tools: false,
            has_images: false,
            warm_local_kv: true,
            event: TurnEvent::UserMessage,
            held_target: Some(&held_opus),
            local_slot: 54,
            local_logit_bias: -4.0,
            cloud_candidates: &auto_slots,
            policy: RoutingPolicy::PreferLocal {
                min_p_pass: 0.99,
                value_usd: 1.5,
            },
            pin_pii_classes: &HashSet::new(),
        },
    )
    .expect("decide no stale pullback");
    assert_ne!(d_no_stale_pullback.basis, RoutingBasis::WarmKvStayLocal);
}

#[cfg(feature = "cuda")]
#[test]
#[allow(clippy::cognitive_complexity)]
fn arc_encoder_and_heads_match_goldens() {
    use warp34_host::model::Qwen3_5Engine;

    let Some(dir) = cached_arc_dir() else {
        eprintln!("skipping: nchapman/ARC-1.0-W4A16-HGPTQ not cached");
        return;
    };
    let heads = ArcHeads::load(&dir).expect("load ARC heads");
    let mut engine = Qwen3_5Engine::load_for_encoder_on_device(&dir, 0, 2048, 4)
        .expect("load ARC-1.0-W4A16-HGPTQ on GPU 0");

    // 1. Verify signals.pii.payload probabilities against all 54 cases in goldens.signals.jsonl.
    let sig_txt =
        std::fs::read_to_string(dir.join("goldens.signals.jsonl")).expect("goldens.signals.jsonl");
    let mut checked = 0usize;
    let mut max_abs_diff = 0.0f64;
    let mut sum_abs_diff = 0.0f64;
    for line in sig_txt.lines().filter(|l| !l.trim().is_empty()) {
        let row: serde_json::Value = serde_json::from_str(line).expect("parse signal golden");
        let fmt = row
            .pointer("/request/request_format")
            .and_then(|v| v.as_str())
            .unwrap_or("openai_chat");
        let proto = match fmt {
            "anthropic_messages" => ApiProtocol::AnthropicMessages,
            "openai_responses" => ApiProtocol::OpenAiResponses,
            _ => ApiProtocol::OpenAiChat,
        };
        let Some(req_body) = row.pointer("/request/request") else {
            continue;
        };
        let Some(ref_any) = row
            .pointer("/signals/pii.payload/scopes/conversation/probabilities/any")
            .and_then(|v| v.as_f64())
        else {
            continue;
        };
        let turn = serialize_request(proto, req_body);
        let mut flat = Vec::new();
        let mut offsets = Vec::new();
        for b in &turn.blocks {
            let toks = engine.encode(&b.serialized()).expect("encode");
            if !toks.is_empty() {
                flat.extend_from_slice(&toks);
                offsets.push(flat.len() - 1);
            }
        }
        let rows = engine
            .encode_readouts_at_offsets_f32(&flat, 0, &offsets)
            .expect("forward");
        let readout = rows.last().expect("last row");
        let pred = heads
            .pii
            .predict(
                readout,
                heads.catalog.pii_threshold_for_tokens(flat.len()),
                heads.catalog.pii_type_threshold,
            )
            .expect("pii predict");
        let diff = (pred.presence_prob - ref_any).abs();
        if diff > 0.1 || checked < 3 {
            eprintln!(
                "  signal[{checked}] ref={ref_any:.4} got={:.4} diff={diff:.4} toks={} block0={:?}",
                pred.presence_prob,
                flat.len(),
                turn.blocks
                    .first()
                    .map(|b| b.serialized())
                    .unwrap_or_default()
                    .chars()
                    .take(120)
                    .collect::<String>()
            );
        }
        max_abs_diff = max_abs_diff.max(diff);
        sum_abs_diff += diff;
        checked += 1;
    }
    let mean_abs_diff = if checked > 0 {
        sum_abs_diff / checked as f64
    } else {
        0.0
    };
    eprintln!(
        "ARC goldens.signals.jsonl checked={checked} mean_abs_diff={mean_abs_diff:.4} max_abs_diff={max_abs_diff:.4}"
    );
    assert_eq!(checked, 26, "expected 26 golden signal rows");
    assert!(
        mean_abs_diff < 0.08,
        "expected tight PII parity with canonical arc-role-blocks-v2 (got mean_abs_diff={mean_abs_diff:.4})"
    );

    // 2. Verify 54/54 exact token_count + Stage-1/Stage-2 + decide() invariants across goldens.jsonl
    //    and across all 51 real ARC-1.0 Stage-2 actions in arc-runtime.json.
    let runtime_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("runtime/arc-runtime.json"))
            .expect("runtime/arc-runtime.json"),
    )
    .expect("parse arc-runtime.json");
    let raw_actions = runtime_json
        .pointer("/actions")
        .and_then(|v| v.as_array())
        .expect("runtime/arc-runtime.json actions");
    assert_eq!(raw_actions.len(), heads.catalog.actions.len());
    assert_eq!(raw_actions.len(), 51, "ARC-1.0 defines 51 Stage-2 actions");
    for (idx, act) in heads.catalog.actions.iter().enumerate() {
        assert!(
            matches!(act.effort.as_str(), "default" | "low" | "medium" | "high"),
            "ARC-1.0 action[{idx}] unexpected base effort {:?}",
            act.effort
        );
        if let Some(lv) = act.level.as_deref() {
            assert!(
                matches!(lv, "down" | "none" | "up"),
                "ARC-1.0 action[{idx}] unexpected steering level {lv:?}"
            );
        }
    }
    // Verify generalized 3-archetype remote slot resolution against the real 56-slot ARC-1.0 catalog.
    for (tier, id, prices, expected_slot) in [
        (
            None,
            "opencode/claude-sonnet-5-5",
            Some([2.0, 0.2, 2.5, 10.0]),
            2usize,
        ),
        (
            None,
            "opencode/claude-opus-5-5",
            Some([4.0, 0.4, 5.0, 20.0]),
            1usize,
        ),
        (None, "openai/gpt-5.4", Some([2.5, 0.25, 0.0, 15.0]), 2usize),
        (
            None,
            "openai/gpt-5.4-pro",
            Some([30.0, 3.0, 0.0, 180.0]),
            1usize,
        ),
        (Some("standard"), "custom/model-a", None, 2usize),
        (Some("sonnet"), "custom/model-b", None, 2usize),
        (Some("frontier"), "custom/model-c", None, 1usize),
        (Some("opus"), "custom/model-d", None, 1usize),
    ] {
        assert_eq!(
            warp34_host::arc_router::policy::infer_remote_slot(
                tier,
                id,
                id,
                prices,
                &heads.catalog,
            )
            .expect("infer_remote_slot"),
            expected_slot,
            "expected ({tier:?}, {id}) to map to remote archetype slot {expected_slot}"
        );
    }

    let gold_txt = std::fs::read_to_string(dir.join("goldens.jsonl")).expect("goldens.jsonl");
    let mut gold_checked = 0usize;
    let mut exact_tok_matches = 0usize;
    let active_cands: Vec<usize> = [0usize, 1, 2, 3, 10, 53, 54, 55]
        .into_iter()
        .filter_map(|m| heads.catalog.stage_one_candidate_for_model(m))
        .collect();
    let empty_pins = HashSet::new();
    for line in gold_txt.lines().filter(|l| !l.trim().is_empty()) {
        let row: serde_json::Value = serde_json::from_str(line).expect("parse decision golden");
        let Some(req_body) = row.pointer("/request/request") else {
            continue;
        };
        let turn = serialize_request(ApiProtocol::AnthropicMessages, req_body);
        let mut flat = Vec::new();
        let mut offsets = Vec::new();
        let mut roles = Vec::new();
        for b in &turn.blocks {
            let toks = engine.encode(&b.serialized()).expect("encode");
            if !toks.is_empty() {
                flat.extend_from_slice(&toks);
                offsets.push(flat.len() - 1);
                roles.push(b.role.clone());
            }
        }
        if let Some(ref_toks) = row
            .pointer("/expected/token_count")
            .and_then(|v| v.as_u64())
            && flat.len() == ref_toks as usize
        {
            exact_tok_matches += 1;
        }
        let rows = engine
            .encode_readouts_at_offsets_f32(&flat, 0, &offsets)
            .expect("forward");
        let preds = heads
            .stage_one
            .predict(rows.last().unwrap(), heads.catalog.stage_one_regime)
            .expect("stage_one");
        assert_eq!(preds.len(), heads.catalog.models.len());

        // Verify sparse active-only Stage-1 prediction (P3) matches full prediction bit-for-bit.
        if gold_checked == 0 {
            let sparse = heads
                .stage_one
                .predict_selected(
                    rows.last().unwrap(),
                    heads.catalog.stage_one_regime,
                    &active_cands,
                )
                .expect("stage_one sparse");
            assert_eq!(sparse.len(), preds.len());
            for &cand_idx in &active_cands {
                assert!((sparse[cand_idx].p_pass - preds[cand_idx].p_pass).abs() < 1e-12);
                assert_eq!(sparse[cand_idx].usage, preds[cand_idx].usage);
            }
        }
        // Verify Stage-2 multi-block pooled prediction succeeds cleanly on every golden case
        // and that real `decide()` on Ternary-Bonsai-2.5-27B (slot 55) & Apex-35B (slot 54)
        // never collapses thinking effort onto `"none"`.
        let all_arms: Vec<usize> = (0..8).collect();
        let s2 = heads
            .stage_two
            .predict_arms(&rows, &roles, &[], &all_arms)
            .expect("stage_two");
        assert!(!s2.is_empty());
        for local_slot in [54usize, 55usize] {
            let dec = decide(
                &heads,
                &DecideInput {
                    block_readouts: &rows,
                    block_roles: &roles,
                    prompt_tokens: flat.len(),
                    has_tools: turn.has_tools,
                    has_images: turn.has_images,
                    warm_local_kv: false,
                    event: TurnEvent::TaskStart,
                    held_target: None,
                    local_slot,
                    local_logit_bias: 0.0,
                    cloud_candidates: &[],
                    policy: RoutingPolicy::PreferLocal {
                        min_p_pass:
                            warp34_host::arc_router::policy::DEFAULT_PREFER_LOCAL_MIN_P_PASS,
                        value_usd: 1.0,
                    },
                    pin_pii_classes: &empty_pins,
                },
            )
            .expect("decide");
            assert!(
                matches!(
                    dec.reasoning_effort.as_str(),
                    "low" | "medium" | "high" | "xhigh"
                ),
                "golden[{gold_checked}] slot={local_slot} unexpectedly collapsed effort to {:?}",
                dec.reasoning_effort
            );
        }
        gold_checked += 1;
    }
    eprintln!(
        "ARC goldens.jsonl checked={gold_checked} exact_token_count_matches={exact_tok_matches}/{gold_checked}"
    );
    assert_eq!(gold_checked, 54, "expected 54 decision golden cases");
    assert_eq!(
        exact_tok_matches, 54,
        "expected 54/54 exact token_count parity on goldens.jsonl, got {exact_tok_matches}/{gold_checked}"
    );

    // 3. Verify incremental prefix extension matches fresh full prefill.
    let prefix_req = serde_json::json!({
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Explain why quicksort is O(n log n) on average."}
        ]
    });
    let full_req = serde_json::json!({
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Explain why quicksort is O(n log n) on average."},
            {"role": "assistant", "content": "Because random pivots split the array evenly in expectation."},
            {"role": "user", "content": "Now give the recurrence relation."}
        ]
    });
    let p_turn = serialize_request(ApiProtocol::OpenAiChat, &prefix_req);
    let f_turn = serialize_request(ApiProtocol::OpenAiChat, &full_req);
    let mut p_flat = Vec::new();
    let mut p_off = Vec::new();
    for b in &p_turn.blocks {
        p_flat.extend_from_slice(&engine.encode(&b.serialized()).expect("encode"));
        p_off.push(p_flat.len() - 1);
    }
    let mut f_flat = Vec::new();
    let mut f_off = Vec::new();
    for b in &f_turn.blocks {
        f_flat.extend_from_slice(&engine.encode(&b.serialized()).expect("encode"));
        f_off.push(f_flat.len() - 1);
    }
    assert_eq!(&f_flat[..p_flat.len()], &p_flat[..]);
    let fresh_rows = engine
        .encode_readouts_at_offsets_f32(&f_flat, 0, &f_off)
        .expect("fresh");
    // Now run prefix, then incrementally extend with the new blocks.
    let _ = engine
        .encode_readouts_at_offsets_f32(&p_flat, 0, &p_off)
        .expect("prefix");
    let new_off: Vec<usize> = f_off
        .iter()
        .copied()
        .filter(|&o| o >= p_flat.len())
        .collect();
    let inc_rows = engine
        .encode_readouts_at_offsets_f32(&f_flat, p_flat.len(), &new_off)
        .expect("incremental");
    let max_inc_diff = fresh_rows
        .last()
        .unwrap()
        .iter()
        .zip(inc_rows.last().unwrap().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("ARC incremental prefix extension max_abs_diff={max_inc_diff:.6}");
    assert!(
        max_inc_diff < 5e-3,
        "incremental prefix extension diverged from fresh prefill: {max_inc_diff}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn dp1_warm_kv_and_stateful_responses_routing() {
    use std::sync::Arc;
    use std::time::Duration;
    use warp34_host::{
        arc_router::{ArcRouter, ArcRouterConfig},
        cache::KvFormatPolicy,
        model::Qwen3_5Engine,
        server::{OpenAiServerState, ServeOptions, serve},
    };

    let Some(arc_dir) = cached_arc_dir() else {
        eprintln!("skipping: nchapman/ARC-1.0-W4A16-HGPTQ not cached");
        return;
    };
    let Some(local_dir) = std::env::var("HOME")
        .ok()
        .map(|h| {
            PathBuf::from(h)
                .join(".cache/huggingface/hub/models--nchapman--Qwen3.5-0.8B-W4A16-HGPTQ/snapshots")
        })
        .and_then(|p| std::fs::read_dir(p).ok()?.next()?.ok().map(|e| e.path()))
    else {
        eprintln!("skipping: nchapman/Qwen3.5-0.8B-W4A16-HGPTQ not cached");
        return;
    };

    let arc_router = Arc::new(
        ArcRouter::load(ArcRouterConfig {
            model_dir: arc_dir,
            device_id: 0,
            local_model_name: "Qwen3.5-0.8B".to_string(),
            local_logit_bias: None,
            cloud_model_filter: Vec::new(),
            policy: RoutingPolicy::PreferLocal {
                min_p_pass: warp34_host::arc_router::policy::DEFAULT_PREFER_LOCAL_MIN_P_PASS,
                value_usd: 1.0,
            },
            pin_pii_classes: HashSet::new(),
            offline: true,
        })
        .expect("load ArcRouter"),
    );

    let local_dir_str = local_dir.to_string_lossy().into_owned();
    let state = OpenAiServerState::from_loader_with_options(
        "Qwen3.5-0.8B".into(),
        false,
        ServeOptions {
            webui: false,
            max_num_seqs: 2,
            arc_router: Some(arc_router),
            ..Default::default()
        },
        move || {
            Qwen3_5Engine::load_for_serving(
                &local_dir_str,
                1,
                KvFormatPolicy::Fp16,
                Some(256),
                false,
                2,
            )
            .map_err(|e| e.to_string())
        },
    )
    .expect("boot DP=1 server with ArcRouter");

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
            let server_state = Arc::clone(&state);
            let server = tokio::spawn(async move {
                serve(listener, server_state, async {
                    let _ = stop_rx.await;
                })
                .await
            });
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .unwrap();

            // >= 64 tokens so Turn 0 donates at least one complete 64-token radix + GDN block.
            let long_sys = "You are a systems programming assistant specializing in Rust, CUDA, \
                memory allocators, lock-free data structures, radix prefix caching, and clean \
                deterministic verification. Always answer concisely with exact facts. "
                .repeat(4);

            // 1. Headerless DP=1 `/v1/chat/completions`: Turn 0 cold -> Turn 1 warm_kv_stay_local.
            let t0_resp = client
                .post(format!("{base}/v1/chat/completions"))
                .json(&serde_json::json!({
                    "model": "router",
                    "max_tokens": 8,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": long_sys},
                        {"role": "user", "content": "Reply with OK."}
                    ]
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(t0_resp.status(), 200);
            assert_eq!(
                t0_resp
                    .headers()
                    .get("x-warp34-route-basis")
                    .and_then(|v| v.to_str().ok()),
                Some("local_only_effort"),
                "cold Turn 0 on local-only router must report local_only_effort"
            );
            let t0_effort = t0_resp
                .headers()
                .get("x-warp34-route-effort")
                .and_then(|v| v.to_str().ok())
                .expect("x-warp34-route-effort header");
            assert!(
                matches!(t0_effort, "low" | "medium" | "high" | "xhigh"),
                "ARC-1.0 Stage-2 route effort must keep thinking enabled, got {t0_effort:?}"
            );
            let t0_json: serde_json::Value = t0_resp.json().await.unwrap();
            let asst_reply = t0_json["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("OK");

            let t1_resp = client
                .post(format!("{base}/v1/chat/completions"))
                .json(&serde_json::json!({
                    "model": "router",
                    "max_tokens": 8,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": long_sys},
                        {"role": "user", "content": "Reply with OK."},
                        {"role": "assistant", "content": asst_reply},
                        {"role": "user", "content": "Now reply with DONE."}
                    ]
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(t1_resp.status(), 200);
            assert_eq!(
                t1_resp
                    .headers()
                    .get("x-warp34-route-basis")
                    .and_then(|v| v.to_str().ok()),
                Some("warm_kv_stay_local"),
                "Turn 1 headerless continuation on DP=1 must detect live radix+GDN KV"
            );

            // 2. Stateful `/v1/responses` (`"store": true` + Bearer scope): cold Turn 0 must NOT
            //    claim warm_kv_stay_local; follow-up via `previous_response_id` MUST detect warm_kv_stay_local.
            let r0_resp = client
                .post(format!("{base}/v1/responses"))
                .bearer_auth("test-session-scope")
                .json(&serde_json::json!({
                    "model": "router",
                    "store": true,
                    "max_output_tokens": 8,
                    "temperature": 0.0,
                    "instructions": long_sys,
                    "input": "Reply with ALPHA."
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(r0_resp.status(), 200);
            assert_eq!(
                r0_resp
                    .headers()
                    .get("x-warp34-route-basis")
                    .and_then(|v| v.to_str().ok()),
                Some("local_only_effort"),
                "cold stateful /v1/responses Turn 0 must report local_only_effort, not warm_kv_stay_local"
            );
            let r0_json: serde_json::Value = r0_resp.json().await.unwrap();
            let prev_id = r0_json["id"].as_str().expect("response id");

            let r1_resp = client
                .post(format!("{base}/v1/responses"))
                .bearer_auth("test-session-scope")
                .json(&serde_json::json!({
                    "model": "router",
                    "previous_response_id": prev_id,
                    "max_output_tokens": 8,
                    "temperature": 0.0,
                    "input": "Reply with BETA."
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(r1_resp.status(), 200);
            assert_eq!(
                r1_resp
                    .headers()
                    .get("x-warp34-route-basis")
                    .and_then(|v| v.to_str().ok()),
                Some("warm_kv_stay_local"),
                "previous_response_id continuation must detect live radix+GDN KV"
            );

            let metrics: serde_json::Value = client
                .get(format!("{base}/api/metrics"))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(metrics["arc_router"]["total_requests"], 4);
            assert_eq!(
                metrics["arc_router"]["warm_kv_stayed_local"], 2,
                "only the 2 genuine warm-KV continuations should increment warm_kv_stayed_local"
            );

            stop_tx.send(()).unwrap();
            server.await.unwrap().unwrap();
            state.shutdown_all().await.unwrap();
        });
}
