//! Golden parity and policy unit tests for the native ARC-1.0 router.

use std::{collections::HashSet, path::PathBuf};
use warp34_host::arc_router::{
    heads::ArcHeads,
    input::{ApiProtocol, serialize_request},
    policy::{
        DecideInput, RouteTarget, RoutingBasis, RoutingPolicy, UpstreamTarget, decide,
        infer_local_slot, resolve_cloud_candidates,
    },
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
    assert!(
        media_turn.serialized_text().len() < 256,
        "base64 media payload was not bounded: {} bytes",
        media_turn.serialized_text().len()
    );

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
    assert_eq!(infer_local_slot("Qwen3.5-35B-A3B", &heads.catalog), 53);
    assert_eq!(infer_local_slot("Qwen3.8-Flash-Next", &heads.catalog), 53);
    assert_eq!(infer_local_slot("Qwen3.5-27B", &heads.catalog), 55);
    assert_eq!(infer_local_slot("Qwen3.5-9B", &heads.catalog), 54);
    assert_eq!(infer_local_slot("Qwen3.5-4B", &heads.catalog), 54);

    let or_slots = resolve_cloud_candidates(&UpstreamTarget::OpenRouter, &heads.catalog, &[])
        .expect("openrouter");
    assert_eq!(or_slots.len(), 11);
    let filtered = resolve_cloud_candidates(
        &UpstreamTarget::OpenRouter,
        &heads.catalog,
        &[
            "medium=meta/muse-spark-1.3-contributor".to_string(),
            "strong=google/gemini-4-neon".to_string(),
        ],
    )
    .expect("filtered");
    assert_eq!(filtered.len(), 2);
    assert_eq!(filtered[0].wire, "meta/muse-spark-1.3-contributor");
    assert_eq!(filtered[1].wire, "google/gemini-4-neon");

    // Warm local KV must stay local even with cloud slots available.
    let zero_readouts = vec![vec![0.0f32; 1024]];
    let roles = vec!["user".to_string()];
    let d = decide(
        &heads,
        &DecideInput {
            block_readouts: &zero_readouts,
            block_roles: &roles,
            prompt_tokens: 128,
            has_tools: false,
            warm_local_kv: true,
            local_slot: 54,
            cloud_candidates: &or_slots,
            policy: RoutingPolicy::Utility { value_usd: 4.5 },
            pin_pii_classes: &HashSet::new(),
        },
    )
    .expect("decide");
    assert!(matches!(d.target, RouteTarget::Local));
    assert_eq!(d.basis, RoutingBasis::WarmKvStayLocal);
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

    // 2. Verify 54/54 exact token_count + Stage-1/Stage-2 consistency across goldens.jsonl.
    let gold_txt = std::fs::read_to_string(dir.join("goldens.jsonl")).expect("goldens.jsonl");
    let mut gold_checked = 0usize;
    let mut exact_tok_matches = 0usize;
    let active_cands: Vec<usize> = [0usize, 1, 2, 3, 10, 53, 54, 55]
        .into_iter()
        .filter_map(|m| heads.catalog.stage_one_candidate_for_model(m))
        .collect();
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
        // Verify Stage-2 multi-block pooled prediction succeeds cleanly on every golden case.
        let all_arms: Vec<usize> = (0..8).collect();
        let s2 = heads
            .stage_two
            .predict_arms(&rows, &roles, &[], &all_arms)
            .expect("stage_two");
        assert!(!s2.is_empty());
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
