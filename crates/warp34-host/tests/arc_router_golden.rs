//! Golden parity and policy unit tests for the native ARC-1.0 router.

use std::{collections::HashSet, path::PathBuf};
use warp34_host::arc_router::{
    heads::ArcHeads,
    input::{ApiProtocol, serialize_request},
    policy::{
        DecideInput, RouteTarget, RoutingBasis, RoutingPolicy, UpstreamTarget, decide,
        infer_local_slot, resolve_cloud_slots,
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
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hello"}
        ],
        "tools": [{"type": "function", "function": {"name": "bash"}}]
    });
    let turn = serialize_request(ApiProtocol::OpenAiChat, &openai);
    assert!(turn.has_tools);
    assert_eq!(turn.last_role, "user");
    assert!(
        turn.text
            .contains("<|im_start|>system\n\"Be terse.\\n\\nAvailable tools: bash\"<|im_end|>")
    );
    assert!(turn.text.contains("<|im_start|>user\n\"Hello\"<|im_end|>"));

    let anth = serde_json::json!({
        "system": "Sys prompt",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    });
    let turn = serialize_request(ApiProtocol::AnthropicMessages, &anth);
    assert!(!turn.has_tools);
    assert!(
        turn.text
            .contains("<|im_start|>system\n\"Sys prompt\"<|im_end|>")
    );
    assert!(turn.text.contains("<|im_start|>user\n\"Hi\"<|im_end|>"));

    let resp = serde_json::json!({
        "instructions": "Code carefully",
        "input": "Fix bug"
    });
    let turn = serialize_request(ApiProtocol::OpenAiResponses, &resp);
    assert!(
        turn.text
            .contains("<|im_start|>system\n\"Code carefully\"<|im_end|>")
    );
    assert!(
        turn.text
            .contains("<|im_start|>user\n\"Fix bug\"<|im_end|>")
    );
}

#[test]
fn heads_catalog_slot_inference_and_policy_invariants() {
    let Some(dir) = cached_arc_dir() else {
        eprintln!("skipping: nchapman/ARC-1.0-W4A16-HGPTQ not cached");
        return;
    };
    let heads = ArcHeads::load(&dir).expect("load ARC heads");
    assert_eq!(infer_local_slot("Qwen3.5-35B-A3B", &heads.catalog), 53);
    assert_eq!(infer_local_slot("Qwen3.8-Flash-Next", &heads.catalog), 53);
    assert_eq!(infer_local_slot("Qwen3.5-27B", &heads.catalog), 55);
    assert_eq!(infer_local_slot("Qwen3.5-9B", &heads.catalog), 54);
    assert_eq!(infer_local_slot("Qwen3.5-4B", &heads.catalog), 54);

    let or_slots =
        resolve_cloud_slots(&UpstreamTarget::OpenRouter, &heads.catalog, &[]).expect("openrouter");
    assert_eq!(or_slots.len(), 11);
    let filtered = resolve_cloud_slots(
        &UpstreamTarget::OpenRouter,
        &heads.catalog,
        &["anthropic/claude-opus-5.5".to_string()],
    )
    .expect("filtered");
    assert_eq!(filtered.len(), 1);

    // Warm local KV must stay local even with cloud slots available.
    let zero_readout = vec![0.0f32; 1024];
    let d = decide(
        &heads,
        &DecideInput {
            readout: &zero_readout,
            last_role: "user",
            prompt_tokens: 128,
            has_tools: false,
            warm_local_kv: true,
            local_slot: 54,
            cloud_slots: &or_slots,
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
fn arc_encoder_and_heads_match_goldens() {
    use warp34_host::model::Qwen3_5Engine;

    let Some(dir) = cached_arc_dir() else {
        eprintln!("skipping: nchapman/ARC-1.0-W4A16-HGPTQ not cached");
        return;
    };
    let heads = ArcHeads::load(&dir).expect("load ARC heads");
    let mut engine = Qwen3_5Engine::load_for_encoder_on_device(&dir, 0, 256)
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
        let tokens = engine.encode(&turn.text).expect("encode");
        let readout = engine.encode_last_readout_f32(&tokens).expect("forward");
        let pred = heads
            .pii
            .predict(
                &readout,
                heads.catalog.pii_threshold_for_tokens(tokens.len()),
                heads.catalog.pii_type_threshold,
            )
            .expect("pii predict");
        let diff = (pred.presence_prob - ref_any).abs();
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
    assert!(checked > 0, "expected golden signal rows");

    // 2. Verify PII signal head on synthetic clean vs secret prompts.
    let clean = "<|im_start|>user\n\"What is 2 + 2?\"<|im_end|>\n";
    let secret = "<|im_start|>user\n\"My full name is John Smith, email john.smith@acme.com, phone 555-867-5309, SSN 123-45-6789, AWS key AKIAIOSFODNN7EXAMPLE, secret wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY.\"<|im_end|>\n";
    let clean_toks = engine.encode(clean).unwrap();
    let clean_r = engine.encode_last_readout_f32(&clean_toks).unwrap();
    let clean_pii = heads
        .pii
        .predict(
            &clean_r,
            heads.catalog.pii_threshold_for_tokens(clean_toks.len()),
            heads.catalog.pii_type_threshold,
        )
        .unwrap();

    let secret_toks = engine.encode(secret).unwrap();
    let secret_r = engine.encode_last_readout_f32(&secret_toks).unwrap();
    let secret_pii = heads
        .pii
        .predict(
            &secret_r,
            heads.catalog.pii_threshold_for_tokens(secret_toks.len()),
            heads.catalog.pii_type_threshold,
        )
        .unwrap();

    eprintln!(
        "PII clean: prob={:.4} classes={:?} | secret: prob={:.4} classes={:?}",
        clean_pii.presence_prob,
        clean_pii.triggered_classes,
        secret_pii.presence_prob,
        secret_pii.triggered_classes
    );
    assert!(
        secret_pii.presence_prob > clean_pii.presence_prob,
        "expected secret prompt PII prob ({}) > clean ({})",
        secret_pii.presence_prob,
        clean_pii.presence_prob
    );
    assert!(
        !secret_pii.triggered_classes.is_empty(),
        "expected PII classes triggered on secret prompt"
    );

    // 3. Verify Stage 1 differentiates easy vs hard prompt on local slot 54 (9B).
    let hard = "<|im_start|>user\n\"Prove that every finite division ring is a field (Wedderburn's little theorem) using cyclotomic polynomials and class equations, then formalize the proof in Lean 4 without sorry.\"<|im_end|>\n";
    let hard_toks = engine.encode(hard).unwrap();
    let hard_r = engine.encode_last_readout_f32(&hard_toks).unwrap();
    let p_easy = heads.stage_one.predict(&clean_r, 1).unwrap()[54].p_pass;
    let p_hard = heads.stage_one.predict(&hard_r, 1).unwrap()[54].p_pass;
    eprintln!("Stage-1 slot[54] P(pass): easy={p_easy:.4} hard={p_hard:.4}");
    assert!(
        p_easy > p_hard,
        "expected easy P(pass) ({p_easy:.4}) > hard ({p_hard:.4})"
    );
}
