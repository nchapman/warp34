#![cfg(feature = "cuda")]
mod common;

use std::{sync::atomic::Ordering, time::Duration};
use warp34_host::{
    cache::KvFormatPolicy,
    engine_worker::{EngineWorker, GenerationEvent, GenerationStream},
    model::Qwen3_5Engine,
    parser::SamplingParams,
    proposer::{SpeculativeConfig, SpeculativeKind},
    scheduler::SchedulerConfig,
};

fn worker_with_policy(kind: Option<SpeculativeKind>, policy: KvFormatPolicy) -> EngineWorker {
    EngineWorker::from_loader(
        move || {
            let mut engine =
                Qwen3_5Engine::load_for_serving("Qwen/Qwen3.5-4B", 1, policy, Some(256), true, 16)
                    .map_err(|e| e.to_string())?;
            if let Some(kind) = kind {
                engine
                    .enable_speculative(&SpeculativeConfig {
                        kind,
                        mtp_k: 3,
                        draft_model: (kind == SpeculativeKind::DFlash2)
                            .then(|| "nchapman/Qwen3.5-4B-DFlash2".into()),
                    })
                    .map_err(|e| e.to_string())?;
            }
            Ok(engine)
        },
        SchedulerConfig::default(),
    )
    .unwrap()
}

fn worker(kind: SpeculativeKind) -> EngineWorker {
    worker_with_policy(Some(kind), KvFormatPolicy::Tq4)
}

fn params() -> SamplingParams {
    SamplingParams {
        max_tokens: 64,
        ignore_eos: true,
        temperature: 0.0,
        ..SamplingParams::default()
    }
}

async fn collect(stream: &mut GenerationStream, mut tokens: Vec<i32>) -> Result<Vec<i32>, String> {
    while let Some(event) = stream.events.recv().await {
        match event {
            GenerationEvent::Token { id, .. } => tokens.push(id),
            GenerationEvent::Finished {
                generated_tokens, ..
            } => {
                if generated_tokens != 64 || tokens.len() != 64 {
                    return Err(format!("counts: {generated_tokens}, {}", tokens.len()));
                }
                return Ok(tokens);
            }
            GenerationEvent::Error(e) => return Err(e.to_string()),
        }
    }
    Err("stream closed without Finished".into())
}

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// Run with flock /tmp/warp34-gpu1.lock and CUDA_VISIBLE_DEVICES=1.
#[test]
fn native_speculation_yields_to_late_peer_losslessly() {
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "native_speculation_yields_to_late_peer_losslessly",
    )
    .is_none()
        || common::require_or_skip_cached_model(
            "nchapman/Qwen3.5-4B-DFlash2",
            "native_speculation_yields_to_late_peer_losslessly",
        )
        .is_none()
    {
        return;
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    for kind in [SpeculativeKind::Mtp, SpeculativeKind::DFlash2] {
        // Loading, inference and destruction stay on the worker OS thread.
        // Use an unambiguous factual prompt rather than "The capital of France is", which on
        // sm_89 TQ4 hits an exact 1-FP16-ULP (0.015625) tie at token 10 ("\n" vs "\n\n") when
        // DFlash2 yields from M=16 Tensor Core verify to M=2 cuBLAS batched decode.
        let baseline_worker = worker(kind);
        let prompt: Vec<i32> = baseline_worker
            .tokenizer
            .as_ref()
            .unwrap()
            .encode("In computer architecture, cache coherence ensures", false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&id| id as i32)
            .collect();
        let baseline = rt.block_on(async {
            let mut stream = baseline_worker
                .submit(prompt.clone(), params(), None)
                .unwrap();
            let result =
                tokio::time::timeout(Duration::from_secs(120), collect(&mut stream, vec![])).await;
            baseline_worker.shutdown().await.unwrap();
            assert!(baseline_worker.stats.verify_rounds.load(Ordering::Relaxed) > 0);
            result.unwrap().unwrap()
        });
        let worker = worker(kind);
        let result = rt.block_on(async {
            let result = tokio::time::timeout(Duration::from_secs(120), async {
                let mut first = worker.submit(prompt.clone(), params(), None)?;
                let mut prefix = Vec::new();
                while prefix.len() < 2 {
                    match first.events.recv().await {
                        Some(GenerationEvent::Token { id, .. }) => prefix.push(id),
                        other => return Err(format!("premature event: {other:?}")),
                    }
                }
                assert!(worker.stats.verify_rounds.load(Ordering::Relaxed) > 0);
                assert_eq!(worker.stats.completed.load(Ordering::Relaxed), 0);
                let mut peer = worker.submit(prompt.clone(), params(), None)?;
                // Drain both concurrently: output backpressure must not cancel either.
                let (first, peer) =
                    tokio::join!(collect(&mut first, prefix), collect(&mut peer, vec![]));
                let first = first?;
                peer?;
                Ok::<_, String>(first)
            })
            .await;
            worker.shutdown().await.unwrap();
            result
        });
        assert_eq!(
            result.unwrap().unwrap(),
            baseline,
            "{kind:?}: exact greedy output"
        );
        assert!(
            worker.stats.peak_active.load(Ordering::Relaxed) >= 2,
            "{kind:?}: late peer was held until speculative completion"
        );
        assert_eq!(worker.stats.completed.load(Ordering::Relaxed), 2);
        assert_eq!(worker.stats.cancelled.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn batched_mtp_b4_matches_single_stream_losslessly() {
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "batched_mtp_b4_matches_single_stream_losslessly",
    )
    .is_none()
    {
        return;
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let texts = [
        "The capital of France is",
        "In computer architecture, cache coherence ensures",
        "Rust ownership rules guarantee memory safety by",
        "Binary search finds an element in a sorted array in",
    ];
    let baseline_worker = worker(SpeculativeKind::Mtp);
    let prompts: Vec<Vec<i32>> = texts
        .iter()
        .map(|s| {
            baseline_worker
                .tokenizer
                .as_ref()
                .unwrap()
                .encode(*s, false)
                .unwrap()
                .get_ids()
                .iter()
                .map(|&id| id as i32)
                .collect()
        })
        .collect();
    let baselines = rt.block_on(async {
        let mut outs = Vec::with_capacity(prompts.len());
        for p in &prompts {
            let mut st = baseline_worker.submit(p.clone(), params(), None).unwrap();
            let got = tokio::time::timeout(Duration::from_secs(120), collect(&mut st, vec![]))
                .await
                .unwrap()
                .unwrap();
            outs.push(got);
        }
        baseline_worker.shutdown().await.unwrap();
        outs
    });

    let batch_worker = worker(SpeculativeKind::Mtp);
    let got_all = rt.block_on(async {
        let mut s0 = batch_worker
            .submit(prompts[0].clone(), params(), None)
            .unwrap();
        let mut s1 = batch_worker
            .submit(prompts[1].clone(), params(), None)
            .unwrap();
        let mut s2 = batch_worker
            .submit(prompts[2].clone(), params(), None)
            .unwrap();
        let mut s3 = batch_worker
            .submit(prompts[3].clone(), params(), None)
            .unwrap();
        let res = tokio::time::timeout(Duration::from_secs(120), async {
            let (o0, o1, o2, o3) = tokio::join!(
                collect(&mut s0, vec![]),
                collect(&mut s1, vec![]),
                collect(&mut s2, vec![]),
                collect(&mut s3, vec![]),
            );
            Ok::<_, String>(vec![o0?, o1?, o2?, o3?])
        })
        .await
        .unwrap()
        .unwrap();
        batch_worker.shutdown().await.unwrap();
        res
    });

    assert_eq!(
        got_all, baselines,
        "batched B=4 MTP (TQ4) must match single-stream B=1 MTP greedy bit-for-bit across all 4 slots"
    );
    assert_eq!(batch_worker.stats.peak_active.load(Ordering::Relaxed), 4);
    assert!(
        batch_worker.stats.verify_rounds.load(Ordering::Relaxed) >= 40,
        "expected batched MTP verify rounds across all 4 slots, got {}",
        batch_worker.stats.verify_rounds.load(Ordering::Relaxed)
    );
    assert!(
        batch_worker.stats.accepted_drafts.load(Ordering::Relaxed) > 0,
        "expected accepted MTP drafts in batched B=4 verify"
    );

    // 2. Verify B=2 batched MTP (`M=8` CUDA Graph slot) matches single-stream B=1 bit-for-bit,
    //    then verify B=2 GPU-resident stochastic verification (`verify_batched_speculative_gpu_steps`).
    let b2_worker = worker(SpeculativeKind::Mtp);
    let (got_b2, gpu_stoch_a, gpu_stoch_b) = rt.block_on(async {
        let mut s0 = b2_worker
            .submit(prompts[0].clone(), params(), None)
            .unwrap();
        let mut s1 = b2_worker
            .submit(prompts[1].clone(), params(), None)
            .unwrap();
        let got_b2 = tokio::time::timeout(Duration::from_secs(120), async {
            let (o0, o1) = tokio::join!(collect(&mut s0, vec![]), collect(&mut s1, vec![]));
            Ok::<_, String>(vec![o0?, o1?])
        })
        .await
        .unwrap()
        .unwrap();

        // 2a. Plain greedy (`ignore_eos = false`) exercises the `any_gpu_sampler == false`
        //     (`RejectionSampler::greedy`) branch in `step_scheduled_batched_mtp_verify_round`,
        //     while `params()` (`ignore_eos = true`) above exercised `verify_batched_speculative_gpu_steps`.
        let mut plain_greedy = params();
        plain_greedy.ignore_eos = false;
        let rounds_before_plain = b2_worker.stats.verify_rounds.load(Ordering::Relaxed);
        let mut g0 = b2_worker
            .submit(prompts[2].clone(), plain_greedy.clone(), None)
            .unwrap();
        let mut g1 = b2_worker
            .submit(prompts[3].clone(), plain_greedy, None)
            .unwrap();
        let (gpu_stoch_a, gpu_stoch_b) = tokio::time::timeout(Duration::from_secs(120), async {
            let (o0, o1) = tokio::join!(collect(&mut g0, vec![]), collect(&mut g1, vec![]));
            Ok::<_, String>((o0?, o1?))
        })
        .await
        .unwrap()
        .unwrap();
        assert!(
            b2_worker.stats.verify_rounds.load(Ordering::Relaxed) > rounds_before_plain,
            "B=2 plain-greedy path must execute speculative verify rounds"
        );

        b2_worker.shutdown().await.unwrap();
        (got_b2, gpu_stoch_a, gpu_stoch_b)
    });
    assert_eq!(
        &got_b2[..],
        &baselines[..2],
        "B=2 batched MTP must match B=1"
    );
    assert_eq!(
        gpu_stoch_a, baselines[2],
        "B=2 plain-greedy slot 0 must match baseline[2]"
    );
    assert_eq!(
        gpu_stoch_b, baselines[3],
        "B=2 plain-greedy slot 1 must match baseline[3]"
    );
}

async fn collect_b5(worker: &EngineWorker, prompts: &[Vec<i32>; 5]) -> Vec<Vec<i32>> {
    // Submit the entire burst without awaiting; use the worker's existing coalescing.
    let [mut s0, mut s1, mut s2, mut s3, mut s4] = prompts
        .each_ref()
        .map(|prompt| worker.submit(prompt.clone(), params(), None).unwrap());
    let result = tokio::time::timeout(Duration::from_secs(120), async {
        let (o0, o1, o2, o3, o4) = tokio::join!(
            collect(&mut s0, vec![]),
            collect(&mut s1, vec![]),
            collect(&mut s2, vec![]),
            collect(&mut s3, vec![]),
            collect(&mut s4, vec![]),
        );
        Ok::<_, String>(vec![o0?, o1?, o2?, o3?, o4?])
    })
    .await
    .unwrap()
    .unwrap();
    worker.shutdown().await.unwrap();
    let outputs = result;
    assert_eq!(worker.stats.peak_active.load(Ordering::Relaxed), 5);
    assert_eq!(worker.stats.verify_rounds.load(Ordering::Relaxed), 0);
    assert_eq!(worker.stats.completed.load(Ordering::Relaxed), 5);
    assert_eq!(worker.stats.cancelled.load(Ordering::Relaxed), 0);
    assert_eq!(outputs[4], outputs[0], "B=5 duplicate slots must match");
    outputs
}

#[test]
fn batched_mtp_b5_fallback_matches_non_speculative_losslessly() {
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "batched_mtp_b5_fallback_matches_non_speculative_losslessly",
    )
    .is_none()
    {
        return;
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let fallback_worker = worker(SpeculativeKind::Mtp);
    let prompts = [
        "The capital of France is",
        "In computer architecture, cache coherence ensures",
        "Rust ownership rules guarantee memory safety by",
        "Binary search finds an element in a sorted array in",
        "The capital of France is",
    ]
    .map(|text| {
        fallback_worker
            .tokenizer
            .as_ref()
            .unwrap()
            .encode(text, false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&id| id as i32)
            .collect()
    });
    // B=5 exceeds MAX_MTP_BATCH_SLOTS=4 and must use ordinary batched decode.
    // Its oracle is fresh non-speculative B=5 with identical prompts/order, not B=1:
    // cuBLAS can choose different reduction algorithms at different M dimensions.
    // This avoids conflating cross-geometry numerical differences with fallback bugs;
    // it does not establish the cause of any particular B=5 versus B=1 mismatch.
    let fallback = rt.block_on(collect_b5(&fallback_worker, &prompts));
    let baseline_worker = worker_with_policy(None, KvFormatPolicy::Tq4);
    let baseline = rt.block_on(collect_b5(&baseline_worker, &prompts));
    assert_eq!(
        fallback, baseline,
        "B=5 MTP fallback must match non-speculative B=5 across all five complete outputs"
    );
}

async fn collect_with_cached(
    stream: &mut GenerationStream,
    expected_len: usize,
) -> Result<(Vec<i32>, usize), String> {
    let mut tokens = Vec::new();
    while let Some(event) = stream.events.recv().await {
        match event {
            GenerationEvent::Token { id, .. } => tokens.push(id),
            GenerationEvent::Finished {
                cached_tokens,
                generated_tokens,
                ..
            } => {
                if generated_tokens != expected_len || tokens.len() != expected_len {
                    return Err(format!(
                        "counts: finished={generated_tokens}, got={}, expected={expected_len}",
                        tokens.len()
                    ));
                }
                return Ok((tokens, cached_tokens));
            }
            GenerationEvent::Error(e) => return Err(e.to_string()),
        }
    }
    Err("stream closed without Finished".into())
}

#[test]
fn batched_mtp_max4_handles_8_burst_and_reuses_warm_prefixes() {
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "batched_mtp_max4_handles_8_burst_and_reuses_warm_prefixes",
    )
    .is_none()
    {
        return;
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let lane_tags = [
        "[Lane Alpha] ",
        "[Lane Beta] ",
        "[Lane Gamma] ",
        "[Lane Delta] ",
    ];
    let shared_prefix = "System architecture notes: requests carry bounded deadlines, workers pull from bounded queues, and cache invalidations execute strictly after transaction commit to preserve serializability across concurrent readers and writers. ".repeat(6);
    let suffixes = [
        "List two race conditions in cache invalidation:",
        "Explain how bounded queues prevent memory exhaustion:",
        "Describe safe cancellation semantics for streaming RPCs:",
        "Summarize retry backoff invariants under overload:",
        "Give pseudocode for atomic lease renewal:",
        "Identify lock-ordering hazards in connection pools:",
        "Explain read-your-writes consistency with replicas:",
        "Outline a deterministic test for timeout cleanup:",
    ];
    let followup_tail = " Next turn: provide a concise checklist for code review.";

    let mut short_params = params();
    short_params.max_tokens = 32;

    let baseline_worker = EngineWorker::from_loader(
        || {
            Qwen3_5Engine::load_for_serving_with_speculative(
                "Qwen/Qwen3.5-4B",
                1,
                KvFormatPolicy::Tq4,
                Some(1024),
                true,
                1,
                Some(&SpeculativeConfig {
                    kind: SpeculativeKind::Mtp,
                    draft_model: None,
                    mtp_k: 3,
                }),
            )
            .map_err(|e| e.to_string())
        },
        SchedulerConfig {
            max_num_seqs: 1,
            ..SchedulerConfig::default()
        },
    )
    .unwrap();

    let tok = baseline_worker.tokenizer.as_ref().unwrap().clone();
    let encode = |text: &str| -> Vec<i32> {
        tok.encode(text, false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&id| id as i32)
            .collect()
    };
    // Pair `(i, i + 4)` shares a >= 128-token prefix while wave 1 (`0..4`) starts cold in both
    // `B=1` and `B=4`, so both workers traverse the exact same 1-hop `WHT128-Q8_0` checkpoints.
    let burst_prompts: Vec<Vec<i32>> = suffixes
        .iter()
        .enumerate()
        .map(|(i, s)| encode(&format!("{}{shared_prefix}{s}", lane_tags[i % 4])))
        .collect();
    assert!(
        burst_prompts.iter().all(|p| p.len() >= 192),
        "shared prefix must cross >= 2 Mamba 64-token checkpoints"
    );
    let followup_suffix_ids = encode(followup_tail);

    let (burst_baselines, followup_prompts, followup_baselines) = rt.block_on(async {
        let mut burst_outs = Vec::with_capacity(burst_prompts.len());
        for p in &burst_prompts {
            let mut st = baseline_worker
                .submit(p.clone(), short_params.clone(), None)
                .unwrap();
            let (out, _) =
                tokio::time::timeout(Duration::from_secs(120), collect_with_cached(&mut st, 32))
                    .await
                    .unwrap()
                    .unwrap();
            burst_outs.push(out);
        }
        let mut follow_prompts = Vec::with_capacity(4);
        let mut follow_outs = Vec::with_capacity(4);
        for i in 0..4 {
            let mut fp = burst_prompts[i].clone();
            fp.extend_from_slice(&burst_outs[i]);
            fp.extend_from_slice(&followup_suffix_ids);
            let mut st = baseline_worker
                .submit(fp.clone(), short_params.clone(), None)
                .unwrap();
            let (out, _) =
                tokio::time::timeout(Duration::from_secs(120), collect_with_cached(&mut st, 32))
                    .await
                    .unwrap()
                    .unwrap();
            follow_prompts.push(fp);
            follow_outs.push(out);
        }
        baseline_worker.shutdown().await.unwrap();
        (burst_outs, follow_prompts, follow_outs)
    });

    let burst_worker = EngineWorker::from_loader(
        || {
            Qwen3_5Engine::load_for_serving_with_speculative(
                "Qwen/Qwen3.5-4B",
                1,
                KvFormatPolicy::Tq4,
                Some(1024),
                true,
                4,
                Some(&SpeculativeConfig {
                    kind: SpeculativeKind::Mtp,
                    draft_model: None,
                    mtp_k: 3,
                }),
            )
            .map_err(|e| e.to_string())
        },
        SchedulerConfig {
            max_num_seqs: 4,
            ..SchedulerConfig::default()
        },
    )
    .unwrap();

    let (burst_got, rounds_after_burst, followup_got, rounds_after_followup) = rt.block_on(async {
        // Submit all 8 burst requests simultaneously while max_num_seqs == 4.
        let mut streams: Vec<_> = burst_prompts
            .iter()
            .map(|p| {
                burst_worker
                    .submit(p.clone(), short_params.clone(), None)
                    .unwrap()
            })
            .collect();
        let mut s_iter = streams.drain(..);
        let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5, mut s6, mut s7) = (
            s_iter.next().unwrap(),
            s_iter.next().unwrap(),
            s_iter.next().unwrap(),
            s_iter.next().unwrap(),
            s_iter.next().unwrap(),
            s_iter.next().unwrap(),
            s_iter.next().unwrap(),
            s_iter.next().unwrap(),
        );
        let burst_got = tokio::time::timeout(Duration::from_secs(120), async {
            let (r0, r1, r2, r3, r4, r5, r6, r7) = tokio::join!(
                collect_with_cached(&mut s0, 32),
                collect_with_cached(&mut s1, 32),
                collect_with_cached(&mut s2, 32),
                collect_with_cached(&mut s3, 32),
                collect_with_cached(&mut s4, 32),
                collect_with_cached(&mut s5, 32),
                collect_with_cached(&mut s6, 32),
                collect_with_cached(&mut s7, 32),
            );
            Ok::<_, String>(vec![r0?, r1?, r2?, r3?, r4?, r5?, r6?, r7?])
        })
        .await
        .unwrap()
        .unwrap();
        let rounds_after_burst = burst_worker.stats.verify_rounds.load(Ordering::Relaxed);

        // Submit a 4-request multi-turn follow-up wave extending the first 4 burst conversations.
        let mut f_streams: Vec<_> = followup_prompts
            .iter()
            .map(|p| {
                burst_worker
                    .submit(p.clone(), short_params.clone(), None)
                    .unwrap()
            })
            .collect();
        let mut f_iter = f_streams.drain(..);
        let (mut f0, mut f1, mut f2, mut f3) = (
            f_iter.next().unwrap(),
            f_iter.next().unwrap(),
            f_iter.next().unwrap(),
            f_iter.next().unwrap(),
        );
        let followup_got = tokio::time::timeout(Duration::from_secs(120), async {
            let (r0, r1, r2, r3) = tokio::join!(
                collect_with_cached(&mut f0, 32),
                collect_with_cached(&mut f1, 32),
                collect_with_cached(&mut f2, 32),
                collect_with_cached(&mut f3, 32),
            );
            Ok::<_, String>(vec![r0?, r1?, r2?, r3?])
        })
        .await
        .unwrap()
        .unwrap();
        let rounds_after_followup = burst_worker.stats.verify_rounds.load(Ordering::Relaxed);

        burst_worker.shutdown().await.unwrap();
        (
            burst_got,
            rounds_after_burst,
            followup_got,
            rounds_after_followup,
        )
    });

    assert_eq!(burst_worker.stats.peak_active.load(Ordering::Relaxed), 4);
    assert!(
        rounds_after_burst >= 80,
        "8-request burst under max_num_seqs=4 must stay in batched MTP instead of plain fallback, got {rounds_after_burst} rounds"
    );
    assert!(
        rounds_after_followup >= rounds_after_burst + 30,
        "4-request follow-up wave must execute batched MTP verify rounds, before={rounds_after_burst} after={rounds_after_followup}"
    );
    assert!(
        burst_worker.stats.accepted_drafts.load(Ordering::Relaxed) > 0,
        "expected accepted drafts across burst and follow-up waves"
    );
    for (i, ((got, _), expected)) in burst_got.iter().zip(&burst_baselines).enumerate() {
        assert_eq!(
            got, expected,
            "burst slot {i} must match B=1 greedy baseline"
        );
    }
    assert!(
        burst_got[4..].iter().all(|(_, cached)| *cached >= 128),
        "second 4-request wave of shared-prefix burst must reuse warm radix+MTP prefix, got {:?}",
        burst_got
            .iter()
            .map(|(_, cached)| *cached)
            .collect::<Vec<_>>()
    );
    for (i, ((got, cached), expected)) in followup_got.iter().zip(&followup_baselines).enumerate() {
        assert_eq!(
            got, expected,
            "follow-up turn {i} must match B=1 greedy baseline"
        );
        assert!(
            *cached >= 192,
            "multi-turn follow-up {i} in batched MTP must reuse prompt+generated prefix, got cached={cached}"
        );
    }
}

#[test]
fn batched_mtp_stochastic_and_tp2_greedy_stay_speculative() {
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "batched_mtp_stochastic_and_tp2_greedy_stay_speculative",
    )
    .is_none()
    {
        return;
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    // 1. Stochastic B=4 batched MTP must stay in speculative verify rounds (not fall back to plain decode).
    let stoch_worker = EngineWorker::from_loader(
        || {
            let mut engine = Qwen3_5Engine::load_for_serving(
                "Qwen/Qwen3.5-4B",
                1,
                KvFormatPolicy::Tq4,
                Some(512),
                true,
                16,
            )
            .map_err(|e| e.to_string())?;
            engine
                .enable_speculative(&SpeculativeConfig {
                    kind: SpeculativeKind::Mtp,
                    mtp_k: 3,
                    draft_model: None,
                })
                .map_err(|e| e.to_string())?;
            Ok(engine)
        },
        SchedulerConfig {
            max_num_seqs: 4,
            ..SchedulerConfig::default()
        },
    )
    .unwrap();

    let prompts: Vec<Vec<i32>> = (0..4)
        .map(|i| {
            let mut p: Vec<i32> = (0..96).map(|j| 200 + ((j * 5) % 89)).collect();
            p.extend((0..16).map(|j| 600 + (i * 19 + j)));
            p
        })
        .collect();

    let (stoch_got, stoch_rounds, stoch_accepted) = rt.block_on(async {
        let mut streams: Vec<_> = prompts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let sp = SamplingParams {
                    max_tokens: 32,
                    ignore_eos: true,
                    temperature: 0.7,
                    top_p: 0.9,
                    seed: 1000 + i as u64,
                    ..SamplingParams::default()
                };
                stoch_worker.submit(p.clone(), sp, None).unwrap()
            })
            .collect();
        let mut it = streams.drain(..);
        let (mut s0, mut s1, mut s2, mut s3) = (
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
        );
        let out = tokio::time::timeout(Duration::from_secs(120), async {
            let (r0, r1, r2, r3) = tokio::join!(
                collect_with_cached(&mut s0, 32),
                collect_with_cached(&mut s1, 32),
                collect_with_cached(&mut s2, 32),
                collect_with_cached(&mut s3, 32),
            );
            Ok::<_, String>(vec![r0?, r1?, r2?, r3?])
        })
        .await
        .unwrap()
        .unwrap();
        let rounds = stoch_worker.stats.verify_rounds.load(Ordering::Relaxed);
        let acc = stoch_worker.stats.accepted_drafts.load(Ordering::Relaxed);
        stoch_worker.shutdown().await.unwrap();
        (out, rounds, acc)
    });
    assert!(
        stoch_rounds >= 20,
        "stochastic B=4 must execute batched MTP verify rounds, got {stoch_rounds}"
    );
    assert!(
        stoch_accepted > 0,
        "stochastic B=4 batched MTP must accept draft tokens"
    );
    assert!(stoch_got.iter().all(|(toks, _)| toks.len() == 32));

    // 2. TP=2 greedy B=4 batched MTP must match B=1 non-speculative greedy bit-for-bit.
    let tp2_plain = EngineWorker::from_loader(
        || {
            Qwen3_5Engine::load_for_serving(
                "Qwen/Qwen3.5-4B",
                2,
                KvFormatPolicy::Tq4,
                Some(512),
                true,
                16,
            )
            .map_err(|e| e.to_string())
        },
        SchedulerConfig {
            max_num_seqs: 1,
            ..SchedulerConfig::default()
        },
    )
    .unwrap();
    let greedy_short = SamplingParams {
        max_tokens: 32,
        ignore_eos: true,
        temperature: 0.0,
        ..SamplingParams::default()
    };
    let tp2_expected = rt.block_on(async {
        let mut out = Vec::with_capacity(prompts.len());
        for p in &prompts {
            let mut s = tp2_plain
                .submit(p.clone(), greedy_short.clone(), None)
                .unwrap();
            let (toks, _) =
                tokio::time::timeout(Duration::from_secs(90), collect_with_cached(&mut s, 32))
                    .await
                    .unwrap()
                    .unwrap();
            out.push(toks);
        }
        tp2_plain.shutdown().await.unwrap();
        out
    });

    let tp2_mtp = EngineWorker::from_loader(
        || {
            let mut engine = Qwen3_5Engine::load_for_serving(
                "Qwen/Qwen3.5-4B",
                2,
                KvFormatPolicy::Tq4,
                Some(512),
                true,
                16,
            )
            .map_err(|e| e.to_string())?;
            engine
                .enable_speculative(&SpeculativeConfig {
                    kind: SpeculativeKind::Mtp,
                    mtp_k: 3,
                    draft_model: None,
                })
                .map_err(|e| e.to_string())?;
            Ok(engine)
        },
        SchedulerConfig {
            max_num_seqs: 4,
            ..SchedulerConfig::default()
        },
    )
    .unwrap();
    let (tp2_got, tp2_rounds, tp2_accepted) = rt.block_on(async {
        let mut streams: Vec<_> = prompts
            .iter()
            .map(|p| {
                tp2_mtp
                    .submit(p.clone(), greedy_short.clone(), None)
                    .unwrap()
            })
            .collect();
        let mut it = streams.drain(..);
        let (mut s0, mut s1, mut s2, mut s3) = (
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
        );
        let out = tokio::time::timeout(Duration::from_secs(120), async {
            let (r0, r1, r2, r3) = tokio::join!(
                collect_with_cached(&mut s0, 32),
                collect_with_cached(&mut s1, 32),
                collect_with_cached(&mut s2, 32),
                collect_with_cached(&mut s3, 32),
            );
            Ok::<_, String>(vec![r0?, r1?, r2?, r3?])
        })
        .await
        .unwrap()
        .unwrap();
        let rounds = tp2_mtp.stats.verify_rounds.load(Ordering::Relaxed);
        let acc = tp2_mtp.stats.accepted_drafts.load(Ordering::Relaxed);
        tp2_mtp.shutdown().await.unwrap();
        (out, rounds, acc)
    });
    assert!(
        tp2_rounds >= 30,
        "TP=2 B=4 batched MTP must execute verify rounds, got {tp2_rounds}"
    );
    assert!(tp2_accepted > 0, "TP=2 B=4 batched MTP must accept drafts");
    for (i, ((got, _), exp)) in tp2_got.iter().zip(&tp2_expected).enumerate() {
        assert_eq!(got, exp, "TP=2 B=4 slot {i} must match greedy baseline");
    }
}

#[test]
fn batched_mtp_full_window_budgets_stay_speculative_and_preserve_radix_cache() {
    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "batched_mtp_full_window_budgets_stay_speculative_and_preserve_radix_cache",
    )
    .is_none()
    {
        return;
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    // kv_pages = 512 => 8,192 total tokens, MTP batch slot_cap = (512 / 4) * 16 = 2,048 tokens.
    // Four concurrent requests with `max_tokens = 8192 - prompt_len` (exceeding `slot_cap` and
    // exceeding 100% of the KV pool in sum) must still admit together on tick 1 (`peak_active = 4`),
    // run batched MTP verify rounds (`verify_rounds > 0`), and leave an earlier warm session's
    // MambaRadixCache prefix intact via incremental KV slab allocation.
    let worker = EngineWorker::from_loader(
        || {
            let mut engine = Qwen3_5Engine::load_for_serving(
                "Qwen/Qwen3.5-4B",
                1,
                KvFormatPolicy::Tq4,
                Some(512),
                true,
                8,
            )
            .map_err(|e| e.to_string())?;
            engine
                .enable_speculative(&SpeculativeConfig {
                    kind: SpeculativeKind::Mtp,
                    mtp_k: 3,
                    draft_model: None,
                })
                .map_err(|e| e.to_string())?;
            Ok(engine)
        },
        SchedulerConfig {
            max_num_seqs: 4,
            ..SchedulerConfig::default()
        },
    )
    .unwrap();
    let max_len = worker.max_model_len();
    assert_eq!(max_len, 8192);

    let tok = worker.tokenizer.as_ref().unwrap().clone();
    let encode = |text: &str| -> Vec<i32> {
        tok.encode(text, false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&id| id as i32)
            .collect()
    };

    let warm_prefix_text = "Distributed consensus invariant log: every committed raft entry carries a monotonic term and index across quorum replicas, and follower snapshots advance state machines atomically at chunk boundaries. ".repeat(6);
    let warm_seed_prompt = encode(&format!(
        "{warm_prefix_text}Summarize leader election safety in one phrase:"
    ));
    let warm_probe_prompt = encode(&format!(
        "{warm_prefix_text}Summarize log matching safety in one phrase:"
    ));
    assert!(
        warm_seed_prompt.len() >= 192,
        "warm session prefix must span >= 3 Mamba 64-token checkpoints"
    );

    let burst_prompts: Vec<Vec<i32>> = [
        "The capital of France is",
        "In computer architecture, cache coherence ensures",
        "Rust ownership rules guarantee memory safety by",
        "Binary search finds an element in a sorted array in",
    ]
    .iter()
    .map(|s| encode(s))
    .collect();

    async fn collect_first_n_then_cancel(
        mut stream: GenerationStream,
        target: usize,
    ) -> Result<Vec<i32>, String> {
        let mut out = Vec::with_capacity(target);
        while let Some(ev) = stream.events.recv().await {
            match ev {
                GenerationEvent::Token { id, .. } => {
                    out.push(id);
                    if out.len() >= target {
                        drop(stream);
                        return Ok(out);
                    }
                }
                GenerationEvent::Finished { .. } => return Ok(out),
                GenerationEvent::Error(e) => return Err(e.to_string()),
            }
        }
        Err("stream ended before target tokens".into())
    }

    let (rounds_delta, probe_cached) = rt.block_on(async {
        // 1. Seed a warm 192+-token session into MambaRadixCache.
        let mut seed_stream = worker
            .submit(
                warm_seed_prompt,
                SamplingParams {
                    max_tokens: 16,
                    ignore_eos: true,
                    temperature: 0.0,
                    ..SamplingParams::default()
                },
                None,
            )
            .unwrap();
        let _ = tokio::time::timeout(
            Duration::from_secs(60),
            collect_with_cached(&mut seed_stream, 16),
        )
        .await
        .unwrap()
        .unwrap();

        // 2. Launch 4 concurrent full-window requests (`max_tokens = max_len - prompt.len()`,
        //    each >7,900 tokens > slot_cap=2,048, sum >31,000 tokens >> 8,192 KV pool).
        let rounds_before = worker.stats.verify_rounds.load(Ordering::Relaxed);
        let mut streams: Vec<_> = burst_prompts
            .iter()
            .map(|p| {
                let sp = SamplingParams {
                    max_tokens: max_len - p.len(),
                    ignore_eos: true,
                    temperature: 0.0,
                    ..SamplingParams::default()
                };
                worker
                    .submit_with_images_lora_and_budget_mode(
                        p.clone(),
                        Vec::new(),
                        sp,
                        None,
                        warp34_host::lora::BASE_MODEL_LORA_ID,
                        true,
                    )
                    .unwrap()
            })
            .collect();
        let mut it = streams.drain(..);
        let (s0, s1, s2, s3) = (
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
        );
        tokio::time::timeout(Duration::from_secs(120), async {
            let (r0, r1, r2, r3) = tokio::join!(
                collect_first_n_then_cancel(s0, 32),
                collect_first_n_then_cancel(s1, 32),
                collect_first_n_then_cancel(s2, 32),
                collect_first_n_then_cancel(s3, 32),
            );
            r0?;
            r1?;
            r2?;
            r3?;
            Ok::<_, String>(())
        })
        .await
        .unwrap()
        .unwrap();
        let rounds_after = worker.stats.verify_rounds.load(Ordering::Relaxed);

        // 3. Probe the earlier warm prefix: incremental KV slab allocation must have kept it
        //    resident in MambaRadixCache rather than evicting the pool for 4 full-window budgets.
        let mut probe_stream = worker
            .submit(
                warm_probe_prompt,
                SamplingParams {
                    max_tokens: 16,
                    ignore_eos: true,
                    temperature: 0.0,
                    ..SamplingParams::default()
                },
                None,
            )
            .unwrap();
        let (_, cached) = tokio::time::timeout(
            Duration::from_secs(60),
            collect_with_cached(&mut probe_stream, 16),
        )
        .await
        .unwrap()
        .unwrap();
        worker.shutdown().await.unwrap();
        (rounds_after - rounds_before, cached)
    });

    assert_eq!(
        worker.stats.peak_active.load(Ordering::Relaxed),
        4,
        "4 concurrent full-window soft-budget requests must admit together instead of serializing"
    );
    assert!(
        rounds_delta >= 20,
        "4 concurrent full-window requests must execute batched MTP verify rounds, got {rounds_delta}"
    );
    assert!(
        probe_cached >= 128,
        "warm MambaRadixCache prefix must survive concurrent full-window requests via incremental KV slabs, got cached={probe_cached}"
    );
}

#[test]
fn realistic_swe_trajectory_caches_across_plain_mtp_and_dflash2() {
    use serde_json::{Value, json};
    use warp34_host::{
        chatml::{ChatMessage, format_chatml_prompt_with_options, parse_history_tools},
        parser::Qwen3Parser,
    };

    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "realistic_swe_trajectory_caches_across_plain_mtp_and_dflash2",
    )
    .is_none()
        || common::require_or_skip_cached_model(
            "nchapman/Qwen3.5-4B-DFlash2",
            "realistic_swe_trajectory_caches_across_plain_mtp_and_dflash2",
        )
        .is_none()
    {
        return;
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    // Load the first realistic trajectory from `benchmarks/perf/fixtures/open_swe_intact_16k.json`.
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/perf/fixtures/open_swe_intact_16k.json");
    let fixture_json: Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture_path).unwrap()).unwrap();
    let traj = &fixture_json["trajectories"][0];
    let system_prompt = traj["system"].as_str().unwrap();
    let turns = traj["turns"].as_array().unwrap();

    let run_trajectory = |kind: Option<SpeculativeKind>| -> (Vec<Vec<i32>>, Vec<usize>, usize) {
        let worker = EngineWorker::from_loader(
            move || {
                let mut engine = Qwen3_5Engine::load_for_serving(
                    "Qwen/Qwen3.5-4B",
                    1,
                    KvFormatPolicy::Tq4,
                    Some(512),
                    true,
                    16,
                )
                .map_err(|e| e.to_string())?;
                if let Some(kind) = kind {
                    engine
                        .enable_speculative(&SpeculativeConfig {
                            kind,
                            mtp_k: 3,
                            draft_model: (kind == SpeculativeKind::DFlash2)
                                .then(|| "nchapman/Qwen3.5-4B-DFlash2".into()),
                        })
                        .map_err(|e| e.to_string())?;
                }
                Ok(engine)
            },
            SchedulerConfig {
                max_num_seqs: 1,
                ..SchedulerConfig::default()
            },
        )
        .unwrap();
        let tok = worker.tokenizer.as_ref().unwrap().clone();
        let encode = |text: &str| -> Vec<i32> {
            tok.encode(text, false)
                .unwrap()
                .get_ids()
                .iter()
                .map(|&id| id as i32)
                .collect()
        };
        let decode = |ids: &[i32]| -> String {
            let u32_ids: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
            tok.decode(&u32_ids, false).unwrap()
        };

        let mut history = vec![ChatMessage {
            role: "system".into(),
            content: Some(system_prompt.into()),
            reasoning_content: None,
            tool_calls: vec![],
        }];
        let mut step_outputs = Vec::new();
        let mut step_cached = Vec::new();
        let mut prev_prompt_len = 0usize;

        for (step, turn) in turns.iter().take(4).enumerate() {
            if let Some(user_text) = turn["user"].as_str() {
                history.push(ChatMessage {
                    role: "user".into(),
                    content: Some(user_text.into()),
                    reasoning_content: None,
                    tool_calls: vec![],
                });
            }
            if let Some(tools) = turn["tools"].as_array()
                && !tools.is_empty()
            {
                let calls_val: Vec<Value> = tools
                    .iter()
                    .map(|t| {
                        json!({
                            "function": {
                                "name": t["name"],
                                "arguments": t["arguments"]
                            }
                        })
                    })
                    .collect();
                let calls = parse_history_tools(Some(&Value::Array(calls_val))).unwrap();
                history.push(ChatMessage {
                    role: "assistant".into(),
                    content: None,
                    reasoning_content: None,
                    tool_calls: calls,
                });
                for t in tools {
                    history.push(ChatMessage {
                        role: "tool".into(),
                        content: Some(t["result"].as_str().unwrap_or("").into()),
                        reasoning_content: None,
                        tool_calls: vec![],
                    });
                }
            }

            let prompt_str = format_chatml_prompt_with_options(&history, None, true, true);
            let prompt_ids = encode(&prompt_str);
            let (out_ids, cached) = rt.block_on(async {
                let mut stream = worker
                    .submit(
                        prompt_ids.clone(),
                        SamplingParams {
                            max_tokens: 24,
                            ignore_eos: true,
                            temperature: 0.0,
                            ..SamplingParams::default()
                        },
                        None,
                    )
                    .unwrap();
                tokio::time::timeout(
                    Duration::from_secs(60),
                    collect_with_cached(&mut stream, 24),
                )
                .await
                .unwrap()
                .unwrap()
            });

            if step > 0 {
                let expected_min_cached = (prev_prompt_len / 64) * 64;
                assert!(
                    cached >= expected_min_cached,
                    "mode={kind:?} step={step}: expected cached >= {expected_min_cached} (prev_prompt_len={prev_prompt_len}), got cached={cached}"
                );
            }
            prev_prompt_len = prompt_ids.len();
            step_cached.push(cached);

            let raw_text = decode(&out_ids);
            let parsed = Qwen3Parser::parse_complete(&raw_text, true);
            history.push(ChatMessage {
                role: "assistant".into(),
                content: parsed.content,
                reasoning_content: parsed.reasoning_content,
                tool_calls: vec![],
            });
            step_outputs.push(out_ids);
        }

        // Also verify mid-turn cancellation preserves committed prefix for the follow-up turn.
        let mut cancel_history = history.clone();
        cancel_history.push(ChatMessage {
            role: "user".into(),
            content: Some("Now run pytest on the modified module and verify.".into()),
            reasoning_content: None,
            tool_calls: vec![],
        });
        let cancel_prompt_ids = encode(&format_chatml_prompt_with_options(
            &cancel_history,
            None,
            true,
            true,
        ));
        let post_cancel_cached = rt.block_on(async {
            let mut stream = worker
                .submit(
                    cancel_prompt_ids.clone(),
                    SamplingParams {
                        max_tokens: 48,
                        ignore_eos: true,
                        temperature: 0.0,
                        ..SamplingParams::default()
                    },
                    None,
                )
                .unwrap();
            // Collect 4 tokens then drop `stream` to trigger client cancellation.
            let mut got = 0usize;
            while let Some(ev) = stream.events.recv().await {
                if matches!(ev, GenerationEvent::Token { .. }) {
                    got += 1;
                    if got >= 4 {
                        break;
                    }
                }
            }
            drop(stream);

            // Follow-up request with the same prompt + extra user suffix must hit the cache.
            let mut follow_ids = cancel_prompt_ids.clone();
            follow_ids.extend_from_slice(&encode("\n Confirm exit code 0."));
            let mut follow_stream = worker
                .submit(
                    follow_ids,
                    SamplingParams {
                        max_tokens: 8,
                        ignore_eos: true,
                        temperature: 0.0,
                        ..SamplingParams::default()
                    },
                    None,
                )
                .unwrap();
            let (_, cached) = tokio::time::timeout(
                Duration::from_secs(60),
                collect_with_cached(&mut follow_stream, 8),
            )
            .await
            .unwrap()
            .unwrap();
            worker.shutdown().await.unwrap();
            cached
        });

        let expected_cancel_cached = ((cancel_prompt_ids.len() - 1) / 64) * 64;
        assert!(
            post_cancel_cached >= expected_cancel_cached,
            "mode={kind:?}: post-cancel follow-up must preserve committed prefix >= {expected_cancel_cached}, got {post_cancel_cached}"
        );

        (step_outputs, step_cached, post_cancel_cached)
    };

    let (plain_outs, plain_cached, _) = run_trajectory(None);
    let (mtp_outs, mtp_cached, _) = run_trajectory(Some(SpeculativeKind::Mtp));
    let (dflash_outs, dflash_cached, _) = run_trajectory(Some(SpeculativeKind::DFlash2));

    assert_eq!(
        mtp_outs, plain_outs,
        "MTP trajectory tokens must match Plain bit-for-bit"
    );
    assert_eq!(
        dflash_outs, plain_outs,
        "DFlash2 trajectory tokens must match Plain bit-for-bit"
    );
    assert_eq!(
        mtp_cached, plain_cached,
        "MTP cached prefix depths must match Plain on every trajectory turn"
    );
    assert_eq!(
        dflash_cached, plain_cached,
        "DFlash2 cached prefix depths must match Plain on every trajectory turn"
    );
}

#[test]
fn mtp_plain_memo_siblings_and_dp_l2_warmup_and_hol_yield_regressions() {
    use std::sync::Arc;
    use warp34_host::dp_prefix_pool::SharedDpPrefixPool;

    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "mtp_plain_memo_siblings_and_dp_l2_warmup_and_hol_yield_regressions",
    )
    .is_none()
    {
        return;
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let l2_pool = Arc::new(SharedDpPrefixPool::new(512 * 1024 * 1024, 16));

    // Phase 1: `max_tokens = 1` warmup on an MTP worker (`1024` KV pages -> `max_tap_rows = 256`
    // while `mtp_prefill_max_rows = 2048`) must publish BOTH KV+GDN AND `proposer_features` to L2.
    let pool_w0 = Arc::clone(&l2_pool);
    let w0 = EngineWorker::from_loader(
        move || {
            let mut engine = Qwen3_5Engine::load_for_serving(
                "Qwen/Qwen3.5-4B",
                1,
                KvFormatPolicy::Tq4,
                Some(1024),
                true,
                16,
            )
            .map_err(|e| e.to_string())?;
            engine
                .enable_speculative(&SpeculativeConfig {
                    kind: SpeculativeKind::Mtp,
                    mtp_k: 3,
                    draft_model: None,
                })
                .map_err(|e| e.to_string())?;
            engine.set_dp_prefix_pool(0, pool_w0);
            Ok(engine)
        },
        SchedulerConfig::default(),
    )
    .unwrap();

    let base_ids: Vec<i32> = w0
        .tokenizer
        .as_ref()
        .unwrap()
        .encode(
            "Explain modern CPU out-of-order execution, register renaming, reorder buffers, and speculative branch recovery in detail. ",
            false,
        )
        .unwrap()
        .get_ids()
        .iter()
        .map(|&id| id as i32)
        .collect();
    let mut harness_2112 = Vec::with_capacity(2120);
    while harness_2112.len() < 2120 {
        harness_2112.extend_from_slice(&base_ids);
    }
    harness_2112.truncate(2120); // > 2048 -> donates 2112-token (33-chunk) prefix

    rt.block_on(async {
        let mut warm = w0
            .submit(
                harness_2112.clone(),
                SamplingParams {
                    max_tokens: 1,
                    ignore_eos: true,
                    temperature: 0.0,
                    ..SamplingParams::default()
                },
                None,
            )
            .unwrap();
        let (_, _) =
            tokio::time::timeout(Duration::from_secs(60), collect_with_cached(&mut warm, 1))
                .await
                .unwrap()
                .unwrap();
        w0.shutdown().await.unwrap();
    });

    assert!(
        l2_pool.stats().publishes >= 1,
        "max_tokens=1 MTP warmup must publish to SharedDpPrefixPool"
    );
    assert_eq!(
        l2_pool.peek_longest_match(-1, &harness_2112, 0),
        Some((2112, true)),
        "max_tokens=1 MTP warmup must include proposer_features in L2"
    );

    // Phase 2: Fresh MTP worker `w1` hydrates the 2112-token prefix from L2, runs a concurrent
    // `B=6` burst sharing that >2048 harness (forcing `step_plain`), and then runs Turn 2 on
    // sibling #5 — proving `init_plain_memo` did NOT steal the shared harness memo from siblings
    // 1..5 and that Turn 2 resumes MTP-3 with full cached prefix!
    let pool_w1 = Arc::clone(&l2_pool);
    let w1 = EngineWorker::from_loader(
        move || {
            let mut engine = Qwen3_5Engine::load_for_serving(
                "Qwen/Qwen3.5-4B",
                1,
                KvFormatPolicy::Tq4,
                Some(1024),
                true,
                16,
            )
            .map_err(|e| e.to_string())?;
            engine
                .enable_speculative(&SpeculativeConfig {
                    kind: SpeculativeKind::Mtp,
                    mtp_k: 3,
                    draft_model: None,
                })
                .map_err(|e| e.to_string())?;
            engine.set_dp_prefix_pool(1, pool_w1);
            Ok(engine)
        },
        SchedulerConfig::default(),
    )
    .unwrap();

    rt.block_on(async {
        // Hydrate L2 onto `w1` and confirm MTP-3 verify rounds fire immediately.
        let mut hydrated_prompt = harness_2112[..2112].to_vec();
        hydrated_prompt.extend_from_slice(&base_ids[..16]);
        let mut s_hyd = w1
            .submit(
                hydrated_prompt,
                SamplingParams {
                    max_tokens: 16,
                    ignore_eos: true,
                    temperature: 0.0,
                    ..SamplingParams::default()
                },
                None,
            )
            .unwrap();
        let (_, hyd_cached) =
            tokio::time::timeout(Duration::from_secs(60), collect_with_cached(&mut s_hyd, 16))
                .await
                .unwrap()
                .unwrap();
        assert_eq!(hyd_cached, 2112);
        assert!(w1.stats.verify_rounds.load(Ordering::Relaxed) > 0);

        // Concurrent B=6 burst (> MAX_MTP_BATCH_SLOTS = 4) sharing the 2112-token harness:
        // all 6 enter `step_plain`, and sibling #5 must still retain its plain_memo for Turn 2!
        let mut streams = Vec::new();
        let mut prompts_b6 = Vec::new();
        for idx in 0..6usize {
            let mut p = harness_2112[..2112].to_vec();
            p.push(base_ids[idx % base_ids.len()]);
            p.extend_from_slice(&base_ids[..20]);
            streams.push(
                w1.submit(
                    p.clone(),
                    SamplingParams {
                        max_tokens: 16,
                        ignore_eos: true,
                        temperature: 0.0,
                        ..SamplingParams::default()
                    },
                    None,
                )
                .unwrap(),
            );
            prompts_b6.push(p);
        }
        let mut outs_b6 = Vec::new();
        for s in &mut streams {
            let (toks, cached) =
                tokio::time::timeout(Duration::from_secs(60), collect_with_cached(s, 16))
                    .await
                    .unwrap()
                    .unwrap();
            assert_eq!(cached, 2112, "every B=6 sibling must hit the 2112-token prefix");
            outs_b6.push(toks);
        }

        // Turn 2 of sibling #5 (2112 + 21 + 16 = 2149 tokens + 16 new suffix = 2165 tokens,
        // so aligned 2112+32 = 2144 > 2112!) must resume MTP-3 cleanly without losing cache!
        let rounds_before_turn2 = w1.stats.verify_rounds.load(Ordering::Relaxed);
        let mut sib5_turn2 = prompts_b6[5].clone();
        sib5_turn2.extend_from_slice(&outs_b6[5]);
        sib5_turn2.extend_from_slice(&base_ids[..16]);
        let mut s_turn2 = w1
            .submit(
                sib5_turn2,
                SamplingParams {
                    max_tokens: 16,
                    ignore_eos: true,
                    temperature: 0.0,
                    ..SamplingParams::default()
                },
                None,
            )
            .unwrap();
        let (_, turn2_cached) =
            tokio::time::timeout(Duration::from_secs(60), collect_with_cached(&mut s_turn2, 16))
                .await
                .unwrap()
                .unwrap();
        assert!(
            turn2_cached >= 2112,
            "sibling #5 Turn 2 must hit full cached prefix >= 2112, got {turn2_cached}"
        );
        assert!(
            w1.stats.verify_rounds.load(Ordering::Relaxed) > rounds_before_turn2,
            "sibling #5 Turn 2 must resume MTP-3 verify rounds (plain_memo was not stolen by sibling #0)"
        );

        // Phase 3: Insert a distinct 2112-token prefix into L2 WITHOUT proposer_features
        // (simulating a plain-mode spillover entry) and submit it WHILE an MTP request is
        // actively decoding. The worker must yield MTP to `step_plain` instead of starving
        // `sched.waiting` in an infinite head-of-line stall!
        w1.shutdown().await.unwrap();
    });

    let plain_l2 = Arc::new(SharedDpPrefixPool::new(512 * 1024 * 1024, 16));
    let alt_tok = *base_ids.iter().find(|&&t| t != base_ids[0]).unwrap();
    let mut distinct_2120 = harness_2112.clone();
    distinct_2120[0] = alt_tok;
    {
        let pool_p = Arc::clone(&plain_l2);
        let wp = EngineWorker::from_loader(
            move || {
                let mut engine = Qwen3_5Engine::load_for_serving(
                    "Qwen/Qwen3.5-4B",
                    1,
                    KvFormatPolicy::Tq4,
                    Some(512),
                    true,
                    16,
                )
                .map_err(|e| e.to_string())?;
                engine.set_dp_prefix_pool(0, pool_p);
                Ok(engine)
            },
            SchedulerConfig::default(),
        )
        .unwrap();
        rt.block_on(async {
            let mut s = wp
                .submit(
                    distinct_2120.clone(),
                    SamplingParams {
                        max_tokens: 1,
                        ignore_eos: true,
                        temperature: 0.0,
                        ..SamplingParams::default()
                    },
                    None,
                )
                .unwrap();
            let _ = tokio::time::timeout(Duration::from_secs(60), collect_with_cached(&mut s, 1))
                .await
                .unwrap()
                .unwrap();
            wp.shutdown().await.unwrap();
        });
    }
    assert_eq!(
        plain_l2.peek_longest_match(-1, &distinct_2120, 0),
        Some((2112, false)),
        "plain-mode max_tokens=1 warmup must publish KV+GDN with proposer_features=None"
    );

    let pool_w2 = Arc::clone(&plain_l2);
    let w2 = EngineWorker::from_loader(
        move || {
            let mut engine = Qwen3_5Engine::load_for_serving(
                "Qwen/Qwen3.5-4B",
                1,
                KvFormatPolicy::Tq4,
                Some(1024),
                true,
                16,
            )
            .map_err(|e| e.to_string())?;
            engine
                .enable_speculative(&SpeculativeConfig {
                    kind: SpeculativeKind::Mtp,
                    mtp_k: 3,
                    draft_model: None,
                })
                .map_err(|e| e.to_string())?;
            engine.set_dp_prefix_pool(1, pool_w2);
            Ok(engine)
        },
        SchedulerConfig::default(),
    )
    .unwrap();

    rt.block_on(async {
        // Start an active MTP request first, wait for 2 tokens so it is mid-decode in MTP batch mode,
        // then enqueue the 2112-token unwarmed L2 request behind it!
        let mut active_mtp = w2
            .submit(
                base_ids.clone(),
                SamplingParams {
                    max_tokens: 48,
                    ignore_eos: true,
                    temperature: 0.0,
                    ..SamplingParams::default()
                },
                None,
            )
            .unwrap();
        let mut prefix_out = Vec::new();
        tokio::time::timeout(Duration::from_secs(30), async {
            while prefix_out.len() < 2 {
                match active_mtp.events.recv().await {
                    Some(GenerationEvent::Token { id, .. }) => prefix_out.push(id),
                    Some(GenerationEvent::Finished { .. }) => {
                        panic!("active_mtp finished early before 2 tokens")
                    }
                    Some(GenerationEvent::Error(e)) => panic!("active_mtp errored early: {e}"),
                    None => panic!("active_mtp stream closed before 2 tokens"),
                }
            }
        })
        .await
        .expect("timed out waiting for first 2 active_mtp tokens");
        let mut unwarmed_req = distinct_2120[..2112].to_vec();
        unwarmed_req.extend_from_slice(&base_ids[..16]);
        let mut s_unwarmed = w2
            .submit(
                unwarmed_req,
                SamplingParams {
                    max_tokens: 16,
                    ignore_eos: true,
                    temperature: 0.0,
                    ..SamplingParams::default()
                },
                None,
            )
            .unwrap();
        // Both must finish promptly without head-of-line queue starvation, and `s_unwarmed`
        // must hydrate all 2112 tokens from L2!
        let (res_active, res_unwarmed) = tokio::time::timeout(Duration::from_secs(30), async {
            let mut rem_active = prefix_out;
            while let Some(ev) = active_mtp.events.recv().await {
                match ev {
                    GenerationEvent::Token { id, .. } => rem_active.push(id),
                    GenerationEvent::Finished { .. } => break,
                    GenerationEvent::Error(e) => return Err(e.to_string()),
                }
            }
            let (_, cached) = collect_with_cached(&mut s_unwarmed, 16).await?;
            Ok::<_, String>((rem_active.len(), cached))
        })
        .await
        .expect("unwarmed >=2048 radix arrival must not deadlock behind active MTP")
        .unwrap();
        assert_eq!(res_active, 48);
        assert_eq!(res_unwarmed, 2112);
        w2.shutdown().await.unwrap();
    });
}

#[test]
fn non_gdn_l2_hydration_allocates_unique_state_slots_without_corrupting_pool() {
    use std::sync::Arc;
    use warp34_host::dp_prefix_pool::SharedDpPrefixPool;

    if common::require_or_skip_cached_model(
        "Qwen/Qwen3.5-4B",
        "non_gdn_l2_hydration_allocates_unique_state_slots_without_corrupting_pool",
    )
    .is_none()
    {
        return;
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let l2 = Arc::new(SharedDpPrefixPool::new(256 * 1024 * 1024, 16));
    let mut e0 = Qwen3_5Engine::load_for_serving(
        "Qwen/Qwen3.5-4B",
        1,
        KvFormatPolicy::Tq4,
        Some(128),
        false,
        8,
    )
    .unwrap();
    // Configure `e0` as a pure full-attention + PLE model (`!is_hybrid_gdn && uses_ple()`),
    // and truncate `kv_pool` downward to verify immutable page strides on truncated TP ranks.
    e0.cfg.num_hidden_layers = e0.cfg.num_full_attention_layers();
    e0.cfg.ple_embed_dim = 64;
    e0.cfg.ple_layer_ids = vec![0];
    e0.ranks[0].kv_pool.truncate_pages(96);
    e0.ranks[0].state_pool.init_ple_state(64, 0).unwrap();
    e0.ranks[0].checkpoint_pool = None;
    assert!(!e0.has_checkpoint_pool());
    e0.set_dp_prefix_pool(0, Arc::clone(&l2));

    // Publish two distinct 256-token prefixes from e0 into L2 with distinct PLE n-gram contexts
    // AND distinct nonzero `ple_conv_state` buffers.
    let ple_slot_bytes = e0.ranks[0].state_pool.ple_conv_slot_bytes();
    assert!(ple_slot_bytes > 0);
    let conv_a = vec![0xA5u8; ple_slot_bytes];
    let conv_b = vec![0x5Au8; ple_slot_bytes];
    let seq_a: Vec<i32> = (10..266).collect();
    let seq_b: Vec<i32> = (500..756).collect();
    let mut rad0 = warp34_host::radix::MambaRadixCache::new(16, 64);
    for (seq, ple_ctx, conv_bytes) in [(&seq_a, [42, 99], &conv_a), (&seq_b, [77, 88], &conv_b)] {
        let mut pages = Vec::with_capacity(16);
        for _ in 0..16 {
            pages.push(e0.ranks[0].kv_pool.alloc_page().unwrap());
        }
        let slot = e0.alloc_state_slot_from(&mut rad0).unwrap();
        e0.ranks[0].state_pool.ple_ngram_context[slot as usize] = ple_ctx;
        e0.ranks[0]
            .state_pool
            .import_ple_conv_host_bytes_to_slot(slot, conv_bytes, &e0.ranks[0].stream)
            .unwrap();
        let ins = rad0.insert(seq, &pages, &[], slot, 0);
        e0.try_publish_radix_node_to_shared_pool_with_features(
            &rad0,
            ins.terminal_node,
            -1,
            seq,
            false,
            || None,
        );
    }
    assert_eq!(l2.stats().entries, 2);

    // Hydrate BOTH prefixes onto e1 while slot 0 is actively owned by an in-flight sequence,
    // verify PLE contexts AND `ple_conv_state` round-trip into distinct `state_pool` slots (`!= 0`),
    // then evict the radix cache and verify `state_pool.free_count()` never pushes duplicate `0`s!
    let mut e1 = Qwen3_5Engine::load_for_serving(
        "Qwen/Qwen3.5-4B",
        1,
        KvFormatPolicy::Tq4,
        Some(128),
        false,
        8,
    )
    .unwrap();
    e1.cfg.num_hidden_layers = e1.cfg.num_full_attention_layers();
    e1.cfg.ple_embed_dim = 64;
    e1.cfg.ple_layer_ids = vec![0];
    e1.ranks[0].kv_pool.truncate_pages(80);
    e1.ranks[0].state_pool.init_ple_state(64, 0).unwrap();
    e1.ranks[0].checkpoint_pool = None;
    e1.set_dp_prefix_pool(1, Arc::clone(&l2));
    let total_slots = e1.ranks[0].state_pool.total_slots;
    let active_slot = e1.ranks[0].state_pool.alloc_slot().unwrap();
    assert_eq!(active_slot, 0);
    let conv_active = vec![0xCCu8; ple_slot_bytes];
    e1.ranks[0].state_pool.ple_ngram_context[0] = [111, 222];
    e1.ranks[0]
        .state_pool
        .import_ple_conv_host_bytes_to_slot(0, &conv_active, &e1.ranks[0].stream)
        .unwrap();

    let mut q_a = seq_a.clone();
    q_a.push(999);
    let mut q_b = seq_b.clone();
    q_b.push(999);
    let mut rad = warp34_host::radix::MambaRadixCache::new(16, 64);
    assert!(
        e1.try_hydrate_waiting_from_shared_pool(&mut rad, -1, &q_a)
            .is_some()
    );
    assert!(
        e1.try_hydrate_waiting_from_shared_pool(&mut rad, -1, &q_b)
            .is_some()
    );
    assert_eq!(e1.ranks[0].state_pool.free_count(), total_slots - 3);
    assert_eq!(
        e1.ranks[0].state_pool.ple_ngram_context[0],
        [111, 222],
        "active slot 0 PLE context must not be clobbered by !is_hybrid_gdn L2 hydration"
    );
    assert_eq!(
        e1.ranks[0]
            .state_pool
            .export_ple_conv_slot_to_host_bytes(0, &e1.ranks[0].stream)
            .unwrap(),
        conv_active,
        "active slot 0 PLE conv state must not be clobbered"
    );
    let node_a = rad.match_prefix(&seq_a).last_node;
    let node_b = rad.match_prefix(&seq_b).last_node;
    let (_, _, slot_a) = rad.node_kv_pages_and_mamba(node_a).unwrap();
    let (_, _, slot_b) = rad.node_kv_pages_and_mamba(node_b).unwrap();
    assert_ne!(slot_a, 0);
    assert_ne!(slot_b, 0);
    assert_ne!(slot_a, slot_b);
    assert_eq!(
        e1.ranks[0].state_pool.ple_ngram_context[slot_a as usize],
        [42, 99]
    );
    assert_eq!(
        e1.ranks[0].state_pool.ple_ngram_context[slot_b as usize],
        [77, 88]
    );
    assert_eq!(
        e1.ranks[0]
            .state_pool
            .export_ple_conv_slot_to_host_bytes(slot_a, &e1.ranks[0].stream)
            .unwrap(),
        conv_a
    );
    assert_eq!(
        e1.ranks[0]
            .state_pool
            .export_ple_conv_slot_to_host_bytes(slot_b, &e1.ranks[0].stream)
            .unwrap(),
        conv_b
    );

    // Evict all hydrated radix entries and release their state slots back to `e1`.
    let ev = rad.evict_full_pages(64);
    e1.free_kv_pages(&ev.freed_kv_pages);
    e1.free_state_slots(&ev.freed_mamba_slots);
    assert_eq!(e1.ranks[0].state_pool.free_count(), total_slots - 1);

    // Allocate all remaining free slots: NONE of them may alias `active_slot == 0`!
    let mut seen = std::collections::HashSet::new();
    seen.insert(active_slot);
    while let Some(s) = e1.ranks[0].state_pool.alloc_slot() {
        assert!(
            seen.insert(s),
            "duplicate state_pool slot {s} after !is_hybrid_gdn L2 eviction!"
        );
    }
    assert_eq!(seen.len(), total_slots);
}

#[test]
fn dflash2_projected_ctx_plain_recovery_and_batched_draft_parity() {
    let target_model = "nchapman/Qwen3.5-4B-W4A16-HGPTQ";
    let draft_model = "nchapman/Qwen3.5-4B-DFlash2";
    if common::require_or_skip_cached_model(
        target_model,
        "dflash2_projected_ctx_plain_recovery_and_batched_draft_parity",
    )
    .is_none()
        || common::require_or_skip_cached_model(
            draft_model,
            "dflash2_projected_ctx_plain_recovery_and_batched_draft_parity",
        )
        .is_none()
    {
        return;
    }
    let _gpu_guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let worker = EngineWorker::from_loader(
        move || {
            Qwen3_5Engine::load_for_serving_with_speculative(
                target_model,
                1,
                KvFormatPolicy::Tq4,
                Some(2048),
                true,
                4,
                Some(&SpeculativeConfig {
                    kind: SpeculativeKind::DFlash2,
                    draft_model: Some(draft_model.into()),
                    mtp_k: 0,
                }),
            )
            .map_err(|e| e.to_string())
        },
        SchedulerConfig {
            max_num_seqs: 4,
            ..SchedulerConfig::default()
        },
    )
    .unwrap();

    let tok = worker.tokenizer.as_ref().unwrap().clone();
    let encode = |text: &str| -> Vec<i32> {
        tok.encode(text, false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&id| id as i32)
            .collect()
    };

    let shared_prefix = "Systems engineering notes: lock-free ring buffers use acquire-release atomic indices, cache-line padding avoids false sharing across producer and consumer threads, and power-of-two capacities replace modulo with bitmasks. ".repeat(6);
    let t1_prompt = encode(&format!(
        "{shared_prefix}Turn 1: Write a concise Rust RingBuffer struct."
    ));
    let t2_suffix = encode(" Turn 2: Add push with Acquire/Release ordering.");
    let t3_suffix = encode(" Turn 3: Add pop and explain empty/full checks.");
    let t4a_suffix = encode(" Turn 4A: Write a unit test for wrap-around.");
    let t4b_suffix = encode(" Turn 4B: Summarize memory ordering guarantees.");

    let mut greedy = params();
    greedy.max_tokens = 32;
    let mut plain_logprobs = greedy.clone();
    plain_logprobs.logprobs = true;
    plain_logprobs.top_logprobs = 2;

    let (t3_prompt, out3, t4a_prompt, out4a, t4b_prompt, out4b) = rt.block_on(async {
        // Turn 1: initial DFlash2 speculative turn.
        let r0 = worker.stats.telemetry.snapshot().speculation.rounds;
        let mut st1 = worker.submit(t1_prompt.clone(), greedy.clone(), None).unwrap();
        let (out1, _) = tokio::time::timeout(Duration::from_secs(120), collect_with_cached(&mut st1, 32))
            .await
            .unwrap()
            .unwrap();
        assert!(!out1.is_empty());
        let r1 = worker.stats.telemetry.snapshot().speculation.rounds;
        assert!(r1 > r0, "Turn 1 should execute DFlash2 speculative rounds (r0={r0}, r1={r1})");

        // Turn 2: two concurrent `logprobs=true` requests force genuine `step_plain`
        // (batched speculation rejects `logprobs`, unlike a lone single-slot request),
        // recording projected `ctx` into `plain_memos` with zero speculative rounds.
        let mut t2_prompt = t1_prompt.clone();
        t2_prompt.extend_from_slice(&out1);
        t2_prompt.extend_from_slice(&t2_suffix);
        let mut t2_peer = t1_prompt.clone();
        t2_peer.extend_from_slice(&encode(" Peer plain request with logprobs."));
        let mut st2 = worker
            .submit(t2_prompt.clone(), plain_logprobs.clone(), None)
            .unwrap();
        let mut st2_peer = worker
            .submit(t2_peer, plain_logprobs.clone(), None)
            .unwrap();
        let (res2, res2_peer) = tokio::time::timeout(Duration::from_secs(120), async {
            tokio::join!(
                collect_with_cached(&mut st2, 32),
                collect_with_cached(&mut st2_peer, 32),
            )
        })
        .await
        .unwrap();
        let (out2, cached2) = res2.unwrap();
        let (out2_peer, _) = res2_peer.unwrap();
        assert!(!out2.is_empty() && !out2_peer.is_empty());
        assert!(cached2 > 0, "Turn 2 should reuse warm radix prefix from Turn 1");
        let r2 = worker.stats.telemetry.snapshot().speculation.rounds;
        assert_eq!(
            r2, r1,
            "Turn 2 concurrent logprobs should run in step_plain without speculative rounds (r1={r1}, r2={r2})"
        );

        // Turn 3: greedy continuation MUST recover DFlash2 from projected `ctx` `plain_memos`
        // while preserving the warm radix prefix from Turn 2.
        let mut t3_prompt = t2_prompt.clone();
        t3_prompt.extend_from_slice(&out2);
        t3_prompt.extend_from_slice(&t3_suffix);
        let mut st3 = worker.submit(t3_prompt.clone(), greedy.clone(), None).unwrap();
        let (out3, cached3) =
            tokio::time::timeout(Duration::from_secs(120), collect_with_cached(&mut st3, 32))
                .await
                .unwrap()
                .unwrap();
        assert!(!out3.is_empty());
        assert!(
            cached3 >= t2_prompt.len().saturating_sub(64),
            "Turn 3 should hit warm radix prefix after step_plain recovery (cached={cached3}, t2_len={})",
            t2_prompt.len()
        );
        let r3 = worker.stats.telemetry.snapshot().speculation.rounds;
        assert!(
            r3 > r2,
            "Turn 3 should recover DFlash2 speculation from projected ctx plain_memos (r2={r2}, r3={r3})"
        );

        // Turn 4: concurrent B=2 greedy continuations exercise batched DFlash2 drafting
        // (`try_ingest_and_propose_dflash_batch` + `batched_select_greedy_graphs`).
        let mut t4a_prompt = t3_prompt.clone();
        t4a_prompt.extend_from_slice(&out3);
        t4a_prompt.extend_from_slice(&t4a_suffix);
        let mut t4b_prompt = t3_prompt.clone();
        t4b_prompt.extend_from_slice(&out3);
        t4b_prompt.extend_from_slice(&t4b_suffix);

        let mut st4a = worker.submit(t4a_prompt.clone(), greedy.clone(), None).unwrap();
        let mut st4b = worker.submit(t4b_prompt.clone(), greedy.clone(), None).unwrap();
        let (res4a, res4b) = tokio::time::timeout(
            Duration::from_secs(120),
            async {
                tokio::join!(
                    collect_with_cached(&mut st4a, 32),
                    collect_with_cached(&mut st4b, 32),
                )
            },
        )
        .await
        .unwrap();
        let (out4a, cached4a) = res4a.unwrap();
        let (out4b, cached4b) = res4b.unwrap();
        assert!(!out4a.is_empty());
        assert!(!out4b.is_empty());
        assert!(cached4a > 0 && cached4b > 0, "B=2 DFlash2 turns should reuse warm radix prefix");
        let r4 = worker.stats.telemetry.snapshot().speculation.rounds;
        assert!(
            r4 > r3,
            "Turn 4 B=2 continuations should advance DFlash2 speculative rounds (r3={r3}, r4={r4})"
        );

        (t3_prompt, out3, t4a_prompt, out4a, t4b_prompt, out4b)
    });

    drop(worker);

    // Verify lossless greedy parity against a non-speculative baseline for Turn 3 (recovered
    // from `plain_memos`) and Turn 4A/4B (batched DFlash2 drafting).
    let plain_worker = EngineWorker::from_loader(
        move || {
            Qwen3_5Engine::load_for_serving_with_speculative(
                target_model,
                1,
                KvFormatPolicy::Tq4,
                Some(2048),
                true,
                4,
                None,
            )
            .map_err(|e| e.to_string())
        },
        SchedulerConfig {
            max_num_seqs: 4,
            ..SchedulerConfig::default()
        },
    )
    .unwrap();
    rt.block_on(async {
        for (label, prompt, expected) in [
            ("Turn 3 recovered", t3_prompt, out3),
            ("Turn 4A batched", t4a_prompt, out4a),
            ("Turn 4B batched", t4b_prompt, out4b),
        ] {
            let mut st = plain_worker.submit(prompt, greedy.clone(), None).unwrap();
            let (plain_out, _) =
                tokio::time::timeout(Duration::from_secs(120), collect_with_cached(&mut st, 32))
                    .await
                    .unwrap()
                    .unwrap();
            assert_eq!(
                expected, plain_out,
                "{label} DFlash2 tokens must match non-speculative greedy baseline"
            );
        }
    });
    drop(plain_worker);
}
