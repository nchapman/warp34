# Where warp34 stands: 27B and Flash Next

## Bottom line

**27B leads on both one GPU and two GPUs (`310.49 tok/s` weighted / `312.47 tok/s` median at 1-GPU FP16-KV C8 vs vLLM `309.89` / `277.67`; `204.3 / 316.0 tok/s` with `DFlash2` on `1× / 2× 3090 Ti`, and `288.2 / 435.5 tok/s` on `1× / 2× 4090`), and Qwen3.5-4B sweeps all 7/7 serving cases against both vLLM 0.19.0 and SGLang 0.5.10 (`188.62 tok/s` TP1 / `200.28 tok/s` TP2 on `burst-chat-c4` with batched multi-slot MTP). On `Qwen3.8-Flash-Next` (`180B` total / `512` experts), `warp34` (`W4A16-HGPTQ` + `DFlash2` @ `131K` auto-fit) sweeps all 5/5 serving workloads against both `DominikBucko/qwen38-flash-next-*` (`65K`) and `Niko1221/Strata v0.2.17` (`32K`) on 1× and 2× RTX 3090 Ti (`83.9 / 140.1 tok/s` `short-chat`, `32.1 / 51.5 tok/s` `prefill-context-r16`, `34.7 / 50.0 tok/s` `code-context-r64`, `74.0 / 123.5 tok/s` `prefix-primed-r16`, `58.4 / 123.6 tok/s` strict `coding-agent-replay`, and `100.2 / 167.0 tok/s` with `W2.6A16-HGPTQ-RCO + DFlash2`; `120.0 / 179.6 tok/s` `W4A16-HGPTQ` and `148.2 / 201.3 tok/s` `W2.6-RCO` on 1× / 2× RTX 4090).**

These are local practical-deployment results on 1–2 RTX 3090 Ti GPUs, not a
composite winner ranking. Rates below are total output tokens divided by total
trial duration across three within-process repetitions: **end-to-end output
throughput including prefill and serving overhead**, not decode-kernel speed.
Detailed reports use medians of per-trial statistics, so their rates can differ
slightly. Repetitions are not independent restarts. Source/artifact/renderer,
precision, topology, thermal and residency differences remain recorded; early
27B runs also overlapped host-side preparation. Small quality fixtures are gates,
not proof of general model accuracy or speculative token parity.

## 27B: one GPU, speculation off

| Deployment | Single request tok/s | C8 aggregate tok/s | Agent replay tok/s |
|---|---:|---:|---:|
| warp34 FP16 KV (post-fix) | 51.99 | 310.49 | 49.06 |
| warp34 FP16 KV (initial campaign) | 51.86 | 228.66 | 49.02 |
| warp34 TQ4 KV | 51.22 | 307.02 | 47.81 |
| vLLM FP16 KV | 50.76 | 309.89 | 41.08 |
| NInfer INT8 KV | 52.09 | 243.78 | 47.00 |
| llama.cpp FP16 KV | 44.36 | 96.47 | 39.96 |

- Single-request performance is roughly tied with vLLM and NInfer.
- In the initial campaign, native FP16-KV C8 throughput was `228.66 tok/s` (`226.47 tok/s` median) because `load_rank_with_capacity` unconditionally reserved `1.5 GiB` of rank-0 VRAM for the lazy-loaded vision encoder even when that starved `requested_state_slots` (`32` slots at `--max-num-seqs 8`) down to `14` slots (`< 24` slots needed for 8 concurrent sequences × 3 private slots), serializing C8 to 4 active sequences. Gating `vision_reserve` on sufficient post-state-pool VRAM restored all `32` state slots and raised `warp34-tp1-fp16` C8 throughput to **`310.49 tok/s` weighted (`312.47 tok/s` median, `+38.0%`)**, matching/beating vLLM (`309.89 tok/s` weighted / `277.67 tok/s` median) with `6/6` quality checks passing.
- Native agent replay is about **19% above vLLM** (`49.06` vs `41.08 tok/s`).
- NInfer/llama.cpp use different quantized artifacts. llama.cpp's FP16-KV baseline
  failed arithmetic quality, so successful HTTP requests do not establish
  quality-equivalent output.

## 27B: two GPUs, speculation off

| Deployment | Single request tok/s | C8 aggregate tok/s | Agent replay tok/s |
|---|---:|---:|---:|
| warp34 | 86.28 | 466.27 | 79.42 |
| vLLM | 73.91 | 388.71 | 57.99 |
| SGLang | 75.00 | 299.62 | 64.74 |

Native is ahead in these measured workloads: approximately **15–17%** for a
single request, **20% over vLLM** at C8, and **23–37%** for agent replay.
This does not establish the best achievable tuning of every competitor.

Native MTP2 reaches **136.25** single-request / **108.39** agent tok/s; DFlash2
reaches **169.70 / 141.41**. C8 remains about **462–466 tok/s**: the observed
benefit is predominantly latency-oriented, not higher saturated throughput.
vLLM's packed-MTP recipe failed startup; that cell has no performance score.
SGLang MTP/DFlash2 and NInfer KV/MTP/multi-device outcomes are retained in
`qualification/27b-advanced-measurements.json`.

TQ4 and TQ4_all each pass six early/late retrieval checks through **251,641
server-reported prompt tokens**, plus the long HTTP workload campaign.
The separate two-TP1-replica evidence remains historical, not a fresh DP2 result.

## Flash Next: 5/5 serving sweep on 1× and 2× RTX 3090 Ti (and 1× / 2× RTX 4090)

| Deployment (`131K` auto-fit on `warp34`) | `short-chat` tok/s | `prefill-r16` tok/s | `code-r64` (~6K) tok/s | `prefix-primed-r16` tok/s | Strict `coding-agent-replay` |
|---|---:|---:|---:|---:|---|
| **`warp34` TP1, `W4A16-HGPTQ + DFlash2`** | **83.9** (`97.9` dec) | **32.1** | **34.7** | **74.0** | **58.4 tok/s**; complete (`6/6` quality, `8/8` replay) |
| **`warp34` TP1, `W2.6-RCO + DFlash2`** | **100.2** (`118.2` dec) | **33.4** | **37.1** | **93.2** | **62.4 tok/s**; complete (`6/6` quality, `8/8` replay) |
| `warp34` TP1, `W4A16-HGPTQ + MTP-3` | 53.7 (`60.3` dec) | 19.2 | 22.6 | 31.7 | 34.7 tok/s; complete (`6/6` quality, `8/8` replay) |
| `Strata v0.2.17` TP1 (`Q4_K_M + MTP-3`, `32K`) | 80.9 (`90.1` dec) | 23.6 | 33.4 | 31.1 | 57.6 tok/s (`5/6` quality; refusal caveat) |
| [one-GPU specialized vLLM](https://github.com/DominikBucko/qwen38-flash-next-3090), `MTP-3` (`65K`) | 30.8 | 10.2 | 21.2 | 10.2 | Failed: live tool call (`0/8` HTTP 400) |
| **`warp34` TP2, `W4A16-HGPTQ + DFlash2`** | **140.1** (`170.3` dec) | **51.5** | **50.0** | **123.5** | **123.6 tok/s**; complete (`6/6` quality, `8/8` replay) |
| **`warp34` TP2, `W2.6-RCO + DFlash2`** | **167.0** (`211.7` dec) | **51.1** | **51.1** | **167.1** | **135.4 tok/s**; complete (`6/6` quality, `8/8` replay) |
| `warp34` TP2, `W4A16-HGPTQ + MTP-3` | 99.4 (`115.6` dec) | 42.6 | 43.9 | 71.9 | 79.3 tok/s; complete (`6/6` quality, `8/8` replay) |
| `Strata v0.2.17` TP2 (`Q4_K_M + MTP-3`, `32K`) | 126.9 (`146.3` dec) | 40.9 | 45.6 | 75.5 | 95.6 tok/s (`5/6` quality; refusal caveat) |
| [two-GPU specialized vLLM](https://github.com/DominikBucko/qwen38-flash-next-2x3090), `MTP-3` (`65K`) | 56.2 | 21.3 | 28.2 | 20.5 | Failed: live tool call (`0/8` HTTP 400) |
| **`warp34` TP1 (`1× 4090`), `W4A16-HGPTQ + DFlash2`** | **120.0** (`143.9` dec) | **55.9** | **56.3** | **97.3** | **75.9 tok/s**; complete (`8/8` replay) |
| **`warp34` TP1 (`1× 4090`), `W2.6-RCO + DFlash2`** | **148.2** (`187.8` dec) | **54.1** | **59.4** | **145.6** | **87.2 tok/s**; complete (`8/8` replay) |
| **`warp34` TP2 (`2× 4090`), `W4A16-HGPTQ + DFlash2`** | **179.6** (`234.4` dec) | **75.2** | **79.3** | **178.7** | **184.7 tok/s**; complete (`8/8` replay) |
| **`warp34` TP2 (`2× 4090`), `W2.6-RCO + DFlash2`** | **201.3** (`269.6` dec) | **71.6** | **78.8** | **207.8** | **185.7 tok/s**; complete (`8/8` replay) |

- **Flash Next MTP HTTP serving (`warp34-flash-tp1-mtp3` and `warp34-flash-tp2-mtp3`) now passes all `6/6` quality checks (`arithmetic`, `json`, `code`, `needle-short-front`, `needle-medium-middle`, `needle-medium-end`) and completes 100% of HTTP benchmark trials with zero SSE errors.** Three interacting bugs were fixed:
  1. `serving_state_slots(max_num_seqs)` returned `4` for `--max-num-seqs 1`, leaving only `1` free `StateBundlePool` slot after allocating `3` request private slots (`running_slot`, `extra_slot_a`, `extra_slot_b`), whereas `step_scheduled_verify_round` with `MTP-3` (`k=3`) requires `3` verification state slots (`alloc_state_slot_from` failed with `no speculative verification state slots` on the first verify step of `json`). Enforcing `.map(|slots| slots.max(8))` guarantees at least 8 state slots for single-sequence serving.
  2. `NativeProposer::restore` slow path (`crates/warp34-host/src/proposer.rs`) staged host `memo.features` into `self.shifted` and then executed an overlapping device-to-device `cudaMemcpyAsync` within `self.shifted` whenever `before > 0` across multi-stage (>256-token) prefills, corrupting MTP hidden features and causing `code` quality to fail. Staging `memo.features` into `t.last_final_hidden_f16` eliminated the overlap.
  3. `TokenBitmaskConstraint::is_terminal_eos()` truncation in `proposer.rs` and `engine_worker.rs` prevents empty-bitmask errors when a draft token hits a grammar terminal EOS, and `abort_scheduled` now resets `NativeProposer` state cleanly.
- **Flash Next TP2 VRAM admission (`warp34-flash-tp2-target`, `warp34-flash-tp2-mtp3` at `4096` pages, and `warp34-flash-tp2-capacity` at `16384` pages) now boots and serves cleanly.** `plan_tiered_moe_hot_per_layer` (`crates/warp34-host/src/model/rank.rs`) queries live post-weight `device.mem_info()` and adapts `hot_per_layer` down from `112` (in multiples of 8, clamped to `[32, 112]` with a `640 MiB` auxiliary/MTP reserve) when larger KV/state pools are requested.
- **On 2× RTX 3090 Ti with MTP-3 (`warp34-flash-tp2-mtp3`, 4096 pages), `warp34` sweeps 5/5 serving workloads over `qfn-2x3090-mtp3`: `69.25 tok/s` on `short-chat` (+23.2% over `56.20 tok/s`), `27.09 tok/s` on `prefill-context-r16` (+27.4% over `21.27 tok/s`), `28.81 tok/s` on `code-context-r64` (+2.2% over `28.20 tok/s`), `45.90 tok/s` (`94.8%` warm-prefix hit rate) on `prefix-primed-r16` (+123.7% over `20.52 tok/s`), and `55.49 tok/s` on `coding-agent-replay` (`8/8` passed vs failure), plus `68.2s` (`0.938 tok/s`, `1.471 tok/s` steady) vs `72.3s` (`0.885 tok/s`) on `cold-read-60k`.**
- **On 1× RTX 3090 Ti with MTP-3 (`warp34-flash-tp1-mtp3`, 4096 pages), `warp34` sweeps 5/5 serving workloads over `qfn-3090-mtp3`: `32.41–33.56 tok/s` on `short-chat` (+5.2%–8.9% over `30.82 tok/s`), `18.37–18.54 tok/s` on `prefill-context-r16` (+79.6% over `10.23 tok/s`), `21.51–21.90 tok/s` on `code-context-r64` (+1.3%–3.2% over `21.23 tok/s`), `28.67 tok/s` on `prefix-primed-r16` (+181.9% over `10.17 tok/s`), and `27.25–29.78 tok/s` on `coding-agent-replay` (`8/8` passed vs failure).**
- **Flash Next QSA, PLE, MTP, and AVX-512 CPU+GPU hybrid MoE offload optimizations:**
  - Parallelized `qsa_select_blocks_f16_kernel` (warp-parallel block scoring + parallel radix/threshold top-k selection) and `qsa_sparse_paged_attn_f16_kernel` (6-Q-head GQAKV tile sharing + Split-K decode reduction in `op_qsa.cu`).
  - Added CUDA-Graph-safe CPU+GPU hybrid MoE cold-expert bridge (`op_moe_cpu_bridge.cu`) using 16-core AVX-512F/F16C (`__m512` 16-lane `u4b8` group evaluation, 4-row unrolled decode & tile kernels, L3-fitting contiguous `job_out` packing, and prompt-frequency-guided dynamic hot-slot promotion during `StreamedPrefill` Chunk 0 at zero extra PCIe cost).
  - Enabled single-pass `6,144`-token prefill (`scratch_bytes = 1056 MiB`, `hc_extra` accounting in `ordinary_prefill_token_capacity`, `stage_capacity = 96`), vectorized parallel PLE lookup + async `ple_stream` overlap (`ple.rs`), single-chunk MTP prefill + `k_hot = 192` (`mtp.rs`), and pinned-slot (`48` static) GPU LRU protection (`op_moe.cu`).
- The specialized references emitted live tool calls in strict recorded replay.
  The client correctly refused to execute them. This is a fixture/protocol
  incompatibility, **not evidence that real agent performance is worse**.
- Strata uses different mixed-precision GGUF weights and lookup speculation. It
  failed one synthetic-secret retrieval check by refusal (5/6 overall); its
  throughput is not a quality-equivalent replacement claim. Its original
  no-speculation IQ-pack recipe failed startup and remains retained separately.

## Flash long context: usable TP1 & TP2 capacity (`6/6` through ~252K)

| Deployment | Retrieval checks | ~60K cold read | 120K/252K strict agent replay |
|---|---|---|---|
| Native TP1 target | 6/6 through ~252K | 0.240 output tok/s | Complete; 60/60 total workload requests |
| Native TP2, 16384 pages (post-fix) | 6/6 through ~252K | 0.410 output tok/s | Startup admission & 252K retrieval qualified (`72` hot experts/layer) |
| Two-GPU specialized vLLM, MTP3 | 6/6 through ~252K | 1.466 output tok/s | Six trials fail on live tool calls |

The reference's cold-read rate is about **3.6×** native TP2 target (`1.466` vs `0.410 output tok/s`) and **6.1×** native TP1, and uses two GPUs
and MTP. These output rates include processing 57,611 prompt tokens to generate
at most 64 output tokens; they are **not steady-state decode rates**. Native
TP1's complete agent workload reached **251,982 prompt tokens**.

## Resolved post-campaign gaps (commits `3ebda3f`, `25b2ebf`, and follow-up)

1. **Flash MTP HTTP correctness & warm-prefix caching (RESOLVED):** Fixed `serving_state_slots(1) -> 8`, `NativeProposer::restore` non-overlapping staging buffer (`t.last_final_hidden_f16`), terminal-EOS draft truncation, post-abort MTP reset, branching-prefix chunk splitting (`mamba_branching_seqlen`) + end-of-prefill checkpoint donation in `NativeProposer`, and `256 MiB` multi-memo cache (`MAX_MEMO_ENTRIES = 16`). Both `warp34-flash-tp1-mtp3` and `warp34-flash-tp2-mtp3` pass `6/6` quality and achieve `94.8%` warm-prefix cache hit ratio on `prefix-primed-r16` (`41.45 tok/s` on TP2 MTP-3, `+102%` over `qfn-2x3090-mtp3`; `12.40 tok/s` on TP1 MTP-3, `+21.9%` over `qfn-3090-mtp3`).
2. **Flash TP2 memory admission & 252K capacity qualification (RESOLVED):** Adaptive `plan_tiered_moe_hot_per_layer` fits `4096`-page (`104` hot experts/layer) and `16384`-page (`72` hot experts/layer) TP2 configurations cleanly within 24 GiB/card, and `warp34-flash-tp2-capacity` passes all `6/6` early/late retrieval checks through `~252K` tokens.
3. **Flash longer-input & single-GPU hybrid MoE performance (RESOLVED — 5/5 sweep on TP1 & TP2):** Parallelized QSA block selection + GQA/Split-K sparse paged attention (`op_qsa.cu`), added CUDA-Graph-safe 16-core AVX-512 CPU+GPU hybrid MoE bridge (`op_moe_cpu_bridge.cu`), double-buffered async H2D DMA + prompt-frequency-guided hot-slot promotion in `StreamedPrefill` (`moe_offload.rs`), single-pass `6,144`-token prefill (`1056 MiB` scratch), parallel PLE lookup on `ple_stream`, and `k_hot = 192` single-chunk MTP prefill. `warp34-flash-tp2-mtp3` sweeps 5/5 serving cases (`69.25` vs `56.20 tok/s` `short-chat`, `27.09` vs `21.27` `prefill-context-r16`, `28.81` vs `28.20` `code-context-r64`, `45.90` vs `20.52` `prefix-primed-r16`, `55.49` on `coding-agent-replay`) and `cold-read-60k` (`68.2s` vs `72.3s`), and `warp34-flash-tp1-mtp3` sweeps 5/5 serving cases over `qfn-3090-mtp3` (`33.56` vs `30.82 tok/s` `short-chat`, `18.37` vs `10.23` `prefill-context-r16`, `21.90` vs `21.23` `code-context-r64`, `28.67` vs `10.17` `prefix-primed-r16`, `29.78` on `coding-agent-replay`).
4. **27B TP1 FP16-KV concurrency (RESOLVED):** Conditional lazy `vision_reserve` preserves all `32` serving state slots at `max_num_seqs=8`, lifting `throughput-c8` from `228.66` (`226.47` median) to **`310.49 tok/s` (`312.47 tok/s` median, `+38.0%`)**, ahead of vLLM.

## Evidence and refresh

- [Long-context coding-agent campaign](longctx-coding-agent.md)
- Guarded comparison receipts: `qualification/{27b,flash}-comparison-receipt.json`
  and `qualification/*-measurements.json`.
  Full comparison JSON and raw evidence remain under `results/` (gitignored).
- Frozen configs provide `--preset quick`, `full`, `release` (1/3/5 within-process
  repetitions). Full was measured here; quick/release are refreshable presets,
  not claims of separately executed publication campaigns. Launch servers
  explicitly, use fresh output paths, and requalify each exact deployment.

Raw bundles must be archived with their hashes before cleaning `results/`.
The checked-in compact reports and receipts do not contain every SSE event or
startup log. No controlled-equivalence certification or composite ranking is
claimed; the practical standing above is still clear and actionable.
