# warp34 — Comprehensive Front-to-Back Feature Inventory

`warp34` is a clean-room Rust + CUDA inference engine engineered for **Qwen 3.5 / 3.6 / 3.8 hybrid models** (GDN linear attention + full attention + MoE + Vision + MTP/DFlash2 + Paged TQ4/FP8 KV) targeting **1–4× NVIDIA RTX 30-series (`sm_86`) and RTX 40-series (`sm_89`) GPUs** (primary targets: RTX 3090 / 3090 Ti, RTX 4090, RTX 3060 Ti, and RTX 4060 Ti).

This document inventories implemented capabilities, APIs, execution contracts, kernel specializations, and tooling. Verified benchmark and hardware qualification results live in [`STATE.md`](STATE.md).

The 22 section numbers are stable inventory references. Readiness IDs group their bullets into independently assessable contracts; implementation details inherit the assessment of their owning capability, not a separate production guarantee.

---

## 1. Command-Line Interface (CLI) & Operational Tooling
*Location: `crates/warp34-host/src/cli/`, `crates/warp34-host/src/main.rs`*

* **CLI Dispatch & Subcommands:**
  * `generate`: Standalone single/multimodal generation supporting plain, MTP, and DFlash2 execution across TP=1/2/4.
  * `bench`: Multi-stage benchmarking framework (`--stage decode | radix | scheduler | vision | all`) with configurable prompt/generation lengths, batch sizes, custom or realistic coding prompts (`--prompt`, `--prompt-file`, `--realistic-prompt` / `--realistic`), and JSON export (`--out <path>`, alias `--out-json <path>`).
  * `greedy-diff`: Automated token-for-token validation harness against reference golden outputs (`--golden <dir|json>`) with `--verify-lossless` (speculative parity) and `--verify-radix-warm` (prefix cache parity) verification.
  * `dump-diff`: Layer-by-layer tensor inspection harness dumping per-layer inputs/outputs across norms, attention, GDN, and MLP against `vLLM` / `SGLang` Safetensors fixtures (`--compare-tp2`).
  * `profile-diff`: Side-by-side SQLite/JSON performance profiler comparing round latencies (`ms/round`) and launch counts across 7 kernel categories against `vLLM 0.34.0` and `SGLang v0.5.23`.
  * `serve`: HTTP server implementing OpenAI Chat, OpenAI Responses, Anthropic Messages, and TypeSafe Decision model APIs.
  * `chat`: Interactive multi-turn terminal REPL (`warp34 chat <model>`) and single-turn runner (`--prompt`) formatting ChatML prompts, streaming reasoning and tool blocks via `Qwen3Parser`, and reporting generation statistics (prompt/cached/generated tokens, TTFT, decode tok/s) to stderr.
* **CLI Flags & Argument Validation:**
  * **Tensor Parallelism:** `--tp 1 | 2 | 4 | auto` (aliases: `--tp-size`, `--tensor-parallel-size`).
  * **KV Cache Formats:** `--kv-format auto | tq4 | tq4_all | q8_0 | fp8 | fp16 | bf16` (default: `auto`, selecting `tq4` and auto-stepping down to `tq4_all` in `serve` when needed for full context; `fp8` requires `sm_89+`).
  * **Speculative Decoding & Context:** `--speculative auto | mtp | dflash2 | off` (default: `auto` in `serve`/`chat`/`generate`, `off` in `bench`/oracle), `--draft-model <id|path>`, `--mtp-k 1..3` (default 3), `-c` / `--ctx-size <auto|tokens>` or `--kv-pages <n>` (defaults to `auto` up to `256k` / `16,384` pages in `serve`, `16k` / `1,024` pages in `chat`, or `4k` / `256` pages in `generate`).
  * **Serving, Reasoning & LoRA Options:** `--api-key <val>` or `--api-key-env <NAME>`, `--served-model-name <alias>` (`--alias`, `-a`), `--no-webui`, `--lora <name=path_or_repo>` (`--lora-modules`, `--active-lora`), `--max-num-seqs <1..16>` (default 16), `--max-model-len <tokens>` (supports `k`/`M` suffixes), `--reasoning-effort <level>`, `--reasoning-budget <auto|off|n>`, `--reasoning-budget-message <text>`, `--no-reasoning-effort-budget`.
  * **Model Hub & Offline Options:** `--revision <ref>`, `--draft-revision <ref>`, `--offline` (reads strictly from `HF_HUB_CACHE`).
  * **Multimodal Options:** `--image <path|url>`, `--allow-remote-images` (SSRF-protected HTTP/HTTPS retrieval).

---

## 2. Multi-API HTTP Gateway & Serving Architecture
*Location: `crates/warp34-host/src/server/`, `crates/warp34-host/src/server/http.rs`*

* **Asynchronous Axum Server Architecture:**
  * Complete decoupling of HTTP parsing, JSON serialization, and vision preprocessing onto an asynchronous CPU worker pool (`preprocessing`), protecting the bounded GPU `EngineWorker` from event-loop stalls.
  * Admission control semaphore (`admission`) limiting concurrent in-flight requests and guarding against memory exhaustion under burst traffic.
  * Bounded payload size (`BODY_LIMIT = 32 MiB`) and body timeout (`BODY_TIMEOUT = 10s`) returning HTTP 408/413 appropriately.
* **OpenAI Chat Completions API (`POST /v1/chat/completions`):**
  * Full ChatML compatibility with non-streaming JSON responses and streaming Server-Sent Events (`SSE`, `text/event-stream`).
  * Per-token streaming chunks with monotonic ping keep-alives and `usage` emission (emitted as a separate chunk with `choices: [] + usage` when `stream_options.include_usage` is set).
  * `reasoning_effort` mapping (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`) emitting real-time streaming `reasoning_content` deltas.
  * Parallel tool-calling support with incremental function call argument streaming.
  * Dedicated input token counting endpoint: `POST /v1/chat/completions/input_tokens` (and unversioned alias `/chat/completions/input_tokens`).
  * Legacy completions fallback: `POST /v1/completions` with parallel token/logprob arrays.
* **OpenAI Responses API (`POST /v1/responses`):**
  * Implementation of the Responses routes and behaviors documented in [`GENERATIVE_APIS.md`](GENERATIVE_APIS.md), not a blanket claim of upstream specification completeness (`crates/warp34-host/src/responses_api/`).
  * In-memory bounded session store (`ResponseStore`, default capacity 256 responses, 64 per credential scope) partitioned by credential-derived scope and supporting request chaining via `previous_response_id`.
  * Cryptographically unguessable 128-bit response IDs (`resp_<32hex>`) generated via `/dev/urandom`.
  * Stateful lifecycle routes: `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}`, `POST /v1/responses/{id}/cancel`, and `POST /v1/responses/input_tokens`.
  * Specialized 0-indexed monotonic `sequence_number` SSE event streaming (`response.created`, `response.in_progress`, `response.output_item.added`, `response.output_text.delta`, `response.completed`).
* **Anthropic Messages API (`POST /v1/messages`):**
  * Direct ingestion of Anthropic Messages wire format (`crates/warp34-host/src/anthropic_api.rs`) translating system blocks, thinking blocks, user images, and `tool_use`/`tool_result` content.
  * Transparent Claude Code `cch=` billing-header stripping to maximize `MambaRadixCache` prefix-sharing hits.
  * Supports thinking budgets (`budget_tokens >= 1024`), adaptive thinking, and redacted thinking compatibility.
  * Streaming Anthropic SSE event format (`message_start`, `content_block_start`, `content_block_delta`, `message_delta`, `message_stop`) with typed tool use IDs (`toolu_*`).
  * Dedicated token count endpoint: `POST /v1/messages/count_tokens`.
* **Health, Discovery & Prometheus Telemetry:**
  * `GET /health`: Fast probe returning `{"status":"ok"}`.
  * `GET /v1/models`: OpenAI-compatible model catalog reporting active model ID, LoRA aliases, and `max_model_len`.
  * `POST /tokenize` & `POST /detokenize`: Direct tokenization and detokenization endpoints with admission limiting.
  * `GET /metrics`: Standard Prometheus metrics export (`text/plain; version=0.0.4`) exposing TTFT, per-request time-per-output-token, queue latencies, and cache utilization.
  * `GET /api/metrics`: Real-time JSON telemetry snapshot (`no-store`) powering operational dashboards.

---

## 3. Specialized Decision-Model Serving Engine
*Location: `crates/warp34-host/src/decision.rs`, `crates/warp34-host/src/decision/`*

* **TypeSafe Decision Model Endpoints:**
  * Exposes `POST /v1/systemone` (and unversioned `/systemone`) and `POST /v1/decisions` (and `/decisions`) for specialized zero-decode decision pipelines without modifying generative routes.
  * **`CodeReadout` (`GLiDE` — `fastino/GLiDE-*`, `nchapman/GLiDE-*-W4A16-HGPTQ`, and `GLiDE` Decision LoRA adapters):** Projects post-final-`GemmaRMSNorm` hidden states through `CodeReadoutHead` (`readout.safetensors`, up to 64 option codes, optional `lm_head` omitted), supporting `noncausal_full_attention`, `midloop` damped-Euler recurrence (`warp34_midloop_damped_euler_f16`), `true`-first `noul` ordering, and `Top1Top2Margin` confidence.
  * **`CodeReadout` (`d3` — `vllm-sr/d3-*`, `nchapman/d3-*-W4A16-HGPTQ`):** Projects post-final-`GemmaRMSNorm` hidden states at each question's `Answer:` token via `CodeReadoutHead`, supporting `header-v1` multi-question summary layout, `state_once` single-prefill multi-question taps, `prefix_bidir` bidirectional state attention, `temperature_by_questions`, `false`-first `noul` ordering, and `NormalizedEntropy` confidence.
  * **`CausalLm` (Default Qwen Generative Scorer):** Evaluates any standard Qwen 3.5 / 3.6 / 3.8 model via last-row `lm_head` candidate-token logits across single-token labels `A..X` (up to 24 options) with `"state"`-first JSON formatting and LCP shared-prefix batching across `MambaRadixCache`.
  * **Positional Bias Cancellation (`order_passes: 2`):** Evaluates both forward and reverse option permutations and averages probabilities in `f64`.

---

## 4. API Authentication & Web Dashboard
*Location: `crates/warp34-host/src/api_auth.rs`, `crates/warp34-host/src/dashboard.rs`*

* **Digest-Based API Credential Layer:**
  * Opt-in protection for all inference, tokenization, model catalog, and response routes (while leaving health, metrics, and dashboard public).
  * `ApiKey` holds exclusively a SHA-256 fixed-length digest (`[u8; 32]`) and deliberately omits `Debug` implementations to prevent log leakage.
  * Constant-time comparison (`subtle::ConstantTimeEq`) guards against timing side-channels.
  * Accepts both RFC 6750 `Authorization: Bearer <token>` and `x-api-key` headers.
* **Embedded Self-Contained Web Dashboard:**
  * Self-contained React dashboard and interactive API Playground compiled and embedded directly into the binary via `include_dir!`.
  * Strict Content Security Policy (`CSP: default-src 'none'; script-src 'self'; style-src 'self' ...`), anti-sniff headers, and path traversal guards (`.` and `..` rejection).
  * Serves `GET /`, `GET /playground`, `GET /playground/{*path}`, `GET /connect` (`index.html`), `GET /favicon.svg`, and immutable cached static assets (`/assets/*`).

---

## 5. Native Rust XGrammar Structured Outputs & Constraints
*Location: `crates/warp34-host/src/constraints.rs`, `crates/warp34-host/src/request_constraints.rs`*

* **High-Performance Grammar Compilation:**
  * Pinned integration with native `xgrammar-rs 0.2.8` and Hugging Face `tokenizers 0.22`, eliminating custom Python/Triton grammars and runtime fallbacks.
  * Shared 512 MiB grammar compiler cache reusing compiled schemas across concurrent requests.
  * Supports JSON Schema (`response_format` and `structured_outputs`), JSON Object mode, regular expressions, Lark grammars, and EBNF syntax (`::=` auto-detection).
  * Choice lists: Compiles arbitrary lists of strings into structural tag alternatives (`choice: [...]`).
  * Schema sanitization and validation: Rejects non-disjoint unions, unsupported `$ref` scopes, and ambiguous constraint specs before dispatching to the model.
* **GPU Token Bitmasking & Speculative Rollback (`TokenBitmaskConstraint`):**
  * Directly computes packed 32-bit bitmasks (`valid_mask: Arc<Vec<u32>>`) applied inside the GPU sampler.
  * Speculative rejection rollback: Tracks matcher token history with `try_rollback_tokens` capability to roll back grammar state on rejected speculative drafts.
  * Decoupled reasoning gate: Maintains `<think>...</think>` tokens outside grammar constraints, activating grammar enforcement only upon reasoning termination.
  * Request stop-token injection: Binds request-specific stop tokens into the grammar EOS set, masking premature EOS emission until grammar terminal states are reached.

---

## 6. Tool-Calling & Reasoning Parsing Subsystem
*Location: `crates/warp34-host/src/parser/`, `crates/warp34-host/src/tool_constraints.rs`, `crates/warp34-host/src/chatml.rs`*

* **Qwen3 XML Structural Tool System:**
  * Canonical Qwen XML envelope representation: `<tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>`.
  * Pinned `max_whitespace_cnt: 1` in native XGrammar rules prevents infinite empty-argument whitespace loops.
  * Independent per-parameter native schema compilation (`native_schema::compile`): Resolves local `$ref` targets using a bounded traversal budget and compiles grammars for argument typing.
  * Supports `tool_choice`: `auto`, `none` (disables tool generation and extraction), `required` (enforces at least one tool call), and named functions (`{type: "function", function: {name: ...}}`).
  * Non-strict schema compliance: Parameters accept raw strings, numbers, booleans, and JSON objects/arrays without requiring rigid schema strictness.
* **Streaming Parser & Reasoning State Machine (`Qwen3Parser`):**
  * Token-by-token state tracking across Reasoning (`<think>`), Content, and Tool Envelopes.
  * Emits incremental deltas (`reasoning_delta`, `content_delta`, and completed `tool_calls`).
  * Implicit `</think>` auto-transition: Automatically terminates reasoning mode when a `<tool_call>` tag appears inside the thinking block.
  * Duplicate tag absorption: Silently absorbs duplicate `</think>` tags emitted in content state.

---

## 7. Execution Engine & Continuous Batching Scheduler
*Location: `crates/warp34-host/src/scheduler.rs`, `crates/warp34-host/src/engine_worker/`*

* **Token-Budget Scheduler (`SchedulerV1`):**
  * Continuous batching with unified multi-request scheduling up to `max_num_batched_tokens` (default: 2,048) and `max_num_seqs` (default: 16).
  * Chunked prefill strictly aligned to the `MambaRadixCache` checkpoint boundary (`C = k * 64` tokens).
  * Mixed-batch scheduling: Bounds prefill chunk size to `mixed_chunk_prefill_size` (default: 256 tokens) whenever active decode/speculative verify sequences are present, protecting inter-token latency (p99 ITL).
  * Dynamic scratch-arena sizing: Automatically selects scratch allocations (`COMPACT_SCRATCH_ARENA_MAX_BYTES` 256 MiB, `EXTREME_LONGCTX_SCRATCH_ARENA_BYTES` 336 MiB, `LARGE_MODEL_TP1_SCRATCH_ARENA_BYTES` 384 MiB, `DEFAULT_SCRATCH_ARENA_BYTES` 512 MiB, `TIERED_MOE_SCRATCH_ARENA_BYTES` 1,056 MiB).
* **Engine Worker & Pipeline Overlap:**
  * Lock-free bounded channel ingress with immediate cancellation on caller disconnect, avoiding head-of-line blocking.
  * Double-buffered pipelined decode loop overlapping host scheduler bookkeeping and sampling setup with GPU execution.
  * Transactional multi-rank execution (`coordinated::replay_ranks`): Replays CUDA Graph submissions across TP ranks through an explicit `Upload -> Capture -> Launch` transaction barrier.

---

## 8. CUDA Graphs & Memory Arena Architecture
*Location: `crates/warp34-host/src/cuda.rs`, `crates/warp34-host/src/model/rank_runtime.rs`*

* **CUDA Graph Engine:**
  * Pre-captures execution graphs across plain decode (`decode_graph`), batched decode (`batched_decode_graphs` for batch sizes 1..16), and speculative verification (`spec_verify_graph` for M_verify 2..16).
  * Valid-token/valid-req masking: Padded rows write to designated sink locations, preventing corrupted recurrent states, RNG drift, or invalid KV writes.
  * Cold-start warmup protocol (`prepare_execution_plan()`): Resolves all dynamic symbols and executes warmup iterations before graph capture to prevent CUDA 13 lazy-loading deadlocks.
* **Pinned Host & Device Arenas:**
  * `DeviceArena` and sub-allocated buffers: Zero-allocation runtime operations using pre-allocated contiguous memory pools.
  * Zero-copy mapped-host embedding fallback (`plan_host_embed_offload`): Automatically offloads untied `embed_tokens` to pinned system memory on VRAM-constrained GPUs (e.g. 9B models on 8 GB RTX 3060 Ti) while preserving GPU execution for active layers.

---

## 9. Hybrid Radix Prefix Caching (`MambaRadixCache`)
*Location: `crates/warp34-host/src/radix.rs`, `crates/warp34-host/src/cache/checkpoint_pool.rs` (Donor: `SGLang v0.5.23` `mamba_radix_cache.py`)*

* **SGLang-Donor Hybrid Radix Tree:**
  * Unifies full-attention KV cache pages and GDN recurrent `StateBundle` checkpoints into a single prefix tree.
  * **Dual-Lock Reference Counting:** Enforces `full_lock_ref >= mamba_lock_ref`. Retaining a node locks `mamba_lock_ref` on that node and `full_lock_ref` along its entire ancestral path `[node..root)`.
  * **Dual LRU Eviction Lists:** Manages `full_lru` and `mamba_lru` independently. Prefill hits refresh `full_lru` along the path to the root and `mamba_lru` on the terminal node.
  * **Internal Tombstoning:** Intermediate recurrent states can be evicted (`mamba_value = None`) to reclaim `~72 MiB` GDN state bundles while keeping underlying KV pages intact for long-context descendants.
  * **Ping-Pong Track Buffers (`ReqRadixState`):** 2-slot ping-pong tracking buffers capture intermediate GDN states at 64-token boundaries, donating only fully committed (`emitted_len`) snapshots into the global tree upon verify success.
  * **Tiered `WHT128-Q8_0` GDN Checkpoint Pool (`cache/checkpoint_pool.rs`):** Compresses donated radix GDN checkpoints `4.06×` (`143.8 MiB -> 35.4 MiB` on 27B `TP=1`, `71.9 MiB -> 17.7 MiB/rank` on `TP=2`) using per-row Fast Walsh-Hadamard Transform (`WHT-128`) + symmetric `Q8_0` quantization across an L1 VRAM pool and an L2 pinned host RAM overflow pool while keeping active decode/verify `StateBundlePool` slots in full `FP32`.

---

## 10. Decoupled Memory Pools & State Transactions
*Location: `crates/warp34-host/src/cache.rs`*

* **Typed KV Pool (`KvPool`):**
  * Logical `page_id = token_pos / 16` addressing completely decoupled from per-layer physical byte strides.
  * Layer-specific `KvLayerLayout` descriptors configure base pointers, head dimensions (128 or 256), formats, transforms, and stride bytes per layer.
  * Unified page allocation bitmap across all full-attention layers with lockstep TP-rank clamping.
* **Transactional GDN State Pool (`StateBundlePool` & `StateTx`):**
  * Pre-allocated pool of persistent GDN state bundles (`conv_state` FP16 + `ssm_state` FP32 in global memory).
  * Explicit 6-phase state machine: `Idle`, `Prefilling`, `Decoding`, `SpecDrafting`, `SpecVerifying`, `RolledBack`.
  * Winner-take-all speculative resolution: Reclaims losing speculative state slots upon rejection without data movement or memory fragmentation.

---

## 11. Multimodal & Vision Transformer Pipeline
*Location: `crates/warp34-host/src/vision/`, `crates/warp34-host/src/scheduled_vision.rs`, `crates/warp34-kernels/csrc/ops/op_vision*.cu`*

* **Qwen3-VL Vision Architecture:**
  * Native implementation of `Qwen3_VisionTransformer` parameterized by model `VisionConfig` (e.g. 24 layers, 1,024 hidden size, 4,096 intermediate size, 16 heads on 4B/8B).
  * High-fidelity image preprocessing: `smart_resize` preserving aspect ratios up to 200:1, bicubic resampling, and normalization.
  * 2D bilinear `pos_embed` interpolation, 2D axial rotary position embeddings (RoPE), and `PatchMerger` with spatial merging.
  * 3D `mRoPE` coordinate generation across temporal, height, and width axes (`[3, N_max]`).
* **Multimodal Prefix Caching & Security:**
  * FNV-1a visual token content hashing integrates image feature embeddings directly into `MambaRadixCache` keys, enabling instant warm prefix restoration on repeated image queries.
  * SSRF-hardened remote image loader (`remote_images.rs`): Enforces 16 MiB download limits, 10s deadlines, zero redirect following, and strict IP blacklisting against private, loopback, link-local, carrier-grade NAT, and cloud metadata subnets.
* **Vision CUDA Kernels (`op_vision.cu`, `op_vision_fa2.cu`):**
  * Custom 2D RoPE and patch-merging CUDA kernels.
  * Dedicated Vision FlashAttention-2 (`op_vision_fa2.cu`) optimized for non-causal visual token self-attention.

---

## 12. Speculative Decoding Subsystem (MTP & DFlash2)
*Location: `crates/warp34-host/src/speculative.rs`, `crates/warp34-host/src/mtp/`, `crates/warp34-host/src/dflash.rs`*

* **Native Multi-Token Prediction (`MTP`):**
  * Clean-room implementation of `Qwen3_5MTP` / `Qwen3_5MoeMTP` (vLLM donor, `crates/warp34-host/src/mtp.rs`).
  * Single-layer TP-sharded predictor with projection, gated attention, dense/MoE FFN, and borrowed target LM head.
  * Online LM-head quantization: Quantizes unquantized target LM heads into a private `MarlinW4A16` (group size 64) proposal head on the fly when VRAM headroom permits.
  * Pre-captured proposal CUDA Graphs supporting up to 16 concurrent speculative slots (`MAX_MTP_BATCH_SLOTS = 16` when `all_pages >= 512`, `4` otherwise).
* **DFlash2 Block Speculation:**
  * Clean-room 5-layer native BF16 speculative drafter (`DFLASH2_NUM_LAYERS = 5`, SGLang `DFlashWorkerV2` donor), supporting both unquantized `BF16` and quantized `W4A16-HGPTQ` (`Linear::MarlinBf16U4B8`, `warp34_marlin_gemm_bf16u4` + `warp34_block_hadamard_bf16`).
  * Verifies `M = K + 1` proposals (up to 16 tokens, e.g. `M=8` on `27B` and `180B` `Flash-Next` models, `M=16` on `4B` models) in a single batched forward pass.
  * Projects committed target taps into an isolated, disposable `DFlashDraftContext` KV namespace (`head_dim = 128`).
  * Low-rank candidate selector kernel (`op_dflash2.cu`, `DFLASH2_MAX_TOP_K = 16`, selector rank up to 1,024).
* **Speculative Transaction Contract & Rejection Sampling:**
  * Enforces the 4-length transactional invariant across verify rounds: `processed_len` (target consumed), `emitted_len` (caller visible), `reserved_len` (allocated headroom), and `draft_context_len` (proposer cache).
  * Speculative commit & rollback (`crates/warp34-host/src/speculative.rs`): Atomically commits accepted drafts, emits bonus anchor tokens, and rolls back grammar matcher state.
  * Lossless dynamic speculative handoff (`tests/speculative_handoff.rs`): Dynamically yields speculative resources to incoming batched requests without dropping tokens or breaking state.

---

## 13. Model Architecture Support Matrix
*Location: `crates/warp34-host/src/model/`, `crates/warp34-kernels/csrc/ops/op_hc.cu`, `op_ple.cu`, `op_qsa.cu`*

* **Target Model Family Implementations:**
  * **Qwen 3.5 Dense Models (0.8B, 2B, 4B, 9B, 27B):** Hybrid architecture interleaving 3× Gated DeltaNet (GDN) linear attention layers with 1× Full Attention layer.
  * **Qwen 3.6 MoE Models (35B-A3B):** Hybrid GDN + Full Attention with routed experts (`num_experts_per_tok = 8`) plus shared expert.
  * **Qwen 3.8-Flash-Next MoE (`180B` total / `6B` active: `125B` 48-layer `40 GDN + 8 QSA` backbone + `51.2B` 8-layer PLE table + `4B` MTP drafter, `qwen4_exp`):**
    * 512 total experts with 10 active routed experts + 1 shared expert per token (`num_experts_per_tok = 10`).
    * **HyperConnection (`HC`, `op_hc.cu`):** 4-stream multi-residual gating and mixing with per-stream `GemmaRMSNorm`, low-rank SiLU projections, and sigmoid stream routing.
    * **Per-Layer Embeddings (`PLE`, `op_ple.cu`, `ple.rs`):** Sharded zero-copy mmap `lut4-gptq` (`PleTableFormat::Lut4Gptq`, `3.63×` smaller) and `F8_E4M3` (`PleTableFormat::F8E4m3`) embedding tables across `8` PLE layers (`[5,11,17,23,29,35,41,47]`) with stateful causal dilated depthwise conv1d (`kernel = 4`, `dilation = 3`, `state_len = 9`) and signed-square-root sigmoid gating.
    * **Query-Sparse Attention (`QSA`, `op_qsa.cu`):** Top-512 block indexer scoring (`2,048` active tokens out of up to `262K`) with 4:1 compressed key cache, ReLU-sum MQA, and sparse paged GQA decode.
* **Precision Policy Invariants:**
  * End-to-end `FP16` target model spine (hidden states, residuals, norms, attention Q/K/V/O, Marlin activations).
  * Native `BF16` preservation for DFlash2 drafters to prevent dynamic range overflow.
  * `FP32` persistent recurrent states and reduction trees (`ssm_state` `[H_v, 128, 128]`, RMSNorm variance, softmax denominators, and logit distributions).

---

## 14. Model Loading & Marlin Quantization Engine
*Location: `crates/warp34-host/src/loader/`, `crates/warp34-kernels/csrc/marlin/`, `op_marlin.cu`*

* **Universal Checkpoint Ingestion:**
  * Direct memory-mapped SafeTensors loading with automated shard resolution from local directories or Hugging Face Hub (`HF_HUB_CACHE`).
  * Supported formats: Unquantized `FP16`/`BF16`, `W4A16-HGPTQ` (block-Hadamard `H_64`/`H_16`), mixed-precision `W2.6A16-HGPTQ-RCO` & `W2.2A16-HGPTQ-RCO` (`2-bit U2B2` + `4-bit U4B8` + online `8-bit U8B128` `lm_head`), `GPTQ`, `AWQ`, `Compressed-Tensors` (`pack-quantized`), and `AutoRound`.
* **Hardware-Accelerated Marlin Repack:**
  * Loads quantized checkpoints and repacks weights at startup into unified Marlin tile formats (`W4A16` `U4B8`, `W2A16` `U2B2`, `W4A16-ZP`, `W8A16` `U8B128`, `BF16` `U4B8`, and shared-weight `W4A8` INT8-IMMA prefill `warp34_marlin_gemm_s8u4_shared`).
  * Supports group sizes 0 (channelwise), 32, 64, and 128 with static/dynamic act-order.
  * Marlin GEMM kernels optimized for `sm_86` and `sm_89` with asynchronous global-to-shared copies (`cp.async`) and mma tensor core operations.
  * 1-Slice grid invariance: Enforces deterministic tile-count divisor CTA grids for `M <= 16` shapes, guaranteeing identical FP32 reduction trees between TP=1 and TP=2.

---

## 15. Tiered MoE & Asynchronous CPU Offload Subsystem
*Location: `crates/warp34-host/src/moe_offload/`, `crates/warp34-host/src/moe_cpu_bridge/`, `crates/warp34-kernels/csrc/ops/op_moe*.cu`*

* **512-Expert Tiered Offloading Engine:**
  * Designed for memory-constrained execution of `180B` models (`Qwen3.8-Flash-Next`) on 16 GB–24 GB GPUs (`TP=1` and `TP=2 / EP=2`) and 100% VRAM-resident execution on `TP=4 / EP=4`.
  * `ExpertPackLayout`: Contiguous 4KiB-aligned DMA layout containing `U4B8` or `U2B2` weights and unpermuted FP16 scales.
  * **Dynamic TinyLFU + Clock-Pro & Smooth BIP-LRU Cache:** Manages VRAM hot expert cache with on-GPU TinyLFU + Clock-Pro frequency-gated promotion (`warp34_moe_hybrid_plan_and_submit_dynamic`, `TIERED_MOE_DYNAMIC_LRU_MIN_HOT_EXPERTS = 64`) on `TP=1` and a smoothed `k_protected` BIP-LRU floor (`tail_budget = (total_pairs <= 40) ? 40 : min(40, k_hot / 2)`) on `TP=2`, preserving anchor slots during speculative verify rounds.
  * Dual execution modes: `Auto` dynamically selects between zero-sync device-LRU / hybrid CPU bridge (for decode, `M=4`/`M=8` verify, and short warm-prefix suffixes `num_tokens <= 40` via 8-token sliced launches + slice-0 shared-expert GPU overlap) and `StreamedPrefill` (with `dynamic_short_adapt` hot-slot promotion across `32..1024`-token warm turns and prompt-frequency promotion on `>= 1024`-token prompts).
* **Asynchronous Hybrid CPU Bridge:**
  * Offloads cold expert evaluation for decode and speculative verify tokens (`M <= 8`) to multi-threaded host workers (AVX-512 and AVX2 optimized).
  * Double-buffered pinned host mailbox with lock-free sequence counters and atomic job stealing.
  * Asynchronously accumulates CPU-evaluated cold expert down-projections into GPU activation buffers over PCIe while the GPU computes resident hot experts.
* **Unified GPU LRU Pull Kernel (`op_moe_router.cu`):**
  * Consolidated `moe_pull_mapped_experts_kernel` performs DMA transfers from host memory to VRAM hot slots in a single GPU-driven launch.

---

## 16. Recurrent Linear Attention Kernels (Gated DeltaNet)
*Location: `crates/warp34-kernels/csrc/ops/op_gdn.cu`, `op_gdn_wy.cu`, `op_gdn_wy_chunk64.cu`*

* **Native CUDA C++ GDN Implementation:**
  * Ported and hardened from `vllm.cpp` (`50d425bec`), verified against `vLLM` FLA golden dumps.
  * State dtypes: `conv_state` in `FP16` (`[slots, conv_dim, 3]`), `ssm_state` in `FP32` (`[slots, H_v, 128, 128]`).
  * `gdn_causal_conv1d_f16_kernel`: Stateful causal 1D convolution with SiLU activation and slot-based snapshot writing.
  * `gdn_fused_recurrent_delta_rule_kernel`: Decode (`q_len = 1`) and speculative verify (`q_len = 1 + k`) recurrence with fused sigmoid gating and delta update rules.
* **Chunked GDN-WY Prefill Kernels:**
  * 3-stage chunked prefill algorithm over 64-token tiles (`BT = 64`):
    * Stage 1: Computes intra-chunk Householder-like representation (`W` and `U` matrices).
    * Stage 2: Evaluates inter-chunk recurrent state propagation (`Delta H`).
    * Stage 3: Produces final outputs with FP32 accumulators.
  * Continuation/checkpoint tests exercise exact parity for their declared geometries; this is not a universal 0-ULP guarantee across every prefill kernel and full-model configuration.

---

## 17. Attention & TurboQuant 4-Bit KV Cache Kernels
*Location: `crates/warp34-kernels/csrc/tq/`, `op_kv_cache.cu`, `op_hadamard.cu`, `op_paged_attn.cu`, `op_paged_fa2.cu`*

* **TurboQuant 4-Bit KV Cache (`TQ4`):**
  * Pinned in-house implementation with **`3.07×` effective KV compression** for the documented 27B layout (21,376 bytes/token). Historical 262k-context runs on a single 24 GB GPU are recorded in [`STATE.md`](STATE.md); capacity and quality depend on the checkpoint, other memory allocations, and workload, and require release-specific qualification.
  * `block_tq4_0`: 66 bytes per 128 elements (4.125 bpw) with independent per-128 least-squares FP16 scales.
  * Randomized orthonormal rotation: Seed-42 Rademacher + Fast Walsh-Hadamard Transform (WHT-256) eliminates outlier dimensions prior to quantization.
  * 16-centroid `Beta(256)` Lloyd-Max codebook stored in `__constant__` memory and register shuffle caches.
  * **Boundary-Anchored KV Policy:** The last 4 full-attention layers remain in rotated `Q8_0` (272 B/row) or `FP8_E4M3` (264 B/row on `sm_89+`) while preceding layers use rotated `TQ4` (132 B/row), limiting quantization error (`3.07×`–`3.10×` effective KV compression; `--kv-format tq4_all` applies `TQ4` uniformly across all layers for `3.88×` compression). TQ4 is lossy: historical greedy token agreement varies by model and prompt; see [`STATE.md`](STATE.md).
* **Paged Attention & FlashAttention Kernels:**
  * Cooperative 1-warp GQA decode kernel with register-shuffled TQ4 centroid dequantization.
  * Tensor core MMA decode and prefill kernels (`mma.sync.m16n8k16`) with Split-KV grid reductions.
  * Integrated Paged FlashAttention-2 (`op_paged_fa2.cu`) supporting variable-length causal batches.

---

## 18. Fused Layer Norms, RoPE, Activations & Small-N GEMMs
*Location: `crates/warp34-kernels/csrc/ops/op_norms_rope_act.cu`*

* **Fused Operators:**
  * `GemmaRMSNorm`: Vectorized (vec8) RMSNorm with `(1.0 + weight)` unit offset and FP32 variance accumulation.
  * `fused_qk_rmsnorm_rope_gate`: Fuses Q/K split, per-head RMSNorm, partial interleaved 3D `mRoPE` (rotary dimension 64, sections `[11, 11, 10]`), and output projection gating into a single memory pass.
  * `silu_and_mul`: Fused SwiGLU / SiLU activation and elementwise multiplication with optional Hadamard transform injection.
  * `RMSNormGated`: Fused post-GDN normalization with SiLU gating (`norm_before_gate = true`).
* **M-Invariant Small-N GEMM Kernels:**
  * Custom `small_n_gemm_4x4_f16_kernel`, `small_n_gemm_dual_4x4_f16_kernel`, and `small_n_gemm_dual_8x4_f16_kernel` handling skinny projection matrices (`N <= 2560`, PLE projections `N = 10240`).
  * Eliminates cuBLAS kernel switching between `M = 1`, `M = 4`, and `M = 8`, ensuring bit-exact 0-ULP reduction invariance across plain and speculative decode steps.

---

## 19. Cross-GPU Collectives & Tensor Parallelism
*Location: `crates/warp34-kernels/csrc/ops/op_collectives.cu`*

* **High-Efficiency TP Collectives:**
  * Primary verified implementation: Pinned-sysmem TP=2 (`op_collectives.cu`) over PCIe Gen4 x8 (e.g. 2× RTX 3090 Ti / 3060 Ti / 4060 Ti / 4090).
  * System-scope release/acquire memory fences (`st.release.sys.global` / `ld.acquire.sys.global`) with monotonic epoch parity: small in-kernel transfers use one barrier; the large Copy-Engine DMA path uses two barriers.
  * Dynamic DMA cutoff (`kDmaThresholdBytes = 64 KiB`): Selects in-kernel register writeback for small transfers and asynchronous Copy-Engine DMA for bulk payloads.
  * Strictly typed collective entrypoints: Typed routines for `FP16` hidden states, `BF16` draft tokens, `FP32` reductions, and `INT32` token IDs (never casting integer tokens through float collectives).
  * Implemented NVLink P2P VRAM staging and TP=4 execution plans (`crates/warp34-host/src/cuda/collectives.rs`), with historical 4× RTX 3090 qualification in [`STATE.md`](STATE.md). These require separate hardware/topology requalification for a release.

---

## 20. GPU Sampler & Speculative Rejection
*Location: `crates/warp34-kernels/csrc/ops/op_sampler.cu`, `op_rejection.cu`*

* **9-Step Persistent GPU Sampler:**
  * GPU sampling uses device scratch buffers, reductions, and specialized fast paths; random draws are supplied by host `SamplerV1`. The policy includes:
    1. Grammar token bitmask application (`-INFINITY` masking).
    2. Logit bias adjustments.
    3. Repetition, frequency, and presence penalty applications.
    4. Temperature scaling.
    5. Min-p dynamic threshold filtering.
    6. Top-k and Top-p (nucleus) sorting and cumulative mass truncation.
    7. Seeded sampling using host-generated, position-keyed SplitMix64 uniform draws passed in the row configuration.
    8. Fast-greedy argmax shortcut with warp/block reductions for eligible rows (not a one-cycle latency guarantee).
    9. FP32 logprob extraction with configurable `top_logprobs` (up to 20 in the full sampler).
* **GPU Speculative Rejection Sampler:**
  * Parallel verification of speculative draft tokens against target acceptance probabilities (`p / q`).
  * Supports both greedy acceptance and stochastic residual distribution sampling (`sample max(p - q, 0)`).

---

## 21. Real-Time Telemetry & Verification Framework
*Location: `crates/warp34-host/src/telemetry.rs`, `tools/oracle/`*

* **Thread-Safe Metrics & Prometheus Monitoring:**
  * Non-blocking CPU snapshots tracking request counts (submitted, running, completed, failed, cancelled).
  * Token accounting: Separates prompt, computed, radix-cached, and generated tokens.
  * High-resolution latency histograms: Time-To-First-Token (`ttft`), Queue Wait Time (`queue`), End-to-End Latency (`e2e`), and Inter-Token Latency (`time_per_output_token`).
  * Memory pool utilization: Tracks live capacity, active pages, cached/evictable pages, and free physical pages across KV and GDN pools.
* **Continuous Parity Oracle Suite (`tools/oracle/`):**
  * Automated Python/Triton test harnesses running `vLLM 0.34.0` and `SGLang v0.5.23` side-by-side with `warp34`.
  * Provides a 12-prompt canonical multi-turn/reasoning oracle suite. Exact token matches, numerical tensor tolerances, and adjudicated divergences are distinct outcomes; historical results are not a guarantee of universal token equivalence or a fresh release pass.

---

## 22. Containerization & Multi-Architecture Fatbinary Packaging
*Location: `Dockerfile`*

* **Optimized Docker Deployment:**
  * Multi-stage build process decoupling the Node/pnpm React dashboard compilation from the CUDA/Rust binary build.
  * Runtime image based on `ubuntu:26.04` (shared-link runtime filesystem size ~775.6 MB; the ~575.4 MB static variant was an experiment, not the current default) without shipping multi-GB CUDA toolkits, headers, or driver stubs.
  * Shared cuBLAS dynamic linkage (`libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`) stored in a dedicated cacheable filesystem layer.
  * Standard Hugging Face cache & health checks: Binds `0.0.0.0` via `WARP34_HOST=0.0.0.0`, supports standard `-v ~/.cache/huggingface:/root/.cache/huggingface` mounts (plus non-root overrides via `--user "$(id -u):$(id -g)" -e HF_HOME=/cache`), and defines container health checks querying `GET /health`.
* **Multi-Arch Fatbinary Support:**
  * Compiles native SASS cubins for Ampere (`sm_86`) and Ada Lovelace (`sm_89`) plus `compute_89` PTX fallback.
  * Embedded device code compression enabled (`WARP34_NVCC_EXTRA=--compress-mode=size`).
  * Dynamic hardware detection (`warp34_device_props`) adapts CTA grid dimensions and shared memory allocations at runtime to the specific GPU SM count (from 28 SMs on RTX 3060 to 142 SMs on RTX 4090).
