# Project State

Current verification status, architectural capabilities, and benchmark records for `warp34` across NVIDIA A100 (`sm_80`), RTX 30-series (`sm_86`), and RTX 40-series (`sm_89`) GPUs.

---

## 1. Engine Status & Capabilities Summary

`warp34` is feature-complete and verified across **1–2× A100 (`sm_80`)**, **1–4× RTX 30-series (`sm_86`, 8–24 GB)**, and **1–2× RTX 40-series (`sm_89`, 16–24 GB)** GPUs:

- **Model Families & Multi-Format Loader (`crates/warp34-host/src/{hub,loader,model}/`, [`MODELS.md`](MODELS.md)):**
  - Native download-on-first-use via `hf-hub = 1.0.0` (`--revision`, `--draft-revision`, `--offline`, Xet concurrent streaming).
  - **Qwen 3.5 Dense (`0.8B`, `2B`, `4B`, `9B`)** and **Qwen 3.8 Dense (`27B`)** hybrid `3:1` Gated DeltaNet (`GDN`) + full GQA backbones.
  - **Qwen 3.6 Resident MoE (`35B-A3B`)**: 256 routed experts (top-8 softmax) + gated shared expert, fitting on a single 24 GB GPU (`W4A16-HGPTQ`) or a single 16 GB GPU (`W2.6A16-HGPTQ-RCO`).
  - **Qwen 3.8 Flash-Next (`qwen4_exp`, `80B-A3B`, [`QWEN_FLASH_NEXT.md`](QWEN_FLASH_NEXT.md))**: 512 routed experts (top-10 sigmoid) + shared expert + 4-stream **HyperConnection (`HC`)** + **Per-Layer Embeddings (`PLE`)** + **Qwen Sparse Attention (`QSA`)**, served on 8–24 GB GPUs via tiered VRAM hot-expert caches (`HotExpertCache`) + zero-copy pinned-host `HostExpertStore` + SIMD CPU bridge (`AVX-512` / `AVX2` / portable scalar).
  - **Multimodal Vision (`Qwen3_VisionTransformer`)**: Native 2D-RoPE ViT + 2×2 spatial merger + `mRoPE` 3D position IDs + content-hashed image-token radix caching across `2B`, `4B`, `9B`, `27B`, `35B-A3B`, and `Flash-Next`.
  - **Speculative Decoding (`Qwen3_5MTP` & `DFlash2`)**: Per-rank CUDA-Graphed native multi-token prediction (`k = 1..3`, online `MarlinW4A16` `draft_lm_head`, `B = 1..4` batched verification) and external `BF16` / `MarlinBf16U4B8` `DFlash2` block diffusion drafters (`M = 16` on `4B`, `M = 8` on `27B` and `Flash-Next`, tap layers `[5, 15, 25, 35, 45]` with stream-averaged `BF16` `hc_combine_tap_f16`).
  - **Decision Models ([`DECISION_MODELS.md`](DECISION_MODELS.md))**: Headless `CodeReadout` (`d3-edge`, `d3-lite`, `d3-ultra`, `GLiDE-0.8B/4B/9B/27B`) and `CausalLm` (`Clef`, `Kev`, `Nimble`, `Tev1`, `OpenJev`, `Xor`/`Lev`) with `noncausal_full_attention`, `prefix_bidir` state-once prefill, `midloop` damped-Euler recurrence, and `order_passes = 1 | 2` permutation averaging.
  - **Runtime Multi-Adapter LoRA (`crates/warp34-host/src/lora.rs`)**: Hot-swappable per-request PEFT adapters (`--lora name=path`, `--max-loras`, `--max-cpu-loras`, `--max-lora-rank`) with fused GPU shrink/expand kernels (`op_lora.cu`) across dense, GDN, and MoE projections.
  - **Image Generation & Editing (`Qwen-Image-2.1` / `Qwen-Image-2.1-Turbo`, `crates/warp34-host/src/qwen_image/`, `crates/warp34-host/src/server/image_http.rs`)**: Native `Qwen3-VL-8B` conditioning encoder (27-block ViT + 36L text encoder + DeepStack + `head_dim=128` FA2), 60-block single-stream joint DiT (`TP=1` & `TP=2`, Step-0 KV caching + whole-step CUDA Graph + fused `gated_residual + adaLN + H64` + 8-chunk pipelined PCIe `TP=2` AllReduce), 64-channel RGBA 3D Causal VAE (`im2col + cuBLAS` with L2-aware strip tiling, power-of-two `RMSNorm` rescaling to prevent 5-stage `FP16` overflow, auto-tiled `>1536 px` decode), CLI (`image`, `bench-image`, `dump-diff`), OpenAI Images API (`/v1/images/generations`, `/v1/images/edits`), and Web Playground with 1-click Edit Image.

- **Weight & KV Quantization ([`HGPTQ.md`](HGPTQ.md), `crates/warp34-kernels/csrc/{marlin,ops,tq}/`):**
  - **Weights**: `MarlinW4A16` (`U4B8` symmetric) & `MarlinW4A16Zp` (`U4` asymmetric) across `GPTQ`, `AWQ`, `compressed-tensors`, and `AutoRound` (`group_size ∈ {32, 64, 128}` or channelwise `-1`); native 2-bit **`MarlinW2A16` (`kU2B2`, `0.25 B/weight`)** and mixed `W2`/`W4`/`W8` **`HGPTQ-RCO`** (`-36.0%` MoE expert footprint); **`W4A8` INT8 IMMA (`m16n8k32`)** dynamic activation quantization for dense `M >= 128` prefill (`1.93–1.99×` prefill speedup on `sm_86`/`sm_89`); `MarlinW8A16`; producer-fused in-register **`H_64` / `H_16` block-Hadamard (`HGPTQ`)**; 4-bit **`lut4-gptq`** (`11.1 GiB`) and **`F8_E4M3`** (`40.4 GiB`) `PLE` mmap tables; `DenseF16`; and `DFlash2` `DenseBf16` / `MarlinBf16U4B8`.
  - **Paged KV Cache (`KvFormatPolicy`)**: `fp16`, `bf16`, `q8_0` (`1.88×`), `fp8` (`E4M3` on `sm_89+`), tail-anchored **`tq4` (`3.07×`, 4 unrotated `Q8_0` tail layers + 12 rotated LSQ `Beta(256)` WHT-256 `TQ4` interior layers)**, and **`tq4_all` (`3.88×`)**. Passes **8/8** single-GPU `27B` needle-in-a-haystack evaluations across `32K / 64K / 128K / 262K` contexts and live `252K`-token multi-turn coding-agent trajectories.

- **Multi-GPU Collectives & Capacity Planning (`crates/warp34-kernels/csrc/ops/op_collectives.cu`, `crates/warp34-host/src/model/rank_capacity.rs`):**
  - **`TP=1`, `TP=2`, and `TP=4` (`EP=2` / `EP=4`)**: Automatic 1-barrier NVLink P2P VRAM collectives (`CommBuffer::DeviceVram`, `36` CTAs) when direct peer access is available, and two-tier PCIe pinned-sysmem collectives (`CommBuffer::MappedHost`) combining `>= 64 KiB` Copy-Engine `cudaMemcpyAsync` DMA TX + 2-barrier SM RX (`5.8×` faster `M = 512` collectives on non-`iommu=pt` hosts) with `< 64 KiB` 1-barrier `st.global.wb.v4.u32` streaming stores and fused `all_reduce_residual_gemma_rms_norm_f16` / `all_reduce_hc_combine_tap_f16`.
  - **Adaptive Capacity Budgeting**: Pre-flight VRAM validation, automatic zero-copy `MappedHostBuffer` `embed_tokens` offload when GPU residency would leave `< 2.0 GiB` slack (`HOST_EMBED_GPU_RESIDENCY_MIN_SLACK_BYTES`), multi-tier `Flash-Next` hot-expert + KV auto-planning (`plan_tiered_moe_hot_per_layer_with_options`), and dedicated serving GDN state-slot budgets (`serving_state_slots`).

- **Serving, Radix Cache, Structured Outputs & Telemetry ([`SERVING.md`](SERVING.md), [`GENERATIVE_APIS.md`](GENERATIVE_APIS.md), [`STRUCTURED_OUTPUTS.md`](STRUCTURED_OUTPUTS.md), [`METRICS.md`](METRICS.md)):**
  - **`MambaRadixCache` + `HybridReqToTokenPool` + L2 Host-RAM KV Virtual Memory (`HostKvSwapPool`)**: Coupled page-aligned KV + FP32 GDN (`conv_state` + `ssm_state`) prefix tree with prompt-frontier preservation (`is_decode_tail`), sparse `8,192`-token (`ANCHOR_CHECKPOINT_STRIDE_PAGES = 512`) + penultimate anchor checkpoint preservation, in-place chunked-prefill checkpoint recycling (`can_recycle_prefill_checkpoint`), `8 GiB/rank` (`215`-slot on `27B TP=2`) WHT128-Q8_0 GDN checkpoint pool, lazy `256`-page layer-major L2 Host-RAM KV swap slabs (`HostKvSwapPool`, `16 GiB/rank` = `786,432` extra tokens on `27B` for `1,048,576` total L1+L2 KV tokens) with contiguous DMA run coalescing and non-destructive L1 ↔ L2 demotion/promotion, elastic power-of-two `MTP-3` batch slot scaling (`16 <-> 8 <-> 4 <-> 2` slots, scaling per-slot capacity from `16,384` up to `65,536–131,072` tokens via in-place D2D KV relocation), 3-tier LRU speculative batch-slot affinity, and GPU D2D cross-slot KV + QSA side-cache cloning (`copy_kv_from_slot`).
  - **Structured Outputs**: Native Rust `xgrammar-rs = 0.2.8` (`512 MiB` shared grammar cache, `<think>`-aware reasoning gating) supporting JSON Schema, Regex, EBNF, Lark, structural tags, and Qwen3-XML tool calls across OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`), Anthropic (`/v1/messages`, `/v1/messages/count_tokens`), and Decision (`/v1/systemone`, `/v1/decisions`) endpoints, plus Prometheus `/metrics`, `/api/metrics`, and the embedded React/Vite dashboard.

---

## 2. Verified Head-to-Head & Campaign Benchmarks

### 2.1 `Qwen3.8-Flash-Next` (`80B-A3B`, 512-Expert Tiered MoE) vs Specialized Engines

3-repetition `warp34perf` HTTP medians (`tok/s`, `6/6` quality gates passed, `100%` lossless verification) across all 5 `flash-next-hgptq.toml` / `flash-next-serving.toml` workloads (`short-chat`, `prefill-context-r16`, `code-context-r64`, `prefix-primed-r16`, `coding-agent-replay`):

| Hardware | Engine / Profile | `short-chat` | `prefill-r16` | `code-r64` | `prefix-r16` | `agent-replay` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **1× RTX 3090 Ti** (`TP=1`) | **`warp34` (`W2.6A16-HGPTQ-RCO`, `DFlash2`, `k_hot=192`)** | **100.2** | **33.4** | **37.1** | **93.2** | **62.4** |
| **1× RTX 3090 Ti** (`TP=1`) | **`warp34` (`W4A16-HGPTQ`, `DFlash2`, `k_hot=72`)** | **83.9** | **32.1** | **34.7** | **74.0** | **58.4** |
| **1× RTX 3090 Ti** (`TP=1`) | `Strata v0.2.17` (`Q4_K_M` GGUF, `DFlash2 b16`, lossy `TAIL_SKIP=7`) | 80.9 | 23.6 | 25.0 | 82.9 | 64.7 |
| **1× RTX 3090 Ti** (`TP=1`) | **`warp34` (`W4A16-HGPTQ`, `MTP-3`, `k_hot=112`)** | **53.7** | **26.3** | **26.9** | **49.6** | **42.6** |
| **1× RTX 3090 Ti** (`TP=1`) | `qwen38-flash-next-3090` (`FP8PLE`, `MTP-3`) | 31.8 | 10.6 | 15.6 | 10.6 | 15.1 |
| **2× RTX 3090 Ti** (`TP=2`) | **`warp34` (`W2.6A16-HGPTQ-RCO`, `DFlash2`, `k_hot=192/rank`)** | **167.0** | **51.1** | **51.1** | **167.1** | **135.4** |
| **2× RTX 3090 Ti** (`TP=2`) | **`warp34` (`W4A16-HGPTQ`, `DFlash2`, `k_hot=144/rank`)** | **140.1** | **51.5** | **50.0** | **123.5** | **123.6** |
| **2× RTX 3090 Ti** (`TP=2`) | **`warp34` (`W4A16-HGPTQ`, `MTP-3`, `k_hot=144/rank`)** | **99.0** | **43.9** | **43.2** | **90.5** | **91.9** |
| **2× RTX 3090 Ti** (`TP=2`) | `Strata v0.2.17` (`Q4_K_M` GGUF, `DFlash2 b16`, lossy `TAIL_SKIP=7`) | 94.8 | 26.8 | 28.1 | 94.6 | 80.5 |
| **2× RTX 3090 Ti** (`TP=2`) | `qwen38-flash-next-2x3090` (`FP8PLE`, `MTP-3`, 128 GB RAM) | 56.2 | 21.3 | 28.2 | 20.5 | 34.9 |
| **1× RTX 4090** (`TP=1`) | **`warp34` (`W2.6A16-HGPTQ-RCO`, `DFlash2`, `k_hot=192`)** | **148.2** | **54.1** | **59.4** | **145.6** | **87.2** |
| **1× RTX 4090** (`TP=1`) | **`warp34` (`W4A16-HGPTQ`, `DFlash2`, `k_hot=72`)** | **120.0** | **55.9** | **56.3** | **97.3** | **75.9** |
| **1× RTX 4090** (`TP=1`) | `Strata v0.2.17` (`Q4_K_M` GGUF, `DFlash2 b16`, lossy `TAIL_SKIP=7`) | 109.4 | 39.8 | 41.3 | 98.7 | 89.2 |
| **2× RTX 4090** (`TP=2`) | **`warp34` (`W2.6A16-HGPTQ-RCO`, `DFlash2`, `k_hot=192/rank`)** | **201.3** | **71.6** | **78.8** | **207.8** | **185.7** |
| **2× RTX 4090** (`TP=2`) | **`warp34` (`W4A16-HGPTQ`, `DFlash2`, `k_hot=144/rank`)** | **179.6** | **75.2** | **79.3** | **178.7** | **184.7** |
| **2× RTX 4090** (`TP=2`) | `Strata v0.2.17` (`Q4_K_M` GGUF, `DFlash2 b16`, lossy `TAIL_SKIP=7`) | 132.7 | 51.1 | 52.8 | 113.1 | 112.6 |

### 2.2 `Qwen3.5-9B` (`W4A16-HGPTQ` + `W4A8` INT8 IMMA Prefill) Head-to-Head (`9b-head-to-head.toml`)

| Hardware | Engine / Mode | Cold 4K Prefill (`tok/s`) | Single-Stream (`tok/s`) | `C=16` Batch (`tok/s`) | 60K Cold-Read (`tok/s`) | 115K Agent (`tok/s`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **1× RTX 3090 Ti** | **`warp34` (`MTP-3` / `Plain` prefill)** | **8,223 / 8,632** | **233.5** (`103.5` Plain) | **1,346.4** (`1,348.8` Plain) | **42.3** | **19.5** |
| **1× RTX 3090 Ti** | `vLLM 0.34.0` (`MTP-3`) | 4,672 | 164.1 | 1,073.4 | 18.8 | 10.7 |
| **1× RTX 3090 Ti** | `SGLang v0.5.23` (`MTP-3`) | 4,648 | 146.9 | 767.4 | 15.2 | 12.1 |
| **1× RTX 3090 Ti** | `llama.cpp` (`Q4_K_M`) | 4,562 | 94.7 | 506.9 | 12.3 | 5.2 |
| **1× RTX 3090 Ti** | `ninfer-3090` (`115dee6`, `MTP-3`) | 2,953 | 134.9 | 134.1 (`C=1` only) | 12.3 | 2.6 |
| **2× RTX 3090 Ti** | **`warp34` (`TP=2`, `MTP-3` / `Plain` prefill)** | **8,408 / 9,012** | **338.6** (`167.8` Plain) | **1,675.8** | **49.8** | **26.4** |
| **2× RTX 3090 Ti** | `vLLM 0.34.0` (`TP=2`, `MTP-3`) | 5,054 | 173.4 | 878.8 | 30.8 | 18.0 |
| **2× RTX 3090 Ti** | `SGLang v0.5.23` (`TP=2`, `MTP-3`) | 4,475 | 170.5 | 864.8 | 21.8 | 18.5 |
| **1× RTX 4090** | **`warp34` (`MTP-3` / `Plain` prefill)** | **14,548 / 15,939** | **252.6** (`108.9` Plain) | **1,750.3** (`1,771.7` Plain) | **69.5** | **36.2** |
| **1× RTX 4090** | `vLLM 0.34.0` (`MTP-3`) | 8,708 | 182.9 | 1,354.6 | 38.6 | 22.9 |
| **2× RTX 4090** | **`warp34` (`TP=2`, `MTP-3` / `Plain` prefill)** | **14,519 / 16,104** | **369.2** (`177.9` Plain) | **2,319.2** | **82.0** | **51.0** |
| **2× RTX 4090** | `vLLM 0.34.0` (`TP=2`, `MTP-3`) | 10,803 | 243.5 | 1,701.7 | 65.2 | 41.3 |

### 2.3 `Qwen3.8-27B` & `Qwen3.6-35B-A3B` Serving & Long-Context Standings

- **`Qwen3.8-27B` (`nchapman/Qwen3.8-27B-W4A16-HGPTQ`):**
  - **1× RTX 3090 Ti (`TP=1`)**: `FP16` KV (`27b-fp16-first.toml`) reaches **`52.12 tok/s`** `decode-c1` (`140.0 ms` TTFT vs `vLLM` `45.32`, `SGLang` `26.86`, `llama.cpp` `39.17`), **`311.75 tok/s`** `throughput-c8` (vs `vLLM` `222.74`, `SGLang` `152.93`), **`49.21 tok/s`** `agent-c1`, and **`51.98 tok/s`** `prefill-primed-c1` (`31.2 ms` TTFT, `100%` hit). With `TQ4` KV + speculation, single-stream decode reaches **`150.1 tok/s`** (`MTP-3`) and **`247.9 tok/s`** (`DFlash2 M=8`). On `252K` `Open-SWE` coding-agent replay (`27b-tp1-262k-coding-agent.toml`, `29/29` requests), `warp34` reaches **`12.02 tok/s`** (`115K`) and **`6.90 tok/s`** (`252K`, `93.3%` prefix hit) where `vLLM` and `SGLang` OOM at `120K+` on 24 GB.
  - **2× RTX 3090 Ti (`TP=2`)**: `27b-spec-and-tp2.toml` reaches **`82.97 tok/s`** Plain / **`146.36 tok/s`** `MTP-2` / **`157.31 tok/s`** `DFlash2` on HTTP `decode-c1` (**`252.1 tok/s`** `MTP-3` and **`351.0–365.1 tok/s`** `DFlash2` on `warp34 bench`), **`405.82 tok/s`** (`TQ4`) / **`466.27 tok/s`** (`FP16`) on `throughput-c8`, and **`22.29 tok/s`** (`MTP-3`, `677 ms` median warm TTFT) on `115K` coding-agent replay.
  - **1–2× RTX 4090 (`sm_89`)**: `TP=1` reaches **`53.5 tok/s`** Plain (`6,161 tok/s` `W4A8` 4K prefill), **`162.0 tok/s`** `MTP-3`, and **`258.6 tok/s`** `DFlash2`; `TP=2` reaches **`88.5 tok/s`** Plain (`5,926 tok/s` 4K prefill), **`230.1–246.0 tok/s`** `MTP-3`, **`333.6 tok/s`** `DFlash2`, and **`593.0 tok/s`** HTTP `throughput-c8`.
- **`Qwen3.6-35B-A3B` Resident MoE (`nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ`):**
  - **1× RTX 3090 Ti (`TP=1`, `TQ4`)**: Fits on a single 24 GB GPU with `MTP-3` + Vision where `vLLM` OOMs, reaching **`205.0 tok/s`** Plain / **`429.1 tok/s`** `MTP-3` on `warp34 bench`, and **`196.83 tok/s`** `decode-c1` / **`729.10 tok/s`** `throughput-c8` on `35b-a3b-moe-serving.toml` (outperforming 2-GPU `vLLM TP=2` `191.42 tok/s` on 1 GPU).
  - **2× RTX 3090 Ti (`TP=2`)**: Reaches **`281.3 tok/s`** Plain / **`553.4 tok/s`** `MTP-3` (`7,579 tok/s` prefill) on `warp34 bench`, and **`261.21 tok/s`** Plain / **`389.60 tok/s`** `MTP-2` `decode-c1`, **`654.52 tok/s`** `spec-c4`, and **`1,019.54 tok/s`** `throughput-c8` on `35b-a3b-moe-serving.toml`.
  - **1–2× RTX 4090 (`sm_89`)**: `TP=1` reaches **`225.5 tok/s`** Plain (`11,890 tok/s` prefill) and **`501.9–512.0 tok/s`** `MTP-3`; `TP=2` reaches **`299.3 tok/s`** Plain (`10,291 tok/s` prefill) and **`643.2–672.0 tok/s`** `MTP-3`.

### 2.4 Multi-Client Intact `Open-SWE-Traces` Serving & 1-Hour Soak (`open_swe_intact_16k.json`)

- **Intact `Open-SWE-Traces` Multi-Client Campaigns (`swe-c4-shared` / `swe-c4-unique` / `swe-c8-burst`, `0` errors):**
  - **`Qwen3.8-27B` (`27b-swe-multiclient.toml`)**:
    - `TP=1` (`1× 3090 Ti`): **`101.04 / 96.49 / 110.44 tok/s`** (`MTP-3`, `88.5%` cache hit, **`3.34–4.82×`** `vLLM` `28.16 / 24.52 / 22.90` & `SGLang` `30.21 / 26.73 / 23.38`, beating 2-GPU `SGLang TP=2` plain on 1 GPU) and **`88.16 / 89.62 / 83.04 tok/s`** (`DFlash2`).
    - `TP=2` (`2× 3090 Ti`): **`144.45 / 142.62 / 180.68 tok/s`** (`MTP-3`) and **`145.45 / 129.99 / 174.95 tok/s`** (`DFlash2`, **`2.21–3.31×`** `SGLang TP=2 DFlash2` `61.14 / 58.70 / 52.90`).
  - **`Qwen3.6-35B-A3B` (`35b-swe-multiclient.toml`)**:
    - `TP=1` (`1× 3090 Ti`): **`341.89 / 349.89 / 353.78 tok/s`** (`MTP-3`, `1.8–1.9×` faster TTFT vs `vLLM` `244.92 / 232.87 / 350.52`; `SGLang` fails at startup).
    - `TP=2` (`2× 3090 Ti`): **`400.45 / 424.86 / 473.62 tok/s`** (`MTP-3`, **`1.23–1.49×`** `vLLM` `273.97 / 285.26 / 385.41`).
  - **`Qwen3.8-Flash-Next` (`flash-next-swe-multiclient.toml`)**:
    - `TP=2` (`2× 3090 Ti`, `MTP-3`): **`56.34` (`smoke`) / `40.72` / `42.35` / `42.93 tok/s`** (`86.9–88.9%` hit), beating `Strata` (`34.55 / 40.66 / 40.97 / 29.65 tok/s`) across all 4 workloads (`+44.8%` on `swe-c8-burst`) at 100% lossless verification.
    - `TP=1` (`1× 3090 Ti`): **`30.45 / 28.61 / 28.83 / 28.23 tok/s`** (`MTP-3`) and **`28.13 / 28.47 / 29.71 / 28.75 tok/s`** (`DFlash2`), beating `Strata TP=1` on `swe-c8-burst` (`17.83 tok/s`) by **`+58.3%` to `+61.2%`**.
- **1-Hour Intact `Open-SWE` Soak (`soak-swe-27b.toml`, `1× RTX 3090 Ti`, `27B W4A16-HGPTQ`, `MTP-3`, `TQ4`, `4` sustained / `8` burst clients):**
  - Completed **`3,675.22 s`** with **`2,920` completed responses + `153` injected local disconnects** across `326` multi-turn trajectories (`16,221,154` prompt tokens, `365,786` completion tokens = **`99.53 tok/s` sustained over 1 hour on 1 GPU**), **`0` unexpected errors / invalid usage / SLO breaches**, final post-drain inference probe `"ok"`, GPU VRAM delta **`+8.0 MiB`** (`22,719 → 22,727 MiB`, budget `<= 256 MiB`), and host RSS delta **`-760.10 MiB`** post-warmup (`4,510 → 3,750 MiB`).
- **Live `SWE-bench Verified` (`0:10`) Agentic Coding & Thinking-Effort Study (`docs/SWEBENCH.md`, `benchmarks/swebench/run_swebench.py`, `2× RTX 3090 Ti`, `TP=2`, `MTP-3`, omitted `max_tokens`):**
  - **`Qwen3.8-27B-W4A16-HGPTQ`**: **`6 / 10 (60.0%)` RESOLVED** in **`7m 33s`** (`200` turns, `84,797` output tokens, `50,735` thinking tokens, `94.74%` prefix cache hit, `0` patch/format errors) under default `medium` (`16,384` thinking ceiling), and **`9 / 10 (90.0%)` RESOLVED** in **`47m 40s`** (`507` turns up to `118,161` context tokens including `111` turns above `65,536` tokens across 4 concurrent workers up to `245,163` live context tokens, `381,981` output tokens, **`98.37%` exact prefix cache hit rate** across `20,929,445` prompt tokens with `482 / 507` exact `"hit"` turns and `0` multi-turn `"no_checkpoint"` cascades, **`121,307` `MTP-3` verify rounds** with `185,493 / 250,452 = 74.06%` draft tokens accepted) under `--reasoning-effort max` with L2 `HostKvSwapPool` + demand-sized per-slot `MTP-3` page partitions (`reconfigure_batch_slot_pages` + `adapt_mtp_batch_slot_layout`).
  - **`Qwen3.6-35B-A3B-W4A16-HGPTQ`**: **`6 / 10 (60.0%)` RESOLVED** in **`11m 31s`** (`416` turns, `174,760` output tokens, `96,857` thinking tokens, `96.61%` prefix cache hit, `0` format errors) under default `medium`, vs `4 / 10` (`5 / 10` normalized `-p1`, `16m 06s`, `422` turns, `230,381` output tokens) under `--reasoning-effort max`.
  - Across all **`1,494` turns** in the 4 runs, natural thinking peaked at `8,826` tokens (`p99 = 2,634..4,886`) with **`0` false `thinking_budget_exhausted` cutoffs**.

---

## 3. Hardware & Subsystem Qualification Matrix

| Target Hardware | Architecture & Topology | Verified `warp34 bench` / Campaign Highlights (`TQ4` KV, `100%` Lossless Speculative Parity) |
| --- | --- | --- |
| **1–2× A100-SXM4-80GB** | `sm_80`, 108 SMs, 40 MiB L2, 2.04 TB/s HBM2e, NVLink P2P | **27B `TP=1`**: `63.5 tok/s` Plain / `150.0` `MTP-3` / **`199.3 tok/s` `DFlash2`** (`4,108 tok/s` prefill). **27B `TP=2` NVLink**: `83.1 tok/s` Plain / `200.8` `MTP-3` / **`241.3 tok/s` `DFlash2`** (`6,710 tok/s` prefill). **4B `TP=1`**: `146.1` Plain / `428.9` `MTP-3` / **`973.0 tok/s` `DFlash2`**. |
| **4× RTX 3090 Dual-NVLink** | `sm_86`, 82 SMs, `NV4` P2P (`0<->3`, `1<->2`) + `TP=4 / EP=4` | **27B `TP=2` NVLink P2P**: `91.3 tok/s` Plain (`+60.4%` TTFT vs PCIe) / `154.6` `MTP-3` / **`197.0 tok/s` `DFlash2`**. **27B `TP=4`**: `129.1 tok/s` Plain / `291.0` `MTP-3` / **`438.3 tok/s` `DFlash2`** (`2.28 ms/tok`). **Flash-Next `TP=4 / EP=4`** (100% VRAM-resident 512 experts): `90.6 tok/s` Plain / **`147.2 tok/s` `MTP-3`** (`2,741 tok/s` prefill). |
| **1–2× RTX 4090 (24 GB)** | `sm_89`, 128 SMs, 72 MiB L2, 1,008 GB/s, PCIe 4.0 x16 | **4B**: `TP=1` `222.2 / 566.8 / 1,409.3 tok/s`, `TP=2` `325.8 / 765.9 /` **`1,645.5 tok/s`** (`Plain / MTP-3 / DFlash2`). **27B**: `TP=1` `53.5 / 162.0 / 258.6 tok/s`, `TP=2` `88.5 / 230.1 / 333.6 tok/s`. **35B-A3B**: `TP=1` `225.5 / 501.9 tok/s`, `TP=2` `299.3 / 651.2 tok/s`. |
| **1–2× RTX 3090 Ti (24 GB)** | `sm_86`, 84 SMs, 6 MiB L2, 1,008 GB/s, PCIe 4.0 x8 | **2B**: `TP=1` `343.2 / 851.6 tok/s`, `TP=2` `478.8 /` **`1,131.3 tok/s`**. **4B**: `TP=1` `211.5 / 534.2 / 1,288.0 tok/s`, `TP=2` `305.5 / 766.0 /` **`1,488.2 tok/s`**. **9B**: `TP=1` `150.2 / 356.9 tok/s`, `TP=2` `213.3 / 545.9 tok/s`. |
| **1–2× RTX 4060 Ti (16 GB)** | `sm_89`, 34 SMs, 32 MiB L2, 288 GB/s | **2B**: `TP=1` `138.6 / 347.7`, `TP=2` `240.5 / 591.9 tok/s`. **4B**: `TP=1` `77.1 / 191.7 / 487.6`, `TP=2` `137.0 / 335.7 /` **`767.3 tok/s`**. **27B `TP=2`**: `31.1 / 85.1 / 111.9 tok/s` (`TP=1 RCO`: `48.5 tok/s`). **35B-A3B**: `TP=1 RCO` `190.8 tok/s`, `TP=2 W4` `154.4 / 331.2 tok/s`. **Flash-Next**: `TP=1 W4` `25.2 / 34.7`, `TP=2 W4` `30.7 / 43.2 tok/s`. |
| **1–2× RTX 3060 Ti (8 GB)** | `sm_86`, 38 SMs, 3 MiB L2, 448 GB/s | **2B**: `TP=1` `117.3 / 240.3`, `TP=2` `161.5 / 279.7 tok/s`. **4B**: `TP=1` `85.4 / 182.1 / 378.8`, `TP=2` `123.8 / 212.1 /` **`411.8 tok/s`**. **9B**: `TP=1 W4` (mapped-host `embed_tokens`) `63.7 / 121.5 tok/s` (`TP=1 RCO`: **`151.5 tok/s`**), `TP=2 W4` `100.4 / 167.0 tok/s`. **Flash-Next `TP=2`** (`2× 8 GB` compact `k_hot=16`): `21.1 / 21.8 tok/s`. |

### Additional Qualified Subsystems

- **Native 2-Bit (`W2A16` / `kU2B2`) & Mixed `W2.6A16-HGPTQ-RCO` (`op_marlin.cu`, `op_moe.cu`, `moe_cpu_bridge/eval.rs`):**
  - 0-ULP bit-for-bit parity against dilated `U4B8` across Marlin, all 5 GPU MoE regimes, and AVX-512/AVX2/scalar CPU bridges.
  - **`Qwen3.5-9B-W2.6A16-HGPTQ-RCO`**: `128/128` greedy parity vs `vLLM`; `TP=1` `175.0 tok/s` Plain / **`389.1 tok/s` `MTP-3`** (`6.77 GiB` weights, fits on 8 GB GPUs); `TP=2` `250.5 tok/s` Plain / **`584.2 tok/s` `MTP-3`**.
  - **`Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO`**: `-6.00 GiB` VRAM (`19.22 → 13.21 GiB` on `TP=1`, fitting on 16 GB GPUs); `TP=1` `223.4 tok/s` Plain / **`435.7 tok/s` `MTP-3`**; `TP=2` `264.2 tok/s` Plain / **`565.8 tok/s` `MTP-3`**.
  - **`Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO`**: `-36.0%` MoE expert footprint (`246,272 B` vs `385,024 B`/expert), `-3.40 GiB` pinned host RAM (`9.45 → 6.05 GiB`); `warp34 bench` decode reaches **`82.3 tok/s`** (`TP=1 DFlash2`) and **`237.7 tok/s` (`291.4 tok/s` warm radix)** (`TP=2 DFlash2`).
- **Decision Models (`JevBench` `128` Public + `64` Multi-Question Items, `0` Errors on RTX 3090 Ti):**
  - **`d3-edge-W4A16-HGPTQ` (`TP=1`, 12L)**: Public **`90/128 (70.31%)`** at **`149.9 dec/s`** (`c=1`, `p50 = 3.91 ms`) / **`232.1 dec/s`** (`c=16`); MultiQ **`46/64 (71.88%)`** at **`399.0 dec/s`** (`c=8`).
  - **`d3-lite-W4A16-HGPTQ` (`TP=1`, 24L)**: Public **`101/128 (78.91%)`** at **`77.4 dec/s`** (`c=1`) / **`115.6 dec/s`** (`c=16`); MultiQ **`58/64 (90.62%)`** at **`190.7 dec/s`** (`c=8`).
  - **`GLiDE-0.8B-W4A16-HGPTQ` (`TP=1`, 24L)**: Public **`113/128 (88.28%)`** at **`82.3 dec/s`** (`c=1`) / **`125.1 dec/s`** (`c=16`); MultiQ **`63/64 (98.44%)`** at **`213.8 dec/s`** (`c=8`).
  - **`GLiDE-9B-W4A16-HGPTQ` (`TP=1`, 32L + `midloop 2×`)**: Public **`124/128 (96.88%)`** at **`9.0 dec/s`** (`c=1`) / **`9.8 dec/s`** (`c=16`); MultiQ **`64/64 (100.00%)`** at **`15.7 dec/s`** (`c=8`).
  - **`d3-ultra-W4A16-HGPTQ` (`TP=2` & `TP=1`, 64L 27B + `prefix_bidir`)**: Public **`122/128 (95.31%)`** at **`3.5 dec/s`** (`c=1`) / **`3.9 dec/s`** (`c=16`); MultiQ **`64/64 (100.00%)`** at **`3.6 dec/s`** (`c=8`).
- **Runtime Multi-Adapter LoRA (`Qwen3.5-9B` + Rank-16 Adapters on RTX 3090 Ti):**
  - `TP=1` (`MTP-3` + LoRA) **`262.17 tok/s`** vs `vLLM` (`MTP-3` + LoRA) `145.65 tok/s` (**`1.80×`**) and `SGLang` (Plain + LoRA) `76.02 tok/s` (**`3.45×`**); `TP=2` (`MTP-3` + LoRA) **`401.18 tok/s`** vs `vLLM` `126.90 tok/s` (**`3.16×`**) and `SGLang` `106.79 tok/s` (**`3.76×`**), with `max_abs_diff <= 1.25e-2` tensor parity and 100% greedy token match against HF PEFT.
- **Image Generation & Editing (`Qwen-Image-2.1` & `Qwen-Image-2.1-Turbo`, 8 steps vs `diffusers` `BF16`):**
  - **1–2× A100 80GB (`sm_80`)**: `FP16` `512×512` **`0.72 s`** (`TP=1`) / **`0.66 s`** (`TP=2`, **`1.41×`** vs `diffusers` `0.93 s`, `48.6 dB` PSNR vs `diffusers`, `57.8 dB` `TP=1 vs TP=2`); `1024×1024` **`2.91 s`** (`TP=1`) / **`2.66 s`** (`TP=2`, **`1.30×`** vs `3.46 s`, `56.6 dB` `TP=1 vs TP=2`); `W4A16-HGPTQ` (`9.82 GiB`) `512×512` **`0.79 s / 0.71 s`**, `1024×1024` **`3.32 s / 2.85 s`**.
  - **1–2× RTX 4090 (`sm_89`)**: `W4A16-HGPTQ` `TP=1` (1 GPU vs 2-GPU `diffusers`) `512×512` **`0.83 s`** (**`1.25×`** vs `1.05 s`, `52.4 dB` `TP=1 vs TP=2`), `1024×1024` **`3.68 s`** (**`1.23×`** vs `4.54 s`, `52.2 dB` `TP=1 vs TP=2`); `FP16` (2× 4090) `512×512` **`0.88 s`** (`52.0–54.3 dB` vs `diffusers`), `1024×1024` **`3.73 s`** (`55.4 dB` `TP=1 vs TP=2`).
  - **1–2× RTX 3090 Ti (`sm_86`)**: `W4A16-HGPTQ` (with automatic `W4A8` INT8 IMMA activation quantization on `sm_86`/`sm_89`) `512×512` **`0.98 s`** (`1.02 img/s`, `TP=1`, **`14.47×`** vs `diffusers` CPU-offloaded `14.18 s`, **`1.47×`** faster than `FP16 TP=2` on 1 card) / **`1.01 s`** (`TP=2`), `1024×1024` **`4.47 s`** (`TP=1`, **`4.66×`** vs `20.82 s`, **`1.31×`** faster than `FP16 TP=2` on 1 card) / **`4.19 s`** (`TP=2`, **`4.96×`** vs `20.82 s`); `FP16` `512×512` **`1.44 s`** (`TP=2`, `50.1 dB`), `1024×1024` **`5.84 s`** (`TP=2`, `48.8 dB`), `1024×1024` RGBA transparent **`37.23 dB`** T2I / **`42.70 dB`** edit (`0` `NaN`s), `Qwen-Image-2.1` (`CFG=4.0`) `512×512` **`3.42 s`** (**`5.04×`** vs `17.23 s`).
- **Upstream `lm-eval==0.4.13` Quality & Speed Verification (`ifeval` + `gsm8k`, `--limit 50`, `FP16` KV):**
  - **`27B-W4A16-HGPTQ` (`TP=1`, `MTP-3`)**: `ifeval` **`0.9000`**, `gsm8k` **`0.9200`** in **`249.3 s`** (`2.00×` faster than `SGLang` `497.8 s`, `2.04×` faster than `vLLM` `507.4 s`, `2.30×` faster than `llama.cpp` `574.3 s`).
  - **`35B-A3B-W4A16-HGPTQ` (`TP=2`, `MTP-3`)**: `ifeval` **`0.9000`**, `gsm8k` **`0.9000`** in **`84.1 s`** (`1.54×` faster than `vLLM` `129.1 s`).
  - **`Flash-Next-W4A16-HGPTQ` (`TP=2`, `MTP-3`)**: `ifeval` **`0.9200`**, `gsm8k` **`0.9400`** in **`379.0 s`** (`1.23×` faster than `Strata` `467.4 s`, `2.97×` faster than `qfn-3090` `1,127.4 s`).

---

## 4. Key Architectural Invariants & Verification Gates

- **Tiered-MoE (`Qwen3.8-Flash-Next`) Zero-Sync Decode & Warm-Prefix Adaptation (`moe_offload.rs`, `moe_offload/engine.rs`, `model/weights/moe.rs`):**
  - When `k_hot >= TIERED_MOE_DYNAMIC_LRU_MIN_HOT_EXPERTS` (`64`), `HotExpertRankings::plan_layer_capacities` activates TinyLFU `k_static = 1` (`k_hot - 1` dynamic GPU LRU slots) with overlapped shared-expert GPU evaluation while the CPU bridge evaluates cold misses over the half-bandwidth `FP16` pinned mailbox (`TIERED_MOE_CPU_BRIDGE_MAX_DECODE_TOKENS = 8`).
  - Suffixes of `9..=40` tokens execute as zero-sync 8-token sliced device-bridge launches (with slice-0 shared-expert overlap), `32..1024`-token suffixes enable `dynamic_short_adapt` to promote newly hot experts during `StreamedPrefill`, and `>= 1024`-token prompts promote top-frequency cold experts on Chunk 0 so subsequent chunks and decode run against an already-warmed `HotExpertCache`.
  - `M=1` (`Plain`), `M=4` (`MTP-3`), and `M=8` (`DFlash2`) share a 0-ULP M-invariant reduction tree across QSA (`target_splits = 8` for `num_tokens <= 16`), routed MoE (`moe_sparse_vec4_kernel` through `num_tokens <= 8` for `group_size == 64 && top_k >= 10`), and unquantized `DenseF16` small-N projections (`warp34_small_n_gemm_f16` through `N <= 10240`).
- **CUDA 13 `ptxas` Alignment & Multi-Arch Fatbinary (`crates/warp34-kernels/build.rs`, `tq4_kv_common.cuh`):**
  - Default `WARP34_ARCH=sm_80,sm_86,sm_89` builds native SASS cubins for all 3 architectures plus PTX with `--compress-mode=size` and shared `cuBLAS`/`cuBLASLt` runtime linkage (`Dockerfile` copies only `libcudart.so.13`, `libcublas.so.13`, and `libcublasLt.so.13` into `ubuntu:26.04`).
  - `block_tq4_0` and `qsa_block_tq4_0` enforce `static_assert(sizeof == 66 && alignof == 2)` without `#pragma pack(push, 1)` so CUDA 13.0 `ptxas` emits aligned 16-bit `STG.E.U16` / `LDG.E.U16` scale accesses instead of miscompiling 1-byte `__half` stores.
- **Continuous Release Gate (`mise run check`):**
  - `check-quick` (`fmt-check`, `clippy`, `clippy-cpu`, `doc-check` with `RUSTDOCFLAGS='-D warnings'`, `simd-check`, `cuda-bin-check` static `STACK == 0 && LOCAL == 0` audit across `sm_80`/`sm_86`/`sm_89`), `dashboard-check` (Vitest + TypeScript + Prettier), `oracle-check`, `perf-check` (manifest & campaign coverage validation across `suite.toml` and `{27b,35b,flash-next}-swe-multiclient.toml`), full GPU unit + 44 integration test targets (`cargo test --release --features cuda`), and `python3 tools/check_cuda_sanitizer.py` (`compute-sanitizer` `memcheck` + `racecheck` + `synccheck` across all 6 CUDA subsystems and `TP=2`).
