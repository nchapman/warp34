# Benchmarks & Performance

Detailed benchmark results, competitor head-to-heads, and methodology for `warp34`. For raw campaign manifests and runner configs, see [`benchmarks/perf/README.md`](../benchmarks/perf/README.md) and [`benchmarks/perf/campaigns/standings.md`](../benchmarks/perf/campaigns/standings.md).

---

## Head-to-head: `Qwen3.5-9B` (`MTP-3`) vs. `vLLM`, `SGLang`, and `llama.cpp`

End-to-end HTTP serving p50 (`tok/s`, `FP16` KV). We benchmark `9B` here because `vLLM` and `SGLang` OOM trying to fit `Qwen3.8-27B` with full context, Vision, and `MTP-3` on a single 24 GB GPU.

### `1× RTX 3090 Ti` (`11/11` clean sweep)

| Workload | **`warp34`** | `vLLM 0.34.0` | `SGLang v0.5.23` | `llama.cpp` | `warp34` lead |
|---|---:|---:|---:|---:|---:|
| **Cold 4K prefill** | **`8,223`** | `4,801` | `4,539` | `4,892` | **`+68%–81%`** |
| **Cold 115K prefill** | **`4,097`** | `2,875` | `2,885` | `3,124` | **`+31%–43%`** |
| **Single-stream decode (`C=1`)** | **`233.5`** | `180.6` | `179.2` | `194.4` | **`+20%–30%`** |
| **Batched decode (`C=16`)** | **`1,346.4`** | `1,276.8` | `881.5` | *crash* | **`+5%–53%`** |
| **Coding agent (`64K` ctx)** | **`42.3`** | `22.9` | `27.2` | `31.8` | **`+33%–85%`** |
| **Coding agent (`120K` ctx)** | **`19.5`** | `11.2` | `13.2` | `15.1` | **`+29%–74%`** |

### `2× RTX 3090 Ti` PCIe (`11/11` clean sweep)

| Workload | **`warp34`** | `vLLM 0.34.0` | `SGLang v0.5.23` | `llama.cpp` | `warp34` lead |
|---|---:|---:|---:|---:|---:|
| **Cold 4K prefill** | **`8,408`** | `5,282` | `5,044` | `4,169` | **`+59%–102%`** |
| **Cold 115K prefill** | **`5,492`** | `3,902` | `3,742` | `3,119` | **`+41%–76%`** |
| **Single-stream decode (`C=1`)** | **`338.6`** | `232.6` | `251.4` | `217.0` | **`+35%–56%`** |
| **Batched decode (`C=16`)** | **`1,675.8`** | `1,354.0` | `1,264.5` | *crash* | **`+24%–33%`** |
| **Coding agent (`64K` ctx)** | **`49.8`** | `27.5` | `34.1` | `30.5` | **`+46%–81%`** |
| **Coding agent (`120K` ctx)** | **`26.4`** | `14.2` | `17.2` | `15.2` | **`+53%–86%`** |

---

## Head-to-head vs. specialized consumer-GPU engines (`1–2× RTX 3090 Ti` & `1–2× RTX 4090`)

Each row pits `warp34` (`W4A16-HGPTQ`) against a dedicated single-model engine on the exact model, GPU count, and speculative mode it was built for (end-to-end HTTP serving p50, `tok/s`):

| Engine *(upstream HEAD)* | Specialty | Workload | Competitor | **`warp34`** | `warp34` lead |
|---|---|---|---:|---:|---:|
| **[`ninfer-3090`](https://github.com/Don-Chad/ninfer-3090)** (`115dee6`) | `27B` W4 on `1× 3090 Ti` *(MTP)* | `C=1` / `C=8` / agent / primed | `94.1` / `260.3` / `69.2` / `39.8` | **`109.1` / `417.2` / `96.0` / `109.6`** | **`+16%` – `2.75×`** |
| **[`qwen38-flash-next-3090`](https://github.com/DominikBucko/qwen38-flash-next-3090)** (`10b24c2`) | `180B Flash-Next` W4 on `1× 3090 Ti` *(MTP)* | chat / `6K` code / primed | `30.6` / `21.2` / `10.2` | **`53.7` / `26.3` / `49.6`** | **`+24%` – `4.86×`** |
| **[`qwen38-flash-next-2x3090`](https://github.com/DominikBucko/qwen38-flash-next-2x3090)** (`4dca58f`) | `180B Flash-Next` W4 on `2× 3090 Ti` *(MTP)* | chat / `6K` code / primed | `55.6` / `28.3` / `20.5` | **`99.4` / `39.1` / `93.1`** | **`+38%` – `4.54×`** |
| **[`Strata v0.2.17`](https://github.com/Niko1221/Strata)** (`3142b66`) | `180B Flash-Next` W4 on `1× 3090 Ti` *(DFlash2)* | chat / prefill / primed / agent | `80.9` / `23.6` / **`82.9`** / **`64.7`** | **`83.9`** / **`32.1`** / `74.0` / `58.4` | **`+4%` chat, `+36%` prefill; `-10%` primed/agent** |
| **[`Strata v0.2.17`](https://github.com/Niko1221/Strata)** (`3142b66`) | `180B Flash-Next` W4 on `1× 4090` *(DFlash2)* | chat / prefill / primed / agent | `118.0` / `40.3` / `109.4` / `98.0` | **`120.0` / `49.0` / `118.6` / `101.1`** | **`+2%` – `+22%`** |
| **[`Strata v0.2.17`](https://github.com/Niko1221/Strata)** (`3142b66`) | `180B Flash-Next` W4 on `2× 3090 Ti` *(DFlash2)* | chat / prefill / primed / agent | `120.9` / `37.1` / `112.8` / `96.1` | **`140.1` / `51.0` / `123.5` / `123.6`** | **`+10%` – `+38%`** |
| **[`Strata v0.2.17`](https://github.com/Niko1221/Strata)** (`3142b66`) | `180B Flash-Next` W4 on `2× 4090` *(DFlash2)* | chat / prefill / primed / agent | `132.7` / `51.1` / `113.1` / `112.6` | **`179.6` / `68.5` / `178.7` / `184.7`** | **`+34%` – `+64%`** |

### Competitor configuration & `W2.6-RCO` notes

- **`ninfer-3090`**: Runs its mixed Q4/Q5 `.ninfer` checkpoint (`MTP-2`, `32K` KV) vs `warp34` `W4A16-HGPTQ` (`MTP-3`, `32K` `TQ4` KV).
- **`qwen38-flash-next-{3090,2x3090}`**: Both engines run 4-bit `Flash-Next` with `MTP-3`.
- **`Strata v0.2.17`**: Runs its out-of-the-box `UD-IQ4_XS` (~3.5 bpw) + `DFlash2` + `32K` streamed KV vs `warp34` 4-bit `W4A16-HGPTQ` (`131K` resident `TQ4` KV, `DFlash2`).
- **`warp34` `W2.6A16-HGPTQ-RCO` + `DFlash2` on `Flash-Next`** (chat / prefill / primed / agent, `tok/s`):
  - `1× 3090 Ti`: `100.6 / 40.0 / 99.6 / 92.9`
  - `1× 4090`: `139.1 / 50.0 / 135.5 / 118.4`
  - `2× 3090 Ti`: `167.0 / 73.1 / 170.3 / 174.3`
  - `2× 4090`: `201.3 / 77.6 / 189.6 / 200.4`

---

## Single-stream decode across Q4 models (`W4A16-HGPTQ` + `MTP-3`)

| Model | 1× RTX 3090 Ti | 1× RTX 4090 | 2× RTX 3090 Ti | 2× RTX 4090 |
|---|---:|---:|---:|---:|
| **`Qwen3.5-9B`** | `234 tok/s` | `253 tok/s` | `339 tok/s` | `369 tok/s` |
| **`Qwen3.8-27B`** | `150 tok/s` | `162 tok/s` | `252 tok/s` | `246 tok/s` |
| **`Qwen3.6-35B-A3B`** | `429 tok/s` | `512 tok/s` | `553 tok/s` | `672 tok/s` |
| **`Qwen3.8-Flash-Next`** *(180B MoE)* | `54 tok/s` *(DFlash2 `84`)* | `58 tok/s` *(DFlash2 `120`)* | `99 tok/s` *(DFlash2 `140`)* | `117 tok/s` *(DFlash2 `180`)* |

- `9B` and `Flash-Next` are end-to-end HTTP serving p50s.
- `27B` and `35B-A3B` are `warp34 bench` peak acceptance; realistic coding prompts achieve `96–113 tok/s` on `1× 3090 Ti` and `156–184 tok/s` on `2×` GPUs for `27B MTP-3`.

---

## `W2.6A16-HGPTQ-RCO` vs. `W4A16-HGPTQ` (non-speculative `warp34 bench`)

Measured head-to-head (`--stage decode`, `prompt=128`, `gen=128`; prefill also shown at `prompt=512`):

| Model / GPU | `W4A16-HGPTQ` | `W2.6A16-HGPTQ-RCO` | `W2.6` vs. `W4` |
|---|---:|---:|---:|
| **`27B` `1× 3090 Ti` Decode** | `53.8 tok/s` | **`71.6 tok/s`** | **`+33%`** |
| **`27B` `2× 3090 Ti` Decode** | `90.4 tok/s` | **`107.6 tok/s`** | **`+19%`** |
| **`27B` `1× 3090 Ti` Prefill (128 / 512)** | `2,279 / 2,742 tok/s` | **`2,379 / 2,795 tok/s`** | **`+2%–4%`** |
| **`27B` `1× 4090` Decode / Prefill (512)** | — | **`75.6 tok/s`** / **`5,134 tok/s`** | `91.5%` HBM roofline |
| **`27B` `2× 4090` Decode** | — | **`115.5 tok/s`** | `8.66 ms/tok` |
| **`35B-A3B` `1× 3090 Ti` / `1× 4090` Decode** | — | **`243.5`** / **`283.7 tok/s`** | `5,488` / `8,653` prefill |
| **`9B` `1× 3090 Ti` / `1× 4090` / `2× 4090` Decode** | — | **`185.1`** / **`207.2`** / **`290.4 tok/s`** | `7,061` / `11,736` / `7,310` prefill |

---

## Budget & mid-range GPUs (8–16 GB)

Single-stream `MTP-3` decode measurements with `128K` `tq4` KV and zero manual flags (see [`docs/STATE.md`](STATE.md) for full hardware matrix):

| GPU | Model | Footprint | Decode (`MTP-3`) |
|---|---|---:|---:|
| **RTX 3060 Ti (8 GB)** | `Qwen3.5-9B-W2.6A16-HGPTQ-RCO` | `6.0 GiB` | `151.5 tok/s` |
| **RTX 4060 Ti (16 GB)** | `Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO` | `14.6 GiB` | `190.8 tok/s` |
| **RTX 4060 Ti (16 GB)** | `Qwen3.8-27B-W2.6A16-HGPTQ-RCO` | `14.6 GiB` | `48.5 tok/s` |
