# SWE-bench Verified Agentic Evaluation & Thinking-Effort Study

This document records the methodology, tooling (`benchmarks/swebench/run_swebench.py`),
and empirical findings from evaluating `warp34` on **SWE-bench Verified**
(`princeton-nlp/SWE-Bench_Verified`, `test` split, slice `0:10`) across
**`Qwen3.6-35B-A3B-W4A16-HGPTQ`** (MoE) and **`Qwen3.8-27B-W4A16-HGPTQ`** (dense)
under **`medium` (default)** vs **`max` (`--reasoning-effort max`)** thinking
configurations on `2× NVIDIA GeForce RTX 3090 Ti` (`TP=2`, `MTP-3`, `TQ4` KV).

---

## 1. Evaluation Setup & Invariants

1. **Agent Harness (`mini-swe-agent v2.4.6`):**
   - Invoked via `uvx --from mini-swe-agent mini-extra swebench --subset verified --split test --slice 0:10` with the standard `swebench.yaml` prompt/tool template (`bash` tool inside isolated per-instance Docker containers `docker.io/swebench/sweb.eval.x86_64.*:latest`).
   - **`max_tokens` is deliberately omitted** on all client chat requests to match out-of-the-box coding-agent defaults and exercise `warp34`'s automatic OpenRouter-style thinking-effort budget (`crates/warp34-host/src/server.rs:34-90`).
2. **Thinking Effort & Hard Cutoff Semantics:**
   - **`medium` (default):** `chat_template.jinja` (`qwen3.8-froggeric-v22.5`) renders `effort = 'medium'`; when `max_tokens` is omitted, `warp34` caps thinking at `50% × 32,768 = 16,384` reasoning tokens per turn before injecting `"I've hit my thinking limit. I'll move forward with the solution now.\n</think>\n"`.
   - **`max` (`--reasoning-effort max`):** `chat_template.jinja` renders `effort = 'xhigh'`; when `max_tokens` is omitted, `warp34` caps thinking at `95% × 32,768 = 31,129` reasoning tokens per turn.
3. **Deterministic Grading (`swebench==3.0.17`):**
   - No LLM judge is used. `python3 -m swebench.harness.run_evaluation` applies each generated patch inside the pinned historical repository container alongside the upstream PR's ground-truth `test_patch` and runs `pytest`.
   - An instance is marked **RESOLVED** if and only if every `FAIL_TO_PASS` test passes and every `PASS_TO_PASS` regression test continues to pass (`returncode == 0`).

---

## 2. 4-Way Summary Comparison (`35B-A3B` vs `27B` × `medium` vs `max`)

| Metric | `Qwen3.6-35B-A3B` `medium` | `Qwen3.6-35B-A3B` `max` | `Qwen3.8-27B` `medium` | `Qwen3.8-27B` `max` |
| --- | ---: | ---: | ---: | ---: |
| **Chat Template Effort (`chat_template.jinja`)** | `medium` | `xhigh` | `medium` | `xhigh` |
| **Hard Thinking Ceiling / Turn** | `16,384` | `31,129` | `16,384` | `31,129` |
| **SWE-bench Verified Resolved (raw `preds.json`)** | `6 / 10 (60%)` | `4 / 10 (40%)` | `6 / 10 (60%)` | **`8 / 10 (80%)`** |
| **SWE-bench Verified Resolved (normalized `-p1`)** | `6 / 10 (60%)` | `5 / 10 (50%)`* | `6 / 10 (60%)` | **`8 / 10 (80%)`** |
| **Patch Apply Errors (raw `preds.json`)** | `2` | `2` | **`0`** | **`0`** |
| **Total Agent Turns (across 10 tasks)** | `416` | `422` | **`200`** | `456` |
| **Total Output Tokens** | `174,760` | `230,381` | **`84,797`** | `322,306` |
| **Total Thinking Tokens (`%` of output)** | `96,857` (`55.4%`) | `142,778` (`62.0%`) | **`50,735` (`59.8%`)** | `226,249` (`70.2%`) |
| **Thinking Tokens / Turn (`p50` / `p90` / `p99` / `max`)** | `48` / `629` / `3,611` / `4,901` | `45` / `996` / `3,498` / `7,294` | `45` / `882` / `2,634` / `3,183` | `119` / `1,402` / `4,886` / `8,826` |
| **`thinking_budget_exhausted` Cutoffs** | `0` | `0` | `0` | `0` |
| **Tool-Call Format Errors** | **`0`** | `1` | **`0`** | `1` |
| **Cumulative Prompt Tokens (`MambaRadixCache` hit `%`)** | `9.72M` (`96.61%`) | `12.49M` (`96.66%`) | **`2.66M` (`94.74%`)** | `19.11M` (`95.01%`) |
| **Per-Stream Decode Rate (`p50` / `mean` / `p90`)** | `99.3` / `105.7` / `124.1 tok/s` | `99.4` / `99.6` / `118.1 tok/s` | `58.6` / `66.5` / `93.7 tok/s` | `56.4` / `61.8` / `89.2 tok/s` |
| **Wall-Clock Time (10 tasks)** | `11m 31s` (`w=4`) | `16m 06s` (`w=4`) | **`7m 33s` (`w=4`)** | `~55m` (`w=2`, `384k` KV) |

*\*In `35B-A3B max`, `astropy__astropy-13579` produced a valid fix that passed the upstream test suite once its `diff -u file.bak file` (`-p0`) header was normalized to `--- a/... +++ b/...` (`-p1`).*

---

## 3. Per-Instance Resolution Matrix (`0:10`)

| Instance ID | `35B-A3B` `medium` | `35B-A3B` `max` | `27B` `medium` | `27B` `max` | Subsystem / Bug Summary |
| --- | :---: | :---: | :---: | :---: | --- |
| `astropy__astropy-12907` | **PASS** (21t) | **PASS** (41t) | **PASS** (9t) | **PASS** (17t) | `separability_matrix` nested `CompoundModel` |
| `astropy__astropy-13033` | FAIL (33t) | FAIL (22t) | FAIL (21t) | FAIL (17t) | `TimeSeries` required-column `ValueError` wording |
| `astropy__astropy-13236` | **PASS** (53t) | **PASS** (73t) | FAIL (8t) | **PASS** (49t) | `Table` structured `ndarray` auto-`NdarrayMixin` deprecation |
| `astropy__astropy-13398` | FAIL (64t) | FAIL (41t) | FAIL (41t) | **PASS (96t)** | Direct `ITRS <-> CIRS/AltAz` topocentric frame transformation |
| `astropy__astropy-13453` | **PASS** (64t) | **PASS** (45t) | **PASS** (28t) | **PASS** (68t) | HTML table writer `formats=` argument forwarding |
| `astropy__astropy-13579` | **PASS** (41t) | **PASS*** (56t) | **PASS** (19t) | **PASS** (45t) | `SlicedLowLevelWCS.world_to_pixel_values` coupled dimensions |
| `astropy__astropy-13977` | FAIL (38t) | FAIL (26t) | FAIL (28t) | FAIL (62t) | `Quantity.__array_ufunc__` returning `NotImplemented` on duck arrays |
| `astropy__astropy-14096` | **PASS** (30t) | FAIL (35t) | **PASS** (19t) | **PASS** (32t) | Subclassed `SkyCoord` custom property `AttributeError` propagation |
| `astropy__astropy-14182` | FAIL (56t) | FAIL (72t) | **PASS** (15t) | **PASS** (40t) | `ascii.rst` writer `header_rows` multi-line support |
| `astropy__astropy-14309` | **PASS** (16t) | **PASS** (11t) | **PASS** (12t) | **PASS** (30t) | `io.fits.connect.is_fits` empty `args` `IndexError` guard |
| **Total Resolved** | **`6 / 10`** | **`4–5 / 10`** | **`6 / 10`** | **`8 / 10`** | |

---

## 4. Key Findings

### 4.1 Default `medium` (`16,384` Thinking Ceiling) Causes Zero False Cutoffs
Across all **`1,494` assistant turns** in all four runs:
- Natural thinking lengths under `medium` peaked at **`4,901` tokens** on `35B-A3B` (`p99 = 3,611`) and **`3,183` tokens** on `27B` (`p99 = 2,634`).
- Natural thinking lengths under `max` peaked at **`7,294` tokens** on `35B-A3B` (`p99 = 3,498`) and **`8,826` tokens** on `27B` (`p99 = 4,886`).
- `inference.thinking_budget_exhausted` fired **`0` times across `1,494` turns**. The default `medium` ceiling (`16,384` tokens when `max_tokens` is omitted) sits **`3.3×–5.1×` above the longest legitimate reasoning step**, so it never clips productive thinking while acting as a hard circuit breaker against pathological infinite-thinking loops.

### 4.2 Dense `Qwen3.8-27B` (`medium`) Finishes `1.53×` Faster Than MoE `35B-A3B` (`medium`)
- Even though `35B-A3B` has higher single-stream decode throughput (`105.7 tok/s` mean vs `66.5 tok/s` on `27B`), **`27B medium` completed the 10-task slice in `7m 33s` vs `11m 31s` on `35B-A3B medium`**.
- `27B medium` needed **half as many turns (`200` vs `416`)**, **`2.06×` fewer output tokens (`84,797` vs `174,760`)**, and **`3.66×` fewer cumulative prompt tokens (`2.66M` vs `9.72M`)**, and emitted **`0` malformed patch submissions** (vs `2` non-`-p1` submissions on `35B-A3B`).

### 4.3 `--reasoning-effort max` Hurts `35B-A3B` (`60% -> 40–50%`) but Lifts `27B` to `80%` (`8/10`)
- On **`Qwen3.6-35B-A3B`**, switching from `medium` to `max` increased thinking tokens by **`+47.4%`** (`96.9K -> 142.8K`) and wall-clock time by **`+47.8%`**, but caused the model to second-guess its clean initial fix on `astropy-14096` and replace it with a broken descriptor workaround (`6/10 -> 4–5/10`).
- On **`Qwen3.8-27B`**, switching from `medium` to `max` increased thinking tokens by **`4.46×`** (`50.7K -> 226.2K`) and turns by **`2.28×`** (`200 -> 456`), which unlocked two additional hard tasks (`6/10 -> 8/10`):
  - `astropy-13236` (where `27B medium` exited prematurely after `8` turns, whereas `27B max` verified the deprecation test suite across `49` turns and passed).
  - `astropy-13398` (a 96-turn 3D coordinate transformation task that all three other runs failed).

### 4.4 `MambaRadixCache` Multi-Session Tombstone Cascade Under KV Pool Exhaustion ([Issue #23](https://github.com/nchapman/warp34/issues/23))
- Running `27B max` with `4` concurrent workers against the default `262,144`-token KV pool exposed a prefix-cache cliff once the 4 active sessions' combined contexts reached **`291,214` tokens** (`109.7K + 76.7K + 56.5K + 48.4K > 262,144` KV capacity) at turn `#284`:
  1. Over 70+ linear turns per session, `MambaRadixCache::evict_mamba` Priority 1 (`is_superseded_linear_internal`, `crates/warp34-host/src/radix.rs:914-917`) tombstones every intermediate turn node's `mamba_value = None` between the shared `432`-token system prompt and the session's current tip leaf.
  2. When `reserve_kv_pages_from` (`crates/warp34-host/src/model/batch.rs:71`) calls `evict_full_pages(missing - avail)` to reclaim a small KV page deficit (`~64–128` pages), `delete_leaf_and_cleanup_tombstones` (`crates/warp34-host/src/radix.rs:730-755`) deletes one idle session's tip leaf and cascades through all tombstoned ancestor nodes back to token `432`, freeing all `40K–110K` tokens (`2,500–6,800` KV pages) at once.
  3. On that session's next turn, it sees `kv_matched_tokens = 432, cached_prompt_tokens = 0, reason = "no_checkpoint"` and cold-prefills `50K–110K` tokens (`250–350s` TTFT), which in turn evicts and cascades the next session's branch.
- **Workaround until [#23](https://github.com/nchapman/warp34/issues/23) is fixed:** When running `--reasoning-effort max` on long-horizon agent benchmarks where individual trajectories can reach `100K+` tokens, ensure `workers × max_trajectory_tokens < ctx_size` (e.g., `--ctx-size 384k --workers 2` on `27B TP=2`).
