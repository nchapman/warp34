# SWE-bench Verified Agentic Evaluation Harness

`benchmarks/swebench/run_swebench.py` runs **SWE-bench Verified** slices through
[`mini-swe-agent`](https://github.com/SWE-agent/mini-swe-agent) against `warp34 serve`,
evaluates generated patches deterministically inside the upstream
`swebench==3.0.17` Docker test containers (`FAIL_TO_PASS` + `PASS_TO_PASS` `pytest`
suites), and aggregates turn-level thinking and `MambaRadixCache` telemetry from
`warp34`'s structured JSON server logs.

By default, `max_tokens` is **omitted** on client requests so `warp34`'s
out-of-the-box thinking-effort budget (`--reasoning-effort medium` -> `16,384`
thinking tokens/turn, `--reasoning-effort max` -> `31,129` thinking tokens/turn)
is exercised under real multi-turn coding-agent conditions.

See [`docs/SWEBENCH.md`](../../docs/SWEBENCH.md) for the full 4-way empirical
findings (`Qwen3.6-35B-A3B-W4A16-HGPTQ` vs `Qwen3.8-27B-W4A16-HGPTQ` × `medium`
vs `max`).

## Quick Start

### 1. One-Command Run + Docker Evaluation (`--spawn-server --evaluate`)

```sh
export PATH="/opt/cuda/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1

# Qwen3.8-27B-W4A16-HGPTQ (TP=2, MTP-3, default medium thinking effort)
python3 benchmarks/swebench/run_swebench.py run \
  --spawn-server --evaluate \
  --model nchapman/Qwen3.8-27B-W4A16-HGPTQ \
  --served-model-name qwen3.8-27b \
  --tp 2 --speculative mtp --mtp-k 3 \
  --slice 0:10 --workers 4 \
  --output /tmp/swebench-27b-medium-10

# Qwen3.8-27B-W4A16-HGPTQ (TP=2, MTP-3, max thinking effort, 384k KV pool)
python3 benchmarks/swebench/run_swebench.py run \
  --spawn-server --evaluate \
  --model nchapman/Qwen3.8-27B-W4A16-HGPTQ \
  --served-model-name qwen3.8-27b \
  --tp 2 --speculative mtp --mtp-k 3 \
  --ctx-size 384k --reasoning-effort max \
  --slice 0:10 --workers 2 \
  --output /tmp/swebench-27b-max-10
```

### 2. Evaluate or Re-Analyze an Existing Output Directory

```sh
# Run official swebench==3.0.17 Docker test suite against <output>/preds.json
python3 benchmarks/swebench/run_swebench.py eval \
  --output /tmp/swebench-27b-medium-10 \
  --workers 4

# Aggregate trajectories + warp34 JSON server logs into Markdown and summary.json
python3 benchmarks/swebench/run_swebench.py analyze \
  --output /tmp/swebench-27b-medium-10 \
  --server-log /tmp/warp34-27b-swebench-medium.log
```

### 3. CPU Unit Tests

```sh
python3 -m unittest discover -s benchmarks/swebench/tests -v
```
