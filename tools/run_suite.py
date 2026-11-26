#!/usr/bin/env python3
"""Unified verification & release qualification orchestrator for warp34.

Provides three structured, non-redundant verification tiers plus a standalone soak:
  - `smoke`   (~1–2 min, 1 GPU):  Fast inner-loop lint, static kernel/SIMD gates,
                                  harness tests, doctests, and GPU `--lib` unit tests.
  - `gate`    (~10–15 min, 2 GPUs): Pre-merge confidence: full static/dashboard checks,
                                  release build, full GPU unit + integration tests,
                                  multi-GPU compute-sanitizer, 4B bench-gate, and
                                  lossless greedy oracle.
  - `qualify` (~35–50 min, 2 GPUs): Release qualification: deduplicated `gate` +
                                  full-fleet bench-gate, TP=1 vs TP=2 oracle-dump,
                                  27B 32K–262K needle-gates (auto-pinned to 1 GPU),
                                  live multi-agent/DP=2/image soak, and pinned
                                  5-task SWE-bench Verified slice.
  - `soak`    (~12–15 min, 2 GPUs): Standalone managed live-server release soak.

Features:
  - Fast preflight (<2s): GPU squatter detection, GPU count/arch check, CUDA PATH
    setup, and cached checkpoint verification (`prep_test_models.py --check`).
  - Strict sequential GPU execution with automatic per-stage `CUDA_VISIBLE_DEVICES`
    pinning (single-GPU stages pin to the first visible GPU automatically).
  - Machine-readable `summary.json`, human-readable `summary.md`, and per-stage
    logs in `outputs/<tier>-<timestamp>/`.
  - Resumability via `--resume <dir>`, `--from <stage>`, `--skip <stage>`, and
    clean process-group teardown on SIGINT/SIGTERM.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TIERS: tuple[str, ...] = ("smoke", "gate", "qualify", "soak")


@dataclasses.dataclass(frozen=True)
class Stage:
    name: str
    cmd: tuple[str, ...]
    gpu_mode: str = "none"  # "none" | "single" | "multi"
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    shell: bool = False
    description: str = ""
    timeout_s: float | None = None


@dataclasses.dataclass
class StageResult:
    name: str
    status: str  # "passed" | "failed" | "skipped" | "resumed"
    exit_code: int
    elapsed_s: float
    gpu_mode: str
    log_path: str | None = None
    note: str | None = None


def discover_visible_gpus(env_cvd: str | None = None) -> list[str]:
    """Return list of visible GPU indices/UUIDs respecting `CUDA_VISIBLE_DEVICES`."""
    cvd = env_cvd if env_cvd is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        cleaned = cvd.strip()
        if not cleaned or cleaned in ("-1", "none", "NoDevFiles"):
            return []
        return [tok.strip() for tok in cleaned.split(",") if tok.strip()]
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.SubprocessError):
        return []


def check_gpu_squatters(visible_gpus: list[str]) -> tuple[list[str], str | None]:
    """Return `(active_processes, query_error)` on visible GPUs."""
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    ]
    if visible_gpus:
        cmd.extend(["-i", ",".join(visible_gpus)])
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=5.0)
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()], None
    except FileNotFoundError:
        return [], "nvidia-smi not found on PATH"
    except subprocess.SubprocessError as exc:
        return [], f"nvidia-smi query failed: {exc}"


def git_metadata() -> dict[str, Any]:
    sha = "unknown"
    dirty = False
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = bool(status)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return {"git_sha": sha, "git_dirty": dirty}


def build_perf_check_shell_cmd() -> str:
    return (
        "cd benchmarks/perf && "
        "python3 -m unittest discover -s tests -v && "
        "python3 -m warp34perf validate suite.toml && "
        "for f in campaigns/*.toml; do python3 -m warp34perf validate \"$f\" || exit 1; done"
    )


def build_oracle_greedy_shell_cmd() -> str:
    return (
        "./target/release/warp34 greedy-diff --offline --model Qwen/Qwen3.5-4B "
        "--golden tools/oracle/goldens/4B --max-prompts 4 --max-tokens 32 "
        "--kv-format tq4 --speculative mtp --mtp-k 3 --verify-lossless --verify-radix-warm && "
        "./target/release/warp34 greedy-diff --offline --model Qwen/Qwen3.5-4B "
        "--golden tools/oracle/goldens/4B --max-prompts 4 --max-tokens 32 "
        "--kv-format tq4 --speculative dflash2 --draft-model nchapman/Qwen3.5-4B-DFlash2 "
        "--verify-lossless --verify-radix-warm"
    )


def build_oracle_dump_shell_cmd() -> str:
    return (
        "./target/release/warp34 dump-diff --offline --model Qwen/Qwen3.5-4B --kv-format tq4 --compare-tp2 && "
        "./target/release/warp34 dump-diff --offline --model Qwen/Qwen3.5-4B --kv-format tq4 --image rgb://https://warp34.test/sample.ppm --compare-tp2 && "
        "./target/release/warp34 dump-diff --offline --model Qwen/Qwen3.5-4B --kv-format tq4 --speculative mtp --mtp-k 3 --compare-tp2 && "
        "./target/release/warp34 dump-diff --offline --model Qwen/Qwen3.5-4B --kv-format tq4 --speculative dflash2 --draft-model nchapman/Qwen3.5-4B-DFlash2 --prompt \"$(printf '<|im_start|>user\\nWhat is the capital of France?<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n')\" --compare-tp2 && "
        "./target/release/warp34 dump-diff --offline --model nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ --kv-format tq4 --kv-pages 64 --compare-tp2 && "
        "./target/release/warp34 dump-diff --offline --model nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ --kv-format tq4 --kv-pages 64 --speculative mtp --mtp-k 3 --compare-tp2"
    )


def build_tier_stages(
    tier: str,
    run_dir: pathlib.Path,
    *,
    duration_scale: float = 1.0,
    swebench_slice: str = "0:5",
    swebench_min_resolved: int = 3,
    strict_bench: bool = True,
) -> list[Stage]:
    """Construct the ordered, deduplicated stage list for `tier`."""
    if tier == "soak":
        return [
            Stage(
                name="soak",
                cmd=(
                    sys.executable,
                    "tools/run_release_soak.py",
                    "--out-dir",
                    str(run_dir / "soak"),
                    "--duration-scale",
                    str(duration_scale),
                ),
                gpu_mode="multi",
                description="Managed live-server soak (27B DFlash2 SWE, 4B DP=2, Qwen-Image-2.1, Bonsai 5-Trit)",
            )
        ]

    # Shared fast static & CPU harness stages.
    smoke_cpu: list[Stage] = [
        Stage("fmt-check", ("cargo", "fmt", "--check"), "none", description="Rust formatting"),
        Stage(
            "dashboard-build",
            ("bash", "-euo", "pipefail", "-c", "pnpm --dir dashboard install --frozen-lockfile && pnpm --dir dashboard build"),
            "none",
            description="Build embedded dashboard assets once",
        ),
        Stage(
            "clippy-cuda",
            ("cargo", "clippy", "--features", "cuda", "--all-targets", "--", "-D", "warnings"),
            "none",
            description="Clippy with CUDA feature",
        ),
        Stage(
            "simd-check",
            (sys.executable, "tools/check_cpu_simd.py"),
            "none",
            description="Static CPU SIMD / llvm-mca audit",
        ),
        Stage(
            "cuda-bin-check",
            (sys.executable, "tools/check_cuda_binaries.py"),
            "none",
            description="Static cuobjdump STACK/LOCAL/SHARED/REG audit (sm_86, sm_89)",
        ),
        Stage(
            "oracle-check",
            (sys.executable, "-m", "unittest", "discover", "-s", "tools/oracle", "-p", "test_*.py", "-v"),
            "none",
            description="Oracle & gate harness unit tests",
        ),
        Stage(
            "perf-check",
            (build_perf_check_shell_cmd(),),
            "none",
            shell=True,
            description="Benchmark harness unit tests + all campaign TOML validation",
        ),
        Stage(
            "swebench-check",
            (sys.executable, "-m", "unittest", "discover", "-s", "benchmarks/swebench/tests", "-v"),
            "none",
            description="SWE-bench runner & analyzer unit tests",
        ),
        Stage(
            "doc-test",
            ("cargo", "test", "--doc"),
            "none",
            description="Rust library doc-tests",
        ),
    ]

    if tier == "smoke":
        return smoke_cpu + [
            Stage(
                "test-quick",
                ("cargo", "test", "--release", "--features", "cuda", "--lib", "--", "--test-threads=1"),
                "single",
                description="GPU library unit tests (serialized)",
            )
        ]

    # Extra static checks promoted to `gate` & `qualify` so `smoke` stays fast.
    gate_extra_cpu: list[Stage] = [
        Stage(
            "clippy-cpu",
            ("cargo", "clippy", "--all-targets", "--", "-D", "warnings"),
            "none",
            description="Clippy without CUDA feature",
        ),
        Stage(
            "doc-check",
            ("cargo", "doc", "--no-deps", "--features", "cuda"),
            "none",
            env={"RUSTDOCFLAGS": "-D warnings"},
            description="Rustdoc check with warnings denied",
        ),
        Stage(
            "dashboard-check",
            ("pnpm", "--dir", "dashboard", "check"),
            "none",
            description="Dashboard TypeScript, Prettier, and Vitest suite",
        ),
        Stage(
            "build-release",
            ("cargo", "build", "--release", "--features", "cuda"),
            "none",
            description="Release binary build (shared across oracle/bench/soak stages)",
        ),
    ]

    gate_gpu: list[Stage] = [
        Stage(
            "test-lib",
            ("cargo", "test", "--release", "--features", "cuda", "--lib", "--", "--test-threads=1"),
            "single",
            description="GPU library unit tests",
        ),
        Stage(
            "test-integration",
            ("cargo", "test", "--release", "--features", "cuda", "--tests", "--", "--test-threads=1"),
            "multi",
            description="Full release integration test suite (serialized)",
        ),
        Stage(
            "cuda-sanitize",
            (sys.executable, "tools/check_cuda_sanitizer.py", "--require-multi-gpu"),
            "multi",
            env={"WARP34_PREBUILT_TESTS": "1"},
            description="compute-sanitizer (memcheck + racecheck + synccheck) across all subsystems + TP=2",
        ),
    ]

    bench_strict_flag = ("--strict",) if strict_bench else ()

    if tier == "gate":
        return smoke_cpu + gate_extra_cpu + gate_gpu + [
            Stage(
                "bench-gate",
                (
                    sys.executable,
                    "tools/check_bench_regressions.py",
                    "--live",
                    "--suite",
                    "quick",
                    *bench_strict_flag,
                    "--out-dir",
                    str(run_dir / "bench-gate"),
                ),
                "single",
                description="4B throughput & parity regression gate (Plain, MTP-3, DFlash2)",
            ),
            Stage(
                "oracle-greedy",
                (build_oracle_greedy_shell_cmd(),),
                "single",
                shell=True,
                description="4B lossless speculative + warm radix greedy token parity",
            ),
        ]

    if tier == "qualify":
        # Deduplicated: skips `bench-gate` (quick) because `bench-gate-full` is a strict superset.
        return smoke_cpu + gate_extra_cpu + gate_gpu + [
            Stage(
                "oracle-greedy",
                (build_oracle_greedy_shell_cmd(),),
                "single",
                shell=True,
                description="4B lossless speculative + warm radix greedy token parity",
            ),
            Stage(
                "oracle-dump",
                (build_oracle_dump_shell_cmd(),),
                "multi",
                shell=True,
                description="TP=1 vs TP=2 tensor parity (4B Plain/Vision/MTP/DFlash2 + 35B MoE)",
            ),
            Stage(
                "bench-gate-full",
                (
                    sys.executable,
                    "tools/check_bench_regressions.py",
                    "--live",
                    "--suite",
                    "full",
                    *bench_strict_flag,
                    "--out-dir",
                    str(run_dir / "bench-gate-full"),
                ),
                "multi",
                description="Full-fleet throughput & parity regression gate (4B, 9B+LoRA, 27B, 35B MoE, Flash-Next)",
            ),
            Stage(
                "needle-gates",
                (
                    sys.executable,
                    "tools/oracle/needle_matrix.py",
                    "--out-dir",
                    str(run_dir / "needle-gates"),
                ),
                "single",
                description="1-GPU 27B TQ4/TQ4-all 32K–262K needle-in-a-haystack gates",
            ),
            Stage(
                "soak",
                (
                    sys.executable,
                    "tools/run_release_soak.py",
                    "--out-dir",
                    str(run_dir / "soak"),
                    "--duration-scale",
                    str(duration_scale),
                ),
                "multi",
                description="Managed live-server soak (27B DFlash2 SWE, 4B DP=2, Qwen-Image-2.1, Bonsai 5-Trit)",
            ),
            Stage(
                "swebench-slice",
                (
                    sys.executable,
                    "benchmarks/swebench/run_swebench.py",
                    "run",
                    "--spawn-server",
                    "--evaluate",
                    "--model",
                    "nchapman/Qwen3.8-27B-W4A16-HGPTQ",
                    "--tp",
                    "2",
                    "--speculative",
                    "mtp",
                    "--mtp-k",
                    "3",
                    "--slice",
                    swebench_slice,
                    "--workers",
                    "4",
                    "--output",
                    str(run_dir / "swebench"),
                    "--min-submitted",
                    "5",
                    "--min-resolved",
                    str(swebench_min_resolved),
                    "--max-server-errors",
                    "0",
                    "--max-budget-exhausted",
                    "0",
                    "--max-format-errors",
                    "1",
                    "--min-prefix-cache-hit-pct",
                    "85.0",
                    "--min-stream-tps-p50",
                    "45.0",
                ),
                "multi",
                description="Pinned 5-task SWE-bench Verified live solve + deterministic server gates",
            ),
        ]

    raise ValueError(f"unknown tier {tier!r}")


def required_gpus_for_stages(stages: list[Stage]) -> int:
    if any(s.gpu_mode == "multi" for s in stages):
        return 2
    if any(s.gpu_mode == "single" for s in stages):
        return 1
    return 0


def run_preflight(
    *,
    tier: str,
    stages: list[Stage],
    allow_squatters: bool = False,
    skip_model_check: bool = False,
) -> tuple[list[str], list[str]]:
    """Verify host readiness before starting GPU work.

    Returns `(errors, visible_gpus)`.
    """
    errors: list[str] = []
    visible_gpus = discover_visible_gpus()
    needed_gpus = required_gpus_for_stages(stages)
    if len(visible_gpus) < needed_gpus:
        errors.append(
            f"tier '{tier}' requires {needed_gpus} visible GPU(s), found {len(visible_gpus)} "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r})"
        )

    if needed_gpus > 0 and not allow_squatters:
        squatters, query_err = check_gpu_squatters(visible_gpus)
        if query_err:
            errors.append(
                f"unable to verify GPU idle state ({query_err}); pass --allow-squatters to override"
            )
        elif squatters:
            errors.append(
                "active GPU compute process(es) detected (pass --allow-squatters to override): "
                + "; ".join(squatters)
            )

    if not skip_model_check and needed_gpus > 0:
        prep_cmd = [sys.executable, "tools/prep_test_models.py", "--check"]
        if tier == "qualify":
            prep_cmd.append("--all")
        try:
            proc = subprocess.run(
                prep_cmd,
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            if proc.returncode != 0:
                remediation = "mise run test-prep-all" if tier == "qualify" else "mise run test-prep"
                errors.append(
                    f"required test checkpoints missing/unverified (run `{remediation}`): "
                    + (proc.stderr.strip() or proc.stdout.strip())[:300]
                )
        except subprocess.TimeoutExpired:
            errors.append("preflight model check (`prep_test_models.py --check`) timed out after 30s")

    if any(s.name == "swebench-slice" for s in stages):
        if shutil.which("docker") is None:
            errors.append(
                "swebench-slice requires `docker` on PATH (or pass --skip swebench-slice)"
            )
        if shutil.which("uvx") is None:
            errors.append(
                "swebench-slice requires `uvx` on PATH (or pass --skip swebench-slice)"
            )

    return errors, visible_gpus


def stage_env_for(stage: Stage, visible_gpus: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    opt_cuda_bin = "/opt/cuda/bin"
    if pathlib.Path(opt_cuda_bin).is_dir():
        cur_path = env.get("PATH", "")
        if opt_cuda_bin not in cur_path.split(":"):
            env["PATH"] = f"{opt_cuda_bin}:{cur_path}" if cur_path else opt_cuda_bin
    if stage.gpu_mode == "single" and visible_gpus:
        env["CUDA_VISIBLE_DEVICES"] = visible_gpus[0]
    elif stage.gpu_mode == "multi" and visible_gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpus)
    env.update(stage.env)
    return env


def _terminate_proc_group(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            proc.kill()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass


def run_stage(
    stage: Stage,
    *,
    run_dir: pathlib.Path,
    visible_gpus: list[str],
) -> StageResult:
    log_path = run_dir / f"{stage.name}.log"
    env = stage_env_for(stage, visible_gpus)
    t0 = time.monotonic()
    print(f"\n[suite] >>> {stage.name} ({stage.gpu_mode}) — {stage.description}", file=sys.stderr)
    proc: subprocess.Popen[Any] | None = None
    note: str | None = None
    with log_path.open("w", encoding="utf-8") as log_f:
        try:
            if stage.shell:
                proc = subprocess.Popen(
                    stage.cmd[0],
                    cwd=REPO_ROOT,
                    env=env,
                    shell=True,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            else:
                proc = subprocess.Popen(
                    list(stage.cmd),
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            rc = proc.wait(timeout=stage.timeout_s)
        except OSError as exc:
            log_f.write(f"failed to launch stage {stage.name}: {exc}\n")
            rc = 127
            note = f"launch error: {exc}"
        except subprocess.TimeoutExpired:
            if proc is not None:
                _terminate_proc_group(proc)
            log_f.write(f"\nstage {stage.name} timed out after {stage.timeout_s}s\n")
            rc = 124
            note = f"timed out after {stage.timeout_s}s"
        except KeyboardInterrupt:
            if proc is not None:
                _terminate_proc_group(proc)
            raise

    elapsed = time.monotonic() - t0
    status = "passed" if rc == 0 else "failed"
    print(
        f"[suite] <<< {stage.name}: {status.upper()} ({elapsed:.1f}s) [log: {log_path}]",
        file=sys.stderr,
    )
    if rc != 0:
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            if tail:
                print(f"--- tail of {log_path} ---", file=sys.stderr)
                for line in tail:
                    print(f"  {line}", file=sys.stderr)
        except OSError:
            pass
    return StageResult(
        name=stage.name,
        status=status,
        exit_code=rc,
        elapsed_s=elapsed,
        gpu_mode=stage.gpu_mode,
        log_path=str(log_path),
        note=note,
    )


def suite_passed(results: list[StageResult]) -> bool:
    """Return True iff at least one stage passed/resumed and zero stages failed."""
    return any(r.status in ("passed", "resumed") for r in results) and all(
        r.status != "failed" for r in results
    )


def render_summary_markdown(
    tier: str,
    run_dir: pathlib.Path,
    results: list[StageResult],
    meta: dict[str, Any],
) -> str:
    failed = sum(1 for r in results if r.status == "failed")
    passed = sum(1 for r in results if r.status in ("passed", "resumed"))
    skipped = sum(1 for r in results if r.status == "skipped")
    total_s = sum(r.elapsed_s for r in results)
    overall = "PASS" if suite_passed(results) else ("NO-OP" if failed == 0 and passed == 0 else "FAIL")
    dirty_suffix = " (dirty)" if meta.get("git_dirty") else ""
    lines = [
        f"## `warp34` Verification Receipt — `{tier}` (`{overall}`)",
        "",
        f"- **Git:** `{meta.get('git_sha', 'unknown')}`{dirty_suffix}",
        f"- **Run Dir:** `{run_dir}`",
        f"- **Stages:** `{passed}` passed, `{skipped}` skipped, `{failed}` failed (`{total_s:.1f}s` total)",
        "",
        "| Stage | GPU | Status | Duration | Exit | Log |",
        "| :--- | :-: | :-: | ---: | ---: | :--- |",
    ]
    for r in results:
        badge = {
            "passed": "**PASS**",
            "resumed": "PASS (resumed)",
            "skipped": "SKIP",
            "failed": "**FAIL**",
        }.get(r.status, r.status)
        log_col = f"`{pathlib.Path(r.log_path).name}`" if r.log_path else "—"
        lines.append(
            f"| `{r.name}` | `{r.gpu_mode}` | {badge} | {r.elapsed_s:.1f}s | {r.exit_code} | {log_col} |"
        )
    return "\n".join(lines)


def write_receipt(
    tier: str,
    run_dir: pathlib.Path,
    results: list[StageResult],
    meta: dict[str, Any],
) -> str:
    md = render_summary_markdown(tier, run_dir, results, meta)
    (run_dir / "summary.md").write_text(md + "\n", encoding="utf-8")
    payload = {
        "tier": tier,
        "run_dir": str(run_dir),
        **meta,
        "passed": suite_passed(results),
        "stages": [dataclasses.asdict(r) for r in results],
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return md


def load_resumed_passed_stages(resume_dir: pathlib.Path) -> set[str]:
    summary_path = resume_dir / "summary.json"
    if not summary_path.is_file():
        return set()
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(
            f"[suite] warning: ignoring corrupt resume receipt {summary_path}: {exc}",
            file=sys.stderr,
        )
        return set()
    if not isinstance(data, dict) or not isinstance(data.get("stages"), list):
        print(
            f"[suite] warning: ignoring malformed resume receipt {summary_path}",
            file=sys.stderr,
        )
        return set()
    return {
        str(s["name"])
        for s in data["stages"]
        if isinstance(s, dict) and s.get("status") in ("passed", "resumed") and "name" in s
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified warp34 verification suite orchestrator")
    parser.add_argument("tier", choices=TIERS, help="Verification tier to execute")
    parser.add_argument("--list", action="store_true", help="List ordered stages and exit")
    parser.add_argument("--dry-run", action="store_true", help="Run preflight and print commands without executing")
    parser.add_argument("--from", dest="from_stage", default=None, help="Start execution at this stage name")
    parser.add_argument("--skip", action="append", default=[], help="Skip named stage(s) (repeatable)")
    parser.add_argument(
        "--resume",
        type=pathlib.Path,
        default=None,
        help="Resume an existing run directory, skipping stages that already passed",
    )
    parser.add_argument("--out-dir", type=pathlib.Path, default=None, help="Override receipt/log output directory")
    parser.add_argument("--keep-going", action="store_true", help="Continue remaining stages after a failure")
    parser.add_argument("--allow-squatters", action="store_true", help="Do not fail preflight on active GPU processes")
    parser.add_argument("--skip-model-check", action="store_true", help="Skip prep_test_models.py --check in preflight")
    parser.add_argument("--no-strict-bench", action="store_true", help="Allow skipped bench-gate cases")
    parser.add_argument("--duration-scale", type=float, default=1.0, help="Soak duration scale factor")
    parser.add_argument("--swebench-slice", default="0:5", help="SWE-bench slice for qualify (default: 0:5)")
    parser.add_argument("--swebench-min-resolved", type=int, default=3, help="Min resolved tasks for qualify SWE-bench")
    args = parser.parse_args(argv)

    run_dir = (
        args.resume
        or args.out_dir
        or (REPO_ROOT / "outputs" / f"{args.tier}-{time.strftime('%Y%m%d-%H%M%S')}")
    )
    all_stages = build_tier_stages(
        args.tier,
        run_dir,
        duration_scale=args.duration_scale,
        swebench_slice=args.swebench_slice,
        swebench_min_resolved=args.swebench_min_resolved,
        strict_bench=not args.no_strict_bench,
    )

    if args.list:
        print(f"Tier `{args.tier}` ({len(all_stages)} stages):")
        for idx, st in enumerate(all_stages, 1):
            cmd_str = st.cmd[0] if st.shell else " ".join(st.cmd)
            print(f"  {idx:2d}. {st.name:<18} [{st.gpu_mode:<6}] {st.description}")
            print(f"      cmd: {cmd_str}")
        return 0

    skip_set = set(args.skip)
    if args.from_stage:
        names = [s.name for s in all_stages]
        if args.from_stage not in names:
            print(f"error: unknown --from stage {args.from_stage!r}; valid: {', '.join(names)}", file=sys.stderr)
            return 2
        cut = names.index(args.from_stage)
        skip_set.update(names[:cut])

    resumed_passed = load_resumed_passed_stages(args.resume) if args.resume else set()
    active_stages = [s for s in all_stages if s.name not in skip_set and s.name not in resumed_passed]

    preflight_errors, visible_gpus = run_preflight(
        tier=args.tier,
        stages=active_stages,
        allow_squatters=args.allow_squatters,
        skip_model_check=args.skip_model_check,
    )
    if preflight_errors:
        print(f"[suite] preflight FAILED for `{args.tier}`:", file=sys.stderr)
        for err in preflight_errors:
            print(f"  - {err}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"[suite] dry-run `{args.tier}` (visible GPUs: {visible_gpus}):")
        for st in active_stages:
            cmd_str = st.cmd[0] if st.shell else " ".join(st.cmd)
            print(f"  - {st.name} [{st.gpu_mode}]: {cmd_str}")
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    meta = git_metadata()
    results: list[StageResult] = []

    for st in all_stages:
        if st.name in resumed_passed:
            results.append(StageResult(st.name, "resumed", 0, 0.0, st.gpu_mode))
            continue
        if st.name in skip_set:
            results.append(StageResult(st.name, "skipped", 0, 0.0, st.gpu_mode))
            continue
        res = run_stage(st, run_dir=run_dir, visible_gpus=visible_gpus)
        results.append(res)
        write_receipt(args.tier, run_dir, results, meta)
        if res.status == "failed" and not args.keep_going:
            break

    md = write_receipt(args.tier, run_dir, results, meta)
    print("\n" + md)
    return 0 if suite_passed(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
