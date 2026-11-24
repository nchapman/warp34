#!/usr/bin/env python3
"""End-to-end SWE-bench Verified runner, evaluator, and telemetry analyzer for warp34.

Manages optional `warp34 serve` lifecycle, invokes `mini-swe-agent` (`mini-extra swebench`)
with omitted `max_tokens` by default (testing out-of-the-box coding-agent thinking budgets),
evaluates generated patches via the deterministic `swebench==3.0.17` Docker test harness,
and aggregates turn-level thinking/radix-cache telemetry from `warp34` JSON logs.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_DATASET = "princeton-nlp/SWE-Bench_Verified"
DEFAULT_SWEBENCH_VERSION = "3.0.17"
DEFAULT_SUBSET = "verified"
DEFAULT_SPLIT = "test"
DEFAULT_SLICE = "0:10"
DEFAULT_WORKERS = 4
DEFAULT_PORT = 18034
DEFAULT_HOST = "127.0.0.1"


@dataclass
class InstanceSummary:
    instance_id: str
    exit_status: str
    turns: int
    format_errors: int
    patch_bytes: int
    resolved: bool | None = None
    resolved_normalized: bool | None = None


@dataclass
class ServerLogSummary:
    total_turns: int = 0
    cancelled_turns: int = 0
    error_events: int = 0
    thinking_budget_exhausted_events: int = 0
    wall_time_s: float = 0.0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    prefix_cache_hit_pct: float = 0.0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    content_tokens: int = 0
    reasoning_share_pct: float = 0.0
    wall_output_tok_per_s: float = 0.0
    stream_tps_p50: float = 0.0
    stream_tps_mean: float = 0.0
    stream_tps_p90: float = 0.0
    stream_tps_max: float = 0.0
    ttft_ms_p50: float = 0.0
    ttft_ms_p90: float = 0.0
    ttft_ms_max: float = 0.0
    thinking_tokens_p50: int = 0
    thinking_tokens_p90: int = 0
    thinking_tokens_p99: int = 0
    thinking_tokens_max: int = 0
    output_tokens_p50: int = 0
    output_tokens_p90: int = 0
    output_tokens_p99: int = 0
    output_tokens_max: int = 0


@dataclass
class RunReport:
    output_dir: str
    submitted_count: int
    resolved_count: int | None
    resolved_normalized_count: int | None
    total_trajectory_turns: int
    total_format_errors: int
    instances: list[InstanceSummary] = field(default_factory=list)
    server: ServerLogSummary | None = None


def percentile_sorted(sorted_vals: list[int | float], q: float) -> int | float:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def build_warp34_serve_cmd(
    *,
    binary: str,
    model: str,
    served_model_name: str,
    host: str,
    port: int,
    tp: int,
    speculative: str,
    mtp_k: int,
    ctx_size: str | None,
    reasoning_effort: str | None,
    reasoning_budget: str | None,
    offline: bool = True,
) -> list[str]:
    cmd = [
        binary,
        "serve",
        model,
        "--tp",
        str(tp),
        "--speculative",
        speculative,
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if offline:
        cmd.append("--offline")
    if speculative == "mtp":
        cmd.extend(["--mtp-k", str(mtp_k)])
    if ctx_size:
        cmd.extend(["--ctx-size", ctx_size])
    if reasoning_effort:
        cmd.extend(["--reasoning-effort", reasoning_effort])
    if reasoning_budget:
        cmd.extend(["--reasoning-budget", reasoning_budget])
    return cmd


def build_minisweagent_cmd(
    *,
    output_dir: pathlib.Path,
    served_model_name: str,
    api_base: str,
    subset: str = DEFAULT_SUBSET,
    split: str = DEFAULT_SPLIT,
    slice_spec: str = DEFAULT_SLICE,
    workers: int = DEFAULT_WORKERS,
    max_tokens: int | None = None,
) -> list[str]:
    model_id = (
        served_model_name
        if served_model_name.startswith("openai/")
        else f"openai/{served_model_name}"
    )
    cmd = [
        "uvx",
        "--from",
        "mini-swe-agent",
        "mini-extra",
        "swebench",
        "--subset",
        subset,
        "--split",
        split,
        "--slice",
        slice_spec,
        "--workers",
        str(workers),
        "--output",
        str(output_dir),
        "-m",
        model_id,
        "-c",
        "swebench.yaml",
        "-c",
        "model.cost_tracking=ignore_errors",
        "-c",
        f"model.model_kwargs.api_base={api_base}",
        "-c",
        "model.model_kwargs.api_key=dummy",
        "-c",
        "agent.cost_limit=0",
    ]
    if max_tokens is not None:
        cmd.extend(["-c", f"model.model_kwargs.max_tokens={max_tokens}"])
    return cmd


def build_swebench_eval_cmd(
    *,
    predictions_path: pathlib.Path,
    run_id: str,
    max_workers: int = DEFAULT_WORKERS,
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    swebench_version: str = DEFAULT_SWEBENCH_VERSION,
    instance_ids: list[str] | None = None,
) -> list[str]:
    cmd = [
        "uvx",
        "--from",
        f"swebench=={swebench_version}",
        "python3",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
    ]
    if instance_ids:
        cmd.append("--instance_ids")
        cmd.extend(instance_ids)
    return cmd


_DIFF_HEADER_P0_RE = re.compile(
    r"^--- (?P<old>(?!a/)[^\t\n]+?)(?:\.bak|\.orig)?(?:\t.*)?\n\+\+\+ (?P<new>(?!b/)[^\t\n]+?)(?:\t.*)?$",
    re.MULTILINE,
)


def normalize_patch_Or_extract_from_traj(
    patch: str,
    traj_data: dict[str, Any] | None = None,
) -> tuple[str, bool]:
    """Normalizes `-p0` unified diff headers to `-p1` (`a/...`, `b/...`) or extracts the last
    `git diff` output from the trajectory when the submitted patch is raw file contents.
    """
    if "--- a/" in patch and "+++ b/" in patch:
        return patch, False

    if _DIFF_HEADER_P0_RE.search(patch):
        fixed = _DIFF_HEADER_P0_RE.sub(
            lambda m: f"--- a/{m.group('new').lstrip('./')}\n+++ b/{m.group('new').lstrip('./')}",
            patch,
        )
        return fixed, fixed != patch

    if "diff --git " in patch:
        return patch, False

    if traj_data:
        for msg in reversed(traj_data.get("messages", [])):
            content = str(msg.get("content", ""))
            if "<output>\n" in content and "diff --git " in content:
                out_block = content.split("<output>\n", 1)[1].rsplit("</output>", 1)[0]
                if "diff --git " in out_block:
                    extracted = "diff --git " + out_block.split("diff --git ", 1)[1]
                    return extracted, True
    return patch, False


def _load_json_safe(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(errors="replace"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def parse_server_logs(log_paths: list[pathlib.Path]) -> ServerLogSummary:
    turns: list[dict[str, Any]] = []
    cancelled = 0
    exhausted = 0
    errors = 0
    for path in log_paths:
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.startswith("{"):
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(ev, dict):
                        continue
                    event = ev.get("event")
                    if event == "inference.turn_end":
                        if ev.get("finish_reason") == "cancelled":
                            cancelled += 1
                        else:
                            try:
                                _ = float(ev["timestamp_unix_ms"])
                                _ = float(ev.get("duration_ms", 0.0))
                                _ = int(ev.get("prompt_tokens", 0))
                                _ = int(ev.get("cached_prompt_tokens", 0))
                                _ = int(ev.get("output_tokens", 0))
                                _ = int(ev.get("reasoning_tokens", 0))
                            except (KeyError, TypeError, ValueError):
                                errors += 1
                                continue
                            turns.append(ev)
                    elif event == "inference.thinking_budget_exhausted":
                        exhausted += 1
                    elif ev.get("level") in ("warn", "error"):
                        errors += 1
        except OSError:
            continue

    if not turns:
        return ServerLogSummary(
            cancelled_turns=cancelled,
            error_events=errors,
            thinking_budget_exhausted_events=exhausted,
        )

    t0 = min(t["timestamp_unix_ms"] - t.get("duration_ms", 0.0) for t in turns)
    t1 = max(t["timestamp_unix_ms"] for t in turns)
    wall_s = max(1e-3, (t1 - t0) / 1000.0)
    tot_prompt = sum(int(t.get("prompt_tokens", 0)) for t in turns)
    tot_cached = sum(int(t.get("cached_prompt_tokens", 0)) for t in turns)
    tot_out = sum(int(t.get("output_tokens", 0)) for t in turns)
    tot_think = sum(int(t.get("reasoning_tokens", 0)) for t in turns)
    tot_content = max(0, tot_out - tot_think)

    stream_tps = sorted(
        float(t["output_tokens_per_s"])
        for t in turns
        if t.get("output_tokens_per_s") is not None
    )
    ttfts = sorted(float(t.get("ttft_ms", 0.0)) for t in turns)
    think_toks = sorted(int(t.get("reasoning_tokens", 0)) for t in turns)
    out_toks = sorted(int(t.get("output_tokens", 0)) for t in turns)

    return ServerLogSummary(
        total_turns=len(turns),
        cancelled_turns=cancelled,
        error_events=errors,
        thinking_budget_exhausted_events=exhausted,
        wall_time_s=round(wall_s, 2),
        prompt_tokens=tot_prompt,
        cached_prompt_tokens=tot_cached,
        prefix_cache_hit_pct=round(100.0 * tot_cached / max(1, tot_prompt), 2),
        output_tokens=tot_out,
        reasoning_tokens=tot_think,
        content_tokens=tot_content,
        reasoning_share_pct=round(100.0 * tot_think / max(1, tot_out), 2),
        wall_output_tok_per_s=round(tot_out / wall_s, 1),
        stream_tps_p50=round(float(percentile_sorted(stream_tps, 0.50)), 1),
        stream_tps_mean=round(sum(stream_tps) / max(1, len(stream_tps)), 1),
        stream_tps_p90=round(float(percentile_sorted(stream_tps, 0.90)), 1),
        stream_tps_max=round(float(stream_tps[-1]) if stream_tps else 0.0, 1),
        ttft_ms_p50=round(float(percentile_sorted(ttfts, 0.50)), 1),
        ttft_ms_p90=round(float(percentile_sorted(ttfts, 0.90)), 1),
        ttft_ms_max=round(float(ttfts[-1]) if ttfts else 0.0, 1),
        thinking_tokens_p50=int(percentile_sorted(think_toks, 0.50)),
        thinking_tokens_p90=int(percentile_sorted(think_toks, 0.90)),
        thinking_tokens_p99=int(percentile_sorted(think_toks, 0.99)),
        thinking_tokens_max=int(think_toks[-1]) if think_toks else 0,
        output_tokens_p50=int(percentile_sorted(out_toks, 0.50)),
        output_tokens_p90=int(percentile_sorted(out_toks, 0.90)),
        output_tokens_p99=int(percentile_sorted(out_toks, 0.99)),
        output_tokens_max=int(out_toks[-1]) if out_toks else 0,
    )


def analyze_output_dir(
    output_dir: pathlib.Path,
    server_logs: list[pathlib.Path] | None = None,
) -> RunReport:
    trajs = sorted(output_dir.glob("*/*.traj.json"))
    eval_report_path = output_dir / "eval_report.json"
    eval_norm_path = output_dir / "eval_report_normalized.json"
    resolved_set: set[str] | None = None
    resolved_norm_set: set[str] | None = None

    data = _load_json_safe(eval_report_path)
    if data is not None:
        resolved_set = set(data.get("resolved_ids", []))
    data_norm = _load_json_safe(eval_norm_path)
    if data_norm is not None:
        resolved_norm_set = set(data_norm.get("resolved_ids", []))
    elif resolved_set is not None:
        resolved_norm_set = set(resolved_set)

    instances: list[InstanceSummary] = []
    tot_turns = 0
    tot_fmt_errs = 0
    for tpath in trajs:
        d = _load_json_safe(tpath)
        if d is None:
            continue
        info = d.get("info", {})
        sub = info.get("submission") or ""
        msgs = d.get("messages", [])
        assistant_turns = sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "assistant")
        fmt_errs = sum(
            1
            for m in msgs
            if isinstance(m, dict)
            and m.get("role") == "user"
            and (
                "Tool call error:" in str(m.get("content", ""))
                or "reached the output token limit" in str(m.get("content", ""))
            )
        )
        iid = tpath.parent.name
        tot_turns += assistant_turns
        tot_fmt_errs += fmt_errs
        instances.append(
            InstanceSummary(
                instance_id=iid,
                exit_status=str(info.get("exit_status", "Unknown")),
                turns=assistant_turns,
                format_errors=fmt_errs,
                patch_bytes=len(sub),
                resolved=(iid in resolved_set) if resolved_set is not None else None,
                resolved_normalized=(
                    (iid in resolved_norm_set) if resolved_norm_set is not None else None
                ),
            )
        )

    logs_to_read = server_logs if server_logs is not None else [output_dir / "warp34-server.log"]
    server_summary = parse_server_logs(logs_to_read)
    if server_summary.total_turns == 0 and server_summary.cancelled_turns == 0:
        server_summary = None

    return RunReport(
        output_dir=str(output_dir),
        submitted_count=len(instances),
        resolved_count=len(resolved_set) if resolved_set is not None else None,
        resolved_normalized_count=(
            len(resolved_norm_set) if resolved_norm_set is not None else None
        ),
        total_trajectory_turns=tot_turns,
        total_format_errors=tot_fmt_errs,
        instances=instances,
        server=server_summary,
    )


def format_markdown_report(report: RunReport) -> str:
    lines = [
        f"## SWE-bench Verified Report (`{report.output_dir}`)",
        "",
        f"- **Submitted:** `{report.submitted_count}`",
    ]
    if report.resolved_count is not None:
        pct = 100.0 * report.resolved_count / max(1, report.submitted_count)
        lines.append(
            f"- **Resolved (raw `preds.json`):** `{report.resolved_count} / {report.submitted_count} ({pct:.1f}%)`"
        )
    if (
        report.resolved_normalized_count is not None
        and report.resolved_normalized_count != report.resolved_count
    ):
        pct_n = 100.0 * report.resolved_normalized_count / max(1, report.submitted_count)
        lines.append(
            f"- **Resolved (normalized `-p1` diff):** `{report.resolved_normalized_count} / {report.submitted_count} ({pct_n:.1f}%)`"
        )
    lines.extend(
        [
            f"- **Total Agent Turns:** `{report.total_trajectory_turns}`",
            f"- **Tool-Call Format Errors:** `{report.total_format_errors}`",
        ]
    )
    if report.server:
        s = report.server
        lines.extend(
            [
                f"- **Wall-Clock Inference Span:** `{s.wall_time_s:.1f} s` (`{s.wall_output_tok_per_s:.1f} tok/s` net output)",
                f"- **Prompt Tokens (`MambaRadixCache` hit `%`):** `{s.prompt_tokens:,}` (`{s.cached_prompt_tokens:,}` cached, `{s.prefix_cache_hit_pct:.2f}%`)",
                f"- **Output Tokens:** `{s.output_tokens:,}` (`{s.reasoning_tokens:,}` thinking `[{s.reasoning_share_pct:.1f}%]`, `{s.content_tokens:,}` tool/content)",
                f"- **Thinking Tokens / Turn (`p50` / `p90` / `p99` / `max`):** `{s.thinking_tokens_p50}` / `{s.thinking_tokens_p90}` / `{s.thinking_tokens_p99}` / `{s.thinking_tokens_max}`",
                f"- **`thinking_budget_exhausted` Cutoffs:** `{s.thinking_budget_exhausted_events}`",
                f"- **Per-Stream Decode TPS (`p50` / `mean` / `p90` / `max`):** `{s.stream_tps_p50}` / `{s.stream_tps_mean}` / `{s.stream_tps_p90}` / `{s.stream_tps_max} tok/s`",
                f"- **TTFT (`p50` / `p90` / `max`):** `{s.ttft_ms_p50}` / `{s.ttft_ms_p90}` / `{s.ttft_ms_max} ms`",
            ]
        )

    lines.extend(
        [
            "",
            "| Instance ID | Exit Status | Outcome | Turns | Format Errors | Patch Bytes |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for inst in report.instances:
        if inst.resolved is True:
            outcome = "**RESOLVED**"
        elif inst.resolved_normalized is True:
            outcome = "**RESOLVED*** (`-p1`)"
        elif inst.resolved is False:
            outcome = "Unresolved"
        else:
            outcome = "—"
        lines.append(
            f"| `{inst.instance_id}` | `{inst.exit_status}` | {outcome} | {inst.turns} | {inst.format_errors} | {inst.patch_bytes:,} B |"
        )
    return "\n".join(lines) + "\n"


def wait_for_server_health(host: str, port: int, timeout_s: float = 120.0) -> None:
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for warp34 server health at {url}")


def stop_process_cleanly(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass


def run_evaluation_harness(
    output_dir: pathlib.Path,
    *,
    max_workers: int = DEFAULT_WORKERS,
    normalize_p1: bool = True,
    timeout_s: float | None = None,
) -> None:
    preds_path = output_dir / "preds.json"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {preds_path}")

    run_id = f"warp34-{output_dir.name}"
    cmd = build_swebench_eval_cmd(
        predictions_path=preds_path,
        run_id=run_id,
        max_workers=max_workers,
    )
    subprocess.run(cmd, check=True, timeout=timeout_s)

    cwd = pathlib.Path.cwd()
    reports = sorted(cwd.glob(f"*.{run_id}.json"), key=lambda p: p.stat().st_mtime)
    if reports:
        shutil.move(str(reports[-1]), str(output_dir / "eval_report.json"))
    if (cwd / "logs" / "run_evaluation" / run_id).exists():
        shutil.rmtree(cwd / "logs" / "run_evaluation" / run_id, ignore_errors=True)
        if (cwd / "logs" / "run_evaluation").exists() and not any(
            (cwd / "logs" / "run_evaluation").iterdir()
        ):
            shutil.rmtree(cwd / "logs", ignore_errors=True)

    if not normalize_p1:
        return

    preds = _load_json_safe(preds_path)
    if not preds:
        return
    changed_ids: list[str] = []
    for iid, entry in preds.items():
        if not isinstance(entry, dict):
            continue
        traj_path = output_dir / iid / f"{iid}.traj.json"
        traj_data = _load_json_safe(traj_path)
        fixed_patch, changed = normalize_patch_Or_extract_from_traj(
            str(entry.get("model_patch", "")),
            traj_data,
        )
        if changed:
            entry["model_patch"] = fixed_patch
            changed_ids.append(iid)

    if not changed_ids:
        return

    norm_preds_path = output_dir / "preds_normalized.json"
    norm_preds_path.write_text(json.dumps(preds, indent=2))
    norm_run_id = f"{run_id}-norm"
    norm_cmd = build_swebench_eval_cmd(
        predictions_path=norm_preds_path,
        run_id=norm_run_id,
        max_workers=min(max_workers, len(changed_ids)),
        instance_ids=changed_ids,
    )
    subprocess.run(norm_cmd, check=True, timeout=timeout_s)
    norm_reports = sorted(cwd.glob(f"*.{norm_run_id}.json"), key=lambda p: p.stat().st_mtime)
    if norm_reports and (output_dir / "eval_report.json").exists():
        base_eval = _load_json_safe(output_dir / "eval_report.json") or {}
        recheck_eval = _load_json_safe(norm_reports[-1]) or {}
        resolved_ids = sorted(
            set(base_eval.get("resolved_ids", [])) | set(recheck_eval.get("resolved_ids", []))
        )
        base_eval["resolved_ids"] = resolved_ids
        base_eval["resolved_instances"] = len(resolved_ids)
        (output_dir / "eval_report_normalized.json").write_text(json.dumps(base_eval, indent=2))
        norm_reports[-1].unlink(missing_ok=True)
    if (cwd / "logs" / "run_evaluation" / norm_run_id).exists():
        shutil.rmtree(cwd / "logs" / "run_evaluation" / norm_run_id, ignore_errors=True)
        if (cwd / "logs" / "run_evaluation").exists() and not any(
            (cwd / "logs" / "run_evaluation").iterdir()
        ):
            shutil.rmtree(cwd / "logs", ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SWE-bench Verified runner, Docker test evaluator, and telemetry analyzer for warp34."
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # 1. `run` subcommand
    p_run = sub.add_parser("run", help="Run mini-swe-agent on a SWE-bench slice (and optionally evaluate)")
    p_run.add_argument("--output", "-o", type=pathlib.Path, required=True, help="Output directory")
    p_run.add_argument("--model", default="nchapman/Qwen3.8-27B-W4A16-HGPTQ", help="Model repo or path")
    p_run.add_argument("--served-model-name", default="qwen3.8-27b", help="Served model alias")
    p_run.add_argument("--binary", default="./target/release/warp34", help="Path to warp34 binary")
    p_run.add_argument("--spawn-server", action="store_true", help="Launch and manage warp34 serve process")
    p_run.add_argument("--host", default=DEFAULT_HOST)
    p_run.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_run.add_argument("--tp", type=int, default=2)
    p_run.add_argument("--speculative", default="mtp")
    p_run.add_argument("--mtp-k", type=int, default=3)
    p_run.add_argument("--ctx-size", default=None, help="Optional KV pool capacity (e.g. 384k)")
    p_run.add_argument("--reasoning-effort", default=None, help="Server --reasoning-effort level")
    p_run.add_argument("--reasoning-budget", default=None, help="Server --reasoning-budget value")
    p_run.add_argument("--subset", default=DEFAULT_SUBSET)
    p_run.add_argument("--split", default=DEFAULT_SPLIT)
    p_run.add_argument("--slice", dest="slice_spec", default=DEFAULT_SLICE)
    p_run.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p_run.add_argument("--max-tokens", type=int, default=None, help="Client max_tokens (omitted by default)")
    p_run.add_argument("--timeout", type=float, default=None, help="Optional subprocess timeout in seconds")
    p_run.add_argument("--evaluate", action="store_true", help="Run swebench Docker test evaluation after generation")

    # 2. `eval` subcommand
    p_eval = sub.add_parser("eval", help="Run official swebench==3.0.17 Docker test evaluation on preds.json")
    p_eval.add_argument("--output", "-o", type=pathlib.Path, required=True, help="Output directory containing preds.json")
    p_eval.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p_eval.add_argument("--timeout", type=float, default=None, help="Optional evaluation timeout in seconds")
    p_eval.add_argument("--no-normalize-p1", action="store_true", help="Disable -p0 -> -p1 diff header normalization check")

    # 3. `analyze` subcommand
    p_an = sub.add_parser("analyze", help="Analyze trajectories, eval reports, and warp34 server JSON logs")
    p_an.add_argument("--output", "-o", type=pathlib.Path, required=True, help="Output directory")
    p_an.add_argument(
        "--server-log",
        type=pathlib.Path,
        action="append",
        default=None,
        help="Path(s) to warp34 JSON server log (defaults to <output>/warp34-server.log)",
    )
    p_an.add_argument("--json-out", type=pathlib.Path, default=None, help="Write JSON summary to path")

    args = parser.parse_args(argv)

    if args.subcommand == "run":
        args.output.mkdir(parents=True, exist_ok=True)
        server_proc: subprocess.Popen[Any] | None = None
        server_log_file = None
        server_log_path = args.output / "warp34-server.log"
        try:
            if args.spawn_server:
                serve_cmd = build_warp34_serve_cmd(
                    binary=args.binary,
                    model=args.model,
                    served_model_name=args.served_model_name,
                    host=args.host,
                    port=args.port,
                    tp=args.tp,
                    speculative=args.speculative,
                    mtp_k=args.mtp_k,
                    ctx_size=args.ctx_size,
                    reasoning_effort=args.reasoning_effort,
                    reasoning_budget=args.reasoning_budget,
                )
                server_log_file = server_log_path.open("a", encoding="utf-8")
                server_proc = subprocess.Popen(
                    serve_cmd,
                    stdout=server_log_file,
                    stderr=subprocess.STDOUT,
                )
                wait_for_server_health(args.host, args.port)

            api_base = f"http://{args.host}:{args.port}/v1"
            mswea_cmd = build_minisweagent_cmd(
                output_dir=args.output,
                served_model_name=args.served_model_name,
                api_base=api_base,
                subset=args.subset,
                split=args.split,
                slice_spec=args.slice_spec,
                workers=args.workers,
                max_tokens=args.max_tokens,
            )
            env = os.environ.copy()
            env.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
            env.setdefault("OPENAI_API_KEY", "dummy")
            subprocess.run(mswea_cmd, env=env, check=True, timeout=args.timeout)
        finally:
            stop_process_cleanly(server_proc)
            if server_log_file is not None:
                server_log_file.close()

        if args.evaluate:
            run_evaluation_harness(
                args.output,
                max_workers=args.workers,
                timeout_s=args.timeout,
            )

        report = analyze_output_dir(args.output)
        (args.output / "summary.json").write_text(json.dumps(asdict(report), indent=2))
        sys.stdout.write(format_markdown_report(report))
        return 0

    if args.subcommand == "eval":
        run_evaluation_harness(
            args.output,
            max_workers=args.workers,
            normalize_p1=not args.no_normalize_p1,
            timeout_s=args.timeout,
        )
        report = analyze_output_dir(args.output)
        (args.output / "summary.json").write_text(json.dumps(asdict(report), indent=2))
        sys.stdout.write(format_markdown_report(report))
        return 0

    if args.subcommand == "analyze":
        report = analyze_output_dir(args.output, server_logs=args.server_log)
        out_json = args.json_out or (args.output / "summary.json")
        out_json.write_text(json.dumps(asdict(report), indent=2))
        sys.stdout.write(format_markdown_report(report))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
