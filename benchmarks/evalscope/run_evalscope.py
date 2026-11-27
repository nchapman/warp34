#!/usr/bin/env python3
"""Managed warp34 + EvalScope evaluation runner for base-model & ARC-router sweeps.

Boots `warp34 serve` on the selected GPU(s), runs a standardized EvalScope suite
(`ifeval`, `gpqa_diamond`, `math_500`, `aime25`) via the OpenAI-compatible chat API,
captures `/api/metrics` deltas per benchmark (decode tok/s, prompt/output tokens, and
ARC router local/cloud split + USD spend when `--router` is enabled), and writes
`summary.json` + `summary.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
WARP34_BIN = REPO_ROOT / "target" / "release" / "warp34"

MATH_LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]


def pick_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def fetch_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_server(proc: subprocess.Popen[str], base_url: str, log_path: pathlib.Path, timeout_s: float = 180.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-3000:] if log_path.exists() else ""
            raise RuntimeError(f"warp34 exited early with code {proc.returncode}:\n{tail}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"warp34 did not become healthy within {timeout_s}s (see {log_path})")


def build_preset_suites(preset: str, custom_tasks: list[str] | None) -> list[dict[str, Any]]:
    """Returns per-benchmark `(dataset, limit, dataset_args)` specs."""
    if preset == "smoke":
        specs = [
            {"dataset": "ifeval", "limit": 2, "dataset_args": {}},
            {"dataset": "gpqa_diamond", "limit": 2, "dataset_args": {}},
            {"dataset": "math_500", "limit": 1, "dataset_args": {"math_500": {"few_shot_num": 0, "subset_list": ["Level 1", "Level 5"]}}},
            {"dataset": "aime25", "limit": 2, "dataset_args": {}},
        ]
    elif preset == "screen":
        # 50 ifeval + 50 gpqa_diamond + 50 stratified math_500 (10 x L1..L5) + full 30 aime25 = 180 prompts.
        specs = [
            {"dataset": "ifeval", "limit": 50, "dataset_args": {}},
            {"dataset": "gpqa_diamond", "limit": 50, "dataset_args": {}},
            {"dataset": "math_500", "limit": 10, "dataset_args": {"math_500": {"few_shot_num": 0, "subset_list": MATH_LEVELS}}},
            {"dataset": "aime25", "limit": None, "dataset_args": {}},
        ]
    elif preset == "full":
        specs = [
            {"dataset": "ifeval", "limit": None, "dataset_args": {}},
            {"dataset": "gpqa_diamond", "limit": None, "dataset_args": {}},
            {"dataset": "math_500", "limit": None, "dataset_args": {"math_500": {"few_shot_num": 0, "subset_list": MATH_LEVELS}}},
            {"dataset": "aime25", "limit": None, "dataset_args": {}},
        ]
    else:
        raise ValueError(f"unknown preset: {preset}")

    if custom_tasks:
        wanted = set(custom_tasks)
        specs = [s for s in specs if s["dataset"] in wanted]
    return specs


def extract_evalscope_score(
    report_obj: Any, work_dir: pathlib.Path, dataset: str, min_mtime: float | None = None
) -> dict[str, Any]:
    """Extracts primary score, per-subset breakdown, and client perf stats from EvalScope v2 report JSON."""
    reports_dir = work_dir / "reports"
    report_files = list(reports_dir.rglob(f"*{dataset}*.json")) if reports_dir.exists() else []
    if min_mtime is not None:
        fresh = [p for p in report_files if p.stat().st_mtime >= min_mtime - 1.0]
        if fresh:
            report_files = fresh
    if report_files:
        chosen = max(report_files, key=lambda p: p.stat().st_mtime)
        raw = json.loads(chosen.read_text())
        primary_id = raw.get("primary_metric_identity") or {}
        primary_name = primary_id.get("name")
        metrics_list = raw.get("metrics") or []
        score = raw.get("score")
        metrics_out: dict[str, Any] = {}
        for idx, m in enumerate(metrics_list):
            ident = m.get("identity") or {}
            m_name = ident.get("name") or m.get("name") or f"metric_{idx}"
            m_score = m.get("score")
            if score is None and (m_name == primary_name or (primary_name is None and idx == 0)):
                score = m_score
            metrics_out[m_name] = {
                "score": m_score,
                "num": m.get("num"),
                "num_correct": m.get("num_correct"),
                "subsets": {
                    sub.get("name", "default"): {
                        "score": sub.get("score"),
                        "num": sub.get("num"),
                        "num_correct": sub.get("num_correct"),
                    }
                    for cat in (m.get("categories") or [])
                    for sub in (cat.get("subsets") or [])
                },
            }
        if score is None and metrics_list:
            score = metrics_list[0].get("score")
        perf_summary = (raw.get("perf_metrics") or {}).get("summary") or {}
        usage = perf_summary.get("usage") or {}
        thpt = perf_summary.get("throughput") or {}
        lat = perf_summary.get("latency") or {}
        client_perf = {
            "n_samples": perf_summary.get("n_samples"),
            "mean_latency_s": lat.get("mean"),
            "avg_output_tps": thpt.get("avg_output_tps"),
            "total_input_tokens": usage.get("total_input_tokens"),
            "total_output_tokens": usage.get("total_output_tokens"),
        }
        return {
            "score": score,
            "primary_metric": primary_name,
            "metrics": metrics_out,
            "client_perf": client_perf,
            "execution_summary": raw.get("execution_summary"),
            "report_file": str(chosen),
        }

    # Fallback to in-memory return value from run_task
    if isinstance(report_obj, dict) and dataset in report_obj:
        r = report_obj[dataset]
        score = getattr(r, "score", None) if not isinstance(r, dict) else r.get("score")
        return {"score": score, "metrics": {}, "client_perf": {}}
    return {"score": None, "metrics": {}, "client_perf": {}}


def metrics_delta(before: dict[str, Any], after: dict[str, Any], wall_s: float) -> dict[str, Any]:
    """Computes local engine + MTP + ARC router deltas between two `/api/metrics` snapshots."""
    tok_b = before.get("tokens") or {}
    tok_a = after.get("tokens") or {}
    req_b = before.get("requests") or {}
    req_a = after.get("requests") or {}
    spec_b = before.get("speculation") or {}
    spec_a = after.get("speculation") or {}

    prompt_toks = int(tok_a.get("prompt", 0)) - int(tok_b.get("prompt", 0))
    cached_toks = int(tok_a.get("cached", 0)) - int(tok_b.get("cached", 0))
    gen_toks = int(tok_a.get("generated", 0)) - int(tok_b.get("generated", 0))
    reqs_ok = int(req_a.get("completed", 0)) - int(req_b.get("completed", 0))
    reqs_fail = int(req_a.get("failed", 0)) - int(req_b.get("failed", 0))

    spec_rounds = int(spec_a.get("rounds", 0)) - int(spec_b.get("rounds", 0))
    spec_accepted = int(spec_a.get("accepted_tokens", 0)) - int(spec_b.get("accepted_tokens", 0))
    spec_drafted = int(spec_a.get("draft_tokens") or 0) - int(spec_b.get("draft_tokens") or 0)

    out: dict[str, Any] = {
        "wall_seconds": round(wall_s, 2),
        "local_prompt_tokens": prompt_toks,
        "local_cached_tokens": cached_toks,
        "local_generation_tokens": gen_toks,
        "effective_local_gen_tok_per_s": round(gen_toks / wall_s, 2) if wall_s > 0 else 0.0,
        "requests_completed": reqs_ok,
        "requests_failed": reqs_fail,
        "speculation_rounds": spec_rounds,
        "speculation_accepted_tokens": spec_accepted,
        "speculation_draft_tokens": spec_drafted,
        "mtp_accept_rate": round(spec_accepted / spec_drafted, 4) if spec_drafted > 0 else None,
    }
    ar_after = after.get("arc_router")
    ar_before = before.get("arc_router") or {}
    if isinstance(ar_after, dict):
        b_models = {m["model"]: m for m in (ar_before.get("models") or []) if "model" in m}
        model_deltas = []
        for m in ar_after.get("models") or []:
            name = m.get("model", "")
            bm = b_models.get(name, {})
            d_req = int(m.get("requests", 0)) - int(bm.get("requests", 0))
            if d_req > 0:
                model_deltas.append(
                    {
                        "model": name,
                        "is_local": bool(m.get("is_local", False)),
                        "requests": d_req,
                        "prompt_tokens": int(m.get("prompt_tokens", 0)) - int(bm.get("prompt_tokens", 0)),
                        "completion_tokens": int(m.get("completion_tokens", 0)) - int(bm.get("completion_tokens", 0)),
                        "spend_usd": round(float(m.get("spend_usd", 0.0)) - float(bm.get("spend_usd", 0.0)), 6),
                    }
                )
        out["arc_router"] = {
            "total_requests": int(ar_after.get("total_requests", 0)) - int(ar_before.get("total_requests", 0)),
            "local_requests": int(ar_after.get("local_requests", 0)) - int(ar_before.get("local_requests", 0)),
            "cloud_requests": int(ar_after.get("cloud_requests", 0)) - int(ar_before.get("cloud_requests", 0)),
            "warm_kv_stayed_local": int(ar_after.get("warm_kv_stayed_local", 0))
            - int(ar_before.get("warm_kv_stayed_local", 0)),
            "pii_pinned_requests": int(ar_after.get("pii_pinned_requests", 0))
            - int(ar_before.get("pii_pinned_requests", 0)),
            "effort": {
                "none": int(ar_after.get("effort_none", 0)) - int(ar_before.get("effort_none", 0)),
                "low": int(ar_after.get("effort_low", 0)) - int(ar_before.get("effort_low", 0)),
                "medium": int(ar_after.get("effort_medium", 0)) - int(ar_before.get("effort_medium", 0)),
                "high": int(ar_after.get("effort_high", 0)) - int(ar_before.get("effort_high", 0)),
            },
            "router_fallbacks": int(ar_after.get("router_fallbacks", 0)) - int(ar_before.get("router_fallbacks", 0)),
            "cloud_upstream_errors": int(ar_after.get("cloud_upstream_errors", 0))
            - int(ar_before.get("cloud_upstream_errors", 0)),
            "spend_usd": round(
                float(ar_after.get("total_spend_usd", 0.0)) - float(ar_before.get("total_spend_usd", 0.0)), 6
            ),
            "models": model_deltas,
        }
    return out


def format_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"# EvalScope Summary — `{summary['label']}`",
        "",
        f"- **Model:** `{summary['model']}` (queried as `{summary['api_model']}`)",
        f"- **GPU(s):** `CUDA_VISIBLE_DEVICES={summary['gpu']}` (`tp={summary['tp']}`, `speculative={summary['speculative']}`)",
        f"- **Preset:** `{summary['preset']}` (`eval_batch_size={summary['eval_batch_size']}`, `max_tokens={summary['max_tokens']}`)",
        f"- **Router:** `{'enabled (' + (summary['router_cloud_models'] or 'local-only') + ')' if summary['router'] else 'disabled (base model direct)'}`",
        f"- **Total Wall Time:** `{summary['total_wall_seconds']:.1f}s`",
        "",
        "| Benchmark | Score | Out Tokens | Wall (s) | Local tok/s | MTP Accept | Cloud % | Cloud Spend ($) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    scores = []
    for task in summary["tasks"]:
        sc = task.get("score")
        if isinstance(sc, (int, float)):
            # EvalScope scores are 0..1 floats in report JSON; normalize to percentage for readability
            sc_pct = sc * 100.0 if sc <= 1.0 else sc
            scores.append(sc_pct)
            sc_str = f"{sc_pct:.2f}%"
        else:
            sc_str = "N/A"
        perf = task["telemetry"]
        cperf = task.get("client_perf") or {}
        c_out = cperf.get("total_output_tokens")
        out_toks = c_out if c_out is not None else perf["local_generation_tokens"]
        mtp_acc = perf.get("mtp_accept_rate")
        mtp_str = f"{100.0 * mtp_acc:.1f}%" if isinstance(mtp_acc, (int, float)) else "—"
        ar = perf.get("arc_router")
        if ar and ar["total_requests"] > 0:
            cloud_pct = f"{100.0 * ar['cloud_requests'] / ar['total_requests']:.1f}%"
            spend = f"${ar['spend_usd']:.4f}"
        else:
            cloud_pct = "—"
            spend = "—"
        lines.append(
            f"| `{task['dataset']}` | **{sc_str}** | {out_toks:,} | "
            f"{perf['wall_seconds']:.1f} | {perf['effective_local_gen_tok_per_s']:.1f} | {mtp_str} | {cloud_pct} | {spend} |"
        )
    if scores:
        mean_pct = sum(scores) / len(scores)
        ov_mtp = summary["overall_telemetry"].get("mtp_accept_rate")
        ov_mtp_str = f"{100.0 * ov_mtp:.1f}%" if isinstance(ov_mtp, (int, float)) else "—"
        ov_ar = summary["overall_telemetry"].get("arc_router")
        if ov_ar and ov_ar["total_requests"] > 0:
            ov_cloud = f"{100.0 * ov_ar['cloud_requests'] / ov_ar['total_requests']:.1f}%"
            ov_spend = f"${ov_ar['spend_usd']:.4f}"
        else:
            ov_cloud = "—"
            ov_spend = "—"
        lines.append(
            f"| **Macro Average** | **{mean_pct:.2f}%** | {summary['total_output_tokens']:,} | "
            f"{summary['total_wall_seconds']:.1f} | {summary['overall_local_gen_tok_per_s']:.1f} | {ov_mtp_str} | {ov_cloud} | {ov_spend} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_compare(dirs: list[str]) -> int:
    rows = []
    all_datasets: list[str] = []
    for d in dirs:
        p = pathlib.Path(d)
        sj = p / "summary.json" if p.is_dir() else p
        data = json.loads(sj.read_text())
        rows.append(data)
        for t in data["tasks"]:
            if t["dataset"] not in all_datasets:
                all_datasets.append(t["dataset"])

    hdr = "| Candidate | " + " | ".join(f"`{ds}`" for ds in all_datasets) + " | Macro Avg | Total Wall (s) | Eff. tok/s | Cloud Spend |"
    sep = "| :--- | " + " | ".join("---:" for _ in all_datasets) + " | ---: | ---: | ---: | ---: |"
    print(hdr)
    print(sep)
    for data in rows:
        by_ds = {t["dataset"]: t for t in data["tasks"]}
        cells = []
        vals = []
        total_spend = 0.0
        has_router = False
        for ds in all_datasets:
            t = by_ds.get(ds)
            sc = t.get("score") if t else None
            if isinstance(sc, (int, float)):
                pct = sc * 100.0 if sc <= 1.0 else sc
                vals.append(pct)
                cells.append(f"{pct:.2f}%")
            else:
                cells.append("—")
            if t and "arc_router" in t.get("telemetry", {}):
                has_router = True
                total_spend += t["telemetry"]["arc_router"]["spend_usd"]
        avg_s = f"**{sum(vals) / len(vals):.2f}%**" if vals else "—"
        spend_s = f"${total_spend:.4f}" if has_router else "—"
        print(
            f"| `{data['label']}` | "
            + " | ".join(cells)
            + f" | {avg_s} | {data['total_wall_seconds']:.1f}s | {data['overall_local_gen_tok_per_s']:.1f} | {spend_s} |"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare", nargs="+", help="Compare two or more completed run directories")
    ap.add_argument("--label", default=None, help="Short run label for tables")
    ap.add_argument("--model", default="nchapman/Apex-35B-V1.0-W4A16-HGPTQ", help="Local HF repo or path")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value (e.g. '0', '1', '0,1')")
    ap.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
    ap.add_argument("--dp", type=int, default=1, help="Data parallelism degree")
    ap.add_argument("--speculative", default="mtp", choices=["auto", "mtp", "dflash2", "off"])
    ap.add_argument("--draft-model", default=None, help="Optional DFlash2 drafter repo/path")
    ap.add_argument("--kv-format", default="tq4")
    ap.add_argument("--kv-pages", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--reasoning-effort", default="high", help="Server default reasoning effort (high = 80%% cap + 20%% answer reserve)")
    ap.add_argument("--port", type=int, default=0, help="HTTP port (0 = auto free port)")
    ap.add_argument("--preset", default="screen", choices=["smoke", "screen", "full"])
    ap.add_argument("--tasks", nargs="*", default=None, help="Optional subset of tasks to run")
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--temperature", type=float, default=None, help="Override sampling temperature (default: model generation_config.json)")
    ap.add_argument("--top-p", type=float, default=None, help="Override top_p (default: model generation_config.json)")
    ap.add_argument("--top-k", type=int, default=None, help="Override top_k (default: model generation_config.json)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offline", action="store_true", default=True, help="Pass --offline to warp34 serve")
    ap.add_argument("--online-hub", action="store_true", help="Allow HF Hub network checks in warp34 serve")
    ap.add_argument("--router", action="store_true", help="Enable native ARC-1.0 router")
    ap.add_argument("--router-cloud-models", default=None, help="--router-cloud-models value when --router is set")
    ap.add_argument("--router-policy", default=None, help="--router-policy (prefer-local, sage, swift)")
    ap.add_argument("--router-min-p-pass", type=float, default=None)
    ap.add_argument("--router-value", type=float, default=None)
    ap.add_argument("--work-dir", required=False, default=None, help="Output directory")
    args = ap.parse_args()

    if args.compare:
        return run_compare(args.compare)

    from evalscope import TaskConfig, run_task

    short_model = args.model.rstrip("/").split("/")[-1]
    label = args.label or (f"{short_model}+router" if args.router else short_model)
    ts = time.strftime("%Y%m%d-%H%M%S")
    work_dir = pathlib.Path(args.work_dir) if args.work_dir else (REPO_ROOT / "outputs" / f"evalscope-{label}-{ts}")
    work_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = work_dir / "warp34-serve.log"

    port = args.port or pick_free_port()
    base_url = f"http://127.0.0.1:{port}"
    api_model = "router" if args.router else args.model

    cmd = [
        str(WARP34_BIN),
        "serve",
        "--model",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp",
        str(args.tp),
        "--dp",
        str(args.dp),
        "--speculative",
        args.speculative,
        "--kv-format",
        args.kv_format,
        "--kv-pages",
        str(args.kv_pages),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        str(args.max_model_len),
        "--reasoning-effort",
        args.reasoning_effort,
    ]
    if args.speculative == "mtp":
        cmd += ["--mtp-k", "3"]
    if args.draft_model:
        cmd += ["--draft-model", args.draft_model]
    if args.offline and not args.online_hub:
        cmd.append("--offline")
    if args.router:
        cmd.append("--router")
        if args.router_cloud_models:
            cmd += ["--router-cloud-models", args.router_cloud_models]
        if args.router_policy:
            cmd += ["--router-policy", args.router_policy]
        if args.router_min_p_pass is not None:
            cmd += ["--router-min-p-pass", str(args.router_min_p_pass)]
        if args.router_value is not None:
            cmd += ["--router-value", str(args.router_value)]

    env = os.environ.copy()
    env["PATH"] = f"/opt/cuda/bin:{env.get('PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"[{label}] Starting warp34 on GPU {args.gpu} port {port} -> {work_dir}", flush=True)
    proc: subprocess.Popen[str] | None = None
    with open(server_log_path, "w", encoding="utf-8") as server_log:
        try:
            proc = subprocess.Popen(cmd, env=env, stdout=server_log, stderr=subprocess.STDOUT, text=True)
            wait_for_server(proc, base_url, server_log_path)
            print(f"[{label}] Server healthy at {base_url}; starting EvalScope `{args.preset}` suite...", flush=True)

            suites = build_preset_suites(args.preset, args.tasks)
            task_results: list[dict[str, Any]] = []
            had_task_error = False
            t_suite_start = time.time()
            m_suite_before = fetch_json(f"{base_url}/api/metrics")

            for spec in suites:
                ds = spec["dataset"]
                ds_work_dir = work_dir / f"eval_{ds}"
                ds_work_dir.mkdir(parents=True, exist_ok=True)
                print(f"[{label}] >>> Running `{ds}` (limit={spec['limit']}) ...", flush=True)

                t0 = time.time()
                try:
                    m_before = fetch_json(f"{base_url}/api/metrics")
                    gen_cfg: dict[str, Any] = {
                        "max_tokens": args.max_tokens,
                        "n": 1,
                    }
                    if args.temperature is not None:
                        gen_cfg["temperature"] = args.temperature
                    if args.top_p is not None:
                        gen_cfg["top_p"] = args.top_p
                    if args.top_k is not None:
                        gen_cfg["top_k"] = args.top_k

                    cfg_kwargs: dict[str, Any] = {
                        "model": api_model,
                        "api_url": f"{base_url}/v1",
                        "api_key": "EMPTY",
                        "eval_type": "openai_api",
                        "datasets": [ds],
                        "dataset_args": spec["dataset_args"],
                        "eval_batch_size": args.eval_batch_size,
                        "seed": args.seed,
                        "timeout": 3600,
                        "work_dir": str(ds_work_dir),
                        "no_timestamp": True,
                        "generation_config": gen_cfg,
                    }
                    if spec["limit"] is not None:
                        cfg_kwargs["limit"] = spec["limit"]

                    report_obj = run_task(TaskConfig(**cfg_kwargs))
                    wall_s = time.time() - t0
                    m_after = fetch_json(f"{base_url}/api/metrics")

                    extracted = extract_evalscope_score(report_obj, ds_work_dir, ds, min_mtime=t0)
                    telem = metrics_delta(m_before, m_after, wall_s)
                    sc = extracted["score"]
                    sc_disp = (
                        f"{(sc * 100.0 if isinstance(sc, (int, float)) and sc <= 1.0 else sc):.2f}%"
                        if isinstance(sc, (int, float))
                        else str(sc)
                    )
                    print(
                        f"[{label}] <<< `{ds}` done in {wall_s:.1f}s: score={sc_disp}, "
                        f"local_gen_toks={telem['local_generation_tokens']:,}, "
                        f"eff_tok_s={telem['effective_local_gen_tok_per_s']:.1f}, "
                        f"mtp_accept={telem['mtp_accept_rate']}",
                        flush=True,
                    )
                    task_results.append(
                        {
                            "dataset": ds,
                            "limit": spec["limit"],
                            "score": extracted["score"],
                            "primary_metric": extracted.get("primary_metric"),
                            "metrics": extracted["metrics"],
                            "client_perf": extracted.get("client_perf"),
                            "execution_summary": extracted.get("execution_summary"),
                            "report_file": extracted.get("report_file"),
                            "telemetry": telem,
                        }
                    )
                except Exception as exc:
                    had_task_error = True
                    wall_s = time.time() - t0
                    print(f"[{label}] !!! `{ds}` failed after {wall_s:.1f}s: {exc}", file=sys.stderr, flush=True)
                    task_results.append(
                        {
                            "dataset": ds,
                            "limit": spec["limit"],
                            "score": None,
                            "error": str(exc),
                            "metrics": {},
                            "client_perf": {},
                            "telemetry": {
                                "wall_seconds": round(wall_s, 2),
                                "local_prompt_tokens": 0,
                                "local_cached_tokens": 0,
                                "local_generation_tokens": 0,
                                "effective_local_gen_tok_per_s": 0.0,
                                "requests_completed": 0,
                                "requests_failed": 0,
                                "speculation_rounds": 0,
                                "speculation_accepted_tokens": 0,
                                "speculation_draft_tokens": 0,
                                "mtp_accept_rate": None,
                            },
                        }
                    )

            total_wall_s = time.time() - t_suite_start
            try:
                m_suite_after = fetch_json(f"{base_url}/api/metrics")
                overall = metrics_delta(m_suite_before, m_suite_after, total_wall_s)
            except Exception:
                overall = metrics_delta(m_suite_before, m_suite_before, total_wall_s)

            c_outs = [
                (t.get("client_perf") or {}).get("total_output_tokens")
                for t in task_results
                if (t.get("client_perf") or {}).get("total_output_tokens") is not None
            ]
            total_out_toks = sum(c_outs) if c_outs else overall["local_generation_tokens"]

            summary = {
                "label": label,
                "model": args.model,
                "api_model": api_model,
                "gpu": args.gpu,
                "tp": args.tp,
                "dp": args.dp,
                "speculative": args.speculative,
                "preset": args.preset,
                "eval_batch_size": args.eval_batch_size,
                "max_tokens": args.max_tokens,
                "router": args.router,
                "router_cloud_models": args.router_cloud_models,
                "router_policy": args.router_policy,
                "total_wall_seconds": round(total_wall_s, 2),
                "total_local_prompt_tokens": overall["local_prompt_tokens"],
                "total_local_generation_tokens": overall["local_generation_tokens"],
                "total_output_tokens": total_out_toks,
                "overall_local_gen_tok_per_s": overall["effective_local_gen_tok_per_s"],
                "overall_telemetry": overall,
                "tasks": task_results,
            }
            (work_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            md = format_markdown_summary(summary)
            (work_dir / "summary.md").write_text(md)
            print("\n" + md, flush=True)
            return 1 if had_task_error else 0
        finally:
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
