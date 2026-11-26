#!/usr/bin/env python3
"""Managed live-server release soak runner for warp34.

Boots `warp34 serve` per subphase in an isolated process group, drives realistic
multi-agent / concurrency / cancellation / image workloads over HTTP, and verifies
strict post-drain resource conservation via `/api/metrics` + `nvidia-smi` + RSS.

Subphases:
  1. `swe_27b_dflash2`    — 27B TP=2 + DFlash2 multi-agent Open-SWE replay +
                            interleaved `logprobs` + mid-stream SSE cancellations.
  2. `dp2_4b_concurrency` — 4B `--dp 2` prefix-sharing, balanced worker dispatch,
                            and cancellation leak check across both replicas.
  3. `qwen_image_http`    — Qwen-Image-2.1-Turbo `/v1/images/generations` (RGB +
                            RGBA) and `/v1/images/edits` PNG validation.
  4. `bonsai_5trit`       — PrismML Ternary Bonsai 5-Trit (`PTQ1_0`) concurrent
                            SSE chat + cancellation conservation smoke.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import dataclasses
import http.client
import json
import os
import pathlib
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_WARP34_BIN = REPO_ROOT / "target" / "release" / "warp34"
DEFAULT_SWE_FIXTURE = REPO_ROOT / "benchmarks" / "perf" / "fixtures" / "open_swe_intact_16k.json"
DEFAULT_SUBPHASES: tuple[str, ...] = (
    "swe_27b_dflash2",
    "dp2_4b_concurrency",
    "qwen_image_http",
    "bonsai_5trit",
)
BONSAI_CANDIDATE_MODELS: tuple[str, ...] = (
    "nchapman/Ternary-Bonsai-2-4B-warp34",
    "nchapman/Ternary-Bonsai-2-9B-warp34",
    "nchapman/Ternary-Bonsai-2-4B-5Trit",
    "nchapman/Ternary-Bonsai-2-9B-5Trit",
)


@dataclasses.dataclass(frozen=True)
class SubphaseResult:
    name: str
    passed: bool
    skipped: bool
    elapsed_s: float
    requests_sent: int
    requests_ok: int
    requests_cancelled: int
    requests_failed: int
    output_tokens: int
    output_tok_per_s: float
    prefix_cache_hit_pct: float | None
    mean_accepted_drafts: float | None
    vram_growth_mib: float | None
    rss_growth_mib: float | None
    failures: tuple[str, ...]
    server_log: str | None = None


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def make_test_png(width: int = 64, height: int = 64) -> bytes:
    """Create a valid RGB PNG in memory using only Python stdlib."""
    raw_rows = bytearray()
    for y in range(height):
        raw_rows.append(0)  # filter type 0 (None)
        for x in range(width):
            raw_rows.extend(((x * 4) % 256, (y * 4) % 256, 128))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw_rows), level=6)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def parse_png_dimensions(png_bytes: bytes) -> tuple[int, int, int] | None:
    """Return `(width, height, color_type)` for a valid PNG, else `None`."""
    if len(png_bytes) < 33 or png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    if png_bytes[12:16] != b"IHDR":
        return None
    width, height, _bit_depth, color_type = struct.unpack(">IIBB", png_bytes[16:26])
    return width, height, color_type


def query_vram_used_mib(cuda_visible_devices: str | None = None) -> float | None:
    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    cvd = (
        cuda_visible_devices
        if cuda_visible_devices is not None
        else os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if cvd is not None:
        cleaned = cvd.strip()
        if not cleaned or cleaned in ("-1", "none", "NoDevFiles"):
            return None
        cmd.extend(["-i", cleaned])
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=5.0)
        vals = [float(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
        return sum(vals) if vals else None
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def query_process_rss_mib(pid: int) -> float | None:
    status_path = pathlib.Path(f"/proc/{pid}/status")
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024.0
    except OSError:
        return None
    return None


def _hf_snapshot_dirs(repo_id: str) -> list[pathlib.Path]:
    hub_dir = pathlib.Path(
        os.environ.get("HF_HUB_CACHE")
        or (
            pathlib.Path(os.environ["HF_HOME"]) / "hub"
            if os.environ.get("HF_HOME")
            else pathlib.Path.home() / ".cache" / "huggingface" / "hub"
        )
    )
    snapshots = hub_dir / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted([p for p in snapshots.iterdir() if p.is_dir()], reverse=True)


def find_cached_hf_repo(repo_id: str, *, kind: str = "llm") -> bool:
    """Return True iff a complete cached snapshot with weights/configs exists for `repo_id`."""
    for snap in _hf_snapshot_dirs(repo_id):
        if kind == "image":
            if (
                (snap / "model_index.json").is_file()
                and (snap / "transformer" / "config.json").is_file()
                and (snap / "vae" / "config.json").is_file()
            ):
                return True
        else:
            has_weights = (snap / "model.safetensors").is_file() or (
                snap / "model.safetensors.index.json"
            ).is_file()
            if (snap / "config.json").is_file() and (snap / "tokenizer.json").is_file() and has_weights:
                return True
    return False


def repo_supports_mtp(repo_id: str) -> bool:
    """Return True if the cached checkpoint advertises MTP layers or carries `model_mtp.safetensors`."""
    for snap in _hf_snapshot_dirs(repo_id):
        if (snap / "model_mtp.safetensors").is_file():
            return True
        cfg_path = snap / "config.json"
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
                if int(text_cfg.get("mtp_num_hidden_layers", 0)) > 0:
                    return True
            except (OSError, ValueError, TypeError):
                pass
    return False


def evaluate_metrics_conservation(
    metrics: dict[str, Any],
    *,
    expect_dp_workers: int | None = None,
    min_prefix_cache_hit_pct: float | None = None,
    min_mean_accepted_drafts: float | None = None,
    vram_growth_mib: float | None = None,
    max_vram_growth_mib: float | None = None,
    rss_growth_mib: float | None = None,
    max_rss_growth_mib: float | None = None,
    output_tok_per_s: float | None = None,
    min_output_tok_per_s: float | None = None,
) -> tuple[list[str], float | None, float | None]:
    """Validate `/api/metrics` post-drain conservation and performance floors.

    Returns `(failures, prefix_cache_hit_pct, mean_accepted_drafts)`.
    """
    failures: list[str] = []

    def _check_snapshot(snap: dict[str, Any], label: str) -> None:
        reqs = snap.get("requests") or {}
        http_r = snap.get("http_responses") or {}
        pools = snap.get("pools") or {}
        if int(reqs.get("running", 0)) != 0:
            failures.append(f"{label}: requests.running={reqs.get('running')} != 0 after drain")
        if int(reqs.get("waiting", 0)) != 0:
            failures.append(f"{label}: requests.waiting={reqs.get('waiting')} != 0 after drain")
        if int(reqs.get("failed", 0)) != 0:
            failures.append(f"{label}: requests.failed={reqs.get('failed')} != 0")
        if int(http_r.get("server_error", 0)) != 0:
            failures.append(
                f"{label}: http_responses.server_error={http_r.get('server_error')} != 0"
            )
        kv = pools.get("kv") or {}
        if int(kv.get("capacity", 0)) > 0:
            if kv.get("active") is None:
                failures.append(f"{label}: pools.kv.active is null (allocator accounting underflow)")
            elif int(kv["active"]) != 0:
                failures.append(f"{label}: leaked KV pages pools.kv.active={kv['active']} != 0")
        gdn = pools.get("gdn") or {}
        if int(gdn.get("capacity", 0)) > 0:
            if gdn.get("active") is None:
                failures.append(f"{label}: pools.gdn.active is null (allocator accounting underflow)")
            elif int(gdn["active"]) != 0:
                failures.append(f"{label}: leaked GDN slots pools.gdn.active={gdn['active']} != 0")

    _check_snapshot(metrics, "aggregate")

    workers = metrics.get("workers") or []
    if expect_dp_workers is not None:
        if len(workers) != expect_dp_workers:
            failures.append(
                f"expected {expect_dp_workers} DP workers in /api/metrics, got {len(workers)}"
            )
        for idx, worker in enumerate(workers):
            _check_snapshot(worker, f"dp_worker[{idx}]")
            w_reqs = worker.get("requests") or {}
            if int(w_reqs.get("completed", 0)) <= 0:
                failures.append(
                    f"dp_worker[{idx}]: completed={w_reqs.get('completed', 0)} (unbalanced dispatch)"
                )

    tokens = metrics.get("tokens") or {}
    prompt_toks = int(tokens.get("prompt", 0))
    cached_toks = int(tokens.get("cached", 0))
    cache_hit_pct = (100.0 * cached_toks / prompt_toks) if prompt_toks > 0 else None
    if min_prefix_cache_hit_pct is not None:
        if cache_hit_pct is None or cache_hit_pct < min_prefix_cache_hit_pct:
            actual = f"{cache_hit_pct:.1f}%" if cache_hit_pct is not None else "n/a"
            failures.append(
                f"prefix_cache_hit_pct {actual} < floor {min_prefix_cache_hit_pct:.1f}%"
            )

    spec = metrics.get("speculation") or {}
    rounds = int(spec.get("rounds", 0))
    accepted = int(spec.get("accepted_tokens", 0))
    mean_acc = (accepted / rounds) if rounds > 0 else None
    if min_mean_accepted_drafts is not None:
        if mean_acc is None or mean_acc < min_mean_accepted_drafts:
            actual = f"{mean_acc:.2f}" if mean_acc is not None else "n/a"
            failures.append(
                f"mean_accepted_drafts {actual} < floor {min_mean_accepted_drafts:.2f}"
            )

    if (
        max_vram_growth_mib is not None
        and vram_growth_mib is not None
        and vram_growth_mib > max_vram_growth_mib
    ):
        failures.append(
            f"VRAM growth {vram_growth_mib:.1f} MiB > ceiling {max_vram_growth_mib:.1f} MiB"
        )
    if (
        max_rss_growth_mib is not None
        and rss_growth_mib is not None
        and rss_growth_mib > max_rss_growth_mib
    ):
        failures.append(
            f"host RSS growth {rss_growth_mib:.1f} MiB > ceiling {max_rss_growth_mib:.1f} MiB"
        )
    if (
        min_output_tok_per_s is not None
        and output_tok_per_s is not None
        and output_tok_per_s < min_output_tok_per_s
    ):
        failures.append(
            f"output throughput {output_tok_per_s:.1f} tok/s < floor {min_output_tok_per_s:.1f} tok/s"
        )

    return failures, cache_hit_pct, mean_acc


class ManagedServer:
    """Context manager that runs `warp34 serve` in its own process group."""

    def __init__(
        self,
        cmd: list[str],
        host: str,
        port: int,
        log_path: pathlib.Path,
        env: dict[str, str] | None = None,
        startup_timeout_s: float = 300.0,
    ) -> None:
        self.cmd = cmd
        self.host = host
        self.port = port
        self.log_path = log_path
        self.env = env
        self.startup_timeout_s = startup_timeout_s
        self.proc: subprocess.Popen[Any] | None = None
        self._log_file: Any = None

    def __enter__(self) -> ManagedServer:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("w", encoding="utf-8")
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                env=self.env,
                start_new_session=True,
            )
            self._wait_for_health()
        except BaseException:
            self._cleanup()
            raise
        return self

    def _wait_for_health(self) -> None:
        assert self.proc is not None
        url = f"http://{self.host}:{self.port}/health"
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            rc = self.proc.poll()
            if rc is not None:
                raise RuntimeError(
                    f"warp34 serve exited early with code {rc}; see {self.log_path}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2.0) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            time.sleep(0.5)
        raise RuntimeError(
            f"timed out waiting {self.startup_timeout_s:.0f}s for {url}; see {self.log_path}"
        )

    def fetch_metrics(self) -> dict[str, Any]:
        url = f"http://{self.host}:{self.port}/api/metrics"
        with urllib.request.urlopen(url, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _is_snapshot_drained(self, snap: dict[str, Any]) -> bool:
        reqs = snap.get("requests") or {}
        if int(reqs.get("running", 0)) != 0 or int(reqs.get("waiting", 0)) != 0:
            return False
        pools = snap.get("pools") or {}
        for key in ("kv", "gdn"):
            pool = pools.get(key) or {}
            if int(pool.get("capacity", 0)) > 0 and pool.get("active") not in (0, None):
                return False
        for worker in snap.get("workers") or []:
            if not self._is_snapshot_drained(worker):
                return False
        return True

    def wait_for_drain(self, timeout_s: float = 30.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        last: dict[str, Any] = {}
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                last = self.fetch_metrics()
                last_err = None
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                last_err = exc
                time.sleep(0.25)
                continue
            if self._is_snapshot_drained(last):
                return last
            time.sleep(0.25)
        if not last and last_err is not None:
            raise RuntimeError(f"failed to fetch /api/metrics during drain: {last_err}") from last_err
        return last

    def _cleanup(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                pgid = os.getpgid(self.proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except OSError:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except OSError:
                    self.proc.kill()
                try:
                    self.proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __exit__(self, *_exc: Any) -> None:
        self._cleanup()


def send_chat_request(
    host: str,
    port: int,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 96,
    stream: bool = True,
    cancel_after_chunks: int | None = None,
    logprobs: bool = False,
    session_id: str | None = None,
    timeout_s: float = 120.0,
) -> tuple[str, int, str | None]:
    """Send one `/v1/chat/completions` request.

    Returns `(status, completion_chunks, error_msg)` where `status` is
    `"ok"`, `"cancelled"`, or `"error"`.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 2
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["x-session-id"] = session_id

    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
        resp = conn.getresponse()
        if resp.status != 200:
            err = resp.read().decode("utf-8", errors="replace")[:300]
            return "error", 0, f"HTTP {resp.status}: {err}"
        if not stream:
            data = json.loads(resp.read().decode("utf-8"))
            usage = data.get("usage") or {}
            return "ok", int(usage.get("completion_tokens", 1)), None

        chunks = 0
        saw_done = False
        saw_logprobs = False
        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                saw_done = True
                break
            chunks += 1
            if logprobs and '"logprobs"' in data_str and '"logprobs":null' not in data_str.replace(" ", ""):
                saw_logprobs = True
            if cancel_after_chunks is not None and chunks >= cancel_after_chunks:
                conn.close()
                return "cancelled", chunks, None
        if not saw_done:
            return "error", chunks, f"truncated SSE stream (no [DONE], chunks={chunks})"
        if chunks == 0:
            return "error", 0, "empty SSE stream (0 chunks before [DONE])"
        if logprobs and not saw_logprobs:
            return "error", chunks, "logprobs=true requested but no logprobs payload received"
        return "ok", chunks, None
    except Exception as exc:
        return "error", 0, str(exc)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def load_swe_trajectories(
    fixture_path: pathlib.Path,
    *,
    max_trajectories: int = 4,
    max_turns_per_trajectory: int = 4,
) -> list[list[list[dict[str, Any]]]]:
    """Load intact multi-turn Open-SWE trajectories from `open_swe_intact_16k.json`.

    Returns a list of trajectories; each trajectory is an ordered list of cumulative
    ChatML `messages` prefixes so worker threads replay turns sequentially and exercise
    real multi-turn prefix reuse in `MambaRadixCache`.
    """
    if not fixture_path.is_file():
        raise FileNotFoundError(f"SWE soak fixture not found: {fixture_path}")
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    raw_trajs = data.get("trajectories") if isinstance(data, dict) else data
    if not isinstance(raw_trajs, list):
        raise ValueError(f"malformed SWE fixture {fixture_path}: expected 'trajectories' list")

    parsed: list[list[list[dict[str, Any]]]] = []
    for traj in raw_trajs[:max_trajectories]:
        if not isinstance(traj, dict):
            continue
        turns = traj.get("turns")
        if not isinstance(turns, list) or not turns:
            continue
        history: list[dict[str, Any]] = []
        sys_prompt = traj.get("system")
        if isinstance(sys_prompt, str) and sys_prompt.strip():
            history.append({"role": "system", "content": sys_prompt})
        turn_prompts: list[list[dict[str, Any]]] = []
        for t_idx, turn in enumerate(turns[:max_turns_per_trajectory]):
            if not isinstance(turn, dict):
                continue
            if isinstance(turn.get("messages"), list) and turn["messages"]:
                turn_prompts.append(list(turn["messages"]))
                continue
            user_text = turn.get("user")
            if isinstance(user_text, str) and user_text.strip():
                history.append({"role": "user", "content": user_text})
            tools = turn.get("tools")
            if isinstance(tools, list) and tools:
                obs = "\n".join(str(t) for t in tools[:4])
                history.append({"role": "user", "content": f"Tool output:\n{obs}"})
            if history:
                turn_prompts.append(list(history))
                history.append(
                    {
                        "role": "assistant",
                        "content": f"Acknowledged step {t_idx + 1}; continuing investigation.",
                    }
                )
        if turn_prompts:
            parsed.append(turn_prompts)
    if not parsed:
        raise ValueError(f"SWE fixture {fixture_path} yielded 0 replayable trajectories")
    return parsed


def run_swe_27b_dflash2_soak(
    *,
    warp34_bin: pathlib.Path,
    out_dir: pathlib.Path,
    duration_scale: float,
    cuda_visible_devices: str | None,
) -> SubphaseResult:
    t0 = time.monotonic()
    port = pick_free_port()
    log_path = out_dir / "swe_27b_dflash2.server.log"
    model = "nchapman/Qwen3.8-27B-W4A16-HGPTQ"
    draft = "nchapman/Qwen3.8-27B-DFlash2-W4A16-HGPTQ"
    cmd = [
        str(warp34_bin),
        "serve",
        model,
        "--tp",
        "2",
        "--speculative",
        "dflash2",
        "--draft-model",
        draft,
        "--kv-format",
        "tq4",
        "--offline",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    turns_per_agent = max(3, int(round(4 * duration_scale)))
    trajectories = load_swe_trajectories(
        DEFAULT_SWE_FIXTURE,
        max_trajectories=4,
        max_turns_per_trajectory=turns_per_agent,
    )
    total_turns = sum(len(t) for t in trajectories)
    failures: list[str] = []
    ok_cnt = cancel_cnt = fail_cnt = 0

    try:
        with ManagedServer(cmd, "127.0.0.1", port, log_path) as srv:
            # B=4 warmup burst so CUDA graphs & workspaces settle before measuring VRAM/RSS delta.
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(trajectories)) as wpool:
                wfuts = [
                    wpool.submit(
                        send_chat_request,
                        "127.0.0.1",
                        port,
                        model,
                        traj[0],
                        max_tokens=24,
                        stream=True,
                        session_id=f"swe-warmup-{i}",
                    )
                    for i, traj in enumerate(trajectories)
                ]
                for fut in concurrent.futures.as_completed(wfuts):
                    st, _, err = fut.result()
                    if st != "ok":
                        failures.append(f"warmup failed: {err}")
            warmup_metrics = srv.wait_for_drain(timeout_s=30.0)
            gen_before = int((warmup_metrics.get("tokens") or {}).get("generated", 0))
            vram_before = query_vram_used_mib(cuda_visible_devices)
            rss_before = query_process_rss_mib(srv.proc.pid) if srv.proc else None

            def _run_agent(agent_idx: int, turn_list: list[list[dict[str, Any]]]) -> tuple[int, int, int, list[str]]:
                a_ok = a_cancel = a_fail = 0
                a_errs: list[str] = []
                for t_idx, msgs in enumerate(turn_list):
                    global_idx = agent_idx * len(turn_list) + t_idx
                    cancel_at = 3 if (global_idx % 7 == 6) else None
                    use_logprobs = (global_idx % 5 == 4) and cancel_at is None
                    st, _, err = send_chat_request(
                        "127.0.0.1",
                        port,
                        model,
                        msgs,
                        max_tokens=96,
                        stream=True,
                        cancel_after_chunks=cancel_at,
                        logprobs=use_logprobs,
                        session_id=f"swe-agent-{agent_idx}",
                    )
                    if st == "ok":
                        a_ok += 1
                    elif st == "cancelled":
                        a_cancel += 1
                    else:
                        a_fail += 1
                        a_errs.append(f"agent {agent_idx} turn {t_idx}: {err}")
                return a_ok, a_cancel, a_fail, a_errs

            load_t0 = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(trajectories)) as pool:
                futs = [pool.submit(_run_agent, i, traj) for i, traj in enumerate(trajectories)]
                for fut in concurrent.futures.as_completed(futs):
                    a_ok, a_cancel, a_fail, a_errs = fut.result()
                    ok_cnt += a_ok
                    cancel_cnt += a_cancel
                    fail_cnt += a_fail
                    failures.extend(a_errs)
            load_elapsed = max(0.001, time.monotonic() - load_t0)

            metrics = srv.wait_for_drain(timeout_s=30.0)
            gen_after = int((metrics.get("tokens") or {}).get("generated", 0))
            vram_after = query_vram_used_mib(cuda_visible_devices)
            rss_after = query_process_rss_mib(srv.proc.pid) if srv.proc else None

            # Post-drain deterministic probe.
            st, _, err = send_chat_request(
                "127.0.0.1",
                port,
                model,
                [{"role": "user", "content": "Reply with OK."}],
                max_tokens=16,
                stream=False,
            )
            if st != "ok":
                failures.append(f"post-drain probe failed: {err}")
            metrics = srv.wait_for_drain(timeout_s=15.0)

            vram_growth = (
                (vram_after - vram_before)
                if vram_before is not None and vram_after is not None
                else None
            )
            rss_growth = (
                (rss_after - rss_before)
                if rss_before is not None and rss_after is not None
                else None
            )
            gen_toks = max(0, gen_after - gen_before)
            tps = gen_toks / load_elapsed

            cons_failures, cache_hit, mean_acc = evaluate_metrics_conservation(
                metrics,
                min_prefix_cache_hit_pct=40.0,
                min_mean_accepted_drafts=1.2,
                vram_growth_mib=vram_growth,
                max_vram_growth_mib=256.0,
                rss_growth_mib=rss_growth,
                # 27B TP=2 lazily faults in the L2 pinned-host GDN checkpoint & projected ctx memo pool (~2 GiB).
                max_rss_growth_mib=2560.0,
            )
            failures.extend(cons_failures)
            return SubphaseResult(
                name="swe_27b_dflash2",
                passed=(len(failures) == 0 and fail_cnt == 0),
                skipped=False,
                elapsed_s=time.monotonic() - t0,
                requests_sent=total_turns,
                requests_ok=ok_cnt,
                requests_cancelled=cancel_cnt,
                requests_failed=fail_cnt,
                output_tokens=gen_toks,
                output_tok_per_s=tps,
                prefix_cache_hit_pct=cache_hit,
                mean_accepted_drafts=mean_acc,
                vram_growth_mib=vram_growth,
                rss_growth_mib=rss_growth,
                failures=tuple(failures),
                server_log=str(log_path),
            )
    except Exception as exc:
        failures.append(str(exc))
        return SubphaseResult(
            name="swe_27b_dflash2",
            passed=False,
            skipped=False,
            elapsed_s=time.monotonic() - t0,
            requests_sent=total_turns,
            requests_ok=ok_cnt,
            requests_cancelled=cancel_cnt,
            requests_failed=fail_cnt,
            output_tokens=0,
            output_tok_per_s=0.0,
            prefix_cache_hit_pct=None,
            mean_accepted_drafts=None,
            vram_growth_mib=None,
            rss_growth_mib=None,
            failures=tuple(failures),
            server_log=str(log_path),
        )


def run_dp2_4b_soak(
    *,
    warp34_bin: pathlib.Path,
    out_dir: pathlib.Path,
    duration_scale: float,
    cuda_visible_devices: str | None,
) -> SubphaseResult:
    t0 = time.monotonic()
    port = pick_free_port()
    log_path = out_dir / "dp2_4b_concurrency.server.log"
    model = "nchapman/Qwen3.5-4B-W4A16-HGPTQ"
    cmd = [
        str(warp34_bin),
        "serve",
        model,
        "--dp",
        "2",
        "--speculative",
        "mtp",
        "--mtp-k",
        "3",
        "--kv-format",
        "tq4",
        "--offline",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    total_reqs = max(12, int(round(24 * duration_scale)))
    shared_prefix = (
        "Shared coding-agent system context for DP=2 prefix-routing verification. " * 60
    )
    failures: list[str] = []
    ok_cnt = cancel_cnt = fail_cnt = 0

    try:
        with ManagedServer(cmd, "127.0.0.1", port, log_path) as srv:
            # Warmup burst across both DP workers so CUDA graphs & host pools settle before measuring VRAM/RSS delta.
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as wpool:
                wfuts = [
                    wpool.submit(
                        send_chat_request,
                        "127.0.0.1",
                        port,
                        model,
                        [
                            {"role": "system", "content": shared_prefix},
                            {"role": "user", "content": f"Warmup session {i}."},
                        ],
                        max_tokens=16,
                        stream=True,
                        session_id=f"dp-warmup-{i}",
                    )
                    for i in range(4)
                ]
                for fut in concurrent.futures.as_completed(wfuts):
                    st, _, err = fut.result()
                    if st != "ok":
                        failures.append(f"dp2 warmup failed: {err}")
            warmup_metrics = srv.wait_for_drain(timeout_s=20.0)
            gen_before = int((warmup_metrics.get("tokens") or {}).get("generated", 0))
            vram_before = query_vram_used_mib(cuda_visible_devices)
            rss_before = query_process_rss_mib(srv.proc.pid) if srv.proc else None

            load_t0 = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futs = []
                for idx in range(total_reqs):
                    cancel_at = 3 if (idx % 6 == 5) else None
                    msgs = [
                        {"role": "system", "content": shared_prefix},
                        {
                            "role": "user",
                            "content": f"Session {idx % 4}, step {idx}: summarize warp34 DP routing in 2 sentences.",
                        },
                    ]
                    futs.append(
                        pool.submit(
                            send_chat_request,
                            "127.0.0.1",
                            port,
                            model,
                            msgs,
                            max_tokens=64,
                            stream=True,
                            cancel_after_chunks=cancel_at,
                            session_id=f"dp-session-{idx % 4}",
                        )
                    )
                for fut in concurrent.futures.as_completed(futs):
                    st, _, err = fut.result()
                    if st == "ok":
                        ok_cnt += 1
                    elif st == "cancelled":
                        cancel_cnt += 1
                    else:
                        fail_cnt += 1
                        failures.append(f"dp2 request error: {err}")
            load_elapsed = max(0.001, time.monotonic() - load_t0)

            metrics = srv.wait_for_drain(timeout_s=30.0)
            gen_after = int((metrics.get("tokens") or {}).get("generated", 0))
            vram_after = query_vram_used_mib(cuda_visible_devices)
            rss_after = query_process_rss_mib(srv.proc.pid) if srv.proc else None
            st, _, err = send_chat_request(
                "127.0.0.1",
                port,
                model,
                [{"role": "user", "content": "Reply with OK."}],
                max_tokens=16,
                stream=False,
            )
            if st != "ok":
                failures.append(f"dp2 post-drain probe failed: {err}")
            metrics = srv.wait_for_drain(timeout_s=15.0)
            vram_growth = (
                (vram_after - vram_before)
                if vram_before is not None and vram_after is not None
                else None
            )
            rss_growth = (
                (rss_after - rss_before)
                if rss_before is not None and rss_after is not None
                else None
            )
            gen_toks = max(0, gen_after - gen_before)
            tps = gen_toks / load_elapsed

            cons_failures, cache_hit, mean_acc = evaluate_metrics_conservation(
                metrics,
                expect_dp_workers=2,
                min_prefix_cache_hit_pct=50.0,
                vram_growth_mib=vram_growth,
                max_vram_growth_mib=256.0,
                rss_growth_mib=rss_growth,
                max_rss_growth_mib=512.0,
            )
            failures.extend(cons_failures)
            return SubphaseResult(
                name="dp2_4b_concurrency",
                passed=(len(failures) == 0 and fail_cnt == 0),
                skipped=False,
                elapsed_s=time.monotonic() - t0,
                requests_sent=total_reqs,
                requests_ok=ok_cnt,
                requests_cancelled=cancel_cnt,
                requests_failed=fail_cnt,
                output_tokens=gen_toks,
                output_tok_per_s=tps,
                prefix_cache_hit_pct=cache_hit,
                mean_accepted_drafts=mean_acc,
                vram_growth_mib=vram_growth,
                rss_growth_mib=rss_growth,
                failures=tuple(failures),
                server_log=str(log_path),
            )
    except Exception as exc:
        failures.append(str(exc))
        return SubphaseResult(
            name="dp2_4b_concurrency",
            passed=False,
            skipped=False,
            elapsed_s=time.monotonic() - t0,
            requests_sent=total_reqs,
            requests_ok=ok_cnt,
            requests_cancelled=cancel_cnt,
            requests_failed=fail_cnt,
            output_tokens=0,
            output_tok_per_s=0.0,
            prefix_cache_hit_pct=None,
            mean_accepted_drafts=None,
            vram_growth_mib=None,
            rss_growth_mib=None,
            failures=tuple(failures),
            server_log=str(log_path),
        )


def _post_multipart_image_edit(
    host: str,
    port: int,
    model: str,
    png_bytes: bytes,
    prompt: str,
    size: str = "512x512",
    timeout_s: float = 120.0,
) -> tuple[int, dict[str, Any] | str]:
    boundary = f"----warp34soak{uuid.uuid4().hex}"
    parts: list[bytes] = []

    def _add_field(name: str, val: str) -> None:
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{val}\r\n".encode("utf-8")
        )

    _add_field("model", model)
    _add_field("prompt", prompt)
    _add_field("size", size)
    _add_field("response_format", "b64_json")
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="input.png"\r\n'
        f"Content-Type: image/png\r\n\r\n".encode("utf-8")
        + png_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)

    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        conn.request(
            "POST",
            "/v1/images/edits",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", errors="replace")
        if resp.status != 200:
            return resp.status, raw[:300]
        return 200, json.loads(raw)
    finally:
        conn.close()


def run_qwen_image_http_soak(
    *,
    warp34_bin: pathlib.Path,
    out_dir: pathlib.Path,
    cuda_visible_devices: str | None,
) -> SubphaseResult:
    t0 = time.monotonic()
    model = "nchapman/Qwen-Image-2.1-Turbo-W4A16-HGPTQ"
    if not find_cached_hf_repo(model, kind="image"):
        return SubphaseResult(
            name="qwen_image_http",
            passed=True,
            skipped=True,
            elapsed_s=0.0,
            requests_sent=0,
            requests_ok=0,
            requests_cancelled=0,
            requests_failed=0,
            output_tokens=0,
            output_tok_per_s=0.0,
            prefix_cache_hit_pct=None,
            mean_accepted_drafts=None,
            vram_growth_mib=None,
            rss_growth_mib=None,
            failures=(),
        )

    port = pick_free_port()
    log_path = out_dir / "qwen_image_http.server.log"
    env = os.environ.copy()
    if cuda_visible_devices:
        first_gpu = cuda_visible_devices.split(",")[0].strip()
        if first_gpu:
            env["CUDA_VISIBLE_DEVICES"] = first_gpu
    cmd = [
        str(warp34_bin),
        "serve",
        model,
        "--tp",
        "1",
        "--offline",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    failures: list[str] = []
    ok_cnt = fail_cnt = 0

    def _run_image_cycle(cycle_tag: str) -> tuple[int, int]:
        c_ok = c_fail = 0
        for label, extra, expected_color_type in (
            (f"{cycle_tag}_rgb_t2i", {"background": "opaque"}, 2),
            (f"{cycle_tag}_rgba_t2i", {"background": "transparent"}, 6),
        ):
            body = json.dumps(
                {
                    "model": model,
                    "prompt": "A crisp studio photo of a brass compass on dark slate",
                    "size": "512x512",
                    "response_format": "b64_json",
                    **extra,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/images/generations",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                png_bytes = base64.b64decode(data["data"][0]["b64_json"])
                dims = parse_png_dimensions(png_bytes)
                if dims is None or dims[0] != 512 or dims[1] != 512 or dims[2] != expected_color_type:
                    c_fail += 1
                    failures.append(
                        f"{label}: invalid PNG dims/color_type {dims} (expected 512x512 color_type={expected_color_type})"
                    )
                else:
                    c_ok += 1
            except Exception as exc:
                c_fail += 1
                failures.append(f"{label}: {exc}")

        status, edit_resp = _post_multipart_image_edit(
            "127.0.0.1",
            port,
            model,
            make_test_png(64, 64),
            "Turn this into a warm golden sunset gradient",
            size="512x512",
        )
        if status != 200 or not isinstance(edit_resp, dict):
            c_fail += 1
            failures.append(f"{cycle_tag}_image_edit: HTTP {status} {edit_resp}")
        else:
            png_bytes = base64.b64decode(edit_resp["data"][0]["b64_json"])
            dims = parse_png_dimensions(png_bytes)
            if dims is None or dims[0] != 512 or dims[1] != 512 or dims[2] != 2:
                c_fail += 1
                failures.append(
                    f"{cycle_tag}_image_edit: invalid PNG dims/color_type {dims} (expected 512x512 color_type=2)"
                )
            else:
                c_ok += 1
        return c_ok, c_fail

    try:
        with ManagedServer(cmd, "127.0.0.1", port, log_path, env=env) as srv:
            # Cycle 1 warms T2I + RGBA + Edit (including lazy vision-encoder load).
            w_ok, w_fail = _run_image_cycle("warmup")
            ok_cnt += w_ok
            fail_cnt += w_fail
            srv.wait_for_drain(timeout_s=15.0)
            vram_before = query_vram_used_mib(env.get("CUDA_VISIBLE_DEVICES"))
            rss_before = query_process_rss_mib(srv.proc.pid) if srv.proc else None

            # Cycle 2 verifies steady-state VRAM/RSS leak freedom across all 3 endpoints.
            m_ok, m_fail = _run_image_cycle("steady")
            ok_cnt += m_ok
            fail_cnt += m_fail

            metrics = srv.wait_for_drain(timeout_s=15.0)
            vram_after = query_vram_used_mib(env.get("CUDA_VISIBLE_DEVICES"))
            rss_after = query_process_rss_mib(srv.proc.pid) if srv.proc else None
            vram_growth = (
                (vram_after - vram_before)
                if vram_before is not None and vram_after is not None
                else None
            )
            rss_growth = (
                (rss_after - rss_before)
                if rss_before is not None and rss_after is not None
                else None
            )
            cons_failures, _, _ = evaluate_metrics_conservation(
                metrics,
                vram_growth_mib=vram_growth,
                max_vram_growth_mib=256.0,
                rss_growth_mib=rss_growth,
                max_rss_growth_mib=512.0,
            )
            failures.extend(cons_failures)
            return SubphaseResult(
                name="qwen_image_http",
                passed=(len(failures) == 0 and fail_cnt == 0),
                skipped=False,
                elapsed_s=time.monotonic() - t0,
                requests_sent=6,
                requests_ok=ok_cnt,
                requests_cancelled=0,
                requests_failed=fail_cnt,
                output_tokens=0,
                output_tok_per_s=0.0,
                prefix_cache_hit_pct=None,
                mean_accepted_drafts=None,
                vram_growth_mib=vram_growth,
                rss_growth_mib=rss_growth,
                failures=tuple(failures),
                server_log=str(log_path),
            )
    except Exception as exc:
        failures.append(str(exc))
        return SubphaseResult(
            name="qwen_image_http",
            passed=False,
            skipped=False,
            elapsed_s=time.monotonic() - t0,
            requests_sent=6,
            requests_ok=ok_cnt,
            requests_cancelled=0,
            requests_failed=fail_cnt,
            output_tokens=0,
            output_tok_per_s=0.0,
            prefix_cache_hit_pct=None,
            mean_accepted_drafts=None,
            vram_growth_mib=None,
            rss_growth_mib=None,
            failures=tuple(failures),
            server_log=str(log_path),
        )


def run_bonsai_5trit_soak(
    *,
    warp34_bin: pathlib.Path,
    out_dir: pathlib.Path,
    bonsai_model: str | None,
    cuda_visible_devices: str | None,
) -> SubphaseResult:
    t0 = time.monotonic()
    model = bonsai_model
    if not model:
        for cand in BONSAI_CANDIDATE_MODELS:
            if find_cached_hf_repo(cand):
                model = cand
                break
    if not model or not find_cached_hf_repo(model):
        return SubphaseResult(
            name="bonsai_5trit",
            passed=True,
            skipped=True,
            elapsed_s=0.0,
            requests_sent=0,
            requests_ok=0,
            requests_cancelled=0,
            requests_failed=0,
            output_tokens=0,
            output_tok_per_s=0.0,
            prefix_cache_hit_pct=None,
            mean_accepted_drafts=None,
            vram_growth_mib=None,
            rss_growth_mib=None,
            failures=(),
        )

    port = pick_free_port()
    log_path = out_dir / "bonsai_5trit.server.log"
    env = os.environ.copy()
    if cuda_visible_devices:
        first_gpu = cuda_visible_devices.split(",")[0].strip()
        if first_gpu:
            env["CUDA_VISIBLE_DEVICES"] = first_gpu
    cmd = [
        str(warp34_bin),
        "serve",
        model,
        "--tp",
        "1",
        "--kv-format",
        "tq4",
        "--offline",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if repo_supports_mtp(model):
        cmd.extend(["--speculative", "mtp", "--mtp-k", "3"])
    failures: list[str] = []
    ok_cnt = cancel_cnt = fail_cnt = 0
    total_reqs = 12

    try:
        with ManagedServer(cmd, "127.0.0.1", port, log_path, env=env) as srv:
            # Warmup burst so CUDA graphs settle before measuring VRAM/RSS delta.
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as wpool:
                wfuts = [
                    wpool.submit(
                        send_chat_request,
                        "127.0.0.1",
                        port,
                        model,
                        [{"role": "user", "content": f"Warmup {i}."}],
                        max_tokens=16,
                        stream=True,
                    )
                    for i in range(4)
                ]
                for fut in concurrent.futures.as_completed(wfuts):
                    st, _, err = fut.result()
                    if st != "ok":
                        failures.append(f"bonsai warmup failed: {err}")
            warmup_metrics = srv.wait_for_drain(timeout_s=20.0)
            gen_before = int((warmup_metrics.get("tokens") or {}).get("generated", 0))
            vram_before = query_vram_used_mib(env.get("CUDA_VISIBLE_DEVICES"))
            rss_before = query_process_rss_mib(srv.proc.pid) if srv.proc else None
            load_t0 = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futs = []
                for idx in range(total_reqs):
                    cancel_at = 3 if (idx % 4 == 3) else None
                    futs.append(
                        pool.submit(
                            send_chat_request,
                            "127.0.0.1",
                            port,
                            model,
                            [
                                {
                                    "role": "user",
                                    "content": f"Request {idx}: briefly explain ternary 5-trit quantization.",
                                }
                            ],
                            max_tokens=48,
                            stream=True,
                            cancel_after_chunks=cancel_at,
                        )
                    )
                for fut in concurrent.futures.as_completed(futs):
                    st, _, err = fut.result()
                    if st == "ok":
                        ok_cnt += 1
                    elif st == "cancelled":
                        cancel_cnt += 1
                    else:
                        fail_cnt += 1
                        failures.append(f"bonsai request error: {err}")
            load_elapsed = max(0.001, time.monotonic() - load_t0)
            metrics = srv.wait_for_drain(timeout_s=20.0)
            gen_after = int((metrics.get("tokens") or {}).get("generated", 0))
            vram_after = query_vram_used_mib(env.get("CUDA_VISIBLE_DEVICES"))
            rss_after = query_process_rss_mib(srv.proc.pid) if srv.proc else None
            st, _, err = send_chat_request(
                "127.0.0.1",
                port,
                model,
                [{"role": "user", "content": "Reply with OK."}],
                max_tokens=16,
                stream=False,
            )
            if st != "ok":
                failures.append(f"bonsai post-drain probe failed: {err}")
            metrics = srv.wait_for_drain(timeout_s=15.0)
            vram_growth = (
                (vram_after - vram_before)
                if vram_before is not None and vram_after is not None
                else None
            )
            rss_growth = (
                (rss_after - rss_before)
                if rss_before is not None and rss_after is not None
                else None
            )
            gen_toks = max(0, gen_after - gen_before)
            tps = gen_toks / load_elapsed
            cons_failures, cache_hit, mean_acc = evaluate_metrics_conservation(
                metrics,
                vram_growth_mib=vram_growth,
                max_vram_growth_mib=256.0,
                rss_growth_mib=rss_growth,
                max_rss_growth_mib=512.0,
            )
            failures.extend(cons_failures)
            return SubphaseResult(
                name="bonsai_5trit",
                passed=(len(failures) == 0 and fail_cnt == 0),
                skipped=False,
                elapsed_s=time.monotonic() - t0,
                requests_sent=total_reqs,
                requests_ok=ok_cnt,
                requests_cancelled=cancel_cnt,
                requests_failed=fail_cnt,
                output_tokens=gen_toks,
                output_tok_per_s=tps,
                prefix_cache_hit_pct=cache_hit,
                mean_accepted_drafts=mean_acc,
                vram_growth_mib=vram_growth,
                rss_growth_mib=rss_growth,
                failures=tuple(failures),
                server_log=str(log_path),
            )
    except Exception as exc:
        failures.append(str(exc))
        return SubphaseResult(
            name="bonsai_5trit",
            passed=False,
            skipped=False,
            elapsed_s=time.monotonic() - t0,
            requests_sent=total_reqs,
            requests_ok=ok_cnt,
            requests_cancelled=cancel_cnt,
            requests_failed=fail_cnt,
            output_tokens=0,
            output_tok_per_s=0.0,
            prefix_cache_hit_pct=None,
            mean_accepted_drafts=None,
            vram_growth_mib=None,
            rss_growth_mib=None,
            failures=tuple(failures),
            server_log=str(log_path),
        )


def render_soak_markdown(results: list[SubphaseResult]) -> str:
    passed = sum(1 for r in results if r.passed and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if not r.passed)
    overall = "PASS" if failed == 0 else "FAIL"
    lines = [
        f"### `warp34` Release Soak Report (`{overall}`: {passed} passed, {skipped} skipped, {failed} failed)",
        "",
        "| Subphase | Status | Elapsed | Reqs (ok/cancel/fail) | Out tok/s | Cache Hit | Accept | VRAM Δ | RSS Δ |",
        "| :--- | :-: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        status = "SKIP" if r.skipped else ("**PASS**" if r.passed else "**FAIL**")
        tps = f"{r.output_tok_per_s:.1f}" if r.output_tok_per_s > 0 else "—"
        hit = f"{r.prefix_cache_hit_pct:.1f}%" if r.prefix_cache_hit_pct is not None else "—"
        acc = f"{r.mean_accepted_drafts:.2f}" if r.mean_accepted_drafts is not None else "—"
        vram = f"{r.vram_growth_mib:+.0f} MiB" if r.vram_growth_mib is not None else "—"
        rss = f"{r.rss_growth_mib:+.0f} MiB" if r.rss_growth_mib is not None else "—"
        lines.append(
            f"| `{r.name}` | {status} | {r.elapsed_s:.1f}s | {r.requests_ok}/{r.requests_cancelled}/{r.requests_failed} | {tps} | {hit} | {acc} | {vram} | {rss} |"
        )
    all_failures = [(r.name, f) for r in results for f in r.failures]
    if all_failures:
        lines.append("")
        lines.append("#### Soak Failures")
        for name, msg in all_failures:
            lines.append(f"- `{name}`: {msg}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    def _handle_sigterm(_signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signal.SIGTERM)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    parser = argparse.ArgumentParser(description="Managed live-server release soak for warp34")
    parser.add_argument(
        "--warp34-bin",
        type=pathlib.Path,
        default=DEFAULT_WARP34_BIN,
        help="Path to compiled warp34 release binary",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="Directory for server logs, summary.json, and summary.md",
    )
    parser.add_argument(
        "--subphase",
        action="append",
        choices=DEFAULT_SUBPHASES,
        default=[],
        help="Run only specific subphase(s) (repeatable; default: all)",
    )
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="Scale factor for request volume (e.g. 4.0 for extended 1h soak; default: 1.0)",
    )
    parser.add_argument(
        "--bonsai-model",
        default=None,
        help="Explicit Hugging Face repo ID for the bonsai_5trit subphase",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail non-zero if any requested subphase is skipped due to missing cached model",
    )
    args = parser.parse_args(argv)

    if not args.warp34_bin.is_file():
        print(
            f"error: warp34 binary not found at {args.warp34_bin}; run `mise run build` first",
            file=sys.stderr,
        )
        return 2

    out_dir = args.out_dir or (
        REPO_ROOT / "outputs" / f"soak-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = tuple(args.subphase) if args.subphase else DEFAULT_SUBPHASES
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    results: list[SubphaseResult] = []

    for name in selected:
        print(f"[soak] === running subphase: {name} ===", file=sys.stderr)
        if name == "swe_27b_dflash2":
            res = run_swe_27b_dflash2_soak(
                warp34_bin=args.warp34_bin,
                out_dir=out_dir,
                duration_scale=args.duration_scale,
                cuda_visible_devices=cvd,
            )
        elif name == "dp2_4b_concurrency":
            res = run_dp2_4b_soak(
                warp34_bin=args.warp34_bin,
                out_dir=out_dir,
                duration_scale=args.duration_scale,
                cuda_visible_devices=cvd,
            )
        elif name == "qwen_image_http":
            res = run_qwen_image_http_soak(
                warp34_bin=args.warp34_bin,
                out_dir=out_dir,
                cuda_visible_devices=cvd,
            )
        elif name == "bonsai_5trit":
            res = run_bonsai_5trit_soak(
                warp34_bin=args.warp34_bin,
                out_dir=out_dir,
                bonsai_model=args.bonsai_model,
                cuda_visible_devices=cvd,
            )
        else:
            raise ValueError(f"unknown subphase {name!r}")
        if args.strict and res.skipped:
            res = dataclasses.replace(
                res,
                passed=False,
                failures=res.failures + (f"strict mode: subphase {name} skipped (missing cached model)",),
            )
        results.append(res)

    md = render_soak_markdown(results)
    print(md)
    (out_dir / "summary.md").write_text(md + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps([dataclasses.asdict(r) for r in results], indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
