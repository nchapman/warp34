import importlib.util
import json
import pathlib
import tempfile
import unittest

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "run_swebench.py"
SPEC = importlib.util.spec_from_file_location("run_swebench", MODULE_PATH)
assert SPEC and SPEC.loader
run_swebench = importlib.util.module_from_spec(SPEC)
import sys

sys.modules["run_swebench"] = run_swebench
SPEC.loader.exec_module(run_swebench)


class RunSwebenchTests(unittest.TestCase):
    def test_build_warp34_serve_cmd_defaults_and_overrides(self) -> None:
        cmd = run_swebench.build_warp34_serve_cmd(
            binary="./target/release/warp34",
            model="nchapman/Qwen3.8-27B-W4A16-HGPTQ",
            served_model_name="qwen3.8-27b",
            host="127.0.0.1",
            port=18027,
            tp=2,
            speculative="mtp",
            mtp_k=3,
            ctx_size="384k",
            reasoning_effort="max",
            reasoning_budget=None,
        )
        self.assertIn("--offline", cmd)
        self.assertIn("--mtp-k", cmd)
        self.assertEqual(cmd[cmd.index("--ctx-size") + 1], "384k")
        self.assertEqual(cmd[cmd.index("--reasoning-effort") + 1], "max")
        self.assertNotIn("--reasoning-budget", cmd)

    def test_build_minisweagent_cmd_omits_max_tokens_by_default(self) -> None:
        out_dir = pathlib.Path("/tmp/swebench-test")
        cmd = run_swebench.build_minisweagent_cmd(
            output_dir=out_dir,
            served_model_name="qwen3.8-27b",
            api_base="http://127.0.0.1:18027/v1",
            slice_spec="0:10",
            workers=4,
            max_tokens=None,
        )
        self.assertIn("openai/qwen3.8-27b", cmd)
        self.assertFalse(any("max_tokens=" in arg for arg in cmd))

        cmd_with_max = run_swebench.build_minisweagent_cmd(
            output_dir=out_dir,
            served_model_name="openai/qwen3.8-27b",
            api_base="http://127.0.0.1:18027/v1",
            max_tokens=16384,
        )
        self.assertIn("model.model_kwargs.max_tokens=16384", cmd_with_max)

    def test_normalize_patch_p0_and_extract_from_traj(self) -> None:
        p0 = (
            "--- astropy/wcs/wcsapi/wrappers/sliced_wcs.py.bak\t2026-11-22 20:43:01\n"
            "+++ astropy/wcs/wcsapi/wrappers/sliced_wcs.py\t2026-11-22 20:49:31\n"
            "@@ -1,3 +1,3 @@\n-foo\n+bar\n"
        )
        fixed, changed = run_swebench.normalize_patch_Or_extract_from_traj(p0)
        self.assertTrue(changed)
        self.assertIn("--- a/astropy/wcs/wcsapi/wrappers/sliced_wcs.py\n", fixed)
        self.assertIn("+++ b/astropy/wcs/wcsapi/wrappers/sliced_wcs.py\n", fixed)

        raw_file = "# Licensed under a 3-clause BSD style license\nclass Foo:\n    pass\n"
        traj = {
            "messages": [
                {
                    "role": "tool",
                    "content": (
                        "<returncode>0</returncode>\n"
                        "<output>\n"
                        "diff --git a/astropy/io/ascii/rst.py b/astropy/io/ascii/rst.py\n"
                        "--- a/astropy/io/ascii/rst.py\n"
                        "+++ b/astropy/io/ascii/rst.py\n"
                        "@@ -1 +1 @@\n-old\n+new\n"
                        "</output>"
                    ),
                }
            ]
        }
        extracted, changed2 = run_swebench.normalize_patch_Or_extract_from_traj(raw_file, traj)
        self.assertTrue(changed2)
        self.assertTrue(extracted.startswith("diff --git a/astropy/io/ascii/rst.py"))

        git_with_p0 = (
            "diff --git a/foo.py b/foo.py\n"
            "--- foo.py.bak\t2026-11-22\n"
            "+++ foo.py\t2026-11-22\n"
            "@@ -1 +1 @@\n-a\n+b\n"
        )
        fixed_git_p0, changed3 = run_swebench.normalize_patch_Or_extract_from_traj(git_with_p0)
        self.assertTrue(changed3)
        self.assertIn("--- a/foo.py\n+++ b/foo.py\n", fixed_git_p0)

    def test_percentile_sorted_and_malformed_log_resilience(self) -> None:
        self.assertEqual(run_swebench.percentile_sorted([10, 20], 0.50), 10)
        self.assertEqual(run_swebench.percentile_sorted([10, 20, 30], 0.50), 20)
        with tempfile.TemporaryDirectory() as tmp:
            log_path = pathlib.Path(tmp) / "server.log"
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps({"event": "inference.turn_end", "finish_reason": "tool_calls"}),
                        json.dumps(
                            {
                                "event": "inference.turn_end",
                                "timestamp_unix_ms": 5000,
                                "duration_ms": 500.0,
                                "prompt_tokens": 1000,
                                "cached_prompt_tokens": 900,
                                "output_tokens": 100,
                                "reasoning_tokens": 40,
                                "finish_reason": "tool_calls",
                            }
                        ),
                    ]
                )
            )
            summary = run_swebench.parse_server_logs([log_path])
            self.assertEqual(summary.total_turns, 1)
            self.assertEqual(summary.error_events, 1)

    def test_analyze_output_dir_and_format_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            inst_dir = root / "astropy__astropy-12907"
            inst_dir.mkdir()
            (inst_dir / "astropy__astropy-12907.traj.json").write_text(
                json.dumps(
                    {
                        "info": {
                            "exit_status": "Submitted",
                            "submission": "diff --git a/x.py b/x.py\n",
                        },
                        "messages": [
                            {"role": "user", "content": "fix bug"},
                            {"role": "assistant", "content": "step 1"},
                            {"role": "assistant", "content": "step 2"},
                        ],
                    }
                )
            )
            (root / "eval_report.json").write_text(
                json.dumps({"resolved_ids": ["astropy__astropy-12907"]})
            )
            (root / "warp34-server.log").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event": "inference.turn_end",
                                "timestamp_unix_ms": 10000,
                                "duration_ms": 1000.0,
                                "ttft_ms": 150.0,
                                "prompt_tokens": 2000,
                                "cached_prompt_tokens": 1800,
                                "output_tokens": 200,
                                "reasoning_tokens": 120,
                                "output_tokens_per_s": 80.0,
                                "finish_reason": "tool_calls",
                            }
                        ),
                        json.dumps(
                            {
                                "event": "inference.turn_end",
                                "timestamp_unix_ms": 13000,
                                "duration_ms": 2000.0,
                                "ttft_ms": 250.0,
                                "prompt_tokens": 4000,
                                "cached_prompt_tokens": 3800,
                                "output_tokens": 300,
                                "reasoning_tokens": 180,
                                "output_tokens_per_s": 100.0,
                                "finish_reason": "tool_calls",
                            }
                        ),
                    ]
                )
            )
            report = run_swebench.analyze_output_dir(root)
            self.assertEqual(report.submitted_count, 1)
            self.assertEqual(report.resolved_count, 1)
            self.assertEqual(report.total_trajectory_turns, 2)
            assert report.server is not None
            self.assertEqual(report.server.total_turns, 2)
            self.assertEqual(report.server.output_tokens, 500)
            self.assertEqual(report.server.reasoning_tokens, 300)
            md = run_swebench.format_markdown_report(report)
            self.assertIn("**RESOLVED**", md)
            self.assertIn("`1 / 1 (100.0%)`", md)

    def test_synthesize_eval_report_from_logs_recovers_partial_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            preds_path = root / "preds.json"
            preds_path.write_text(
                json.dumps(
                    {
                        "repo__ok-1": {"model_patch": "diff --git a/a.py b/a.py\n"},
                        "repo__fail-2": {"model_patch": "diff --git a/b.py b/b.py\n"},
                        "repo__empty-3": {"model_patch": "   "},
                        "repo__crashed-4": {"model_patch": "diff --git a/d.py b/d.py\n"},
                    }
                )
            )
            log_dir = root / "logs" / "run_evaluation" / "run1"
            for iid, resolved in [("repo__ok-1", True), ("repo__fail-2", False)]:
                rdir = log_dir / "model" / iid
                rdir.mkdir(parents=True)
                (rdir / "report.json").write_text(json.dumps({iid: {"resolved": resolved}}))

            synth = run_swebench._synthesize_eval_report_from_logs(preds_path, log_dir)
            assert synth is not None
            self.assertEqual(synth["submitted_instances"], 4)
            self.assertEqual(synth["completed_instances"], 2)
            self.assertEqual(synth["resolved_ids"], ["repo__ok-1"])
            self.assertEqual(synth["unresolved_ids"], ["repo__fail-2"])
            self.assertEqual(synth["empty_patch_ids"], ["repo__empty-3"])
            self.assertEqual(synth["error_ids"], ["repo__crashed-4"])

    def test_run_evaluation_harness_recovers_on_timeout_or_reraises(self) -> None:
        from unittest import mock
        import os
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            out_dir = root / "out"
            out_dir.mkdir()
            (out_dir / "preds.json").write_text(
                json.dumps({"repo__ok-1": {"model_patch": "diff --git a/a.py b/a.py\n"}})
            )
            log_dir = root / "logs" / "run_evaluation" / "warp34-out" / "model" / "repo__ok-1"
            log_dir.mkdir(parents=True)
            (log_dir / "report.json").write_text(json.dumps({"repo__ok-1": {"resolved": True}}))

            old_cwd = pathlib.Path.cwd()
            try:
                os.chdir(root)
                with mock.patch.object(
                    subprocess,
                    "run",
                    side_effect=subprocess.TimeoutExpired(cmd=["uvx"], timeout=1),
                ):
                    run_swebench.run_evaluation_harness(
                        out_dir, max_workers=1, normalize_p1=False, timeout_s=1
                    )
                recovered = json.loads((out_dir / "eval_report.json").read_text())
                self.assertEqual(recovered["resolved_ids"], ["repo__ok-1"])

                # Without logs or report files, TimeoutExpired is re-raised.
                (out_dir / "eval_report.json").unlink()
                with mock.patch.object(
                    subprocess,
                    "run",
                    side_effect=subprocess.TimeoutExpired(cmd=["uvx"], timeout=1),
                ):
                    with self.assertRaises(subprocess.TimeoutExpired):
                        run_swebench.run_evaluation_harness(
                            out_dir, max_workers=1, normalize_p1=False, timeout_s=1
                        )
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
