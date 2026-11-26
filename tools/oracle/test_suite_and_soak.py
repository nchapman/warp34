from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

TOOLS_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import check_bench_regressions as bench_reg
import run_release_soak as soak
import run_suite as suite


class ReleaseSoakTests(unittest.TestCase):
    def test_make_and_parse_png_roundtrip(self) -> None:
        png = soak.make_test_png(64, 48)
        self.assertEqual(soak.parse_png_dimensions(png), (64, 48, 2))
        self.assertIsNone(soak.parse_png_dimensions(b"not-a-png"))

    def test_conservation_passes_on_clean_dp2_metrics(self) -> None:
        metrics = {
            "requests": {"running": 0, "waiting": 0, "failed": 0, "completed": 24},
            "http_responses": {"success": 24, "client_error": 0, "server_error": 0},
            "pools": {
                "kv": {"capacity": 1024, "active": 0, "cached": 120, "free": 904},
                "gdn": {"capacity": 32, "active": 0, "cached": 8, "free": 24},
            },
            "tokens": {"prompt": 10000, "cached": 7500, "generated": 1200},
            "speculation": {"rounds": 400, "accepted_tokens": 900},
            "workers": [
                {
                    "requests": {"running": 0, "waiting": 0, "failed": 0, "completed": 12},
                    "http_responses": {"server_error": 0},
                    "pools": {"kv": {"active": 0}, "gdn": {"active": 0}},
                },
                {
                    "requests": {"running": 0, "waiting": 0, "failed": 0, "completed": 12},
                    "http_responses": {"server_error": 0},
                    "pools": {"kv": {"active": 0}, "gdn": {"active": 0}},
                },
            ],
        }
        failures, hit, acc = soak.evaluate_metrics_conservation(
            metrics,
            expect_dp_workers=2,
            min_prefix_cache_hit_pct=50.0,
            min_mean_accepted_drafts=1.5,
            vram_growth_mib=12.0,
            max_vram_growth_mib=256.0,
            rss_growth_mib=40.0,
            max_rss_growth_mib=512.0,
        )
        self.assertEqual(failures, [])
        self.assertAlmostEqual(hit or 0.0, 75.0)
        self.assertAlmostEqual(acc or 0.0, 2.25)

    def test_conservation_catches_leaks_and_unbalanced_dp(self) -> None:
        metrics = {
            "requests": {"running": 1, "waiting": 0, "failed": 2},
            "http_responses": {"server_error": 1},
            "pools": {"kv": {"active": 3}, "gdn": {"active": 1}},
            "tokens": {"prompt": 1000, "cached": 100, "generated": 50},
            "speculation": {"rounds": 10, "accepted_tokens": 5},
            "workers": [
                {
                    "requests": {"running": 0, "waiting": 0, "failed": 0, "completed": 10},
                    "http_responses": {"server_error": 0},
                    "pools": {"kv": {"active": 0}, "gdn": {"active": 0}},
                },
                {
                    "requests": {"running": 0, "waiting": 0, "failed": 0, "completed": 0},
                    "http_responses": {"server_error": 0},
                    "pools": {"kv": {"active": 2}, "gdn": {"active": 0}},
                },
            ],
        }
        failures, _, _ = soak.evaluate_metrics_conservation(
            metrics,
            expect_dp_workers=2,
            min_prefix_cache_hit_pct=50.0,
            min_mean_accepted_drafts=1.2,
            vram_growth_mib=300.0,
            max_vram_growth_mib=256.0,
        )
        joined = "\n".join(failures)
        self.assertIn("requests.running=1", joined)
        self.assertIn("leaked KV pages", joined)
        self.assertIn("leaked GDN slots", joined)
        self.assertIn("unbalanced dispatch", joined)
        self.assertIn("prefix_cache_hit_pct", joined)
        self.assertIn("mean_accepted_drafts", joined)
        self.assertIn("VRAM growth", joined)


class RunSuiteTests(unittest.TestCase):
    def test_tier_stage_graphs_are_deduplicated_and_ordered(self) -> None:
        tmp = pathlib.Path("/tmp/warp34-test-suite")
        smoke_names = [s.name for s in suite.build_tier_stages("smoke", tmp)]
        gate_names = [s.name for s in suite.build_tier_stages("gate", tmp)]
        qualify_names = [s.name for s in suite.build_tier_stages("qualify", tmp)]

        # Smoke stays lean: no clippy-cpu, no doc-check, no dashboard-check.
        self.assertIn("fmt-check", smoke_names)
        self.assertIn("swebench-check", smoke_names)
        self.assertIn("doc-test", smoke_names)
        self.assertIn("test-quick", smoke_names)
        self.assertNotIn("clippy-cpu", smoke_names)

        # Gate runs --lib once (test-lib) + --tests once (test-integration) + sanitize + bench-gate.
        self.assertIn("clippy-cpu", gate_names)
        self.assertIn("test-lib", gate_names)
        self.assertIn("test-integration", gate_names)
        self.assertNotIn("test-quick", gate_names)
        self.assertIn("cuda-sanitize", gate_names)
        self.assertIn("bench-gate", gate_names)
        self.assertIn("oracle-greedy", gate_names)

        # Qualify deduplicates bench-gate (runs bench-gate-full once) and pins needle-gates to single GPU.
        self.assertNotIn("bench-gate", qualify_names)
        self.assertIn("bench-gate-full", qualify_names)
        self.assertIn("oracle-dump", qualify_names)
        self.assertIn("needle-gates", qualify_names)
        self.assertIn("soak", qualify_names)
        self.assertIn("swebench-slice", qualify_names)
        needle_stage = next(s for s in suite.build_tier_stages("qualify", tmp) if s.name == "needle-gates")
        self.assertEqual(needle_stage.gpu_mode, "single")

    def test_stage_env_pins_single_gpu_and_preserves_multi_gpu(self) -> None:
        single = suite.Stage("needle-gates", ("echo",), "single")
        multi = suite.Stage("oracle-dump", ("echo",), "multi")
        self.assertEqual(suite.stage_env_for(single, ["0", "1"])["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(suite.stage_env_for(multi, ["0", "1"])["CUDA_VISIBLE_DEVICES"], "0,1")

    def test_resume_loads_previously_passed_stages(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = pathlib.Path(td)
            results = [
                suite.StageResult("fmt-check", "passed", 0, 1.2, "none"),
                suite.StageResult("clippy-cuda", "failed", 1, 5.0, "none"),
            ]
            suite.write_receipt("smoke", run_dir, results, {"git_sha": "abc1234", "git_dirty": False})
            self.assertEqual(suite.load_resumed_passed_stages(run_dir), {"fmt-check"})

            # Corrupt / truncated receipt degrades cleanly instead of crashing.
            (run_dir / "summary.json").write_text("{corrupt", encoding="utf-8")
            self.assertEqual(suite.load_resumed_passed_stages(run_dir), set())

    def test_all_skipped_suite_is_not_passed(self) -> None:
        self.assertFalse(
            suite.suite_passed([suite.StageResult("fmt-check", "skipped", 0, 0.0, "none")])
        )
        self.assertTrue(
            suite.suite_passed(
                [
                    suite.StageResult("fmt-check", "passed", 0, 1.0, "none"),
                    suite.StageResult("doc-test", "skipped", 0, 0.0, "none"),
                ]
            )
        )

    def test_run_stage_handles_missing_binary_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            res = suite.run_stage(
                suite.Stage("bad-bin", ("/nonexistent/binary/warp34-nope",), "none"),
                run_dir=pathlib.Path(td),
                visible_gpus=[],
            )
            self.assertEqual(res.status, "failed")
            self.assertEqual(res.exit_code, 127)
            self.assertIn("launch error", res.note or "")

    def test_bench_detect_host_arch_respects_cuda_visible_devices(self) -> None:
        fake = mock.Mock(stdout="8.9\n", returncode=0)
        with mock.patch("subprocess.run", return_value=fake) as run_mock:
            arch, count = bench_reg.detect_host_arch(cuda_visible_devices="1")
        self.assertEqual((arch, count), ("sm_89", 1))
        cmd = run_mock.call_args[0][0]
        self.assertIn("-i", cmd)
        self.assertIn("1", cmd)


if __name__ == "__main__":
    unittest.main()
