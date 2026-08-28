"""tests/unit/test_sync_benchmark.py
Unit tests for the P3-12 benchmark harness (Issue #64).

CI-safe: no MPI, no JVM, no subprocess execution - the orchestrator's
plan/command/rankfile/wrapper builders and the analyzer's table builders
are exercised as pure functions over synthetic inputs.
"""

import csv

from scripts.analyze_sync_benchmark import (
    build_convergence_rows,
    build_summary_rows,
    collect_worker_rows,
    render_markdown,
)
from scripts.run_sync_benchmark import (
    M3_MODE,
    build_command,
    build_run_plan,
    parse_log_metrics,
    render_rankfile,
    render_throttle_wrapper,
)


class TestRunPlan:
    def test_full_grid(self):
        plan = build_run_plan(["none", "gossip"], [2, 4], ["homogeneous", "throttled"], "out")
        assert len(plan) == 2 * 2 * 2
        assert all(s.np_ == s.workers + 1 for s in plan)

    def test_run_dir_naming(self):
        plan = build_run_plan(["gossip"], [4], ["throttled"], "results/benchmark")
        assert plan[0].run_dir == "results/benchmark/throttled/gossip_w4"
        assert plan[0].log_path.endswith("log.txt")

    def test_fanout_only_applies_to_gossip(self):
        plan = build_run_plan(["gossip", "none"], [2], ["homogeneous"], "out", gossip_fanout=1)
        by_mode = {s.mode: s for s in plan}
        assert by_mode["gossip"].gossip_fanout == 1
        assert by_mode["none"].gossip_fanout is None


class TestCommandBuilder:
    def test_wired_mode_command(self):
        spec = build_run_plan(["ps_async"], [2], ["homogeneous"], "out")[0]
        cmd = build_command(spec, "data.csv", 10, 10, python="python")
        assert cmd[:2] == ["mpirun", "--oversubscribe"]
        assert "-np" in cmd and cmd[cmd.index("-np") + 1] == "3"
        assert "--sync-mode" in cmd and cmd[cmd.index("--sync-mode") + 1] == "ps_async"
        assert "--results-dir" in cmd

    def test_gossip_fanout_flag_appended(self):
        spec = build_run_plan(["gossip"], [2], ["homogeneous"], "out", gossip_fanout=1)[0]
        cmd = build_command(spec, "data.csv", 10, 10, python="python")
        assert "--gossip-fanout" in cmd and cmd[cmd.index("--gossip-fanout") + 1] == "1"

    def test_m3_uses_standalone_module_entry(self):
        spec = build_run_plan([M3_MODE], [2], ["homogeneous"], "out")[0]
        cmd = build_command(spec, "data.csv", 10, 10, python="python")
        assert "mpj_spark.applications.logreg.allreduce" in cmd
        assert "--sync-mode" not in cmd

    def test_rankfile_included_when_selected(self):
        spec = build_run_plan(["none"], [2], ["throttled"], "out")[0]
        cmd = build_command(spec, "data.csv", 10, 10, rankfile_path="rf", python="python")
        assert "--rankfile" in cmd and cmd[cmd.index("--rankfile") + 1] == "rf"
        assert "bash" not in cmd

    def test_taskset_wrapper_command_shape(self):
        spec = build_run_plan(["none"], [4], ["throttled"], "out")[0]
        wrapper = spec.run_dir + "/throttle_wrapper.sh"
        cmd = build_command(
            spec,
            "data.csv",
            10,
            10,
            wrapper_path=wrapper,
            throttle_rank=1,
            throttle_cores="0-1",
            python="python",
        )
        assert "-x" in cmd
        assert "THROTTLE_RANK=1" in cmd and "THROTTLE_CORES=0-1" in cmd
        # wrapper sits between -np and the python interpreter
        assert cmd[cmd.index("-np") + 2 : cmd.index("-np") + 4] == ["bash", wrapper]
        assert "--rankfile" not in cmd


class TestRankfile:
    def test_balanced_split(self):
        rf = render_rankfile(3, 22)
        lines = rf.strip().splitlines()
        assert lines[0] == "rank 0=localhost slot=0-6"
        assert lines[1] == "rank 1=localhost slot=7-13"
        assert lines[2] == "rank 2=localhost slot=14-20"

    def test_throttled_rank_gets_reduced_slots(self):
        rf = render_rankfile(3, 22, throttle_rank=1, throttle_slots=2)
        lines = rf.strip().splitlines()
        assert lines[1] == "rank 1=localhost slot=7-8"
        assert len(lines) == 3  # all ranks still covered


class TestThrottleWrapper:
    def test_wrapper_pins_only_the_throttled_rank(self):
        content = render_throttle_wrapper()
        assert "OMPI_COMM_WORLD_RANK" in content
        assert "THROTTLE_RANK" in content
        assert "taskset -c" in content
        # non-throttled ranks exec the command untouched
        assert content.rstrip().endswith('exec "$@"')


class TestLogParsing:
    SNIPPET = """
     Weighted accuracy: 0.6308
     Final |w|        : 0.3228
  Processing Time (avg fit)     28.2992 s
  Total Wall-clock Time         28.9359 s
"""

    def test_parses_all_metrics(self):
        m = parse_log_metrics(self.SNIPPET)
        assert m["final_weight_norm"] == 0.3228
        assert m["weighted_accuracy"] == 0.6308
        assert m["wall_clock_s"] == 28.9359
        assert m["proc_time_s"] == 28.2992

    def test_missing_fields_are_none(self):
        m = parse_log_metrics("nothing here")
        assert m["final_weight_norm"] is None


class TestAnalyzer:
    def _write_worker_csv(self, run_dir, name, rows):
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def _manifest_row(self, run_dir, mode="gossip", arm="homogeneous"):
        return {
            "arm": arm,
            "mode": mode,
            "workers": "2",
            "exit_code": "0",
            "wall_clock_s": "28.9",
            "weighted_accuracy": "0.6308",
            "final_weight_norm": "0.3228",
            "run_dir": str(run_dir),
        }

    def test_collect_worker_rows_covers_both_patterns(self, tmp_path):
        self._write_worker_csv(
            tmp_path,
            "worker_0_gossip_metrics.csv",
            [
                {
                    "worker_id": 0,
                    "iteration": 1,
                    "iter_time_s": 1.5,
                    "weight_norm": 0.32,
                    "gossip_time_s": 0.01,
                }
            ],
        )
        self._write_worker_csv(
            tmp_path,
            "worker_1_logreg_iter_metrics.csv",
            [{"worker_id": 1, "iteration": 1, "iter_time_s": 1.6, "weight_norm": 0.33}],
        )
        rows = collect_worker_rows(str(tmp_path))
        assert len(rows) == 2  # both filename patterns, no double counting

    def test_summary_rows_include_sync_time_for_instrumented_modes(self, tmp_path):
        run_dir = tmp_path / "run"
        self._write_worker_csv(
            run_dir,
            "worker_0_hybrid_metrics.csv",
            [
                {
                    "worker_id": 0,
                    "iteration": i,
                    "iter_time_s": 2.0,
                    "weight_norm": 0.32,
                    "allreduce_time_s": 0.10,
                    "ps_time_s": 0.01,
                }
                for i in (1, 2)
            ],
        )
        rows = build_summary_rows(
            [self._manifest_row(run_dir, mode="hybrid_ps_allreduce")], str(tmp_path)
        )
        assert len(rows) == 1
        # (0.10 + 0.01) x 2 rounds / 1 worker
        assert rows[0]["sync_channel_time_s"] == 0.22
        assert rows[0]["mean_iter_time_s"] == 2.0

    def test_failed_runs_are_skipped(self, tmp_path):
        bad = self._manifest_row(tmp_path)
        bad["exit_code"] = "1"
        assert build_summary_rows([bad], str(tmp_path)) == []

    def test_convergence_rows_group_by_iteration(self, tmp_path):
        run_dir = tmp_path / "run"
        self._write_worker_csv(
            run_dir,
            "worker_0_gossip_metrics.csv",
            [{"worker_id": 0, "iteration": 1, "weight_norm": 0.30}],
        )
        self._write_worker_csv(
            run_dir,
            "worker_1_gossip_metrics.csv",
            [{"worker_id": 1, "iteration": 1, "weight_norm": 0.34}],
        )
        rows = build_convergence_rows([self._manifest_row(run_dir)])
        assert len(rows) == 1
        assert rows[0]["mean_weight_norm"] == 0.32
        assert rows[0]["max_weight_norm"] == 0.34

    def test_markdown_renders_mode_table(self, tmp_path):
        run_dir = tmp_path / "run"
        self._write_worker_csv(
            run_dir,
            "worker_0_gossip_metrics.csv",
            [
                {
                    "worker_id": 0,
                    "iteration": 1,
                    "iter_time_s": 1.5,
                    "weight_norm": 0.32,
                    "gossip_time_s": 0.01,
                }
            ],
        )
        summary = build_summary_rows([self._manifest_row(run_dir)], str(tmp_path))
        md = render_markdown(summary)
        assert "gossip" in md
        assert "| Mode | Workers |" in md
