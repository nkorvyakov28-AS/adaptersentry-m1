"""Tests for CARD-11 benchmark harness — corpus generator, metrics, regression check.

These tests do NOT run the full multiprocessing batch (too slow for unit tests).
They test the corpus generator, metrics dataclass, and regression checker in isolation.
The harness integration test uses a minimal 2-adapter run via a direct worker call.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmarks.corpus import generate_corpus, generate_anomalous_corpus
from benchmarks.check_regression import check_regression
from benchmarks.harness import BenchmarkMetrics, _collect_latencies, _percentile


# ---------------------------------------------------------------------------
# Corpus generator
# ---------------------------------------------------------------------------

class TestGenerateCorpus:
    def test_creates_expected_file_count(self, tmp_path) -> None:
        paths = generate_corpus(5, tmp_path / "corpus")
        assert len(paths) == 5

    def test_files_have_safetensors_extension(self, tmp_path) -> None:
        paths = generate_corpus(3, tmp_path / "corpus")
        for p in paths:
            assert p.suffix == ".safetensors"

    def test_files_are_parseable(self, tmp_path) -> None:
        from safetensors.numpy import load_file
        paths = generate_corpus(2, tmp_path / "corpus", rank=4, n_layers=2)
        for p in paths:
            tensors = load_file(str(p))
            assert len(tensors) > 0

    def test_tensor_names_contain_lora_a_and_b(self, tmp_path) -> None:
        from safetensors.numpy import load_file
        paths = generate_corpus(1, tmp_path / "corpus", rank=4, n_layers=2, n_modules=2)
        tensors = load_file(str(paths[0]))
        keys = list(tensors.keys())
        assert any("lora_A" in k for k in keys)
        assert any("lora_B" in k for k in keys)

    def test_tensor_shapes_match_rank(self, tmp_path) -> None:
        from safetensors.numpy import load_file
        rank, hidden = 8, 32
        paths = generate_corpus(1, tmp_path / "corpus", rank=rank, hidden_dim=hidden, n_layers=1, n_modules=1)
        tensors = load_file(str(paths[0]))
        a_key = next(k for k in tensors if "lora_A" in k)
        b_key = next(k for k in tensors if "lora_B" in k)
        assert list(tensors[a_key].shape) == [rank, hidden]
        assert list(tensors[b_key].shape) == [hidden, rank]

    def test_metadata_contains_rank(self, tmp_path) -> None:
        from safetensors import safe_open
        rank = 16
        paths = generate_corpus(1, tmp_path / "corpus", rank=rank, n_layers=1)
        with safe_open(str(paths[0]), framework="numpy") as f:
            meta = f.metadata()
        assert meta.get("r") == str(rank)

    def test_deterministic_with_same_seed(self, tmp_path) -> None:
        from safetensors.numpy import load_file
        p1 = tmp_path / "a"
        p2 = tmp_path / "b"
        paths1 = generate_corpus(2, p1, seed=7)
        paths2 = generate_corpus(2, p2, seed=7)
        for a, b in zip(paths1, paths2):
            t1 = load_file(str(a))
            t2 = load_file(str(b))
            for key in t1:
                assert (t1[key] == t2[key]).all()

    def test_different_seeds_produce_different_tensors(self, tmp_path) -> None:
        from safetensors.numpy import load_file
        p1 = tmp_path / "a"
        p2 = tmp_path / "b"
        paths1 = generate_corpus(1, p1, seed=1)
        paths2 = generate_corpus(1, p2, seed=2)
        t1 = load_file(str(paths1[0]))
        t2 = load_file(str(paths2[0]))
        key = next(k for k in t1 if "lora_A" in k)
        assert not (t1[key] == t2[key]).all()

    def test_anomalous_corpus_creates_files(self, tmp_path) -> None:
        paths = generate_anomalous_corpus(3, tmp_path / "anomalous")
        assert len(paths) == 3
        for p in paths:
            assert p.exists()

    def test_output_dir_created_if_absent(self, tmp_path) -> None:
        out = tmp_path / "deep" / "nested" / "dir"
        assert not out.exists()
        generate_corpus(1, out)
        assert out.exists()

    def test_m1_analyzer_can_scan_generated_adapter(self, tmp_path) -> None:
        from adaptersentry.analyzer import scan
        from adaptersentry.schemas.adapter_report import ParseStatus

        paths = generate_corpus(1, tmp_path / "corpus", rank=8, n_layers=2, n_modules=2)
        report = scan(paths[0])
        assert report.parse_status != ParseStatus.FAILED
        assert len(report.tensor_records) > 0


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

class TestPercentile:
    def test_empty_returns_zero(self) -> None:
        assert _percentile([], 50) == 0.0

    def test_single_value(self) -> None:
        assert _percentile([100.0], 50) == 100.0
        assert _percentile([100.0], 95) == 100.0

    def test_p50_median(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(vals, 50) == 3.0

    def test_p95_above_most(self) -> None:
        vals = list(range(1, 101))  # 1..100
        assert _percentile(vals, 95) >= 95.0


# ---------------------------------------------------------------------------
# BenchmarkMetrics DoD checks
# ---------------------------------------------------------------------------

class TestBenchmarkMetricsDoD:
    def _make(self, **kwargs) -> BenchmarkMetrics:
        defaults = dict(
            scenario="cold",
            n_adapters=100,
            n_workers=4,
            total_wall_s=50.0,
            throughput_per_min=120.0,
            latency_p50_ms=400.0,
            latency_p95_ms=800.0,
            latency_p99_ms=1000.0,
            peak_total_rss_mb=1000.0,
            peak_per_worker_mb=250.0,
            error_rate=0.0,
            cache_hit_rate=0.0,
        )
        defaults.update(kwargs)
        return BenchmarkMetrics(**defaults)

    def test_cold_pass_when_throughput_ok(self) -> None:
        m = self._make(scenario="cold", n_adapters=100, throughput_per_min=120.0)
        passed, failures = m.passes_dod("cold")
        assert passed
        assert failures == []

    def test_cold_fail_when_throughput_below_100(self) -> None:
        m = self._make(scenario="cold", n_adapters=100, throughput_per_min=80.0)
        passed, failures = m.passes_dod("cold")
        assert not passed
        assert any("throughput" in f for f in failures)

    def test_warm_pass_when_wall_time_under_120s(self) -> None:
        m = self._make(scenario="warm", n_adapters=1000, total_wall_s=90.0)
        passed, failures = m.passes_dod("warm")
        assert passed

    def test_warm_fail_when_wall_time_over_120s(self) -> None:
        m = self._make(scenario="warm", n_adapters=1000, total_wall_s=150.0)
        passed, failures = m.passes_dod("warm")
        assert not passed
        assert any("120s" in f for f in failures)

    def test_memory_fail_when_over_512mb_per_worker(self) -> None:
        m = self._make(peak_per_worker_mb=600.0)
        passed, failures = m.passes_dod("cold")
        assert not passed
        assert any("512MB" in f for f in failures)

    def test_serialisable_to_json(self) -> None:
        m = self._make()
        data = json.dumps(asdict(m))
        restored = json.loads(data)
        assert restored["scenario"] == "cold"
        assert "latency_p95_ms" in restored


# ---------------------------------------------------------------------------
# Latency collection from result files
# ---------------------------------------------------------------------------

class TestCollectLatencies:
    def test_reads_wall_time_ms_from_json_files(self, tmp_path) -> None:
        result_data = {
            "schema_version": "1.0.0",
            "identity": {"wall_time_ms": 1500},
            "status": "ok",
        }
        (tmp_path / "adapter_0000.json").write_text(json.dumps(result_data))
        (tmp_path / "adapter_0001.json").write_text(json.dumps(
            {**result_data, "identity": {"wall_time_ms": 2000}}
        ))
        latencies = _collect_latencies(tmp_path)
        assert sorted(latencies) == [1500.0, 2000.0]

    def test_skips_debug_json_files(self, tmp_path) -> None:
        (tmp_path / "adapter_0000.debug.json").write_text(
            json.dumps({"identity": {"wall_time_ms": 999}})
        )
        latencies = _collect_latencies(tmp_path)
        assert latencies == []

    def test_skips_run_summary(self, tmp_path) -> None:
        (tmp_path / "run_summary.json").write_text(
            json.dumps({"identity": {"wall_time_ms": 999}})
        )
        latencies = _collect_latencies(tmp_path)
        assert latencies == []

    def test_tolerates_missing_wall_time(self, tmp_path) -> None:
        (tmp_path / "adapter_0000.json").write_text(json.dumps({"status": "ok"}))
        latencies = _collect_latencies(tmp_path)
        assert latencies == []

    def test_empty_dir_returns_empty(self, tmp_path) -> None:
        latencies = _collect_latencies(tmp_path)
        assert latencies == []


# ---------------------------------------------------------------------------
# Regression checker
# ---------------------------------------------------------------------------

class TestCheckRegression:
    def _metrics(self, scenario="cold", **kwargs) -> dict:
        base = dict(
            scenario=scenario,
            throughput_per_min=100.0,
            latency_p50_ms=500.0,
            latency_p95_ms=1000.0,
            latency_p99_ms=1200.0,
            error_rate=0.0,
        )
        base.update(kwargs)
        return base

    def test_no_regression_passes(self) -> None:
        baseline = [self._metrics(latency_p95_ms=1000.0, throughput_per_min=100.0)]
        current  = [self._metrics(latency_p95_ms=1100.0, throughput_per_min=95.0)]
        passed, failures = check_regression(current, baseline, factor=2.0)
        assert passed
        assert failures == []

    def test_p95_regression_detected(self) -> None:
        baseline = [self._metrics(latency_p95_ms=1000.0)]
        current  = [self._metrics(latency_p95_ms=2500.0)]  # > 2× baseline
        passed, failures = check_regression(current, baseline, factor=2.0)
        assert not passed
        assert any("p95" in f for f in failures)

    def test_throughput_regression_detected(self) -> None:
        baseline = [self._metrics(throughput_per_min=100.0)]
        current  = [self._metrics(throughput_per_min=40.0)]  # < 0.5× baseline
        passed, failures = check_regression(current, baseline, factor=2.0)
        assert not passed
        assert any("throughput" in f for f in failures)

    def test_error_rate_regression_detected(self) -> None:
        baseline = [self._metrics(error_rate=0.0)]
        current  = [self._metrics(error_rate=0.10)]  # +10pp > +5pp tolerance
        passed, failures = check_regression(current, baseline, factor=2.0)
        assert not passed
        assert any("error rate" in f for f in failures)

    def test_unknown_scenario_skipped(self) -> None:
        baseline = [self._metrics(scenario="cold")]
        current  = [self._metrics(scenario="warm", latency_p95_ms=99999.0)]
        passed, failures = check_regression(current, baseline, factor=2.0)
        assert passed  # no baseline for "warm" — skipped

    def test_multiple_scenarios_both_checked(self) -> None:
        baseline = [
            self._metrics(scenario="cold", latency_p95_ms=1000.0),
            self._metrics(scenario="warm", latency_p95_ms=200.0),
        ]
        current = [
            self._metrics(scenario="cold", latency_p95_ms=1100.0),
            self._metrics(scenario="warm", latency_p95_ms=5000.0),  # regressed
        ]
        passed, failures = check_regression(current, baseline, factor=2.0)
        assert not passed
        assert any("warm" in f for f in failures)
        assert not any("cold" in f for f in failures)

    def test_exactly_at_threshold_passes(self) -> None:
        baseline = [self._metrics(latency_p95_ms=1000.0)]
        current  = [self._metrics(latency_p95_ms=2000.0)]  # exactly 2× — not > 2×
        passed, failures = check_regression(current, baseline, factor=2.0)
        assert passed
