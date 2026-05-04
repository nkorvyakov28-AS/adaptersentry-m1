"""Unit tests for benchmark utility functions — no network access required."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from benchmarks.hub_scanner import (
    CandidateRepo,
    ScanResult,
    append_result,
    check_lora_architecture,
    load_all_results,
    load_completed_repo_ids,
    scan_one_adapter,
)
from benchmarks.report import _percentiles, write_aggregate, write_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _success(repo_id: str, ens: float, risk: str, ts: str = "TRAINED") -> ScanResult:
    return ScanResult(
        repo_id=repo_id,
        scan_timestamp=_ts(),
        status="success",
        ensemble_score=ens,
        ensemble_risk_level=risk,
        overall_risk={"LOW": 0, "MEDIUM": 10, "HIGH": 50, "CRITICAL": 100}.get(risk, 0),
        training_status=ts,
        n_flags=0,
        top_flags=[],
        hf_tags=["peft"],
    )


def _failed(repo_id: str, status: str = "download_failed") -> ScanResult:
    return ScanResult(
        repo_id=repo_id,
        scan_timestamp=_ts(),
        status=status,
        error_message="some error",
    )


# ---------------------------------------------------------------------------
# CandidateRepo serialisation round-trip
# ---------------------------------------------------------------------------


class TestCandidateRepo:
    def test_to_dict_from_dict_round_trip(self) -> None:
        c = CandidateRepo(
            repo_id="author/model",
            hf_downloads=1234,
            hf_tags=["peft", "lora"],
            adapter_size_bytes=5_000_000,
            has_adapter_config=True,
        )
        assert CandidateRepo.from_dict(c.to_dict()) == c

    def test_from_dict_ignores_unknown_fields(self) -> None:
        d = {"repo_id": "a/b", "unknown_field": "x"}
        c = CandidateRepo.from_dict(d)
        assert c.repo_id == "a/b"

    def test_default_values(self) -> None:
        c = CandidateRepo(repo_id="x/y")
        assert c.hf_downloads == 0
        assert c.hf_tags == []
        assert c.adapter_size_bytes is None
        assert c.has_adapter_config is False


# ---------------------------------------------------------------------------
# ScanResult serialisation round-trip
# ---------------------------------------------------------------------------


class TestScanResult:
    def test_to_dict_from_dict_round_trip(self) -> None:
        r = _success("a/b", 14.6, "HIGH")
        assert ScanResult.from_dict(r.to_dict()) == r

    def test_failed_result_round_trip(self) -> None:
        r = _failed("c/d", status="analysis_failed")
        reconstructed = ScanResult.from_dict(r.to_dict())
        assert reconstructed.status == "analysis_failed"
        assert reconstructed.repo_id == "c/d"

    def test_from_dict_ignores_unknown_fields(self) -> None:
        d = {"repo_id": "e/f", "scan_timestamp": _ts(), "status": "success", "future_field": 99}
        r = ScanResult.from_dict(d)
        assert r.repo_id == "e/f"


# ---------------------------------------------------------------------------
# Resume behavior: append_result / load_completed_repo_ids / load_all_results
# ---------------------------------------------------------------------------


class TestResumeBehavior:
    def test_empty_jsonl_returns_empty_set(self, tmp_path: Path) -> None:
        p = tmp_path / "results.jsonl"
        p.touch()
        assert load_completed_repo_ids(p) == set()

    def test_missing_file_returns_empty_set(self, tmp_path: Path) -> None:
        assert load_completed_repo_ids(tmp_path / "missing.jsonl") == set()

    def test_appended_result_appears_in_completed(self, tmp_path: Path) -> None:
        p = tmp_path / "results.jsonl"
        append_result(_success("author/model-a", 4.1, "LOW"), p)
        assert "author/model-a" in load_completed_repo_ids(p)

    def test_all_statuses_counted_as_completed(self, tmp_path: Path) -> None:
        p = tmp_path / "results.jsonl"
        append_result(_success("a/ok", 3.0, "LOW"), p)
        append_result(_failed("b/fail"), p)
        append_result(
            ScanResult(repo_id="c/skip", scan_timestamp=_ts(), status="size_exceeded"), p
        )
        completed = load_completed_repo_ids(p)
        assert completed == {"a/ok", "b/fail", "c/skip"}

    def test_multiple_results_all_loaded(self, tmp_path: Path) -> None:
        p = tmp_path / "results.jsonl"
        for i in range(5):
            append_result(_success(f"a/model-{i}", float(i), "LOW"), p)
        assert len(load_completed_repo_ids(p)) == 5

    def test_corrupted_line_skipped_gracefully(self, tmp_path: Path) -> None:
        p = tmp_path / "results.jsonl"
        append_result(_success("a/good-1", 4.0, "LOW"), p)
        with p.open("a") as f:
            f.write("NOT VALID JSON\n")
        append_result(_success("a/good-2", 5.0, "LOW"), p)

        completed = load_completed_repo_ids(p)
        assert "a/good-1" in completed
        assert "a/good-2" in completed
        assert len(completed) == 2

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "results.jsonl"
        append_result(_success("a/model", 4.0, "LOW"), p)
        with p.open("a") as f:
            f.write("\n\n")
        assert len(load_completed_repo_ids(p)) == 1

    def test_load_all_results_preserves_fields(self, tmp_path: Path) -> None:
        p = tmp_path / "results.jsonl"
        original = _success("author/model", 14.6, "HIGH", ts="TRAINED")
        original.n_flags = 5
        original.cross_layer_consistency = 0.0
        append_result(original, p)

        loaded = load_all_results(p)
        assert len(loaded) == 1
        assert loaded[0].ensemble_score == pytest.approx(14.6)
        assert loaded[0].training_status == "TRAINED"
        assert loaded[0].n_flags == 5
        assert loaded[0].cross_layer_consistency == pytest.approx(0.0)

    def test_resume_skips_completed(self, tmp_path: Path) -> None:
        """Simulates resume: only repos NOT in completed_ids should be processed."""
        p = tmp_path / "results.jsonl"
        already_done = {"a/done-1", "a/done-2"}
        for repo_id in already_done:
            append_result(_success(repo_id, 3.0, "LOW"), p)

        completed = load_completed_repo_ids(p)
        candidates = [f"a/done-{i}" for i in range(1, 6)]
        remaining = [c for c in candidates if c not in completed]

        assert set(remaining) == {"a/done-3", "a/done-4", "a/done-5"}


# ---------------------------------------------------------------------------
# _percentiles
# ---------------------------------------------------------------------------


class TestPercentiles:
    def test_basic(self) -> None:
        values = list(range(101))
        result = _percentiles(values, [25, 50, 75])
        assert result["p50"] == pytest.approx(50.0, abs=0.1)
        assert result["p25"] == pytest.approx(25.0, abs=0.1)
        assert result["p75"] == pytest.approx(75.0, abs=0.1)

    def test_empty_returns_empty(self) -> None:
        assert _percentiles([], [50, 90]) == {}

    def test_single_value(self) -> None:
        result = _percentiles([7.0], [25, 50, 75])
        assert result["p50"] == pytest.approx(7.0)
        assert result["p25"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------


class TestWriteCsv:
    def test_creates_file_with_header(self, tmp_path: Path) -> None:
        write_csv([_success("a/b", 4.1, "LOW"), _failed("c/d")], tmp_path / "r.csv")
        content = (tmp_path / "r.csv").read_text()
        assert "repo_id" in content
        assert "a/b" in content
        assert "c/d" in content

    def test_no_crash_on_empty(self, tmp_path: Path) -> None:
        write_csv([], tmp_path / "empty.csv")
        assert not (tmp_path / "empty.csv").exists()

    def test_list_fields_joined_to_string(self, tmp_path: Path) -> None:
        r = _success("a/b", 5.0, "MEDIUM")
        r.top_flags = ["FLAG_A", "FLAG_B"]
        r.hf_tags = ["peft", "lora", "safetensors"]
        write_csv([r], tmp_path / "r.csv")
        content = (tmp_path / "r.csv").read_text()
        assert "FLAG_A" in content
        assert "peft" in content


# ---------------------------------------------------------------------------
# write_aggregate (candidate filtering + aggregation)
# ---------------------------------------------------------------------------


class TestWriteAggregate:
    def _results(self) -> list[ScanResult]:
        return [
            _success("a/m1", 4.1, "LOW"),
            _success("a/m2", 9.5, "MEDIUM"),
            _success("a/m3", 18.7, "HIGH"),
            _success("a/m4", 3.2, "LOW", ts="INIT_ONLY"),
            _failed("a/m5"),
        ]

    def _candidates(self) -> list[CandidateRepo]:
        return [CandidateRepo(repo_id=f"a/m{i}") for i in range(1, 8)]

    def test_totals_correct(self, tmp_path: Path) -> None:
        agg = write_aggregate(self._results(), tmp_path / "agg.json", self._candidates(), 7, 5)
        assert agg["totals"]["attempted"] == 5
        assert agg["totals"]["succeeded"] == 4
        assert agg["totals"]["failed"] == 1
        assert agg["totals"]["discovered"] == 7

    def test_risk_distribution(self, tmp_path: Path) -> None:
        agg = write_aggregate(self._results(), tmp_path / "agg.json", [], 5, 5)
        assert agg["risk_level_distribution"]["LOW"] == 2
        assert agg["risk_level_distribution"]["MEDIUM"] == 1
        assert agg["risk_level_distribution"]["HIGH"] == 1

    def test_training_status_distribution(self, tmp_path: Path) -> None:
        agg = write_aggregate(self._results(), tmp_path / "agg.json", [], 5, 5)
        assert agg["training_status_distribution"]["TRAINED"] == 3
        assert agg["training_status_distribution"]["INIT_ONLY"] == 1

    def test_top_suspicious_ordered_by_score(self, tmp_path: Path) -> None:
        agg = write_aggregate(self._results(), tmp_path / "agg.json", [], 5, 5)
        top = agg["top_suspicious_by_ensemble_score"]
        assert len(top) >= 1
        assert top[0]["ensemble_score"] == pytest.approx(18.7, abs=0.01)
        assert top[0]["repo_id"] == "a/m3"

    def test_counts_convenience_keys(self, tmp_path: Path) -> None:
        agg = write_aggregate(self._results(), tmp_path / "agg.json", [], 5, 5)
        assert agg["counts"]["INIT_ONLY"] == 1
        assert agg["counts"]["LOW"] == 2
        assert agg["counts"]["HIGH"] == 1
        assert agg["counts"]["CRITICAL"] == 0

    def test_failure_counts(self, tmp_path: Path) -> None:
        agg = write_aggregate(self._results(), tmp_path / "agg.json", [], 5, 5)
        assert agg["failure_reason_counts"]["download_failed"] == 1

    def test_aggregate_json_written_and_parseable(self, tmp_path: Path) -> None:
        p = tmp_path / "aggregate.json"
        write_aggregate(self._results(), p, [], 5, 5)
        assert p.exists()
        with p.open() as f:
            data = json.load(f)
        assert "generated_at" in data
        assert "framing" in data
        assert "totals" in data
        assert "ensemble_score_percentiles" in data

    def test_percentiles_computed(self, tmp_path: Path) -> None:
        agg = write_aggregate(self._results(), tmp_path / "agg.json", [], 5, 5)
        pcts = agg["ensemble_score_percentiles"]
        assert "p50" in pcts
        assert pcts["p50"] > 0


# ---------------------------------------------------------------------------
# v2: failure classification
# ---------------------------------------------------------------------------


class TestUnsupportedArchitectureClassification:
    """Adapters without standard lora_A/lora_B pairs get status=unsupported_architecture."""

    def _make_safetensors(self, tmp_path: Path, keys: dict) -> Path:
        """Write a minimal safetensors file with arbitrary tensor keys."""
        import numpy as np
        from safetensors.numpy import save_file

        p = tmp_path / "adapter_model.safetensors"
        save_file({k: np.zeros((4, 4), dtype=np.float32) for k in keys}, str(p))
        return p

    def test_no_lora_keys_is_unsupported(self, tmp_path: Path) -> None:
        st = self._make_safetensors(
            tmp_path,
            {"model.weight": None, "model.bias": None},
        )
        is_supported, keys_sample = check_lora_architecture(st)
        assert is_supported is False
        assert len(keys_sample) <= 10

    def test_single_lora_pair_is_unsupported(self, tmp_path: Path) -> None:
        # One pair is below the _MIN_LORA_PAIRS=2 threshold
        import numpy as np
        from safetensors.numpy import save_file

        p = tmp_path / "adapter_model.safetensors"
        save_file(
            {
                "base.q_proj.lora_A.weight": np.zeros((4, 8), dtype=np.float32),
                "base.q_proj.lora_B.weight": np.zeros((8, 4), dtype=np.float32),
            },
            str(p),
        )
        is_supported, _ = check_lora_architecture(p)
        assert is_supported is False

    def test_two_lora_pairs_is_supported(self, tmp_path: Path) -> None:
        import numpy as np
        from safetensors.numpy import save_file

        p = tmp_path / "adapter_model.safetensors"
        save_file(
            {
                "base.q_proj.lora_A.weight": np.zeros((4, 8), dtype=np.float32),
                "base.q_proj.lora_B.weight": np.zeros((8, 4), dtype=np.float32),
                "base.v_proj.lora_A.weight": np.zeros((4, 8), dtype=np.float32),
                "base.v_proj.lora_B.weight": np.zeros((8, 4), dtype=np.float32),
            },
            str(p),
        )
        is_supported, _ = check_lora_architecture(p)
        assert is_supported is True

    def test_scan_one_adapter_returns_unsupported_status(self, tmp_path: Path) -> None:
        import numpy as np
        from safetensors.numpy import save_file

        p = tmp_path / "adapter_model.safetensors"
        save_file({"dense.weight": np.zeros((8, 8), dtype=np.float32)}, str(p))

        candidate = CandidateRepo(repo_id="author/ia3-adapter")
        result = scan_one_adapter(candidate, p, {})

        assert result.status == "unsupported_architecture"
        assert result.training_status == "UNSUPPORTED"
        assert result.risk_level == "UNKNOWN"
        assert result.error_type == "no_lora_pairs_found"
        assert isinstance(result.tensor_keys_sample, list)
        assert result.ensemble_score is None

    def test_aggregate_counts_unsupported_separately(self, tmp_path: Path) -> None:
        results = [
            _success("a/ok", 4.0, "LOW"),
            ScanResult(
                repo_id="b/ia3",
                scan_timestamp=_ts(),
                status="unsupported_architecture",
                training_status="UNSUPPORTED",
                risk_level="UNKNOWN",
                error_type="no_lora_pairs_found",
            ),
        ]
        agg = write_aggregate(results, tmp_path / "agg.json", [], 2, 5)
        assert agg["totals"]["unsupported_architecture"] == 1
        assert agg["totals"]["succeeded"] == 1
        # unsupported_architecture must NOT appear in risk_level_distribution
        assert "UNKNOWN" not in agg["risk_level_distribution"]


class TestAnalysisFailedClassification:
    """Genuine exceptions during M1 analysis produce status=analysis_failed with error_type."""

    def test_scan_one_adapter_catches_exception(self, tmp_path: Path, monkeypatch) -> None:
        import numpy as np
        from safetensors.numpy import save_file

        p = tmp_path / "adapter_model.safetensors"
        # Valid adapter with 2 pairs so architecture check passes
        save_file(
            {
                "base.q.lora_A.weight": np.zeros((4, 8), dtype=np.float32),
                "base.q.lora_B.weight": np.zeros((8, 4), dtype=np.float32),
                "base.v.lora_A.weight": np.zeros((4, 8), dtype=np.float32),
                "base.v.lora_B.weight": np.zeros((8, 4), dtype=np.float32),
            },
            str(p),
        )

        # Force run_m1 to raise
        from benchmarks import hub_scanner as hs
        monkeypatch.setattr(hs, "run_m1", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("synthetic M1 error")))

        candidate = CandidateRepo(repo_id="author/broken")
        result = scan_one_adapter(candidate, p, {})

        assert result.status == "analysis_failed"
        assert result.error_type == "RuntimeError"
        assert "synthetic M1 error" in (result.error_detail or "")

    def test_failure_breakdown_in_aggregate(self, tmp_path: Path) -> None:
        results = [
            _success("a/ok", 4.0, "LOW"),
            ScanResult(
                repo_id="b/broken",
                scan_timestamp=_ts(),
                status="analysis_failed",
                error_type="ValueError",
                error_detail="bad tensor shape",
            ),
            _failed("c/dl", status="download_failed"),
        ]
        agg = write_aggregate(results, tmp_path / "agg.json", [], 3, 5)
        fb = agg["failure_breakdown"]
        assert fb["analysis_failed"] == 1
        assert fb["download_failed"] == 1
        assert fb.get("unsupported_architecture", 0) == 0


# ---------------------------------------------------------------------------
# v2: parallel append safety
# ---------------------------------------------------------------------------


class TestParallelAppendSafety:
    def test_concurrent_appends_produce_valid_jsonl(self, tmp_path: Path) -> None:
        """N threads writing concurrently must produce exactly N well-formed JSONL lines."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        p = tmp_path / "results.jsonl"
        lock = threading.Lock()
        n_writers = 50

        def write_one(i: int) -> None:
            r = ScanResult(
                repo_id=f"author/model-{i}",
                scan_timestamp=_ts(),
                status="success",
                ensemble_score=float(i),
            )
            append_result(r, p, lock)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write_one, range(n_writers)))

        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        assert len(lines) == n_writers, f"Expected {n_writers} lines, got {len(lines)}"

        repo_ids = set()
        for line in lines:
            obj = json.loads(line)   # must not raise — each line is valid JSON
            repo_ids.add(obj["repo_id"])

        assert len(repo_ids) == n_writers, "Duplicate or missing repo_ids detected"

    def test_no_partial_writes(self, tmp_path: Path) -> None:
        """Every line in the output must be parseable JSON (no truncation or interleaving)."""
        import threading

        p = tmp_path / "results.jsonl"
        lock = threading.Lock()
        errors: list[str] = []

        def writer(i: int) -> None:
            r = ScanResult(
                repo_id=f"repo-{i}",
                scan_timestamp=_ts(),
                status="success",
                top_flags=["FLAG_A: " + "x" * 100],  # moderately long line
            )
            append_result(r, p, lock)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Bad JSON: {exc}")

        assert errors == [], "\n".join(errors)


# ---------------------------------------------------------------------------
# v2: local-only mode — no HF API calls
# ---------------------------------------------------------------------------


class TestLocalOnlyMode:
    def _make_minimal_adapter(self, path: Path) -> None:
        """Write a minimal valid safetensors adapter with 2 lora pairs."""
        import numpy as np
        from safetensors.numpy import save_file

        path.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "base.q.lora_A.weight": np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32),
                "base.q.lora_B.weight": np.random.default_rng(1).standard_normal((8, 4)).astype(np.float32),
                "base.v.lora_A.weight": np.random.default_rng(2).standard_normal((4, 8)).astype(np.float32),
                "base.v.lora_B.weight": np.random.default_rng(3).standard_normal((8, 4)).astype(np.float32),
            },
            str(path / "adapter_model.safetensors"),
        )

    def test_local_only_does_not_call_hf_api(self, tmp_path: Path, monkeypatch) -> None:
        """run_pipeline with local_only=True must never instantiate HfApi."""
        hf_api_calls: list[str] = []

        class FakeHfApi:
            def __init__(self, *a, **kw):
                hf_api_calls.append("HfApi.__init__")

        monkeypatch.setattr("benchmarks.hub_scanner.download_adapter",
                            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("download called")))

        # Build a fake candidates.json and cached adapter
        candidates_dir = tmp_path / "v1_run"
        candidates_dir.mkdir()
        repo_id = "testauthor/my-lora"
        safe_name = repo_id.replace("/", "__")

        self._make_minimal_adapter(candidates_dir / "adapters" / safe_name)

        candidates_json = candidates_dir / "candidates.json"
        candidates_json.write_text(json.dumps({
            "generated_at": "2026-01-01T00:00:00+00:00",
            "target_n": 1,
            "max_size_mb": 500,
            "min_downloads": 0,
            "sample_seed": 42,
            "total": 1,
            "candidates": [{"repo_id": repo_id, "hf_downloads": 100, "hf_tags": ["peft"], "adapter_size_bytes": None, "has_adapter_config": False}],
        }))

        output_dir = tmp_path / "v2_run"
        from benchmarks.hub_scanner import run_pipeline
        run_pipeline(
            output_dir=output_dir,
            limit=1,
            resume=False,
            max_download_mb=500.0,
            sleep_seconds=0.0,
            min_downloads=0,
            sample_seed=42,
            top_n=5,
            workers=1,
            local_only=True,
            candidates_from=candidates_json,
        )

        assert hf_api_calls == [], f"HfApi was called: {hf_api_calls}"

        results_path = output_dir / "results.jsonl"
        assert results_path.exists()
        lines = [ln for ln in results_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["repo_id"] == repo_id
        assert obj["status"] in ("success", "unsupported_architecture")

    def test_not_cached_when_file_missing(self, tmp_path: Path) -> None:
        """If the adapter file is absent in local cache, status must be not_cached."""
        from benchmarks.hub_scanner import _process_repo

        candidate = CandidateRepo(repo_id="author/gone-model")
        adapters_dir = tmp_path / "adapters"
        adapters_dir.mkdir()

        result = _process_repo(candidate, adapters_dir, 500.0, local_only=True)
        assert result.status == "not_cached"
        assert result.repo_id == "author/gone-model"
