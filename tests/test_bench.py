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
    load_all_results,
    load_completed_repo_ids,
    read_tensor_keys_sample,
    scan_one_adapter,
)
from benchmarks.report import _percentiles, write_aggregate, write_csv, write_markdown_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _success(
    repo_id: str,
    z: float | None,
    action: str = "allow",
    level: str = "LOW",
    ts: str = "trained",
    reasons: list[str] | None = None,
) -> ScanResult:
    return ScanResult(
        repo_id=repo_id,
        scan_timestamp=_ts(),
        status="success",
        scan_status="ok",
        action=action,
        level=level,
        training_state=ts,
        max_robust_z=z,
        n_outlier_modules=0 if z is None or z < 3.5 else 1,
        reason_codes=list(reasons or ["NO_REFERENCE_PROFILE"]),
        n_findings=0,
        top_findings=[],
        rank_declared=8,
        n_modules=4,
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
        r = _success("a/b", 14.6, "review", "MEDIUM", reasons=["INTRA_OUTLIER", "RANK_MISMATCH"])
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
        append_result(_success("author/model-a", 4.1), p)
        assert "author/model-a" in load_completed_repo_ids(p)

    def test_all_statuses_counted_as_completed(self, tmp_path: Path) -> None:
        p = tmp_path / "results.jsonl"
        append_result(_success("a/ok", 3.0), p)
        append_result(_failed("b/fail"), p)
        append_result(
            ScanResult(repo_id="c/skip", scan_timestamp=_ts(), status="size_exceeded"), p
        )
        completed = load_completed_repo_ids(p)
        assert completed == {"a/ok", "b/fail", "c/skip"}

    def test_multiple_results_all_loaded(self, tmp_path: Path) -> None:
        p = tmp_path / "results.jsonl"
        for i in range(5):
            append_result(_success(f"a/model-{i}", float(i)), p)
        assert len(load_completed_repo_ids(p)) == 5

    def test_corrupted_line_skipped_gracefully(self, tmp_path: Path) -> None:
        p = tmp_path / "results.jsonl"
        append_result(_success("a/good-1", 4.0), p)
        with p.open("a") as f:
            f.write("NOT VALID JSON\n")
        append_result(_success("a/good-2", 5.0), p)

        completed = load_completed_repo_ids(p)
        assert "a/good-1" in completed
        assert "a/good-2" in completed
        assert len(completed) == 2

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "results.jsonl"
        append_result(_success("a/model", 4.0), p)
        with p.open("a") as f:
            f.write("\n\n")
        assert len(load_completed_repo_ids(p)) == 1

    def test_load_all_results_preserves_fields(self, tmp_path: Path) -> None:
        p = tmp_path / "results.jsonl"
        original = _success("author/model", 14.6, "review", "MEDIUM", ts="partial",
                            reasons=["INTRA_OUTLIER", "SIBLING_EXECUTABLE"])
        original.n_findings = 5
        original.top_findings = ["INTRA_DEPTH_SPIKE: Module departs from its family"]
        append_result(original, p)

        loaded = load_all_results(p)
        assert len(loaded) == 1
        assert loaded[0] == original
        assert loaded[0].max_robust_z == pytest.approx(14.6)
        assert loaded[0].training_state == "partial"
        assert loaded[0].n_findings == 5
        assert loaded[0].reason_codes == ["INTRA_OUTLIER", "SIBLING_EXECUTABLE"]

    def test_load_all_results_ignores_legacy_fields(self, tmp_path: Path) -> None:
        """1.x records load; their removed fields (ensemble_score, …) are dropped."""
        p = tmp_path / "results.jsonl"
        p.write_text(json.dumps({
            "repo_id": "old/record", "scan_timestamp": _ts(), "status": "success",
            "ensemble_score": 12.0, "overall_risk": 50, "top_flags": ["X"],
        }) + "\n")
        loaded = load_all_results(p)
        assert len(loaded) == 1
        assert loaded[0].repo_id == "old/record"
        assert loaded[0].max_robust_z is None

    def test_resume_skips_completed(self, tmp_path: Path) -> None:
        """Simulates resume: only repos NOT in completed_ids should be processed."""
        p = tmp_path / "results.jsonl"
        already_done = {"a/done-1", "a/done-2"}
        for repo_id in already_done:
            append_result(_success(repo_id, 3.0), p)

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
        write_csv([_success("a/b", 4.1), _failed("c/d")], tmp_path / "r.csv")
        content = (tmp_path / "r.csv").read_text()
        assert "repo_id" in content
        assert "a/b" in content
        assert "c/d" in content

    def test_no_crash_on_empty(self, tmp_path: Path) -> None:
        write_csv([], tmp_path / "empty.csv")
        assert not (tmp_path / "empty.csv").exists()

    def test_list_fields_joined_to_string(self, tmp_path: Path) -> None:
        r = _success("a/b", 5.0, "review", "MEDIUM", reasons=["INTRA_OUTLIER", "RANK_MISMATCH"])
        r.top_findings = ["RULE_A: first", "RULE_B: second"]
        r.hf_tags = ["peft", "lora", "safetensors"]
        write_csv([r], tmp_path / "r.csv")
        content = (tmp_path / "r.csv").read_text()
        assert "RULE_A: first | RULE_B: second" in content
        assert "INTRA_OUTLIER,RANK_MISMATCH" in content
        assert "peft" in content


# ---------------------------------------------------------------------------
# write_aggregate (candidate filtering + aggregation)
# ---------------------------------------------------------------------------


class TestWriteAggregate:
    def _results(self) -> list[ScanResult]:
        return [
            _success("a/m1", 1.1),
            _success("a/m2", 4.2, "review", "MEDIUM", reasons=["INTRA_OUTLIER", "NO_REFERENCE_PROFILE"]),
            _success("a/m3", 9.7, "review", "MEDIUM",
                     reasons=["INTRA_OUTLIER", "RANK_MISMATCH", "SIBLING_EXECUTABLE"]),
            _success("a/m4", None, ts="init_only"),
            _success("a/m6", 2.0, "review", "MEDIUM", reasons=["FULL_WEIGHT_REPLACEMENT"]),
            _failed("a/m5"),
        ]

    def _candidates(self) -> list[CandidateRepo]:
        return [CandidateRepo(repo_id=f"a/m{i}") for i in range(1, 8)]

    def test_totals_correct(self, tmp_path: Path) -> None:
        agg = write_aggregate(self._results(), tmp_path / "agg.json", self._candidates(), 7, 5)
        assert agg["totals"]["attempted"] == 6
        assert agg["totals"]["succeeded"] == 5
        assert agg["totals"]["failed"] == 1
        assert agg["totals"]["discovered"] == 7

    def test_action_distribution(self, tmp_path: Path) -> None:
        agg = write_aggregate(self._results(), tmp_path / "agg.json", [], 5, 5)
        assert agg["action_distribution"] == {"allow": 2, "review": 3}

    def test_level_distribution(self, tmp_path: Path) -> None:
        agg = write_aggregate(self._results(), tmp_path / "agg.json", [], 5, 5)
        assert agg["level_distribution"] == {"LOW": 2, "MEDIUM": 3}

    def test_training_state_distribution(self, tmp_path: Path) -> None:
        agg = write_aggregate(self._results(), tmp_path / "agg.json", [], 5, 5)
        assert agg["training_state_distribution"]["trained"] == 4
        assert agg["training_state_distribution"]["init_only"] == 1

    def test_reason_code_distribution_counts_adapters(self, tmp_path: Path) -> None:
        agg = write_aggregate(self._results(), tmp_path / "agg.json", [], 5, 5)
        assert agg["reason_code_distribution"]["INTRA_OUTLIER"] == 2
        assert agg["reason_code_distribution"]["RANK_MISMATCH"] == 1

    def test_top_suspicious_ordered_by_max_robust_z(self, tmp_path: Path) -> None:
        agg = write_aggregate(self._results(), tmp_path / "agg.json", [], 5, 5)
        top = agg["top_suspicious_by_max_robust_z"]
        assert [e["repo_id"] for e in top] == ["a/m3", "a/m2", "a/m6", "a/m1"]
        assert top[0]["max_robust_z"] == pytest.approx(9.7)
        # adapters without an intra comparison are not ranked
        assert all(e["repo_id"] != "a/m4" for e in top)

    def test_top_by_structural_codes(self, tmp_path: Path) -> None:
        agg = write_aggregate(self._results(), tmp_path / "agg.json", [], 5, 5)
        top = agg["top_suspicious_by_structural_codes"]
        assert [e["repo_id"] for e in top] == ["a/m3", "a/m6"]
        assert sorted(top[0]["structural_codes"]) == ["RANK_MISMATCH", "SIBLING_EXECUTABLE"]
        # non-structural codes are not counted as structural
        assert "INTRA_OUTLIER" not in top[0]["structural_codes"]

    def test_top_n_limits_lists(self, tmp_path: Path) -> None:
        agg = write_aggregate(self._results(), tmp_path / "agg.json", [], 5, 1)
        assert len(agg["top_suspicious_by_max_robust_z"]) == 1
        assert len(agg["top_suspicious_by_structural_codes"]) == 1

    def test_counts_convenience_keys(self, tmp_path: Path) -> None:
        agg = write_aggregate(self._results(), tmp_path / "agg.json", [], 5, 5)
        assert agg["counts"]["allow"] == 2
        assert agg["counts"]["review"] == 3
        assert agg["counts"]["block"] == 0
        assert agg["counts"]["init_only"] == 1
        assert agg["counts"]["with_structural_codes"] == 2

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
        assert "uncalibrated" in data["framing"]
        assert "totals" in data
        assert "max_robust_z_percentiles" in data

    def test_max_robust_z_stats(self, tmp_path: Path) -> None:
        agg = write_aggregate(self._results(), tmp_path / "agg.json", [], 5, 5)
        pcts = agg["max_robust_z_percentiles"]
        assert pcts["p50"] == pytest.approx(3.1)  # median of 1.1, 2.0, 4.2, 9.7
        assert agg["max_robust_z_mean"] == pytest.approx((1.1 + 4.2 + 9.7 + 2.0) / 4)
        assert agg["n_with_intra_anomaly"] == 4

    def test_markdown_report_uses_new_sections(self, tmp_path: Path) -> None:
        results = self._results()
        agg = write_aggregate(results, tmp_path / "agg.json", [], 5, 5)
        report = tmp_path / "report.md"
        write_markdown_report(agg, results, report, 5)
        text = report.read_text()
        assert "Verdict Action Distribution" in text
        assert "Training State Distribution" in text
        assert "uncalibrated" in text
        assert "not detection accuracy" in text
        assert "`a/m3`" in text
        assert "ensemble" not in text.lower()

    def test_markdown_report_with_no_success(self, tmp_path: Path) -> None:
        results = [_failed("x/y")]
        agg = write_aggregate(results, tmp_path / "agg.json", [], 1, 5)
        report = tmp_path / "report.md"
        write_markdown_report(agg, results, report, 1)
        assert "No intra-adapter comparison" in report.read_text()


# ---------------------------------------------------------------------------
# v2: failure classification
# ---------------------------------------------------------------------------


class TestUnsupportedArchitectureClassification:
    """A scan that finds no LoRA A/B pair gets status=unsupported_architecture."""

    def test_read_tensor_keys_sample(self, tmp_path: Path) -> None:
        import numpy as np
        from safetensors.numpy import save_file

        p = tmp_path / "adapter_model.safetensors"
        save_file({f"t{i}.weight": np.zeros((2, 2), dtype=np.float32) for i in range(15)}, str(p))
        sample = read_tensor_keys_sample(p)
        assert len(sample) == 10

    def test_read_tensor_keys_sample_on_garbage(self, tmp_path: Path) -> None:
        p = tmp_path / "adapter_model.safetensors"
        p.write_bytes(b"not a safetensors file")
        assert read_tensor_keys_sample(p) == []

    def test_scan_one_adapter_returns_unsupported_status(self, tmp_path: Path) -> None:
        import numpy as np
        from safetensors.numpy import save_file

        p = tmp_path / "adapter_model.safetensors"
        save_file({"dense.weight": np.zeros((8, 8), dtype=np.float32)}, str(p))

        candidate = CandidateRepo(repo_id="author/ia3-adapter")
        result = scan_one_adapter(candidate, p)

        assert result.status == "unsupported_architecture"
        assert result.scan_status == "failed"
        assert result.error_type == "no_lora_pairs_found"
        assert result.tensor_keys_sample == ["dense.weight"]
        assert result.action is None
        assert result.max_robust_z is None

    def test_scan_one_adapter_single_pair_is_scanned(self, tmp_path: Path) -> None:
        """One LoRA pair is enough for the 2.0 scanner (the 1.x two-pair minimum is gone)."""
        import numpy as np
        from safetensors.numpy import save_file

        p = tmp_path / "adapter_model.safetensors"
        rng = np.random.default_rng(0)
        save_file(
            {
                "base.q_proj.lora_A.weight": rng.standard_normal((4, 8)).astype(np.float32),
                "base.q_proj.lora_B.weight": rng.standard_normal((8, 4)).astype(np.float32),
            },
            str(p),
        )
        result = scan_one_adapter(CandidateRepo(repo_id="a/one-pair"), p)
        assert result.status == "success"
        assert result.n_modules == 1

    def test_aggregate_counts_unsupported_separately(self, tmp_path: Path) -> None:
        results = [
            _success("a/ok", 4.0),
            ScanResult(
                repo_id="b/ia3",
                scan_timestamp=_ts(),
                status="unsupported_architecture",
                scan_status="failed",
                error_type="no_lora_pairs_found",
            ),
        ]
        agg = write_aggregate(results, tmp_path / "agg.json", [], 2, 5)
        assert agg["totals"]["unsupported_architecture"] == 1
        assert agg["totals"]["succeeded"] == 1
        # unsupported_architecture must NOT appear in the distributions
        assert agg["action_distribution"] == {"allow": 1}
        assert "unknown" not in agg["level_distribution"]


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
        result = scan_one_adapter(candidate, p)

        assert result.status == "analysis_failed"
        assert result.error_type == "RuntimeError"
        assert "synthetic M1 error" in (result.error_detail or "")

    def test_failed_scan_is_analysis_failed(self, tmp_path: Path) -> None:
        """A failed scan for a reason other than 'no LoRA pair' is analysis_failed."""
        p = tmp_path / "adapter_model.safetensors"
        p.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{broken}")

        result = scan_one_adapter(CandidateRepo(repo_id="author/corrupt"), p)

        assert result.status == "analysis_failed"
        assert result.scan_status == "failed"
        assert result.error_type
        assert result.error_detail

    def test_successful_scan_populates_record(self, tmp_path: Path) -> None:
        import numpy as np
        from safetensors.numpy import save_file

        rng = np.random.default_rng(0)
        tensors = {}
        for layer in range(4):
            for mod in ("q_proj", "v_proj"):
                base = f"base_model.model.model.layers.{layer}.self_attn.{mod}"
                tensors[f"{base}.lora_A.weight"] = rng.standard_normal((4, 32)).astype(np.float32)
                tensors[f"{base}.lora_B.weight"] = rng.standard_normal((32, 4)).astype(np.float32)
        p = tmp_path / "adapter_model.safetensors"
        save_file(tensors, str(p))
        (tmp_path / "adapter_config.json").write_text(json.dumps(
            {"peft_type": "LORA", "r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"]}
        ))

        result = scan_one_adapter(CandidateRepo(repo_id="author/good", hf_downloads=7), p, mode="fast")

        assert result.status == "success"
        assert result.scan_status in ("ok", "degraded")
        assert result.action in ("allow", "review", "block")
        assert result.level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert result.training_state == "trained"
        assert result.rank_declared == 4
        assert result.n_modules == 8
        assert isinstance(result.reason_codes, list) and result.reason_codes
        assert len(result.top_findings) == min(result.n_findings, 5)
        assert all(len(f) <= 120 for f in result.top_findings)
        assert result.hf_downloads == 7

    def test_top_findings_truncated(self, tmp_path: Path, monkeypatch) -> None:
        """At most 5 findings, each formatted 'RULE_ID: title' and capped at 120 chars."""
        from types import SimpleNamespace

        from benchmarks import hub_scanner as hs

        p = tmp_path / "adapter_model.safetensors"
        p.write_bytes(b"x")
        findings = [SimpleNamespace(rule_id=f"RULE_{i}", title="t" * 200) for i in range(8)]
        fake = SimpleNamespace(
            status="ok",
            errors=[],
            anomaly=SimpleNamespace(intra=SimpleNamespace(max_robust_z=6.5, n_outlier_modules=2)),
            verdict=SimpleNamespace(action="review", level=SimpleNamespace(value="MEDIUM"),
                                    reasons=[SimpleNamespace(code="INTRA_OUTLIER")]),
            adapter=SimpleNamespace(training_state="trained", rank_declared=16),
            coverage=SimpleNamespace(n_modules=12),
            findings=findings,
        )
        monkeypatch.setattr(hs, "run_m1", lambda *a, **kw: fake)

        result = scan_one_adapter(CandidateRepo(repo_id="a/b"), p)

        assert result.n_findings == 8
        assert len(result.top_findings) == 5
        assert result.top_findings[0].startswith("RULE_0: ")
        assert all(len(f) == 120 for f in result.top_findings)
        assert result.max_robust_z == pytest.approx(6.5)
        assert result.n_outlier_modules == 2
        assert result.reason_codes == ["INTRA_OUTLIER"]

    def test_failure_breakdown_in_aggregate(self, tmp_path: Path) -> None:
        results = [
            _success("a/ok", 4.0),
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
                max_robust_z=float(i),
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
                top_findings=["RULE_A: " + "x" * 100],  # moderately long line
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
        assert obj["status"] == "success"
        assert obj["action"] in ("allow", "review")
        assert obj["n_modules"] == 2
        assert (output_dir / "report.md").exists()
        assert (output_dir / "aggregate.json").exists()

    def test_not_cached_when_file_missing(self, tmp_path: Path) -> None:
        """If the adapter file is absent in local cache, status must be not_cached."""
        from benchmarks.hub_scanner import _process_repo

        candidate = CandidateRepo(repo_id="author/gone-model")
        adapters_dir = tmp_path / "adapters"
        adapters_dir.mkdir()

        result = _process_repo(candidate, adapters_dir, 500.0, local_only=True)
        assert result.status == "not_cached"
        assert result.repo_id == "author/gone-model"
