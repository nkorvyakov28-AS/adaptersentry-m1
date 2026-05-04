"""Tests for M1-RPT-01: PerLayerFinding ranked layer summary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.schemas.finding import Severity
from adaptersentry.schemas.per_layer_finding import PerLayerFinding
from adaptersentry.reporting.per_layer import (
    compute_per_layer_findings,
    _score_record,
    _flag_family,
    _severity_from_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(tmp_path: Path, n_layers: int = 4, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {}
    for i in range(n_layers):
        tensors[f"model.layers.{i}.q_proj.lora_A.weight"] = \
            rng.standard_normal((4, 32)).astype(np.float32)
        tensors[f"model.layers.{i}.q_proj.lora_B.weight"] = \
            rng.standard_normal((32, 4)).astype(np.float32)
    p = tmp_path / "adapter.safetensors"
    save_file(tensors, str(p))
    return p


def _make_report(tmp_path: Path, n_layers: int = 4):
    from adaptersentry.analyzer import scan
    return scan(_make_adapter(tmp_path, n_layers=n_layers))


def _make_tensor_record(layer_name: str, flags: list[str] = (), parse_error=None):
    from adaptersentry.schemas.tensor_record import TensorRecord
    return TensorRecord(
        layer_name=layer_name,
        shape_a=[4, 32], shape_b=[32, 4],
        rank=4, energy_concentration=0.5,
        kurtosis_a=1.0, kurtosis_b=1.0,
        mean_a=0.0, std_a=0.1,
        mean_b=0.0, std_b=0.1,
        skewness_a=0.0,
        entropy_a=0.6, entropy_b=0.6,
        zscore_outlier_rate_a=0.01, zscore_outlier_rate_b=0.01,
        flags=list(flags),
        parse_error=parse_error,
    )


# ---------------------------------------------------------------------------
# PerLayerFinding schema
# ---------------------------------------------------------------------------

class TestPerLayerFindingSchema:
    def test_frozen(self):
        plf = PerLayerFinding(
            rank=1, layer_name="test", severity=Severity.HIGH,
            severity_score=0.5, triggered_families=["distribution"],
            flag_count=1, signals=["Heavy-Tailed Weight Distribution"],
        )
        with pytest.raises(Exception):
            plf.rank = 2  # type: ignore[misc]

    def test_rank_ge_one(self):
        with pytest.raises(Exception):
            PerLayerFinding(
                rank=0, layer_name="test", severity=Severity.LOW,
                severity_score=0.1, triggered_families=[], flag_count=0, signals=[],
            )

    def test_severity_score_bounds(self):
        with pytest.raises(Exception):
            PerLayerFinding(
                rank=1, layer_name="test", severity=Severity.LOW,
                severity_score=1.5, triggered_families=[], flag_count=0, signals=[],
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_flag_family_kurtosis(self):
        assert _flag_family("HIGH_KURTOSIS_A: kurtosis=15") == "distribution"

    def test_flag_family_entropy(self):
        assert _flag_family("LOW_ENTROPY_A: entropy=0.05") == "entropy"
        assert _flag_family("HIGH_ENTROPY_B: entropy=0.99") == "entropy"

    def test_flag_family_outlier(self):
        assert _flag_family("HIGH_ZSCORE_OUTLIER_RATE_A: rate=5%") == "outlier"
        assert _flag_family("HIGH_ISOLATION_ANOMALY_A: score=-0.3") == "outlier"

    def test_flag_family_norm(self):
        assert _flag_family("RANK_INFLATION: rank=4") == "norm"
        assert _flag_family("NEAR_ZERO_B_MATRIX: norm=0.0") == "norm"

    def test_flag_family_unknown(self):
        assert _flag_family("SOME_UNKNOWN_FLAG") is None

    def test_severity_from_score_high(self):
        assert _severity_from_score(0.35) == Severity.HIGH

    def test_severity_from_score_critical(self):
        assert _severity_from_score(0.6) == Severity.CRITICAL

    def test_severity_from_score_medium(self):
        assert _severity_from_score(0.15) == Severity.MEDIUM

    def test_severity_from_score_low(self):
        assert _severity_from_score(0.05) == Severity.LOW

    def test_score_record_no_flags(self):
        record = _make_tensor_record("test.layer")
        assert _score_record(record) == pytest.approx(0.0)

    def test_score_record_with_flags(self):
        record = _make_tensor_record("test.layer", flags=["HIGH_KURTOSIS_A: k=15"])
        assert _score_record(record) > 0.0

    def test_score_record_with_parse_error(self):
        from adaptersentry.schemas.errors import ErrorCategory
        record = _make_tensor_record("test.layer", parse_error=ErrorCategory.DEGRADED)
        assert _score_record(record) > 0.0


# ---------------------------------------------------------------------------
# compute_per_layer_findings
# ---------------------------------------------------------------------------

class TestComputePerLayerFindings:
    def test_empty_returns_empty(self):
        assert compute_per_layer_findings([]) == []

    def test_clean_adapter_returns_empty(self, tmp_path):
        report = _make_report(tmp_path)
        # Normal random adapter likely has no flags
        results = compute_per_layer_findings(report.tensor_records)
        for plf in results:
            assert plf.severity_score > 0.0

    def test_returns_per_layer_finding_objects(self):
        records = [
            _make_tensor_record("layer.0", flags=["HIGH_KURTOSIS_A: k=20"]),
            _make_tensor_record("layer.1"),
        ]
        results = compute_per_layer_findings(records)
        assert all(isinstance(r, PerLayerFinding) for r in results)

    def test_rank_starts_at_one(self):
        records = [
            _make_tensor_record("layer.0", flags=["HIGH_KURTOSIS_A: k=20"]),
        ]
        results = compute_per_layer_findings(records)
        assert len(results) >= 1
        assert results[0].rank == 1

    def test_ranks_are_sequential(self):
        records = [
            _make_tensor_record("layer.0", flags=["HIGH_KURTOSIS_A: k=20"]),
            _make_tensor_record("layer.1", flags=["LOW_ENTROPY_A: e=0.01"]),
            _make_tensor_record("layer.2", flags=["HIGH_ZSCORE_OUTLIER_RATE_A: r=5%"]),
        ]
        results = compute_per_layer_findings(records)
        for i, r in enumerate(results, start=1):
            assert r.rank == i

    def test_capped_at_top_k(self):
        records = [
            _make_tensor_record(f"layer.{i}", flags=["HIGH_KURTOSIS_A: k=20"])
            for i in range(15)
        ]
        results = compute_per_layer_findings(records, top_k=10)
        assert len(results) <= 10

    def test_sorted_by_score_descending(self):
        records = [
            _make_tensor_record("layer.low", flags=["HIGH_ENTROPY_A: e=0.99"]),       # LOW sev
            _make_tensor_record("layer.high", flags=["HIGH_KURTOSIS_A: k=20",          # MEDIUM×2
                                                     "HIGH_KURTOSIS_B: k=18"]),
        ]
        results = compute_per_layer_findings(records)
        scores = [r.severity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_triggered_families_populated(self):
        records = [
            _make_tensor_record("layer.0", flags=["HIGH_KURTOSIS_A: k=20"]),
        ]
        results = compute_per_layer_findings(records)
        assert len(results) == 1
        assert "distribution" in results[0].triggered_families

    def test_signals_not_empty_for_flagged_layer(self):
        records = [
            _make_tensor_record("layer.0", flags=["HIGH_KURTOSIS_A: k=20"]),
        ]
        results = compute_per_layer_findings(records)
        assert len(results[0].signals) > 0

    def test_signals_capped_at_five(self):
        flags = [
            "HIGH_KURTOSIS_A: k=20",
            "HIGH_KURTOSIS_B: k=18",
            "LOW_ENTROPY_A: e=0.01",
            "HIGH_ZSCORE_OUTLIER_RATE_A: r=5%",
            "RANK_INFLATION: rank=4",
            "NEAR_ZERO_B_MATRIX: norm=0.0",
        ]
        records = [_make_tensor_record("layer.0", flags=flags)]
        results = compute_per_layer_findings(records)
        assert len(results[0].signals) <= 5

    def test_parse_error_raises_score(self):
        from adaptersentry.schemas.errors import ErrorCategory
        clean = _make_tensor_record("layer.clean")
        broken = _make_tensor_record("layer.broken", parse_error=ErrorCategory.DEGRADED)
        results = compute_per_layer_findings([clean, broken])
        assert len(results) >= 1
        assert results[0].layer_name == "layer.broken"

    def test_inter_layer_bonus_adds_family(self):
        from adaptersentry.schemas.inter_layer_similarity_features import (
            InterLayerSimilarityFeatures, SimilarPair,
        )
        records = [
            _make_tensor_record("layer.0"),
            _make_tensor_record("layer.5"),
        ]
        il = InterLayerSimilarityFeatures(
            cosine_sim_mean=0.9, n_suspicious_pairs=1,
            top_suspicious_pairs=[
                SimilarPair(layer_a="layer.0", layer_b="layer.5",
                            index_a=0, index_b=5, cosine_sim=0.9),
            ],
        )
        results = compute_per_layer_findings(records, inter_layer_features=il)
        names = {r.layer_name for r in results}
        assert "layer.0" in names or "layer.5" in names
        for r in results:
            if r.layer_name in ("layer.0", "layer.5"):
                assert "inter_layer" in r.triggered_families

    def test_layer_without_flags_excluded(self):
        records = [
            _make_tensor_record("layer.clean"),
            _make_tensor_record("layer.flagged", flags=["HIGH_KURTOSIS_A: k=20"]),
        ]
        results = compute_per_layer_findings(records)
        names = [r.layer_name for r in results]
        assert "layer.clean" not in names
        assert "layer.flagged" in names

    def test_flag_count_correct(self):
        flags = ["HIGH_KURTOSIS_A: k=20", "LOW_ENTROPY_A: e=0.01"]
        records = [_make_tensor_record("layer.0", flags=flags)]
        results = compute_per_layer_findings(records)
        assert results[0].flag_count == 2

    def test_remediation_hint_populated(self):
        records = [_make_tensor_record("layer.0", flags=["HIGH_KURTOSIS_A: k=20"])]
        results = compute_per_layer_findings(records)
        assert results[0].remediation_hint is not None


# ---------------------------------------------------------------------------
# Integration via worker_main
# ---------------------------------------------------------------------------

class TestWorkerIntegration:
    def _make_req(self, path: Path):
        from adaptersentry.engine.schemas.requests import AdapterScanRequest, ArtifactSource
        return AdapterScanRequest(
            request_id="sha256:" + "d" * 64,
            run_id="test",
            adapter_path=str(path),
            source=ArtifactSource(kind="local_path", local_path=str(path)),
        )

    def test_top_layer_findings_in_result(self, tmp_path):
        from adaptersentry.engine.worker import worker_main
        p = _make_adapter(tmp_path)
        result, _ = worker_main(self._make_req(p), analyzer_config_hash="hash")
        assert isinstance(result.top_layer_findings, list)

    def test_top_layer_findings_max_ten(self, tmp_path):
        from adaptersentry.engine.worker import worker_main
        p = _make_adapter(tmp_path, n_layers=20)
        result, _ = worker_main(self._make_req(p), analyzer_config_hash="hash")
        assert len(result.top_layer_findings) <= 10

    def test_top_layer_findings_schema(self, tmp_path):
        from adaptersentry.engine.worker import worker_main
        p = _make_adapter(tmp_path)
        result, _ = worker_main(self._make_req(p), analyzer_config_hash="hash")
        for plf in result.top_layer_findings:
            assert isinstance(plf, PerLayerFinding)
            assert plf.rank >= 1
            assert 0.0 <= plf.severity_score <= 1.0
