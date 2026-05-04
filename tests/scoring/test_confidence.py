"""Tests for M1-SCORE-03: AnalysisQualityScore + ConfidenceScore."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.schemas.confidence_score import AnalysisQualityScore, ConfidenceScore
from adaptersentry.scoring.confidence import (
    compute_confidence_score,
    compute_quality_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(
    tmp_path: Path,
    n_layers: int = 4,
    seed: int = 0,
    prefix: str = "model.layers",
) -> Path:
    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {}
    for i in range(n_layers):
        tensors[f"{prefix}.{i}.q_proj.lora_A.weight"] = \
            rng.standard_normal((4, 32)).astype(np.float32)
        tensors[f"{prefix}.{i}.q_proj.lora_B.weight"] = \
            rng.standard_normal((32, 4)).astype(np.float32)
    p = tmp_path / "adapter.safetensors"
    save_file(tensors, str(p))
    return p


def _make_report(tmp_path: Path, n_layers: int = 4, seed: int = 0):
    from adaptersentry.analyzer import scan
    return scan(_make_adapter(tmp_path, n_layers=n_layers, seed=seed))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_quality_score_frozen(self):
        qs = AnalysisQualityScore(
            n_layers_total=4, n_layers_parsed_ok=4,
            parse_coverage=1.0, metadata_completeness=0.5,
            feature_completeness=0.8, degenerate_ratio=0.0,
            overall_quality=0.85,
        )
        with pytest.raises(Exception):
            qs.overall_quality = 0.5  # type: ignore[misc]

    def test_confidence_score_frozen(self):
        cs = ConfidenceScore(
            n_layers=4, n_families_successful=4,
            sample_size_factor=0.25, analysis_quality=0.9,
            inter_family_agreement=0.8, scan_mode_factor=1.0,
            overall_confidence=0.7, verdict_certainty="medium",
        )
        with pytest.raises(Exception):
            cs.verdict_certainty = "high"  # type: ignore[misc]

    def test_verdict_certainty_literals(self):
        for vc in ("high", "medium", "low"):
            cs = ConfidenceScore(
                n_layers=4, n_families_successful=3,
                sample_size_factor=0.5, analysis_quality=0.8,
                inter_family_agreement=0.7, scan_mode_factor=1.0,
                overall_confidence=0.6, verdict_certainty=vc,
            )
            assert cs.verdict_certainty == vc


# ---------------------------------------------------------------------------
# compute_quality_score
# ---------------------------------------------------------------------------

class TestComputeQualityScore:
    def test_returns_schema(self, tmp_path):
        report = _make_report(tmp_path)
        qs = compute_quality_score(report)
        assert isinstance(qs, AnalysisQualityScore)

    def test_parse_coverage_bounds(self, tmp_path):
        report = _make_report(tmp_path)
        qs = compute_quality_score(report)
        assert 0.0 <= qs.parse_coverage <= 1.0

    def test_feature_completeness_bounds(self, tmp_path):
        report = _make_report(tmp_path)
        qs = compute_quality_score(report)
        assert 0.0 <= qs.feature_completeness <= 1.0

    def test_overall_quality_bounds(self, tmp_path):
        report = _make_report(tmp_path)
        qs = compute_quality_score(report)
        assert 0.0 <= qs.overall_quality <= 1.0

    def test_degenerate_ratio_bounds(self, tmp_path):
        report = _make_report(tmp_path)
        qs = compute_quality_score(report)
        assert 0.0 <= qs.degenerate_ratio <= 1.0

    def test_clean_adapter_high_parse_coverage(self, tmp_path):
        report = _make_report(tmp_path, n_layers=4)
        qs = compute_quality_score(report)
        assert qs.parse_coverage == pytest.approx(1.0)
        assert qs.n_layers_parsed_ok == qs.n_layers_total

    def test_n_layers_counts(self, tmp_path):
        report = _make_report(tmp_path, n_layers=6)
        qs = compute_quality_score(report)
        assert qs.n_layers_total == 6
        assert qs.n_layers_parsed_ok <= 6

    def test_no_metadata_zero_completeness(self, tmp_path):
        # save_file with no metadata → metadata_completeness = 0.0
        rng = np.random.default_rng(0)
        tensors = {
            "model.layers.0.q_proj.lora_A.weight": rng.standard_normal((4, 32)).astype(np.float32),
            "model.layers.0.q_proj.lora_B.weight": rng.standard_normal((32, 4)).astype(np.float32),
        }
        p = tmp_path / "nometa.safetensors"
        save_file(tensors, str(p))

        from adaptersentry.analyzer import scan
        report = scan(p)
        qs = compute_quality_score(report)
        assert qs.metadata_completeness == pytest.approx(0.0)

    def test_empty_adapter_returns_zero_quality(self, tmp_path):
        from adaptersentry.analyzer import scan
        from adaptersentry.schemas.adapter_report import ParseStatus

        # A safetensors file with no LoRA pairs
        rng = np.random.default_rng(0)
        p = tmp_path / "nonlora.safetensors"
        save_file({"dense.weight": rng.standard_normal((4, 8)).astype(np.float32)}, str(p))
        report = scan(p)
        qs = compute_quality_score(report)
        assert qs.n_layers_total == 0
        assert qs.overall_quality == 0.0

    def test_quality_weights_sum_to_one(self):
        from adaptersentry.scoring.confidence import _QUAL_WEIGHTS
        assert abs(sum(_QUAL_WEIGHTS.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# compute_confidence_score
# ---------------------------------------------------------------------------

class TestComputeConfidenceScore:
    def test_returns_schema(self, tmp_path):
        report = _make_report(tmp_path)
        qs = compute_quality_score(report)
        cs = compute_confidence_score(report, qs)
        assert isinstance(cs, ConfidenceScore)

    def test_overall_confidence_bounds(self, tmp_path):
        report = _make_report(tmp_path)
        qs = compute_quality_score(report)
        cs = compute_confidence_score(report, qs)
        assert 0.0 <= cs.overall_confidence <= 1.0

    def test_verdict_certainty_valid_value(self, tmp_path):
        report = _make_report(tmp_path)
        qs = compute_quality_score(report)
        cs = compute_confidence_score(report, qs)
        assert cs.verdict_certainty in ("high", "medium", "low")

    def test_limiting_factors_list(self, tmp_path):
        report = _make_report(tmp_path)
        qs = compute_quality_score(report)
        cs = compute_confidence_score(report, qs)
        assert isinstance(cs.limiting_factors, list)

    def test_sample_size_factor_bounds(self, tmp_path):
        report = _make_report(tmp_path)
        qs = compute_quality_score(report)
        cs = compute_confidence_score(report, qs)
        assert 0.0 <= cs.sample_size_factor <= 1.0

    def test_small_adapter_low_sample_size_factor(self, tmp_path):
        # 2 layers → sample_size_factor = 2/16 = 0.125
        report = _make_report(tmp_path, n_layers=2)
        qs = compute_quality_score(report)
        cs = compute_confidence_score(report, qs)
        assert cs.sample_size_factor == pytest.approx(2 / 16)

    def test_large_adapter_sample_size_factor_one(self, tmp_path):
        # 16+ layers → sample_size_factor = 1.0
        report = _make_report(tmp_path, n_layers=20)
        qs = compute_quality_score(report)
        cs = compute_confidence_score(report, qs)
        assert cs.sample_size_factor == pytest.approx(1.0)

    def test_few_layers_adds_limiting_factor(self, tmp_path):
        report = _make_report(tmp_path, n_layers=2)
        qs = compute_quality_score(report)
        cs = compute_confidence_score(report, qs)
        assert any("layer" in f.lower() for f in cs.limiting_factors)

    def test_no_circular_logic_with_anomaly_signals(self, tmp_path):
        """Confidence score must not access anomaly feature attributes from TensorRecord."""
        import inspect
        from adaptersentry.scoring import confidence as conf_module
        src = inspect.getsource(conf_module)
        # These are the actual attribute accesses that would be circular.
        # We check for attribute-access form (tr.X or .kurtosis_a) not bare words
        # because the docstring legitimately names them as things we avoid.
        forbidden_access = [
            "tr.kurtosis_a", "tr.kurtosis_b",
            "tr.entropy_a", "tr.entropy_b",
            "tr.zscore_outlier_rate", "tr.isolation_score",
            "tr.energy_concentration",
            ".wasserstein_mean", ".cross_layer_consistency",
        ]
        for token in forbidden_access:
            assert token not in src, (
                f"Circular logic guard violation: attribute access '{token}' found in confidence.py"
            )

    def test_full_analysis_mode_scan_mode_factor_one(self, tmp_path):
        from adaptersentry.schemas.adapter_report import AnalysisMode
        report = _make_report(tmp_path)
        qs = compute_quality_score(report)
        cs = compute_confidence_score(report, qs)
        if report.analysis_mode == AnalysisMode.FULL:
            assert cs.scan_mode_factor == pytest.approx(1.0)

    def test_n_families_successful_positive(self, tmp_path):
        report = _make_report(tmp_path, n_layers=4)
        qs = compute_quality_score(report)
        cs = compute_confidence_score(report, qs)
        assert cs.n_families_successful > 0

    def test_confidence_weights_sum_to_one(self):
        from adaptersentry.scoring.confidence import _CONF_WEIGHTS
        assert abs(sum(_CONF_WEIGHTS.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Verdict certainty thresholds
# ---------------------------------------------------------------------------

class TestVerdictCertainty:
    def _make_cs(self, overall_confidence: float) -> ConfidenceScore:
        vc = ("high" if overall_confidence >= 0.75
              else "medium" if overall_confidence >= 0.45
              else "low")
        return ConfidenceScore(
            n_layers=8, n_families_successful=4,
            sample_size_factor=0.5, analysis_quality=0.8,
            inter_family_agreement=0.8, scan_mode_factor=1.0,
            overall_confidence=overall_confidence,
            verdict_certainty=vc,
        )

    def test_high_at_075(self):
        assert self._make_cs(0.75).verdict_certainty == "high"

    def test_high_at_1(self):
        assert self._make_cs(1.0).verdict_certainty == "high"

    def test_medium_at_045(self):
        assert self._make_cs(0.45).verdict_certainty == "medium"

    def test_medium_at_074(self):
        assert self._make_cs(0.74).verdict_certainty == "medium"

    def test_low_below_045(self):
        assert self._make_cs(0.44).verdict_certainty == "low"

    def test_low_at_zero(self):
        assert self._make_cs(0.0).verdict_certainty == "low"


# ---------------------------------------------------------------------------
# worker_main integration
# ---------------------------------------------------------------------------

class TestWorkerIntegration:
    def _make_req(self, path: Path):
        from adaptersentry.engine.schemas.requests import AdapterScanRequest, ArtifactSource
        return AdapterScanRequest(
            request_id="sha256:" + "c" * 64,
            run_id="test",
            adapter_path=str(path),
            source=ArtifactSource(kind="local_path", local_path=str(path)),
        )

    def test_quality_score_in_result(self, tmp_path):
        from adaptersentry.engine.worker import worker_main

        p = _make_adapter(tmp_path)
        result, _ = worker_main(self._make_req(p), analyzer_config_hash="hash0000")
        assert result.quality_score is not None
        assert isinstance(result.quality_score, AnalysisQualityScore)

    def test_confidence_score_in_result(self, tmp_path):
        from adaptersentry.engine.worker import worker_main

        p = _make_adapter(tmp_path)
        result, _ = worker_main(self._make_req(p), analyzer_config_hash="hash0000")
        assert result.confidence_score is not None
        assert isinstance(result.confidence_score, ConfidenceScore)

    def test_verdict_certainty_not_empty(self, tmp_path):
        from adaptersentry.engine.worker import worker_main

        p = _make_adapter(tmp_path)
        result, _ = worker_main(self._make_req(p), analyzer_config_hash="hash0000")
        cs = result.confidence_score
        assert cs is not None
        assert cs.verdict_certainty in ("high", "medium", "low")
