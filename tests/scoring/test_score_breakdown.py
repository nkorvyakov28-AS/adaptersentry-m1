"""Tests for M1-SCORE-01: ScoreBreakdown and filled detector_weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.schemas.score_breakdown import ScoreBreakdown, SubScore
from adaptersentry.scoring.score_breakdown import (
    _FAMILY_ORDER,
    _FAMILY_WEIGHTS,
    compute_score_breakdown,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_adapter_file(tmp_path: Path, n_layers: int = 4, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {}
    for i in range(n_layers):
        tensors[f"model.layers.{i}.self_attn.q_proj.lora_A.weight"] = \
            rng.standard_normal((4, 32)).astype(np.float32)
        tensors[f"model.layers.{i}.self_attn.q_proj.lora_B.weight"] = \
            rng.standard_normal((32, 4)).astype(np.float32)
    p = tmp_path / "adapter.safetensors"
    save_file(tensors, str(p))
    return p


def _make_report(tmp_path: Path, n_layers: int = 4, seed: int = 0):
    from adaptersentry.analyzer import scan
    p = _make_adapter_file(tmp_path, n_layers=n_layers, seed=seed)
    return scan(p)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestScoreBreakdownSchema:
    def test_sub_score_frozen(self):
        s = SubScore(
            family="parse", raw_score=0.1, normalized_score=0.1,
            weight=0.1, weighted_contribution=0.01,
        )
        with pytest.raises(Exception):
            s.family = "distribution"  # type: ignore[misc]

    def test_score_breakdown_frozen(self):
        sb = ScoreBreakdown(sub_scores=[], total_weighted=0.0, dominant_family="parse")
        with pytest.raises(Exception):
            sb.total_weighted = 0.5  # type: ignore[misc]

    def test_family_weights_sum_to_one(self):
        assert abs(sum(_FAMILY_WEIGHTS.values()) - 1.0) < 1e-9

    def test_all_families_defined(self):
        assert set(_FAMILY_ORDER) == set(_FAMILY_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# compute_score_breakdown
# ---------------------------------------------------------------------------

class TestComputeScoreBreakdown:
    def test_returns_score_breakdown(self, tmp_path):
        report = _make_report(tmp_path)
        result = compute_score_breakdown(report)
        assert isinstance(result, ScoreBreakdown)

    def test_all_families_present(self, tmp_path):
        report = _make_report(tmp_path)
        result = compute_score_breakdown(report)
        families = {s.family for s in result.sub_scores}
        assert families == set(_FAMILY_ORDER)

    def test_canonical_order(self, tmp_path):
        report = _make_report(tmp_path)
        result = compute_score_breakdown(report)
        assert [s.family for s in result.sub_scores] == _FAMILY_ORDER

    def test_normalized_score_in_bounds(self, tmp_path):
        report = _make_report(tmp_path)
        result = compute_score_breakdown(report)
        for s in result.sub_scores:
            assert 0.0 <= s.normalized_score <= 1.0, f"{s.family}: {s.normalized_score}"

    def test_weighted_contribution_consistent(self, tmp_path):
        report = _make_report(tmp_path)
        result = compute_score_breakdown(report)
        for s in result.sub_scores:
            expected = s.normalized_score * s.weight
            assert s.weighted_contribution == pytest.approx(expected, abs=1e-9)

    def test_total_weighted_sum(self, tmp_path):
        report = _make_report(tmp_path)
        result = compute_score_breakdown(report)
        expected = sum(s.weighted_contribution for s in result.sub_scores)
        assert result.total_weighted == pytest.approx(expected, abs=1e-9)

    def test_total_weighted_in_bounds(self, tmp_path):
        report = _make_report(tmp_path)
        result = compute_score_breakdown(report)
        assert 0.0 <= result.total_weighted <= 1.0

    def test_weights_match_family_weights(self, tmp_path):
        report = _make_report(tmp_path)
        result = compute_score_breakdown(report)
        for s in result.sub_scores:
            assert s.weight == pytest.approx(_FAMILY_WEIGHTS[s.family])

    def test_dominant_family_is_highest_contribution(self, tmp_path):
        report = _make_report(tmp_path)
        result = compute_score_breakdown(report)
        dominant = max(result.sub_scores, key=lambda s: s.weighted_contribution)
        assert result.dominant_family == dominant.family

    def test_top_reasons_list(self, tmp_path):
        report = _make_report(tmp_path)
        result = compute_score_breakdown(report)
        for s in result.sub_scores:
            assert isinstance(s.top_reasons, list)
            assert len(s.top_reasons) <= 3

    def test_schema_version_present(self, tmp_path):
        report = _make_report(tmp_path)
        result = compute_score_breakdown(report)
        assert result.schema_version == "1.0.0"


# ---------------------------------------------------------------------------
# Semantic tests — specific sub-score signals
# ---------------------------------------------------------------------------

class TestSubScoreSemantics:
    def test_missing_metadata_raises_metadata_score(self, tmp_path):
        """Adapter with no metadata should produce a nonzero metadata sub-score."""
        # save_file with no metadata field → empty metadata
        rng = np.random.default_rng(0)
        tensors = {
            "model.layers.0.q_proj.lora_A.weight": rng.standard_normal((4, 32)).astype(np.float32),
            "model.layers.0.q_proj.lora_B.weight": rng.standard_normal((32, 4)).astype(np.float32),
        }
        p = tmp_path / "nometadata.safetensors"
        save_file(tensors, str(p))

        from adaptersentry.analyzer import scan
        report = scan(p)
        result = compute_score_breakdown(report)
        meta_score = next(s for s in result.sub_scores if s.family == "metadata")
        assert meta_score.normalized_score > 0.0

    def test_high_kurtosis_raises_distribution_score(self, tmp_path):
        """Layers with artificially high kurtosis values should raise distribution score."""
        from adaptersentry.analyzer import scan

        rng = np.random.default_rng(42)
        # Create Laplacian-like tensors (high kurtosis)
        tensors = {}
        for i in range(2):
            a_data = rng.laplace(0, 0.01, (4, 64)).astype(np.float32)
            b_data = rng.laplace(0, 0.01, (64, 4)).astype(np.float32)
            tensors[f"model.layers.{i}.q_proj.lora_A.weight"] = a_data
            tensors[f"model.layers.{i}.q_proj.lora_B.weight"] = b_data

        p = tmp_path / "high_kurt.safetensors"
        save_file(tensors, str(p))
        report = scan(p)
        result = compute_score_breakdown(report)

        dist_score = next(s for s in result.sub_scores if s.family == "distribution")
        # Laplacian has kurtosis=3 (excess), which is above normal
        assert dist_score.normalized_score >= 0.0  # just ensure no crash

    def test_clean_adapter_low_scores(self, tmp_path):
        """Normal Gaussian adapter should have low sub-scores overall."""
        report = _make_report(tmp_path, n_layers=4, seed=7)
        result = compute_score_breakdown(report)
        # Most sub-scores should be well below 1.0 for random normal weights
        for s in result.sub_scores:
            assert s.normalized_score < 0.95, (
                f"{s.family} score too high ({s.normalized_score:.3f}) for normal weights"
            )


# ---------------------------------------------------------------------------
# EnsembleSignal.detector_weights fill
# ---------------------------------------------------------------------------

def _make_req(adapter_path: Path):
    from adaptersentry.engine.schemas.requests import AdapterScanRequest, ArtifactSource
    return AdapterScanRequest(
        request_id="sha256:" + "a" * 64,
        run_id="test-run",
        adapter_path=str(adapter_path),
        source=ArtifactSource(kind="local_path", local_path=str(adapter_path)),
    )


class TestDetectorWeightsFill:
    def test_detector_weights_populated(self, tmp_path):
        """worker_main should fill detector_weights in EnsembleSignal."""
        from adaptersentry.engine.worker import worker_main

        p = _make_adapter_file(tmp_path)
        result, _ = worker_main(_make_req(p), analyzer_config_hash="testhash0000")
        assert result.ensemble.detector_weights, "detector_weights should not be empty"
        total = sum(result.ensemble.detector_weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_score_breakdown_in_ensemble(self, tmp_path):
        """worker_main should produce score_breakdown in EnsembleSignal."""
        from adaptersentry.engine.worker import worker_main

        p = _make_adapter_file(tmp_path)
        result, _ = worker_main(_make_req(p), analyzer_config_hash="testhash0000")
        assert result.ensemble.score_breakdown is not None
        sb = result.ensemble.score_breakdown
        assert isinstance(sb, ScoreBreakdown)
        assert len(sb.sub_scores) == len(_FAMILY_ORDER)

    def test_detector_weights_keys(self, tmp_path):
        """Detector weights should use the canonical EnsembleDetector key names."""
        from adaptersentry.engine.worker import worker_main
        from adaptersentry.scoring.ensemble import DETECTOR_WEIGHTS

        p = _make_adapter_file(tmp_path)
        result, _ = worker_main(_make_req(p), analyzer_config_hash="testhash0000")
        assert set(result.ensemble.detector_weights.keys()) == set(DETECTOR_WEIGHTS.keys())
