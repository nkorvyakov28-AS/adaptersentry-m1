"""Tests for ensemble detector, Wasserstein detector, and cross-layer detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.detectors.wasserstein import (
    build_clean_reference,
    compute_wasserstein_distance,
    detect_wasserstein_anomalies,
)
from adaptersentry.detectors.cross_layer import (
    compute_cross_layer_consistency,
    detect_cross_layer_anomalies,
)
from adaptersentry.scoring.ensemble import DETECTOR_WEIGHTS, EnsembleDetector
from adaptersentry.analyzer import analyze


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LAYER = "base_model.model.layers.0.self_attn.q_proj"


def _make_adapter(
    tmp_path: Path,
    tensor_A: np.ndarray,
    tensor_B: np.ndarray,
    metadata: dict[str, str] | None = None,
) -> Path:
    path = tmp_path / "adapter.safetensors"
    save_file(
        {
            f"{_LAYER}.lora_A.weight": tensor_A.astype(np.float32),
            f"{_LAYER}.lora_B.weight": tensor_B.astype(np.float32),
        },
        str(path),
        metadata=metadata or {},
    )
    return path


def _clean_layer_report(n_flags: int = 0) -> dict:
    return {"flags": [f"FAKE_FLAG_{i}" for i in range(n_flags)]}


# ---------------------------------------------------------------------------
# Wasserstein tests
# ---------------------------------------------------------------------------


class TestWassersteinDistance:
    def test_same_tensor_zero_distance(self) -> None:
        t = np.random.default_rng(0).standard_normal((16, 64))
        assert compute_wasserstein_distance(t, t) == pytest.approx(0.0, abs=1e-6)

    def test_shifted_distribution_positive_distance(self) -> None:
        rng = np.random.default_rng(1)
        t_a = rng.standard_normal((16, 64))
        t_b = rng.standard_normal((16, 64)) + 5.0
        assert compute_wasserstein_distance(t_a, t_b) > 1.0

    def test_clean_vs_malicious_exceeds_threshold(self, tmp_path: Path) -> None:
        """W2 distance between clean Gaussian and high-kurtosis spike tensor > 0.15."""
        rng = np.random.default_rng(42)
        clean = rng.standard_normal((16, 512)).astype(np.float32)

        # Bernoulli-Gaussian spike mixture (same as synthetic generator)
        n = 16 * 512
        spike_mask = rng.random(n) < 0.01
        malicious_flat = np.where(
            spike_mask,
            rng.standard_normal(n) * 5.0,
            rng.standard_normal(n) * 0.001,
        )
        malicious = malicious_flat.reshape(16, 512).astype(np.float32)

        ref = build_clean_reference([clean])
        w2, flags = detect_wasserstein_anomalies("q_proj", malicious, ref)
        assert w2 > 0.15
        assert len(flags) == 1
        assert "HIGH_WASSERSTEIN_DISTANCE" in flags[0]

    def test_clean_tensor_no_flag(self) -> None:
        rng = np.random.default_rng(7)
        clean_tensors = [rng.standard_normal((16, 64)) for _ in range(3)]
        ref = build_clean_reference(clean_tensors)
        test = rng.standard_normal((16, 64))
        w2, flags = detect_wasserstein_anomalies("q_proj", test, ref)
        assert flags == []

    def test_size_limit_enforced(self) -> None:
        huge = np.empty(500_000_001, dtype=np.float32)
        with pytest.raises(ValueError, match="exceeding safety limit"):
            compute_wasserstein_distance(huge, np.zeros(10))

    def test_build_clean_reference_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            build_clean_reference([])

    def test_build_clean_reference_returns_required_keys(self) -> None:
        rng = np.random.default_rng(0)
        ref = build_clean_reference([rng.standard_normal((8, 32))])
        assert {"hist", "bin_edges", "bin_centres", "mean_w2"} <= set(ref)

    def test_hist_sums_to_one(self) -> None:
        rng = np.random.default_rng(0)
        ref = build_clean_reference([rng.standard_normal((8, 32))])
        assert ref["hist"].sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Cross-layer tests
# ---------------------------------------------------------------------------


class TestCrossLayerConsistency:
    def test_single_layer_returns_one(self) -> None:
        reports = {"layer0": _clean_layer_report(3)}
        assert compute_cross_layer_consistency(reports) == pytest.approx(1.0)

    def test_empty_returns_one(self) -> None:
        assert compute_cross_layer_consistency({}) == pytest.approx(1.0)

    def test_uniform_flags_high_consistency(self) -> None:
        """All layers have the same flag count → consistency near 1.0."""
        reports = {f"layer{i}": _clean_layer_report(2) for i in range(6)}
        c = compute_cross_layer_consistency(reports)
        assert c > 0.7

    def test_concentrated_flags_low_consistency(self) -> None:
        """One layer has all flags, rest have zero → consistency < 0.3."""
        reports = {f"layer{i}": _clean_layer_report(0) for i in range(9)}
        reports["layer_bad"] = _clean_layer_report(20)
        c = compute_cross_layer_consistency(reports)
        assert c < 0.3

    def test_consistency_in_range(self) -> None:
        rng = np.random.default_rng(0)
        reports = {
            f"l{i}": _clean_layer_report(int(rng.integers(0, 5)))
            for i in range(8)
        }
        c = compute_cross_layer_consistency(reports)
        assert 0.0 <= c <= 1.0


class TestDetectCrossLayerAnomalies:
    def test_clean_no_flags(self) -> None:
        reports = {f"layer{i}": _clean_layer_report(1) for i in range(8)}
        _, flags = detect_cross_layer_anomalies(reports)
        assert flags == []

    def test_concentrated_flags_cross_layer_concentration(self) -> None:
        reports = {f"layer{i}": _clean_layer_report(0) for i in range(9)}
        reports["layer_bad"] = _clean_layer_report(20)
        _, flags = detect_cross_layer_anomalies(reports)
        assert any("CROSS_LAYER_CONCENTRATION" in f for f in flags)

    def test_cluster_flag_fires(self) -> None:
        """Single layer holds >30% of flags in a 10-layer adapter."""
        reports = {f"layer{i}": _clean_layer_report(1) for i in range(9)}
        reports["layer_poison"] = _clean_layer_report(15)
        _, flags = detect_cross_layer_anomalies(reports)
        assert any("SUSPICIOUS_LAYER_CLUSTER" in f for f in flags)

    def test_returns_tuple(self) -> None:
        result = detect_cross_layer_anomalies({"l0": _clean_layer_report(1)})
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# EnsembleDetector tests
# ---------------------------------------------------------------------------


class TestEnsembleWeights:
    def test_weights_sum_to_one(self) -> None:
        detector = EnsembleDetector()
        total = sum(detector.weights.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_raw_weights_sum_close_to_one_after_normalisation(self) -> None:
        # DETECTOR_WEIGHTS (module-level constant) is already normalised
        assert sum(DETECTOR_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)

    def test_custom_weights_renormalised(self) -> None:
        detector = EnsembleDetector(weights={"kurtosis_score": 2.0, "energy_concentration": 2.0})
        assert sum(detector.weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            EnsembleDetector(weights={"kurtosis_score": -1.0})

    def test_all_zero_weights_raises(self) -> None:
        zero_weights = {k: 0.0 for k in DETECTOR_WEIGHTS}
        with pytest.raises(ValueError, match="all be zero"):
            EnsembleDetector(weights=zero_weights)

    def test_weights_property_is_copy(self) -> None:
        d = EnsembleDetector()
        w = d.weights
        w["kurtosis_score"] = 999.0
        assert d.weights["kurtosis_score"] != 999.0


class TestEnsembleScore:
    def _clean_reports(self, n: int = 4) -> dict:
        rng = np.random.default_rng(0)
        return {
            f"layer{i}": {
                "flags": [],
                "kurtosis_A": float(abs(rng.standard_normal())),
                "kurtosis_B": float(abs(rng.standard_normal())),
                "energy_concentration": 0.1,
                "entropy_A": 0.75,
                "entropy_B": 0.72,
                "zscore_outlier_rate_A": 0.001,
                "zscore_outlier_rate_B": 0.001,
                "isolation_score_A": 0.05,
            }
            for i in range(n)
        }

    def _malicious_reports(self) -> dict:
        return {
            "layer0": {
                "flags": ["HIGH_KURTOSIS_A: 262.9 > 10.0"],
                "kurtosis_A": 262.9,
                "kurtosis_B": 0.3,
                "energy_concentration": 0.98,
                "entropy_A": 0.01,
                "entropy_B": 0.70,
                "zscore_outlier_rate_A": 0.08,
                "zscore_outlier_rate_B": 0.002,
                "isolation_score_A": -0.30,
            }
        }

    def test_score_in_range(self) -> None:
        d = EnsembleDetector()
        s = d.score(self._clean_reports())
        assert 0.0 <= s <= 100.0

    def test_malicious_scores_higher_than_clean(self) -> None:
        d = EnsembleDetector()
        clean_score = d.score(self._clean_reports())
        mal_score = d.score(self._malicious_reports())
        assert mal_score > clean_score

    def test_empty_reports_returns_low_score(self) -> None:
        d = EnsembleDetector()
        assert d.score({}) < 50.0

    def test_predict_returns_valid_level(self) -> None:
        d = EnsembleDetector()
        level = d.predict(self._clean_reports())
        assert level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_predict_malicious_not_low(self) -> None:
        d = EnsembleDetector()
        level = d.predict(self._malicious_reports())
        assert level != "LOW"


class TestMajorityVote:
    def test_all_three_anomalous_returns_true(self) -> None:
        d = EnsembleDetector()
        assert d.majority_vote(0.97, 15.0, -0.25) is True

    def test_two_anomalous_returns_true(self) -> None:
        d = EnsembleDetector()
        # spectral + stat fire; iso does not
        assert d.majority_vote(0.97, 15.0, 0.10) is True

    def test_one_anomalous_returns_false(self) -> None:
        d = EnsembleDetector()
        # only kurtosis fires
        assert d.majority_vote(0.50, 15.0, 0.05) is False

    def test_none_anomalous_returns_false(self) -> None:
        d = EnsembleDetector()
        assert d.majority_vote(0.10, 2.0, 0.05) is False

    def test_majority_vote_requires_two(self) -> None:
        """Single detector flag must NOT produce majority."""
        d = EnsembleDetector()
        # Only energy_concentration fires
        result = d.majority_vote(spectral_score=0.97, stat_score=1.0, iso_score=0.0)
        assert result is False


# ---------------------------------------------------------------------------
# Backward-compatibility test
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_existing_json_fields_still_present(self, tmp_path: Path) -> None:
        """Upgrading to ensemble must not drop any existing schema fields."""
        rng = np.random.default_rng(0)
        path = _make_adapter(
            tmp_path,
            rng.standard_normal((8, 64)),
            rng.standard_normal((64, 8)),
            metadata={"r": "8"},
        )
        report = analyze(path)
        required_fields = {
            "adapter_path", "timestamp", "overall_risk", "risk_level",
            "flags", "layers", "metadata", "summary",
        }
        assert required_fields <= set(report), (
            f"Missing fields: {required_fields - set(report)}"
        )

    def test_new_fields_present(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        path = _make_adapter(
            tmp_path,
            rng.standard_normal((8, 64)),
            rng.standard_normal((64, 8)),
        )
        report = analyze(path)
        assert "ensemble_score" in report
        assert "cross_layer_consistency" in report
        assert "wasserstein_distances" in report

    def test_ensemble_score_in_range(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(3)
        path = _make_adapter(
            tmp_path,
            rng.standard_normal((8, 64)),
            rng.standard_normal((64, 8)),
        )
        report = analyze(path)
        assert 0.0 <= report["ensemble_score"] <= 100.0

    def test_report_json_serialisable(self, tmp_path: Path) -> None:
        import json
        rng = np.random.default_rng(5)
        path = _make_adapter(
            tmp_path,
            rng.standard_normal((8, 64)),
            rng.standard_normal((64, 8)),
        )
        report = analyze(path)
        assert json.loads(json.dumps(report)) == report
