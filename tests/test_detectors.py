"""Tests for adaptersentry.detectors — entropy and outlier modules."""

from __future__ import annotations

import numpy as np
import pytest

from adaptersentry.detectors.entropy import compute_entropy, detect_entropy_anomalies
from adaptersentry.detectors.outlier import (
    detect_outlier_anomalies,
    isolation_forest_score,
    zscore_outlier_rate,
)


# ---------------------------------------------------------------------------
# compute_entropy
# ---------------------------------------------------------------------------


class TestComputeEntropy:
    def test_returns_float_in_0_1(self) -> None:
        rng = np.random.default_rng(0)
        t = rng.standard_normal((16, 64))
        e = compute_entropy(t)
        assert 0.0 <= e <= 1.0

    def test_constant_tensor_zero_entropy(self) -> None:
        assert compute_entropy(np.ones((8, 64))) == 0.0

    def test_constant_scalar_tensor_zero_entropy(self) -> None:
        assert compute_entropy(np.array([5.0])) == 0.0

    def test_normal_distribution_moderate_entropy(self) -> None:
        rng = np.random.default_rng(42)
        t = rng.standard_normal(10_000)
        e = compute_entropy(t)
        # Normal distribution should produce moderate-to-high entropy
        assert 0.5 < e < 1.0

    def test_uniform_distribution_high_entropy(self) -> None:
        rng = np.random.default_rng(1)
        t = rng.uniform(-1, 1, size=10_000)
        e = compute_entropy(t)
        # Uniform is maximum entropy — should be close to 1.0
        assert e > 0.95

    def test_near_zero_tensor_low_entropy(self) -> None:
        # Weights clustered tightly near zero → very low entropy
        t = np.zeros((64, 8), dtype=np.float32)
        t[0, 0] = 1e-8  # break strict constant to avoid 0.0 path
        e = compute_entropy(t)
        assert e < 0.1

    def test_empty_tensor_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_entropy(np.array([]))

    def test_accepts_any_shape(self) -> None:
        for shape in [(10,), (4, 8), (2, 4, 8)]:
            e = compute_entropy(np.random.default_rng(0).standard_normal(shape))
            assert 0.0 <= e <= 1.0

    def test_custom_bin_count(self) -> None:
        t = np.random.default_rng(7).standard_normal(1_000)
        e64 = compute_entropy(t, n_bins=64)
        e256 = compute_entropy(t, n_bins=256)
        # Both should be in valid range; finer bins → slightly higher entropy
        assert 0.0 <= e64 <= 1.0
        assert 0.0 <= e256 <= 1.0


# ---------------------------------------------------------------------------
# detect_entropy_anomalies
# ---------------------------------------------------------------------------


class TestDetectEntropyAnomalies:
    def test_low_entropy_flagged(self) -> None:
        flags = detect_entropy_anomalies(0.05, "model.q_proj", "A")
        assert len(flags) == 1
        assert "LOW_ENTROPY_A" in flags[0]

    def test_high_entropy_flagged(self) -> None:
        flags = detect_entropy_anomalies(0.995, "model.q_proj", "B")
        assert len(flags) == 1
        assert "HIGH_ENTROPY_B" in flags[0]

    def test_normal_entropy_no_flag(self) -> None:
        flags = detect_entropy_anomalies(0.75, "model.q_proj", "A")
        assert flags == []

    def test_flag_includes_entropy_value(self) -> None:
        flags = detect_entropy_anomalies(0.05, "model.q_proj", "A")
        assert "0.0500" in flags[0]

    def test_flag_includes_layer_name(self) -> None:
        flags = detect_entropy_anomalies(0.03, "base_model.lm_head", "A")
        assert "base_model.lm_head" in flags[0]

    def test_matrix_label_b_in_flag(self) -> None:
        flags = detect_entropy_anomalies(0.02, "model.v_proj", "B")
        assert "LOW_ENTROPY_B" in flags[0]

    def test_custom_thresholds(self) -> None:
        # With wide thresholds nothing flags
        flags = detect_entropy_anomalies(0.5, "layer", "A", low_threshold=0.0, high_threshold=1.0)
        assert flags == []

    def test_exactly_at_low_threshold_no_flag(self) -> None:
        # Boundary: exactly at low_threshold → not flagged (< threshold required)
        flags = detect_entropy_anomalies(0.1, "layer", "A", low_threshold=0.1)
        assert flags == []

    def test_exactly_at_high_threshold_no_flag(self) -> None:
        flags = detect_entropy_anomalies(0.99, "layer", "A", high_threshold=0.99)
        assert flags == []


# ---------------------------------------------------------------------------
# zscore_outlier_rate
# ---------------------------------------------------------------------------


class TestZscoreOutlierRate:
    def test_returns_required_keys(self) -> None:
        result = zscore_outlier_rate(np.random.default_rng(0).standard_normal((8, 64)))
        assert "outlier_rate" in result
        assert "threshold_sigma" in result

    def test_normal_distribution_low_rate(self) -> None:
        rng = np.random.default_rng(42)
        t = rng.standard_normal(50_000)
        result = zscore_outlier_rate(t)
        # Gaussian: ~0.27% beyond 3σ — should be well under 1%
        assert result["outlier_rate"] < 0.01

    def test_sparse_outlier_tensor_high_rate(self) -> None:
        # A few extreme values in otherwise zero tensor → high outlier rate
        t = np.zeros(1_000, dtype=np.float32)
        t[:50] = 1000.0  # 5% of values at extreme outliers
        result = zscore_outlier_rate(t)
        assert result["outlier_rate"] > 0.02

    def test_constant_tensor_zero_rate(self) -> None:
        result = zscore_outlier_rate(np.ones((8, 8)))
        assert result["outlier_rate"] == 0.0

    def test_threshold_preserved_in_result(self) -> None:
        result = zscore_outlier_rate(np.zeros(10), threshold=4.0)
        assert result["threshold_sigma"] == 4.0

    def test_empty_tensor_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            zscore_outlier_rate(np.array([]))

    def test_accepts_any_shape(self) -> None:
        for shape in [(10,), (4, 8), (2, 4, 8)]:
            result = zscore_outlier_rate(np.random.default_rng(0).standard_normal(shape))
            assert 0.0 <= result["outlier_rate"] <= 1.0


# ---------------------------------------------------------------------------
# isolation_forest_score
# ---------------------------------------------------------------------------


class TestIsolationForestScore:
    def test_returns_required_keys(self) -> None:
        t = np.random.default_rng(0).standard_normal((16, 64))
        result = isolation_forest_score(t)
        assert "mean_score" in result
        assert "anomalous_fraction" in result

    def test_anomalous_fraction_in_0_1(self) -> None:
        t = np.random.default_rng(1).standard_normal((8, 32))
        result = isolation_forest_score(t)
        assert 0.0 <= result["anomalous_fraction"] <= 1.0

    def test_reproducible_with_random_state(self) -> None:
        t = np.random.default_rng(5).standard_normal((16, 32))
        r1 = isolation_forest_score(t, random_state=0)
        r2 = isolation_forest_score(t, random_state=0)
        assert r1["mean_score"] == pytest.approx(r2["mean_score"])

    def test_subsampling_for_large_tensor(self) -> None:
        # >2000 weights → triggers subsampling; should still return valid keys
        t = np.random.default_rng(3).standard_normal((100, 100))  # 10_000 weights
        result = isolation_forest_score(t, max_samples=500)
        assert "mean_score" in result

    def test_minimum_size_2_enforced(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            isolation_forest_score(np.array([1.0]))


# ---------------------------------------------------------------------------
# detect_outlier_anomalies
# ---------------------------------------------------------------------------


class TestDetectOutlierAnomalies:
    def _normal_tensor(self) -> np.ndarray:
        return np.random.default_rng(42).standard_normal((16, 64))

    def _sparse_outlier_tensor(self) -> np.ndarray:
        # 100 elements, 3 at extreme values → 3% outlier rate, above the 2% threshold
        t = np.random.default_rng(0).standard_normal(100)
        t[:3] = 100.0
        return t

    def test_returns_three_item_tuple(self) -> None:
        zs, iso, flags = detect_outlier_anomalies(self._normal_tensor(), "l", "A")
        assert isinstance(zs, dict)
        assert isinstance(iso, dict)
        assert isinstance(flags, list)

    def test_clean_tensor_no_flags(self) -> None:
        _, _, flags = detect_outlier_anomalies(self._normal_tensor(), "model.q_proj", "A")
        assert flags == []

    def test_sparse_tensor_flags_high_zscore(self) -> None:
        _, _, flags = detect_outlier_anomalies(
            self._sparse_outlier_tensor(), "model.q_proj", "A"
        )
        assert any("HIGH_ZSCORE_OUTLIER_RATE_A" in f for f in flags)

    def test_matrix_label_in_flag(self) -> None:
        _, _, flags = detect_outlier_anomalies(
            self._sparse_outlier_tensor(), "model.q_proj", "B",
            run_isolation_forest=False,
        )
        assert all("_B" in f for f in flags)

    def test_no_isolation_forest_when_disabled(self) -> None:
        _, iso, _ = detect_outlier_anomalies(
            self._normal_tensor(), "model.q_proj", "A", run_isolation_forest=False
        )
        assert iso == {}

    def test_isolation_forest_result_present_when_enabled(self) -> None:
        _, iso, _ = detect_outlier_anomalies(
            self._normal_tensor(), "model.q_proj", "A", run_isolation_forest=True
        )
        assert "mean_score" in iso
