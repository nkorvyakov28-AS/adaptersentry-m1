"""Tests for M1-ANAL-01: extended DistributionFeatures + compute_tensor_stats."""

from __future__ import annotations

import numpy as np
import pytest

from adaptersentry.features.distribution import compute_distribution_features
from adaptersentry.features.tensor_stats import compute_tensor_stats
from adaptersentry.schemas.distribution_features import DistributionFeatures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pair(rank: int = 4, out: int = 32, in_: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((rank, in_)).astype(np.float32)
    B = rng.standard_normal((out, rank)).astype(np.float32)
    return A, B


# ---------------------------------------------------------------------------
# compute_tensor_stats — new fields
# ---------------------------------------------------------------------------

class TestComputeTensorStats:
    def test_returns_new_keys(self):
        A, _ = _make_pair()
        stats = compute_tensor_stats(A)
        for key in ("median", "p01", "p99", "iqr", "zero_ratio"):
            assert key in stats, f"missing key: {key}"

    def test_iqr_nonnegative(self):
        A, _ = _make_pair()
        stats = compute_tensor_stats(A)
        assert stats["iqr"] >= 0.0

    def test_zero_ratio_bounds(self):
        A, _ = _make_pair()
        stats = compute_tensor_stats(A)
        assert 0.0 <= stats["zero_ratio"] <= 1.0

    def test_zero_ratio_all_zeros(self):
        z = np.zeros((8, 16), dtype=np.float32)
        stats = compute_tensor_stats(z)
        assert stats["zero_ratio"] == pytest.approx(1.0)

    def test_zero_ratio_no_zeros(self):
        A, _ = _make_pair()
        # standard normal — extremely unlikely to have exact zeros
        stats = compute_tensor_stats(A)
        assert stats["zero_ratio"] < 0.01

    def test_p01_le_median_le_p99(self):
        A, _ = _make_pair(seed=99)
        stats = compute_tensor_stats(A)
        assert stats["p01"] <= stats["median"] <= stats["p99"]

    def test_fast_mode_returns_same_keys(self):
        rng = np.random.default_rng(7)
        large = rng.standard_normal(200_000).astype(np.float32)
        stats = compute_tensor_stats(large, fast=True)
        for key in ("median", "p01", "p99", "iqr", "zero_ratio"):
            assert key in stats

    def test_legacy_keys_still_present(self):
        A, _ = _make_pair()
        stats = compute_tensor_stats(A)
        for key in ("mean", "std", "kurtosis", "skewness"):
            assert key in stats


# ---------------------------------------------------------------------------
# DistributionFeatures schema — new fields
# ---------------------------------------------------------------------------

class TestDistributionFeaturesSchema:
    def test_new_fields_exist(self):
        df = DistributionFeatures(
            delta_kurtosis=1.0, delta_skewness=0.5,
            delta_mean=0.01, delta_std=0.1,
        )
        for field in ("delta_median", "delta_p01", "delta_p99",
                      "delta_iqr", "delta_zero_ratio", "delta_entropy"):
            assert hasattr(df, field)

    def test_new_fields_default_zero(self):
        df = DistributionFeatures(
            delta_kurtosis=0.0, delta_skewness=0.0,
            delta_mean=0.0, delta_std=0.0,
        )
        assert df.delta_median == 0.0
        assert df.delta_p01 == 0.0
        assert df.delta_p99 == 0.0
        assert df.delta_iqr == 0.0
        assert df.delta_zero_ratio == 0.0
        assert df.delta_entropy == 0.0

    def test_backwards_compat_missing_new_fields(self):
        # Simulate old serialized JSON without new fields
        old_dict = {
            "delta_kurtosis": 2.0,
            "delta_skewness": -0.3,
            "delta_mean": 0.001,
            "delta_std": 0.05,
            "computed_on_sample": False,
        }
        df = DistributionFeatures(**old_dict)
        assert df.delta_median == 0.0
        assert df.delta_entropy == 0.0


# ---------------------------------------------------------------------------
# compute_distribution_features — new fields computed
# ---------------------------------------------------------------------------

class TestComputeDistributionFeaturesExtended:
    def test_returns_distribution_features(self):
        A, B = _make_pair()
        result = compute_distribution_features(A, B)
        assert isinstance(result, DistributionFeatures)

    def test_new_fields_populated(self):
        A, B = _make_pair(rank=4, out=16, in_=32)
        df = compute_distribution_features(A, B)
        assert df is not None
        # All new fields should be non-default for a normal random pair
        assert df.delta_p99 > 0.0 or df.delta_p01 < 0.0  # normal distribution
        assert df.delta_iqr >= 0.0
        assert 0.0 <= df.delta_zero_ratio <= 1.0
        assert 0.0 <= df.delta_entropy <= 1.0

    def test_p01_le_median_le_p99(self):
        A, B = _make_pair(seed=42)
        df = compute_distribution_features(A, B)
        assert df is not None
        assert df.delta_p01 <= df.delta_median <= df.delta_p99

    def test_zero_ratio_near_zero_for_normal_delta(self):
        A, B = _make_pair(rank=8, out=64, in_=128)
        df = compute_distribution_features(A, B)
        assert df is not None
        # Standard normal ΔW has very few exact zeros
        assert df.delta_zero_ratio < 0.01

    def test_zero_b_returns_zero_fields(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((4, 32)).astype(np.float32)
        B = np.zeros((16, 4), dtype=np.float32)
        df = compute_distribution_features(A, B)
        assert df is not None
        assert df.delta_kurtosis == 0.0
        assert df.delta_median == 0.0
        assert df.delta_entropy == 0.0
        assert df.delta_zero_ratio == 0.0  # zero-B path returns early

    def test_entropy_bounds(self):
        A, B = _make_pair(rank=4, out=32, in_=64, seed=5)
        df = compute_distribution_features(A, B)
        assert df is not None
        assert 0.0 <= df.delta_entropy <= 1.0

    def test_constant_delta_entropy_is_zero(self):
        # B = 1s, A = 0s → ΔW = 0 everywhere → zero-B guard fires first
        A = np.zeros((4, 32), dtype=np.float32)
        B = np.ones((16, 4), dtype=np.float32)
        df = compute_distribution_features(A, B)
        # ΔW = B @ A = 0 matrix → delta_entropy = 0 via constant path
        assert df is not None
        assert df.delta_entropy == 0.0

    def test_shape_mismatch_returns_none(self):
        A = np.ones((4, 32), dtype=np.float32)
        B = np.ones((16, 8), dtype=np.float32)  # rank mismatch
        assert compute_distribution_features(A, B) is None


# ---------------------------------------------------------------------------
# Memory guard — proxy path for out×in > _MAX_DELTA_NUMEL_FULL (4M)
# ---------------------------------------------------------------------------

class TestLargeDeltaProxyPath:
    """Verify that full mode uses proxy (not ΔW materialisation) for large layers.

    Uses out=2048, in=2560 → delta_numel=5,242,880 > 4M threshold.
    Previously this would materialise a 20MB ΔW; now it uses lora_A rows.
    """

    def _large_pair(self, rank: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        # out=2048, in=2560 → 5.24M elements — above 4M full threshold
        A = rng.standard_normal((rank, 2560)).astype(np.float32)
        B = rng.standard_normal((2048, rank)).astype(np.float32)
        return A, B

    def test_full_mode_sets_computed_on_sample(self):
        A, B = self._large_pair()
        df = compute_distribution_features(A, B, fast=False)
        assert df is not None
        assert df.computed_on_sample is True

    def test_fast_mode_also_proxy(self):
        A, B = self._large_pair()
        df = compute_distribution_features(A, B, fast=True)
        assert df is not None
        assert df.computed_on_sample is True

    def test_proxy_result_has_finite_stats(self):
        A, B = self._large_pair(rank=32, seed=7)
        df = compute_distribution_features(A, B, fast=False)
        assert df is not None
        assert np.isfinite(df.delta_kurtosis)
        assert np.isfinite(df.delta_skewness)
        assert np.isfinite(df.delta_mean)
        assert np.isfinite(df.delta_std)

    def test_below_threshold_is_exact_in_full_mode(self):
        rng = np.random.default_rng(0)
        # out=64, in=64 → 4096 elements — well below 4M threshold
        A = rng.standard_normal((8, 64)).astype(np.float32)
        B = rng.standard_normal((64, 8)).astype(np.float32)
        df = compute_distribution_features(A, B, fast=False)
        assert df is not None
        assert df.computed_on_sample is False


# ---------------------------------------------------------------------------
# feature_extractor integration — new keys in raw_features
# ---------------------------------------------------------------------------

class TestFeatureExtractorDistribution:
    def test_distribution_raw_features_has_new_delta_keys(self):
        from adaptersentry.engine.feature_extractor import FeatureExtractor

        A, B = _make_pair(rank=4, out=32, in_=64)
        extractor = FeatureExtractor()
        _, families, _ = extractor.extract_layer("test.layer", A, B)
        dist = next(f for f in families if f.family == "distribution")
        assert dist.status == "ok"
        for key in ("median_delta", "p01_delta", "p99_delta",
                    "iqr_delta", "zero_ratio_delta", "entropy_delta"):
            assert key in dist.raw_features, f"missing raw_feature key: {key}"

    def test_distribution_raw_features_has_new_ab_keys(self):
        from adaptersentry.engine.feature_extractor import FeatureExtractor

        A, B = _make_pair(rank=4, out=32, in_=64)
        extractor = FeatureExtractor()
        _, families, _ = extractor.extract_layer("test.layer", A, B)
        dist = next(f for f in families if f.family == "distribution")
        assert dist.status == "ok"
        for key in ("median_A", "p01_A", "p99_A", "iqr_A", "zero_ratio_A",
                    "median_B", "p01_B", "p99_B", "iqr_B", "zero_ratio_B",
                    "skewness_A", "skewness_B"):
            assert key in dist.raw_features, f"missing raw_feature key: {key}"

    def test_families_from_record_propagates_new_fields(self):
        from adaptersentry.engine.feature_extractor import FeatureExtractor

        A, B = _make_pair(rank=4, out=32, in_=64)
        extractor = FeatureExtractor()
        record, _, _ = extractor.extract_layer("test.layer", A, B)
        families = FeatureExtractor.families_from_record(record)
        dist = next(f for f in families if f.family == "distribution")
        for key in ("median_delta", "p01_delta", "p99_delta",
                    "iqr_delta", "zero_ratio_delta", "entropy_delta"):
            assert key in dist.raw_features, f"families_from_record missing: {key}"
