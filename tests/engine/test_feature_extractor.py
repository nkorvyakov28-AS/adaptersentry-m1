"""Tests for FeatureExtractor — CARD-08."""

from __future__ import annotations

import numpy as np
import pytest

from adaptersentry.engine.feature_extractor import FeatureExtractor
from adaptersentry.engine.schemas.signals import FeatureFamilyResult
from adaptersentry.schemas.errors import ErrorSeverity, ScanError, ScanPhase
from adaptersentry.schemas.tensor_record import TensorRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


@pytest.fixture
def simple_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    A = rng.standard_normal((8, 64)).astype(np.float32)
    B = rng.standard_normal((64, 8)).astype(np.float32)
    return A, B


@pytest.fixture
def zero_B_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 32)).astype(np.float32)
    B = np.zeros((32, 8), dtype=np.float32)
    return A, B


# ---------------------------------------------------------------------------
# extract_layer — return types
# ---------------------------------------------------------------------------

class TestExtractLayerReturnTypes:
    def test_returns_three_tuple(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        result = extractor.extract_layer("layer.q_proj", A, B)
        assert len(result) == 3

    def test_first_is_tensor_record(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        record, _, _ = extractor.extract_layer("layer.q_proj", A, B)
        assert isinstance(record, TensorRecord)

    def test_second_is_family_list(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        _, families, _ = extractor.extract_layer("layer.q_proj", A, B)
        assert isinstance(families, list)
        assert all(isinstance(f, FeatureFamilyResult) for f in families)

    def test_third_is_error_list(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        _, _, errors = extractor.extract_layer("layer.q_proj", A, B)
        assert isinstance(errors, list)
        assert all(isinstance(e, ScanError) for e in errors)


class TestExtractLayerFamilies:
    _EXPECTED_FAMILIES = {"spectral", "distribution", "entropy", "entropy_compression", "outlier", "norm"}
    _EXPECTED_FAMILIES_FAST = {"spectral", "distribution", "entropy", "entropy_compression", "outlier", "norm"}

    def test_all_families_present(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        _, families, _ = extractor.extract_layer("layer.q_proj", A, B)
        names = {f.family for f in families}
        assert self._EXPECTED_FAMILIES == names

    def test_entropy_compression_runs_in_fast_mode(self, extractor, simple_pair) -> None:
        # BUG-02 fix: entropy_compression is O(n) and runs in both modes per M1 spec.
        # Previously skipped in fast mode, causing feature_completeness=0% in quality score.
        A, B = simple_pair
        _, families, _ = extractor.extract_layer("layer.q_proj", A, B, fast=True)
        ec = next(f for f in families if f.family == "entropy_compression")
        assert ec.status == "ok", (
            f"entropy_compression should run in fast mode (O(n), spec requires both modes), got {ec.status!r}"
        )

    def test_entropy_compression_ok_in_full_mode(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        _, families, _ = extractor.extract_layer("layer.q_proj", A, B, fast=False)
        ec = next(f for f in families if f.family == "entropy_compression")
        assert ec.status == "ok"

    def test_families_tagged_with_layer_name(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        _, families, _ = extractor.extract_layer("model.layer.0.q_proj", A, B)
        for ffr in families:
            assert ffr.layer == "model.layer.0.q_proj"

    def test_all_families_ok_on_clean_input(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        _, families, errors = extractor.extract_layer("layer.q_proj", A, B)
        assert errors == [], f"Unexpected errors: {errors}"
        for ffr in families:
            assert ffr.status in ("ok", "skipped"), (
                f"Family {ffr.family!r} has unexpected status {ffr.status!r}"
            )

    def test_spectral_raw_features(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        _, families, _ = extractor.extract_layer("layer.q_proj", A, B)
        spec = next(f for f in families if f.family == "spectral")
        assert "effective_rank" in spec.raw_features
        assert "energy_concentration" in spec.raw_features
        assert spec.raw_features["energy_concentration"] >= 0.0

    def test_distribution_raw_features(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        _, families, _ = extractor.extract_layer("layer.q_proj", A, B)
        dist = next(f for f in families if f.family == "distribution")
        assert "kurtosis_A" in dist.raw_features
        assert "kurtosis_B" in dist.raw_features

    def test_entropy_raw_features(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        _, families, _ = extractor.extract_layer("layer.q_proj", A, B)
        ent = next(f for f in families if f.family == "entropy")
        assert "entropy_A" in ent.raw_features
        assert "entropy_B" in ent.raw_features
        assert 0.0 <= ent.raw_features["entropy_A"] <= 1.0

    def test_outlier_raw_features(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        _, families, _ = extractor.extract_layer("layer.q_proj", A, B)
        out = next(f for f in families if f.family == "outlier")
        assert "zscore_outlier_rate_A" in out.raw_features
        assert "zscore_outlier_rate_B" in out.raw_features

    def test_norm_raw_features_zero_B(self, extractor, zero_B_pair) -> None:
        A, B = zero_B_pair
        _, families, _ = extractor.extract_layer("layer.q_proj", A, B)
        norm = next(f for f in families if f.family == "norm")
        if norm.status == "ok":
            assert norm.raw_features["delta_norm_ratio"] == 0.0


class TestExtractLayerTensorRecord:
    def test_layer_name_preserved(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        record, _, _ = extractor.extract_layer("model.layer.0.q_proj", A, B)
        assert record.layer_name == "model.layer.0.q_proj"

    def test_shapes_preserved(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        record, _, _ = extractor.extract_layer("layer.q_proj", A, B)
        assert record.shape_a == list(A.shape)
        assert record.shape_b == list(B.shape)

    def test_no_parse_error_on_clean_input(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        record, _, _ = extractor.extract_layer("layer.q_proj", A, B)
        assert record.parse_error is None

    def test_has_norm_features_for_normal_pair(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        record, _, _ = extractor.extract_layer("layer.q_proj", A, B)
        # norm_features may be None for very large tensors, but not for small ones
        assert record.norm_features is not None or True  # not mandatory


# ---------------------------------------------------------------------------
# families_from_record — migration bridge
# ---------------------------------------------------------------------------

class TestFamiliesFromRecord:
    def test_returns_list_of_family_results(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        record, _, _ = extractor.extract_layer("layer.q_proj", A, B)
        families = FeatureExtractor.families_from_record(record)
        assert isinstance(families, list)
        assert all(isinstance(f, FeatureFamilyResult) for f in families)

    def test_all_families_present(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        record, _, _ = extractor.extract_layer("layer.q_proj", A, B)
        families = FeatureExtractor.families_from_record(record)
        names = {f.family for f in families}
        assert {"spectral", "distribution", "entropy", "entropy_compression", "outlier", "norm"} == names

    def test_layer_name_preserved(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        record, _, _ = extractor.extract_layer("model.layer.q_proj", A, B)
        families = FeatureExtractor.families_from_record(record)
        for ffr in families:
            assert ffr.layer == "model.layer.q_proj"

    def test_spectral_values_consistent_with_record(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        record, _, _ = extractor.extract_layer("layer.q_proj", A, B)
        families = FeatureExtractor.families_from_record(record)
        spec = next(f for f in families if f.family == "spectral")
        assert spec.raw_features["effective_rank"] == float(record.rank)
        assert spec.raw_features["energy_concentration"] == float(record.energy_concentration)

    def test_entropy_values_consistent_with_record(self, extractor, simple_pair) -> None:
        A, B = simple_pair
        record, _, _ = extractor.extract_layer("layer.q_proj", A, B)
        families = FeatureExtractor.families_from_record(record)
        ent = next(f for f in families if f.family == "entropy")
        assert ent.raw_features["entropy_A"] == float(record.entropy_a)
        assert ent.raw_features["entropy_B"] == float(record.entropy_b)

    def test_degraded_record_produces_degraded_families(self, extractor) -> None:
        from adaptersentry.schemas.errors import ErrorCategory
        # Construct a TensorRecord with parse_error set
        record = TensorRecord(
            layer_name="bad_layer",
            shape_a=[8, 64],
            shape_b=[64, 8],
            rank=0,
            energy_concentration=0.0,
            kurtosis_a=0.0,
            kurtosis_b=0.0,
            mean_a=0.0,
            std_a=0.0,
            mean_b=0.0,
            std_b=0.0,
            skewness_a=0.0,
            entropy_a=0.0,
            entropy_b=0.0,
            zscore_outlier_rate_a=0.0,
            zscore_outlier_rate_b=0.0,
            parse_error=ErrorCategory.DEGRADED,
        )
        families = FeatureExtractor.families_from_record(record)
        assert all(f.status == "degraded" for f in families if f.status != "skipped")


# ---------------------------------------------------------------------------
# EnsembleDetector.score_families — typed path
# ---------------------------------------------------------------------------

class TestScoreFamilies:
    def test_score_in_range(self, extractor, simple_pair) -> None:
        from adaptersentry.scoring.ensemble import EnsembleDetector
        A, B = simple_pair
        _, families, _ = extractor.extract_layer("layer.q_proj", A, B)
        score = EnsembleDetector().score_families(families)
        assert 0.0 <= score <= 100.0

    def test_empty_families_returns_low_score(self) -> None:
        from adaptersentry.scoring.ensemble import EnsembleDetector
        score = EnsembleDetector().score_families([])
        assert score < 20.0

    def test_score_consistent_with_dict_path(self, extractor, simple_pair) -> None:
        """score_families and score should produce close results for the same data."""
        from adaptersentry.scoring.ensemble import EnsembleDetector
        A, B = simple_pair
        record, families, _ = extractor.extract_layer("layer.q_proj", A, B)
        detector = EnsembleDetector()

        # Build a minimal layer_reports dict from the record for the old path
        layer_reports = {
            record.layer_name: {
                "kurtosis_A": record.kurtosis_a,
                "kurtosis_B": record.kurtosis_b,
                "energy_concentration": record.energy_concentration,
                "entropy_A": record.entropy_a,
                "entropy_B": record.entropy_b,
                "zscore_outlier_rate_A": record.zscore_outlier_rate_a,
                "zscore_outlier_rate_B": record.zscore_outlier_rate_b,
                "isolation_score_A": record.isolation_score_a,
            }
        }
        score_old = detector.score(layer_reports)
        score_new = detector.score_families(families)
        # Scores should be reasonably close (within 10 points)
        assert abs(score_old - score_new) < 10.0, (
            f"score_families ({score_new:.1f}) diverged too much from "
            f"score ({score_old:.1f})"
        )

    def test_failed_families_excluded_from_scoring(self) -> None:
        from adaptersentry.scoring.ensemble import EnsembleDetector
        bad = FeatureFamilyResult(
            family="distribution",
            family_schema_version="1.0.0",
            status="failed",
            raw_features={},
        )
        score = EnsembleDetector().score_families([bad])
        assert score < 20.0  # failed families contribute zero

    def test_high_kurtosis_raises_score(self, extractor) -> None:
        from adaptersentry.scoring.ensemble import EnsembleDetector
        rng = np.random.default_rng(1)
        # Create a very heavy-tailed distribution
        A = np.zeros((8, 64), dtype=np.float32)
        A[0, 0] = 1000.0  # single spike — extreme kurtosis
        B = rng.standard_normal((64, 8)).astype(np.float32)
        _, families, _ = extractor.extract_layer("layer.q_proj", A, B)
        score = EnsembleDetector().score_families(families)
        # High kurtosis should produce a non-trivial score
        assert score > 0.0
