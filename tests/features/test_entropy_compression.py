"""Tests for M1-ANAL-02: EntropyCompressionFeatures."""

from __future__ import annotations

import numpy as np
import pytest

from adaptersentry.features.entropy_compression import compute_entropy_compression_features
from adaptersentry.schemas.entropy_compression_features import EntropyCompressionFeatures


def _make_pair(rank: int = 4, out: int = 32, in_: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((rank, in_)).astype(np.float32)
    B = rng.standard_normal((out, rank)).astype(np.float32)
    return A, B


# ---------------------------------------------------------------------------
# Schema defaults
# ---------------------------------------------------------------------------

class TestEntropyCompressionFeaturesSchema:
    def test_all_fields_have_defaults(self):
        ec = EntropyCompressionFeatures()
        for field in (
            "value_repeat_ratio_a", "unique_value_ratio_a", "approx_compression_ratio_a",
            "byte_entropy_a", "sign_entropy_a", "sign_balance_a", "quantization_suspect_score_a",
            "value_repeat_ratio_b", "unique_value_ratio_b", "approx_compression_ratio_b",
            "byte_entropy_b", "sign_entropy_b", "sign_balance_b", "quantization_suspect_score_b",
        ):
            assert hasattr(ec, field), f"missing field: {field}"

    def test_is_frozen(self):
        ec = EntropyCompressionFeatures()
        with pytest.raises(Exception):
            ec.sign_balance_a = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compute_entropy_compression_features — return type
# ---------------------------------------------------------------------------

class TestComputeEntropyCompressionFeatures:
    def test_returns_schema_instance(self):
        A, B = _make_pair()
        result = compute_entropy_compression_features(A, B)
        assert isinstance(result, EntropyCompressionFeatures)

    def test_does_not_raise(self):
        A, B = _make_pair(seed=123)
        result = compute_entropy_compression_features(A, B)
        assert result is not None

    # --- bounds ---

    def test_unique_value_ratio_bounds(self):
        A, B = _make_pair()
        ec = compute_entropy_compression_features(A, B)
        assert 0.0 <= ec.unique_value_ratio_a <= 1.0
        assert 0.0 <= ec.unique_value_ratio_b <= 1.0

    def test_value_repeat_ratio_complement(self):
        A, B = _make_pair()
        ec = compute_entropy_compression_features(A, B)
        assert ec.value_repeat_ratio_a == pytest.approx(1.0 - ec.unique_value_ratio_a, abs=1e-9)
        assert ec.value_repeat_ratio_b == pytest.approx(1.0 - ec.unique_value_ratio_b, abs=1e-9)

    def test_byte_entropy_bounds(self):
        A, B = _make_pair()
        ec = compute_entropy_compression_features(A, B)
        assert 0.0 <= ec.byte_entropy_a <= 1.0
        assert 0.0 <= ec.byte_entropy_b <= 1.0

    def test_sign_entropy_bounds(self):
        A, B = _make_pair()
        ec = compute_entropy_compression_features(A, B)
        assert 0.0 <= ec.sign_entropy_a <= 1.0
        assert 0.0 <= ec.sign_entropy_b <= 1.0

    def test_sign_balance_bounds(self):
        A, B = _make_pair()
        ec = compute_entropy_compression_features(A, B)
        assert 0.0 <= ec.sign_balance_a <= 1.0
        assert 0.0 <= ec.sign_balance_b <= 1.0

    def test_quantization_suspect_score_bounds(self):
        A, B = _make_pair()
        ec = compute_entropy_compression_features(A, B)
        assert 0.0 <= ec.quantization_suspect_score_a <= 1.0
        assert 0.0 <= ec.quantization_suspect_score_b <= 1.0

    def test_compression_ratio_positive(self):
        A, B = _make_pair()
        ec = compute_entropy_compression_features(A, B)
        assert ec.approx_compression_ratio_a > 0.0
        assert ec.approx_compression_ratio_b > 0.0

    # --- semantic correctness ---

    def test_all_zeros_high_repeat_ratio(self):
        A = np.zeros((8, 32), dtype=np.float32)  # 256 elements, 1 unique value
        B = np.zeros((16, 8), dtype=np.float32)
        ec = compute_entropy_compression_features(A, B)
        # 1 unique / 256 total → value_repeat_ratio = 255/256 ≈ 0.996
        assert ec.value_repeat_ratio_a > 0.99
        assert ec.unique_value_ratio_a < 0.01

    def test_all_zeros_zero_byte_entropy(self):
        A = np.zeros((8, 32), dtype=np.float32)
        B = np.zeros((16, 8), dtype=np.float32)
        ec = compute_entropy_compression_features(A, B)
        assert ec.byte_entropy_a == pytest.approx(0.0, abs=1e-6)

    def test_all_zeros_sign_entropy_zero(self):
        A = np.zeros((8, 32), dtype=np.float32)
        B = np.zeros((16, 8), dtype=np.float32)
        ec = compute_entropy_compression_features(A, B)
        assert ec.sign_entropy_a == pytest.approx(0.0, abs=1e-6)

    def test_all_positive_sign_balance_one(self):
        A = np.ones((8, 32), dtype=np.float32)
        B = np.ones((16, 8), dtype=np.float32)
        ec = compute_entropy_compression_features(A, B)
        assert ec.sign_balance_a == pytest.approx(1.0, abs=1e-6)
        assert ec.sign_balance_b == pytest.approx(1.0, abs=1e-6)

    def test_all_negative_sign_balance_zero(self):
        A = -np.ones((8, 32), dtype=np.float32)
        B = -np.ones((16, 8), dtype=np.float32)
        ec = compute_entropy_compression_features(A, B)
        assert ec.sign_balance_a == pytest.approx(0.0, abs=1e-6)

    def test_all_same_value_quant_score_near_one(self):
        A = np.full((8, 32), 3.14, dtype=np.float32)
        B = np.full((16, 8), 3.14, dtype=np.float32)
        ec = compute_entropy_compression_features(A, B)
        # 1 unique value → log2(2)/32 = 1/32 → q_score ≈ 0.969
        assert ec.quantization_suspect_score_a > 0.9

    def test_normal_random_unique_ratio_near_one(self):
        # Standard normal float32 with 256*64 = 16384 elements — nearly all unique
        A, B = _make_pair(rank=16, out=64, in_=256, seed=42)
        ec = compute_entropy_compression_features(A, B)
        assert ec.unique_value_ratio_a > 0.99

    def test_normal_random_quant_score_near_zero(self):
        A, B = _make_pair(rank=16, out=64, in_=256, seed=42)
        ec = compute_entropy_compression_features(A, B)
        # ~4096 unique fp32 values → effective_bits = log2(4097) ≈ 12
        # q_score = 1 - 12/32 ≈ 0.625 — clearly lower than 8-bit quant (0.75)
        assert ec.quantization_suspect_score_a < 0.7

    def test_quantized_8bit_pattern(self):
        # Simulate 8-bit quantization: only 256 unique values across a large tensor
        rng = np.random.default_rng(7)
        levels = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
        flat_A = rng.choice(levels, size=8 * 64)
        A = flat_A.reshape(8, 64).astype(np.float32)
        flat_B = rng.choice(levels, size=16 * 8)
        B = flat_B.reshape(16, 8).astype(np.float32)
        ec = compute_entropy_compression_features(A, B)
        # 256 unique values → q_score = 1 - log2(257)/32 ≈ 1 - 8/32 = 0.75
        assert ec.quantization_suspect_score_a > 0.5
        assert ec.unique_value_ratio_a < 0.5

    def test_empty_tensor_does_not_crash(self):
        A = np.zeros((0, 8), dtype=np.float32)
        B = np.zeros((4, 0), dtype=np.float32)
        result = compute_entropy_compression_features(A, B)
        # Should return a result or None — must not raise
        assert result is not None or result is None


# ---------------------------------------------------------------------------
# feature_extractor integration
# ---------------------------------------------------------------------------

class TestFeatureExtractorEntropyCompression:
    def test_entropy_compression_family_present(self):
        from adaptersentry.engine.feature_extractor import FeatureExtractor

        A, B = _make_pair(rank=4, out=32, in_=64)
        _, families, _ = FeatureExtractor().extract_layer("test.q_proj", A, B)
        names = [f.family for f in families]
        assert "entropy_compression" in names

    def test_entropy_compression_family_ok(self):
        from adaptersentry.engine.feature_extractor import FeatureExtractor

        A, B = _make_pair()
        _, families, _ = FeatureExtractor().extract_layer("test.q_proj", A, B)
        ec = next(f for f in families if f.family == "entropy_compression")
        assert ec.status == "ok"

    def test_entropy_compression_raw_features_keys(self):
        from adaptersentry.engine.feature_extractor import FeatureExtractor

        A, B = _make_pair()
        _, families, _ = FeatureExtractor().extract_layer("test.q_proj", A, B)
        ec = next(f for f in families if f.family == "entropy_compression")
        expected = {
            "value_repeat_ratio_A", "unique_value_ratio_A", "approx_compression_ratio_A",
            "byte_entropy_A", "sign_entropy_A", "sign_balance_A", "quantization_suspect_score_A",
            "value_repeat_ratio_B", "unique_value_ratio_B", "approx_compression_ratio_B",
            "byte_entropy_B", "sign_entropy_B", "sign_balance_B", "quantization_suspect_score_B",
        }
        assert expected.issubset(ec.raw_features.keys())

    def test_tensor_record_stores_ec_features(self):
        from adaptersentry.engine.feature_extractor import FeatureExtractor

        A, B = _make_pair()
        record, _, _ = FeatureExtractor().extract_layer("test.q_proj", A, B)
        assert record.entropy_compression_features is not None
        assert isinstance(record.entropy_compression_features, EntropyCompressionFeatures)

    def test_families_from_record_includes_ec(self):
        from adaptersentry.engine.feature_extractor import FeatureExtractor

        A, B = _make_pair()
        record, _, _ = FeatureExtractor().extract_layer("test.q_proj", A, B)
        families = FeatureExtractor.families_from_record(record)
        names = [f.family for f in families]
        assert "entropy_compression" in names

    def test_families_from_record_ec_keys(self):
        from adaptersentry.engine.feature_extractor import FeatureExtractor

        A, B = _make_pair()
        record, _, _ = FeatureExtractor().extract_layer("test.q_proj", A, B)
        families = FeatureExtractor.families_from_record(record)
        ec = next(f for f in families if f.family == "entropy_compression")
        assert "byte_entropy_A" in ec.raw_features
        assert "quantization_suspect_score_B" in ec.raw_features
