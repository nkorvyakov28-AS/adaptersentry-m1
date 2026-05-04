"""Tests for compute_norm_features — NormFeatures from LoRA A/B pairs."""

from __future__ import annotations

import numpy as np
import pytest

from adaptersentry.features.delta_norm import compute_norm_features
from adaptersentry.schemas.norm_features import NormFeatures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pair(out: int, rank: int, in_: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (tensor_A, tensor_B) of compatible shape for standard LoRA."""
    rng = np.random.default_rng(seed)
    # A: (rank, in_features),  B: (out_features, rank)
    a = rng.standard_normal((rank, in_)).astype(np.float32)
    b = rng.standard_normal((out, rank)).astype(np.float32)
    return a, b


# ---------------------------------------------------------------------------
# Return type contract
# ---------------------------------------------------------------------------


class TestReturnTypeContract:
    def test_returns_norm_features_instance(self) -> None:
        a, b = _pair(64, 8, 64)
        result = compute_norm_features(a, b)
        assert isinstance(result, NormFeatures)

    def test_result_is_frozen(self) -> None:
        a, b = _pair(64, 8, 64)
        result = compute_norm_features(a, b)
        assert result is not None
        with pytest.raises(Exception):
            result.fro_norm_delta = 0.0  # type: ignore[misc]

    def test_all_fields_are_float(self) -> None:
        a, b = _pair(64, 8, 64)
        result = compute_norm_features(a, b)
        assert result is not None
        assert isinstance(result.fro_norm_delta, float)
        assert isinstance(result.max_abs_delta, float)
        assert isinstance(result.mean_abs_delta, float)
        assert isinstance(result.delta_norm_ratio, float)


# ---------------------------------------------------------------------------
# Valid paired tensors
# ---------------------------------------------------------------------------


class TestValidPair:
    def test_fro_norm_positive(self) -> None:
        a, b = _pair(64, 8, 64)
        result = compute_norm_features(a, b)
        assert result is not None
        assert result.fro_norm_delta > 0.0

    def test_delta_norm_ratio_bounded(self) -> None:
        """delta_norm_ratio must be in [0, 1] by submultiplicativity."""
        a, b = _pair(128, 16, 128)
        result = compute_norm_features(a, b)
        assert result is not None
        assert 0.0 <= result.delta_norm_ratio <= 1.0 + 1e-9  # float rounding tolerance

    def test_max_abs_geq_mean_abs(self) -> None:
        a, b = _pair(32, 4, 32)
        result = compute_norm_features(a, b)
        assert result is not None
        assert result.max_abs_delta >= result.mean_abs_delta

    def test_max_abs_geq_zero(self) -> None:
        a, b = _pair(32, 4, 32)
        result = compute_norm_features(a, b)
        assert result is not None
        assert result.max_abs_delta >= 0.0

    def test_fro_matches_manual_computation(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.standard_normal((8, 64)).astype(np.float64)
        b = rng.standard_normal((64, 8)).astype(np.float64)
        result = compute_norm_features(a, b)
        assert result is not None
        # float32 matmul: relative error < 1e-5 is sufficient for anomaly thresholds.
        expected_fro = float(np.linalg.norm(b.astype(np.float32) @ a.astype(np.float32), "fro"))
        assert abs(result.fro_norm_delta - expected_fro) < 1e-4

    def test_delta_derived_from_combined_not_a_alone(self) -> None:
        """Changing B while keeping A constant must change all norm features."""
        a, b1 = _pair(32, 4, 32, seed=0)
        _, b2 = _pair(32, 4, 32, seed=99)
        r1 = compute_norm_features(a, b1)
        r2 = compute_norm_features(a, b2)
        assert r1 is not None and r2 is not None
        assert r1.fro_norm_delta != pytest.approx(r2.fro_norm_delta)

    def test_large_rank_small_outer_dims(self) -> None:
        """High rank with small feature dims — should still work."""
        a, b = _pair(16, 128, 16)
        result = compute_norm_features(a, b)
        assert result is not None
        assert result.fro_norm_delta > 0.0


# ---------------------------------------------------------------------------
# Zero-B initialization special case
# ---------------------------------------------------------------------------


class TestZeroBInit:
    def test_zero_b_returns_all_zeros(self) -> None:
        """Standard LoRA init: B = 0, A = random — delta is zero, not anomalous."""
        rng = np.random.default_rng(7)
        a = rng.standard_normal((8, 64)).astype(np.float32)
        b = np.zeros((64, 8), dtype=np.float32)
        result = compute_norm_features(a, b)
        assert result is not None
        assert result.fro_norm_delta == 0.0
        assert result.max_abs_delta == 0.0
        assert result.mean_abs_delta == 0.0
        assert result.delta_norm_ratio == 0.0

    def test_near_zero_b_treated_as_zero(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.standard_normal((8, 64)).astype(np.float32)
        b = np.full((64, 8), 1e-15, dtype=np.float32)  # below 1e-12 threshold
        result = compute_norm_features(a, b)
        assert result is not None
        assert result.delta_norm_ratio == 0.0

    def test_zero_b_is_valid_norm_features_not_none(self) -> None:
        a = np.ones((4, 8), dtype=np.float32)
        b = np.zeros((8, 4), dtype=np.float32)
        result = compute_norm_features(a, b)
        assert result is not None


# ---------------------------------------------------------------------------
# Missing / incomplete pair
# ---------------------------------------------------------------------------


class TestIncompletePair:
    def test_shape_mismatch_returns_none(self) -> None:
        """Inner dimensions don't match: B(out=16, rank=4), A(rank=8, in=32)."""
        a = np.ones((8, 32), dtype=np.float32)   # rank=8
        b = np.ones((16, 4), dtype=np.float32)   # rank=4 ≠ 8
        result = compute_norm_features(a, b)
        assert result is None

    def test_1d_a_is_reshaped_not_rejected(self) -> None:
        """1-D tensors are reshaped to 2-D before processing."""
        a = np.ones(64, dtype=np.float32)         # reshaped to (1, 64)
        b = np.ones((8, 1), dtype=np.float32)
        result = compute_norm_features(a, b)
        assert result is not None

    def test_wrong_ndim_returns_none(self) -> None:
        """3-D+ tensors that can't be reduced to 2-D return None."""
        a = np.ones((2, 4, 8), dtype=np.float32)
        b = np.ones((8, 2), dtype=np.float32)
        # a.ndim==3, b.shape[1]==2 != a.shape[0]==2... but after reshape check fails
        # The exact behaviour depends on reshape: 3-D is rejected
        result = compute_norm_features(a, b)
        # 3-D a can't be treated as 2-D — should return None
        assert result is None


# ---------------------------------------------------------------------------
# Malformed shape mismatch
# ---------------------------------------------------------------------------


class TestMalformedShapes:
    def test_zero_rank_returns_none_or_zero(self) -> None:
        """Zero rank dimension — degenerate case."""
        a = np.zeros((0, 64), dtype=np.float32)
        b = np.zeros((64, 0), dtype=np.float32)
        # Inner dims match (0 == 0) but result is degenerate
        result = compute_norm_features(a, b)
        # Either None or a zero-filled NormFeatures is acceptable
        assert result is None or isinstance(result, NormFeatures)

    def test_transposed_pair_returns_none(self) -> None:
        """A and B swapped — inner dims won't match for typical shapes."""
        a, b = _pair(64, 8, 32)  # A:(8,32), B:(64,8) — correct
        # Swap them: pass B as A argument and A as B argument
        result = compute_norm_features(b, a)  # b:(64,8) as A, a:(8,32) as B → mismatch
        # B.shape[1]=32 != A.shape[0]=64 — should return None
        assert result is None


# ---------------------------------------------------------------------------
# Pipeline integration — NormFeatures in TensorRecord and AdapterReport
# ---------------------------------------------------------------------------


class TestNormFeaturesInPipeline:
    def _make_adapter(self, tmp_path, metadata=None):
        from pathlib import Path
        from safetensors.numpy import save_file

        rng = np.random.default_rng(42)
        tensors = {
            "model.layers.0.q_proj.lora_A.weight": rng.standard_normal((8, 64)).astype(np.float32),
            "model.layers.0.q_proj.lora_B.weight": rng.standard_normal((64, 8)).astype(np.float32),
            "model.layers.1.v_proj.lora_A.weight": rng.standard_normal((8, 64)).astype(np.float32),
            "model.layers.1.v_proj.lora_B.weight": rng.standard_normal((64, 8)).astype(np.float32),
        }
        path = tmp_path / "adapter.safetensors"
        save_file(tensors, str(path), metadata=metadata or {"r": "8"})
        return path

    def test_tensor_records_have_norm_features(self, tmp_path) -> None:
        from adaptersentry.analyzer import scan

        path = self._make_adapter(tmp_path)
        report = scan(path)
        assert all(
            tr.norm_features is not None for tr in report.tensor_records
        ), "every fully-analysed layer must have norm_features"

    def test_norm_features_schema_type(self, tmp_path) -> None:
        from adaptersentry.analyzer import scan

        path = self._make_adapter(tmp_path)
        report = scan(path)
        for tr in report.tensor_records:
            assert tr.norm_features is None or isinstance(tr.norm_features, NormFeatures)

    def test_zero_b_layer_has_zero_ratio(self, tmp_path) -> None:
        """Layer with zero-init B produces norm_features.delta_norm_ratio == 0.0."""
        from pathlib import Path
        from safetensors.numpy import save_file
        from adaptersentry.analyzer import scan

        rng = np.random.default_rng(0)
        path = tmp_path / "zero_b.safetensors"
        save_file(
            {
                "layer.lora_A.weight": rng.standard_normal((8, 64)).astype(np.float32),
                "layer.lora_B.weight": np.zeros((64, 8), dtype=np.float32),
                # Need a second pair so the adapter has ≥ 2 pairs
                "layer2.lora_A.weight": rng.standard_normal((8, 64)).astype(np.float32),
                "layer2.lora_B.weight": np.zeros((64, 8), dtype=np.float32),
            },
            str(path),
            metadata={"r": "8"},
        )
        report = scan(path)
        zero_records = [tr for tr in report.tensor_records if tr.norm_features is not None]
        assert all(tr.norm_features.delta_norm_ratio == 0.0 for tr in zero_records)

    def test_text_reporter_includes_delta_norm_section(self, tmp_path) -> None:
        from adaptersentry.analyzer import scan
        from adaptersentry.reporters.text import render

        path = self._make_adapter(tmp_path)
        report = scan(path)
        output = render(report, no_color=True)
        assert "ΔW norm" in output
        assert "delta_norm_ratio" in output
        assert "ratio=" in output


# ---------------------------------------------------------------------------
# Memory guard — Cholesky path for out×in > _MAX_DELTA_NUMEL_FULL (4M)
# ---------------------------------------------------------------------------


class TestLargeDeltaCholeskyPath:
    """Verify that full mode uses Cholesky (not ΔW materialisation) for large layers.

    Uses out=2048, in=2560 → delta_numel=5,242,880 > 4M threshold.
    Cholesky path: fro_norm computed exactly; max_abs=mean_abs=0.0.
    """

    def _large_pair(self, rank: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        # out=2048, in=2560 → 5.24M elements — above 4M full threshold
        a = rng.standard_normal((rank, 2560)).astype(np.float32)
        b = rng.standard_normal((2048, rank)).astype(np.float32)
        return a, b

    def test_full_mode_max_abs_is_zero(self) -> None:
        a, b = self._large_pair()
        result = compute_norm_features(a, b, fast=False)
        assert result is not None
        assert result.max_abs_delta == 0.0

    def test_full_mode_mean_abs_is_zero(self) -> None:
        a, b = self._large_pair()
        result = compute_norm_features(a, b, fast=False)
        assert result is not None
        assert result.mean_abs_delta == 0.0

    def test_full_mode_fro_norm_positive(self) -> None:
        a, b = self._large_pair(rank=32, seed=3)
        result = compute_norm_features(a, b, fast=False)
        assert result is not None
        assert result.fro_norm_delta > 0.0

    def test_full_mode_ratio_bounded(self) -> None:
        a, b = self._large_pair(rank=32, seed=5)
        result = compute_norm_features(a, b, fast=False)
        assert result is not None
        assert 0.0 <= result.delta_norm_ratio <= 1.0 + 1e-9

    def test_below_threshold_has_exact_max_abs(self) -> None:
        # out=64, in=64 → 4096 elements — well below 4M threshold
        a, b = _pair(out=64, rank=8, in_=64)
        result = compute_norm_features(a, b, fast=False)
        assert result is not None
        assert result.max_abs_delta > 0.0  # exact path populates this
