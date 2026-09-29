"""Tests for ESR per-module features: parity with explicit ΔW, invariances, heads."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import kurtosis

from adaptersentry.features.spectral import (
    gini,
    head_features,
    hoyer,
    infer_head_dim,
    module_features,
    top_k_share,
)

RNG = np.random.default_rng(7)


def _ab(r: int = 8, d: int = 256, k: int = 192) -> tuple[np.ndarray, np.ndarray]:
    a = RNG.standard_normal((r, k)).astype(np.float32)
    b = (RNG.standard_normal((d, r)) * 0.01 * np.exp(-np.arange(r) / 3)).astype(np.float32)
    return a, b


def _dw(a: np.ndarray, b: np.ndarray, s: float = 1.0) -> np.ndarray:
    return s * (b.astype(np.float64) @ a.astype(np.float64))


class TestHelpers:
    def test_hoyer_extremes(self) -> None:
        assert hoyer(np.ones(8)) == pytest.approx(0.0)
        assert hoyer(np.array([1.0, 0, 0, 0])) == pytest.approx(1.0)
        assert hoyer(np.zeros(4)) is None

    def test_gini_extremes(self) -> None:
        assert gini(np.ones(8)) == pytest.approx(0.0)
        assert gini(np.array([0, 0, 0, 5.0])) == pytest.approx(0.75)
        assert gini(np.zeros(3)) is None

    def test_top_k_share(self) -> None:
        x = np.array([3.0, 1.0, 1.0, 1.0])
        assert top_k_share(x, 1) == pytest.approx(9 / 12)
        assert top_k_share(np.zeros(3)) is None


class TestParityWithExplicit:
    def test_matches_features_of_explicit_matrix(self) -> None:
        a, b = _ab()
        f, _ = module_features(a, b, 2.0)
        dw = _dw(a, b, 2.0)
        s = np.linalg.svd(dw, compute_uv=False)[:8]
        e = s ** 2 / np.sum(s ** 2)
        rows = np.linalg.norm(dw, axis=1)
        cols = np.linalg.norm(dw, axis=0)
        assert f.log_rms == pytest.approx(math.log(np.linalg.norm(dw) / math.sqrt(dw.size)), abs=1e-5)
        assert f.log_sigma1 == pytest.approx(math.log(s[0]), abs=1e-5)
        assert f.top1_e == pytest.approx(e[0], rel=1e-5)
        assert f.srank_r == pytest.approx((1 / e[0]) / 8, rel=1e-5)
        assert f.pr_r == pytest.approx((1 / np.sum(e ** 2)) / 8, rel=1e-5)
        assert f.row_hoyer == pytest.approx(hoyer(rows), rel=1e-4)
        assert f.col_hoyer == pytest.approx(hoyer(cols), rel=1e-4)
        assert f.kurt_dw == pytest.approx(kurtosis(dw.ravel(), fisher=False), rel=0.05)

    def test_spectral_entropy_bounds(self) -> None:
        a, b = _ab()
        f, _ = module_features(a, b, 1.0)
        assert 0.0 <= f.spec_h <= 1.0


class TestInvariances:
    def test_shape_features_ignore_scale(self) -> None:
        a, b = _ab()
        f1, _ = module_features(a, b, 1.0)
        f2, _ = module_features(a, b, 37.0)
        for name in ("top1_e", "srank_r", "pr_r", "spec_h", "excess_conc", "row_hoyer", "row_gini",
                     "row_top8", "col_hoyer", "kurt_dw"):
            assert getattr(f1, name) == pytest.approx(getattr(f2, name), rel=1e-5), name
        assert f2.log_rms - f1.log_rms == pytest.approx(math.log(37.0), abs=1e-5)

    def test_gauge_invariance(self) -> None:
        a, b = _ab()
        g = RNG.standard_normal((8, 8)) + 5 * np.eye(8)
        f1, _ = module_features(a, b, 1.0)
        f2, _ = module_features((np.linalg.inv(g) @ a).astype(np.float32), (b @ g).astype(np.float32), 1.0)
        assert f1.top1_e == pytest.approx(f2.top1_e, rel=1e-3)
        assert f1.row_hoyer == pytest.approx(f2.row_hoyer, rel=1e-3)

    def test_zero_update_has_no_features(self) -> None:
        a, b = _ab()
        f, core = module_features(a, np.zeros_like(b), 1.0)
        assert core.is_zero
        assert all(v is None for v in f.model_dump().values())


class TestInjectionSignals:
    def test_row_concentration_flags_output_targeted_update(self) -> None:
        a, b = _ab(16, 2048, 1024)
        clean, _ = module_features(a, b, 1.0)
        b2 = b.copy()
        u = np.zeros(2048, np.float32)
        u[RNG.choice(2048, 8, replace=False)] = 0.2
        b2[:, -1] = u  # one rank slot writes into 8 outputs only
        inj, _ = module_features(a, b2, 1.0)
        assert inj.row_top8 > 5 * clean.row_top8
        assert inj.row_hoyer > clean.row_hoyer
        assert inj.kurt_dw > clean.kurt_dw

    def test_single_direction_update_has_high_concentration(self) -> None:
        a, b = _ab()
        b1 = np.zeros_like(b)
        b1[:, 0] = b[:, 0] * 50  # all energy in one direction
        f, _ = module_features(a, b1, 1.0)
        assert f.top1_e == pytest.approx(1.0) and f.excess_conc > 0


class TestEmbeddingTransposed:
    def test_token_rows_are_columns_of_ba(self) -> None:
        # embedding LoRA: A (r, vocab=300), B (dim=32, r); ΔW = (B·A)ᵀ is vocab × dim.
        a = RNG.standard_normal((4, 300)).astype(np.float32)
        b = RNG.standard_normal((32, 4)).astype(np.float32)
        a[:, 17] *= 40  # one token receives most of the update
        f, _ = module_features(a, b, 1.0, transposed=True)
        token_rows = np.linalg.norm(_dw(a, b).T, axis=1)
        assert f.row_hoyer == pytest.approx(hoyer(token_rows), rel=1e-4)
        assert f.kurt_dw == pytest.approx(kurtosis(_dw(a, b).T.ravel(), fisher=False), rel=0.1)


class TestHeads:
    def test_output_split_heads_partition_energy(self) -> None:
        a = RNG.standard_normal((8, 256)).astype(np.float32)
        b = RNG.standard_normal((512, 8)).astype(np.float32)  # 4 heads of 128
        heads = head_features(a, b, 1.0, "q_proj")
        assert heads is not None and [h.head for h in heads] == [0, 1, 2, 3]
        total = sum(math.exp(2 * h.features.log_rms) * 128 * 256 for h in heads)
        assert total == pytest.approx(np.linalg.norm(_dw(a, b)) ** 2, rel=1e-4)

    def test_input_split_for_o_proj(self) -> None:
        a = RNG.standard_normal((8, 512)).astype(np.float32)
        b = RNG.standard_normal((256, 8)).astype(np.float32)
        heads = head_features(a, b, 1.0, "o_proj")
        assert heads is not None and len(heads) == 4

    def test_one_poisoned_head_stands_out(self) -> None:
        a = RNG.standard_normal((8, 256)).astype(np.float32)
        b = (RNG.standard_normal((1024, 8)) * 0.01).astype(np.float32)
        b[3 * 128:4 * 128, 0] *= 60  # head 3 dominated by one direction
        heads = head_features(a, b, 1.0, "v_proj")
        top1 = [h.features.top1_e for h in heads]
        assert int(np.argmax(top1)) == 3

    def test_mlp_has_no_heads(self) -> None:
        a, b = _ab()
        assert head_features(a, b, 1.0, "down_proj") is None

    def test_infer_head_dim(self) -> None:
        assert infer_head_dim(8192) == 128
        assert infer_head_dim(1024) == 128
        assert infer_head_dim(192) == 64
        assert infer_head_dim(100) is None
