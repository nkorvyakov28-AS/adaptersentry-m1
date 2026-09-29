"""Parity tests: every core statistic must equal the explicit computation on B·A."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kurtosis

from adaptersentry.features.lowrank_core import (
    compute_core,
    delta_cosine,
    delta_inner,
    kurtosis_dw,
)

RNG = np.random.default_rng(42)


def _ab(r: int = 8, d: int = 96, k: int = 64, *, b_scale: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    a = RNG.standard_normal((r, k)).astype(np.float32)
    b = (RNG.standard_normal((d, r)) * b_scale).astype(np.float32)
    return a, b


def _explicit(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    return s * (b.astype(np.float64) @ a.astype(np.float64))


class TestParity:
    @pytest.mark.parametrize("r,d,k", [(1, 32, 48), (8, 96, 64), (16, 64, 200), (32, 40, 40)])
    def test_spectrum_matches_full_svd(self, r: int, d: int, k: int) -> None:
        a, b = _ab(r, d, k)
        core = compute_core(a, b, 2.0)
        ref = np.linalg.svd(_explicit(a, b, 2.0), compute_uv=False)[:r]
        np.testing.assert_allclose(core.sigma, ref, rtol=1e-5, atol=1e-9)

    def test_fro_row_col_norms(self) -> None:
        a, b = _ab()
        dw = _explicit(a, b, 0.5)
        core = compute_core(a, b, 0.5)
        assert core.fro == pytest.approx(np.linalg.norm(dw), rel=1e-6)
        np.testing.assert_allclose(core.row_norms, np.linalg.norm(dw, axis=1), rtol=1e-5)
        np.testing.assert_allclose(core.col_norms, np.linalg.norm(dw, axis=0), rtol=1e-5)
        assert core.shape == dw.shape

    def test_rank_deficient_a(self) -> None:
        a, b = _ab(8, 64, 64)
        a[3:] = 0.0  # only 3 directions carry information
        core = compute_core(a, b)
        ref = np.linalg.svd(_explicit(a, b, 1.0), compute_uv=False)[:8]
        np.testing.assert_allclose(core.sigma, ref, rtol=1e-5, atol=1e-7)
        assert np.count_nonzero(core.significant_sigma()) == 3

    def test_ill_conditioned_b(self) -> None:
        a, _ = _ab(16, 256, 256)
        u = RNG.standard_normal((256, 16))
        b = (u * np.logspace(0, -5, 16)).astype(np.float32)
        core = compute_core(a, b)
        ref = np.linalg.svd(_explicit(a, b, 1.0), compute_uv=False)[:16]
        np.testing.assert_allclose(core.sigma[:5], ref[:5], rtol=1e-5)

    def test_zero_update(self) -> None:
        a, b = _ab()
        core = compute_core(a, np.zeros_like(b))
        assert core.is_zero and np.all(core.sigma == 0) and core.significant_sigma().size == 0


class TestInvariances:
    def test_gauge_invariance(self) -> None:
        # B·A = (B·G)(G⁻¹·A): re-factoring must not change any statistic of ΔW.
        a, b = _ab()
        g = RNG.standard_normal((8, 8)) + 4 * np.eye(8)
        a2 = (np.linalg.inv(g) @ a).astype(np.float32)
        b2 = (b @ g).astype(np.float32)
        c1, c2 = compute_core(a, b), compute_core(a2, b2)
        np.testing.assert_allclose(c1.sigma, c2.sigma, rtol=1e-4)
        np.testing.assert_allclose(c1.row_norms, c2.row_norms, rtol=1e-4)

    def test_scale_multiplies_everything(self) -> None:
        a, b = _ab()
        c1, c3 = compute_core(a, b, 1.0), compute_core(a, b, 3.0)
        np.testing.assert_allclose(c3.sigma, 3 * c1.sigma, rtol=1e-6)
        assert c3.fro == pytest.approx(3 * c1.fro)

    def test_rescale_attack_is_invisible_after_scaling(self) -> None:
        # B / c with scale × c gives the same ΔW as the original.
        a, b = _ab()
        np.testing.assert_allclose(
            compute_core(a, b / 1e4, 1e4).sigma, compute_core(a, b, 1.0).sigma, rtol=1e-4,
        )

    def test_rejects_bad_shapes(self) -> None:
        a, b = _ab()
        with pytest.raises(ValueError):
            compute_core(a, b[:, :4])


class TestInnerProducts:
    def test_inner_product_matches_explicit(self) -> None:
        a1, b1 = _ab(8, 64, 48)
        a2, b2 = _ab(4, 64, 48)
        ref = float(np.sum(_explicit(a1, b1, 2.0) * _explicit(a2, b2, 0.5)))
        assert delta_inner(a1, b1, 2.0, a2, b2, 0.5) == pytest.approx(ref, rel=1e-5)

    def test_cosine_of_identical_updates_is_one(self) -> None:
        a, b = _ab()
        assert delta_cosine(a, b, 1.0, a, b, 7.0) == pytest.approx(1.0, rel=1e-6)

    def test_cosine_of_zero_update_is_none(self) -> None:
        a, b = _ab()
        assert delta_cosine(a, b, 1.0, a, np.zeros_like(b), 1.0) is None


class TestKurtosis:
    def test_close_to_full_kurtosis(self) -> None:
        a, b = _ab(16, 512, 512)
        full = kurtosis(_explicit(a, b, 1.0).ravel(), fisher=False)
        assert kurtosis_dw(a, b) == pytest.approx(full, rel=0.05)

    def test_rare_spiky_rows_are_not_missed(self) -> None:
        # 3 rows out of 4096 carry a huge update: the case uniform sampling misses.
        a = RNG.standard_normal((16, 1024)).astype(np.float32)
        b = (RNG.standard_normal((4096, 16)) * 0.01).astype(np.float32)
        b[[7, 1500, 3000]] *= 100.0
        full = kurtosis(_explicit(a, b, 1.0).ravel(), fisher=False)
        est = kurtosis_dw(a, b)
        assert full > 100
        assert est == pytest.approx(full, rel=0.15)

    def test_nonzero_mean_update(self) -> None:
        a, b = _ab(8, 256, 256)
        a += 0.5  # shifts the entry mean away from zero
        full = kurtosis(_explicit(a, b, 1.0).ravel(), fisher=False)
        assert kurtosis_dw(a, b) == pytest.approx(full, rel=0.05)

    def test_scale_invariant(self) -> None:
        a, b = _ab()
        assert kurtosis_dw(a, b, 1.0) == pytest.approx(kurtosis_dw(a, b, 50.0), rel=1e-6)

    def test_reproducible(self) -> None:
        a, b = _ab()
        assert kurtosis_dw(a, b) == kurtosis_dw(a, b)

    def test_zero_update_returns_none(self) -> None:
        a, b = _ab()
        assert kurtosis_dw(a, np.zeros_like(b)) is None
