"""Tests for the intra-adapter comparison (robust z within module families)."""

from __future__ import annotations

import numpy as np
import pytest

from adaptersentry.detectors.intra_adapter import (
    MIN_FAMILY,
    OUTLIER_Z,
    REVIEW_Z,
    ModuleEntry,
    compare_within_adapter,
    detrend,
    robust_z,
)
from adaptersentry.features.spectral_v2 import module_features
from adaptersentry.schemas.result import ModuleFeatures


def _adapter_entries(
    n_layers: int = 24,
    *,
    inject_layer: int | None = None,
    seed: int = 0,
    d: int = 512,
    k: int = 384,
    r: int = 8,
) -> list[ModuleEntry]:
    """Realistic-looking synthetic adapter: two families, smooth growth with depth."""
    rng = np.random.default_rng(seed)
    entries: list[ModuleEntry] = []
    idx = 0
    for layer in range(n_layers):
        growth = 1.0 + 0.04 * layer  # update strength rises smoothly with depth
        for module, kind in (("q_proj", "attention"), ("down_proj", "mlp")):
            a = rng.standard_normal((r, k)).astype(np.float32)
            decay = np.exp(-np.arange(r) / 3.0)
            b = (rng.standard_normal((d, r)) * 0.01 * growth * decay).astype(np.float32)
            if module == "down_proj" and layer == inject_layer:
                spike = np.zeros(d, np.float32)
                spike[rng.choice(d, 6, replace=False)] = 0.25 * growth
                b[:, -1] = spike  # one rank slot writes into 6 outputs only
            feats, core = module_features(a, b, 2.0)
            entries.append(ModuleEntry(
                index=idx, module=module, kind=kind, layer=layer, expert=None,
                features=feats, energy=core.fro ** 2,
            ))
            idx += 1
    return entries


class TestHelpers:
    def test_robust_z_ignores_outlier_in_scale(self) -> None:
        z = robust_z(np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 6.0]))
        assert z is not None and z[-1] > 50 and abs(z[0]) < 1

    def test_robust_z_degenerate(self) -> None:
        assert robust_z(np.ones(8)) is None

    def test_robust_z_mad_zero_fallback(self) -> None:
        z = robust_z(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 5.0]))
        assert z is not None and z[-1] > 0

    def test_detrend_removes_line_not_spike(self) -> None:
        layers = np.arange(12, dtype=float)
        x = 1.0 + 0.1 * layers
        x[6] += 0.35
        res = detrend(layers, x)
        res = res - np.median(res)  # Theil–Sen intercept is median-based; z-scores re-centre anyway
        assert abs(res[0]) < 1e-9 and res[6] == pytest.approx(0.35, abs=1e-9)


class TestComparison:
    def test_clean_adapter_does_not_reach_review(self) -> None:
        for seed in range(3):
            result = compare_within_adapter(_adapter_entries(seed=seed))
            assert result is not None
            assert result.summary.max_robust_z < REVIEW_Z, seed

    def test_injected_layer_is_found(self) -> None:
        result = compare_within_adapter(_adapter_entries(inject_layer=13))
        assert result is not None
        assert result.summary.max_robust_z >= REVIEW_Z
        top = result.summary.depth_spikes[0]
        assert (top.module_type, top.layer) == ("down_proj", 13)
        assert result.top_contributors[0].layer == 13

    def test_depth_trend_alone_is_not_an_outlier(self) -> None:
        # Strong smooth growth with depth: detrending must absorb it.
        result = compare_within_adapter(_adapter_entries(n_layers=40, seed=5))
        assert result is not None
        assert all(abs(s.z) < REVIEW_Z for s in result.summary.depth_spikes)

    def test_per_module_z_recorded(self) -> None:
        entries = _adapter_entries(inject_layer=5)
        result = compare_within_adapter(entries)
        assert result is not None
        target = next(e.index for e in entries if e.module == "down_proj" and e.layer == 5)
        assert max(abs(v) for v in result.z_by_module[target].values()) >= REVIEW_Z

    def test_summary_fields(self) -> None:
        result = compare_within_adapter(_adapter_entries())
        assert result is not None
        s = result.summary
        assert 0.0 <= s.energy_gini <= 1.0
        assert set(s.logrms_dispersion) == {"q_proj", "down_proj"}
        assert s.attn_mlp_ratio is not None
        assert len(result.top_contributors) <= 10

    def test_small_adapter_has_no_comparison(self) -> None:
        assert compare_within_adapter(_adapter_entries(n_layers=MIN_FAMILY - 1)) is None

    def test_zero_updates_are_skipped_not_flagged(self) -> None:
        entries = _adapter_entries(n_layers=10)
        empty = [
            ModuleEntry(index=100 + i, module="v_proj", kind="attention", layer=i, expert=None,
                        features=ModuleFeatures(), energy=0.0)
            for i in range(10)
        ]
        result = compare_within_adapter(entries + empty)
        assert result is not None and "v_proj" not in result.summary.logrms_dispersion

    def test_moe_experts_share_family(self) -> None:
        base = _adapter_entries(n_layers=4)
        experts = [
            ModuleEntry(index=500 + i, module="gate_proj", kind="mlp", layer=i // 4, expert=i % 4,
                        features=base[0].features, energy=1.0)
            for i in range(16)
        ]
        result = compare_within_adapter(experts)
        assert result is None or result.summary.max_robust_z < OUTLIER_Z
