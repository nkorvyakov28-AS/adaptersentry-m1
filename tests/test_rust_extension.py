"""Rust extension safety on hostile inputs (skipped when the extension is not built)."""

from __future__ import annotations

import numpy as np
import pytest

rs = pytest.importorskip("adaptersentry_rs")


def _hostile() -> np.ndarray:
    return np.array([1.0, np.nan, 3.0, np.inf, -2.0, -np.inf, 0.5] * 1000, dtype=np.float32)


def test_tensor_stats_with_nan_does_not_panic() -> None:
    stats = rs.tensor_stats_f32(_hostile())
    median, p01, p99 = stats[4], stats[5], stats[6]
    assert all(np.isfinite([median, p01, p99]))


def test_percentiles_clamp_out_of_range_q() -> None:
    x = np.arange(100, dtype=np.float32)
    assert rs.percentiles_f32(x, [-50.0, 500.0]) == [0.0, 99.0]


def test_percentiles_all_nan() -> None:
    x = np.full(16, np.nan, dtype=np.float32)
    assert rs.percentiles_f32(x, [50.0]) == [0.0]


def test_isolation_score_with_nan_does_not_panic() -> None:
    mean, rate, std = rs.isolation_score_1d(_hostile(), 0.55)
    assert 0.0 <= mean <= 1.0
