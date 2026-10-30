"""Shared feature-vector construction for ESR family reference profiles.

One source of truth for the transform and the fixed per-adapter vector, used by
both the offline profile builders (``benchmarks/build_profile.py``,
``benchmarks/build_calibration.py``) and the online loader
(``scoring/reference.py``). Keeping the layout in one place is what guarantees
the vector scored at scan time is the same vector the conformal calibration was
fit on; a silent drift between the two would void the FPR guarantee.

Pure numpy; no torch, no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

# ESR features and the transform applied before detrending, mirroring
# ``detectors/intra_adapter.py``: log for strictly positive heavy-tailed
# quantities, identity otherwise. A module missing a feature is skipped for that
# feature only.
FEATURE_TRANSFORM: dict[str, str] = {
    "log_rms": "id",
    "top1_e": "id",
    "srank_r": "id",
    "spec_h": "id",
    "excess_conc": "id",
    "kurt_dw": "log",
    "row_hoyer": "id",
    "row_top8": "log",
    "col_hoyer": "id",
}


def transform_value(value: float | None, kind: str) -> float | None:
    """Apply a feature's transform; ``None`` when the value cannot be used."""
    if value is None or not math.isfinite(value):
        return None
    if kind == "log":
        return math.log(value) if value > 0.0 else None
    return value


def transformed_values(source: Mapping[str, Any] | Any) -> dict[str, float]:
    """Transformed features of one module, from a dict or a ``ModuleFeatures``."""
    get = source.get if isinstance(source, Mapping) else lambda k: getattr(source, k, None)
    out: dict[str, float] = {}
    for feat, kind in FEATURE_TRANSFORM.items():
        t = transform_value(get(feat), kind)
        if t is not None:
            out[feat] = t
    return out


def adapter_vector(
    modules: Iterable[tuple[str, float, Mapping[str, float]]],
    log_rank: float,
    profile: Mapping[str, Any],
) -> np.ndarray:
    """Fixed ``[median z, p90 |z|]`` per feature for one adapter.

    ``modules`` is an iterable of ``(module_name, rel_depth, transformed_values)``.
    Each module's feature is standardised against the family trend
    ``z = (x - (c0 + c1·rel_depth + c2·log_rank) - resid_median) / resid_scale``;
    a feature the profile could not fit, or the adapter does not carry, is
    skipped. A feature seen in no module becomes 0 (treated as typical).
    """
    feat_names = [str(f) for f in profile["features"]]
    m_index = {str(m): i for i, m in enumerate(profile["modules"])}
    coef, med, scale = profile["coef"], profile["resid_med"], profile["resid_scale"]

    modules = list(modules)
    vec = np.zeros(2 * len(feat_names), dtype=np.float64)
    for fi, feat in enumerate(feat_names):
        zs: list[float] = []
        for name, rel, vals in modules:
            mi = m_index.get(str(name))
            if mi is None or feat not in vals:
                continue
            c, s = coef[mi, fi], scale[mi, fi]
            if not np.all(np.isfinite(c)) or not s > 0.0:
                continue
            pred = c[0] + c[1] * rel + c[2] * log_rank
            zs.append((vals[feat] - pred - med[mi, fi]) / s)
        if zs:
            vec[2 * fi] = float(np.median(zs))
            vec[2 * fi + 1] = float(np.percentile(np.abs(zs), 90))
    return vec


def mahalanobis_sq(X: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of each row of ``X`` to ``mu`` under ``prec``."""
    d = X - mu
    return np.einsum("ij,jk,ik->i", d, prec, d)
