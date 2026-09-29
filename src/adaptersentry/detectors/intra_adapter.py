"""Intra-adapter comparison: which modules differ from their own family.

No reference profile is needed. A 70B adapter has ~80 modules of each kind
(q_proj, down_proj, …); they should look alike, apart from a smooth trend with
depth. A module that departs from its family is suspicious.

Per family (same module name) and per feature:
  1. if layer indices are known, remove the depth trend with a Theil–Sen line
     (median of pairwise slopes; insensitive to the outliers we look for);
  2. robust z of the residuals: (x − median) / (1.4826 · MAD), with the
     mean-absolute-deviation fallback when MAD = 0.

Thresholds (UNCALIBRATED — to be fitted on the benign reference corpus):
    OUTLIER_Z = 3.5   module reported as an outlier (Iglewicz–Hoaglin)
    REVIEW_Z  = 8.0   strong enough to raise the verdict to "review"
A large adapter yields thousands of z-values, so a few exceed 3.5 by chance;
REVIEW_Z is set far above that level for this reason.

Security Notes:
    - Pure numpy/scipy on feature values; no I/O.
    - Degenerate families (fewer than MIN_FAMILY modules, zero spread) are
      skipped and never produce spurious z-values.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import theilslopes

from adaptersentry.schemas.result import (
    Contributor,
    DepthSpike,
    IntraAnomaly,
    ModuleFeatures,
)

OUTLIER_Z = 3.5
REVIEW_Z = 8.0
MIN_FAMILY = 5
MIN_DETREND = 6
_MAD_TO_SIGMA = 1.4826
_MEANAD_TO_SIGMA = 1.2533

# Features compared within a family, with the transform applied first.
# log for strictly positive heavy-tailed quantities; identity otherwise.
_FEATURES: dict[str, str] = {
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


@dataclass(frozen=True)
class ModuleEntry:
    """One analysed module as seen by the intra-adapter comparison."""

    index: int                 # position in the scan's module list
    module: str                # family key, e.g. "down_proj"
    kind: str                  # attention | mlp | …
    layer: int | None
    expert: int | None
    features: ModuleFeatures
    energy: float              # ‖ΔW‖²_F, for concentration across modules


@dataclass
class IntraResult:
    """Summary plus per-module z-values (index → {feature: z})."""

    summary: IntraAnomaly
    z_by_module: dict[int, dict[str, float]] = field(default_factory=dict)
    top_contributors: list[Contributor] = field(default_factory=list)


def robust_z(x: np.ndarray) -> np.ndarray | None:
    """Robust z-scores; None when the spread is zero (degenerate family)."""
    med = float(np.median(x))
    dev = np.abs(x - med)
    mad = float(np.median(dev))
    if mad > 0.0:
        return (x - med) / (_MAD_TO_SIGMA * mad)
    mean_ad = float(np.mean(dev))
    if mean_ad > 0.0:
        return (x - med) / (_MEANAD_TO_SIGMA * mean_ad)
    return None


def detrend(layers: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residuals of x after removing a Theil–Sen line over depth."""
    if np.unique(layers).size < 2:
        return x - float(np.median(x))
    with warnings.catch_warnings():
        # theilslopes also computes a confidence interval we do not use; on
        # constant data that emits a harmless sqrt-of-negative warning.
        warnings.simplefilter("ignore", RuntimeWarning)
        slope, intercept, _, _ = theilslopes(x, layers)
    return x - (intercept + slope * layers)


def _transform(value: float | None, kind: str) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    if kind == "log":
        return math.log(value) if value > 0.0 else None
    return value


def _gini(values: np.ndarray) -> float | None:
    n = values.size
    total = float(values.sum())
    if n < 2 or total <= 0.0:
        return None
    s = np.sort(values)
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(np.sum((2.0 * i - n - 1.0) * s) / (n * total))


def compare_within_adapter(entries: list[ModuleEntry], *, top_k: int = 10) -> IntraResult | None:
    """Run the intra-adapter comparison.

    Returns:
        IntraResult, or None if no family has at least MIN_FAMILY modules.
    """
    families: dict[str, list[ModuleEntry]] = defaultdict(list)
    for e in entries:
        families[e.module].append(e)

    z_by_module: dict[int, dict[str, float]] = defaultdict(dict)
    contributions: list[tuple[float, Contributor]] = []
    spikes: list[DepthSpike] = []
    dispersion: dict[str, float] = {}
    compared = False

    for name, members in families.items():
        if len(members) < MIN_FAMILY:
            continue
        compared = True
        log_rms = np.array([m.features.log_rms for m in members if m.features.log_rms is not None])
        if log_rms.size >= MIN_FAMILY:
            dispersion[name] = float(np.median(np.abs(log_rms - np.median(log_rms))))

        for feat, kind in _FEATURES.items():
            pts = [(m, _transform(getattr(m.features, feat), kind)) for m in members]
            pts = [(m, v) for m, v in pts if v is not None]
            if len(pts) < MIN_FAMILY:
                continue
            x = np.array([v for _, v in pts], dtype=np.float64)
            with_layer = all(m.layer is not None for m, _ in pts)
            if with_layer and len(pts) >= MIN_DETREND:
                layers = np.array([m.layer for m, _ in pts], dtype=np.float64)
                x = detrend(layers, x)
            z = robust_z(x)
            if z is None:
                continue
            for (m, raw), zi in zip(pts, z):
                z_by_module[m.index][feat] = float(zi)
                contributions.append((abs(float(zi)), Contributor(
                    feature=feat, module_type=name, layer=m.layer, expert=m.expert,
                    value=float(getattr(m.features, feat)), z=float(zi),
                )))

    if not compared:
        return None

    per_module_max: dict[int, float] = {
        idx: max(abs(v) for v in zs.values()) for idx, zs in z_by_module.items() if zs
    }
    by_index = {e.index: e for e in entries}
    n_outliers = sum(1 for v in per_module_max.values() if v > OUTLIER_Z)
    for idx, v in sorted(per_module_max.items(), key=lambda kv: -kv[1]):
        e = by_index[idx]
        if v > OUTLIER_Z and e.layer is not None and len(spikes) < top_k:
            signed = max(z_by_module[idx].values(), key=abs)
            spikes.append(DepthSpike(module_type=e.module, layer=e.layer, z=float(signed)))

    energies = np.array([e.energy for e in entries if e.energy > 0.0], dtype=np.float64)
    attn = [e.features.log_rms for e in entries if e.kind == "attention" and e.features.log_rms is not None]
    mlp = [e.features.log_rms for e in entries if e.kind == "mlp" and e.features.log_rms is not None]
    contributions.sort(key=lambda c: -c[0])

    summary = IntraAnomaly(
        max_robust_z=max(per_module_max.values(), default=0.0),
        n_outlier_modules=n_outliers,
        depth_spikes=spikes,
        energy_gini=_gini(energies),
        logrms_dispersion=dispersion,
        attn_mlp_ratio=float(np.median(attn) - np.median(mlp)) if attn and mlp else None,
    )
    return IntraResult(
        summary=summary,
        z_by_module=dict(z_by_module),
        top_contributors=[c for _, c in contributions[:top_k]],
    )
