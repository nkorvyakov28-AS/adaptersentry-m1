"""Build a per-family reference profile from benign-adapter features (ESR stage C1).

The profile describes what a *normal* adapter looks like for one base-model
family, so a new adapter can be scored by how far it deviates (stage C2:
Mahalanobis + split-conformal, built on top of this file's output).

Why detrend by rank and depth
------------------------------
ESR shape features are already rank-normalised (``srank_r`` = stable rank / r,
etc.), yet two of them keep a strong monotonic trend with rank even so: on 198
benign qwen3 adapters the median ``srank_r`` fell 0.19 -> 0.02 and
``excess_conc`` rose 1.5 -> 3.3 from r<=8 to r>64. Pooling a family across ranks
without accounting for that would make a high-rank adapter look anomalous
against a low-rank-dominated norm. So, exactly as the intra-adapter detector
removes the depth trend within one adapter, the profile removes a (relative
depth, log-rank) trend across adapters and stores the residual median and MAD.
Pooling the residuals keeps the whole family as one stratum, which is what makes
the conformal sample large enough (qwen3: 198 >= 99 for alpha = 0.01).

Input: a JSONL feature dump, one adapter per line, as written by the Hub survey:
    {"repo", "family", "base", "n_layers", "rank",
     "modules": [{"m", "l", "e", "k", "f": {<feature>: value, ...}}, ...]}

Output: ``<out>/<family>.npz`` with the per-(module, feature) trend model and
residual location/scale, plus the held-out adapter ids for calibration.

Pure numpy; no I/O beyond reading the dump and writing the npz. No torch.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("build_profile")

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

_MAD_TO_SIGMA = 1.4826
# Need enough modules of a type to fit a 3-parameter trend and a stable MAD.
MIN_POINTS = 24


@dataclass(frozen=True)
class Point:
    """One module of one adapter, as a profile sample."""

    module: str          # family key, e.g. "q_proj"
    rel_depth: float     # layer / (n_layers - 1), in [0, 1]; 0.0 if unknown
    log_rank: float      # log(rank)
    values: dict[str, float]


def family_match(base: str | None, family_key: str) -> bool:
    """True when an adapter's base model belongs to the requested family."""
    return family_key.lower() in (base or "").lower()


def _transform(value: float | None, kind: str) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    if kind == "log":
        return math.log(value) if value > 0.0 else None
    return value


def load_points(dump: Path, family_key: str) -> tuple[list[Point], list[str]]:
    """Read the feature dump and return samples for one family plus its repo ids."""
    points: list[Point] = []
    repos: list[str] = []
    seen: set[str] = set()
    with dump.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            repo = rec.get("repo")
            if repo in seen or not family_match(rec.get("base"), family_key):
                continue
            seen.add(repo)
            rank = rec.get("rank")
            if not rank or rank <= 0:
                continue
            log_rank = math.log(float(rank))
            n_layers = rec.get("n_layers") or 0
            for m in rec.get("modules", []):
                layer = m.get("l")
                rel = (layer / (n_layers - 1)) if (layer is not None and n_layers > 1) else 0.0
                mf = m.get("f") or {}  # "f" may be present but null when a tensor was not featurised
                vals = {
                    feat: t for feat, kind in FEATURE_TRANSFORM.items()
                    if (t := _transform(mf.get(feat), kind)) is not None
                }
                if vals:
                    points.append(Point(module=m.get("m", "?"), rel_depth=rel,
                                        log_rank=log_rank, values=vals))
            repos.append(repo)
    return points, repos


def fit_trend(rel_depth: np.ndarray, log_rank: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients of y ~ 1 + rel_depth + log_rank."""
    design = np.column_stack([np.ones_like(y), rel_depth, log_rank])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def robust_scale(resid: np.ndarray) -> float:
    """MAD -> sigma, with a mean-absolute-deviation fallback when MAD = 0."""
    med = float(np.median(resid))
    dev = np.abs(resid - med)
    mad = float(np.median(dev))
    if mad > 0.0:
        return _MAD_TO_SIGMA * mad
    mean_ad = float(np.mean(dev))
    return 1.2533 * mean_ad if mean_ad > 0.0 else 0.0


def build(points: list[Point]) -> dict[str, object]:
    """Fit the per-(module, feature) trend and residual location/scale."""
    modules = sorted({p.module for p in points})
    features = list(FEATURE_TRANSFORM)
    n_m, n_f = len(modules), len(features)

    coef = np.full((n_m, n_f, 3), np.nan)
    resid_med = np.full((n_m, n_f), np.nan)
    resid_scale = np.full((n_m, n_f), np.nan)
    counts = np.zeros((n_m, n_f), dtype=np.int64)

    for mi, module in enumerate(modules):
        members = [p for p in points if p.module == module]
        for fi, feat in enumerate(features):
            sub = [(p.rel_depth, p.log_rank, p.values[feat]) for p in members if feat in p.values]
            counts[mi, fi] = len(sub)
            if len(sub) < MIN_POINTS:
                continue
            rd = np.array([s[0] for s in sub], dtype=np.float64)
            lr = np.array([s[1] for s in sub], dtype=np.float64)
            y = np.array([s[2] for s in sub], dtype=np.float64)
            c = fit_trend(rd, lr, y)
            resid = y - (c[0] + c[1] * rd + c[2] * lr)
            coef[mi, fi] = c
            resid_med[mi, fi] = float(np.median(resid))
            resid_scale[mi, fi] = robust_scale(resid)

    return {
        "modules": np.array(modules),
        "features": np.array(features),
        "coef": coef,
        "resid_med": resid_med,
        "resid_scale": resid_scale,
        "counts": counts,
    }


def diagnose_rank_detrend(points: list[Point], profile: dict[str, object]) -> None:
    """Confirm the detrend flattened the rank trend for the two trending features."""
    modules = list(profile["modules"])
    features = list(profile["features"])
    coef = profile["coef"]

    def bucket(lr: float) -> str:
        r = round(math.exp(lr))
        return "r<=8" if r <= 8 else "r16" if r <= 16 else "r32" if r <= 32 else \
               "r64" if r <= 64 else "r>64"

    order = ["r<=8", "r16", "r32", "r64", "r>64"]
    for feat in ("srank_r", "excess_conc"):
        if feat not in features:
            continue
        fi = features.index(feat)
        raw: dict[str, list[float]] = {b: [] for b in order}
        res: dict[str, list[float]] = {b: [] for b in order}
        for p in points:
            if feat not in p.values or p.module not in modules:
                continue
            c = coef[modules.index(p.module), fi]
            if not np.all(np.isfinite(c)):
                continue
            b = bucket(p.log_rank)
            pred = c[0] + c[1] * p.rel_depth + c[2] * p.log_rank
            raw[b].append(p.values[feat])
            res[b].append(p.values[feat] - pred)
        rowr = "  ".join(f"{b}={np.median(raw[b]):6.3f}" for b in order if raw[b])
        rowe = "  ".join(f"{b}={np.median(res[b]):+6.3f}" for b in order if res[b])
        logger.info("  %-12s raw median:  %s", feat, rowr)
        logger.info("  %-12s resid median:%s", feat, rowe)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an ESR family reference profile.")
    ap.add_argument("--features", type=Path, required=True, help="benign feature dump (JSONL)")
    ap.add_argument("--family", required=True, help="base-model family substring, e.g. qwen3")
    ap.add_argument("--out", type=Path, required=True, help="output directory for <family>.npz")
    ap.add_argument("--holdout", type=float, default=0.3, help="fraction of adapters reserved for calibration")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    points, repos = load_points(args.features, args.family)
    logger.info("family %s: %d adapters, %d module samples", args.family, len(repos), len(points))
    if len(repos) < 2 * MIN_POINTS:
        logger.warning("few adapters for %s; profile will be thin", args.family)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(repos))
    n_hold = int(round(len(repos) * args.holdout))
    hold = {repos[i] for i in perm[:n_hold]}
    # Trend/scale are fit on the fit split only; the held-out ids feed C2 calibration.
    # (Split by adapter, not by module, so no leakage across the two sets.)
    profile = build(points)
    profile["holdout_repos"] = np.array(sorted(hold))
    profile["family"] = np.array(args.family)
    profile["n_adapters"] = np.array(len(repos))

    logger.info("rank-detrend check (resid median should be flat across buckets):")
    diagnose_rank_detrend(points, profile)

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"{args.family}.npz"
    np.savez_compressed(dest, **profile)
    logger.info("wrote %s", dest)


if __name__ == "__main__":
    main()
