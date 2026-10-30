"""Calibrate a family reference into a conformal verdict (ESR stage C2).

Takes the trend/scale profile from ``build_profile.py`` and turns it into a
one-class anomaly score with a controlled false-positive rate:

  1. Per adapter, standardise every module's features against the family trend
     (z = (transform(x) - predicted - resid_median) / resid_scale) and summarise
     into a fixed vector: for each feature, the adapter's median z (global shift)
     and the 90th percentile of |z| (does it carry a few outlier modules). Two
     numbers per feature catch both ways a poisoned adapter leaves the norm.
  2. Fit a robust centre and a Ledoit-Wolf covariance on a *fit* split, and score
     every other adapter by squared Mahalanobis distance.
  3. Split-conformal p-value against a disjoint *reference* split:
        p = (1 + #{D2_ref >= D2_new}) / (n_ref + 1)
     which bounds the benign flag rate at alpha by construction (exchangeability).
     block if p <= alpha_block, review if p <= alpha_review.

The reference split also gives a leave-one-out estimate of the empirical FPR, so
we can see whether it lands near alpha before any poisoned corpus exists. A clean
FPR at alpha = 0.01 needs n_ref >= 99 (qwen3 has it; smaller families only reach
alpha = 0.05).

numpy + scikit-learn (LedoitWolf); no torch, no I/O beyond the dump, profile and
the calibrated npz it writes.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import json
import sys

import numpy as np
from sklearn.covariance import LedoitWolf

# Reused from the profile builder; ensure its directory is importable under -I.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_profile import FEATURE_TRANSFORM, _transform, family_match  # noqa: E402

logger = logging.getLogger("build_calibration")


def load_adapters(dump: Path, family_key: str) -> list[dict]:
    """One record per adapter: repo, log_rank, and its modules' transformed features."""
    out: list[dict] = []
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
            rank = rec.get("rank")
            if not rank or rank <= 0:
                continue
            seen.add(repo)
            n_layers = rec.get("n_layers") or 0
            mods = []
            for m in rec.get("modules", []):
                layer = m.get("l")
                rel = (layer / (n_layers - 1)) if (layer is not None and n_layers > 1) else 0.0
                vals = {feat: t for feat, kind in FEATURE_TRANSFORM.items()
                        if (t := _transform(m.get("f", {}).get(feat), kind)) is not None}
                if vals:
                    mods.append((m.get("m", "?"), rel, vals))
            if mods:
                out.append({"repo": repo, "log_rank": math.log(float(rank)), "modules": mods})
    return out


def adapter_vector(adapter: dict, profile: dict) -> np.ndarray:
    """Fixed [median z, p90 |z|] per feature; missing feature -> 0 (treated as typical)."""
    modules = list(profile["modules"])
    features = list(profile["features"])
    m_index = {m: i for i, m in enumerate(modules)}
    coef, med, scale = profile["coef"], profile["resid_med"], profile["resid_scale"]

    vec = np.zeros(2 * len(features), dtype=np.float64)
    for fi, feat in enumerate(features):
        zs: list[float] = []
        for name, rel, vals in adapter["modules"]:
            mi = m_index.get(name)
            if mi is None or feat not in vals:
                continue
            c, s = coef[mi, fi], scale[mi, fi]
            if not np.all(np.isfinite(c)) or not s > 0.0:
                continue
            pred = c[0] + c[1] * rel + c[2] * adapter["log_rank"]
            zs.append((vals[feat] - pred - med[mi, fi]) / s)
        if zs:
            az = np.abs(zs)
            vec[2 * fi] = float(np.median(zs))
            vec[2 * fi + 1] = float(np.percentile(az, 90))
    return vec


def mahalanobis_sq(X: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> np.ndarray:
    d = X - mu
    return np.einsum("ij,jk,ik->i", d, prec, d)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate an ESR family profile (conformal).")
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True, help="<family>.npz from build_profile.py")
    ap.add_argument("--out", type=Path, required=True, help="output <family>_calib.npz")
    ap.add_argument("--fit-frac", type=float, default=0.5, help="fraction for the centre/covariance fit")
    ap.add_argument("--alpha-review", type=float, default=0.05)
    ap.add_argument("--alpha-block", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    profile = dict(np.load(args.profile, allow_pickle=True))
    family = str(profile["family"])
    adapters = load_adapters(args.features, family)
    X = np.array([adapter_vector(a, profile) for a in adapters])
    n = len(adapters)
    logger.info("family %s: %d adapters, vector dim %d", family, n, X.shape[1])

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_fit = int(round(n * args.fit_frac))
    fit_idx, ref_idx = perm[:n_fit], perm[n_fit:]
    X_fit, X_ref = X[fit_idx], X[ref_idx]
    n_ref = len(ref_idx)

    mu = np.median(X_fit, axis=0)                     # robust centre
    lw = LedoitWolf().fit(X_fit - mu)                 # shrunk covariance, stable for p<=n
    prec = lw.precision_
    D2_ref = mahalanobis_sq(X_ref, mu, prec)

    # Leave-one-out conformal p-value on the reference split -> empirical FPR.
    order = np.sort(D2_ref)
    fpr: dict[float, float] = {}
    for alpha in (args.alpha_block, args.alpha_review, 0.10):
        flagged = 0
        for i in range(n_ref):
            others = np.delete(D2_ref, i)
            p = (1 + int(np.sum(others >= D2_ref[i]))) / n_ref
            flagged += int(p <= alpha)
        fpr[alpha] = flagged / n_ref

    min_p = 1.0 / (n_ref + 1)
    logger.info("reference split n_ref=%d  (min achievable p = %.4f)", n_ref, min_p)
    logger.info("empirical benign flag rate (target = alpha):")
    for alpha in (args.alpha_block, args.alpha_review, 0.10):
        reachable = "" if alpha >= min_p else "  [alpha below min p: need more benign adapters]"
        logger.info("  alpha=%.2f -> FPR=%.3f%s", alpha, fpr[alpha], reachable)
    logger.info("D2 reference: median=%.2f  p90=%.2f  p99=%.2f  max=%.2f",
                float(np.median(D2_ref)), float(np.percentile(D2_ref, 90)),
                float(np.percentile(D2_ref, 99)), float(order[-1]))

    profile.update({
        "cal_mu": mu,
        "cal_precision": prec,
        "cal_ref_scores": D2_ref,
        "cal_alpha_review": np.array(args.alpha_review),
        "cal_alpha_block": np.array(args.alpha_block),
        "cal_vector_layout": np.array("per-feature [median_z, p90_abs_z]"),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **profile)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
