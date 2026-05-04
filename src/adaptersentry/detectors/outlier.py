"""Outlier detection for LoRA weight tensors.

Two complementary detectors — like running both signature-based and
anomaly-based IDS on the same traffic:

- Z-score:           fraction of weights beyond ±N sigma. Interpretable and
                     fast — like counting packets outside baseline thresholds.
- Isolation Forest:  ensemble anomaly scoring treating each weight as an
                     independent sample. Catches non-Gaussian outlier patterns
                     that Z-score misses.

A clean LoRA adapter should have <0.3% of weights beyond 3σ (Gaussian bound)
and an Isolation Forest mean score near zero or positive.

Security Notes:
    - No file I/O; pure numpy + sklearn computation — no attack surface.
    - Large tensors (>2000 weights) are subsampled for IsolationForest to
      bound runtime while preserving detection power.
    - sklearn IsolationForest is initialized with a fixed random_state to
      ensure reproducible scores.
    - Edge case: constant tensors (std == 0) return outlier_rate 0.0 — no
      division by zero.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# OPT-04: use Rust ECDF-based 1D isolation score when available (334× faster).
# Falls back to sklearn IsolationForest transparently if the extension is absent.
try:
    from adaptersentry_rs import isolation_score_1d as _rs_isolation_score_1d
    _RUST_ISOLATION_AVAILABLE = True
except ImportError:
    _RUST_ISOLATION_AVAILABLE = False

# Thresholds
_ZSCORE_SIGMA: float = 3.0          # standard deviations for outlier boundary
_ZSCORE_RATE_THRESHOLD: float = 0.02  # >2% of weights beyond 3σ → flag
_ISO_SCORE_THRESHOLD: float = -0.1   # mean decision score < -0.1 → flag
_ISO_MAX_SAMPLES: int = 2_000        # subsample cap for Isolation Forest
# 20 trees give equivalent anomaly detection on 2000 1-D samples; reduces
# per-call cost ~120ms → ~25ms without measurable detection loss.
_ISO_N_ESTIMATORS: int = 20
_ISO_RANDOM_STATE: int = 42
_ISO_CONTAMINATION: float = 0.1


def zscore_outlier_rate(
    tensor: np.ndarray,
    threshold: float = _ZSCORE_SIGMA,
) -> dict[str, float]:
    """Compute the fraction of tensor weights beyond ±threshold standard deviations.

    Args:
        tensor: Weight matrix of any shape; flattened internally.
        threshold: Number of standard deviations for the outlier boundary.

    Returns:
        Dict with keys:
            "outlier_rate"    — fraction of weights beyond ±threshold sigma.
            "threshold_sigma" — the sigma threshold used.

    Raises:
        ValueError: If tensor is empty.
    """
    flat = tensor.astype(np.float64).flatten()
    if flat.size == 0:
        raise ValueError("Cannot compute Z-score outlier rate for an empty tensor.")

    std = float(flat.std())
    if std == 0.0:
        return {"outlier_rate": 0.0, "threshold_sigma": threshold}

    z = np.abs((flat - float(flat.mean())) / std)
    outlier_rate = float((z > threshold).mean())
    return {"outlier_rate": outlier_rate, "threshold_sigma": threshold}


def isolation_forest_score(
    tensor: np.ndarray,
    n_estimators: int = _ISO_N_ESTIMATORS,
    random_state: int = _ISO_RANDOM_STATE,
    max_samples: int = _ISO_MAX_SAMPLES,
) -> dict[str, float]:
    """Compute isolation anomaly score over weight values.

    Each individual weight is treated as a 1-D sample.

    When the Rust extension (adaptersentry_rs) is available, uses an exact
    ECDF-based isolation score — the closed-form solution of IsolationForest
    for 1D data with infinite trees. This is 334× faster than sklearn and
    more accurate (no variance from finite tree count).

    Score convention (both paths):
        "mean_score" < 0 → anomalous  (< _ISO_SCORE_THRESHOLD → flag)
        "mean_score" ≥ 0 → normal

    Args:
        tensor:        Weight matrix of any shape; flattened internally.
        n_estimators:  Number of trees (sklearn fallback only; ignored for Rust).
        random_state:  Seed for reproducibility.
        max_samples:   Maximum number of weights to sample.

    Returns:
        Dict with "mean_score" (negative = anomalous) and "anomalous_fraction".

    Raises:
        ValueError: If tensor has fewer than 2 elements.
    """
    flat = tensor.astype(np.float32).flatten()
    if flat.size < 2:
        raise ValueError(
            f"Isolation score requires at least 2 samples; got {flat.size}."
        )

    if flat.size > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(flat.size, max_samples, replace=False)
        flat = flat[idx]
        logger.debug("Isolation score: subsampled %d → %d weights", flat.size, max_samples)

    if _RUST_ISOLATION_AVAILABLE:
        # ECDF-based exact 1D score (OPT-04).
        # rs_mean ∈ [0,1]: 0.5 = normal density, >0.5 = anomalous.
        # Map to sklearn convention: sklearn_score = 1.0 - 2.0 * rs_mean
        #   rs_mean=0.5 → 0.0 (normal);  rs_mean=0.6 → -0.2 (anomalous)
        # Anomalous fraction: points scoring > 0.55 (≈ sklearn score < 0).
        _RS_OUTLIER_THR = 0.55
        rs_mean, rs_outlier_rate, _ = _rs_isolation_score_1d(flat, _RS_OUTLIER_THR)
        return {
            "mean_score": 1.0 - 2.0 * rs_mean,
            "anomalous_fraction": rs_outlier_rate,
        }

    # Fallback: sklearn IsolationForest
    from sklearn.ensemble import IsolationForest
    data = flat.astype(np.float64).reshape(-1, 1)
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=_ISO_CONTAMINATION,
        random_state=random_state,
    )
    clf.fit(data)
    scores = clf.decision_function(data)
    return {
        "mean_score": float(scores.mean()),
        "anomalous_fraction": float((scores < 0).mean()),
    }


# In fast mode IsolationForest is always skipped — only z-score runs.
# Measured overhead: ~650ms per call (100 sklearn trees) regardless of sample size.
# For adapters with 24–224 layers that cost dominates all other per-layer work.
# Full mode retains IsolationForest for thorough anomaly detection.
_MAX_NUMEL_ISOLATION_FAST = 0  # unused; kept for schema compat


def detect_outlier_anomalies(
    tensor: np.ndarray,
    layer_name: str,
    matrix_label: str,
    run_isolation_forest: bool = True,
    *,
    fast: bool = False,
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Run Z-score and Isolation Forest detectors and return results + flags.

    Args:
        tensor: Weight matrix to analyze.
        layer_name: Layer identifier — included in flag context.
        matrix_label: "A" or "B" — which LoRA matrix was measured.
        run_isolation_forest: If False, skip IsolationForest (faster, reduced
                              detection). Default True.
        fast: If True, skip IsolationForest entirely (z-score only).

    Returns:
        Tuple of:
            zs_result  — dict from zscore_outlier_rate.
            iso_result — dict from isolation_forest_score (or empty dict).
            flags      — list of anomaly flag strings.
    """
    flags: list[str] = []

    zs = zscore_outlier_rate(tensor)
    if zs["outlier_rate"] > _ZSCORE_RATE_THRESHOLD:
        flags.append(
            f"HIGH_ZSCORE_OUTLIER_RATE_{matrix_label}:"
            f" rate={zs['outlier_rate']:.4f} > {_ZSCORE_RATE_THRESHOLD}"
            f" at {zs['threshold_sigma']:.0f}σ in {layer_name}"
            " (excess outlier weights — possible sparse injection)"
        )

    if fast:
        run_isolation_forest = False

    iso: dict[str, float] = {}
    if run_isolation_forest and tensor.size >= 2:
        iso = isolation_forest_score(tensor)
        if iso["mean_score"] < _ISO_SCORE_THRESHOLD:
            flags.append(
                f"HIGH_ISOLATION_ANOMALY_{matrix_label}:"
                f" mean_score={iso['mean_score']:.4f} < {_ISO_SCORE_THRESHOLD}"
                f" in {layer_name} (IsolationForest detects anomalous weight pattern)"
            )

    return zs, iso, flags
