"""Init-only adapter detector — separates LoRA initialisation artifacts from real anomalies.

Standard HuggingFace PEFT LoRA initialisation zeroes the B matrix so the adapter is a
mathematical no-op at start of training.  This produces systematic false positives:
  NEAR_ZERO_B_MATRIX  — B never trained from zero
  HIGH_ENTROPY_A/B    — A drawn from uniform-random at kaiming init
  LOW_ENTROPY_B       — B=0 exactly → entropy=0

This module detects the init-only pattern and allows the analyzer to suppress those
flags, preventing alert fatigue on CI/library test adapters.

Key design decision: BOTH conditions must hold simultaneously (B near-zero AND A
near-uniform), so a legitimately suspicious zero-B adapter with Gaussian A is NOT
suppressed.  Partial-training (some layers trained, others still at init) gets its
own SUSPICIOUS_PARTIAL_TRAINING flag rather than blanket suppression.

Security Notes:
    - Pure computation on already-loaded dicts — no I/O, no eval/exec/pickle.
    - Suppression is applied only when ALL layers satisfy the init-only criteria.
    - Partial-training emits SUSPICIOUS_PARTIAL_TRAINING — an attacker hiding
      malicious weights in a handful of layers while keeping others at init would
      trigger this.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Both conditions required simultaneously for a layer to qualify as init-only.
_B_STD_THRESHOLD: float = 1e-6
_A_ENTROPY_THRESHOLD: float = 0.98

# Flag prefixes suppressed when the adapter is classified INIT_ONLY.
SUPPRESSED_PREFIXES: tuple[str, ...] = (
    "NEAR_ZERO_B_MATRIX",
    "HIGH_ENTROPY_A",
    "HIGH_ENTROPY_B",
    "LOW_ENTROPY_B",
)


def is_init_only_adapter(layer_stats: dict) -> bool:
    """Return True if a single layer is in the LoRA zero-initialisation state.

    Both conditions must hold simultaneously:
      1. ``std_B < 1e-6``  — B matrix never updated from zero init
      2. ``entropy_A > 0.98`` — A matrix drawn from uniform-random distribution

    Args:
        layer_stats: Per-layer report dict from ``analyzer.analyze()``.

    Returns:
        True if both init-only conditions are satisfied.
    """
    std_b = layer_stats.get("std_B")
    entropy_a = layer_stats.get("entropy_A")

    if std_b is None or entropy_a is None:
        return False

    return float(std_b) < _B_STD_THRESHOLD and float(entropy_a) > _A_ENTROPY_THRESHOLD


def get_adapter_training_status(layer_reports: dict) -> str:
    """Classify the adapter's overall training state from per-layer statistics.

    Args:
        layer_reports: Dict mapping layer name to per-layer report dict.

    Returns:
        ``"TRAINED"``, ``"INIT_ONLY"``, or ``"PARTIALLY_TRAINED"``.
    """
    if not layer_reports:
        return "TRAINED"

    init_flags = [is_init_only_adapter(stats) for stats in layer_reports.values()]
    n_init = sum(init_flags)
    n_total = len(init_flags)

    if n_init == n_total:
        logger.debug("Adapter classified INIT_ONLY: all %d layers at init state", n_total)
        return "INIT_ONLY"
    if n_init == 0:
        return "TRAINED"

    logger.debug(
        "Adapter classified PARTIALLY_TRAINED: %d/%d layers at init state",
        n_init, n_total,
    )
    return "PARTIALLY_TRAINED"


def suppress_init_flags(flags: list[str]) -> tuple[list[str], int]:
    """Remove init-only artifact flags from a flag list.

    Args:
        flags: List of anomaly flag strings.

    Returns:
        Tuple of (kept_flags, suppressed_count).
    """
    kept: list[str] = []
    suppressed = 0
    for flag in flags:
        if any(flag.startswith(prefix) for prefix in SUPPRESSED_PREFIXES):
            suppressed += 1
        else:
            kept.append(flag)
    return kept, suppressed
