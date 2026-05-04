"""Layer-level anomaly detection rules.

Applies M1 per-layer rules to a single LoRA (A, B) matrix pair, returning
a list of anomaly flag strings.  Each flag encodes the rule name, the
observed value, the threshold, and contextual information for triage.

Security Notes:
    - Pure computation on pre-loaded numpy arrays; no I/O.
    - High-risk module detection uses exact string membership, not pattern
      matching, to avoid false-negatives from partial matches.
"""

from __future__ import annotations

from typing import Any

# Thresholds matching the README "Known Anomaly Patterns" table
_KURTOSIS_THRESHOLD = 10.0
_ENERGY_CONCENTRATION_THRESHOLD = 0.95
_HIGH_RISK_TARGET_MODULES = {"embed_tokens", "lm_head"}
_MAX_SAFE_METADATA_DEPTH = 5


def detect_layer_anomalies(
    layer_name: str,
    stats_A: dict[str, float],
    stats_B: dict[str, float],
    svd: dict[str, Any],
    claimed_rank: int | None,
) -> list[str]:
    """Apply M1 anomaly rules to a single LoRA layer pair.

    Args:
        layer_name: Canonical layer identifier (checked for high-risk modules).
        stats_A: Stats dict for the lora_A matrix.
        stats_B: Stats dict for the lora_B matrix.
        svd: SVD stats dict from compute_svd_stats applied to lora_A.
        claimed_rank: Rank r declared in adapter metadata, or None.

    Returns:
        List of anomaly flag strings (empty list = no anomalies detected).
    """
    flags: list[str] = []

    if stats_A["kurtosis"] > _KURTOSIS_THRESHOLD:
        flags.append(
            f"HIGH_KURTOSIS_A: {stats_A['kurtosis']:.2f} > {_KURTOSIS_THRESHOLD}"
            " (heavy-tailed weights, possible sparse malicious injection)"
        )
    if stats_B["kurtosis"] > _KURTOSIS_THRESHOLD:
        flags.append(
            f"HIGH_KURTOSIS_B: {stats_B['kurtosis']:.2f} > {_KURTOSIS_THRESHOLD}"
        )

    if svd["energy_concentration"] > _ENERGY_CONCENTRATION_THRESHOLD:
        flags.append(
            f"HIGH_ENERGY_CONCENTRATION: {svd['energy_concentration']:.4f}"
            " (single dominant direction — potential backdoor trigger)"
        )

    if claimed_rank is not None and svd["effective_rank"] > claimed_rank * 2:
        flags.append(
            f"RANK_INFLATION: effective_rank={svd['effective_rank']}"
            f" vs claimed_rank={claimed_rank} (possible rank inflation attack)"
        )

    if stats_B["std"] < 1e-6:
        flags.append("NEAR_ZERO_B_MATRIX: lora_B weights near zero (untrained adapter?)")

    for module in _HIGH_RISK_TARGET_MODULES:
        if module in layer_name:
            flags.append(f"HIGH_RISK_TARGET_MODULE: {module} found in layer path")

    return flags
