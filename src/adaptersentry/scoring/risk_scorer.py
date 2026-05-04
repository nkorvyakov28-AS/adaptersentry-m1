"""M1 Risk Scorer — aggregate anomaly flags into an overall adapter risk score.

Primary path: EnsembleDetector.score() — weighted, sigmoid-normalised ensemble
              using layer-level statistics (kurtosis, energy, entropy, etc.).
Fallback path: rule-based additive flag scoring (used when layer_reports are
              unavailable or EnsembleDetector import fails).

Security Notes:
    - Pure computation, no I/O — no attack surface.
    - Custom weights are validated (non-negative ints) on construction to prevent
      score manipulation via unexpected inputs.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: dict[str, int] = {
    # Core analyzer flags (from analyzer.py; see README.md "Known Anomaly Patterns")
    "RANK_INFLATION": 30,           # rank inflation attack — high confidence
    "HIGH_ENERGY_CONCENTRATION": 30,  # single dominant direction — backdoor trigger
    "HIGH_KURTOSIS": 20,            # heavy-tailed weights — sparse injection
    "HIGH_RISK_TARGET_MODULE": 25,  # embed_tokens / lm_head — critical paths
    "NEAR_ZERO_B_MATRIX": 15,       # untrained or zeroed B matrix — suspicious
    "METADATA_DEPTH": 20,           # deep nesting — evasion attempt
    # Entropy detector flags (from detectors/entropy.py)
    "LOW_ENTROPY": 15,              # constant/sparse weights — suspicious structure
    "HIGH_ENTROPY": 10,             # near-uniform — possible noise injection
    # Outlier detector flags (from detectors/outlier.py)
    "HIGH_ZSCORE_OUTLIER_RATE": 15,  # excess outlier weights — sparse injection
    "HIGH_ISOLATION_ANOMALY": 20,    # IsolationForest anomaly pattern
}

_LEVEL_THRESHOLDS: tuple[tuple[int, str], ...] = (
    (75, "CRITICAL"),
    (50, "HIGH"),
    (25, "MEDIUM"),
    (0, "LOW"),
)


class RiskScorer:
    """Aggregate anomaly flags into a 0–100 risk score with severity label.

    Scoring is additive: each flag whose prefix matches a weight entry
    contributes that weight. Multiple flags with the same prefix accumulate
    independently — more affected layers → higher risk. The total is capped at
    100.

    Args:
        weights: Optional override mapping from flag prefix to integer weight.
                 Merged with DEFAULT_WEIGHTS; missing keys keep their defaults.

    Raises:
        ValueError: If any override weight is not a non-negative integer.

    Security Notes:
        - Weights are validated on construction; non-integer or negative values
          raise ValueError to prevent score manipulation.
    """

    def __init__(self, weights: dict[str, int] | None = None) -> None:
        """Initialize with optional custom flag-weight overrides."""
        self._weights: dict[str, int] = dict(DEFAULT_WEIGHTS)
        if weights:
            for key, val in weights.items():
                if not isinstance(val, int) or val < 0:
                    raise ValueError(
                        f"Weight for flag prefix {key!r} must be a non-negative int,"
                        f" got {val!r}."
                    )
                self._weights[key] = val

    @property
    def weights(self) -> dict[str, int]:
        """Read-only view of current flag weights."""
        return dict(self._weights)

    def score_flags(self, flags: list[str]) -> int:
        """Derive a 0–100 risk score from a list of anomaly flag strings.

        Args:
            flags: List of anomaly flag strings (e.g., from analyzer + detectors).

        Returns:
            Integer risk score in [0, 100].
        """
        score = 0
        for flag in flags:
            for prefix, weight in self._weights.items():
                if flag.startswith(prefix):
                    score += weight
                    break
        return min(score, 100)

    def risk_level(self, score: int) -> str:
        """Map a numeric risk score to a severity label.

        Args:
            score: Integer in [0, 100].

        Returns:
            One of "LOW", "MEDIUM", "HIGH", "CRITICAL".
        """
        for threshold, level in _LEVEL_THRESHOLDS:
            if score >= threshold:
                return level
        return "LOW"

    def score_report(self, report: dict[str, Any]) -> tuple[int, str]:
        """Compute risk score from a full M1 analyzer report dict.

        Uses rule-based additive flag scoring for the schema-stable overall_risk
        field (backward-compatible). Callers that want the ensemble score should
        read the separate "ensemble_score" field produced by analyzer.analyze().

        Args:
            report: M1 JSON report dict conforming to the schema in README.md.

        Returns:
            Tuple of (overall_risk: int, risk_level: str).
        """
        all_flags: list[str] = list(report.get("flags", []))
        score = self.score_flags(all_flags)
        return score, self.risk_level(score)
