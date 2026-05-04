"""AnalyzerConfig — deterministic hash of the active detector configuration.

The analyzer_config_hash is the cache invalidation key for detector logic.
Any change to weights, thresholds, enabled families, schema version, or tool
version produces a different hash, automatically invalidating stale cache entries.

This is deliberately conservative: bumping the adaptersentry version always
changes analyzer_config_hash. This prevents silent false-negative regressions
from serving cached results produced by an older, less capable analyzer.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from adaptersentry.scoring.ensemble import DETECTOR_WEIGHTS


class ScanMode(str, Enum):
    """Scan depth mode — controls the trade-off between detection coverage and speed.

    FULL  All detectors run at full depth. Full SVD on every matrix. IsolationForest
          on every layer. No sampling. Use for security audits, suspicious adapters,
          and final verification before deployment.

    FAST  Optimised for throughput screening of large corpora. Three changes vs FULL:
          1. Truncated SVD (top-50 singular values) for matrices >= 512×512.
          2. IsolationForest skipped for tensors with > 5M elements.
          3. Statistical moments computed on a 50K-element random sample for
             tensors with > 100K elements (deterministic seed — reproducible).
          Detection quality for typical backdoor patterns is preserved: energy
          concentration and kurtosis signals live in the top singular values and
          in the tail of the distribution, both of which sampling captures.
    """

    FULL = "full"
    FAST = "fast"


# Current schema and tool versions — bumped here on release
_SCHEMA_VERSION = "1.0.0"
_TOOL_VERSION: str  # resolved lazily to avoid circular import


def _get_tool_version() -> str:
    global _TOOL_VERSION
    try:
        _TOOL_VERSION
    except NameError:
        from adaptersentry.version import __version__
        _TOOL_VERSION = __version__
    return _TOOL_VERSION


# Default detection thresholds (mirror the values in ensemble.py and layer_stats.py)
_DEFAULT_THRESHOLDS: dict[str, float] = {
    "kurtosis_anomaly":     10.0,
    "energy_concentration": 0.95,
    "isolation_anomaly":    -0.1,
    "entropy_low":          0.10,
    "entropy_high":         0.99,
    "zscore_outlier_rate":  0.02,
    "delta_norm_ratio_high": 0.90,
    "delta_kurtosis_high":   6.0,
    "delta_entropy_low":     0.10,
}

_DEFAULT_ENABLED_FAMILIES: list[str] = [
    "norm", "distribution", "entropy", "outlier", "spectral",
]


class AnalyzerConfig(BaseModel):
    """Canonical representation of the active analyzer configuration.

    All fields that affect analysis output must be included here.
    The config_hash is derived from a canonical JSON dump of this model,
    so any field addition or value change produces a new hash.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_version: str = _SCHEMA_VERSION
    tool_version: str = Field(default_factory=_get_tool_version)
    enabled_families: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_ENABLED_FAMILIES)
    )
    detector_weights: dict[str, float] = Field(
        default_factory=lambda: dict(DETECTOR_WEIGHTS)
    )
    thresholds: dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_THRESHOLDS)
    )
    scan_mode: ScanMode = ScanMode.FULL

    def config_hash(self) -> str:
        """Compute a deterministic SHA-256 hash of this config.

        The canonical form is a JSON object with keys sorted and floats
        rounded to 8 decimal places to avoid floating-point drift across
        Python versions.
        """
        canonical = _canonical_json(self.model_dump())
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return f"sha256:{digest}"


def _canonical_json(obj: Any) -> str:
    """Produce a deterministic JSON string suitable for hashing."""

    def _normalize(v: Any) -> Any:
        if isinstance(v, float):
            return round(v, 8)
        if isinstance(v, dict):
            return {k: _normalize(val) for k, val in sorted(v.items())}
        if isinstance(v, list):
            return [_normalize(item) for item in v]
        return v

    return json.dumps(_normalize(obj), sort_keys=True, separators=(",", ":"))


# Module-level default config — constructed once and reused
_default_config: AnalyzerConfig | None = None


def get_default_config() -> AnalyzerConfig:
    """Return the process-level default AnalyzerConfig (singleton)."""
    global _default_config
    if _default_config is None:
        _default_config = AnalyzerConfig()
    return _default_config
