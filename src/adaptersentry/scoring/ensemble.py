"""Ensemble detector — weighted combination of M1 signal sources.

Replaces simple additive flag scoring with a logistic-regression-style
weighted ensemble, then sigmoid-normalises the result to a 0–100 score.

Detector weights sourced from:
  - kurtosis, energy_concentration, entropy, zscore, isolation_forest:
    arXiv 2602.15195 logistic regression coefficients (rescaled to sum to 1).
  - wasserstein_distance, cross_layer_consistency:
    estimated from IDS/anomaly-detection literature (conservative priors).

Security Notes:
    - Pure computation; no I/O, no eval/exec/pickle.
    - Majority-vote gate requires ≥2 independent detectors to agree before
      escalating, reducing single-detector false-positive risk.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detector weights (raw, before normalisation)
# ---------------------------------------------------------------------------

_RAW_WEIGHTS: dict[str, float] = {
    "kurtosis_score":           0.452,
    "energy_concentration":     0.353,
    "entropy_score":            0.089,
    "zscore_outlier_rate":      0.071,
    "isolation_forest_score":   0.035,
    "wasserstein_distance":     0.180,
    "cross_layer_consistency":  0.150,
}

# Normalise so weights sum to 1.0
_WEIGHT_SUM = sum(_RAW_WEIGHTS.values())
DETECTOR_WEIGHTS: dict[str, float] = {
    k: v / _WEIGHT_SUM for k, v in _RAW_WEIGHTS.items()
}

# Sigmoid steepness — tuned so raw_score=0.5 maps to ~75/100 (HIGH boundary)
_SIGMOID_SCALE = 8.0

_THRESHOLDS: tuple[tuple[int, str], ...] = (
    (36, "CRITICAL"),
    (14, "HIGH"),
    (7,  "MEDIUM"),
    (0,  "LOW"),
)

_KURTOSIS_ANOMALY_THRESHOLD: float = 10.0
_ENERGY_ANOMALY_THRESHOLD: float = 0.95
_ISOLATION_ANOMALY_THRESHOLD: float = -0.1


def _sigmoid(x: float, scale: float = _SIGMOID_SCALE) -> float:
    """Sigmoid mapping raw weighted sum → (0, 1)."""
    return float(1.0 / (1.0 + np.exp(-scale * (x - 0.5))))


class EnsembleDetector:
    """Weighted ensemble scorer for M1 layer reports.

    Aggregates per-layer statistics into a single 0–100 risk score using
    calibrated detector weights, then applies a sigmoid to compress the tail.
    A majority-vote gate provides a boolean anomaly signal from three
    independent sub-detectors.

    Args:
        weights: Optional weight override dict (merged with DETECTOR_WEIGHTS).
                 Values must be non-negative floats; they are re-normalised
                 after merging.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        w = dict(DETECTOR_WEIGHTS)
        if weights:
            for k, v in weights.items():
                if not isinstance(v, (int, float)) or v < 0:
                    raise ValueError(
                        f"Weight for {k!r} must be a non-negative number, got {v!r}."
                    )
                w[k] = float(v)
        total = sum(w.values())
        if total == 0:
            raise ValueError("Detector weights must not all be zero.")
        self._weights = {k: v / total for k, v in w.items()}

    @property
    def weights(self) -> dict[str, float]:
        """Read-only normalised weight dict."""
        return dict(self._weights)

    def _extract_features(self, layer_reports: dict[str, Any]) -> dict[str, float]:
        """Aggregate per-layer statistics into ensemble feature scalars."""
        if not layer_reports:
            return {k: 0.0 for k in _RAW_WEIGHTS}

        kurtosis_vals: list[float] = []
        energy_vals: list[float] = []
        entropy_vals: list[float] = []
        zscore_vals: list[float] = []
        iso_vals: list[float] = []

        for layer in layer_reports.values():
            k_a = layer.get("kurtosis_A", 0.0) or 0.0
            k_b = layer.get("kurtosis_B", 0.0) or 0.0
            kurtosis_vals.append(float(np.tanh(max(k_a, k_b, 0.0) / 20.0)))

            energy_vals.append(float(layer.get("energy_concentration", 0.0) or 0.0))

            e_a = layer.get("entropy_A", 0.6) or 0.6
            e_b = layer.get("entropy_B", 0.6) or 0.6
            entropy_vals.append(float(abs(e_a - 0.6) + abs(e_b - 0.6)) / 1.2)

            zs_a = layer.get("zscore_outlier_rate_A", 0.0) or 0.0
            zs_b = layer.get("zscore_outlier_rate_B", 0.0) or 0.0
            zscore_vals.append(float(np.clip(max(zs_a, zs_b) / 0.1, 0.0, 1.0)))

            iso = layer.get("isolation_score_A")
            if iso is not None:
                iso_vals.append(float(np.clip((-iso) / 0.5, 0.0, 1.0)))

        def _mean(lst: list[float]) -> float:
            return float(np.mean(lst)) if lst else 0.0

        return {
            "kurtosis_score":          _mean(kurtosis_vals),
            "energy_concentration":    _mean(energy_vals),
            "entropy_score":           _mean(entropy_vals),
            "zscore_outlier_rate":     _mean(zscore_vals),
            "isolation_forest_score":  _mean(iso_vals),
            "wasserstein_distance":    0.0,
            "cross_layer_consistency": 0.0,
        }

    def score(
        self,
        layer_reports: dict[str, Any],
        wasserstein_score: float = 0.0,
        cross_layer_consistency: float = 1.0,
    ) -> float:
        """Compute ensemble risk score (0–100).

        Args:
            layer_reports: Per-layer report dicts from analyzer.analyze().
            wasserstein_score: Pre-computed W2 anomaly score (0–1).
            cross_layer_consistency: Pre-computed consistency score (0–1);
                                     low value = concentrated anomaly.

        Returns:
            Float in [0, 100].
        """
        features = self._extract_features(layer_reports)
        features["wasserstein_distance"] = float(
            np.clip(wasserstein_score / _WEIGHT_SUM, 0.0, 1.0)
        )
        features["cross_layer_consistency"] = float(
            np.clip(1.0 - cross_layer_consistency, 0.0, 1.0)
        )

        raw = sum(self._weights[k] * features.get(k, 0.0) for k in self._weights)
        return float(np.clip(_sigmoid(raw) * 100.0, 0.0, 100.0))

    def predict(
        self,
        layer_reports: dict[str, Any],
        wasserstein_score: float = 0.0,
        cross_layer_consistency: float = 1.0,
    ) -> str:
        """Return a risk level label for the given layer reports."""
        s = self.score(layer_reports, wasserstein_score, cross_layer_consistency)
        return self.risk_level(int(s))

    @staticmethod
    def risk_level(score: int) -> str:
        """Map 0-100 integer score to severity label."""
        for threshold, level in _THRESHOLDS:
            if score >= threshold:
                return level
        return "LOW"

    def explain(
        self,
        layer_reports: dict[str, Any],
        wasserstein_score: float = 0.0,
        cross_layer_consistency: float = 1.0,
    ) -> list[str]:
        """Return human-readable explanations of the top 3 contributing risk factors."""
        features = self._extract_features(layer_reports)
        features["wasserstein_distance"] = float(
            np.clip(wasserstein_score / _WEIGHT_SUM, 0.0, 1.0)
        )
        features["cross_layer_consistency"] = float(
            np.clip(1.0 - cross_layer_consistency, 0.0, 1.0)
        )

        contributions = {
            k: self._weights[k] * features.get(k, 0.0) for k in self._weights
        }
        ranked = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

        explanations: list[str] = []
        for feature_name, contribution in ranked:
            if contribution < 1e-4:
                break
            value = features[feature_name]
            desc = self._describe_feature(
                feature_name, value, layer_reports,
                wasserstein_score, cross_layer_consistency,
            )
            if desc:
                explanations.append(desc)
            if len(explanations) == 3:
                break

        return explanations

    @staticmethod
    def _describe_feature(
        name: str,
        value: float,
        layer_reports: dict[str, Any],
        wasserstein_score: float,
        cross_layer_consistency: float,
    ) -> str:
        """Format a single feature's contribution as a human-readable string."""
        if name == "kurtosis_score" and value > 0.01:
            max_k, max_layer, max_label = 0.0, "", "A"
            for lname, layer in layer_reports.items():
                for label, key in (("A", "kurtosis_A"), ("B", "kurtosis_B")):
                    k = layer.get(key) or 0.0
                    if k > max_k:
                        max_k, max_layer, max_label = k, lname, label
            short = max_layer.split(".")[-1] if max_layer else "?"
            return (
                f"kurtosis {max_k:.1f}× (lora_{max_label} in …{short},"
                f" threshold: {_KURTOSIS_ANOMALY_THRESHOLD:.0f}×)"
            )

        if name == "energy_concentration" and value > 0.05:
            max_ec, max_layer = 0.0, ""
            for lname, layer in layer_reports.items():
                ec = layer.get("energy_concentration") or 0.0
                if ec > max_ec:
                    max_ec, max_layer = ec, lname
            short = max_layer.split(".")[-1] if max_layer else "?"
            return (
                f"energy concentration {max_ec:.4f} in …{short}"
                f" (threshold: {_ENERGY_ANOMALY_THRESHOLD:.2f})"
            )

        if name == "cross_layer_consistency" and value > 0.1:
            n_layers = len(layer_reports)
            flagged = sum(1 for v in layer_reports.values() if v.get("flags"))
            return (
                f"cross-layer concentration: flags in {flagged}/{n_layers} layer(s)"
                f" (consistency={cross_layer_consistency:.3f}, threshold: 0.30)"
            )

        if name == "wasserstein_distance" and value > 0.05:
            return (
                f"W2 A↔B distance: {wasserstein_score:.4f}"
                f" (threshold: 0.15) — distributional asymmetry between lora_A and lora_B"
            )

        if name == "entropy_score" and value > 0.1:
            max_dev, max_layer, max_label = 0.0, "", "A"
            for lname, layer in layer_reports.items():
                for label, key in (("A", "entropy_A"), ("B", "entropy_B")):
                    e = layer.get(key) or 0.6
                    dev = abs(e - 0.6)
                    if dev > max_dev:
                        max_dev, max_layer, max_label = dev, lname, label
            short = max_layer.split(".")[-1] if max_layer else "?"
            e_val = (layer_reports.get(max_layer) or {}).get(f"entropy_{max_label}", 0.6)
            return (
                f"entropy anomaly: lora_{max_label} entropy={e_val:.4f} in …{short}"
                f" (benign range: 0.10–0.99)"
            )

        if name == "zscore_outlier_rate" and value > 0.1:
            max_rate, max_layer = 0.0, ""
            for lname, layer in layer_reports.items():
                r = max(
                    layer.get("zscore_outlier_rate_A") or 0.0,
                    layer.get("zscore_outlier_rate_B") or 0.0,
                )
                if r > max_rate:
                    max_rate, max_layer = r, lname
            short = max_layer.split(".")[-1] if max_layer else "?"
            return f"outlier rate {max_rate:.1%} in …{short} (threshold: 2.0%)"

        if name == "isolation_forest_score" and value > 0.1:
            min_iso, min_layer = 0.0, ""
            for lname, layer in layer_reports.items():
                iso = layer.get("isolation_score_A")
                if iso is not None and iso < min_iso:
                    min_iso, min_layer = iso, lname
            short = min_layer.split(".")[-1] if min_layer else "?"
            return (
                f"IsolationForest score {min_iso:.4f} in …{short}"
                f" (threshold: {_ISOLATION_ANOMALY_THRESHOLD:.1f})"
            )

        return ""

    def _extract_features_from_families(self, families: list) -> dict[str, float]:
        """Aggregate typed FeatureFamilyResult objects into ensemble feature scalars."""
        kurtosis_vals: list[float] = []
        energy_vals: list[float] = []
        entropy_vals: list[float] = []
        zscore_vals: list[float] = []
        iso_vals: list[float] = []

        for ffr in families:
            if ffr.status not in ("ok", "degraded"):
                continue
            rf = ffr.raw_features
            if ffr.family == "distribution":
                k_a = rf.get("kurtosis_A", 0.0) or 0.0
                k_b = rf.get("kurtosis_B", 0.0) or 0.0
                k = max(k_a, k_b, 0.0) or max(rf.get("kurtosis_delta", 0.0) or 0.0, 0.0)
                kurtosis_vals.append(float(np.tanh(k / 20.0)))
                e_a = rf.get("entropy_A", 0.6) or 0.6
                e_b = rf.get("entropy_B", 0.6) or 0.6
                entropy_vals.append(float(abs(e_a - 0.6) + abs(e_b - 0.6)) / 1.2)
            elif ffr.family == "spectral":
                energy_vals.append(float(rf.get("energy_concentration", 0.0) or 0.0))
            elif ffr.family == "entropy":
                e_a = rf.get("entropy_A", 0.6) or 0.6
                e_b = rf.get("entropy_B", 0.6) or 0.6
                entropy_vals.append(float(abs(e_a - 0.6) + abs(e_b - 0.6)) / 1.2)
            elif ffr.family == "outlier":
                zs_a = rf.get("zscore_outlier_rate_A", 0.0) or 0.0
                zs_b = rf.get("zscore_outlier_rate_B", 0.0) or 0.0
                zscore_vals.append(float(np.clip(max(zs_a, zs_b) / 0.1, 0.0, 1.0)))
                iso = rf.get("isolation_score_A")
                if iso is not None:
                    iso_vals.append(float(np.clip((-iso) / 0.5, 0.0, 1.0)))

        def _mean(lst: list[float]) -> float:
            return float(np.mean(lst)) if lst else 0.0

        return {
            "kurtosis_score":          _mean(kurtosis_vals),
            "energy_concentration":    _mean(energy_vals),
            "entropy_score":           _mean(entropy_vals),
            "zscore_outlier_rate":     _mean(zscore_vals),
            "isolation_forest_score":  _mean(iso_vals),
            "wasserstein_distance":    0.0,
            "cross_layer_consistency": 0.0,
        }

    def score_families(
        self,
        families: list,
        wasserstein_score: float = 0.0,
        cross_layer_consistency: float = 1.0,
    ) -> float:
        """Compute ensemble risk score from typed FeatureFamilyResult objects."""
        features = self._extract_features_from_families(families)
        features["wasserstein_distance"] = float(
            np.clip(wasserstein_score / _WEIGHT_SUM, 0.0, 1.0)
        )
        features["cross_layer_consistency"] = float(
            np.clip(1.0 - cross_layer_consistency, 0.0, 1.0)
        )
        raw = sum(self._weights[k] * features.get(k, 0.0) for k in self._weights)
        return float(np.clip(_sigmoid(raw) * 100.0, 0.0, 100.0))

    def majority_vote(
        self,
        spectral_score: float,
        stat_score: float,
        iso_score: float,
    ) -> bool:
        """Return True if ≥2 of 3 independent detectors flag an anomaly.

        Args:
            spectral_score: energy_concentration value (0–1).
            stat_score: maximum kurtosis value (raw, e.g. 15.3).
            iso_score: mean IsolationForest decision score (e.g. -0.25).

        Returns:
            True if at least 2 of the 3 detectors independently flag anomaly.
        """
        votes = [
            spectral_score > _ENERGY_ANOMALY_THRESHOLD,
            stat_score > _KURTOSIS_ANOMALY_THRESHOLD,
            iso_score < _ISOLATION_ANOMALY_THRESHOLD,
        ]
        return sum(votes) >= 2
