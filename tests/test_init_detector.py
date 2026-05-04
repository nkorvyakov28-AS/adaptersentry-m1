"""Tests for adaptersentry.detectors.init_detector."""

from __future__ import annotations

import numpy as np
import pytest

from adaptersentry.detectors.init_detector import (
    SUPPRESSED_PREFIXES,
    _A_ENTROPY_THRESHOLD,
    _B_STD_THRESHOLD,
    get_adapter_training_status,
    is_init_only_adapter,
    suppress_init_flags,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layer(std_b: float, entropy_a: float, **extra) -> dict:
    """Minimal layer-stats dict for is_init_only_adapter."""
    return {"std_B": std_b, "entropy_A": entropy_a, **extra}


# ---------------------------------------------------------------------------
# is_init_only_adapter
# ---------------------------------------------------------------------------


class TestIsInitOnlyAdapter:
    def test_both_conditions_true_returns_true(self) -> None:
        # std_B near zero AND entropy_A above threshold
        layer = _layer(std_b=0.0, entropy_a=0.99)
        assert is_init_only_adapter(layer) is True

    def test_b_trained_returns_false(self) -> None:
        # B matrix has been updated (std_B >> 0) → not init-only
        layer = _layer(std_b=0.02, entropy_a=0.99)
        assert is_init_only_adapter(layer) is False

    def test_a_not_uniform_returns_false(self) -> None:
        # Gaussian-init A has entropy ≈ 0.8 < threshold → suspicious zero-B kept
        layer = _layer(std_b=0.0, entropy_a=0.80)
        assert is_init_only_adapter(layer) is False

    def test_both_conditions_false_returns_false(self) -> None:
        layer = _layer(std_b=0.05, entropy_a=0.70)
        assert is_init_only_adapter(layer) is False

    def test_std_b_exactly_at_threshold_returns_false(self) -> None:
        # Boundary: std_B == threshold → NOT init-only (< required)
        layer = _layer(std_b=_B_STD_THRESHOLD, entropy_a=0.99)
        assert is_init_only_adapter(layer) is False

    def test_entropy_exactly_at_threshold_returns_false(self) -> None:
        # Boundary: entropy_A == threshold → NOT init-only (> required)
        layer = _layer(std_b=0.0, entropy_a=_A_ENTROPY_THRESHOLD)
        assert is_init_only_adapter(layer) is False

    def test_missing_std_b_returns_false(self) -> None:
        assert is_init_only_adapter({"entropy_A": 0.99}) is False

    def test_missing_entropy_a_returns_false(self) -> None:
        assert is_init_only_adapter({"std_B": 0.0}) is False

    def test_empty_dict_returns_false(self) -> None:
        assert is_init_only_adapter({}) is False

    def test_extra_keys_ignored(self) -> None:
        layer = _layer(std_b=0.0, entropy_a=0.99, kurtosis_A=1.5, flags=[])
        assert is_init_only_adapter(layer) is True


# ---------------------------------------------------------------------------
# get_adapter_training_status
# ---------------------------------------------------------------------------


class TestGetAdapterTrainingStatus:
    def _init_layer(self) -> dict:
        return _layer(std_b=0.0, entropy_a=0.99)

    def _trained_layer(self) -> dict:
        return _layer(std_b=0.02, entropy_a=0.80)

    def test_all_init_returns_init_only(self) -> None:
        reports = {f"layer{i}": self._init_layer() for i in range(4)}
        assert get_adapter_training_status(reports) == "INIT_ONLY"

    def test_all_trained_returns_trained(self) -> None:
        reports = {f"layer{i}": self._trained_layer() for i in range(4)}
        assert get_adapter_training_status(reports) == "TRAINED"

    def test_mixed_returns_partially_trained(self) -> None:
        reports = {
            "layer0": self._init_layer(),
            "layer1": self._trained_layer(),
            "layer2": self._init_layer(),
            "layer3": self._trained_layer(),
        }
        assert get_adapter_training_status(reports) == "PARTIALLY_TRAINED"

    def test_empty_reports_returns_trained(self) -> None:
        assert get_adapter_training_status({}) == "TRAINED"

    def test_single_init_layer_returns_init_only(self) -> None:
        assert get_adapter_training_status({"only": self._init_layer()}) == "INIT_ONLY"

    def test_single_trained_layer_returns_trained(self) -> None:
        assert get_adapter_training_status({"only": self._trained_layer()}) == "TRAINED"

    def test_one_init_many_trained_partially_trained(self) -> None:
        reports = {f"layer{i}": self._trained_layer() for i in range(7)}
        reports["layer_bad"] = self._init_layer()
        assert get_adapter_training_status(reports) == "PARTIALLY_TRAINED"


# ---------------------------------------------------------------------------
# suppress_init_flags
# ---------------------------------------------------------------------------


class TestSuppressInitFlags:
    def test_near_zero_b_removed(self) -> None:
        flags = ["NEAR_ZERO_B_MATRIX: lora_B weights near zero (untrained adapter?)"]
        kept, n = suppress_init_flags(flags)
        assert kept == []
        assert n == 1

    def test_high_entropy_a_removed(self) -> None:
        flags = ["HIGH_ENTROPY_A: entropy=0.997 > 0.99 in layer.q_proj"]
        kept, n = suppress_init_flags(flags)
        assert kept == []
        assert n == 1

    def test_high_entropy_b_removed(self) -> None:
        flags = ["HIGH_ENTROPY_B: entropy=0.998 > 0.99 in layer.q_proj"]
        kept, n = suppress_init_flags(flags)
        assert kept == []
        assert n == 1

    def test_low_entropy_b_removed(self) -> None:
        flags = ["LOW_ENTROPY_B: entropy=0.0000 < 0.1 in layer.q_proj"]
        kept, n = suppress_init_flags(flags)
        assert kept == []
        assert n == 1

    def test_high_kurtosis_not_removed(self) -> None:
        flags = ["HIGH_KURTOSIS_A: 113.71 > 10.0 (heavy-tailed weights)"]
        kept, n = suppress_init_flags(flags)
        assert kept == flags
        assert n == 0

    def test_rank_inflation_not_removed(self) -> None:
        flags = ["RANK_INFLATION: effective_rank=256 vs claimed_rank=4"]
        kept, n = suppress_init_flags(flags)
        assert kept == flags
        assert n == 0

    def test_low_entropy_a_not_removed(self) -> None:
        # LOW_ENTROPY_A is suspicious even on init adapters (A should be high-entropy)
        flags = ["LOW_ENTROPY_A: entropy=0.02 < 0.1 in layer.q_proj"]
        kept, n = suppress_init_flags(flags)
        assert kept == flags
        assert n == 0

    def test_mixed_flags_partial_suppression(self) -> None:
        flags = [
            "NEAR_ZERO_B_MATRIX: ...",
            "HIGH_KURTOSIS_A: 113.7 ...",
            "HIGH_ENTROPY_A: ...",
            "RANK_INFLATION: ...",
        ]
        kept, n = suppress_init_flags(flags)
        assert n == 2
        assert len(kept) == 2
        assert all("KURTOSIS" in f or "RANK" in f for f in kept)

    def test_empty_flags_returns_empty(self) -> None:
        kept, n = suppress_init_flags([])
        assert kept == []
        assert n == 0

    def test_suppressed_count_accurate(self) -> None:
        flags = (
            ["NEAR_ZERO_B_MATRIX: x"] * 3
            + ["HIGH_ENTROPY_A: x"] * 2
            + ["LOW_ENTROPY_B: x"] * 1
            + ["HIGH_KURTOSIS_A: x"] * 4
        )
        _, n = suppress_init_flags(flags)
        assert n == 6

    def test_all_suppressed_prefixes_covered(self) -> None:
        """Every entry in SUPPRESSED_PREFIXES actually causes suppression."""
        for prefix in SUPPRESSED_PREFIXES:
            flags = [f"{prefix}: some detail"]
            kept, n = suppress_init_flags(flags)
            assert n == 1, f"Prefix {prefix!r} did not suppress its flag"
            assert kept == []
