"""Tests for adaptersentry.scoring.risk_scorer — RiskScorer class."""

from __future__ import annotations

import pytest

from adaptersentry.scoring.risk_scorer import DEFAULT_WEIGHTS, RiskScorer


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestRiskScorerConstruction:
    def test_default_weights_applied(self) -> None:
        scorer = RiskScorer()
        assert scorer.weights == DEFAULT_WEIGHTS

    def test_custom_weight_overrides_default(self) -> None:
        scorer = RiskScorer(weights={"RANK_INFLATION": 50})
        assert scorer.weights["RANK_INFLATION"] == 50

    def test_custom_weight_merged_with_defaults(self) -> None:
        scorer = RiskScorer(weights={"RANK_INFLATION": 50})
        # Other defaults untouched
        assert scorer.weights["HIGH_KURTOSIS"] == DEFAULT_WEIGHTS["HIGH_KURTOSIS"]

    def test_new_prefix_added(self) -> None:
        scorer = RiskScorer(weights={"CUSTOM_FLAG": 10})
        assert "CUSTOM_FLAG" in scorer.weights

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative int"):
            RiskScorer(weights={"RANK_INFLATION": -5})

    def test_float_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative int"):
            RiskScorer(weights={"RANK_INFLATION": 3.0})  # type: ignore[dict-item]

    def test_string_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative int"):
            RiskScorer(weights={"RANK_INFLATION": "30"})  # type: ignore[dict-item]

    def test_zero_weight_allowed(self) -> None:
        # Zero effectively disables a flag
        scorer = RiskScorer(weights={"RANK_INFLATION": 0})
        assert scorer.weights["RANK_INFLATION"] == 0

    def test_weights_property_returns_copy(self) -> None:
        scorer = RiskScorer()
        w = scorer.weights
        w["RANK_INFLATION"] = 999
        # Mutation of the returned dict does not affect scorer
        assert scorer.weights["RANK_INFLATION"] == DEFAULT_WEIGHTS["RANK_INFLATION"]


# ---------------------------------------------------------------------------
# score_flags
# ---------------------------------------------------------------------------


class TestScoreFlags:
    def test_empty_flags_zero(self) -> None:
        assert RiskScorer().score_flags([]) == 0

    def test_single_known_flag(self) -> None:
        scorer = RiskScorer()
        score = scorer.score_flags(["RANK_INFLATION: effective_rank=256 vs claimed=8"])
        assert score == DEFAULT_WEIGHTS["RANK_INFLATION"]

    def test_capped_at_100(self) -> None:
        flags = ["RANK_INFLATION: x"] * 20  # 20 × 30 = 600
        assert RiskScorer().score_flags(flags) == 100

    def test_unknown_prefix_ignored(self) -> None:
        assert RiskScorer().score_flags(["TOTALLY_UNKNOWN_FLAG: xyz"]) == 0

    def test_multiple_different_flags_accumulate(self) -> None:
        scorer = RiskScorer()
        flags = [
            "RANK_INFLATION: ...",
            "HIGH_ENERGY_CONCENTRATION: ...",
        ]
        expected = min(
            DEFAULT_WEIGHTS["RANK_INFLATION"] + DEFAULT_WEIGHTS["HIGH_ENERGY_CONCENTRATION"],
            100,
        )
        assert scorer.score_flags(flags) == expected

    def test_same_prefix_multiple_times_accumulates(self) -> None:
        scorer = RiskScorer()
        flags = ["HIGH_KURTOSIS_A: 15.0", "HIGH_KURTOSIS_B: 12.0"]
        expected = min(DEFAULT_WEIGHTS["HIGH_KURTOSIS"] * 2, 100)
        assert scorer.score_flags(flags) == expected

    def test_custom_weight_used_in_scoring(self) -> None:
        scorer = RiskScorer(weights={"RANK_INFLATION": 5})
        score = scorer.score_flags(["RANK_INFLATION: ..."])
        assert score == 5

    def test_new_detector_flags_scored(self) -> None:
        scorer = RiskScorer()
        score = scorer.score_flags(["LOW_ENTROPY_A: entropy=0.02 ..."])
        assert score == DEFAULT_WEIGHTS["LOW_ENTROPY"]

    def test_isolation_anomaly_flag_scored(self) -> None:
        scorer = RiskScorer()
        score = scorer.score_flags(["HIGH_ISOLATION_ANOMALY_A: mean_score=-0.25 ..."])
        assert score == DEFAULT_WEIGHTS["HIGH_ISOLATION_ANOMALY"]

    def test_zscore_outlier_flag_scored(self) -> None:
        scorer = RiskScorer()
        score = scorer.score_flags(["HIGH_ZSCORE_OUTLIER_RATE_B: rate=0.05 ..."])
        assert score == DEFAULT_WEIGHTS["HIGH_ZSCORE_OUTLIER_RATE"]


# ---------------------------------------------------------------------------
# risk_level
# ---------------------------------------------------------------------------


class TestRiskLevel:
    def test_zero_is_low(self) -> None:
        assert RiskScorer().risk_level(0) == "LOW"

    def test_24_is_low(self) -> None:
        assert RiskScorer().risk_level(24) == "LOW"

    def test_25_is_medium(self) -> None:
        assert RiskScorer().risk_level(25) == "MEDIUM"

    def test_49_is_medium(self) -> None:
        assert RiskScorer().risk_level(49) == "MEDIUM"

    def test_50_is_high(self) -> None:
        assert RiskScorer().risk_level(50) == "HIGH"

    def test_74_is_high(self) -> None:
        assert RiskScorer().risk_level(74) == "HIGH"

    def test_75_is_critical(self) -> None:
        assert RiskScorer().risk_level(75) == "CRITICAL"

    def test_100_is_critical(self) -> None:
        assert RiskScorer().risk_level(100) == "CRITICAL"


# ---------------------------------------------------------------------------
# score_report
# ---------------------------------------------------------------------------


class TestScoreReport:
    def _empty_report(self) -> dict:
        return {
            "flags": [],
            "layers": {},
        }

    def _report_with_flags(self, flags: list[str]) -> dict:
        return {"flags": flags, "layers": {}}

    def test_empty_report_zero_risk(self) -> None:
        score, level = RiskScorer().score_report(self._empty_report())
        assert score == 0
        assert level == "LOW"

    def test_returns_tuple_int_str(self) -> None:
        result = RiskScorer().score_report(self._empty_report())
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], str)

    def test_uses_global_flags_field(self) -> None:
        report = self._report_with_flags(["RANK_INFLATION: x"])
        score, _ = RiskScorer().score_report(report)
        assert score == DEFAULT_WEIGHTS["RANK_INFLATION"]

    def test_missing_flags_key_treated_as_empty(self) -> None:
        score, level = RiskScorer().score_report({})
        assert score == 0
        assert level == "LOW"

    def test_critical_report(self) -> None:
        # 30 + 30 + 20 + 25 = 105 → capped at 100, CRITICAL
        report = self._report_with_flags([
            "RANK_INFLATION: x",
            "HIGH_ENERGY_CONCENTRATION: x",
            "HIGH_KURTOSIS_A: x",
            "HIGH_RISK_TARGET_MODULE: embed_tokens",
        ])
        score, level = RiskScorer().score_report(report)
        assert level == "CRITICAL"
        assert score == 100

    def test_level_matches_score(self) -> None:
        report = self._report_with_flags(["NEAR_ZERO_B_MATRIX: x"])
        score, level = RiskScorer().score_report(report)
        expected_level = RiskScorer().risk_level(score)
        assert level == expected_level
