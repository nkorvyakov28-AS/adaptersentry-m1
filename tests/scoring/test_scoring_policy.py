"""Tests for M1-SCORE-02: ScoringPolicy, apply_policy, escalation rules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.schemas.score_breakdown import ScoreBreakdown, SubScore
from adaptersentry.schemas.scoring_policy import (
    EscalationCondition,
    EscalationRule,
    ScoringPolicy,
)
from adaptersentry.scoring.score_breakdown import (
    _FAMILY_ORDER,
    _FAMILY_WEIGHTS,
    apply_policy,
    compute_score_breakdown,
    get_default_policy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sub_scores(values: dict[str, float]) -> list[SubScore]:
    """Build a minimal list of SubScores for policy testing."""
    return [
        SubScore(
            family=f,
            raw_score=values.get(f, 0.0),
            normalized_score=values.get(f, 0.0),
            weight=_FAMILY_WEIGHTS[f],
            weighted_contribution=values.get(f, 0.0) * _FAMILY_WEIGHTS[f],
        )
        for f in _FAMILY_ORDER
    ]


def _make_adapter_file(tmp_path: Path, n_layers: int = 4, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {}
    for i in range(n_layers):
        tensors[f"model.layers.{i}.q_proj.lora_A.weight"] = \
            rng.standard_normal((4, 32)).astype(np.float32)
        tensors[f"model.layers.{i}.q_proj.lora_B.weight"] = \
            rng.standard_normal((32, 4)).astype(np.float32)
    p = tmp_path / "adapter.safetensors"
    save_file(tensors, str(p))
    return p


def _make_report(tmp_path: Path, n_layers: int = 4, seed: int = 0):
    from adaptersentry.analyzer import scan
    return scan(_make_adapter_file(tmp_path, n_layers=n_layers, seed=seed))


# ---------------------------------------------------------------------------
# ScoringPolicy schema
# ---------------------------------------------------------------------------

class TestScoringPolicySchema:
    def test_default_policy_version(self):
        p = get_default_policy()
        assert p.version == "1.0.0"

    def test_default_policy_weights_sum_to_one(self):
        p = get_default_policy()
        assert abs(sum(p.family_weights.values()) - 1.0) < 1e-9

    def test_default_policy_covers_all_families(self):
        p = get_default_policy()
        assert set(p.family_weights.keys()) == set(_FAMILY_ORDER)

    def test_default_policy_has_escalation_rules(self):
        p = get_default_policy()
        assert len(p.escalation_rules) > 0

    def test_default_policy_is_frozen(self):
        p = get_default_policy()
        with pytest.raises(Exception):
            p.version = "2.0.0"  # type: ignore[misc]

    def test_escalation_rule_is_frozen(self):
        rule = get_default_policy().escalation_rules[0]
        with pytest.raises(Exception):
            rule.score_bump = 0.99  # type: ignore[misc]

    def test_custom_policy_construction(self):
        policy = ScoringPolicy(
            version="custom-1.0",
            family_weights=dict(_FAMILY_WEIGHTS),
            family_caps={"distribution": 0.8},
            family_floors={"metadata": 0.1},
        )
        assert policy.version == "custom-1.0"
        assert policy.family_caps["distribution"] == 0.8
        assert policy.family_floors["metadata"] == 0.1

    def test_get_default_policy_cached(self):
        p1 = get_default_policy()
        p2 = get_default_policy()
        assert p1 is p2  # same object


# ---------------------------------------------------------------------------
# SubScore new fields
# ---------------------------------------------------------------------------

class TestSubScoreNewFields:
    def test_cap_applied_default_false(self):
        s = SubScore(
            family="parse", raw_score=0.1, normalized_score=0.1,
            weight=0.1, weighted_contribution=0.01,
        )
        assert s.cap_applied is False

    def test_floor_applied_default_false(self):
        s = SubScore(
            family="parse", raw_score=0.1, normalized_score=0.1,
            weight=0.1, weighted_contribution=0.01,
        )
        assert s.floor_applied is False


# ---------------------------------------------------------------------------
# apply_policy — caps
# ---------------------------------------------------------------------------

class TestApplyPolicyCaps:
    def test_cap_limits_normalized_score(self):
        sub_scores = _make_sub_scores({"distribution": 0.9})
        policy = ScoringPolicy(
            version="test",
            family_weights=dict(_FAMILY_WEIGHTS),
            family_caps={"distribution": 0.5},
        )
        adjusted, _, _ = apply_policy(sub_scores, policy)
        dist = next(s for s in adjusted if s.family == "distribution")
        assert dist.normalized_score == pytest.approx(0.5)
        assert dist.cap_applied is True

    def test_cap_not_applied_when_below_cap(self):
        sub_scores = _make_sub_scores({"distribution": 0.3})
        policy = ScoringPolicy(
            version="test",
            family_weights=dict(_FAMILY_WEIGHTS),
            family_caps={"distribution": 0.5},
        )
        adjusted, _, _ = apply_policy(sub_scores, policy)
        dist = next(s for s in adjusted if s.family == "distribution")
        assert dist.normalized_score == pytest.approx(0.3)
        assert dist.cap_applied is False

    def test_weighted_contribution_recalculated_after_cap(self):
        sub_scores = _make_sub_scores({"distribution": 0.9})
        policy = ScoringPolicy(
            version="test",
            family_weights=dict(_FAMILY_WEIGHTS),
            family_caps={"distribution": 0.5},
        )
        adjusted, _, _ = apply_policy(sub_scores, policy)
        dist = next(s for s in adjusted if s.family == "distribution")
        assert dist.weighted_contribution == pytest.approx(0.5 * _FAMILY_WEIGHTS["distribution"])


# ---------------------------------------------------------------------------
# apply_policy — floors
# ---------------------------------------------------------------------------

class TestApplyPolicyFloors:
    def test_floor_raises_normalized_score(self):
        sub_scores = _make_sub_scores({"metadata": 0.0})
        policy = ScoringPolicy(
            version="test",
            family_weights=dict(_FAMILY_WEIGHTS),
            family_floors={"metadata": 0.15},
        )
        adjusted, _, _ = apply_policy(sub_scores, policy)
        meta = next(s for s in adjusted if s.family == "metadata")
        assert meta.normalized_score == pytest.approx(0.15)
        assert meta.floor_applied is True

    def test_floor_not_applied_when_already_above(self):
        sub_scores = _make_sub_scores({"metadata": 0.4})
        policy = ScoringPolicy(
            version="test",
            family_weights=dict(_FAMILY_WEIGHTS),
            family_floors={"metadata": 0.1},
        )
        adjusted, _, _ = apply_policy(sub_scores, policy)
        meta = next(s for s in adjusted if s.family == "metadata")
        assert meta.normalized_score == pytest.approx(0.4)
        assert meta.floor_applied is False


# ---------------------------------------------------------------------------
# apply_policy — escalation rules
# ---------------------------------------------------------------------------

class TestEscalationRules:
    def _make_rule(self, name, conditions, bump):
        return EscalationRule(
            name=name, description="test rule",
            conditions=conditions, score_bump=bump,
            reason=f"{name} fired",
        )

    def test_rule_fires_when_conditions_met(self):
        sub_scores = _make_sub_scores({"distribution": 0.8, "training_pattern": 0.6})
        rule = self._make_rule(
            "TEST_RULE",
            [
                EscalationCondition(family="distribution", operator="gt", threshold=0.5),
                EscalationCondition(family="training_pattern", operator="gt", threshold=0.5),
            ],
            0.15,
        )
        policy = ScoringPolicy(
            version="test",
            family_weights=dict(_FAMILY_WEIGHTS),
            escalation_rules=[rule],
        )
        _, fired, bump = apply_policy(sub_scores, policy)
        assert any("TEST_RULE" in r for r in fired)
        assert bump == pytest.approx(0.15)

    def test_rule_does_not_fire_when_condition_unmet(self):
        sub_scores = _make_sub_scores({"distribution": 0.3})
        rule = self._make_rule(
            "TEST_RULE",
            [EscalationCondition(family="distribution", operator="gt", threshold=0.5)],
            0.15,
        )
        policy = ScoringPolicy(
            version="test",
            family_weights=dict(_FAMILY_WEIGHTS),
            escalation_rules=[rule],
        )
        _, fired, bump = apply_policy(sub_scores, policy)
        assert len(fired) == 0
        assert bump == pytest.approx(0.0)

    def test_multiple_rules_all_fire(self):
        sub_scores = _make_sub_scores({
            "distribution": 0.9, "training_pattern": 0.9, "similarity": 0.9,
        })
        rules = [
            self._make_rule("R1", [EscalationCondition(family="distribution", operator="gt", threshold=0.5)], 0.1),
            self._make_rule("R2", [EscalationCondition(family="similarity", operator="gt", threshold=0.5)], 0.1),
        ]
        policy = ScoringPolicy(version="test", family_weights=dict(_FAMILY_WEIGHTS), escalation_rules=rules)
        _, fired, bump = apply_policy(sub_scores, policy)
        assert len(fired) == 2
        assert bump == pytest.approx(0.2)

    def test_all_conditions_must_be_true(self):
        # Rule requires BOTH conditions; only one is met
        sub_scores = _make_sub_scores({"distribution": 0.9, "metadata": 0.1})
        rule = self._make_rule(
            "TEST_RULE",
            [
                EscalationCondition(family="distribution", operator="gt", threshold=0.5),
                EscalationCondition(family="metadata", operator="gt", threshold=0.5),  # NOT met
            ],
            0.15,
        )
        policy = ScoringPolicy(version="test", family_weights=dict(_FAMILY_WEIGHTS), escalation_rules=[rule])
        _, fired, bump = apply_policy(sub_scores, policy)
        assert len(fired) == 0
        assert bump == pytest.approx(0.0)

    def test_lte_operator(self):
        sub_scores = _make_sub_scores({"parse": 0.0})
        rule = self._make_rule(
            "CLEAN_PARSE",
            [EscalationCondition(family="parse", operator="lte", threshold=0.01)],
            0.0,
        )
        policy = ScoringPolicy(version="test", family_weights=dict(_FAMILY_WEIGHTS), escalation_rules=[rule])
        _, fired, bump = apply_policy(sub_scores, policy)
        assert any("CLEAN_PARSE" in r for r in fired)

    def test_gte_operator(self):
        sub_scores = _make_sub_scores({"metadata": 0.4})
        rule = self._make_rule(
            "TEST",
            [EscalationCondition(family="metadata", operator="gte", threshold=0.4)],
            0.05,
        )
        policy = ScoringPolicy(version="test", family_weights=dict(_FAMILY_WEIGHTS), escalation_rules=[rule])
        _, fired, bump = apply_policy(sub_scores, policy)
        assert len(fired) == 1


# ---------------------------------------------------------------------------
# ScoreBreakdown new fields
# ---------------------------------------------------------------------------

class TestScoreBreakdownPolicyFields:
    def test_policy_version_stored(self, tmp_path):
        report = _make_report(tmp_path)
        result = compute_score_breakdown(report, policy=get_default_policy())
        assert result.applied_policy_version == "1.0.0"

    def test_adjusted_total_gte_total(self, tmp_path):
        # adjusted_total = total + adjustment (bump is non-negative)
        report = _make_report(tmp_path)
        result = compute_score_breakdown(report, policy=get_default_policy())
        assert result.adjusted_total_weighted >= result.total_weighted - 1e-9

    def test_adjusted_total_in_bounds(self, tmp_path):
        report = _make_report(tmp_path)
        result = compute_score_breakdown(report, policy=get_default_policy())
        assert 0.0 <= result.adjusted_total_weighted <= 1.0

    def test_score_adjustment_nonneg(self, tmp_path):
        report = _make_report(tmp_path)
        result = compute_score_breakdown(report, policy=get_default_policy())
        assert result.score_adjustment >= 0.0

    def test_escalation_rules_fired_is_list(self, tmp_path):
        report = _make_report(tmp_path)
        result = compute_score_breakdown(report, policy=get_default_policy())
        assert isinstance(result.escalation_rules_fired, list)

    def test_no_policy_uses_default(self, tmp_path):
        report = _make_report(tmp_path)
        result = compute_score_breakdown(report)  # no policy arg
        assert result.applied_policy_version == "1.0.0"

    def test_custom_policy_version_stored(self, tmp_path):
        report = _make_report(tmp_path)
        custom = ScoringPolicy(version="test-99", family_weights=dict(_FAMILY_WEIGHTS))
        result = compute_score_breakdown(report, policy=custom)
        assert result.applied_policy_version == "test-99"

    def test_worker_produces_policy_fields(self, tmp_path):
        from adaptersentry.engine.schemas.requests import AdapterScanRequest, ArtifactSource
        from adaptersentry.engine.worker import worker_main

        p = _make_adapter_file(tmp_path)
        req = AdapterScanRequest(
            request_id="sha256:" + "b" * 64,
            run_id="test",
            adapter_path=str(p),
            source=ArtifactSource(kind="local_path", local_path=str(p)),
        )
        result, _ = worker_main(req, analyzer_config_hash="hash0000")
        sb = result.ensemble.score_breakdown
        assert sb is not None
        assert sb.applied_policy_version == "1.0.0"
        assert isinstance(sb.escalation_rules_fired, list)
        assert isinstance(sb.adjusted_total_weighted, float)
