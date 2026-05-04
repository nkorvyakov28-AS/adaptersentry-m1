"""ScoringPolicy — versioned weights, caps/floors, and escalation rules.

ScoringPolicy is the configuration object that controls how ScoreBreakdown
sub-scores are weighted and when escalation rules fire.

Separation of concerns
----------------------
ScoringPolicy defines HOW to score (weights, bounds, rules).
ScoreBreakdown records WHAT was scored (the computed results).
ScoringPolicy is config; ScoreBreakdown is data.

Escalation rules fire when a set of AND-conditions on sub-score
normalized_scores are all met. Each fired rule adds score_bump to
adjusted_total_weighted (clamped to [0, 1]).

Cap/floor
---------
Each family can have a cap (max normalized_score) and a floor (min
normalized_score). Applied before escalation rules. Useful for:
- Capping parse score at 0.5 when the file still loaded (degraded ≠ failed)
- Flooring metadata score at 0.1 to ensure it always has weight
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EscalationCondition(BaseModel):
    """A single condition in an escalation rule — checks one family's normalized_score."""

    model_config = ConfigDict(frozen=True)

    family: str = Field(description="Feature family whose normalized_score is evaluated.")
    operator: Literal["gt", "gte", "lt", "lte"] = Field(
        description="Comparison operator: gt (>), gte (>=), lt (<), lte (<=)."
    )
    threshold: float = Field(description="Comparison threshold in [0, 1].")


class EscalationRule(BaseModel):
    """An escalation rule: fires when ALL conditions are met, adds score_bump."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Stable identifier (e.g. 'PARTIALLY_TRAINED_HIGH_KURTOSIS').")
    description: str = Field(description="Human-readable explanation of what this rule detects.")
    conditions: list[EscalationCondition] = Field(
        description="All conditions must be True (AND logic) for the rule to fire."
    )
    score_bump: float = Field(
        ge=0.0, le=1.0,
        description="Amount added to adjusted_total_weighted when the rule fires.",
    )
    reason: str = Field(
        description="Human-readable reason string appended to escalation_rules_fired context.",
    )


class ScoringPolicy(BaseModel):
    """Versioned scoring policy — weights, caps, floors, and escalation rules.

    version
        Bumped on any change that alters numerical outputs. Stored in
        ScoreBreakdown.applied_policy_version for reproducibility.

    family_weights
        Per-family weight overrides. Must sum to 1.0 (validated at use time,
        not here — allows partial override dicts before merging with defaults).

    family_caps / family_floors
        Per-family bounds on normalized_score, applied before escalation rules.
        Keys are family names; absent families use 1.0 / 0.0 respectively.

    escalation_rules
        Ordered list of rules evaluated after caps/floors. All matching rules
        fire; their score_bumps are summed and added to adjusted_total_weighted.
    """

    model_config = ConfigDict(frozen=True)

    version: str = Field(default="1.0.0", description="Policy version string.")
    family_weights: dict[str, float] = Field(
        description="Per-family weights. Applied to normalized_score for weighted_contribution.",
    )
    family_caps: dict[str, float] = Field(
        default_factory=dict,
        description="Max normalized_score per family (absent → 1.0).",
    )
    family_floors: dict[str, float] = Field(
        default_factory=dict,
        description="Min normalized_score per family (absent → 0.0).",
    )
    escalation_rules: list[EscalationRule] = Field(
        default_factory=list,
        description="Escalation rules evaluated after caps/floors.",
    )
