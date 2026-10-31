"""ESR detection-rate harness (benchmarks/esr_eval.py) and the payload generator.

The metric maths is tested deterministically on constructed reference
comparisons: that is the code that will decide, on the GPU-trained corpus,
whether the block gate can be armed. The synthetic weight-space generator is
tested only for the scan -> score plumbing it feeds — not for an AUROC, because
a Gaussian adapter sits outside the benign manifold the family profile was
calibrated on (both clean and poisoned saturate the conformal lower bound), so a
realistic detection rate needs real benign adapters as the negative class.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptersentry.scanner import scan
from adaptersentry.schemas.result import ReferenceComparison
from benchmarks.attacks import ATTACKS, AdapterSpec, build_corpus, write_adapter
from benchmarks.esr_eval import (
    can_arm_block_gate,
    collect_scores,
    esr_score,
    evaluate,
    format_report,
    leave_one_attack_out,
    scored_from_reference,
)

COVERED_BASE = "meta-llama/Meta-Llama-3-8B"  # has a shipped llama-3 profile


def _ref(distance: float, p: float, *, profile: str = "llama-3",
         alpha_block: float = 0.01, alpha_review: float = 0.05) -> ReferenceComparison:
    return ReferenceComparison(
        profile_id=profile, profile_version="esr-c-1+n213", stratum="x",
        n_reference=213, distance=distance, p_value=p,
        alpha_review=alpha_review, alpha_block=alpha_block,
    )


def _benign(n: int, *, distance: float = 1.0, p: float = 0.8) -> list:
    return [scored_from_reference(f"b{i}", _ref(distance, p), 0) for i in range(n)]


def _poisoned(n: int, attack: str, *, distance: float = 50.0, p: float = 0.001) -> list:
    return [scored_from_reference(f"{attack}{i}", _ref(distance, p), 1, attack) for i in range(n)]


class TestScoredFromReference:
    def test_uncovered_when_no_reference(self) -> None:
        s = scored_from_reference("a", None, 0)
        assert s.covered is False
        assert s.score is None and s.p_value is None

    def test_benign_has_no_attack_tag(self) -> None:
        # An attack tag on a benign row is a labelling error; it is dropped.
        assert scored_from_reference("a", _ref(1.0, 0.8), 0, "concentrated_rows").attack is None

    def test_distance_and_neglogp_rank_consistently(self) -> None:
        near, far = _ref(1.0, 0.9), _ref(99.0, 0.001)
        for source in ("distance", "neglogp"):
            lo = scored_from_reference("n", near, 1, "x", source=source).score
            hi = scored_from_reference("f", far, 1, "x", source=source).score
            assert hi > lo

    def test_bad_source_rejected(self) -> None:
        with pytest.raises(ValueError):
            scored_from_reference("a", _ref(1.0, 0.8), 0, source="bogus")

    def test_bad_label_rejected(self) -> None:
        with pytest.raises(ValueError):
            scored_from_reference("a", _ref(1.0, 0.8), 2)


class TestEvaluate:
    def test_perfect_separation(self) -> None:
        rep = evaluate(_benign(20) + _poisoned(10, "x"))
        assert rep.auroc == pytest.approx(1.0)
        assert rep.tpr_at_block == pytest.approx(1.0)      # poisoned p=0.001 <= 0.01
        assert rep.tpr_at_review == pytest.approx(1.0)
        assert rep.fpr_at_block == pytest.approx(0.0)      # benign p=0.8 not <= 0.01
        assert rep.n_benign == 20 and rep.n_poisoned == 10 and rep.n_uncovered == 0

    def test_single_class_has_no_auroc(self) -> None:
        assert evaluate(_benign(10)).auroc is None

    def test_uncovered_excluded_from_auroc_but_counted(self) -> None:
        rows = _benign(10) + _poisoned(5, "x") + [scored_from_reference("u", None, 1, "x")]
        rep = evaluate(rows)
        assert rep.n_uncovered == 1
        assert rep.n_poisoned == 6               # the uncovered poisoned row is still counted
        assert rep.auroc == pytest.approx(1.0)   # but excluded from the ranking metric

    def test_operating_point_uses_conformal_p(self) -> None:
        # Poisoned just above alpha_block are not flagged at block, but are at review.
        rows = _benign(10, p=0.9) + _poisoned(10, "x", p=0.03)
        rep = evaluate(rows)
        assert rep.tpr_at_block == pytest.approx(0.0)      # 0.03 > 0.01
        assert rep.tpr_at_review == pytest.approx(1.0)     # 0.03 <= 0.05


class TestLeaveOneAttackOut:
    def test_partitions_by_attack(self) -> None:
        rows = _benign(20) + _poisoned(6, "alpha") + _poisoned(4, "beta")
        loao = leave_one_attack_out(rows)
        assert set(loao) == {"alpha", "beta"}
        assert loao["alpha"].n_poisoned == 6 and loao["beta"].n_poisoned == 4
        assert loao["alpha"].n_benign == 20          # each attack scored vs all benign

    def test_untagged_poisoned_not_dropped(self) -> None:
        rows = _benign(10) + [scored_from_reference("p", _ref(50, 0.001), 1, None)]
        loao = leave_one_attack_out(rows)
        assert "(untagged)" in loao and loao["(untagged)"].n_poisoned == 1


class TestBlockGateDecision:
    def test_armable_when_strong(self) -> None:
        rows = _benign(50) + _poisoned(30, "a") + _poisoned(30, "b")
        ok, reasons = can_arm_block_gate(evaluate(rows), leave_one_attack_out(rows))
        assert ok and reasons == []

    def test_weak_attack_blocks_arming(self) -> None:
        # One attack the geometry cannot see (score overlaps benign) holds the gate.
        invisible = [scored_from_reference(f"i{i}", _ref(1.0, 0.8), 1, "invisible") for i in range(20)]
        rows = _benign(50) + _poisoned(30, "seen") + invisible
        ok, reasons = can_arm_block_gate(evaluate(rows), leave_one_attack_out(rows))
        assert ok is False
        assert any("invisible" in r for r in reasons)

    def test_high_benign_fpr_blocks_arming(self) -> None:
        # Benign flagged well above alpha (mis-calibrated) must hold the gate.
        rows = _benign(50, p=0.005) + _poisoned(50, "a")
        ok, reasons = can_arm_block_gate(evaluate(rows), leave_one_attack_out(rows))
        assert ok is False
        assert any("FPR@block" in r for r in reasons)


class TestFormatReport:
    def test_renders_overall_and_per_attack(self) -> None:
        rows = _benign(10) + _poisoned(5, "alpha")
        text = format_report(evaluate(rows), leave_one_attack_out(rows))
        assert "OVERALL" in text and "alpha" in text


class TestGeneratorPlumbing:
    """attacks.py -> scan -> collect_scores: structure only, not an AUROC."""

    def test_catalogue_builds_and_scores(self, tmp_path: Path) -> None:
        triples = build_corpus(tmp_path, base_model=COVERED_BASE,
                               n_benign=3, n_per_attack=2, seed=1)
        assert len(triples) == 3 + 2 * len(ATTACKS)
        records = [(scan(d / "adapter_model.safetensors", mode="full"), label, attack)
                   for d, label, attack in triples]
        scored = collect_scores(records)
        # A covered family means every adapter carries a reference comparison.
        assert all(s.covered for s in scored)
        assert {s.profile_id for s in scored} == {"llama-3"}
        benign = [s for s in scored if s.label == 0]
        poisoned = [s for s in scored if s.label == 1]
        assert len(benign) == 3 and len(poisoned) == 2 * len(ATTACKS)
        assert {s.attack for s in poisoned} == set(ATTACKS)
        # The harness runs end to end on the generated corpus without error.
        format_report(evaluate(scored), leave_one_attack_out(scored))

    def test_scale_hide_preserves_effective_update(self, tmp_path: Path) -> None:
        # alpha x8 and B/8 leave ΔW = (alpha/r)·B·A unchanged, so the effective-update
        # features (hence the reference score) match the un-hidden adapter.
        plain = write_adapter(tmp_path / "plain", AdapterSpec(base_model=COVERED_BASE, seed=7))
        hidden = write_adapter(tmp_path / "hidden",
                               AdapterSpec(base_model=COVERED_BASE, attack="scale_hide", seed=7))
        dp, dh = (scan(p, mode="full").anomaly.reference for p in (plain, hidden))
        assert dp is not None and dh is not None
        assert dh.distance == pytest.approx(dp.distance, rel=1e-6)


class TestEsrScoreFromScanResult:
    def test_covered_scan_yields_reference_fields(self, tmp_path: Path) -> None:
        write_adapter(tmp_path, AdapterSpec(base_model=COVERED_BASE, seed=3))
        res = scan(tmp_path / "adapter_model.safetensors", mode="full")
        score, p, a_block, a_review, profile = esr_score(res)
        assert profile == "llama-3"
        assert score is not None and score >= 0.0
        assert 0.0 < p <= 1.0
        assert a_block == pytest.approx(0.01) and a_review == pytest.approx(0.05)

    def test_fast_mode_is_uncovered(self, tmp_path: Path) -> None:
        write_adapter(tmp_path, AdapterSpec(base_model=COVERED_BASE, seed=3))
        res = scan(tmp_path / "adapter_model.safetensors", mode="fast")
        assert esr_score(res) == (None, None, None, None, None)

    def test_unknown_family_is_uncovered(self, tmp_path: Path) -> None:
        write_adapter(tmp_path, AdapterSpec(base_model="bigscience/bloom-560m", seed=3))
        res = scan(tmp_path / "adapter_model.safetensors", mode="full")
        assert esr_score(res)[0] is None
