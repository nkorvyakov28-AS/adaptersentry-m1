"""ESR stage 5 reference path: loader, conformal p-value, and the block gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adaptersentry.scanner import scan
from adaptersentry.scoring import reference as refmod
from adaptersentry.scoring.reference import compare_to_reference
from adaptersentry.scoring.verdict import derive_verdict
from adaptersentry.schemas.result import Policy, ReferenceComparison
from tests.adapter_factory import write_adapter

SHIPPED = ("qwen3", "qwen2.5", "llama-3")


def _adapter(tmp_path: Path, base: str, **kw) -> Path:
    cfg = {
        "peft_type": "LORA", "r": kw.pop("r", 8), "lora_alpha": 16,
        "base_model_name_or_path": base,
        "target_modules": ["q_proj", "v_proj", "down_proj"],
    }
    return write_adapter(tmp_path, config=cfg, **kw)


class TestShippedProfiles:
    def test_three_families_ship_calibrated(self) -> None:
        avail = {fam for fam, _ in refmod._available()}
        assert set(SHIPPED) <= avail

    @pytest.mark.parametrize("family", SHIPPED)
    def test_profile_invariants(self, family: str) -> None:
        path = next(p for fam, p in refmod._available() if fam == family)
        prof = refmod._load(str(path))
        dim = prof["cal_mu"].shape[0]
        assert dim == 2 * len(prof["features"])          # [median z, p90 |z|] per feature
        assert prof["cal_precision"].shape == (dim, dim)
        assert prof["cal_ref_scores"].size >= 1
        assert 0.0 < float(prof["cal_alpha_block"]) < float(prof["cal_alpha_review"]) < 1.0


class TestLoaderSelection:
    def test_covered_family_is_compared(self, tmp_path: Path) -> None:
        ref = scan(_adapter(tmp_path, "meta-llama/Meta-Llama-3-8B")).anomaly.reference
        assert ref is not None
        assert ref.profile_id == "llama-3"
        assert ref.n_reference >= 99                      # strict alpha = 0.01 needs >= 99
        assert 0.0 < ref.p_value <= 1.0
        assert ref.distance >= 0.0
        assert ref.alpha_block == pytest.approx(0.01) and ref.alpha_review == pytest.approx(0.05)
        assert ref.profile_version.startswith("esr-c-1")

    def test_unknown_family_has_no_reference(self, tmp_path: Path) -> None:
        result = scan(_adapter(tmp_path, "google/gemma-2b"))
        assert result.anomaly.reference is None
        assert any(r.code == "NO_REFERENCE_PROFILE" for r in result.verdict.reasons)

    def test_longest_family_key_wins(self, tmp_path: Path) -> None:
        # "qwen2.5" must select the qwen2.5 profile, not a shorter "qwen*" prefix.
        ref = scan(_adapter(tmp_path, "Qwen/Qwen2.5-7B-Instruct")).anomaly.reference
        assert ref is not None and ref.profile_id == "qwen2.5"

    def test_fast_mode_skips_reference(self, tmp_path: Path) -> None:
        # Calibration corpus was dumped in full mode; a fast vector is not exchangeable.
        path = _adapter(tmp_path, "Qwen/Qwen3-8B")
        assert scan(path, mode="full").anomaly.reference is not None
        assert scan(path, mode="fast").anomaly.reference is None

    def test_missing_rank_returns_none(self) -> None:
        # Without a declared or recoverable rank, the log-rank trend term is
        # undefined, so the comparison is declined rather than guessed.
        assert compare_to_reference(
            base_model_declared="Qwen/Qwen3-8B", rank=None, n_model_layers=16, entries=[],
        ) is None
        assert compare_to_reference(
            base_model_declared=None, rank=8, n_model_layers=16, entries=[],
        ) is None


class TestBlockGate:
    """Variant A: the reference may escalate to review, never to block, until the
    poisoned-corpus detection rate is measured."""

    def test_scan_does_not_block_even_below_alpha_block(self, tmp_path: Path) -> None:
        result = scan(_adapter(tmp_path, "meta-llama/Meta-Llama-3-8B"))
        ref = result.anomaly.reference
        assert ref is not None and ref.p_value <= ref.alpha_block   # extreme vs real llama-3
        assert result.verdict.action == "review"                    # gated: not "block"
        assert refmod.REFERENCE_CAN_BLOCK is False

    def test_gate_toggles_block_in_derive_verdict(self) -> None:
        ref = ReferenceComparison(
            profile_id="llama-3", profile_version="esr-c-1+n213", stratum="x",
            n_reference=105, distance=99.0, p_value=0.001, alpha_review=0.05, alpha_block=0.01,
        )
        common = dict(
            status="ok", policy="default", intra=None, red_flags=[],
            metadata_present=True, confidence_limits=[], n_not_analyzed=0,
        )
        gated = derive_verdict(reference=ref, reference_can_block=False, **common)
        armed = derive_verdict(reference=ref, reference_can_block=True, **common)
        assert gated.action == "review"
        assert armed.action == "block"


class TestConformalArithmetic:
    def test_p_value_bounds_and_monotonicity(self, tmp_path: Path) -> None:
        # Farther from the benign centre -> no larger p-value.
        near = scan(_adapter(tmp_path / "a", "Qwen/Qwen3-8B", b_scale=0.01)).anomaly.reference
        far = scan(_adapter(tmp_path / "b", "Qwen/Qwen3-8B", b_scale=1.0)).anomaly.reference
        assert near is not None and far is not None
        n_ref = near.n_reference
        assert near.p_value >= 1.0 / (n_ref + 1)          # conformal lower bound
        assert far.p_value <= near.p_value
