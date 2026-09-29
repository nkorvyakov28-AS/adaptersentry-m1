"""Direct tests of the 2.0 verdict rules, including the reference-profile path."""

from __future__ import annotations

from adaptersentry.schemas.finding import Severity
from adaptersentry.schemas.result import DepthSpike, IntraAnomaly, ReferenceComparison
from adaptersentry.scoring.verdict_v2 import derive_verdict


def _ref(p: float) -> ReferenceComparison:
    return ReferenceComparison(profile_id="llama", profile_version="1", stratum="s", n_reference=300,
                               distance=10.0, p_value=p, alpha_review=0.01, alpha_block=0.001)


def _v(**kw):
    base = dict(status="ok", policy="default", intra=None, reference=None, red_flags=[],
                metadata_present=True, confidence_limits=[], n_not_analyzed=0)
    base.update(kw)
    return derive_verdict(**base)


def test_reference_block_and_critical() -> None:
    assert _v(reference=_ref(0.0005)).action == "block"
    assert _v(reference=_ref(0.0005)).level == Severity.HIGH
    assert _v(reference=_ref(0.00005)).level == Severity.CRITICAL


def test_reference_review_and_allow() -> None:
    assert _v(reference=_ref(0.005)).action == "review"
    ok = _v(reference=_ref(0.4))
    assert ok.action == "allow" and ok.confidence.level == "high"


def test_intra_below_threshold_allows() -> None:
    intra = IntraAnomaly(max_robust_z=5.0, n_outlier_modules=1)
    assert _v(intra=intra).action == "allow"


def test_intra_above_threshold_reviews_with_location() -> None:
    intra = IntraAnomaly(max_robust_z=12.0, n_outlier_modules=1,
                         depth_spikes=[DepthSpike(module_type="down_proj", layer=47, z=12.0)])
    v = _v(intra=intra)
    assert v.action == "review" and "layer 47" in next(r.message for r in v.reasons if r.code == "INTRA_OUTLIER")


def test_strict_policy_needs_red_flag_to_block() -> None:
    assert _v(policy="strict").action == "allow"
    assert _v(policy="strict", red_flags=["ROUTER_ADAPTED"]).action == "block"


def test_failed_is_low_confidence_review() -> None:
    v = _v(status="failed")
    assert v.action == "review" and v.confidence.level == "low" and v.m2_recommended


def test_missing_metadata_recommends_behavioural_check() -> None:
    v = _v(metadata_present=False)
    assert v.action == "allow" and v.m2_recommended
