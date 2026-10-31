"""ESR detection-rate evaluation (ESR stages 5.4-5.5).

The reference comparison (``scoring/reference.py``) bounds the *benign* false
positive rate at alpha by construction (split-conformal). What it does **not**
yet know is its *true* positive rate on poisoned adapters: until that is
measured the reference path is gated to ``review`` and may never ``block``
(``REFERENCE_CAN_BLOCK = False``). This module measures it.

Given a set of scanned adapters, each labelled benign or poisoned and tagged
with the attack that produced it, it reports:

  * **AUROC / PR-AUC** of the reference anomaly score (squared Mahalanobis
    distance) separating benign from poisoned — threshold-free detection power;
  * **detection rate at the calibrated operating points** ``alpha_block`` and
    ``alpha_review`` (poisoned flagged when its conformal p-value <= alpha), with
    the benign false-positive rate at the same points as a conformal sanity
    check (it must land at or below alpha);
  * **leave-one-attack-out**: the same figures computed per attack family, each
    family scored as the "unseen" positive class against the whole benign set.
    Because the detector is calibrated on benign adapters only and never sees an
    attack, holding one attack out changes nothing in the model — so this is
    exactly the per-attack generalisation view, and an attack that collapses the
    AUROC toward 0.5 is one the ESR geometry does not see.

This is what decides whether the block gate can be armed, and at which alpha.

Pure numpy / scikit-learn over already-produced :class:`ScanResult` objects; no
torch, no GPU, no network. The poisoned adapters it scores come either from a
GPU-trained corpus or, for a CPU sanity run and the unit tests, from the
synthetic payload injector in ``benchmarks/attacks.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from adaptersentry.schemas.result import ReferenceComparison, ScanResult, load_scan_result

logger = logging.getLogger("esr_eval")

# Score source: the reference anomaly score used to rank adapters. "distance" is
# the squared Mahalanobis distance (higher = more anomalous); "neglogp" is
# -log(conformal p). Both rank identically within one family, so AUROC is the
# same; distance is the canonical choice and the default.
ScoreSource = str
_SOURCES = ("distance", "neglogp")


@dataclass(frozen=True)
class ScoredAdapter:
    """One evaluated adapter: its label, attack tag and reference score."""

    adapter_id: str
    label: int                      # 0 = benign, 1 = poisoned
    attack: str | None              # attack family for a poisoned adapter, else None
    score: float | None             # ranking score (distance / neg-log-p); None if uncovered
    p_value: float | None           # conformal p-value, for the alpha operating points
    alpha_block: float | None
    alpha_review: float | None
    profile_id: str | None

    @property
    def covered(self) -> bool:
        """True when a reference profile scored this adapter."""
        return self.score is not None


def _ranking_score(ref: ReferenceComparison, source: ScoreSource) -> float:
    """Ranking score from a reference comparison (higher = more anomalous)."""
    if source == "distance":
        return float(ref.distance)
    # neglogp: strictly decreasing in p, so monotone with distance within a family.
    p = min(max(ref.p_value, 1e-12), 1.0)
    return -math.log(p)


def scored_from_reference(
    adapter_id: str,
    ref: ReferenceComparison | None,
    label: int,
    attack: str | None = None,
    *,
    source: ScoreSource = "distance",
) -> ScoredAdapter:
    """Build one :class:`ScoredAdapter` from a reference comparison (or ``None``).

    This is the common path for both a full :class:`ScanResult` and a bare
    comparison computed offline from a feature dump, so the harness is tested
    once and reused for both. ``ref is None`` marks an uncovered adapter.
    """
    if source not in _SOURCES:
        raise ValueError(f"unknown score source {source!r}; choose from {_SOURCES}")
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label!r}")
    if ref is None:
        return ScoredAdapter(adapter_id, label, attack if label == 1 else None,
                             None, None, None, None, None)
    return ScoredAdapter(
        adapter_id=adapter_id, label=label, attack=attack if label == 1 else None,
        score=_ranking_score(ref, source), p_value=float(ref.p_value),
        alpha_block=float(ref.alpha_block), alpha_review=float(ref.alpha_review),
        profile_id=ref.profile_id,
    )


def esr_score(result: ScanResult | dict[str, Any], source: ScoreSource = "distance") -> tuple[
    float | None, float | None, float | None, float | None, str | None
]:
    """Extract ``(score, p_value, alpha_block, alpha_review, profile_id)`` from a scan.

    All five are ``None`` when the adapter carried no reference comparison (no
    profile for its family, no rank, or fast mode); such adapters are
    "uncovered" and excluded from the reference metrics.
    """
    if source not in _SOURCES:
        raise ValueError(f"unknown score source {source!r}; choose from {_SOURCES}")
    res = result if isinstance(result, ScanResult) else load_scan_result(result)
    ref = res.anomaly.reference
    if ref is None:
        return None, None, None, None, None
    return (_ranking_score(ref, source), float(ref.p_value),
            float(ref.alpha_block), float(ref.alpha_review), ref.profile_id)


def collect_scores(
    records: Iterable[tuple[ScanResult | dict[str, Any], int, str | None]],
    *,
    source: ScoreSource = "distance",
) -> list[ScoredAdapter]:
    """Build :class:`ScoredAdapter` rows from ``(result, label, attack)`` triples."""
    scored: list[ScoredAdapter] = []
    for i, (result, label, attack) in enumerate(records):
        res = result if isinstance(result, ScanResult) else load_scan_result(result)
        adapter_id = res.artifact.logical_id if res.artifact else f"adapter-{i}"
        scored.append(scored_from_reference(
            adapter_id, res.anomaly.reference, label, attack, source=source))
    return scored


def _rate_at_alpha(rows: Sequence[ScoredAdapter], label: int, alpha: float | None) -> float | None:
    """Fraction of covered rows of ``label`` whose conformal p-value <= alpha."""
    if alpha is None:
        return None
    members = [r for r in rows if r.label == label and r.p_value is not None]
    if not members:
        return None
    return sum(1 for r in members if r.p_value <= alpha) / len(members)


def _auc(rows: Sequence[ScoredAdapter], metric: str) -> float | None:
    """ROC-AUC ('roc') or average precision ('pr'); None without both classes."""
    covered = [r for r in rows if r.covered]
    labels = [r.label for r in covered]
    if len(set(labels)) < 2:
        return None
    y = np.array(labels)
    s = np.array([r.score for r in covered], dtype=np.float64)
    fn = roc_auc_score if metric == "roc" else average_precision_score
    return float(fn(y, s))


@dataclass(frozen=True)
class EvalReport:
    """Detection figures for one benign-vs-poisoned comparison."""

    n_benign: int
    n_poisoned: int
    n_uncovered: int                     # labelled rows with no reference score
    auroc: float | None
    pr_auc: float | None
    alpha_block: float | None
    alpha_review: float | None
    tpr_at_block: float | None           # poisoned detection rate at alpha_block
    tpr_at_review: float | None
    fpr_at_block: float | None           # benign false-positive rate at alpha_block (conformal check)
    fpr_at_review: float | None

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _first_not_none(rows: Sequence[ScoredAdapter], attr: str) -> float | None:
    for r in rows:
        v = getattr(r, attr)
        if v is not None:
            return v
    return None


def evaluate(scored: Sequence[ScoredAdapter]) -> EvalReport:
    """Overall detection figures over all benign and poisoned adapters."""
    benign = [r for r in scored if r.label == 0]
    poisoned = [r for r in scored if r.label == 1]
    uncovered = sum(1 for r in scored if not r.covered)
    alpha_block = _first_not_none(scored, "alpha_block")
    alpha_review = _first_not_none(scored, "alpha_review")
    return EvalReport(
        n_benign=len(benign),
        n_poisoned=len(poisoned),
        n_uncovered=uncovered,
        auroc=_auc(scored, "roc"),
        pr_auc=_auc(scored, "pr"),
        alpha_block=alpha_block,
        alpha_review=alpha_review,
        tpr_at_block=_rate_at_alpha(scored, 1, alpha_block),
        tpr_at_review=_rate_at_alpha(scored, 1, alpha_review),
        fpr_at_block=_rate_at_alpha(scored, 0, alpha_block),
        fpr_at_review=_rate_at_alpha(scored, 0, alpha_review),
    )


def leave_one_attack_out(scored: Sequence[ScoredAdapter]) -> dict[str, EvalReport]:
    """Per-attack evaluation: each attack family vs the full benign set.

    The detector is calibrated on benign adapters only, so holding an attack out
    of "training" removes nothing — each family is therefore evaluated
    independently against all benign adapters, which is the honest unseen-attack
    generalisation view. A family whose AUROC sits near 0.5 is invisible to the
    ESR geometry and must not be relied on for blocking.
    """
    benign = [r for r in scored if r.label == 0]
    attacks = sorted({r.attack for r in scored if r.label == 1 and r.attack is not None})
    out: dict[str, EvalReport] = {}
    for attack in attacks:
        subset = benign + [r for r in scored if r.label == 1 and r.attack == attack]
        out[attack] = evaluate(subset)
    # Poisoned rows without an attack tag are reported under a synthetic key so
    # they are never silently dropped from the per-attack view.
    untagged = [r for r in scored if r.label == 1 and r.attack is None]
    if untagged:
        out["(untagged)"] = evaluate(benign + untagged)
    return out


def can_arm_block_gate(
    overall: EvalReport,
    per_attack: dict[str, EvalReport],
    *,
    min_auroc: float = 0.90,
    min_tpr_at_block: float = 0.50,
    max_fpr_slack: float = 1.5,
) -> tuple[bool, list[str]]:
    """Whether the measured rates justify arming the reference block gate.

    Conservative, and advisory only: the decision to flip ``REFERENCE_CAN_BLOCK``
    stays a human one. The gate may be armed when, overall and for *every* attack
    family, the detector clears a floor of discrimination and detection, and the
    benign FPR stays within ``max_fpr_slack`` x alpha (conformal tolerance for a
    finite calibration sample).

    Returns ``(ok, reasons)`` where ``reasons`` lists every failed check (empty
    when ``ok``).
    """
    reasons: list[str] = []
    if overall.auroc is None:
        reasons.append("no overall AUROC (missing a class or all uncovered)")
    elif overall.auroc < min_auroc:
        reasons.append(f"overall AUROC {overall.auroc:.3f} < {min_auroc:.2f}")
    if overall.tpr_at_block is not None and overall.tpr_at_block < min_tpr_at_block:
        reasons.append(f"overall TPR@block {overall.tpr_at_block:.3f} < {min_tpr_at_block:.2f}")
    if (overall.fpr_at_block is not None and overall.alpha_block is not None
            and overall.fpr_at_block > max_fpr_slack * overall.alpha_block):
        reasons.append(
            f"benign FPR@block {overall.fpr_at_block:.3f} exceeds "
            f"{max_fpr_slack:g}x alpha_block ({overall.alpha_block:.3f})")
    for attack, rep in per_attack.items():
        if rep.auroc is not None and rep.auroc < min_auroc:
            reasons.append(f"attack {attack!r}: AUROC {rep.auroc:.3f} < {min_auroc:.2f}")
        if rep.tpr_at_block is not None and rep.tpr_at_block < min_tpr_at_block:
            reasons.append(f"attack {attack!r}: TPR@block {rep.tpr_at_block:.3f} < {min_tpr_at_block:.2f}")
    return (not reasons), reasons


# ── reporting / CLI ──────────────────────────────────────────────────────────


def _fmt(v: float | None) -> str:
    return "  n/a" if v is None else f"{v:5.3f}"


def format_report(overall: EvalReport, per_attack: dict[str, EvalReport]) -> str:
    """A compact text table of the overall and per-attack figures."""
    lines = [
        f"benign={overall.n_benign}  poisoned={overall.n_poisoned}  "
        f"uncovered={overall.n_uncovered}  "
        f"alpha_block={_fmt(overall.alpha_block)} alpha_review={_fmt(overall.alpha_review)}",
        "",
        f"{'group':<18} {'n+':>4} {'AUROC':>7} {'PR':>7} "
        f"{'TPR@blk':>8} {'TPR@rev':>8} {'FPR@blk':>8} {'FPR@rev':>8}",
    ]

    def row(name: str, rep: EvalReport) -> str:
        return (f"{name:<18} {rep.n_poisoned:>4} {_fmt(rep.auroc):>7} {_fmt(rep.pr_auc):>7} "
                f"{_fmt(rep.tpr_at_block):>8} {_fmt(rep.tpr_at_review):>8} "
                f"{_fmt(rep.fpr_at_block):>8} {_fmt(rep.fpr_at_review):>8}")

    lines.append(row("OVERALL", overall))
    for attack, rep in per_attack.items():
        lines.append(row(attack, rep))
    return "\n".join(lines)


def _load_manifest(path: Path) -> list[tuple[dict[str, Any], int, str | None]]:
    """Read a JSONL manifest of labelled scan results.

    Each line is a JSON object with ``label`` (0/1), optional ``attack``, and
    either ``result`` (an inline ScanResult document) or ``scan_path`` (a path to
    a ScanResult JSON file). Paths are resolved relative to the manifest.
    """
    records: list[tuple[dict[str, Any], int, str | None]] = []
    base = path.resolve().parent
    with path.open() as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            label = int(rec["label"])
            attack = rec.get("attack")
            if "result" in rec:
                doc = rec["result"]
            elif "scan_path" in rec:
                p = (base / rec["scan_path"]).resolve()
                doc = json.loads(p.read_text())
            else:
                raise ValueError(f"manifest line {n}: needs 'result' or 'scan_path'")
            records.append((doc, label, attack))
    return records


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure ESR detection rate on a labelled corpus.")
    ap.add_argument("--manifest", type=Path, required=True,
                    help="JSONL of {label, attack?, result|scan_path} rows")
    ap.add_argument("--source", choices=_SOURCES, default="distance",
                    help="reference score to rank by (default: squared Mahalanobis distance)")
    ap.add_argument("--json", type=Path, default=None, help="also write the full report as JSON here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    records = _load_manifest(args.manifest)
    scored = collect_scores(records, source=args.source)
    overall = evaluate(scored)
    per_attack = leave_one_attack_out(scored)
    armed, reasons = can_arm_block_gate(overall, per_attack)

    print(format_report(overall, per_attack))
    print()
    print(f"block gate: {'ARMABLE' if armed else 'hold (variant A)'}")
    for r in reasons:
        print(f"  - {r}")

    if args.json is not None:
        args.json.write_text(json.dumps({
            "overall": overall.as_dict(),
            "per_attack": {k: v.as_dict() for k, v in per_attack.items()},
            "block_gate_armable": armed,
            "block_gate_reasons": reasons,
        }, indent=2))
        logger.info("wrote %s", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
