"""
benchmarks/report.py
====================
Report generation for the AdapterSentry HuggingFace Hub benchmark.

Produces four output artefacts from a completed (or partial) results.jsonl:

  results.csv      One row per adapter; sortable/filterable in any spreadsheet tool.
  aggregate.json   Machine-readable statistics, percentiles, and top-suspicious lists.
  report.md        Human-readable benchmark report with methodology and interpretation notes.

All framing follows the "observational benchmark" contract — no claims of classification
accuracy, precision, or recall; high scores are described as investigation candidates only.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def write_csv(results: list, csv_path: Path) -> None:
    """Write per-adapter results to a flat CSV file.

    List-valued fields (top_flags, hf_tags) are joined with ' | ' so every
    cell is a plain string and the CSV can be opened directly in Excel or pandas.
    """
    if not results:
        logger.warning("No results to write — CSV not created")
        return

    # Build fieldnames: scalar fields first, then the two flattened list fields
    scalar_fields = [
        f for f in results[0].__dataclass_fields__
        if f not in ("top_flags", "hf_tags")
    ]
    fieldnames = scalar_fields + ["hf_tags_summary", "top_flags_summary"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = r.to_dict()
            row["hf_tags_summary"] = ",".join((row.pop("hf_tags") or [])[:5])
            row["top_flags_summary"] = " | ".join((row.pop("top_flags") or [])[:3])
            writer.writerow(row)

    logger.info("CSV written: %s (%d rows)", csv_path, len(results))


# ---------------------------------------------------------------------------
# Aggregate JSON
# ---------------------------------------------------------------------------


def _percentiles(values: list[float], pcts: list[int]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    return {f"p{p}": float(np.percentile(arr, p)) for p in pcts}


def _top_entry(r: Any) -> dict[str, Any]:
    return {
        "repo_id": r.repo_id,
        "ensemble_score": r.ensemble_score,
        "ensemble_risk_level": r.ensemble_risk_level,
        "overall_risk": r.overall_risk,
        "training_status": r.training_status,
        "n_flags": r.n_flags,
        "top_flags": (r.top_flags or [])[:3],
        "cross_layer_consistency": r.cross_layer_consistency,
        "hf_downloads": r.hf_downloads,
    }


def write_aggregate(
    results: list,
    agg_path: Path,
    candidates: list,
    limit: int,
    top_n: int,
) -> dict[str, Any]:
    """Compute aggregate statistics and write aggregate.json.

    Returns the aggregate dict so the caller (and report.md) can reuse it
    without re-reading the file.
    """
    success = [r for r in results if r.status == "success"]
    unsupported = [r for r in results if r.status == "unsupported_architecture"]
    failed = [r for r in results if r.status in ("download_failed", "analysis_failed")]
    skipped = [r for r in results if r.status in ("size_exceeded", "skipped", "not_cached")]

    # Per-status counts for failure_breakdown
    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    # Distributions — computed over SUCCESSFULLY scanned repos only.
    # unsupported_architecture is excluded because ensemble_score is not computed for it.
    risk_dist: dict[str, int] = {}
    training_dist: dict[str, int] = {}
    ens_scores: list[float] = []

    for r in success:
        level = r.ensemble_risk_level or "UNKNOWN"
        risk_dist[level] = risk_dist.get(level, 0) + 1

        ts = r.training_status or "UNKNOWN"
        training_dist[ts] = training_dist.get(ts, 0) + 1

        if r.ensemble_score is not None:
            ens_scores.append(r.ensemble_score)

    pcts = _percentiles(ens_scores, [10, 25, 50, 75, 90, 95, 99])

    # Top suspicious by ensemble score (investigation candidates)
    top_by_ens = sorted(
        [r for r in success if r.ensemble_score is not None],
        key=lambda r: r.ensemble_score,  # type: ignore[arg-type]
        reverse=True,
    )[:top_n]

    # Top by rule-based score (flags with non-zero weight)
    top_by_rule = sorted(
        [r for r in success if r.overall_risk],
        key=lambda r: r.overall_risk,  # type: ignore[arg-type]
        reverse=True,
    )[:top_n]

    agg: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "framing": (
            "Observational benchmark only. No labeled ground truth exists. "
            "High scores flag adapters for investigation, not confirmed malicious content."
        ),
        "run_params": {"target_limit": limit, "top_n": top_n},
        "totals": {
            "discovered": len(candidates),
            "attempted": len(results),
            "succeeded": len(success),
            "unsupported_architecture": len(unsupported),
            "failed": len(failed),
            "skipped": len(skipped),
        },
        # Granular per-status counts for debugging and tooling
        "failure_breakdown": {
            "unsupported_architecture": status_counts.get("unsupported_architecture", 0),
            "analysis_failed": status_counts.get("analysis_failed", 0),
            "download_failed": status_counts.get("download_failed", 0),
            "size_exceeded": status_counts.get("size_exceeded", 0),
            "not_cached": status_counts.get("not_cached", 0),
            "skipped": status_counts.get("skipped", 0),
        },
        # Kept for backward compatibility; equivalent to failure_breakdown
        "failure_reason_counts": status_counts,
        # Risk and training distributions cover success rows only
        "risk_level_distribution": risk_dist,
        "training_status_distribution": training_dist,
        "ensemble_score_percentiles": pcts,
        "ensemble_score_mean": float(np.mean(ens_scores)) if ens_scores else None,
        "counts": {
            "INIT_ONLY": training_dist.get("INIT_ONLY", 0),
            "PARTIALLY_TRAINED": training_dist.get("PARTIALLY_TRAINED", 0),
            "LOW": risk_dist.get("LOW", 0),
            "MEDIUM": risk_dist.get("MEDIUM", 0),
            "HIGH": risk_dist.get("HIGH", 0),
            "CRITICAL": risk_dist.get("CRITICAL", 0),
        },
        "top_suspicious_by_ensemble_score": [_top_entry(r) for r in top_by_ens],
        "top_suspicious_by_rule_score": [_top_entry(r) for r in top_by_rule],
    }

    with agg_path.open("w") as f:
        json.dump(agg, f, indent=2)

    logger.info(
        "Aggregate JSON written: %s  (succeeded=%d, HIGH=%d, CRITICAL=%d)",
        agg_path, len(success),
        risk_dist.get("HIGH", 0), risk_dist.get("CRITICAL", 0),
    )
    return agg


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_markdown_report(
    agg: dict[str, Any],
    results: list,
    report_path: Path,
    limit: int,
) -> None:
    """Write a human-readable Markdown benchmark report."""

    totals = agg["totals"]
    risk_dist = agg["risk_level_distribution"]
    training_dist = agg["training_status_distribution"]
    pcts = agg["ensemble_score_percentiles"]
    generated_at = agg["generated_at"]
    total_success = max(totals["succeeded"], 1)  # guard zero-division

    lines: list[str] = []

    def section(level: int, title: str) -> None:
        lines.append("#" * level + " " + title)
        lines.append("")

    def para(text: str) -> None:
        lines.append(text)
        lines.append("")

    def bullets(items: list[str]) -> None:
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    # ── Header ───────────────────────────────────────────────────────────────
    section(1, f"AdapterSentry M1 — HuggingFace Hub Benchmark")
    lines.append(f"**Generated:** {generated_at}  ")
    lines.append(f"**Target sample size:** {limit}  ")
    lines.append(f"**Successfully scanned:** {totals['succeeded']}  ")
    lines.append("")

    # ── Scope and methodology ─────────────────────────────────────────────────
    section(2, "Scope and Methodology")
    para(
        "This report summarises a large-scale static scan of public LoRA adapter repositories "
        "from HuggingFace Hub using AdapterSentry M1. M1 inspects adapter weight tensors "
        "read-only, without loading a base model or running any model inference."
    )
    para(
        "**This is an observational benchmark, not a malware classifier.** "
        "No labeled ground truth exists for the public Hub adapter population. "
        "Adapters with high ensemble scores or anomalous flag patterns are "
        "flagged as *investigation candidates* — they do not constitute confirmed "
        "malicious content. Confirming malicious behaviour requires behavioural "
        "analysis, provenance review, and domain expertise."
    )

    # ── Candidate selection ───────────────────────────────────────────────────
    section(2, "Candidate Selection Criteria")
    bullets([
        "HuggingFace Hub queried with filter `peft`, sorted by download count (most popular first)",
        "Repositories must contain `adapter_model.safetensors` (single-file adapters only)",
        "`adapter_config.json` fetched when present to extract declared LoRA rank",
        "Base model weights are never downloaded",
        f"File size limit: {agg['run_params'].get('target_limit', '?')} repos maximum",
        "Selection is deterministic given the same HF Hub sort order at query time",
    ])

    # ── Limitations ───────────────────────────────────────────────────────────
    section(2, "Limitations")
    bullets([
        "No labeled ground truth — classification accuracy (precision, recall, F1) cannot be reported",
        "Only single-file `adapter_model.safetensors` adapters are covered; sharded multi-file adapters are excluded",
        "M1 only supports standard PEFT LoRA adapters that contain matched `lora_A.weight` / `lora_B.weight` tensor pairs. "
        "Adapters using non-standard layer names or alternative PEFT methods (IA³, LoHa, LoCon, etc.) are classified "
        "`unsupported_architecture` and excluded from risk statistics.",
        "Selection is biased toward popular adapters (sorted by downloads); newly published or niche repos are under-represented",
        "M1 thresholds were calibrated on a small development set; false-positive and false-negative rates at scale are unknown",
        "Private and gated repositories are excluded",
        "Large adapters above the configured size limit are skipped and counted separately",
        "Adapters that trigger download failures (rate limits, network errors, repo deletion) are recorded but not scanned",
    ])

    # ── Aggregate results ─────────────────────────────────────────────────────
    section(2, "Aggregate Results")

    section(3, "Run Summary")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Repos discovered | {totals['discovered']} |")
    lines.append(f"| Repos attempted | {totals['attempted']} |")
    lines.append(f"| Successfully scanned | {totals['succeeded']} |")
    lines.append(f"| Unsupported architecture (excluded from risk stats) | {totals.get('unsupported_architecture', 0)} |")
    lines.append(f"| Download / analysis failures | {totals['failed']} |")
    lines.append(f"| Skipped (size exceeded, not cached, or other) | {totals['skipped']} |")
    lines.append("")

    section(3, "Ensemble Risk Level Distribution")
    lines.append("| Risk Level | Count | Share of scanned |")
    lines.append("|---|---|---|")
    for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"):
        count = risk_dist.get(level, 0)
        if count == 0 and level == "UNKNOWN":
            continue
        lines.append(f"| {level} | {count} | {count / total_success:.1%} |")
    lines.append("")

    section(3, "Training Status Distribution")
    lines.append("| Training Status | Count | Share of scanned |")
    lines.append("|---|---|---|")
    for status in ("TRAINED", "INIT_ONLY", "PARTIALLY_TRAINED", "UNKNOWN"):
        count = training_dist.get(status, 0)
        if count == 0 and status == "UNKNOWN":
            continue
        lines.append(f"| {status} | {count} | {count / total_success:.1%} |")
    lines.append("")

    section(3, "Ensemble Score Distribution (successfully scanned adapters)")
    if pcts:
        lines.append("| Percentile | Score |")
        lines.append("|---|---|")
        for key, val in sorted(pcts.items(), key=lambda x: int(x[0][1:])):
            lines.append(f"| {key} | {val:.2f} |")
        mean_score = agg.get("ensemble_score_mean")
        if mean_score is not None:
            lines.append(f"| mean | {mean_score:.2f} |")
        lines.append("")
    else:
        para("No ensemble score data available.")

    # ── Failure breakdown ─────────────────────────────────────────────────────
    section(2, "Failure Breakdown")
    failure_counts = agg.get("failure_reason_counts", {})
    if failure_counts:
        lines.append("| Failure Type | Count |")
        lines.append("|---|---|")
        for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {count} |")
        lines.append("")
    else:
        para("No failures recorded.")

    # ── Top investigation candidates ──────────────────────────────────────────
    section(2, "Top Investigation Candidates (by Ensemble Score)")
    para(
        "The following adapters produced the highest ensemble scores in this run. "
        "They are listed as investigation candidates and are **not confirmed malicious**. "
        "Each should be reviewed in context: provenance, stated purpose, and behavioural testing."
    )
    top_ens = agg.get("top_suspicious_by_ensemble_score", [])
    if top_ens:
        show = min(len(top_ens), 15)
        lines.append("| Repository | Ensemble | Risk | Rule | Training | Flags | Top signal |")
        lines.append("|---|---|---|---|---|---|---|")
        for entry in top_ens[:show]:
            flags = entry.get("top_flags") or []
            top_flag = flags[0][:55] if flags else "—"
            lines.append(
                f"| `{entry['repo_id']}` "
                f"| {entry['ensemble_score']:.1f} "
                f"| {entry['ensemble_risk_level']} "
                f"| {entry['overall_risk']} "
                f"| {entry['training_status']} "
                f"| {entry['n_flags']} "
                f"| {top_flag} |"
            )
        lines.append("")
    else:
        para("No successful scans to report.")

    # ── Interpretation guide ──────────────────────────────────────────────────
    section(2, "Interpretation Guidance")
    bullets([
        "**LOW (ensemble 0–6):** Well-formed, trained adapter. No anomalous signals.",
        "**MEDIUM (7–13):** Elevated signal; likely benign. Worth a second look before production deployment.",
        "**HIGH (14–35):** Multiple independent detectors agree. Inspect adapter provenance and weights before use.",
        "**CRITICAL (36–100):** Strong multi-signal anomaly. Do not load without thorough review.",
        "**INIT_ONLY:** Standard LoRA zero-initialisation state (B=0, A=uniform). Init-artifact flags are suppressed; ensemble score reflects residual statistical features only.",
        "**PARTIALLY_TRAINED:** Some layers trained while others remain at zero-init. Uncommon in standard fine-tuning; warrants provenance review.",
        "Ensemble score and rule flags are statistical weight-tensor signals, not behavioural evidence. A high score is the beginning of an investigation, not a conclusion.",
    ])

    para(
        "For adapters rated HIGH or CRITICAL, recommended next steps: "
        "(1) verify the adapter's stated training purpose and provenance; "
        "(2) run AdapterSentry M2 behavioral sandbox when available; "
        "(3) if behavioural evidence is found, report to the HuggingFace Hub moderation team."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown report written: %s", report_path)
