"""
benchmarks/report.py
====================
Report generation for the AdapterSentry HuggingFace Hub benchmark.

Produces four output artefacts from a completed (or partial) results.jsonl:

  results.csv      One row per adapter; sortable/filterable in any spreadsheet tool.
  aggregate.json   Machine-readable statistics, percentiles, and top-suspicious lists.
  report.md        Human-readable benchmark report with methodology and interpretation notes.

All framing follows the "observational benchmark" contract — no claims of classification
accuracy, precision, or recall. Records come from ScanResult 2.0.0: verdict action/level,
training state, the intra-adapter robust z, and verdict reason codes. The intra-adapter
thresholds are uncalibrated; distributions describe scanner behaviour, not detection
accuracy. High values mark investigation candidates only.
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

# Verdict reason codes that describe the adapter's structure rather than weight statistics.
STRUCTURAL_REASON_CODES = frozenset({
    "FULL_WEIGHT_REPLACEMENT",
    "ROUTER_ADAPTED",
    "RANK_MISMATCH",
    "SIBLING_EXECUTABLE",
})

_ACTIONS = ("allow", "review", "block")
_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
_TRAINING_STATES = ("trained", "init_only", "partial", "unknown")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def write_csv(results: list, csv_path: Path) -> None:
    """Write per-adapter results to a flat CSV file.

    List-valued fields (hf_tags, reason_codes, top_findings, tensor_keys_sample)
    are joined into one string so every cell is plain text and the CSV can be
    opened directly in Excel or pandas.
    """
    if not results:
        logger.warning("No results to write — CSV not created")
        return

    list_fields = ("hf_tags", "reason_codes", "top_findings", "tensor_keys_sample")
    scalar_fields = [f for f in results[0].__dataclass_fields__ if f not in list_fields]
    fieldnames = scalar_fields + [
        "hf_tags_summary", "reason_codes_summary", "top_findings_summary", "tensor_keys_sample_summary",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = r.to_dict()
            row["hf_tags_summary"] = ",".join((row.pop("hf_tags") or [])[:5])
            row["reason_codes_summary"] = ",".join(row.pop("reason_codes") or [])
            row["top_findings_summary"] = " | ".join((row.pop("top_findings") or [])[:3])
            row["tensor_keys_sample_summary"] = ",".join((row.pop("tensor_keys_sample") or [])[:5])
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


def structural_codes(r: Any) -> list[str]:
    """Verdict reason codes of ``r`` that are structural red flags."""
    return [c for c in (r.reason_codes or []) if c in STRUCTURAL_REASON_CODES]


def _top_entry(r: Any) -> dict[str, Any]:
    return {
        "repo_id": r.repo_id,
        "action": r.action,
        "level": r.level,
        "scan_status": r.scan_status,
        "training_state": r.training_state,
        "max_robust_z": r.max_robust_z,
        "n_outlier_modules": r.n_outlier_modules,
        "structural_codes": structural_codes(r),
        "reason_codes": list(r.reason_codes or []),
        "n_findings": r.n_findings,
        "top_findings": (r.top_findings or [])[:3],
        "hf_downloads": r.hf_downloads,
    }


def _count(values: list[str | None]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        key = v if v is not None else "unknown"
        out[key] = out.get(key, 0) + 1
    return out


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

    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    # Distributions — over successfully scanned repos only (scan status ok or degraded).
    action_dist = _count([r.action for r in success])
    level_dist = _count([r.level for r in success])
    training_dist = _count([r.training_state for r in success])
    scan_status_dist = _count([r.scan_status for r in success])
    reason_code_dist: dict[str, int] = {}
    for r in success:
        for code in set(r.reason_codes or []):
            reason_code_dist[code] = reason_code_dist.get(code, 0) + 1

    z_values = [float(r.max_robust_z) for r in success if r.max_robust_z is not None]
    pcts = _percentiles(z_values, [10, 25, 50, 75, 90, 95, 99])

    # Investigation candidates by intra-adapter robust z (uncalibrated)
    top_by_z = sorted(
        [r for r in success if r.max_robust_z is not None],
        key=lambda r: r.max_robust_z,  # type: ignore[arg-type]
        reverse=True,
    )[:top_n]

    # Investigation candidates by number of structural red flags
    top_by_structural = sorted(
        [r for r in success if structural_codes(r)],
        key=lambda r: (len(structural_codes(r)), r.max_robust_z or 0.0),
        reverse=True,
    )[:top_n]

    agg: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "framing": (
            "Observational benchmark only. No labeled ground truth exists. "
            "Intra-adapter thresholds are uncalibrated; distributions describe scanner "
            "behaviour, not detection accuracy. A 'review' verdict or a high robust z "
            "marks an investigation candidate, not confirmed malicious content."
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
        "failure_breakdown": {
            "unsupported_architecture": status_counts.get("unsupported_architecture", 0),
            "analysis_failed": status_counts.get("analysis_failed", 0),
            "download_failed": status_counts.get("download_failed", 0),
            "size_exceeded": status_counts.get("size_exceeded", 0),
            "not_cached": status_counts.get("not_cached", 0),
            "skipped": status_counts.get("skipped", 0),
        },
        # All record statuses, including success
        "failure_reason_counts": status_counts,
        # Distributions below cover success rows only
        "scan_status_distribution": scan_status_dist,
        "action_distribution": action_dist,
        "level_distribution": level_dist,
        "training_state_distribution": training_dist,
        "reason_code_distribution": reason_code_dist,
        "max_robust_z_percentiles": pcts,
        "max_robust_z_mean": float(np.mean(z_values)) if z_values else None,
        "n_with_intra_anomaly": len(z_values),
        "counts": {
            "allow": action_dist.get("allow", 0),
            "review": action_dist.get("review", 0),
            "block": action_dist.get("block", 0),
            "init_only": training_dist.get("init_only", 0),
            "partial": training_dist.get("partial", 0),
            "degraded": scan_status_dist.get("degraded", 0),
            "with_structural_codes": sum(1 for r in success if structural_codes(r)),
        },
        "top_suspicious_by_max_robust_z": [_top_entry(r) for r in top_by_z],
        "top_suspicious_by_structural_codes": [_top_entry(r) for r in top_by_structural],
    }

    with agg_path.open("w") as f:
        json.dump(agg, f, indent=2)

    logger.info(
        "Aggregate JSON written: %s  (succeeded=%d, review=%d, block=%d)",
        agg_path, len(success), action_dist.get("review", 0), action_dist.get("block", 0),
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
    action_dist = agg["action_distribution"]
    level_dist = agg["level_distribution"]
    training_dist = agg["training_state_distribution"]
    pcts = agg["max_robust_z_percentiles"]
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

    def dist_table(header: str, dist: dict[str, int], order: tuple[str, ...]) -> None:
        lines.append(f"| {header} | Count | Share of scanned |")
        lines.append("|---|---|---|")
        keys = list(order) + sorted(k for k in dist if k not in order)
        for key in keys:
            count = dist.get(key, 0)
            if count == 0 and key not in order:
                continue
            lines.append(f"| {key} | {count} | {count / total_success:.1%} |")
        lines.append("")

    # ── Header ───────────────────────────────────────────────────────────────
    section(1, "AdapterSentry M1 — HuggingFace Hub Benchmark")
    lines.append(f"**Generated:** {generated_at}  ")
    lines.append(f"**Target sample size:** {limit}  ")
    lines.append(f"**Successfully scanned:** {totals['succeeded']}  ")
    lines.append("")

    # ── Scope and methodology ─────────────────────────────────────────────────
    section(2, "Scope and Methodology")
    para(
        "This report summarises a large-scale static scan of public LoRA adapter repositories "
        "from HuggingFace Hub using AdapterSentry M1 (ScanResult schema 2.0). M1 reads the "
        "adapter weight tensors read-only, without loading a base model or running any model "
        "inference, and computes exact statistics of the effective update ΔW = s·B·A per module."
    )
    para(
        "**This is an observational benchmark, not a malware classifier.** "
        "No labeled ground truth exists for the public Hub adapter population, so the "
        "distributions below describe how the scanner behaves on this population — they are "
        "not detection accuracy. Without a calibrated reference profile, the default policy "
        "never blocks: the verdict is `allow` or `review`. The intra-adapter comparison "
        "(robust z of each module against its own family within the same adapter) uses "
        "**uncalibrated** thresholds. A `review` verdict or a high robust z marks an "
        "*investigation candidate*, not confirmed malicious content."
    )

    # ── Candidate selection ───────────────────────────────────────────────────
    section(2, "Candidate Selection Criteria")
    bullets([
        "HuggingFace Hub queried with filter `peft`, sorted by download count (most popular first)",
        "Repositories must contain `adapter_model.safetensors` (single-file adapters only)",
        "`adapter_config.json` fetched when present; the scanner uses it for the declared rank and LoRA scale",
        "Base model weights are never downloaded",
        f"Target sample size: {agg['run_params'].get('target_limit', '?')} repos maximum",
        "Selection is deterministic given the same HF Hub sort order at query time",
    ])

    # ── Limitations ───────────────────────────────────────────────────────────
    section(2, "Limitations")
    bullets([
        "No labeled ground truth — classification accuracy (precision, recall, F1) cannot be reported",
        "Only single-file `adapter_model.safetensors` adapters are covered; sharded multi-file adapters are excluded",
        "Files in which the scanner finds no analysable LoRA A/B pair (IA³, LoHa, full fine-tunes, …) are "
        "classified `unsupported_architecture` and excluded from the distributions.",
        "Selection is biased toward popular adapters (sorted by downloads); newly published or niche repos are under-represented",
        "Intra-adapter thresholds are uncalibrated; false-positive and false-negative rates at scale are unknown",
        "No reference profile of benign adapters was used; population-level comparison is not part of this report",
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
    lines.append(f"| of which degraded scans | {agg['counts'].get('degraded', 0)} |")
    lines.append(f"| Unsupported architecture (excluded from distributions) | {totals.get('unsupported_architecture', 0)} |")
    lines.append(f"| Download / analysis failures | {totals['failed']} |")
    lines.append(f"| Skipped (size exceeded, not cached, or other) | {totals['skipped']} |")
    lines.append("")

    section(3, "Verdict Action Distribution")
    dist_table("Action", action_dist, _ACTIONS)

    section(3, "Verdict Level Distribution")
    dist_table("Level", level_dist, _LEVELS)

    section(3, "Training State Distribution")
    dist_table("Training state", training_dist, _TRAINING_STATES)

    reason_dist = agg.get("reason_code_distribution", {})
    if reason_dist:
        section(3, "Verdict Reason Codes (adapters carrying each code)")
        lines.append("| Reason code | Adapters | Share of scanned |")
        lines.append("|---|---|---|")
        for code, count in sorted(reason_dist.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"| {code} | {count} | {count / total_success:.1%} |")
        lines.append("")

    section(3, "Intra-adapter Max Robust z (uncalibrated)")
    if pcts:
        para(
            f"Computed for {agg.get('n_with_intra_anomaly', 0)} of {totals['succeeded']} scanned "
            "adapters (it needs enough modules per family to compare)."
        )
        lines.append("| Percentile | max robust z |")
        lines.append("|---|---|")
        for key, val in sorted(pcts.items(), key=lambda x: int(x[0][1:])):
            lines.append(f"| {key} | {val:.2f} |")
        mean_z = agg.get("max_robust_z_mean")
        if mean_z is not None:
            lines.append(f"| mean | {mean_z:.2f} |")
        lines.append("")
    else:
        para("No intra-adapter comparison data available.")

    # ── Failure breakdown ─────────────────────────────────────────────────────
    section(2, "Record Status Breakdown")
    failure_counts = agg.get("failure_reason_counts", {})
    if failure_counts:
        lines.append("| Status | Count |")
        lines.append("|---|---|")
        for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {count} |")
        lines.append("")
    else:
        para("No records.")

    # ── Top investigation candidates ──────────────────────────────────────────
    section(2, "Top Investigation Candidates (by Intra-adapter Max Robust z)")
    para(
        "Adapters whose most deviant module departs most from its own family within the same "
        "adapter. The threshold is uncalibrated, and legitimate fine-tunes can concentrate their "
        "update in a few layers. They are **not confirmed malicious**; each should be reviewed in "
        "context: provenance, stated purpose, and behavioural testing."
    )
    top_z = agg.get("top_suspicious_by_max_robust_z", [])
    if top_z:
        lines.append("| Repository | max z | Outlier modules | Action | Level | Training | Top finding |")
        lines.append("|---|---|---|---|---|---|---|")
        for entry in top_z[:15]:
            findings = entry.get("top_findings") or []
            top_finding = findings[0][:60] if findings else "—"
            lines.append(
                f"| `{entry['repo_id']}` "
                f"| {entry['max_robust_z']:.1f} "
                f"| {entry['n_outlier_modules']} "
                f"| {entry['action']} "
                f"| {entry['level']} "
                f"| {entry['training_state']} "
                f"| {top_finding} |"
            )
        lines.append("")
    else:
        para("No intra-adapter comparison results to report.")

    section(2, "Top Investigation Candidates (by Structural Red Flags)")
    para(
        "Adapters carrying structural reason codes ("
        + ", ".join(f"`{c}`" for c in sorted(STRUCTURAL_REASON_CODES))
        + "). These describe what the file contains or how it is configured, not weight "
        "statistics; several have legitimate uses and warrant a provenance check, not a conclusion."
    )
    top_struct = agg.get("top_suspicious_by_structural_codes", [])
    if top_struct:
        lines.append("| Repository | Structural codes | Action | Level | max z |")
        lines.append("|---|---|---|---|---|")
        for entry in top_struct[:15]:
            z = entry.get("max_robust_z")
            z_str = f"{z:.1f}" if z is not None else "—"
            lines.append(
                f"| `{entry['repo_id']}` "
                f"| {', '.join(entry['structural_codes'])} "
                f"| {entry['action']} "
                f"| {entry['level']} "
                f"| {z_str} |"
            )
        lines.append("")
    else:
        para("No scanned adapter carried a structural reason code.")

    # ── Interpretation guide ──────────────────────────────────────────────────
    section(2, "Interpretation Guidance")
    bullets([
        "**allow:** no reason for review was found by the static checks. It is not a guarantee of safety.",
        "**review:** at least one reason (degraded parse, structural red flag, or intra-adapter outlier) "
        "warrants a human look; behavioural verification is recommended.",
        "**block:** only possible with a calibrated reference profile or under the strict policy; "
        "not expected in this report (default policy, no reference profile).",
        "**Level (LOW…CRITICAL):** the severity of the strongest verdict reason, not a probability of malice.",
        "**init_only:** every LoRA B matrix is zero, the standard PEFT initial state — the adapter "
        "changes nothing and weight statistics are not informative.",
        "**partial:** some modules trained while others remain at zero-init. Uncommon in standard "
        "fine-tuning; warrants provenance review.",
        "**max robust z:** the largest robust z of any module feature relative to modules of the same "
        "family in the same adapter. Uncalibrated; a statistical weight signal, not behavioural evidence.",
    ])

    para(
        "For adapters with a `review` verdict, recommended next steps: "
        "(1) verify the adapter's stated training purpose and provenance; "
        "(2) run behavioural verification (AdapterSentry M2, planned); "
        "(3) if behavioural evidence is found, report to the HuggingFace Hub moderation team."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown report written: %s", report_path)
