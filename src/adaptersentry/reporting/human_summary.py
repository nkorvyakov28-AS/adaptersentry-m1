"""render_human_summary — M1-RPT-02 CLI human summary renderer.

Produces a fixed-block text summary from an AdapterReport:

Compact (default):
  VERDICT block    — risk level, confidence, recommended action
  TOP SIGNALS      — top-3 sub-scores with lead reason
  FINDINGS         — one-line finding list (truncated)
  Hint to run --verbose for full breakdown

Verbose (--verbose):
  Everything above, plus:
  SCORE BREAKDOWN  — all 7 sub-scores with weights and reasons
  TOP SUSPICIOUS LAYERS  — PerLayerFinding list (up to 10)
  ANALYSIS QUALITY — parse coverage, metadata completeness, etc.

All wording is machine-stable: no ephemeral timestamps or run-specific
identifiers appear in the VERDICT/SIGNALS/BREAKDOWN blocks.
"""

from __future__ import annotations

from adaptersentry.schemas.adapter_report import AdapterReport, AnalysisMode, ParseStatus
from adaptersentry.schemas.finding import Severity

# ANSI colour codes
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_CYAN   = "\033[36m"
_BLUE   = "\033[34m"

_SEV_COLOUR = {
    Severity.CRITICAL: _RED + _BOLD,
    Severity.HIGH:     _RED,
    Severity.MEDIUM:   _YELLOW,
    Severity.LOW:      _GREEN,
}
_CONF_COLOUR = {"high": _GREEN, "medium": _YELLOW, "low": _RED}
_ACTION_COLOUR = {"allow": _GREEN, "review": _YELLOW, "block": _RED + _BOLD}


def _c(text: str, code: str, no_color: bool) -> str:
    return text if no_color else f"{code}{text}{_RESET}"


def _severity_colour(sev: Severity, no_color: bool) -> str:
    return _c(sev.value, _SEV_COLOUR.get(sev, ""), no_color)


def _shorten_layer(layer_name: str, width: int = 40) -> str:
    """Return a short representation of a layer path for display."""
    parts = layer_name.split(".")
    if len(parts) <= 3:
        return layer_name
    # Keep last 3 parts: e.g. "…self_attn.q_proj"
    suffix = ".".join(parts[-2:])
    try:
        idx = next(p for p in parts if p.isdigit())
        return f"layers[{idx}].{suffix}"
    except StopIteration:
        return f"….{suffix}"


def render_human_summary(
    report: AdapterReport,
    *,
    verbose: bool = False,
    no_color: bool = False,
) -> str:
    """Render a human-readable M1-RPT-02 summary from an AdapterReport.

    Computes score_breakdown, confidence_score, quality_score, and
    top_layer_findings inline — no pre-computation required.

    Args:
        report:   Completed AdapterReport.
        verbose:  If True, render full breakdown and per-layer details.
        no_color: If True, omit ANSI escape sequences.

    Returns:
        Multi-line string ready for stdout.
    """
    from adaptersentry.reporting.per_layer import compute_per_layer_findings
    from adaptersentry.scoring.confidence import compute_confidence_score, compute_quality_score
    from adaptersentry.scoring.score_breakdown import compute_score_breakdown

    # Compute derived scores
    try:
        sb = compute_score_breakdown(report)
    except Exception:
        sb = None

    try:
        qs = compute_quality_score(report)
        cs = compute_confidence_score(report, qs)
    except Exception:
        qs = None
        cs = None

    plf = []
    if verbose:
        try:
            plf = compute_per_layer_findings(
                report.tensor_records,
                inter_layer_features=report.inter_layer_similarity_features,
                top_k=10,
            )
        except Exception:
            plf = []

    lines: list[str] = []
    rs = report.risk_summary

    # ── Header ────────────────────────────────────────────────────────────────
    ps = report.parse_status
    ps_tag = f"  [parse:{ps.value}]"
    ps_col = _RED + _BOLD if ps == ParseStatus.FAILED else _YELLOW if ps == ParseStatus.DEGRADED else _GREEN
    lines.append(
        _c("AdapterSentry M1 — Static Analysis Report", _BOLD, no_color)
        + _c(ps_tag, ps_col, no_color)
    )
    lines.append("=" * 60)

    path_str = report.scan_target.path
    size_str = ""
    if report.scan_target.file_size_bytes is not None:
        size_str = f"  ({report.scan_target.file_size_bytes / (1024*1024):.1f} MB)"
    lines.append(f"Target:   {path_str}{size_str}")
    mode_label = "degraded" if report.analysis_mode == AnalysisMode.DEGRADED else "full"
    lines.append(
        f"Layers:   {rs.n_layers}  ·  mode: {mode_label}"
        f"  ·  training: {rs.training_status.value}"
    )

    if report.analysis_mode == AnalysisMode.DEGRADED:
        lines.append(_c("⚠  DEGRADED ANALYSIS — some detectors failed or skipped", _YELLOW, no_color))
    elif report.analysis_mode == AnalysisMode.FAILED:
        lines.append(_c("✗  ANALYSIS FAILED", _RED + _BOLD, no_color))

    # ── VERDICT block ────────────────────────────────────────────────────────
    lines.append("")
    lines.append(_c("VERDICT", _BOLD, no_color))

    ens_level = rs.ensemble_risk_level
    lines.append(
        f"  Risk:        {_severity_colour(ens_level, no_color)}"
        f"  (ensemble {rs.ensemble_score:.1f}/100 · rule {rs.overall_risk}/100)"
    )
    # Warn when rule score is dramatically higher than ensemble. The rule scorer
    # is additive (each flag adds weight, capped at 100) and inflates on adapters
    # with many layers. The ensemble score is the calibrated metric for verdict.
    if rs.overall_risk >= 75 and rs.ensemble_score < 25:
        lines.append(
            _c(
                f"  ↳ Note: rule score ({rs.overall_risk}/100) is inflated by additive "
                f"flagging across {rs.n_layers} layers. "
                f"Ensemble ({rs.ensemble_score:.1f}/100) is the calibrated risk metric.",
                _DIM, no_color,
            )
        )

    if cs is not None:
        vc = cs.verdict_certainty
        conf_col = _CONF_COLOUR.get(vc, "")
        conf_str = _c(vc, conf_col, no_color)
        lines.append(f"  Confidence:  {conf_str}  (overall {cs.overall_confidence:.2f})")
        if vc != "high":
            upsell = "run in full mode for higher confidence" if mode_label != "full" \
                else "upgrade to paid tier for detailed breakdown"
            lines.append(_c(f"               ↳ {upsell}", _DIM, no_color))
    else:
        lines.append("  Confidence:  n/a")

    # recommended_action
    level_val = ens_level.value
    action = "block" if level_val in ("HIGH", "CRITICAL") else "review" if level_val == "MEDIUM" else "allow"
    m2_note = "  →  M2 behavioral sandbox recommended" if level_val in ("HIGH", "CRITICAL") else ""
    action_col = _ACTION_COLOUR.get(action, "")
    lines.append(f"  Action:      {_c(action, action_col, no_color)}{m2_note}")

    # ── TOP SIGNALS block (top-3 sub-scores by weighted_contribution) ────────
    if sb is not None and sb.sub_scores:
        lines.append("")
        lines.append(_c("TOP RISK SIGNALS  (top-3 sub-scores by contribution)", _BOLD, no_color))
        top3 = sorted(sb.sub_scores, key=lambda s: s.weighted_contribution, reverse=True)[:3]
        for ss in top3:
            score_col = _RED if ss.normalized_score > 0.6 else _YELLOW if ss.normalized_score > 0.3 else ""
            score_str = _c(f"{ss.normalized_score:.2f}", score_col, no_color)
            reason = f"  ·  {ss.top_reasons[0]}" if ss.top_reasons else ""
            lines.append(f"  {ss.family:<16s} {score_str}{reason}")

    # ── Findings (compact) ────────────────────────────────────────────────────
    if report.findings:
        lines.append("")
        sorted_findings = sorted(report.findings, key=lambda f: list(Severity).index(f.severity))
        shown = sorted_findings[:5]
        rest = len(sorted_findings) - len(shown)
        parts = []
        for f in shown:
            col = _SEV_COLOUR.get(f.severity, "")
            parts.append(f"[{_c(f.severity.value, col, no_color)}] {f.rule_id}")
        findings_line = "Findings:  " + "  ".join(parts)
        if rest:
            findings_line += _c(f"  +{rest} more", _DIM, no_color)
        lines.append(findings_line)
    else:
        lines.append("")
        lines.append(_c("No findings.", _GREEN, no_color))

    # ── Verbose sections ──────────────────────────────────────────────────────
    if verbose:
        _render_score_breakdown(lines, sb, no_color)
        _render_per_layer_findings(lines, plf, no_color)
        if qs is not None:
            _render_quality(lines, qs, no_color)
    else:
        lines.append("")
        lines.append(
            _c("Run with --verbose for full score breakdown and per-layer details.", _DIM, no_color)
        )

    lines.append("")
    return "\n".join(lines)


def _render_score_breakdown(lines: list[str], sb, no_color: bool) -> None:
    if sb is None:
        return
    lines.append("")
    lines.append(_c("SCORE BREAKDOWN", _BOLD, no_color))
    lines.append(_c(f"  {'family':<16s}  {'score':>5}  {'weight':>6}  {'contrib':>7}  reason", _DIM, no_color))
    lines.append("  " + "─" * 70)
    for ss in sb.sub_scores:
        score_col = _RED if ss.normalized_score > 0.6 else _YELLOW if ss.normalized_score > 0.3 else _GREEN
        score_str = _c(f"{ss.normalized_score:.3f}", score_col, no_color)
        reason = f"  ·  {ss.top_reasons[0]}" if ss.top_reasons else ""
        cap_flag = _c(" [cap]", _YELLOW, no_color) if ss.cap_applied else ""
        floor_flag = _c(" [floor]", _CYAN, no_color) if ss.floor_applied else ""
        lines.append(
            f"  {ss.family:<16s}  {score_str}  ×{ss.weight:.2f}  ={ss.weighted_contribution:.4f}"
            f"{cap_flag}{floor_flag}{reason}"
        )
    lines.append("  " + "─" * 70)
    lines.append(f"  Total weighted:  {sb.total_weighted:.4f}  →  adjusted: {sb.adjusted_total_weighted:.4f}"
                 f"  (policy v{sb.applied_policy_version or 'n/a'})")
    if sb.escalation_rules_fired:
        for rule in sb.escalation_rules_fired:
            name = rule.split(":")[0].strip()
            lines.append(_c(f"  ↑ Escalation: {name}", _YELLOW, no_color))


def _render_per_layer_findings(lines: list[str], plf: list, no_color: bool) -> None:
    if not plf:
        return
    lines.append("")
    lines.append(_c(f"TOP SUSPICIOUS LAYERS  ({len(plf)} shown)", _BOLD, no_color))
    for finding in plf:
        sev_col = _SEV_COLOUR.get(finding.severity, "")
        sev_str = _c(f"[{finding.severity.value}]", sev_col, no_color)
        short = _shorten_layer(finding.layer_name)
        families = ", ".join(finding.triggered_families)
        lines.append(f"  #{finding.rank}  {short:<40s}  {sev_str}  {families}")
        if finding.signals:
            signal_line = "  ·  " + "  ·  ".join(finding.signals[:3])
            lines.append(_c(f"       {signal_line}", _DIM, no_color))


def _render_quality(lines: list[str], qs, no_color: bool) -> None:
    lines.append("")
    lines.append(_c("ANALYSIS QUALITY", _BOLD, no_color))

    def _pct(v: float) -> str:
        return f"{v * 100:.0f}%"

    def _quality_col(v: float) -> str:
        return _GREEN if v >= 0.9 else _YELLOW if v >= 0.6 else _RED

    for label, val in (
        ("Parse coverage", qs.parse_coverage),
        ("Metadata", qs.metadata_completeness),
        ("Feature complete", qs.feature_completeness),
        ("Degenerate ratio", qs.degenerate_ratio),
    ):
        col = _quality_col(val) if label != "Degenerate ratio" else (_RED if val > 0.1 else _GREEN)
        lines.append(f"  {label:<18s}  {_c(_pct(val), col, no_color)}")
    lines.append(
        _c(f"  Overall quality:   {_pct(qs.overall_quality)}  "
           f"({qs.n_layers_parsed_ok}/{qs.n_layers_total} layers parsed OK)", _DIM, no_color)
    )
