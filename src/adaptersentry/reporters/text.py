"""Human-readable text reporter for AdapterReport.

Produces concise terminal output suitable for developer workflows.
Degraded and malformed states are prominently shown.
"""

from __future__ import annotations

from adaptersentry.schemas.adapter_report import AdapterReport, AnalysisMode, ParseStatus
from adaptersentry.schemas.errors import ErrorCategory
from adaptersentry.schemas.finding import Severity

# ANSI colour codes (disabled when --no-color)
_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_DIM = "\033[2m"

_SEVERITY_COLOUR = {
    Severity.CRITICAL: _RED + _BOLD,
    Severity.HIGH: _RED,
    Severity.MEDIUM: _YELLOW,
    Severity.LOW: _GREEN,
}


def _colour(text: str, code: str, no_color: bool) -> str:
    if no_color:
        return text
    return f"{code}{text}{_RESET}"


def render(report: AdapterReport, no_color: bool = False) -> str:
    """Render AdapterReport as a human-readable text string.

    Args:
        report: Completed M1 AdapterReport.
        no_color: If True, omit ANSI escape sequences.

    Returns:
        Multi-line string ready to write to stdout.
    """
    lines: list[str] = []
    rs = report.risk_summary

    # ── Header ────────────────────────────────────────────────────────────────
    ps = report.parse_status
    ps_tag = f"  [parse:{ps.value}]"
    if ps == ParseStatus.FAILED:
        ps_colour = _RED + _BOLD
    elif ps == ParseStatus.DEGRADED:
        ps_colour = _YELLOW
    else:
        ps_colour = _GREEN
    lines.append(
        _colour("AdapterSentry M1 — Static Analysis Report", _BOLD, no_color)
        + _colour(ps_tag, ps_colour, no_color)
    )
    lines.append("=" * 60)
    lines.append(f"Target:   {report.scan_target.path}")
    if report.scan_target.file_size_bytes is not None:
        size_mb = report.scan_target.file_size_bytes / (1024 * 1024)
        lines.append(f"Size:     {size_mb:.1f} MB")
    lines.append(f"Scanned:  {report.completed_at}")

    # ── Analysis mode warning ──────────────────────────────────────────────────
    if report.analysis_mode == AnalysisMode.DEGRADED:
        lines.append(_colour("⚠  DEGRADED ANALYSIS — some detectors failed", _YELLOW, no_color))
    elif report.analysis_mode == AnalysisMode.FAILED:
        lines.append(_colour("✗  ANALYSIS FAILED", _RED, no_color))

    # ── Risk summary ──────────────────────────────────────────────────────────
    lines.append("")
    ens_level = rs.ensemble_risk_level
    ens_col = _SEVERITY_COLOUR.get(ens_level, "")
    lines.append(
        f"Risk:     {_colour(ens_level.value, ens_col, no_color)}"
        f"  (ensemble {rs.ensemble_score:.1f}/100 · rule {rs.overall_risk}/100)"
    )
    lines.append(
        f"Status:   {rs.training_status.value}"
        f" · {rs.n_layers} layer(s)"
        f" · {rs.false_positive_suppressed} init artifact(s) suppressed"
    )
    if rs.cross_layer_consistency < 0.3 and rs.n_layers > 1:
        lines.append(
            _colour(
                f"  ↳ Low cross-layer consistency ({rs.cross_layer_consistency:.3f}) — anomaly concentration",
                _YELLOW,
                no_color,
            )
        )

    # ── Adapter metadata ──────────────────────────────────────────────────────
    meta = report.adapter_metadata
    if meta.base_model or meta.claimed_rank:
        lines.append("")
        if meta.base_model:
            lines.append(f"Base model: {meta.base_model}")
        if meta.claimed_rank is not None:
            lines.append(f"Rank:       r={meta.claimed_rank}")
        if meta.target_modules:
            lines.append(f"Targets:    {', '.join(meta.target_modules[:6])}")

    # ── Findings ──────────────────────────────────────────────────────────────
    if report.findings:
        lines.append("")
        lines.append(_colour(f"Findings ({len(report.findings)}):", _BOLD, no_color))
        for finding in sorted(
            report.findings,
            key=lambda f: list(Severity).index(f.severity),
        ):
            col = _SEVERITY_COLOUR.get(finding.severity, "")
            sev = _colour(f"[{finding.severity.value:8s}]", col, no_color)
            layer_hint = ""
            if finding.affected_layers:
                layer_hint = f"  ({', '.join(finding.affected_layers[:2])}{'…' if len(finding.affected_layers) > 2 else ''})"
            lines.append(f"  {sev}  {finding.rule_id}{layer_hint}")
            lines.append(_colour(f"             {finding.title}", _DIM, no_color))
    else:
        lines.append("")
        lines.append(_colour("No findings.", _GREEN, no_color))

    # ── Errors ────────────────────────────────────────────────────────────────
    if report.errors:
        lines.append("")
        lines.append(_colour(f"Errors ({len(report.errors)}):", _YELLOW, no_color))
        for err in report.errors:
            cat_col = _RED if err.category == ErrorCategory.MALFORMED else _YELLOW
            lines.append(
                f"  {_colour(err.category.value.upper(), cat_col, no_color)}"
                f"  {err.code}: {err.message}"
            )
            if err.detail:
                lines.append(_colour(f"    {err.detail}", _DIM, no_color))

    # ── Top-10 layers by delta_norm_ratio ─────────────────────────────────────
    ranked = sorted(
        [tr for tr in report.tensor_records if tr.norm_features is not None],
        key=lambda tr: tr.norm_features.delta_norm_ratio,  # type: ignore[union-attr]
        reverse=True,
    )
    if ranked:
        lines.append("")
        lines.append(_colour("ΔW norm — top layers by delta_norm_ratio:", _BOLD, no_color))
        lines.append(
            _colour(
                "  (ratio = ||B@A||_F / (||A||_F·||B||_F); "
                "future: divide by rank/lora_alpha for cross-adapter comparison)",
                _DIM,
                no_color,
            )
        )
        for i, tr in enumerate(ranked[:10], 1):
            nf = tr.norm_features  # type: ignore[union-attr]
            short = tr.layer_name.split(".")[-3] if "." in tr.layer_name else tr.layer_name
            lines.append(
                f"  {i:2d}. {short:<28s}"
                f"  ratio={nf.delta_norm_ratio:.4f}"
                f"  fro={nf.fro_norm_delta:.4f}"
                f"  max={nf.max_abs_delta:.4f}"
                f"  mean={nf.mean_abs_delta:.4f}"
            )
        if len(ranked) > 10:
            lines.append(_colour(f"  … and {len(ranked) - 10} more layer(s)", _DIM, no_color))

    lines.append("")
    return "\n".join(lines)
