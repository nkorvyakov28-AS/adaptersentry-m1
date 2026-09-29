"""Human-readable text report for a ScanResult 2.0.0.

Every string that originates in the adapter file (paths, tensor keys, module
names, messages built from them) goes through ``safe_text`` so control or bidi
characters cannot forge terminal output.
"""

from __future__ import annotations

from adaptersentry.reporting.sanitize import safe_text
from adaptersentry.schemas.severity import Severity
from adaptersentry.schemas.result import ScanResult

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_ACTION_COLOUR = {"allow": _GREEN, "review": _YELLOW, "block": _RED}
_SEV_COLOUR = {Severity.LOW: _DIM, Severity.MEDIUM: _YELLOW, Severity.HIGH: _RED, Severity.CRITICAL: _RED}
_SEV_ORDER = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}


def _c(text: str, code: str, no_color: bool) -> str:
    return text if no_color or not code else f"{code}{text}{_RESET}"


def render(result: ScanResult, *, no_color: bool = False, max_findings: int = 15) -> str:
    """Render a ScanResult as text."""
    v = result.verdict
    lines: list[str] = []
    name = safe_text(result.artifact.provenance.path) if result.artifact else "(unreadable file)"
    lines.append(_c(f"AdapterSentry {safe_text(result.scan.analyzer_version)} · {name}", _BOLD, no_color))
    lines.append("=" * 64)

    behavioural = " · behavioural check recommended" if v.m2_recommended else ""
    lines.append(
        "Verdict:  "
        + _c(v.action.upper(), _ACTION_COLOUR[v.action], no_color)
        + f" ({v.level.value}) · confidence {v.confidence.level}{behavioural}"
    )
    lines.append(f"Status:   {result.status} · policy {result.scan.policy} · mode {result.scan.mode}")
    if v.reasons:
        lines.append("Why:")
        for r in v.reasons:
            lines.append(f"  - [{_c(r.severity.value, _SEV_COLOUR[r.severity], no_color)}] "
                         f"{safe_text(r.code)}: {safe_text(r.message)}")

    a = result.adapter
    if a.format != "unknown":
        rank = a.rank_actual
        rank_txt = f"r={rank.max}" if rank and rank.min == rank.max else (f"r={rank.distinct}" if rank else "r=?")
        if a.rank_declared is not None:
            rank_txt += f" (declared {a.rank_declared})"
        alpha = f", α={a.lora_alpha:g}" if a.lora_alpha is not None else ""
        layers = f", {a.n_model_layers} layers" if a.n_model_layers else ""
        lines.append(
            f"Adapter:  {a.format}, family {safe_text(a.base_family)} ({a.base_family_source}){layers}, "
            f"{rank_txt}{alpha}, scaling {a.scaling}, training {a.training_state}"
        )
    c = result.coverage
    lines.append(f"Coverage: {c.n_modules_analyzed}/{c.n_modules} modules analysed; "
                 f"{c.n_not_analyzed} tensor(s) not analysed")
    for item in c.not_analyzed[:5]:
        lines.append(_c(f"  · {safe_text(item.tensor_key)} — {safe_text(item.reason)}", _DIM, no_color))

    intra = result.anomaly.intra
    if intra is not None:
        gini = f" · energy Gini {intra.energy_gini:.2f}" if intra.energy_gini is not None else ""
        lines.append(f"Intra:    max robust z {intra.max_robust_z:.1f} · "
                     f"{intra.n_outlier_modules} outlier module(s){gini}")
    if result.anomaly.top_contributors:
        lines.append("Top contributors:")
        for tc in result.anomaly.top_contributors[:5]:
            where = f"layer {tc.layer}" if tc.layer is not None else "—"
            lines.append(f"  z={tc.z:+.1f}  {safe_text(tc.feature)} of {safe_text(tc.module_type)}, {where}")

    if result.findings:
        lines.append(f"Findings ({len(result.findings)}):")
        for f in sorted(result.findings, key=lambda f: _SEV_ORDER[f.severity])[:max_findings]:
            loc = ""
            if f.locations:
                l0 = f.locations[0]
                parts = [p for p in (
                    f"layer {l0.layer}" if l0.layer is not None else None,
                    safe_text(l0.module) if l0.module else None,
                ) if p]
                loc = f"  ({', '.join(parts)})" if parts else ""
            lines.append(f"  [{_c(f.severity.value, _SEV_COLOUR[f.severity], no_color)}] "
                         f"{safe_text(f.rule_id)} — {safe_text(f.title)}{loc}")
        if len(result.findings) > max_findings:
            lines.append(_c(f"  … and {len(result.findings) - max_findings} more", _DIM, no_color))

    if v.confidence.limiting_factors:
        lines.append(_c("Limits: " + "; ".join(safe_text(x) for x in v.confidence.limiting_factors),
                        _DIM, no_color))
    for e in result.errors:
        lines.append(_c(f"Error: {safe_text(e.code)}: {safe_text(e.message)}", _RED, no_color))
    return "\n".join(lines) + "\n"
