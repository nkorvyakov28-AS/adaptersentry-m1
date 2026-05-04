"""GitHub Actions integration helper.

AdapterSentry can be used in GitHub Actions workflows via the CLI.
No Python code is needed for the integration itself — the CLI handles
everything, including SARIF output that GitHub code scanning can ingest
directly.

Example workflow step::

    - name: Scan LoRA adapter
      run: |
        pip install adaptersentry
        adaptersentry scan adapter.safetensors --format sarif --output results.sarif
        # Exit code 2 = findings above threshold (if --fail-on used)

    - name: Upload SARIF results
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: results.sarif
      if: always()

This module provides a helper that writes a GitHub Actions output variable
and step summary when run inside a workflow (``GITHUB_OUTPUT`` env var set).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_github_actions() -> bool:
    """Return True when running inside a GitHub Actions runner."""
    return os.environ.get("GITHUB_ACTIONS") == "true"


def set_output(name: str, value: str) -> None:
    """Write a GitHub Actions step output variable.

    Does nothing outside of a GitHub Actions environment.
    """
    if not is_github_actions():
        return
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{name}={value}\n")


def write_step_summary(content: str) -> None:
    """Append content to the GitHub Actions step summary.

    Does nothing outside of a GitHub Actions environment.
    """
    if not is_github_actions():
        return
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(content + "\n")


def emit_scan_summary(report: object) -> None:  # report: AdapterReport
    """Emit scan results as GitHub Actions outputs and step summary.

    Args:
        report: Completed AdapterReport object.
    """
    if not is_github_actions():
        return

    rs = getattr(report, "risk_summary", None)
    if rs is None:
        return

    set_output("risk_level", rs.ensemble_risk_level.value)
    set_output("ensemble_score", f"{rs.ensemble_score:.1f}")
    set_output("n_findings", str(rs.n_findings))

    md_lines = [
        "## AdapterSentry M1 Scan Results",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Risk | **{rs.ensemble_risk_level.value}** (score {rs.ensemble_score:.1f}/100) |",
        f"| Findings | {rs.n_findings} |",
        f"| Layers scanned | {rs.n_layers} |",
        f"| Training status | {rs.training_status.value} |",
    ]
    findings = getattr(report, "findings", [])
    if findings:
        md_lines += ["", "### Top findings", ""]
        for f in findings[:5]:
            md_lines.append(f"- **{f.severity.value}** `{f.rule_id}` — {f.title}")

    write_step_summary("\n".join(md_lines))
