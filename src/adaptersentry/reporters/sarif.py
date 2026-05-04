"""SARIF 2.1.0 reporter for AdapterReport.

Produces SARIF output suitable for ingestion by GitHub code scanning and
other SARIF-compatible security tooling.

GitHub code scanning requirements honoured:
- ``runs[].tool.driver.rules`` populated for every rule_id that appears in results.
- ``results[].level`` set to "error" / "warning" / "note" based on severity.
- ``results[].properties.security-severity`` set to a CVSS-like score (0.0–10.0)
  so GitHub maps findings to its Low/Medium/High/Critical buckets.
- ``results[].locations`` uses ``physicalLocation.artifactLocation`` pointing to
  the adapter file, plus ``logicalLocations`` for the tensor layer name when
  available.  GitHub requires at least one location per result.

Reference: https://docs.github.com/en/code-security/code-scanning/integrating-with-code-scanning/sarif-support-for-github-code-scanning
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any
from urllib.parse import quote as _url_quote

from adaptersentry.schemas.adapter_report import AdapterReport
from adaptersentry.schemas.finding import Finding, Severity, RULE_CATALOG

# SARIF spec URI
_SARIF_SCHEMA = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/"
    "Schemata/sarif-schema-2.1.0.json"
)
_SARIF_VERSION = "2.1.0"

# Map AdapterSentry severity → SARIF level
_LEVEL_MAP: dict[Severity, str] = {
    Severity.CRITICAL: "error",
    Severity.HIGH: "error",
    Severity.MEDIUM: "warning",
    Severity.LOW: "note",
}

# Map AdapterSentry severity → CVSS-like security-severity for GitHub
_SECURITY_SEVERITY_MAP: dict[Severity, str] = {
    Severity.CRITICAL: "9.0",
    Severity.HIGH: "7.5",
    Severity.MEDIUM: "5.0",
    Severity.LOW: "2.5",
}


def _rule_descriptor(rule_id: str, finding: Finding) -> dict[str, Any]:
    """Build a SARIF rule descriptor for a finding."""
    catalog_entry = RULE_CATALOG.get(rule_id)
    if catalog_entry:
        _, title, remediation = catalog_entry
    else:
        title = rule_id.replace("_", " ").title()
        remediation = None

    desc: dict[str, Any] = {
        "id": rule_id,
        "name": _to_camel(rule_id),
        "shortDescription": {"text": title},
        "fullDescription": {
            "text": (
                f"AdapterSentry M1 detected: {title}. "
                f"Severity: {finding.severity.value}."
            )
        },
        "properties": {
            "security-severity": _SECURITY_SEVERITY_MAP[finding.severity],
            "tags": ["security", "ml-model", "lora-adapter"],
        },
    }
    if remediation:
        desc["help"] = {"text": remediation}
    return desc


def _to_camel(rule_id: str) -> str:
    """Convert SNAKE_CASE rule_id to CamelCase rule name."""
    return "".join(part.capitalize() for part in rule_id.lower().split("_"))


def _artifact_uri(path: str) -> str:
    """Produce a file:// URI from an absolute or relative path."""
    p = Path(path)
    if p.is_absolute():
        return p.as_uri()
    # relative path — encode each segment
    encoded = "/".join(_url_quote(seg, safe="") for seg in p.parts)
    return encoded


def _result_for_finding(
    finding: Finding,
    artifact_index: int,
) -> dict[str, Any]:
    """Convert a Finding to a SARIF result object."""
    flags = finding.evidence.get("flags", [])
    message_text = flags[0] if flags else finding.title

    locations: list[dict[str, Any]] = []

    if finding.affected_layers:
        for layer_name in finding.affected_layers[:3]:  # cap at 3 per result
            loc: dict[str, Any] = {
                "physicalLocation": {
                    "artifactLocation": {"index": artifact_index},
                },
                "logicalLocations": [
                    {
                        "name": layer_name,
                        "fullyQualifiedName": layer_name,
                        "kind": "function",  # closest SARIF analogue to a model layer
                    }
                ],
            }
            locations.append(loc)
    else:
        # Adapter-level finding (no specific layer)
        locations.append(
            {"physicalLocation": {"artifactLocation": {"index": artifact_index}}}
        )

    return {
        "ruleId": finding.rule_id,
        "level": _LEVEL_MAP[finding.severity],
        "message": {"text": message_text},
        "locations": locations,
        "properties": {
            "severity": finding.severity.value,
            "security-severity": _SECURITY_SEVERITY_MAP[finding.severity],
            "confidence": finding.confidence,
        },
    }


def render(report: AdapterReport) -> dict[str, Any]:
    """Convert AdapterReport to a SARIF 2.1.0 dict.

    Args:
        report: Completed M1 AdapterReport.

    Returns:
        Dict conforming to SARIF 2.1.0 — ready for json.dumps().
    """
    # De-duplicate rules: one descriptor per unique rule_id
    seen_rules: dict[str, dict[str, Any]] = {}
    for finding in report.findings:
        if finding.rule_id not in seen_rules:
            seen_rules[finding.rule_id] = _rule_descriptor(finding.rule_id, finding)

    rules = list(seen_rules.values())

    artifact_uri = _artifact_uri(report.scan_target.path)
    artifacts: list[dict[str, Any]] = [
        {
            "location": {"uri": artifact_uri},
            "mimeType": "application/octet-stream",
            "description": {"text": "LoRA adapter safetensors file"},
        }
    ]
    if report.scan_target.file_size_bytes is not None:
        artifacts[0]["length"] = report.scan_target.file_size_bytes

    results = [_result_for_finding(f, 0) for f in report.findings]

    # Include errors as SARIF notifications
    notifications: list[dict[str, Any]] = [
        {
            "message": {"text": f"[{err.category.value}] {err.code}: {err.message}"},
            "level": "error" if err.category.value == "malformed" else "warning",
        }
        for err in report.errors
    ]

    run: dict[str, Any] = {
        "tool": {
            "driver": {
                "name": report.tool.name,
                "version": report.tool.version,
                "semanticVersion": report.tool.version,
                "informationUri": report.tool.informationUri,
                "rules": rules,
            }
        },
        "artifacts": artifacts,
        "results": results,
        "properties": {
            "schema_version": report.schema_version,
            "analysis_mode": report.analysis_mode.value,
            "ensemble_score": report.risk_summary.ensemble_score,
            "risk_level": report.risk_summary.ensemble_risk_level.value,
            "training_status": report.risk_summary.training_status.value,
            "n_layers": report.risk_summary.n_layers,
        },
    }

    if notifications:
        run["invocations"] = [
            {
                "executionSuccessful": report.risk_summary.n_layers > 0,
                "toolExecutionNotifications": notifications,
            }
        ]

    return {
        "$schema": _SARIF_SCHEMA,
        "version": _SARIF_VERSION,
        "runs": [run],
    }


def render_json(report: AdapterReport, indent: int = 2) -> str:
    """Render SARIF output as a JSON string."""
    return _json.dumps(render(report), indent=indent)


def write(report: AdapterReport, path: Path, indent: int = 2) -> None:
    """Write SARIF output to a file."""
    path.write_text(render_json(report, indent=indent), encoding="utf-8")
