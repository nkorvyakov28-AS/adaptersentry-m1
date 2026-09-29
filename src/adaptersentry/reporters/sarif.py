"""SARIF 2.1.0 report for a ScanResult 2.0.0 (GitHub code scanning compatible).

- One rule descriptor per rule id; ``security-severity`` maps to GitHub's buckets.
- Every finding is a result; its locations point at the adapter file and, as
  logical locations, at the exact layer / module / tensor key.
- The verdict itself is a result (rule ``ADAPTER_VERDICT``) whenever it is not
  "allow", so a code-scanning gate sees review/block even without findings.
- The artifact URI is the file name unless the scan recorded a full path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from adaptersentry.schemas.severity import Severity
from adaptersentry.schemas.result import Finding, ScanResult

_SARIF_SCHEMA = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"
_LEVEL = {Severity.CRITICAL: "error", Severity.HIGH: "error", Severity.MEDIUM: "warning", Severity.LOW: "note"}
_SECURITY_SEVERITY = {Severity.CRITICAL: "9.0", Severity.HIGH: "7.5", Severity.MEDIUM: "5.0", Severity.LOW: "2.5"}
_ACTION_SEVERITY = {"review": Severity.MEDIUM, "block": Severity.HIGH}


def _uri(path: str) -> str:
    p = Path(path)
    return p.as_uri() if p.is_absolute() else "/".join(quote(seg, safe="") for seg in p.parts)


def _camel(rule_id: str) -> str:
    return "".join(part.capitalize() for part in rule_id.lower().split("_"))


def _rule(rule_id: str, title: str, severity: Severity, remediation: str | None) -> dict[str, Any]:
    desc: dict[str, Any] = {
        "id": rule_id,
        "name": _camel(rule_id),
        "shortDescription": {"text": title},
        "fullDescription": {"text": f"AdapterSentry M1: {title}."},
        "properties": {"security-severity": _SECURITY_SEVERITY[severity],
                       "tags": ["security", "ml-model", "lora-adapter"]},
    }
    if remediation:
        desc["help"] = {"text": remediation}
    return desc


def _locations(finding: Finding) -> list[dict[str, Any]]:
    if not finding.locations:
        return [{"physicalLocation": {"artifactLocation": {"index": 0}}}]
    out = []
    for loc in finding.locations[:3]:
        name = ".".join(p for p in (
            f"layers.{loc.layer}" if loc.layer is not None else None,
            f"experts.{loc.expert}" if loc.expert is not None else None,
            loc.module,
        ) if p) or (loc.tensor_key or "adapter")
        out.append({
            "physicalLocation": {"artifactLocation": {"index": 0}},
            "logicalLocations": [{"name": name, "fullyQualifiedName": loc.tensor_key or name, "kind": "member"}],
        })
    return out


def render(result: ScanResult) -> dict[str, Any]:
    """Convert a ScanResult to a SARIF 2.1.0 document (a dict ready for json.dumps)."""
    rules: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []

    v = result.verdict
    if v.action != "allow":
        sev = _ACTION_SEVERITY[v.action]
        rules["ADAPTER_VERDICT"] = _rule("ADAPTER_VERDICT", f"Adapter requires {v.action}", sev,
                                         "Follow the reasons in the result; run a behavioural check.")
        results.append({
            "ruleId": "ADAPTER_VERDICT",
            "level": _LEVEL[sev],
            "message": {"text": f"{v.action.upper()} ({v.level.value}): "
                                + "; ".join(f"{r.code}: {r.message}" for r in v.reasons)},
            "locations": [{"physicalLocation": {"artifactLocation": {"index": 0}}}],
            "properties": {"security-severity": _SECURITY_SEVERITY[sev], "confidence": v.confidence.level},
        })

    for f in result.findings:
        rules.setdefault(f.rule_id, _rule(f.rule_id, f.title, f.severity, f.remediation))
        results.append({
            "ruleId": f.rule_id,
            "level": _LEVEL[f.severity],
            "message": {"text": f.message or f.title},
            "locations": _locations(f),
            "properties": {"severity": f.severity.value, "security-severity": _SECURITY_SEVERITY[f.severity]},
        })

    path = result.artifact.provenance.path if result.artifact and result.artifact.provenance.path else "adapter"
    artifact: dict[str, Any] = {"location": {"uri": _uri(path)}, "mimeType": "application/octet-stream",
                                "description": {"text": "LoRA adapter safetensors file"}}
    if result.artifact is not None:
        artifact["length"] = result.artifact.file_size_bytes
        artifact["hashes"] = {"sha-256": result.artifact.content_hash.removeprefix("sha256:")}

    run: dict[str, Any] = {
        "tool": {"driver": {
            "name": "AdapterSentry",
            "version": result.scan.analyzer_version,
            "informationUri": "https://github.com/nkorvyakov28-AS/adaptersentry-m1",
            "rules": list(rules.values()),
        }},
        "artifacts": [artifact],
        "results": results,
        "properties": {
            "schema_version": result.schema_version,
            "status": result.status,
            "action": v.action,
            "level": v.level.value,
            "policy": result.scan.policy,
        },
    }
    if result.errors:
        run["invocations"] = [{
            "executionSuccessful": result.status != "failed",
            "toolExecutionNotifications": [
                {"message": {"text": f"{e.code}: {e.message}"}, "level": "error"} for e in result.errors
            ],
        }]
    return {"$schema": _SARIF_SCHEMA, "version": "2.1.0", "runs": [run]}
