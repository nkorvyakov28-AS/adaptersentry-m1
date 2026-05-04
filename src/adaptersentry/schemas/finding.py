"""Finding schema — a single structured anomaly detected by M1.

A Finding maps one-to-one to an anomaly flag but adds structured fields
for machine consumption: severity, confidence, affected layers, and a
structured evidence payload.

This is the primary interface between M1 and future M2-M4 modules.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Severity(str, Enum):
    """Shared severity enumeration used by both Finding and RiskSummary."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Map flag prefix → (severity, title, remediation_hint)
# Used by _flags_to_findings() in the analyzer bridge.
RULE_CATALOG: dict[str, tuple[Severity, str, str]] = {
    "RANK_INFLATION": (
        Severity.HIGH,
        "Rank Inflation",
        "Inspect adapter metadata and effective rank; verify with adapter author.",
    ),
    "HIGH_ENERGY_CONCENTRATION": (
        Severity.HIGH,
        "Near-Rank-1 Energy Concentration",
        "Run M2 behavioral sandbox; test for trigger-word responses.",
    ),
    "HIGH_RISK_TARGET_MODULE": (
        Severity.HIGH,
        "High-Risk Target Module",
        "Audit embed_tokens or lm_head modifications before deployment.",
    ),
    "CROSS_LAYER_CONCENTRATION": (
        Severity.HIGH,
        "Cross-Layer Flag Concentration",
        "Examine flagged layers for targeted injection patterns.",
    ),
    "SUSPICIOUS_LAYER_CLUSTER": (
        Severity.HIGH,
        "Suspicious Layer Cluster",
        "Examine flagged layer cluster for targeted injection patterns.",
    ),
    "SUSPICIOUS_PARTIAL_TRAINING": (
        Severity.HIGH,
        "Suspicious Partial Training",
        "Some layers trained; others at init. Verify with provenance data.",
    ),
    "HIGH_KURTOSIS_A": (
        Severity.MEDIUM,
        "Heavy-Tailed lora_A Distribution",
        "Inspect weight distribution; compare against clean adapters.",
    ),
    "HIGH_KURTOSIS_B": (
        Severity.MEDIUM,
        "Heavy-Tailed lora_B Distribution",
        "Inspect weight distribution; compare against clean adapters.",
    ),
    "HIGH_KURTOSIS": (
        Severity.MEDIUM,
        "Heavy-Tailed Weight Distribution",
        "Inspect weight distribution; compare against clean adapters.",
    ),
    "NEAR_ZERO_B_MATRIX": (
        Severity.MEDIUM,
        "Near-Zero B Matrix",
        "Verify adapter has been fine-tuned; check for selective zeroing.",
    ),
    "METADATA_DEPTH": (
        Severity.MEDIUM,
        "Deep Metadata Nesting",
        "Inspect metadata structure for evasion attempts.",
    ),
    "HIGH_ISOLATION_ANOMALY_A": (
        Severity.MEDIUM,
        "IsolationForest Anomaly in lora_A",
        "Inspect weight distribution for non-Gaussian outlier patterns.",
    ),
    "HIGH_ISOLATION_ANOMALY_B": (
        Severity.MEDIUM,
        "IsolationForest Anomaly in lora_B",
        "Inspect weight distribution for non-Gaussian outlier patterns.",
    ),
    "HIGH_ISOLATION_ANOMALY": (
        Severity.MEDIUM,
        "IsolationForest Anomaly",
        "Inspect weight distribution for non-Gaussian outlier patterns.",
    ),
    "HIGH_ZSCORE_OUTLIER_RATE_A": (
        Severity.MEDIUM,
        "Excess Z-Score Outliers in lora_A",
        "Review high-magnitude weight values in flagged layer.",
    ),
    "HIGH_ZSCORE_OUTLIER_RATE_B": (
        Severity.MEDIUM,
        "Excess Z-Score Outliers in lora_B",
        "Review high-magnitude weight values in flagged layer.",
    ),
    "HIGH_ZSCORE_OUTLIER_RATE": (
        Severity.MEDIUM,
        "Excess Z-Score Outliers",
        "Review high-magnitude weight values in flagged layer.",
    ),
    "LOW_ENTROPY_A": (
        Severity.MEDIUM,
        "Near-Constant lora_A Distribution",
        "Verify layer is not artificially zeroed or collapsed.",
    ),
    "LOW_ENTROPY_B": (
        Severity.MEDIUM,
        "Near-Constant lora_B Distribution",
        "Verify layer is not artificially zeroed or collapsed.",
    ),
    "LOW_ENTROPY": (
        Severity.MEDIUM,
        "Near-Constant Weight Distribution",
        "Verify layer is not artificially zeroed or collapsed.",
    ),
    "HIGH_ENTROPY_A": (
        Severity.LOW,
        "Near-Uniform lora_A Distribution",
        "May indicate noise injection; also common in init-only adapters.",
    ),
    "HIGH_ENTROPY_B": (
        Severity.LOW,
        "Near-Uniform lora_B Distribution",
        "May indicate noise injection; also common in init-only adapters.",
    ),
    "HIGH_ENTROPY": (
        Severity.LOW,
        "Near-Uniform Weight Distribution",
        "May indicate noise injection; also common in init-only adapters.",
    ),
    "INIT_ONLY_ADAPTER": (
        Severity.LOW,
        "Init-Only Adapter",
        "Adapter appears untrained (B=0, A=uniform). Normal for test/CI adapters.",
    ),
}


def _severity_for_rule(rule_id: str) -> Severity:
    """Return the severity for a known rule_id, defaulting to MEDIUM."""
    for prefix, (sev, _, _) in RULE_CATALOG.items():
        if rule_id.startswith(prefix):
            return sev
    return Severity.MEDIUM


def _title_for_rule(rule_id: str) -> str:
    for prefix, (_, title, _) in RULE_CATALOG.items():
        if rule_id.startswith(prefix):
            return title
    return rule_id.replace("_", " ").title()


def _remediation_for_rule(rule_id: str) -> str | None:
    for prefix, (_, _, rem) in RULE_CATALOG.items():
        if rule_id.startswith(prefix):
            return rem
    return None


class Finding(BaseModel):
    """A single structured anomaly detected by M1.

    This is the stable M1 → M2/M3/M4 interface.  Future modules consume
    findings to prioritize behavioral analysis, signature lookups, and
    runtime policy decisions.
    """

    model_config = ConfigDict(frozen=True)

    rule_id: str = Field(description="Machine-readable rule identifier (flag prefix)")
    title: str = Field(description="Human-readable rule name")
    severity: Severity
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Detection confidence in [0, 1]",
    )
    affected_layers: list[str] = Field(
        default_factory=list,
        description="Layer names where the finding was observed",
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw evidence payload (flag strings, scores, etc.)",
    )
    remediation: str | None = Field(
        default=None,
        description="Suggested remediation or investigation step",
    )


def flags_to_findings(flags: list[str], layer_reports: dict | None = None) -> list[Finding]:
    """Convert raw flag strings into structured Finding objects.

    Groups flags by rule_id; each unique rule_id becomes one Finding with
    all affected layers collected.

    Args:
        flags: Global flag strings from analyzer output.
        layer_reports: Optional per-layer dict for layer attribution.

    Returns:
        Deduplicated list of Finding objects.
    """
    # Build layer attribution: rule_id → [layer_names]
    layer_attr: dict[str, list[str]] = {}
    if layer_reports:
        for lname, report in layer_reports.items():
            for flag in report.get("flags", []):
                rid = _extract_rule_id(flag)
                layer_attr.setdefault(rid, []).append(lname)

    # Group global flags by rule_id
    seen: dict[str, list[str]] = {}
    for flag in flags:
        rid = _extract_rule_id(flag)
        seen.setdefault(rid, []).append(flag)

    findings: list[Finding] = []
    for rule_id, raw_flags in seen.items():
        findings.append(Finding(
            rule_id=rule_id,
            title=_title_for_rule(rule_id),
            severity=_severity_for_rule(rule_id),
            affected_layers=sorted(set(layer_attr.get(rule_id, []))),
            evidence={"flags": raw_flags},
            remediation=_remediation_for_rule(rule_id),
        ))

    return findings


def _extract_rule_id(flag: str) -> str:
    """Extract the rule_id prefix from a raw flag string."""
    colon = flag.find(":")
    return flag[:colon].strip() if colon != -1 else flag.split()[0]
