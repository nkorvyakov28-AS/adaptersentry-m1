"""Schemas subpackage — Pydantic models for stable M1 report contracts."""

from .adapter_metadata import AdapterMetadata
from .adapter_report import (
    AdapterReport,
    AnalysisMode,
    RiskSummary,
    ScanTarget,
    ToolInfo,
    TrainingStatus,
)
from .errors import ErrorCategory, ErrorCode, ErrorSeverity, ScanError, ScanPhase
from .finding import Finding, Severity, flags_to_findings
from .tensor_record import TensorRecord

__all__ = [
    "AdapterMetadata",
    "AdapterReport",
    "AnalysisMode",
    "ErrorCategory",
    "ErrorCode",
    "ErrorSeverity",
    "ScanPhase",
    "Finding",
    "RiskSummary",
    "ScanError",
    "ScanTarget",
    "Severity",
    "TensorRecord",
    "ToolInfo",
    "TrainingStatus",
    "flags_to_findings",
]
