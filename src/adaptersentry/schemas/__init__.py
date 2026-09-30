"""Schemas: the ScanResult 2.0.0 contract and shared types."""

from .errors import ErrorCategory, ErrorSeverity, ScanError, ScanPhase
from .result import SCHEMA_VERSION, ScanResult, UnsupportedSchemaVersion, load_scan_result
from .severity import Severity

__all__ = [
    "SCHEMA_VERSION",
    "ErrorCategory",
    "ErrorSeverity",
    "ScanError",
    "ScanPhase",
    "ScanResult",
    "Severity",
    "UnsupportedSchemaVersion",
    "load_scan_result",
]
