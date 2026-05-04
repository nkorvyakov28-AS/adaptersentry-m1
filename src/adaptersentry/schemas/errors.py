"""Error taxonomy for AdapterSentry scan results.

Three normalized error categories model every failure mode the scanner can
encounter, making scan results machine-consumable regardless of whether
analysis succeeded, partially succeeded, or failed entirely.

Categories
----------
malformed   — The input is corrupt, truncated, or structurally invalid.
              The scanner cannot extract useful information.
unsupported — The input is structurally valid but uses a format or
              architecture variant the scanner does not handle.
degraded    — Analysis proceeded but with reduced fidelity; some signals
              are missing or estimates rather than exact values.

Error severity
--------------
FATAL    — the phase or scan cannot continue; result is absent or meaningless.
DEGRADED — fidelity is reduced; a partial result is still produced.
WARNING  — informational; no impact on analysis quality.

Scan phases
-----------
PARSE      — file I/O and safetensors header / tensor loading.
METADATA   — adapter_config / safetensors metadata extraction.
FEATURE    — per-layer feature extraction (norm, distribution, entropy, …).
SCORING    — ensemble scoring and verdict derivation.
REPORTING  — result serialisation and output formatting.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator


class ErrorCategory(str, Enum):
    """Top-level taxonomy for scan errors."""

    MALFORMED = "malformed"
    UNSUPPORTED = "unsupported"
    DEGRADED = "degraded"


class ErrorSeverity(str, Enum):
    """How much a scan error impacts the final result."""

    FATAL = "fatal"
    DEGRADED = "degraded"
    WARNING = "warning"


class ScanPhase(str, Enum):
    """Pipeline phase where the error occurred."""

    PARSE = "parse"
    METADATA = "metadata"
    FEATURE = "feature"
    SCORING = "scoring"
    REPORTING = "reporting"


class ErrorCode(str, Enum):
    # malformed
    INVALID_SAFETENSORS = "INVALID_SAFETENSORS"
    TRUNCATED_FILE = "TRUNCATED_FILE"
    METADATA_PARSE_ERROR = "METADATA_PARSE_ERROR"
    TENSOR_SHAPE_INVALID = "TENSOR_SHAPE_INVALID"
    TENSOR_TOO_LARGE = "TENSOR_TOO_LARGE"

    # unsupported
    NO_LORA_PAIRS = "NO_LORA_PAIRS"
    UNSUPPORTED_PEFT_TYPE = "UNSUPPORTED_PEFT_TYPE"
    SHARDED_ADAPTER = "SHARDED_ADAPTER"

    # degraded
    PARTIAL_LAYER_ANALYSIS = "PARTIAL_LAYER_ANALYSIS"
    ISOLATION_FOREST_SKIPPED = "ISOLATION_FOREST_SKIPPED"
    SVD_FAILED = "SVD_FAILED"
    METADATA_DEPTH_EXCEEDED = "METADATA_DEPTH_EXCEEDED"


ParseErrorClass = ErrorCategory

_CODE_SEVERITY: dict[str, ErrorSeverity] = {
    ErrorCode.INVALID_SAFETENSORS: ErrorSeverity.FATAL,
    ErrorCode.TRUNCATED_FILE: ErrorSeverity.FATAL,
    ErrorCode.TENSOR_TOO_LARGE: ErrorSeverity.FATAL,
    ErrorCode.NO_LORA_PAIRS: ErrorSeverity.FATAL,
    ErrorCode.UNSUPPORTED_PEFT_TYPE: ErrorSeverity.FATAL,
    ErrorCode.SHARDED_ADAPTER: ErrorSeverity.FATAL,
    ErrorCode.METADATA_PARSE_ERROR: ErrorSeverity.DEGRADED,
    ErrorCode.TENSOR_SHAPE_INVALID: ErrorSeverity.DEGRADED,
    ErrorCode.PARTIAL_LAYER_ANALYSIS: ErrorSeverity.DEGRADED,
    ErrorCode.ISOLATION_FOREST_SKIPPED: ErrorSeverity.DEGRADED,
    ErrorCode.SVD_FAILED: ErrorSeverity.DEGRADED,
    ErrorCode.METADATA_DEPTH_EXCEEDED: ErrorSeverity.WARNING,
}


def _infer_severity(category: object, code: object) -> ErrorSeverity:
    """Return the canonical severity for a given error code and category."""
    code_str: str = code.value if isinstance(code, Enum) else str(code)  # type: ignore[union-attr]
    cat_str: str = category.value if isinstance(category, Enum) else str(category)  # type: ignore[union-attr]
    try:
        return _CODE_SEVERITY[ErrorCode(code_str)]
    except (ValueError, KeyError):
        pass
    if cat_str in (ErrorCategory.MALFORMED.value, ErrorCategory.UNSUPPORTED.value):
        return ErrorSeverity.FATAL
    return ErrorSeverity.DEGRADED


class ScanError(BaseModel):
    """Normalized scan error — goes into AdapterReport.errors / ScanResult.errors.

    ``code`` is a short upper-case string identifying the error.  Canonical values
    are defined in ``ErrorCode``; callers should treat ``code`` as advisory — future
    versions may emit codes not present in the current ``ErrorCode`` enum.  Always
    branch on ``category`` for reliable machine processing; use ``code`` only for
    logging and display.
    """

    model_config = ConfigDict(frozen=True)

    category: ErrorCategory
    code: str
    message: str
    detail: str | None = None
    phase: ScanPhase | None = None
    severity: ErrorSeverity = ErrorSeverity.DEGRADED

    @model_validator(mode="before")
    @classmethod
    def _infer_severity_from_code(cls, data: dict) -> dict:
        if isinstance(data, dict) and "severity" not in data:
            data["severity"] = _infer_severity(
                data.get("category", ""),
                data.get("code", ""),
            )
        return data

    @classmethod
    def malformed(cls, code: str, message: str, detail: str | None = None, *, phase: ScanPhase | None = None) -> "ScanError":
        return cls(category=ErrorCategory.MALFORMED, code=code, message=message, detail=detail, phase=phase)

    @classmethod
    def unsupported(cls, code: str, message: str, detail: str | None = None, *, phase: ScanPhase | None = None) -> "ScanError":
        return cls(category=ErrorCategory.UNSUPPORTED, code=code, message=message, detail=detail, phase=phase)

    @classmethod
    def degraded(cls, code: str, message: str, detail: str | None = None, *, phase: ScanPhase | None = None) -> "ScanError":
        return cls(category=ErrorCategory.DEGRADED, code=code, message=message, detail=detail, phase=phase)
