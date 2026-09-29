"""JSON output of a ScanResult 2.0.0."""

from __future__ import annotations

from adaptersentry.schemas.result import ScanResult


def render(result: ScanResult, indent: int = 2) -> str:
    """Serialise a ScanResult to JSON (field order as in the schema)."""
    return result.model_dump_json(indent=indent) + "\n"
