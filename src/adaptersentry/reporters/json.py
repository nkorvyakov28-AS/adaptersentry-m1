"""JSON reporter for AdapterReport.

Serializes the full AdapterReport to JSON via Pydantic's model_dump_json(),
which handles nested models, enums, and None fields correctly without custom
serialization hacks.
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any

from adaptersentry.schemas.adapter_report import AdapterReport


def render(report: AdapterReport, indent: int = 2) -> str:
    """Serialize AdapterReport to a JSON string.

    Args:
        report: Completed M1 AdapterReport.
        indent: JSON indentation level (default 2).

    Returns:
        JSON string with stable field ordering (Pydantic model field order).
    """
    raw = _json.loads(report.model_dump_json())
    return _json.dumps(raw, indent=indent)


def write(report: AdapterReport, path: Path, indent: int = 2) -> None:
    """Write AdapterReport JSON to a file.

    Args:
        report: Completed M1 AdapterReport.
        path: Output file path.
        indent: JSON indentation level.
    """
    path.write_text(render(report, indent=indent), encoding="utf-8")


def to_dict(report: AdapterReport) -> dict[str, Any]:
    """Return AdapterReport as a plain Python dict.

    Args:
        report: Completed M1 AdapterReport.

    Returns:
        Dict representation suitable for further processing.
    """
    return _json.loads(report.model_dump_json())
