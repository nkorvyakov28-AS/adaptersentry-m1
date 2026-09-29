"""Severity levels shared by verdicts, reasons and findings."""

from __future__ import annotations

from enum import Enum


class Severity(str, Enum):
    """Ordered severity: LOW < MEDIUM < HIGH < CRITICAL."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
