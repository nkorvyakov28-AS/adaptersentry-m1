"""Tests for the error taxonomy schema."""

from __future__ import annotations

import pytest

from adaptersentry.schemas.errors import ErrorCategory, ErrorCode, ScanError


class TestErrorCategory:
    def test_enum_values(self) -> None:
        assert ErrorCategory.MALFORMED == "malformed"
        assert ErrorCategory.UNSUPPORTED == "unsupported"
        assert ErrorCategory.DEGRADED == "degraded"

    def test_all_three_present(self) -> None:
        assert len(ErrorCategory) == 3


class TestScanError:
    def test_malformed_factory(self) -> None:
        err = ScanError.malformed(ErrorCode.INVALID_SAFETENSORS, "bad file")
        assert err.category == ErrorCategory.MALFORMED
        assert err.code == ErrorCode.INVALID_SAFETENSORS
        assert err.message == "bad file"
        assert err.detail is None

    def test_unsupported_factory(self) -> None:
        err = ScanError.unsupported(ErrorCode.NO_LORA_PAIRS, "no pairs found")
        assert err.category == ErrorCategory.UNSUPPORTED
        assert err.code == ErrorCode.NO_LORA_PAIRS

    def test_degraded_factory(self) -> None:
        err = ScanError.degraded(ErrorCode.SVD_FAILED, "svd error", detail="division by zero")
        assert err.category == ErrorCategory.DEGRADED
        assert err.detail == "division by zero"

    def test_json_serializable(self) -> None:
        import json
        err = ScanError.malformed("TEST_CODE", "test message", detail="details")
        data = json.loads(err.model_dump_json())
        assert data["category"] == "malformed"
        assert data["code"] == "TEST_CODE"
        assert data["message"] == "test message"
        assert data["detail"] == "details"

    def test_freeform_code_allowed(self) -> None:
        err = ScanError.degraded("UNEXPECTED_EXCEPTION", "something failed")
        assert err.code == "UNEXPECTED_EXCEPTION"

    def test_frozen_immutable(self) -> None:
        err = ScanError.malformed("X", "y")
        with pytest.raises(Exception):
            err.message = "changed"  # type: ignore[misc]
