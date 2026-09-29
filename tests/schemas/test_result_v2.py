"""Contract tests for ScanResult 2.0.0 (schemas/result.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptersentry.schemas.result import (
    MAX_LISTED_NOT_ANALYZED,
    Coverage,
    NotAnalyzed,
    ReferenceComparison,
    ScanResult,
    UnsupportedSchemaVersion,
    load_scan_result,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_SCHEMA_FILE = Path(__file__).resolve().parents[2] / "docs" / "output-schema" / "scan-result-2.0.0.schema.json"


def _fixture() -> dict:
    return json.loads((_FIXTURES / "scan_result_v2.0.0.json").read_text())


def _with(doc: dict, **changes: object) -> dict:
    out = json.loads(json.dumps(doc))
    for dotted, value in changes.items():
        node = out
        *parents, leaf = dotted.split("__")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return out


class TestRoundTrip:
    def test_fixture_loads(self) -> None:
        result = load_scan_result(_fixture())
        assert result.schema_version == "2.0.0"
        assert result.verdict.action == "review"
        assert result.modules is None

    def test_round_trip_is_lossless(self) -> None:
        result = load_scan_result(_fixture())
        again = load_scan_result(result.model_dump_json())
        assert again == result

    def test_json_text_and_bytes_accepted(self) -> None:
        text = json.dumps(_fixture())
        assert load_scan_result(text) == load_scan_result(text.encode())

    def test_unknown_fields_from_later_minor_are_ignored(self) -> None:
        doc = _with(_fixture(), schema_version="2.3.0", future_field={"x": 1})
        assert load_scan_result(doc).schema_version == "2.3.0"


class TestVersionGate:
    def test_v1_document_is_rejected_with_clear_error(self) -> None:
        v1 = json.loads((_FIXTURES / "scan_result_v1.0.0.json").read_text())
        with pytest.raises(UnsupportedSchemaVersion, match="Re-scan"):
            load_scan_result(v1)

    def test_missing_version_rejected(self) -> None:
        doc = _fixture()
        del doc["schema_version"]
        with pytest.raises(UnsupportedSchemaVersion):
            load_scan_result(doc)

    def test_non_object_rejected(self) -> None:
        with pytest.raises(UnsupportedSchemaVersion):
            load_scan_result("[]")


class TestFailClosedInvariants:
    def test_degraded_cannot_allow(self) -> None:
        doc = _with(_fixture(), status="degraded", verdict__action="allow", verdict__level="LOW")
        with pytest.raises(ValidationError, match="fail-closed"):
            ScanResult.model_validate(doc)

    def test_failed_cannot_allow(self) -> None:
        doc = _with(_fixture(), status="failed", verdict__action="allow")
        with pytest.raises(ValidationError, match="fail-closed"):
            ScanResult.model_validate(doc)

    def test_default_policy_cannot_block_without_reference(self) -> None:
        doc = _with(_fixture(), verdict__action="block", verdict__level="HIGH")
        with pytest.raises(ValidationError, match="default"):
            ScanResult.model_validate(doc)

    def test_strict_policy_block_requires_structural_reason(self) -> None:
        doc = _with(_fixture(), scan__policy="strict", verdict__action="block", verdict__level="HIGH")
        with pytest.raises(ValidationError, match="STRICT_POLICY_STRUCTURAL"):
            ScanResult.model_validate(doc)

    def test_strict_policy_block_with_structural_reason_is_valid(self) -> None:
        doc = _with(_fixture(), scan__policy="strict", verdict__action="block", verdict__level="HIGH")
        doc["verdict"]["reasons"].append(
            {"code": "STRICT_POLICY_STRUCTURAL", "severity": "HIGH", "message": "LoRA on MoE router"}
        )
        assert ScanResult.model_validate(doc).verdict.action == "block"

    def test_block_with_reference_is_valid_under_default(self) -> None:
        doc = _with(_fixture(), verdict__action="block", verdict__level="HIGH")
        doc["anomaly"]["reference"] = ReferenceComparison(
            profile_id="llama-3.x", profile_version="1", stratum="llama|r64|alpha_over_r|all_linear",
            n_reference=300, distance=41.2, p_value=0.0033, alpha_review=0.01, alpha_block=0.005,
        ).model_dump()
        assert ScanResult.model_validate(doc).verdict.action == "block"


class TestCoverage:
    def test_analyzed_cannot_exceed_total(self) -> None:
        with pytest.raises(ValidationError):
            Coverage(n_tensors_total=2, n_tensors_analyzed=3, n_modules=1, n_modules_analyzed=1, n_not_analyzed=0)

    def test_listed_cannot_exceed_count(self) -> None:
        item = NotAnalyzed(tensor_key="k", reason="r", category="unsupported")
        with pytest.raises(ValidationError):
            Coverage(n_tensors_total=2, n_tensors_analyzed=1, n_modules=1, n_modules_analyzed=1,
                     n_not_analyzed=0, not_analyzed=[item])

    def test_list_is_capped(self) -> None:
        items = [NotAnalyzed(tensor_key=f"k{i}", reason="r", category="unsupported")
                 for i in range(MAX_LISTED_NOT_ANALYZED + 1)]
        with pytest.raises(ValidationError, match="capped"):
            Coverage(n_tensors_total=100, n_tensors_analyzed=49, n_modules=1, n_modules_analyzed=1,
                     n_not_analyzed=51, not_analyzed=items)


class TestFieldConstraints:
    def test_hash_format_enforced(self) -> None:
        doc = _with(_fixture(), artifact__content_hash="md5:abc")
        with pytest.raises(ValidationError):
            ScanResult.model_validate(doc)

    def test_top_contributors_capped_at_ten(self) -> None:
        doc = _fixture()
        doc["anomaly"]["top_contributors"] = doc["anomaly"]["top_contributors"] * 11
        with pytest.raises(ValidationError):
            ScanResult.model_validate(doc)

    def test_result_is_frozen(self) -> None:
        result = load_scan_result(_fixture())
        with pytest.raises(ValidationError):
            result.status = "failed"  # type: ignore[misc]


class TestPublishedJsonSchema:
    def test_published_schema_matches_model(self) -> None:
        published = json.loads(_SCHEMA_FILE.read_text())
        assert published == ScanResult.model_json_schema(), (
            "docs/output-schema/scan-result-2.0.0.schema.json is stale; regenerate it from "
            "ScanResult.model_json_schema()"
        )
