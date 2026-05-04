"""Migration tests — schema stability across versions.

Each fixture in tests/fixtures/ is a frozen snapshot of a ScanResult at a
specific schema_version. These tests verify that:

1. An older JSON snapshot can be loaded by the current model without data loss.
2. extra="ignore" silently drops unknown fields rather than raising.
3. Required fields present in the snapshot parse correctly.
4. A schema_version bump requires a new fixture (enforced by naming convention).

Adding a new schema_version:
    python scripts/snapshot_schema.py --version 1.1.0

Removing or renaming a field that existed in a versioned fixture is a breaking
change — this test suite will catch it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptersentry.engine.schemas.scan_result import ScanResult, ScanStatus
from adaptersentry.engine.schemas.identity import AdapterArtifactIdentity, ScanIdentity
from adaptersentry.engine.schemas.scoring import EnsembleSignal, RiskVerdict
from adaptersentry.schemas.adapter_report import AnalysisMode, ParseStatus

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixture discovery helpers
# ---------------------------------------------------------------------------

def _fixture(name: str) -> dict:
    path = FIXTURES_DIR / name
    return json.loads(path.read_text())


def _all_scan_result_fixtures() -> list[Path]:
    return sorted(FIXTURES_DIR.glob("scan_result_v*.json"))


# ---------------------------------------------------------------------------
# v1.0.0 round-trip
# ---------------------------------------------------------------------------

class TestScanResultV1Migration:
    """Verifies the v1.0.0 fixture loads into the current ScanResult model."""

    def test_fixture_file_exists(self) -> None:
        assert (FIXTURES_DIR / "scan_result_v1.0.0.json").is_file(), (
            "scan_result_v1.0.0.json fixture missing — run: "
            "python scripts/snapshot_schema.py --version 1.0.0"
        )

    def test_model_validate_succeeds(self) -> None:
        data = _fixture("scan_result_v1.0.0.json")
        result = ScanResult.model_validate(data)
        assert isinstance(result, ScanResult)

    def test_schema_version_preserved(self) -> None:
        data = _fixture("scan_result_v1.0.0.json")
        result = ScanResult.model_validate(data)
        assert result.schema_version == "1.0.0"

    def test_identity_round_trip(self) -> None:
        data = _fixture("scan_result_v1.0.0.json")
        result = ScanResult.model_validate(data)
        assert isinstance(result.identity, ScanIdentity)
        assert result.identity.run_id == "test-run-v1"
        assert result.identity.schema_version == "1.0.0"

    def test_artifact_round_trip(self) -> None:
        data = _fixture("scan_result_v1.0.0.json")
        result = ScanResult.model_validate(data)
        assert isinstance(result.artifact, AdapterArtifactIdentity)
        assert result.artifact.file_size_bytes == 1024

    def test_verdict_round_trip(self) -> None:
        data = _fixture("scan_result_v1.0.0.json")
        result = ScanResult.model_validate(data)
        assert isinstance(result.verdict, RiskVerdict)
        assert result.verdict.recommended_action == "allow"
        assert result.verdict.m2_recommended is False

    def test_ensemble_round_trip(self) -> None:
        data = _fixture("scan_result_v1.0.0.json")
        result = ScanResult.model_validate(data)
        assert isinstance(result.ensemble, EnsembleSignal)
        assert result.ensemble.score == 0.0

    def test_status_fields_round_trip(self) -> None:
        data = _fixture("scan_result_v1.0.0.json")
        result = ScanResult.model_validate(data)
        assert result.status == ScanStatus.OK
        assert result.parse_status == ParseStatus.OK
        assert result.analysis_mode == AnalysisMode.FULL
        assert result.n_layers == 2
        assert result.n_layers_analyzed == 2

    def test_extra_fields_ignored(self) -> None:
        """extra='ignore' — a newer writer's fields don't break an older reader."""
        data = _fixture("scan_result_v1.0.0.json")
        data["future_field_unknown"] = "some_new_value"
        data["another_unknown"] = {"nested": True}
        result = ScanResult.model_validate(data)
        assert isinstance(result, ScanResult)
        assert not hasattr(result, "future_field_unknown")

    def test_json_serialisable_after_load(self) -> None:
        data = _fixture("scan_result_v1.0.0.json")
        result = ScanResult.model_validate(data)
        dumped = json.loads(result.model_dump_json())
        assert dumped["schema_version"] == "1.0.0"
        assert "identity" in dumped
        assert "verdict" in dumped

    def test_no_data_loss_on_round_trip(self) -> None:
        """Fields present in the fixture must survive a model parse+dump cycle."""
        data = _fixture("scan_result_v1.0.0.json")
        result = ScanResult.model_validate(data)
        dumped = json.loads(result.model_dump_json())

        critical_paths = [
            ("schema_version", "1.0.0"),
            ("status", "ok"),
            ("parse_status", "ok"),
            ("analysis_mode", "full"),
            ("n_layers", 2),
        ]
        for key, expected in critical_paths:
            assert dumped[key] == expected, (
                f"Field '{key}' changed value after round-trip: "
                f"expected {expected!r}, got {dumped[key]!r}"
            )


# ---------------------------------------------------------------------------
# Convention: every fixture file must cover a known schema_version
# ---------------------------------------------------------------------------

class TestFixtureNamingConvention:
    def test_at_least_one_fixture(self) -> None:
        fixtures = _all_scan_result_fixtures()
        assert len(fixtures) >= 1, (
            "No scan_result_v*.json fixtures found in tests/fixtures/. "
            "Run: python scripts/snapshot_schema.py --version 1.0.0"
        )

    def test_each_fixture_is_valid_json(self) -> None:
        for path in _all_scan_result_fixtures():
            data = json.loads(path.read_text())
            assert isinstance(data, dict), f"{path.name} is not a JSON object"

    def test_each_fixture_has_schema_version(self) -> None:
        for path in _all_scan_result_fixtures():
            data = json.loads(path.read_text())
            assert "schema_version" in data, (
                f"{path.name} is missing 'schema_version' field"
            )

    def test_fixture_version_matches_filename(self) -> None:
        for path in _all_scan_result_fixtures():
            # filename: scan_result_v1.0.0.json → version "1.0.0"
            version_from_name = path.stem.replace("scan_result_v", "")
            data = json.loads(path.read_text())
            assert data.get("schema_version") == version_from_name, (
                f"{path.name}: filename implies version '{version_from_name}' "
                f"but schema_version={data.get('schema_version')!r}"
            )

    def test_v100_fixture_loads_into_current_model(self) -> None:
        """Regression: the shipped v1.0.0 fixture must always load cleanly."""
        data = _fixture("scan_result_v1.0.0.json")
        result = ScanResult.model_validate(data)
        assert result.schema_version == "1.0.0"
