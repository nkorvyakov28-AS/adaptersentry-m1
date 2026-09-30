"""Tests for the text, JSON and SARIF renderings of a ScanResult 2.0.0."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from adaptersentry.reporters import json as json_reporter
from adaptersentry.reporters import sarif as sarif_reporter
from adaptersentry.reporters import text as text_reporter
from adaptersentry.scanner import scan
from adaptersentry.schemas.result import load_scan_result
from tests.adapter_factory import write_adapter


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> dict:
    base = tmp_path_factory.mktemp("rep")
    return {
        "clean": scan(write_adapter(base / "clean")),
        "inj": scan(write_adapter(base / "inj", inject_layer=6)),
        "mts": scan(write_adapter(base / "mts", extra={
            "base_model.model.lm_head.weight": np.zeros((8, 256), np.float32)})),
        "failed": scan(base / "missing.safetensors"),
    }


class TestJson:
    def test_round_trip(self, results: dict) -> None:
        for r in results.values():
            assert load_scan_result(json_reporter.render(r)) == r


class TestText:
    def test_clean(self, results: dict) -> None:
        out = text_reporter.render(results["clean"], no_color=True)
        assert "Verdict:  ALLOW (LOW)" in out and "Coverage:" in out

    def test_injected_shows_reason_and_location(self, results: dict) -> None:
        out = text_reporter.render(results["inj"], no_color=True)
        assert "REVIEW" in out and "INTRA_OUTLIER" in out and "layer 6" in out

    def test_failed_shows_error(self, results: dict) -> None:
        out = text_reporter.render(results["failed"], no_color=True)
        assert "REVIEW" in out and "Error:" in out

    def test_colours_only_when_requested(self, results: dict) -> None:
        assert "\x1b[" in text_reporter.render(results["inj"], no_color=False)
        assert "\x1b[" not in text_reporter.render(results["inj"], no_color=True)


class TestSarif:
    def test_structure(self, results: dict) -> None:
        doc = sarif_reporter.render(results["inj"])
        assert doc["version"] == "2.1.0"
        run = doc["runs"][0]
        assert run["tool"]["driver"]["name"] == "AdapterSentry"
        rule_ids = {r["id"] for r in run["tool"]["driver"]["rules"]}
        assert {res["ruleId"] for res in run["results"]} <= rule_ids
        assert all(res["level"] in ("error", "warning", "note") for res in run["results"])
        assert all("security-severity" in res["properties"] for res in run["results"])
        json.dumps(doc)

    def test_verdict_result_present_unless_allow(self, results: dict) -> None:
        clean = sarif_reporter.render(results["clean"])["runs"][0]
        assert not any(r["ruleId"] == "ADAPTER_VERDICT" for r in clean["results"])
        mts = sarif_reporter.render(results["mts"])["runs"][0]
        verdict = next(r for r in mts["results"] if r["ruleId"] == "ADAPTER_VERDICT")
        assert verdict["level"] == "warning" and "FULL_WEIGHT_REPLACEMENT" in verdict["message"]["text"]

    def test_logical_location_names_layer(self, results: dict) -> None:
        run = sarif_reporter.render(results["inj"])["runs"][0]
        spike = next(r for r in run["results"] if r["ruleId"] == "INTRA_DEPTH_SPIKE")
        name = spike["locations"][0]["logicalLocations"][0]["name"]
        assert name == "layers.6.down_proj"

    def test_relative_artifact_uri_and_hash(self, results: dict) -> None:
        art = sarif_reporter.render(results["clean"])["runs"][0]["artifacts"][0]
        assert art["location"]["uri"] == "adapter_model.safetensors"
        assert len(art["hashes"]["sha-256"]) == 64

    def test_failed_scan_reports_invocation_error(self, results: dict) -> None:
        run = sarif_reporter.render(results["failed"])["runs"][0]
        assert run["invocations"][0]["executionSuccessful"] is False
