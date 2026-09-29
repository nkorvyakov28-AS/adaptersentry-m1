"""Regression tests for the v1.0.3 parser and verdict hardening.

Each test corresponds to an attack or failure mode found in the 2026-09 audit:
fail-open verdicts, unbounded headers, tensor bombs, unsupported dtypes,
skipped tensors, non-finite weights and non-regular files.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.analyzer import scan, scan_to_result
from adaptersentry.parsers import safetensors as sf
from adaptersentry.parsers.safetensors import (
    _read_sf_header,
    has_lora_pairs,
    load_adapter_checked,
    parse_tensors,
)
from adaptersentry.schemas.adapter_report import ParseStatus
from adaptersentry.schemas.finding import Severity

_RNG = np.random.default_rng(0)


def _pair(rank: int = 4, dim: int = 16) -> tuple[np.ndarray, np.ndarray]:
    a = _RNG.standard_normal((rank, dim)).astype(np.float32)
    b = (_RNG.standard_normal((dim, rank)) * 0.01).astype(np.float32)
    return a, b


def _write_adapter(path: Path, extra: dict[str, np.ndarray] | None = None, n_layers: int = 3) -> Path:
    tensors: dict[str, np.ndarray] = {}
    for i in range(n_layers):
        a, b = _pair()
        tensors[f"model.layers.{i}.self_attn.q_proj.lora_A.weight"] = a
        tensors[f"model.layers.{i}.self_attn.q_proj.lora_B.weight"] = b
    tensors.update(extra or {})
    save_file(tensors, str(path), metadata={"r": "4"})
    return path


def _write_raw(path: Path, header: dict, data: bytes) -> Path:
    body = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(body)) + body + data)
    return path


def _raw_adapter_with(path: Path, extra_key: str, extra_info: dict, extra_bytes: bytes) -> Path:
    """Valid 2-layer float32 adapter plus one crafted header entry."""
    header: dict = {"__metadata__": {"r": "4"}}
    data = b""
    for i in range(2):
        for part, shape in (("lora_A", [4, 16]), ("lora_B", [16, 4])):
            arr = _RNG.standard_normal(shape).astype("<f4").tobytes()
            header[f"model.layers.{i}.q_proj.{part}.weight"] = {
                "dtype": "F32", "shape": shape, "data_offsets": [len(data), len(data) + len(arr)],
            }
            data += arr
    info = dict(extra_info)
    info["data_offsets"] = [len(data), len(data) + len(extra_bytes)]
    header[extra_key] = info
    data += extra_bytes
    return _write_raw(path, header, data)


# ---------------------------------------------------------------------------
# 1.1 Fail-closed verdict
# ---------------------------------------------------------------------------


class TestFailClosedVerdict:
    def test_truncated_file_is_review_not_allow(self, tmp_path: Path) -> None:
        good = _write_adapter(tmp_path / "good.safetensors")
        truncated = tmp_path / "truncated.safetensors"
        truncated.write_bytes(good.read_bytes()[:-64])

        result = scan_to_result(truncated)

        assert result.verdict.recommended_action != "allow"
        assert result.verdict.overall_level != Severity.LOW
        assert result.verdict.m2_recommended is True
        assert "PARSE_FAILED" in {s.name for s in result.verdict.policy_signals}

    def test_failed_scan_report_is_not_low(self, tmp_path: Path) -> None:
        bad = tmp_path / "garbage.safetensors"
        bad.write_bytes(b"\x00" * 3)
        report = scan(bad)
        assert report.parse_status == ParseStatus.FAILED
        assert report.risk_summary.risk_level != Severity.LOW
        assert report.risk_summary.ensemble_risk_level != Severity.LOW

    def test_degraded_scan_is_at_least_review(self, tmp_path: Path) -> None:
        path = _write_adapter(
            tmp_path / "extra.safetensors",
            extra={"lm_head.weight": _RNG.standard_normal((8, 16)).astype(np.float32)},
        )
        result = scan_to_result(path)
        assert result.parse_status == ParseStatus.DEGRADED
        assert result.verdict.recommended_action in ("review", "block")
        assert "DEGRADED_PARSE" in {s.name for s in result.verdict.policy_signals}

    def test_clean_adapter_still_scans_ok(self, tmp_path: Path) -> None:
        path = _write_adapter(tmp_path / "clean.safetensors")
        result = scan_to_result(path)
        assert result.parse_status == ParseStatus.OK
        assert not any(s.name in ("PARSE_FAILED", "DEGRADED_PARSE") for s in result.verdict.policy_signals)


# ---------------------------------------------------------------------------
# 1.2 Header bounds
# ---------------------------------------------------------------------------


class TestHeaderBounds:
    def test_huge_declared_header_rejected_without_allocation(self, tmp_path: Path) -> None:
        path = tmp_path / "bomb.safetensors"
        path.write_bytes(struct.pack("<Q", 2**63) + b"{}")
        with pytest.raises(ValueError, match="exceeds limit"):
            _read_sf_header(path)

    def test_header_longer_than_file_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "short.safetensors"
        path.write_bytes(struct.pack("<Q", 1000) + b"{}")
        with pytest.raises(ValueError, match="exceeds file size"):
            _read_sf_header(path)

    def test_file_shorter_than_prefix_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.safetensors"
        path.write_bytes(b"\x01\x02")
        with pytest.raises(ValueError, match="too short"):
            _read_sf_header(path)

    def test_non_object_header_rejected(self, tmp_path: Path) -> None:
        path = _write_raw(tmp_path / "list.safetensors", [], b"")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="not a JSON object"):
            _read_sf_header(path)

    def test_has_lora_pairs_fails_closed(self, tmp_path: Path) -> None:
        path = tmp_path / "junk.safetensors"
        path.write_bytes(b"not a safetensors file at all")
        assert has_lora_pairs(path) is False


# ---------------------------------------------------------------------------
# 1.3 / 1.4 Header validation and per-tensor loading
# ---------------------------------------------------------------------------


class TestTensorValidation:
    def test_unsupported_dtype_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        path = _raw_adapter_with(
            tmp_path / "f8.safetensors",
            "model.layers.9.q_proj.lora_A.weight",
            {"dtype": "F8_E4M3", "shape": [4, 16]},
            b"\x00" * 64,
        )
        loaded = load_adapter_checked(path)
        assert any("unsupported dtype" in s.reason for s in loaded.skipped)
        assert len(loaded.tensors) == 4  # the valid pairs still load

        result = scan_to_result(path)
        assert result.parse_status == ParseStatus.DEGRADED
        assert result.verdict.recommended_action != "allow"

    def test_u8_dtype_is_skipped(self, tmp_path: Path) -> None:
        path = _raw_adapter_with(
            tmp_path / "u8.safetensors",
            "model.layers.9.q_proj.lora_B.weight",
            {"dtype": "U8", "shape": [16, 4]},
            b"\x01" * 64,
        )
        loaded = load_adapter_checked(path)
        assert any("unsupported dtype" in s.reason for s in loaded.skipped)

    def test_declared_tensor_bomb_never_allocated(self, tmp_path: Path) -> None:
        # A header that declares a huge tensor over a few bytes of data is
        # structurally invalid: the whole file is rejected (fail-closed review),
        # and nothing is allocated for the declared shape.
        path = _raw_adapter_with(
            tmp_path / "bomb.safetensors",
            "model.layers.9.q_proj.lora_A.weight",
            {"dtype": "F32", "shape": [100_000, 100_000]},
            b"\x00" * 16,
        )
        with pytest.raises(ValueError):
            load_adapter_checked(path)
        assert scan_to_result(path).verdict.recommended_action == "review"

    def test_declared_tensor_bomb_reason(self) -> None:
        info = {"dtype": "F32", "shape": [100_000, 100_000], "data_offsets": [0, 16]}
        assert "exceeds limit" in sf._header_entry_problem(info, data_len=16)

    def test_byte_length_mismatch_detected(self) -> None:
        info = {"dtype": "F32", "shape": [4, 16], "data_offsets": [0, 8]}
        assert "does not match" in sf._header_entry_problem(info, data_len=8)

    def test_offsets_outside_data_detected(self) -> None:
        info = {"dtype": "F32", "shape": [4, 16], "data_offsets": [0, 256]}
        assert "outside tensor data" in sf._header_entry_problem(info, data_len=100)

    def test_inconsistent_file_fails_closed(self, tmp_path: Path) -> None:
        path = _raw_adapter_with(
            tmp_path / "mismatch.safetensors",
            "model.layers.9.q_proj.lora_A.weight",
            {"dtype": "F32", "shape": [4, 16]},
            b"\x00" * 8,
        )
        result = scan_to_result(path)
        assert result.parse_status == ParseStatus.FAILED
        assert result.verdict.recommended_action == "review"

    def test_rank_above_limit_skipped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sf, "_MAX_LORA_RANK", 8)
        a = _RNG.standard_normal((16, 32)).astype(np.float32)
        b = _RNG.standard_normal((32, 16)).astype(np.float32)
        path = _write_adapter(
            tmp_path / "rank.safetensors",
            extra={"model.layers.9.v_proj.lora_A.weight": a, "model.layers.9.v_proj.lora_B.weight": b},
        )
        loaded = load_adapter_checked(path)
        assert sum("rank" in s.reason for s in loaded.skipped) == 2
        assert "model.layers.9.v_proj.lora_A.weight" not in loaded.tensors

    def test_total_numel_limit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sf, "_MAX_TOTAL_NUMEL", 10)
        path = _write_adapter(tmp_path / "total.safetensors")
        with pytest.raises(ValueError, match="exceeding limit"):
            load_adapter_checked(path)

    def test_non_lora_tensor_reported_as_not_analysed(self, tmp_path: Path) -> None:
        path = _write_adapter(
            tmp_path / "mts.safetensors",
            extra={"base_model.model.lm_head.weight": _RNG.standard_normal((8, 16)).astype(np.float32)},
        )
        loaded = load_adapter_checked(path)
        assert [s.key for s in loaded.skipped] == ["base_model.model.lm_head.weight"]
        report = scan(path)
        assert any(e.code == "TENSOR_NOT_ANALYZED" for e in report.errors)

    def test_no_complete_pair_fails(self, tmp_path: Path) -> None:
        a, _ = _pair()
        path = tmp_path / "onlyA.safetensors"
        save_file({"model.layers.0.q_proj.lora_A.weight": a}, str(path))
        with pytest.raises(ValueError, match="No complete"):
            load_adapter_checked(path)

    def test_parse_tensors_guards_declared_numel(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sf, "_MAX_TENSOR_NUMEL", 10)
        path = _write_adapter(tmp_path / "pt.safetensors")
        records = parse_tensors(path)
        assert records and all(r.parse_error is not None for r in records)


# ---------------------------------------------------------------------------
# 1.8 Regular files only
# ---------------------------------------------------------------------------


class TestRegularFileOnly:
    def test_symlink_to_device_rejected(self, tmp_path: Path) -> None:
        link = tmp_path / "zero.safetensors"
        link.symlink_to("/dev/zero")
        with pytest.raises(ValueError):
            load_adapter_checked(link)
        assert scan(link).parse_status == ParseStatus.FAILED

    def test_fifo_rejected_without_blocking(self, tmp_path: Path) -> None:
        fifo = tmp_path / "pipe.safetensors"
        os.mkfifo(fifo)
        with pytest.raises(ValueError, match="Not a regular file"):
            load_adapter_checked(fifo)

    def test_suffix_checked_after_resolve(self, tmp_path: Path) -> None:
        target = tmp_path / "payload.bin"
        target.write_bytes(b"\x80\x04K\x01.")  # a pickle stream
        link = tmp_path / "adapter.safetensors"
        link.symlink_to(target)
        with pytest.raises(ValueError, match="Expected .safetensors"):
            load_adapter_checked(link)

    def test_file_size_limit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sf, "_MAX_FILE_BYTES", 16)
        path = _write_adapter(tmp_path / "big.safetensors")
        with pytest.raises(ValueError, match="too large"):
            load_adapter_checked(path)


class TestNonRegularFilesEndToEnd:
    """The whole single-file and batch paths must not hang on a FIFO."""

    def test_scan_to_result_on_fifo_returns_review(self, tmp_path: Path) -> None:
        import signal

        fifo = tmp_path / "pipe.safetensors"
        os.mkfifo(fifo)

        def _timeout(signum: int, frame: object) -> None:
            raise TimeoutError("scan blocked on FIFO")

        previous = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(10)
        try:
            result = scan_to_result(fifo)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        assert result.verdict.recommended_action == "review"

    def test_build_manifest_skips_fifo(self, tmp_path: Path) -> None:
        from adaptersentry.engine.manifest import ManifestDB
        from adaptersentry.engine.orchestrator import build_manifest

        fifo = tmp_path / "pipe.safetensors"
        os.mkfifo(fifo)
        good = _write_adapter(tmp_path / "good.safetensors")
        db = ManifestDB.open(tmp_path / "manifest.sqlite")
        try:
            requests = build_manifest([fifo, good], "run_fifo", db)
        finally:
            db.close()
        assert [Path(r.adapter_path).name for r in requests] == ["good.safetensors"]
