"""Security regression tests for file handling and the fail-closed verdict (2.0 pipeline).

Each test corresponds to an attack or failure mode from the 2026-09 audit:
fail-open verdicts, unbounded headers, tensor bombs, unsupported dtypes,
skipped tensors, non-finite weights, non-regular files and terminal injection.
"""

from __future__ import annotations

import json
import os
import signal
import struct
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.parsers import safetensors as sf
from adaptersentry.parsers.adapter_file import open_adapter
from adaptersentry.parsers.safetensors import header_entry_problem, read_header
from adaptersentry.scanner import scan
from adaptersentry.schemas.severity import Severity
from tests.adapter_factory import write_adapter

_RNG = np.random.default_rng(0)


def _write_raw(path: Path, header: dict | list, data: bytes) -> Path:
    body = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(body)) + body + data)
    return path


def _raw_adapter_with(path: Path, extra_key: str, extra_info: dict, extra_bytes: bytes) -> Path:
    """Valid 6-layer float32 adapter plus one crafted header entry."""
    header: dict = {"__metadata__": {"r": "4"}}
    data = b""
    for i in range(6):
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


def _with_timeout(seconds: int, fn):
    def _raise(signum: int, frame: object) -> None:
        raise TimeoutError("blocked")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


class TestFailClosedVerdict:
    def test_truncated_file_is_review_not_allow(self, tmp_path: Path) -> None:
        good = write_adapter(tmp_path / "t")
        truncated = tmp_path / "t" / "truncated.safetensors"
        truncated.write_bytes(good.read_bytes()[:-64])
        result = scan(truncated)
        assert result.status == "failed"
        assert result.verdict.action == "review" and result.verdict.level != Severity.LOW
        assert result.verdict.m2_recommended
        assert "PARSE_FAILED" in {r.code for r in result.verdict.reasons}

    def test_garbage_file_is_review(self, tmp_path: Path) -> None:
        bad = tmp_path / "garbage.safetensors"
        bad.write_bytes(b"\x00" * 3)
        assert scan(bad).verdict.action == "review"

    def test_non_lora_tensor_degrades_scan(self, tmp_path: Path) -> None:
        path = write_adapter(tmp_path / "x", extra={
            "base_model.model.model.layers.0.self_attn.q_proj.bias": np.zeros(256, np.float32),
        })
        result = scan(path)
        assert result.status == "degraded" and result.verdict.action == "review"
        assert result.coverage.not_analyzed[0].tensor_key.endswith("q_proj.bias")


class TestHeaderBounds:
    def test_huge_declared_header_rejected_without_allocation(self, tmp_path: Path) -> None:
        path = tmp_path / "bomb.safetensors"
        path.write_bytes(struct.pack("<Q", 2**63) + b"{}")
        with pytest.raises(ValueError, match="exceeds limit"):
            read_header(path)
        assert scan(path).status == "failed"

    def test_header_longer_than_file_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "short.safetensors"
        path.write_bytes(struct.pack("<Q", 1000) + b"{}")
        with pytest.raises(ValueError, match="exceeds file size"):
            read_header(path)

    def test_file_shorter_than_prefix_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.safetensors"
        path.write_bytes(b"\x01\x02")
        with pytest.raises(ValueError, match="too short"):
            read_header(path)

    def test_non_object_header_rejected(self, tmp_path: Path) -> None:
        path = _write_raw(tmp_path / "list.safetensors", [], b"")
        with pytest.raises(ValueError, match="not a JSON object"):
            read_header(path)


class TestTensorValidation:
    def test_unsupported_dtype_is_not_analysed(self, tmp_path: Path) -> None:
        path = _raw_adapter_with(
            tmp_path / "f8.safetensors", "model.layers.9.q_proj.lora_A.weight",
            {"dtype": "F8_E4M3", "shape": [4, 16]}, b"\x00" * 64,
        )
        inv = open_adapter(path)
        assert any("unsupported dtype" in s.reason for s in inv.not_analyzed)
        result = scan(path)
        assert result.status == "degraded" and result.verdict.action != "allow"
        assert result.coverage.n_modules_analyzed == 6

    def test_u8_dtype_is_not_analysed(self, tmp_path: Path) -> None:
        path = _raw_adapter_with(
            tmp_path / "u8.safetensors", "model.layers.9.q_proj.lora_B.weight",
            {"dtype": "U8", "shape": [16, 4]}, b"\x01" * 64,
        )
        assert any("unsupported dtype" in s.reason for s in open_adapter(path).not_analyzed)

    def test_declared_tensor_bomb_never_allocated(self, tmp_path: Path) -> None:
        path = _raw_adapter_with(
            tmp_path / "bomb.safetensors", "model.layers.9.q_proj.lora_A.weight",
            {"dtype": "F32", "shape": [100_000, 100_000]}, b"\x00" * 16,
        )
        result = scan(path)
        assert result.status == "failed" and result.verdict.action == "review"

    def test_entry_problems(self) -> None:
        assert "exceeds limit" in header_entry_problem(
            {"dtype": "F32", "shape": [100_000, 100_000], "data_offsets": [0, 16]}, data_len=16)
        assert "does not match" in header_entry_problem(
            {"dtype": "F32", "shape": [4, 16], "data_offsets": [0, 8]}, data_len=8)
        assert "outside tensor data" in header_entry_problem(
            {"dtype": "F32", "shape": [4, 16], "data_offsets": [0, 256]}, data_len=100)

    def test_rank_above_limit_not_analysed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import adaptersentry.parsers.adapter_file as af
        monkeypatch.setattr(af, "MAX_LORA_RANK", 4)
        path = write_adapter(tmp_path / "rank", n_layers=6)
        result = scan(path)
        assert result.status == "failed"  # every pair has r = 8 > 4
        assert any("exceeds limit" in n.reason for n in open_adapter(path).not_analyzed)


class TestNonFiniteWeights:
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_layer_degrades(self, tmp_path: Path, bad: float) -> None:
        a = _RNG.standard_normal((8, 256)).astype(np.float32)
        a[1, 2] = bad
        path = write_adapter(tmp_path / "nf", extra={
            "base_model.model.model.layers.99.self_attn.q_proj.lora_A.weight": a,
            "base_model.model.model.layers.99.self_attn.q_proj.lora_B.weight": np.ones((256, 8), np.float32),
        })
        result = scan(path)
        assert result.status == "degraded" and result.verdict.action == "review"
        assert result.coverage.non_finite_modules == ["base_model.model.model.layers.99.self_attn.q_proj"]


class TestRegularFileOnly:
    def test_symlink_to_device_rejected(self, tmp_path: Path) -> None:
        link = tmp_path / "zero.safetensors"
        link.symlink_to("/dev/zero")
        with pytest.raises(ValueError):
            open_adapter(link)
        assert _with_timeout(10, lambda: scan(link)).verdict.action == "review"

    def test_fifo_does_not_block_scan(self, tmp_path: Path) -> None:
        fifo = tmp_path / "pipe.safetensors"
        os.mkfifo(fifo)
        with pytest.raises(ValueError, match="Not a regular file"):
            open_adapter(fifo)
        assert _with_timeout(10, lambda: scan(fifo)).verdict.action == "review"

    def test_suffix_checked_after_resolve(self, tmp_path: Path) -> None:
        target = tmp_path / "payload.bin"
        target.write_bytes(b"\x80\x04K\x01.")  # a pickle stream
        link = tmp_path / "adapter.safetensors"
        link.symlink_to(target)
        with pytest.raises(ValueError, match="Expected .safetensors"):
            open_adapter(link)

    def test_file_size_limit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sf, "MAX_FILE_BYTES", 16)
        path = write_adapter(tmp_path / "big", n_layers=2)
        with pytest.raises(ValueError, match="too large"):
            open_adapter(path)

    def test_build_manifest_skips_fifo(self, tmp_path: Path) -> None:
        from adaptersentry.engine.manifest import ManifestDB
        from adaptersentry.engine.orchestrator import build_manifest

        fifo = tmp_path / "pipe.safetensors"
        os.mkfifo(fifo)
        good = write_adapter(tmp_path / "good", n_layers=2)
        db = ManifestDB.open(tmp_path / "manifest.sqlite")
        try:
            requests = build_manifest([fifo, good], "run_fifo", db)
        finally:
            db.close()
        assert [Path(r.adapter_path).name for r in requests] == [good.name]


class TestTerminalInjection:
    def test_malicious_tensor_name_is_escaped_in_text_output(self, tmp_path: Path) -> None:
        from adaptersentry.reporters import text as text_reporter

        evil = "base_model.model.model.layers.0.\x1b[2J\x1b[H.q_proj"
        rng = np.random.default_rng(3)
        path = write_adapter(tmp_path / "evil", n_layers=2, extra={
            f"{evil}.lora_A.weight": rng.standard_normal((8, 256)).astype(np.float32),
            f"{evil}.lora_B.weight": np.full((256, 8), np.nan, np.float32),  # non-finite → listed by name
        })
        result = scan(path)
        rendered = text_reporter.render(result, no_color=True)
        assert "\x1b" not in rendered
        assert "\\x1b[2J" in rendered
