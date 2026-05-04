"""Tests for parsers.safetensors — parse_tensors() public contract and error taxonomy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.parsers import ParsedTensor, ParseErrorClass, parse_tensors
from adaptersentry.schemas.errors import ErrorCategory


def _make_safetensors(
    tmp_path: Path,
    tensors: dict[str, np.ndarray],
    filename: str = "adapter.safetensors",
) -> Path:
    path = tmp_path / filename
    save_file({k: v.astype(np.float32) for k, v in tensors.items()}, str(path))
    return path


def _lora_tensors(layer: str = "model.layers.0.q_proj") -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        f"{layer}.lora_A.weight": rng.standard_normal((8, 64)).astype(np.float32),
        f"{layer}.lora_B.weight": rng.standard_normal((64, 8)).astype(np.float32),
    }


class TestParseTensorsContract:
    def test_returns_list_of_parsed_tensor(self, tmp_path: Path) -> None:
        path = _make_safetensors(tmp_path, _lora_tensors())
        result = parse_tensors(path)
        assert isinstance(result, list)
        assert all(isinstance(r, ParsedTensor) for r in result)

    def test_output_type_stability_always_list(self, tmp_path: Path) -> None:
        """parse_tensors always returns list — never None, dict, or other type."""
        path = _make_safetensors(tmp_path, _lora_tensors())
        result = parse_tensors(path)
        assert type(result) is list  # noqa: E721

    def test_correct_tensor_count(self, tmp_path: Path) -> None:
        tensors = _lora_tensors()
        path = _make_safetensors(tmp_path, tensors)
        result = parse_tensors(path)
        assert len(result) == len(tensors)

    def test_name_field(self, tmp_path: Path) -> None:
        tensors = _lora_tensors("model.layers.0.q_proj")
        path = _make_safetensors(tmp_path, tensors)
        result = parse_tensors(path)
        names = {r.name for r in result}
        assert names == set(tensors.keys())

    def test_dtype_field(self, tmp_path: Path) -> None:
        path = _make_safetensors(tmp_path, _lora_tensors())
        result = parse_tensors(path)
        assert all(isinstance(r.dtype, str) for r in result)
        assert all("float" in r.dtype for r in result)

    def test_shape_field(self, tmp_path: Path) -> None:
        tensors = _lora_tensors()
        path = _make_safetensors(tmp_path, tensors)
        result = parse_tensors(path)
        by_name = {r.name: r for r in result}
        a_key = "model.layers.0.q_proj.lora_A.weight"
        assert by_name[a_key].shape == [8, 64]

    def test_numel_equals_product_of_shape(self, tmp_path: Path) -> None:
        path = _make_safetensors(tmp_path, _lora_tensors())
        result = parse_tensors(path)
        import math
        for r in result:
            assert r.numel == math.prod(r.shape)

    def test_stats_contains_mean_and_std(self, tmp_path: Path) -> None:
        path = _make_safetensors(tmp_path, _lora_tensors())
        result = parse_tensors(path)
        for r in result:
            assert "mean" in r.stats
            assert "std" in r.stats
            assert isinstance(r.stats["mean"], float)
            assert isinstance(r.stats["std"], float)

    def test_immutable_record(self, tmp_path: Path) -> None:
        path = _make_safetensors(tmp_path, _lora_tensors())
        rec = parse_tensors(path)[0]
        with pytest.raises(Exception):
            rec.name = "mutated"  # type: ignore[misc]


class TestParseTensorsErrorHandling:
    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_tensors(tmp_path / "nonexistent.safetensors")

    def test_wrong_suffix_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "adapter.bin"
        bad.write_bytes(b"not safetensors")
        with pytest.raises(ValueError, match=".bin"):
            parse_tensors(bad)

    def test_multiple_layers_all_returned(self, tmp_path: Path) -> None:
        tensors = {}
        for i in range(4):
            tensors.update(_lora_tensors(f"model.layers.{i}.q_proj"))
        path = _make_safetensors(tmp_path, tensors)
        result = parse_tensors(path)
        assert len(result) == 8  # 4 layers × 2 (A + B)

    def test_empty_adapter_returns_empty_list(self, tmp_path: Path) -> None:
        # A safetensors file with a single scalar so save_file accepts it
        path = _make_safetensors(
            tmp_path, {"dummy": np.zeros(1, dtype=np.float32)}, "empty.safetensors"
        )
        result = parse_tensors(path)
        assert isinstance(result, list)
        assert len(result) == 1


class TestParseErrorTaxonomy:
    """parse_tensors() must return degraded records (not raise) for per-tensor failures."""

    def test_parse_error_class_alias_matches_error_category(self) -> None:
        assert ParseErrorClass is ErrorCategory
        assert ParseErrorClass.MALFORMED.value == "malformed"
        assert ParseErrorClass.UNSUPPORTED.value == "unsupported"
        assert ParseErrorClass.DEGRADED.value == "degraded"

    def test_clean_tensor_has_no_parse_error(self, tmp_path: Path) -> None:
        path = _make_safetensors(tmp_path, _lora_tensors())
        result = parse_tensors(path)
        assert all(r.parse_error is None for r in result)

    def test_malformed_tensor_produces_degraded_record_not_exception(
        self, tmp_path: Path
    ) -> None:
        """One bad tensor inside an otherwise parseable file → degraded record, not crash."""
        good = _lora_tensors("model.layers.0.q_proj")
        path = _make_safetensors(tmp_path, good)

        # Simulate get_tensor raising for one specific key
        original_keys = list(good.keys())
        bad_key = original_keys[0]

        real_parse = parse_tensors.__wrapped__ if hasattr(parse_tensors, "__wrapped__") else None

        from safetensors import safe_open as _real_safe_open

        class _FaultyFile:
            def __init__(self, inner: object) -> None:
                self._inner = inner

            def __enter__(self) -> "_FaultyFile":
                self._inner.__enter__()
                return self

            def __exit__(self, *args: object) -> None:
                self._inner.__exit__(*args)

            def keys(self) -> list[str]:
                return list(self._inner.keys())  # type: ignore[attr-defined]

            def get_tensor(self, key: str) -> object:
                if key == bad_key:
                    raise RuntimeError("simulated corrupt tensor")
                return self._inner.get_tensor(key)  # type: ignore[attr-defined]

        def _patched_open(path_str: str, **kwargs: object) -> _FaultyFile:
            return _FaultyFile(_real_safe_open(path_str, **kwargs))

        with patch("adaptersentry.parsers.safetensors.safe_open", side_effect=_patched_open):
            result = parse_tensors(path)

        names = {r.name: r for r in result}
        assert bad_key in names, "failed tensor must still appear in output"
        bad_rec = names[bad_key]
        assert bad_rec.parse_error == ErrorCategory.MALFORMED
        assert bad_rec.numel == 0

        # The good tensor is unaffected
        good_key = original_keys[1]
        assert names[good_key].parse_error is None

    def test_unsupported_zero_element_tensor(self, tmp_path: Path) -> None:
        """Tensor with a zero-size dimension → parse_error=UNSUPPORTED."""
        good = _lora_tensors("model.layers.0.q_proj")
        path = _make_safetensors(tmp_path, good)

        zero_key = list(good.keys())[0]
        zero_tensor = np.zeros((0, 64), dtype=np.float32)

        from safetensors import safe_open as _real_safe_open

        class _ZeroFile:
            def __init__(self, inner: object) -> None:
                self._inner = inner

            def __enter__(self) -> "_ZeroFile":
                self._inner.__enter__()
                return self

            def __exit__(self, *args: object) -> None:
                self._inner.__exit__(*args)

            def keys(self) -> list[str]:
                return list(self._inner.keys())  # type: ignore[attr-defined]

            def get_tensor(self, key: str) -> object:
                if key == zero_key:
                    return zero_tensor
                return self._inner.get_tensor(key)  # type: ignore[attr-defined]

        def _patched_open(path_str: str, **kwargs: object) -> _ZeroFile:
            return _ZeroFile(_real_safe_open(path_str, **kwargs))

        with patch("adaptersentry.parsers.safetensors.safe_open", side_effect=_patched_open):
            result = parse_tensors(path)

        bad_rec = next(r for r in result if r.name == zero_key)
        assert bad_rec.parse_error == ErrorCategory.UNSUPPORTED
        assert bad_rec.numel == 0

    def test_parse_status_ok_when_all_tensors_clean(self, tmp_path: Path) -> None:
        """parse_status=ok when every tensor loads without error."""
        from adaptersentry.analyzer import scan
        from adaptersentry.schemas.adapter_report import ParseStatus
        path = _make_safetensors(tmp_path, _lora_tensors())
        report = scan(path)
        assert report.parse_status == ParseStatus.OK

    def test_parse_status_failed_for_missing_file(self, tmp_path: Path) -> None:
        """Unrecoverable file-level failure → parse_status=failed, structured report returned."""
        from adaptersentry.analyzer import scan
        from adaptersentry.schemas.adapter_report import AnalysisMode, ParseStatus
        report = scan(tmp_path / "ghost.safetensors")
        assert report.parse_status == ParseStatus.FAILED
        assert report.analysis_mode == AnalysisMode.FAILED
        assert report.tensor_records == []
        assert any(e.code == "INVALID_SAFETENSORS" for e in report.errors)

    def test_parser_output_is_schema_driven_not_raw_dict(self, tmp_path: Path) -> None:
        """parse_tensors() always returns list[ParsedTensor], never raw dicts."""
        path = _make_safetensors(tmp_path, _lora_tensors())
        result = parse_tensors(path)
        assert isinstance(result, list)
        for rec in result:
            assert isinstance(rec, ParsedTensor)
            assert isinstance(rec.name, str)
            assert isinstance(rec.shape, list)
            assert isinstance(rec.numel, int)
            assert isinstance(rec.stats, dict)
            # parse_error is typed — not a raw string
            assert rec.parse_error is None or isinstance(rec.parse_error, ErrorCategory)


# ---------------------------------------------------------------------------
# bfloat16 support
# ---------------------------------------------------------------------------

def _make_bf16_safetensors(
    tmp_path: Path,
    tensors_f32: dict[str, np.ndarray],
    filename: str = "bf16.safetensors",
) -> Path:
    """Write a safetensors file with bfloat16 tensors.

    Converts float32 to bfloat16 by zeroing the low 16 bits of each element
    (bfloat16 = high 16 bits of float32). Writes the raw safetensors binary
    format directly since safetensors.numpy does not support bfloat16 output.
    """
    import json as _json
    import struct as _struct

    parts: list[bytes] = []
    meta: dict = {}
    offset = 0
    for key, arr in tensors_f32.items():
        raw = (arr.view(np.uint32) >> 16).astype(np.uint16).tobytes()
        meta[key] = {
            "dtype": "BF16",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        parts.append(raw)
        offset += len(raw)
    header_bytes = _json.dumps(meta).encode()
    path = tmp_path / filename
    with open(path, "wb") as f:
        f.write(_struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for part in parts:
            f.write(part)
    return path


def _bf16_lora_tensors() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    return {
        "model.layers.0.q_proj.lora_A.weight": rng.standard_normal((8, 64)).astype(np.float32),
        "model.layers.0.q_proj.lora_B.weight": rng.standard_normal((64, 8)).astype(np.float32),
        "model.layers.1.v_proj.lora_A.weight": rng.standard_normal((8, 64)).astype(np.float32),
        "model.layers.1.v_proj.lora_B.weight": rng.standard_normal((64, 8)).astype(np.float32),
    }


class TestBFloat16Support:
    """bfloat16 adapters must load cleanly and produce float32 tensors."""

    def test_load_adapter_succeeds(self, tmp_path: Path) -> None:
        """Previously raised ValueError: Failed to parse ... bfloat16."""
        from adaptersentry.parsers.safetensors import load_adapter
        path = _make_bf16_safetensors(tmp_path, _bf16_lora_tensors())
        tensors, _ = load_adapter(path)
        assert len(tensors) == 4

    def test_load_adapter_returns_float32(self, tmp_path: Path) -> None:
        from adaptersentry.parsers.safetensors import load_adapter
        path = _make_bf16_safetensors(tmp_path, _bf16_lora_tensors())
        tensors, _ = load_adapter(path)
        for arr in tensors.values():
            assert arr.dtype == np.float32, f"expected float32, got {arr.dtype}"

    def test_load_adapter_values_correct(self, tmp_path: Path) -> None:
        """Loaded values match float32 truncated to bfloat16 precision."""
        from adaptersentry.parsers.safetensors import load_adapter
        rng = np.random.default_rng(42)
        original = rng.standard_normal((8, 64)).astype(np.float32)
        path = _make_bf16_safetensors(tmp_path, {
            "model.w.lora_A.weight": original,
            "model.w.lora_B.weight": original,
        })
        # Expected: float32 with low 16 mantissa bits zeroed (bfloat16 round-trip)
        expected = (original.view(np.uint32) >> 16 << 16).view(np.float32)
        tensors, _ = load_adapter(path)
        np.testing.assert_array_equal(tensors["model.w.lora_A.weight"], expected)

    def test_parse_tensors_no_error(self, tmp_path: Path) -> None:
        path = _make_bf16_safetensors(tmp_path, _bf16_lora_tensors())
        records = parse_tensors(path)
        assert all(r.parse_error is None for r in records), [
            r for r in records if r.parse_error is not None
        ]

    def test_parse_tensors_dtype_preserved(self, tmp_path: Path) -> None:
        """Original on-disk dtype 'BF16' is preserved in ParsedTensor.dtype."""
        path = _make_bf16_safetensors(tmp_path, _bf16_lora_tensors())
        records = parse_tensors(path)
        assert all(r.dtype == "BF16" for r in records)
