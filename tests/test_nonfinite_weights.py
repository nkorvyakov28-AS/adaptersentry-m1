"""Regression tests: NaN/Inf weights must mark the scan degraded, never clean.

NaN makes every threshold comparison False, so a layer full of NaN used to
produce no flags at all and could crash native sort paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from adaptersentry.analyzer import scan, scan_to_result
from adaptersentry.schemas.adapter_report import ParseStatus

_RNG = np.random.default_rng(1)


def _write(path: Path, bad: tuple[np.ndarray, np.ndarray]) -> Path:
    tensors: dict[str, np.ndarray] = {}
    for i in range(3):
        tensors[f"model.layers.{i}.q_proj.lora_A.weight"] = _RNG.standard_normal((4, 16)).astype(np.float32)
        tensors[f"model.layers.{i}.q_proj.lora_B.weight"] = (
            _RNG.standard_normal((16, 4)) * 0.01
        ).astype(np.float32)
    tensors["model.layers.9.q_proj.lora_A.weight"] = bad[0]
    tensors["model.layers.9.q_proj.lora_B.weight"] = bad[1]
    save_file(tensors, str(path), metadata={"r": "4"})
    return path


def _pair() -> tuple[np.ndarray, np.ndarray]:
    return (
        _RNG.standard_normal((4, 16)).astype(np.float32),
        (_RNG.standard_normal((16, 4)) * 0.01).astype(np.float32),
    )


class TestNonFiniteWeights:
    def test_nan_layer_marked_degraded(self, tmp_path: Path) -> None:
        a, b = _pair()
        b[0, 0] = np.nan
        path = _write(tmp_path / "nan.safetensors", (a, b))
        report = scan(path)
        assert report.parse_status == ParseStatus.DEGRADED
        assert any("NON_FINITE_WEIGHTS" in f for tr in report.tensor_records for f in tr.flags)
        assert scan_to_result(path).verdict.recommended_action != "allow"

    def test_inf_layer_marked_degraded(self, tmp_path: Path) -> None:
        a, b = _pair()
        a[1, 2] = np.inf
        path = _write(tmp_path / "inf.safetensors", (a, b))
        assert scan(path).parse_status == ParseStatus.DEGRADED
