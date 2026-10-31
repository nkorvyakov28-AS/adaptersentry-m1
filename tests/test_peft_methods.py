"""Scanner handling of PEFT Level-1 methods: VeRA/non-LoRA, DoRA, PiSSA init."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.scanner import scan

_MODULES = (("self_attn.q_proj", 256, 256), ("mlp.down_proj", 256, 512))


def _write(directory: Path, tensors: dict, config: dict | None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "adapter_model.safetensors"
    save_file(tensors, str(path), metadata={"format": "pt"})
    if config is not None:
        (directory / "adapter_config.json").write_text(json.dumps(config))
    return path


def _lora_tensors(rng: np.random.Generator, n_layers: int = 16) -> dict[str, np.ndarray]:
    t: dict[str, np.ndarray] = {}
    for layer in range(n_layers):
        for mod, out, inp in _MODULES:
            base = f"base_model.model.model.layers.{layer}.{mod}"
            t[f"{base}.lora_A.weight"] = rng.standard_normal((8, inp)).astype(np.float32)
            t[f"{base}.lora_B.weight"] = (rng.standard_normal((out, 8)) * 0.02).astype(np.float32)
    return t


def _write_dora(directory: Path, base_model: str, *, spike: bool, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    t = _lora_tensors(rng)
    for layer in range(16):
        for mod, out, _ in _MODULES:
            m = np.ones(out, np.float32) + rng.standard_normal(out).astype(np.float32) * 0.01
            if spike and layer == 8 and mod == "mlp.down_proj":
                m[rng.choice(out, 3, replace=False)] += 5.0
            t[f"base_model.model.model.layers.{layer}.{mod}.lora_magnitude_vector.weight"] = m
    cfg = {"peft_type": "LORA", "r": 8, "lora_alpha": 16, "use_dora": True,
           "base_model_name_or_path": base_model, "target_modules": ["q_proj", "down_proj"]}
    return _write(directory, t, cfg)


def _write_vera(directory: Path, *, with_config: bool) -> Path:
    rng = np.random.default_rng(0)
    r = 256
    t: dict[str, np.ndarray] = {
        "base_model.model.vera_A": rng.standard_normal((r, 256)).astype(np.float32),
        "base_model.model.vera_B": rng.standard_normal((256, r)).astype(np.float32),
    }
    for layer in range(16):
        for mod, out, _ in _MODULES:
            b = f"base_model.model.model.layers.{layer}.{mod}"
            t[f"{b}.vera_lambda_b"] = (rng.standard_normal(out) * 0.1).astype(np.float32)
            t[f"{b}.vera_lambda_d"] = (rng.standard_normal(r) * 0.1).astype(np.float32)
    cfg = {"peft_type": "VERA", "r": r, "base_model_name_or_path": "meta-llama/Meta-Llama-3-8B",
           "target_modules": ["q_proj", "down_proj"]} if with_config else None
    return _write(directory, t, cfg)


class TestUnsupportedMethodRecognition:
    def test_vera_recognised_not_parse_failed(self, tmp_path: Path) -> None:
        res = scan(_write_vera(tmp_path, with_config=True))
        assert res.status == "failed"              # cannot statically analyse a non-LoRA method
        assert res.adapter.peft_type == "VERA"
        assert res.verdict.action == "review"      # fail-closed, not "allow"
        codes = {r.code for r in res.verdict.reasons}
        assert "UNSUPPORTED_PEFT_METHOD" in codes
        assert "PARSE_FAILED" not in codes         # no longer a misleading parse failure
        assert any(f.rule_id == "UNSUPPORTED_PEFT_METHOD" for f in res.findings)
        assert res.verdict.m2_recommended is True
        # The file parsed cleanly: it is recognised, not broken.
        assert all(e.code != "INVALID_SAFETENSORS" for e in res.errors)

    def test_vera_recognised_by_keys_without_config(self, tmp_path: Path) -> None:
        res = scan(_write_vera(tmp_path, with_config=False))
        assert res.status == "failed"
        assert any(r.code == "UNSUPPORTED_PEFT_METHOD" for r in res.verdict.reasons)

    def test_unknown_no_pair_file_still_parse_failed(self, tmp_path: Path) -> None:
        # A 2-D tensor that is neither a LoRA pair nor a recognised method stays a
        # plain parse failure — recognition must not swallow genuinely unknown files.
        _write(tmp_path, {"mystery.weight": np.zeros((4, 4), np.float32)}, None)
        res = scan(tmp_path / "adapter_model.safetensors")
        assert res.status == "failed"
        assert any(r.code == "PARSE_FAILED" for r in res.verdict.reasons)
        assert not any(r.code == "UNSUPPORTED_PEFT_METHOD" for r in res.verdict.reasons)


class TestDoraMagnitude:
    def test_magnitudes_counted_as_analysed(self, tmp_path: Path) -> None:
        res = scan(_write_dora(tmp_path, "meta-llama/Meta-Llama-3-8B", spike=False))
        # 2 modules x 16 layers: 32 pairs (64 tensors) + 32 magnitude vectors = 96.
        assert res.coverage.n_tensors_total == 96
        assert res.coverage.n_tensors_analyzed == 96    # magnitudes no longer silently dropped
        assert res.coverage.n_not_analyzed == 0
        assert res.status == "ok"                       # not blindly degraded
        assert res.adapter.use_dora is True

    def test_spike_raises_informational_finding(self, tmp_path: Path) -> None:
        res = scan(_write_dora(tmp_path, "bigscience/bloom-560m", spike=True))
        dm = [f for f in res.findings if f.rule_id == "DORA_MAGNITUDE"]
        assert len(dm) >= 1
        assert dm[0].severity.value == "LOW"            # information only
        assert dm[0].evidence["hoyer"] >= 0.6

    def test_flat_magnitudes_do_not_fire(self, tmp_path: Path) -> None:
        res = scan(_write_dora(tmp_path, "bigscience/bloom-560m", spike=False))
        assert not any(f.rule_id == "DORA_MAGNITUDE" for f in res.findings)

    def test_magnitude_does_not_change_verdict(self, tmp_path: Path) -> None:
        # Uncovered family (no reference): the only difference between the two scans
        # is the magnitude spike, which is informational and must not move the action.
        flat = scan(_write_dora(tmp_path / "flat", "bigscience/bloom-560m", spike=False))
        spiked = scan(_write_dora(tmp_path / "spiked", "bigscience/bloom-560m", spike=True))
        assert spiked.verdict.action == flat.verdict.action


class TestDecompositionInitNote:
    def test_pissa_adds_confidence_limit(self, tmp_path: Path) -> None:
        t = _lora_tensors(np.random.default_rng(0))
        cfg = {"peft_type": "LORA", "r": 8, "lora_alpha": 16, "init_lora_weights": "pissa",
               "base_model_name_or_path": "meta-llama/Meta-Llama-3-8B",
               "target_modules": ["q_proj", "down_proj"]}
        res = scan(_write(tmp_path, t, cfg))
        assert res.adapter.init_method == "pissa"
        assert any("re-parameterises A/B" in f for f in res.verdict.confidence.limiting_factors)

    def test_default_init_has_no_such_note(self, tmp_path: Path) -> None:
        t = _lora_tensors(np.random.default_rng(0))
        cfg = {"peft_type": "LORA", "r": 8, "lora_alpha": 16,
               "base_model_name_or_path": "meta-llama/Meta-Llama-3-8B",
               "target_modules": ["q_proj", "down_proj"]}
        res = scan(_write(tmp_path, t, cfg))
        assert not any("re-parameterises A/B" in f for f in res.verdict.confidence.limiting_factors)
