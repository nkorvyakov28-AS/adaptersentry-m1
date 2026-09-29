"""Tests for adapter_config.json reading and effective LoRA scale resolution."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from adaptersentry.parsers.adapter_config import (
    AdapterConfig,
    parse_adapter_config,
    read_adapter_config,
    resolve_scale,
)


class TestAdapterConfig:
    def test_parses_peft_fields(self) -> None:
        cfg, problems = parse_adapter_config({
            "peft_type": "LORA", "r": 16, "lora_alpha": 32, "use_rslora": True, "use_dora": False,
            "target_modules": ["q_proj", "v_proj"], "modules_to_save": ["lm_head"],
            "rank_pattern": {"q_proj": 8}, "alpha_pattern": {"v_proj": 4.0},
            "base_model_name_or_path": "meta-llama/Llama-3.3-70B-Instruct",
            "trainable_token_indices": [5, 7],
        })
        assert problems == []
        assert (cfg.r, cfg.lora_alpha, cfg.use_rslora) == (16, 32.0, True)
        assert cfg.modules_to_save == ["lm_head"] and cfg.trainable_token_indices is True

    def test_invalid_fields_ignored_and_reported(self) -> None:
        cfg, problems = parse_adapter_config({
            "r": "16", "lora_alpha": -1, "rank_pattern": {"q": "x", "v": 4}, "use_rslora": "yes",
        })
        assert cfg.r is None and cfg.lora_alpha is None and cfg.use_rslora is False
        assert cfg.rank_pattern == {"v": 4}
        assert any("r:" in p for p in problems) and any("lora_alpha" in p for p in problems)
        assert any("rank_pattern" in p for p in problems)

    def test_read_missing_config(self, tmp_path: Path) -> None:
        result = read_adapter_config(tmp_path / "adapter_model.safetensors")
        assert result.config is None and result.problems == []

    def test_read_valid_config(self, tmp_path: Path) -> None:
        (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 16}))
        result = read_adapter_config(tmp_path / "adapter_model.safetensors")
        assert result.config is not None and result.config.lora_alpha == 16.0

    def test_read_non_object_and_garbage(self, tmp_path: Path) -> None:
        (tmp_path / "adapter_config.json").write_text("[1, 2]")
        assert "not a JSON object" in read_adapter_config(tmp_path / "a.safetensors").problems[0]
        (tmp_path / "adapter_config.json").write_text("{not json")
        assert "unreadable" in read_adapter_config(tmp_path / "a.safetensors").problems[0]

    def test_read_oversized_config(self, tmp_path: Path) -> None:
        (tmp_path / "adapter_config.json").write_text(" " * 1_000_001)
        assert "exceeds" in read_adapter_config(tmp_path / "a.safetensors").problems[0]

    def test_read_deeply_nested_config(self, tmp_path: Path) -> None:
        (tmp_path / "adapter_config.json").write_text("[" * 100_000 + "]" * 100_000)
        result = read_adapter_config(tmp_path / "a.safetensors")
        assert result.config is None and result.problems


class TestResolveScale:
    def test_no_config_means_alpha_equals_r(self) -> None:
        res = resolve_scale("m.layers.0.q_proj", 16, None)
        assert res.scale == 1.0 and res.source == "default_alpha_equals_r"

    def test_standard_lora(self) -> None:
        res = resolve_scale("m.layers.0.q_proj", 16, AdapterConfig(r=16, lora_alpha=32))
        assert res.scale == pytest.approx(2.0)

    def test_rslora(self) -> None:
        res = resolve_scale("m.layers.0.q_proj", 16, AdapterConfig(r=16, lora_alpha=32, use_rslora=True))
        assert res.scale == pytest.approx(32 / 4)

    def test_rescale_attack_is_undone(self) -> None:
        # B scaled down by 1e4 and alpha up by 1e4: the effective scale carries the factor back.
        normal = resolve_scale("m.q_proj", 16, AdapterConfig(r=16, lora_alpha=16)).scale
        attack = resolve_scale("m.q_proj", 16, AdapterConfig(r=16, lora_alpha=16 * 1e4)).scale
        assert attack / normal == pytest.approx(1e4)

    def test_literal_alpha_pattern(self) -> None:
        cfg = AdapterConfig(r=16, lora_alpha=16, alpha_pattern={"layers.3.self_attn.q_proj": 64})
        assert resolve_scale("base.model.layers.3.self_attn.q_proj", 16, cfg).scale == pytest.approx(4.0)
        assert resolve_scale("base.model.layers.13.self_attn.q_proj", 16, cfg).scale == pytest.approx(1.0)

    def test_regex_pattern_is_never_executed(self) -> None:
        cfg = AdapterConfig(r=16, lora_alpha=16, alpha_pattern={"(a+)+$": 999})
        started = time.perf_counter()
        res = resolve_scale("a" * 5000 + "!", 16, cfg)
        assert time.perf_counter() - started < 0.1
        assert res.unresolved_pattern is True and res.scale == pytest.approx(1.0)


