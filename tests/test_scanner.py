"""End-to-end tests of the 2.0 scan pipeline (adaptersentry.scanner.scan)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.scanner import scan
from adaptersentry.schemas.result import load_scan_result

_MODULES = (("self_attn.q_proj", 256, 256), ("self_attn.v_proj", 128, 256), ("mlp.down_proj", 256, 512))


def _write_adapter(
    directory: Path,
    *,
    n_layers: int = 12,
    r: int = 8,
    inject_layer: int | None = None,
    config: dict | None = None,
    extra: dict[str, np.ndarray] | None = None,
    b_scale: float = 0.01,
    seed: int = 0,
    name: str = "adapter_model.safetensors",
) -> Path:
    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {}
    for layer in range(n_layers):
        for mod, out, inp in _MODULES:
            base = f"base_model.model.model.layers.{layer}.{mod}"
            a = rng.standard_normal((r, inp)).astype(np.float32)
            b = (rng.standard_normal((out, r)) * b_scale * np.exp(-np.arange(r) / 3)).astype(np.float32)
            if inject_layer == layer and mod == "mlp.down_proj":
                spike = np.zeros(out, np.float32)
                spike[rng.choice(out, 4, replace=False)] = 0.3
                b[:, -1] = spike
            tensors[f"{base}.lora_A.weight"] = a
            tensors[f"{base}.lora_B.weight"] = b
    tensors.update(extra or {})
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    save_file(tensors, str(path), metadata={"format": "pt"})
    cfg = {"peft_type": "LORA", "r": r, "lora_alpha": 2 * r,
           "base_model_name_or_path": "meta-llama/Llama-3.2-1B",
           "target_modules": ["q_proj", "v_proj", "down_proj"]}
    if config is not None:
        cfg = config
    if cfg:
        (directory / "adapter_config.json").write_text(json.dumps(cfg))
    return path


class TestCleanAdapter:
    def test_clean_adapter_allows_with_medium_confidence(self, tmp_path: Path) -> None:
        result = scan(_write_adapter(tmp_path / "clean"))
        assert result.status == "ok"
        assert result.verdict.action == "allow"
        assert result.verdict.confidence.level == "medium"
        codes = {r.code for r in result.verdict.reasons}
        assert codes == {"NO_REFERENCE_PROFILE"}
        assert result.anomaly.intra is not None and result.anomaly.intra.max_robust_z < 8

    def test_adapter_info(self, tmp_path: Path) -> None:
        a = scan(_write_adapter(tmp_path / "info")).adapter
        assert a.format == "peft_lora" and a.base_family == "llama" and a.base_family_source == "config"
        assert a.rank_declared == 8 and a.rank_actual.distinct == [8]
        assert a.lora_alpha == 16 and a.scaling == "alpha_over_r"
        assert a.n_model_layers == 12 and a.training_state == "trained"
        assert set(a.target_modules_actual) == {"q_proj", "v_proj", "down_proj"}

    def test_coverage_complete(self, tmp_path: Path) -> None:
        c = scan(_write_adapter(tmp_path / "cov")).coverage
        assert (c.n_modules, c.n_modules_analyzed, c.n_not_analyzed) == (36, 36, 0)
        assert c.n_tensors_analyzed == c.n_tensors_total == 72

    def test_file_name_only_by_default(self, tmp_path: Path) -> None:
        path = _write_adapter(tmp_path / "paths")
        assert scan(path).artifact.provenance.path == "adapter_model.safetensors"
        assert scan(path, full_paths=True).artifact.provenance.path == str(path.resolve())

    def test_result_round_trips_through_json(self, tmp_path: Path) -> None:
        result = scan(_write_adapter(tmp_path / "rt"), include_modules=True)
        assert load_scan_result(result.model_dump_json()) == result

    def test_modules_opt_in(self, tmp_path: Path) -> None:
        path = _write_adapter(tmp_path / "mods")
        assert scan(path).modules is None
        mods = scan(path, include_modules=True).modules
        assert mods is not None and len(mods) == 36 and mods[0].heads is None
        assert all(m.status == "ok" and m.features is not None for m in mods)
        with_heads = scan(path, include_heads=True).modules
        assert any(m.heads for m in with_heads if m.kind == "attention")


class TestDetection:
    def test_injected_layer_raises_review(self, tmp_path: Path) -> None:
        result = scan(_write_adapter(tmp_path / "inj", inject_layer=7))
        assert result.verdict.action == "review"
        assert "INTRA_OUTLIER" in {r.code for r in result.verdict.reasons}
        spike = next(f for f in result.findings if f.rule_id == "INTRA_DEPTH_SPIKE")
        assert spike.locations[0].layer == 7 and spike.locations[0].module == "down_proj"

    def test_rescale_attack_does_not_hide_the_update(self, tmp_path: Path) -> None:
        # B shrunk by 1e4 and lora_alpha raised by 1e4: effective ΔW is unchanged.
        normal = scan(_write_adapter(tmp_path / "n", inject_layer=7))
        attack = scan(_write_adapter(
            tmp_path / "a", inject_layer=7, b_scale=0.01,
            config={"peft_type": "LORA", "r": 8, "lora_alpha": 16 * 1e4},
        ))
        assert attack.verdict.action == "review"
        assert attack.anomaly.intra.max_robust_z == pytest.approx(normal.anomaly.intra.max_robust_z, rel=1e-3)

    def test_init_only_adapter(self, tmp_path: Path) -> None:
        result = scan(_write_adapter(tmp_path / "init", b_scale=0.0))
        assert result.adapter.training_state == "init_only"


class TestStructural:
    def test_full_lm_head_is_red_flag(self, tmp_path: Path) -> None:
        path = _write_adapter(tmp_path / "mts", extra={
            "base_model.model.lm_head.weight": np.zeros((64, 256), np.float32),
        })
        result = scan(path)
        assert "FULL_WEIGHT_REPLACEMENT" in {r.code for r in result.verdict.reasons}
        assert result.verdict.action == "review" and result.status == "degraded"

    def test_strict_policy_blocks_on_red_flag(self, tmp_path: Path) -> None:
        path = _write_adapter(tmp_path / "strict", extra={
            "base_model.model.lm_head.weight": np.zeros((64, 256), np.float32),
        })
        result = scan(path, policy="strict")
        assert result.verdict.action == "block"
        assert "STRICT_POLICY_STRUCTURAL" in {r.code for r in result.verdict.reasons}

    def test_default_policy_never_blocks_without_reference(self, tmp_path: Path) -> None:
        path = _write_adapter(tmp_path / "def", inject_layer=3, extra={
            "base_model.model.lm_head.weight": np.zeros((64, 256), np.float32),
        })
        assert scan(path).verdict.action == "review"

    def test_executable_sibling(self, tmp_path: Path) -> None:
        path = _write_adapter(tmp_path / "sib")
        (tmp_path / "sib" / "modeling_custom.py").write_text("print('hi')\n")
        result = scan(path)
        assert "SIBLING_EXECUTABLE" in {r.code for r in result.verdict.reasons}
        assert result.artifact.provenance.sibling_files[0].kind == "code"

    def test_rank_mismatch(self, tmp_path: Path) -> None:
        path = _write_adapter(tmp_path / "rank", config={"peft_type": "LORA", "r": 4, "lora_alpha": 8})
        assert "RANK_MISMATCH" in {r.code for r in scan(path).verdict.reasons}

    def test_router_lora(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(1)
        path = _write_adapter(tmp_path / "moe", extra={
            "base_model.model.model.layers.0.mlp.gate.lora_A.weight": rng.standard_normal((8, 256)).astype(np.float32),
            "base_model.model.model.layers.0.mlp.gate.lora_B.weight": rng.standard_normal((8, 8)).astype(np.float32),
        })
        assert "ROUTER_ADAPTED" in {r.code for r in scan(path).verdict.reasons}

    def test_no_config_limits_confidence(self, tmp_path: Path) -> None:
        result = scan(_write_adapter(tmp_path / "nocfg", config={}))
        assert result.adapter.config_present is False and result.adapter.scaling == "unknown"
        assert any("adapter_config.json" in f for f in result.verdict.confidence.limiting_factors)


class TestFailures:
    def test_truncated_file_fails_closed(self, tmp_path: Path) -> None:
        good = _write_adapter(tmp_path / "t")
        bad = tmp_path / "t" / "broken.safetensors"
        bad.write_bytes(good.read_bytes()[:-100])
        result = scan(bad)
        assert result.status == "failed" and result.verdict.action == "review"
        assert result.errors and result.errors[0].code == "INVALID_SAFETENSORS"

    def test_missing_file(self, tmp_path: Path) -> None:
        result = scan(tmp_path / "nope.safetensors")
        assert result.status == "failed" and result.artifact is None

    def test_non_lora_file(self, tmp_path: Path) -> None:
        path = tmp_path / "plain.safetensors"
        save_file({"weight": np.zeros((4, 4), np.float32)}, str(path))
        result = scan(path)
        assert result.status == "failed" and result.verdict.action == "review"

    def test_nan_layer_degrades(self, tmp_path: Path) -> None:
        a = np.full((8, 256), np.nan, np.float32)
        b = np.zeros((256, 8), np.float32)
        path = _write_adapter(tmp_path / "nan", extra={
            "base_model.model.model.layers.99.self_attn.q_proj.lora_A.weight": a,
            "base_model.model.model.layers.99.self_attn.q_proj.lora_B.weight": b,
        })
        result = scan(path)
        assert result.status == "degraded" and result.verdict.action == "review"
        assert result.coverage.non_finite_modules

    def test_deterministic_scan_id(self, tmp_path: Path) -> None:
        path = _write_adapter(tmp_path / "det")
        assert scan(path).scan.scan_id == scan(path).scan.scan_id
        assert scan(path).scan.scan_id != scan(path, policy="strict").scan.scan_id
