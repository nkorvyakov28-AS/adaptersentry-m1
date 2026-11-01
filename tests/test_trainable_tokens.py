"""Scanner handling of PEFT trainable-token deltas (selective-token tuning).

The delta is read (not left unanalysed), its per-token concentration surfaced as
an information-only finding, and it never changes the verdict — mirroring the
DoRA-magnitude discipline. The saved original-embedding copy stays recorded as a
reference copy in not_analyzed, so a token-modifying adapter keeps its review.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.scanner import scan

_MODULES = (("self_attn.q_proj", 256, 256), ("mlp.down_proj", 256, 512))
_UNCOVERED = "bigscience/bloom-560m"  # no reference profile -> isolate verdict effects
_EMBED = "base_model.model.model.embed_tokens"
_N_TOKENS, _DIM, _N_LAYERS = 20, 256, 4


def _write(directory: Path, tensors: dict, config: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "adapter_model.safetensors"
    save_file(tensors, str(path), metadata={"format": "pt"})
    (directory / "adapter_config.json").write_text(json.dumps(config))
    return path


def _lora(rng: np.random.Generator) -> dict[str, np.ndarray]:
    t: dict[str, np.ndarray] = {}
    for layer in range(_N_LAYERS):
        for mod, out, inp in _MODULES:
            base = f"base_model.model.model.layers.{layer}.{mod}"
            t[f"{base}.lora_A.weight"] = rng.standard_normal((8, inp)).astype(np.float32)
            t[f"{base}.lora_B.weight"] = (rng.standard_normal((out, 8)) * 0.02).astype(np.float32)
    return t


def _write_tokens(directory: Path, *, spike: bool, with_original: bool,
                  delta_shape: tuple[int, ...] | None = None, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    t = _lora(rng)
    shape = delta_shape or (_N_TOKENS, _DIM)
    delta = rng.standard_normal(shape).astype(np.float32)
    if spike and delta.ndim == 2:
        delta[0] *= 50.0  # one token's row moved far more than the rest
    t[f"{_EMBED}.trainable_tokens_delta"] = delta
    if with_original:
        t[f"{_EMBED}.trainable_tokens_original"] = rng.standard_normal((_N_TOKENS, _DIM)).astype(np.float32)
    cfg = {"peft_type": "LORA", "r": 8, "lora_alpha": 16,
           "base_model_name_or_path": _UNCOVERED, "target_modules": ["q_proj", "down_proj"]}
    return _write(directory, t, cfg)


# 2 modules x 4 layers = 8 pairs (16 tensors); plus the token delta (and original).
_N_PAIR_TENSORS = 2 * _N_LAYERS * 2


class TestTokenDeltaCoverage:
    def test_delta_counted_as_analysed(self, tmp_path: Path) -> None:
        res = scan(_write_tokens(tmp_path, spike=False, with_original=False))
        assert res.coverage.n_tensors_total == _N_PAIR_TENSORS + 1
        assert res.coverage.n_tensors_analyzed == _N_PAIR_TENSORS + 1  # delta no longer dropped
        assert res.coverage.n_not_analyzed == 0
        assert res.status == "ok"
        assert res.adapter.trainable_token_indices is True

    def test_readable_delta_not_listed_as_not_analyzed(self, tmp_path: Path) -> None:
        res = scan(_write_tokens(tmp_path, spike=False, with_original=False))
        keys = {na.tensor_key for na in res.coverage.not_analyzed}
        assert f"{_EMBED}.trainable_tokens_delta" not in keys

    def test_original_kept_as_reference_copy(self, tmp_path: Path) -> None:
        res = scan(_write_tokens(tmp_path, spike=False, with_original=True))
        keys = {na.tensor_key for na in res.coverage.not_analyzed}
        assert f"{_EMBED}.trainable_tokens_original" in keys
        assert f"{_EMBED}.trainable_tokens_delta" not in keys
        assert res.coverage.n_not_analyzed == 1
        assert res.status == "degraded"
        assert res.verdict.action == "review"  # token-modifying adapter still reviewed


class TestTokenDeltaFinding:
    def test_spike_raises_informational_finding(self, tmp_path: Path) -> None:
        res = scan(_write_tokens(tmp_path, spike=True, with_original=False))
        tf = [f for f in res.findings if f.rule_id == "TRAINABLE_TOKEN_DELTA"]
        assert len(tf) == 1
        assert tf[0].severity.value == "LOW"              # information only
        assert tf[0].evidence["hoyer"] >= 0.6
        assert tf[0].evidence["n_tokens"] == _N_TOKENS
        assert tf[0].evidence["top_token_rows"][0] == 0   # the spiked row

    def test_flat_delta_does_not_fire(self, tmp_path: Path) -> None:
        res = scan(_write_tokens(tmp_path, spike=False, with_original=False))
        assert not any(f.rule_id == "TRAINABLE_TOKEN_DELTA" for f in res.findings)

    def test_finding_does_not_change_verdict(self, tmp_path: Path) -> None:
        # Delta-only, uncovered family: the only difference is the token spike,
        # which is informational and must not move the action.
        flat = scan(_write_tokens(tmp_path / "flat", spike=False, with_original=False))
        spiked = scan(_write_tokens(tmp_path / "spiked", spike=True, with_original=False))
        assert spiked.verdict.action == flat.verdict.action == "allow"


class TestTokenDeltaMalformed:
    def test_non_2d_delta_is_surfaced_not_misread(self, tmp_path: Path) -> None:
        res = scan(_write_tokens(tmp_path, spike=False, with_original=False, delta_shape=(_N_TOKENS,)))
        keys = {na.tensor_key for na in res.coverage.not_analyzed}
        assert f"{_EMBED}.trainable_tokens_delta" in keys
        assert not any(f.rule_id == "TRAINABLE_TOKEN_DELTA" for f in res.findings)
        assert res.status != "ok"  # fail-closed: an unreadable delta is a gap, not a pass
