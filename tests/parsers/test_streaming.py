"""Tests for streaming adapter access: key parsing, inventory, one-pair-at-a-time iteration."""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.parsers.adapter_file import iter_pairs, open_adapter
from adaptersentry.parsers.names import locate, parse_key

_RNG = np.random.default_rng(0)


def _pair(r: int, out: int, inp: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        _RNG.standard_normal((r, inp)).astype(np.float32),
        (_RNG.standard_normal((out, r)) * 0.01).astype(np.float32),
    )


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> raw bfloat16 bits (truncation), stored as uint16."""
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


# ── names ──────────────────────────────────────────────────────────────────────


class TestKeyParsing:
    @pytest.mark.parametrize("key,role,base", [
        ("base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight", "lora_A",
         "base_model.model.model.layers.3.self_attn.q_proj"),
        ("m.layers.3.mlp.down_proj.lora_B.default.weight", "lora_B", "m.layers.3.mlp.down_proj"),
        ("m.embed_tokens.lora_embedding_A", "embedding_A", "m.embed_tokens"),
        ("m.layers.1.self_attn.q_proj.lora_magnitude_vector", "dora_magnitude", "m.layers.1.self_attn.q_proj"),
        ("m.embed_tokens.token_adapter.trainable_tokens_delta", "trainable_tokens", "m.embed_tokens.token_adapter"),
        ("base_model.model.lm_head.weight", "other", None),
    ])
    def test_roles(self, key: str, role: str, base: str | None) -> None:
        parsed = parse_key(key)
        assert (parsed.role, parsed.base) == (role, base)

    @pytest.mark.parametrize("base,layer,expert,module,kind", [
        ("base_model.model.model.layers.47.mlp.down_proj", 47, None, "down_proj", "mlp"),
        ("base_model.model.model.layers.3.self_attn.q_proj", 3, None, "q_proj", "attention"),
        ("model.layers.5.mlp.experts.12.gate_proj", 5, 12, "gate_proj", "mlp"),
        ("model.layers.5.mlp.gate", 5, None, "gate", "router"),
        ("transformer.h.7.attn.c_attn", 7, None, "c_attn", "attention"),
        ("base_model.model.lm_head", None, None, "lm_head", "lm_head"),
        ("base_model.model.model.embed_tokens", None, None, "embed_tokens", "embedding"),
    ])
    def test_locate(self, base: str, layer: int | None, expert: int | None, module: str, kind: str) -> None:
        loc = locate(base)
        assert (loc.layer, loc.expert, loc.module, loc.kind) == (layer, expert, module, kind)


# ── inventory and iteration ────────────────────────────────────────────────────


class TestInventory:
    def test_buckets_every_tensor(self, tmp_path: Path) -> None:
        a, b = _pair(4, 16, 16)
        ea, eb = _pair(4, 16, 32)  # embedding LoRA: A (r, vocab=32), B (dim=16, r)
        tensors = {
            "m.layers.0.self_attn.q_proj.lora_A.weight": a,
            "m.layers.0.self_attn.q_proj.lora_B.weight": b,
            "m.embed_tokens.lora_embedding_A": ea,
            "m.embed_tokens.lora_embedding_B": eb,
            "m.layers.0.self_attn.q_proj.lora_magnitude_vector": np.ones(16, np.float32),
            "base_model.model.lm_head.weight": np.zeros((32, 16), np.float32),
            "m.layers.0.self_attn.q_proj.bias": np.zeros(16, np.float32),
            "m.layers.1.self_attn.v_proj.lora_A.weight": a,  # orphan A
            "m.layers.2.self_attn.k_proj.lora_A.weight": a,  # rank mismatch below
            "m.layers.2.self_attn.k_proj.lora_B.weight": np.zeros((16, 5), np.float32),
        }
        path = tmp_path / "adapter_model.safetensors"
        save_file(tensors, str(path))
        inv = open_adapter(path)

        assert inv.n_tensors_total == len(tensors)
        assert [p.base for p in inv.pairs] == ["m.embed_tokens", "m.layers.0.self_attn.q_proj"]
        emb = inv.pairs[0]
        assert emb.transposed and emb.location.kind == "embedding"
        assert [w.location.kind for w in inv.full_weights] == ["lm_head"]
        assert inv.dora_magnitudes == ["m.layers.0.self_attn.q_proj.lora_magnitude_vector"]
        reasons = {s.key: s.reason for s in inv.not_analyzed}
        assert "without its A/B counterpart" in reasons["m.layers.1.self_attn.v_proj.lora_A.weight"]
        assert "rank mismatch" in reasons["m.layers.2.self_attn.k_proj.lora_A.weight"]
        assert "1-D tensor" in reasons["m.layers.0.self_attn.q_proj.bias"]
        accounted = 2 * len(inv.pairs) + len(inv.full_weights) + len(inv.dora_magnitudes) + len(inv.not_analyzed)
        assert accounted == inv.n_tensors_total

    def test_structurally_invalid_file_fails_up_front(self, tmp_path: Path) -> None:
        good = tmp_path / "good.safetensors"
        a, b = _pair(4, 16, 16)
        save_file({"m.layers.0.q_proj.lora_A.weight": a, "m.layers.0.q_proj.lora_B.weight": b}, str(good))
        bad = tmp_path / "bad.safetensors"
        bad.write_bytes(good.read_bytes()[:-40])
        with pytest.raises(ValueError):
            open_adapter(bad)


class TestIteration:
    def test_values_for_f32_f16_bf16(self, tmp_path: Path) -> None:
        a, b = _pair(4, 16, 16)
        tensors = {
            "m.layers.0.q_proj.lora_A.weight": a, "m.layers.0.q_proj.lora_B.weight": b,
            "m.layers.1.q_proj.lora_A.weight": a.astype(np.float16),
            "m.layers.1.q_proj.lora_B.weight": b.astype(np.float16),
        }
        path = tmp_path / "mixed.safetensors"
        save_file(tensors, str(path))
        loaded = {pd.pair.base: pd for pd in iter_pairs(open_adapter(path))}
        np.testing.assert_array_equal(loaded["m.layers.0.q_proj"].a, a)
        np.testing.assert_allclose(loaded["m.layers.1.q_proj"].b, b, rtol=1e-3)
        assert all(pd.a.dtype == np.float32 for pd in loaded.values())

    def test_bf16_pair(self, tmp_path: Path) -> None:
        a, b = _pair(4, 16, 16)
        body = {
            "m.layers.0.q_proj.lora_A.weight": {"dtype": "BF16", "shape": [4, 16]},
            "m.layers.0.q_proj.lora_B.weight": {"dtype": "BF16", "shape": [16, 4]},
        }
        data = b""
        for (key, info), arr in zip(body.items(), (a, b)):
            raw = _bf16_bits(arr).tobytes()
            info["data_offsets"] = [len(data), len(data) + len(raw)]
            data += raw
        hdr = json.dumps(body).encode()
        path = tmp_path / "bf16.safetensors"
        path.write_bytes(len(hdr).to_bytes(8, "little") + hdr + data)
        (pd,) = list(iter_pairs(open_adapter(path)))
        np.testing.assert_allclose(pd.a, a, rtol=1e-2, atol=1e-2)

    def test_pairs_ordered_by_layer(self, tmp_path: Path) -> None:
        tensors = {}
        for layer in (10, 2, 33):
            a, b = _pair(2, 8, 8)
            tensors[f"m.layers.{layer}.mlp.down_proj.lora_A.weight"] = a
            tensors[f"m.layers.{layer}.mlp.down_proj.lora_B.weight"] = b
        path = tmp_path / "order.safetensors"
        save_file(tensors, str(path))
        assert [pd.pair.location.layer for pd in iter_pairs(open_adapter(path))] == [2, 10, 33]

    def test_peak_memory_bounded_by_one_pair(self, tmp_path: Path) -> None:
        # 24 pairs of a wide MLP shape: whole adapter ~50 MB in float32, one pair ~2 MB.
        tensors = {}
        for layer in range(24):
            a = _RNG.standard_normal((16, 4096)).astype(np.float32)
            b = _RNG.standard_normal((28672, 16)).astype(np.float32)
            tensors[f"m.layers.{layer}.mlp.down_proj.lora_A.weight"] = a
            tensors[f"m.layers.{layer}.mlp.down_proj.lora_B.weight"] = b
        path = tmp_path / "wide.safetensors"
        save_file(tensors, str(path))
        del tensors

        inv = open_adapter(path)
        total = sum(4 * (p.a.numel + p.b.numel) for p in inv.pairs)
        tracemalloc.start()
        checksum = 0.0
        for pd in iter_pairs(inv):
            checksum += float(pd.b[0, 0])
            del pd
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < 3 * inv.max_pair_bytes_f32
        assert peak < total / 5
