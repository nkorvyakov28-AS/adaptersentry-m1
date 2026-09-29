"""Synthetic PEFT LoRA adapters for tests (safetensors.numpy; no torch)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

MODULES = (("self_attn.q_proj", 256, 256), ("self_attn.v_proj", 128, 256), ("mlp.down_proj", 256, 512))

DEFAULT_CONFIG = {
    "peft_type": "LORA",
    "r": 8,
    "lora_alpha": 16,
    "base_model_name_or_path": "meta-llama/Llama-3.2-1B",
    "target_modules": ["q_proj", "v_proj", "down_proj"],
}


def write_adapter(
    directory: Path,
    *,
    n_layers: int = 16,
    r: int = 8,
    inject_layer: int | None = None,
    config: dict | None = None,
    extra: dict[str, np.ndarray] | None = None,
    b_scale: float = 0.01,
    seed: int = 0,
    name: str = "adapter_model.safetensors",
    metadata: dict[str, str] | None = None,
) -> Path:
    """Write a synthetic adapter (and adapter_config.json) into *directory*.

    Args:
        inject_layer: if set, one rank slot of that layer's down_proj writes into
            4 outputs only — a toy "payload" the intra-adapter comparison should find.
        config: adapter_config.json content; {} writes no config file.
        extra: additional tensors (e.g. full weights, NaN layers).
    """
    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {}
    for layer in range(n_layers):
        for mod, out, inp in MODULES:
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
    save_file(tensors, str(path), metadata=metadata if metadata is not None else {"format": "pt"})
    cfg = dict(DEFAULT_CONFIG, r=r, lora_alpha=2 * r) if config is None else config
    if cfg:
        (directory / "adapter_config.json").write_text(json.dumps(cfg))
    return path
