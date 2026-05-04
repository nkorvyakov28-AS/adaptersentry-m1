"""Synthetic LoRA adapter corpus generator for benchmarking.

Generates deterministic .safetensors files with realistic LoRA A/B tensor pairs.
All files are structurally valid and parseable by the M1 analyzer.

Tensor naming follows HuggingFace PEFT convention so the parser recognises
them as lora_A / lora_B pairs and groups them correctly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file as _save_file

logger = logging.getLogger(__name__)

# Default layer names that mirror common LLM target modules
_DEFAULT_MODULE_NAMES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def generate_corpus(
    n: int,
    output_dir: Path,
    *,
    rank: int = 8,
    n_layers: int = 4,
    hidden_dim: int = 64,
    n_modules: int = 4,
    seed: int = 42,
) -> list[Path]:
    """Generate N synthetic .safetensors adapter files in output_dir.

    Each file contains n_layers * n_modules lora_A / lora_B tensor pairs.
    Tensors are random float32 drawn from N(0, 0.02) — typical LoRA init scale.
    File names are zero-padded: adapter_0000.safetensors ... adapter_NNNN.safetensors.

    Args:
        n:          Number of adapter files to generate.
        output_dir: Directory to write files into (created if absent).
        rank:       LoRA rank r — lora_A shape is (r, hidden_dim),
                    lora_B shape is (hidden_dim, r).
        n_layers:   Number of transformer layers per adapter.
        hidden_dim: Hidden dimension of the base model.
        n_modules:  Number of target modules per layer (max 7).
        seed:       Base RNG seed; each adapter uses seed + adapter_index.

    Returns:
        Sorted list of generated file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modules = _DEFAULT_MODULE_NAMES[:n_modules]
    paths: list[Path] = []

    for i in range(n):
        rng = np.random.default_rng(seed + i)
        tensors: dict[str, np.ndarray] = {}

        for layer_idx in range(n_layers):
            for module in modules:
                key_a = f"model.layers.{layer_idx}.{module}.lora_A.weight"
                key_b = f"model.layers.{layer_idx}.{module}.lora_B.weight"
                tensors[key_a] = rng.normal(0.0, 0.02, size=(rank, hidden_dim)).astype(np.float32)
                tensors[key_b] = np.zeros((hidden_dim, rank), dtype=np.float32)

        metadata = {
            "r": str(rank),
            "lora_alpha": str(rank * 2),
            "peft_type": "LORA",
            "base_model_name_or_path": "benchmark/synthetic-base",
            "target_modules": ",".join(modules),
        }

        out_path = output_dir / f"adapter_{i:04d}.safetensors"
        _save_file(tensors, str(out_path), metadata=metadata)
        paths.append(out_path)

    logger.info("Generated %d synthetic adapters in %s", n, output_dir)
    return sorted(paths)


def generate_anomalous_corpus(
    n: int,
    output_dir: Path,
    *,
    rank: int = 8,
    n_layers: int = 4,
    hidden_dim: int = 64,
    n_modules: int = 4,
    seed: int = 999,
) -> list[Path]:
    """Generate N adapters with injected anomalies for error-rate validation.

    Anomalies: heavy-tailed lora_A (Cauchy distribution), non-zero lora_B,
    high kurtosis. These should score HIGH/CRITICAL in the ensemble.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modules = _DEFAULT_MODULE_NAMES[:n_modules]
    paths: list[Path] = []

    for i in range(n):
        rng = np.random.default_rng(seed + i)
        tensors: dict[str, np.ndarray] = {}

        for layer_idx in range(n_layers):
            for module in modules:
                key_a = f"model.layers.{layer_idx}.{module}.lora_A.weight"
                key_b = f"model.layers.{layer_idx}.{module}.lora_B.weight"
                a = rng.standard_cauchy(size=(rank, hidden_dim)).astype(np.float32)
                np.clip(a, -100, 100, out=a)
                b = rng.normal(0.0, 1.0, size=(hidden_dim, rank)).astype(np.float32)
                tensors[key_a] = a
                tensors[key_b] = b

        metadata = {"r": str(rank), "peft_type": "LORA"}
        out_path = output_dir / f"anomalous_{i:04d}.safetensors"
        _save_file(tensors, str(out_path), metadata=metadata)
        paths.append(out_path)

    logger.info("Generated %d anomalous adapters in %s", n, output_dir)
    return sorted(paths)
