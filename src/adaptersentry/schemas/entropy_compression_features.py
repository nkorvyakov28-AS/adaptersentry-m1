"""EntropyCompressionFeatures — compression and entropy signals for LoRA weight matrices.

All statistics are computed per-tensor (lora_A and lora_B separately) because
quantization and compression artifacts affect the *stored* matrices, not the
combined ΔW. Per-matrix stats are supplementary signals per the M1 contract.

value_repeat_ratio_{a,b}
    Fraction of elements that share a value with at least one other element.
    High values indicate quantization snapping or deliberate value repetition.

unique_value_ratio_{a,b}
    Fraction of unique values relative to total element count.
    Low values suggest quantization or pattern injection.

approx_compression_ratio_{a,b}
    zlib-compressed size / raw float32 byte size (level=1, fast).
    Low ratio → highly compressible → repetitive byte patterns.
    Normal trained weights: ~0.95–1.05. Quantized: can drop to ~0.5–0.8.

byte_entropy_{a,b}
    Normalized Shannon entropy of the raw float32 byte stream (256 symbols).
    Near-zero → constant byte patterns (quantized or uninitialized).
    Near-one → uniform byte distribution.

sign_entropy_{a,b}
    Normalized Shannon entropy of the sign pattern (+1/−1/0), base log2(3).
    Near-zero → all-positive or all-negative (bias injection signal).
    Near-one → balanced mixed signs.

sign_balance_{a,b}
    Fraction of elements strictly greater than zero.
    Values near 0 or 1 combined with low sign_entropy are anomalous.

quantization_suspect_score_{a,b}
    Heuristic in [0, 1]. Derived from effective_bits = log2(num_unique + 1).
    Score = max(0, 1 − effective_bits / 32). Interpretation:
      0.0 → all values unique (normal full-precision training)
      0.75 → ~256 unique values (consistent with 8-bit post-training quant)
      0.875 → ~16 unique values (4-bit quantization)
      1.0 → all values identical (degenerate / uninitialized)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EntropyCompressionFeatures(BaseModel):
    """Entropy and compression statistics for lora_A and lora_B matrices."""

    model_config = ConfigDict(frozen=True)

    # --- lora_A ---
    value_repeat_ratio_a: float = Field(
        default=0.0,
        description="Fraction of lora_A elements that share a value with another element.",
    )
    unique_value_ratio_a: float = Field(
        default=0.0,
        description="Fraction of unique values in lora_A (unique_count / numel).",
    )
    approx_compression_ratio_a: float = Field(
        default=0.0,
        description="zlib compressed / raw byte size for lora_A (float32 bytes, level=1).",
    )
    byte_entropy_a: float = Field(
        default=0.0,
        description="Normalized Shannon entropy of lora_A raw bytes (0–1, 256 symbols).",
    )
    sign_entropy_a: float = Field(
        default=0.0,
        description="Normalized Shannon entropy of lora_A sign pattern (0–1, base log2(3)).",
    )
    sign_balance_a: float = Field(
        default=0.5,
        description="Fraction of lora_A elements strictly > 0.",
    )
    quantization_suspect_score_a: float = Field(
        default=0.0,
        description=(
            "Heuristic quantization score for lora_A in [0, 1]. "
            "Near 0 = all unique (normal fp32); near 1 = very few unique values."
        ),
    )

    # --- lora_B ---
    value_repeat_ratio_b: float = Field(
        default=0.0,
        description="Fraction of lora_B elements that share a value with another element.",
    )
    unique_value_ratio_b: float = Field(
        default=0.0,
        description="Fraction of unique values in lora_B (unique_count / numel).",
    )
    approx_compression_ratio_b: float = Field(
        default=0.0,
        description="zlib compressed / raw byte size for lora_B (float32 bytes, level=1).",
    )
    byte_entropy_b: float = Field(
        default=0.0,
        description="Normalized Shannon entropy of lora_B raw bytes (0–1, 256 symbols).",
    )
    sign_entropy_b: float = Field(
        default=0.0,
        description="Normalized Shannon entropy of lora_B sign pattern (0–1, base log2(3)).",
    )
    sign_balance_b: float = Field(
        default=0.5,
        description="Fraction of lora_B elements strictly > 0.",
    )
    quantization_suspect_score_b: float = Field(
        default=0.0,
        description=(
            "Heuristic quantization score for lora_B in [0, 1]. "
            "Near 0 = all unique (normal fp32); near 1 = very few unique values."
        ),
    )
