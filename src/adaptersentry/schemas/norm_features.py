"""NormFeatures schema — per-layer LoRA delta magnitude features.

Captures magnitude signals from the composed weight update ΔW = B @ A.
All values are computed from the combined delta, never from A or B alone.

Future normalization hook
--------------------------
delta_norm_ratio is currently unnormalized. Once claimed_rank and lora_alpha
are reliably available from AdapterMetadata, a cross-adapter comparable score
can be derived as:

    normalized_fro = fro_norm_delta * (claimed_rank / lora_alpha)

That computation belongs in the scorer, not here.  This schema intentionally
exposes the raw magnitude so the scorer can choose its own normalization.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class NormFeatures(BaseModel):
    """Magnitude features derived from ΔW = B @ A for one LoRA layer pair.

    delta_norm_ratio is bounded in [0, 1] by Frobenius submultiplicativity:
        ||B @ A||_F ≤ ||B||_F × ||A||_F

    A ratio near 0 means the matrices nearly cancel (or B is near zero-init).
    A ratio near 1 means A and B are maximally aligned — the composed update
    concentrates its energy along a single direction, which is a potential
    backdoor signal when combined with high energy_concentration.

    Security Notes:
        - Computed exclusively from ΔW = B @ A, never from A or B in isolation.
        - delta_norm_ratio = 0.0 when B is at LoRA zero-init state — not anomalous.
        - max_abs_delta and mean_abs_delta are 0.0 when the full delta matrix
          is too large to materialise (see _MAX_DELTA_NUMEL in delta_norm.py).
    """

    model_config = ConfigDict(frozen=True)

    fro_norm_delta: float = Field(
        description="Frobenius norm of ΔW = B @ A",
    )
    max_abs_delta: float = Field(
        description=(
            "Maximum absolute element of ΔW. "
            "0.0 when the delta matrix exceeds the memory guard."
        ),
    )
    mean_abs_delta: float = Field(
        description=(
            "Mean absolute element of ΔW. "
            "0.0 when the delta matrix exceeds the memory guard."
        ),
    )
    delta_norm_ratio: float = Field(
        description=(
            "fro_norm_delta / (||A||_F × ||B||_F). "
            "Bounded in [0, 1]; 0.0 = zero-B init (not anomalous). "
            "Unnormalized — future scorer divides by (rank / lora_alpha)."
        ),
    )
