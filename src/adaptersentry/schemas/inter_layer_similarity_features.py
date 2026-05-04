"""InterLayerSimilarityFeatures — adapter-level cross-layer weight similarity.

Measures pairwise similarity between ΔW = B @ A matrices across all LoRA
layers. High cosine similarity between non-adjacent layers is a backdoor signal:
legitimate adapters learn different updates for different transformer modules;
backdoored adapters may replicate similar patterns across multiple layers to
maximize trigger effectiveness.

fast mode: uses lora_A rows as a ΔW proxy (O(n) per layer, no matmul).
full mode: materializes ΔW = B @ A when out × in ≤ 16M; falls back to proxy.

Grouping is by module type (q_proj, v_proj, etc.) extracted from the layer name.
Pairwise comparisons are restricted to layers of the same (A_shape, B_shape)
to ensure vectors have the same dimension.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SimilarPair(BaseModel):
    """A suspicious layer pair with high cosine similarity."""

    model_config = ConfigDict(frozen=True)

    layer_a: str
    layer_b: str
    index_a: int = Field(description="Position of layer_a in the original layer order.")
    index_b: int = Field(description="Position of layer_b in the original layer order.")
    cosine_sim: float = Field(description="Cosine similarity of ΔW vectors.")
    pearson: float | None = Field(
        default=None,
        description="Pearson correlation of ΔW vectors (None if degenerate).",
    )


class InterLayerSimilarityFeatures(BaseModel):
    """Adapter-level inter-layer ΔW similarity statistics."""

    model_config = ConfigDict(frozen=True)

    cosine_sim_mean: float = Field(
        default=0.0,
        description="Mean pairwise cosine similarity across all same-shape layer pairs.",
    )
    cosine_sim_std: float = Field(
        default=0.0,
        description="Std of pairwise cosine similarities.",
    )
    pearson_mean: float = Field(
        default=0.0,
        description="Mean pairwise Pearson correlation across same-shape layer pairs.",
    )
    n_pairs_computed: int = Field(
        default=0,
        description="Number of layer pairs for which similarity was computed.",
    )
    n_suspicious_pairs: int = Field(
        default=0,
        description="Non-adjacent pairs with cosine_sim > 0.85.",
    )
    top_suspicious_pairs: list[SimilarPair] = Field(
        default_factory=list,
        description="Top-5 suspicious non-adjacent pairs ranked by cosine_sim descending.",
    )
    module_group_similarities: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-module-type mean cosine similarity. "
            "Keys are module names extracted from layer paths (e.g. 'q_proj', 'v_proj')."
        ),
    )
    computed_on_proxy: bool = Field(
        default=False,
        description=(
            "True when any layer's vector was computed from lora_A rows instead of "
            "the full ΔW = B @ A (fast mode or delta too large to materialize)."
        ),
    )
