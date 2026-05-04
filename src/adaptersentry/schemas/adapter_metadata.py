"""AdapterMetadata schema — structured representation of adapter provenance."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AdapterMetadata(BaseModel):
    """Structured adapter metadata derived from safetensors header and config.

    Field name mapping vs. task spec
    ---------------------------------
    rank          → claimed_rank  (existing convention; "claimed" signals untrusted provenance)
    base_model_name → base_model  (existing convention)
    lora_alpha    → lora_alpha    (new)
    target_modules → target_modules (unchanged)
    peft_type     → peft_type     (unchanged)
    """

    model_config = ConfigDict(frozen=True)

    claimed_rank: int | None = Field(
        default=None,
        description="LoRA rank r declared in metadata; None if not specified",
    )
    lora_alpha: float | None = Field(
        default=None,
        description="LoRA alpha scaling factor declared in metadata; None if absent",
    )
    base_model: str | None = Field(
        default=None,
        description="base_model_name_or_path from adapter config (task: base_model_name)",
    )
    target_modules: list[str] = Field(
        default_factory=list,
        description="LoRA target module names from adapter config",
    )
    peft_type: str | None = Field(
        default=None,
        description="PEFT adapter type (e.g. 'LORA')",
    )
    framework: str | None = Field(
        default=None,
        description="ML framework hint from metadata",
    )
    metadata_depth: int = Field(
        default=0,
        description="Maximum nesting depth of raw safetensors metadata (security signal)",
    )
    metadata_present: bool = Field(
        default=False,
        description=(
            "True if the safetensors header contained any metadata. "
            "False is a weak suspicious signal — legitimate distributed adapters "
            "usually carry provenance metadata."
        ),
    )
    raw: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw safetensors header metadata (string-valued)",
    )

    @classmethod
    def from_parsed(cls, parsed: dict) -> "AdapterMetadata":
        """Construct from output of parsers.metadata.parse_adapter_metadata()."""
        return cls(
            claimed_rank=parsed.get("claimed_rank"),
            lora_alpha=parsed.get("lora_alpha"),
            base_model=parsed.get("base_model"),
            target_modules=parsed.get("target_modules", []),
            peft_type=parsed.get("peft_type"),
            framework=parsed.get("framework"),
            metadata_depth=parsed.get("depth", 0),
            metadata_present=parsed.get("metadata_present", False),
            raw=parsed.get("raw", {}),
        )
