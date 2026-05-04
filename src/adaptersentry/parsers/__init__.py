"""Parsers subpackage — file loading and metadata extraction."""

from adaptersentry.schemas.errors import ParseErrorClass

from .safetensors import (
    ParsedTensor,
    check_lora_architecture,
    load_adapter,
    parse_tensors,
    _group_lora_layers,
)
from .metadata import MetadataExtractor, _metadata_depth, parse_adapter_metadata

__all__ = [
    "ParseErrorClass",
    "ParsedTensor",
    "parse_tensors",
    "load_adapter",
    "check_lora_architecture",
    "_group_lora_layers",
    "MetadataExtractor",
    "_metadata_depth",
    "parse_adapter_metadata",
]
