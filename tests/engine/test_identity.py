"""Tests for ArtifactIdentityResolver and schema contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from adaptersentry.engine.identity import ArtifactIdentityResolver, _sha256_file
from adaptersentry.engine.schemas.identity import AdapterArtifactIdentity
from adaptersentry.engine.schemas.requests import ArtifactSource


def _make_adapter(tmp_path: Path, name: str = "adapter.safetensors") -> Path:
    rng = np.random.default_rng(42)
    tensors = {
        "model.layers.0.q_proj.lora_A.weight": rng.standard_normal((8, 64)).astype(np.float32),
        "model.layers.0.q_proj.lora_B.weight": rng.standard_normal((64, 8)).astype(np.float32),
    }
    path = tmp_path / name
    save_file(tensors, str(path), metadata={"r": "8"})
    return path


class TestContentHash:
    def test_same_file_same_hash(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        h1 = _sha256_file(path)
        h2 = _sha256_file(path)
        assert h1 == h2

    def test_hash_is_prefixed(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        h = _sha256_file(path)
        assert h.startswith("sha256:")
        assert len(h) == len("sha256:") + 64

    def test_different_files_different_hashes(self, tmp_path: Path) -> None:
        path1 = _make_adapter(tmp_path, "a.safetensors")
        path2 = tmp_path / "b.safetensors"
        rng = np.random.default_rng(99)
        save_file(
            {"x.lora_A.weight": rng.standard_normal((8, 64)).astype(np.float32)},
            str(path2),
        )
        assert _sha256_file(path1) != _sha256_file(path2)


class TestArtifactIdentityResolver:
    def test_returns_identity_instance(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        source = ArtifactSource(kind="local_path", local_path=str(path))
        identity = ArtifactIdentityResolver.resolve(path, source)
        assert isinstance(identity, AdapterArtifactIdentity)

    def test_content_hash_matches_manual(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        source = ArtifactSource(kind="local_path", local_path=str(path))
        identity = ArtifactIdentityResolver.resolve(path, source)
        expected = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        assert identity.content_hash == expected

    def test_header_hash_is_populated(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        source = ArtifactSource(kind="local_path", local_path=str(path))
        identity = ArtifactIdentityResolver.resolve(path, source)
        assert identity.header_hash.startswith("sha256:")

    def test_header_hash_differs_from_content_hash(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        source = ArtifactSource(kind="local_path", local_path=str(path))
        identity = ArtifactIdentityResolver.resolve(path, source)
        assert identity.header_hash != identity.content_hash

    def test_file_size_correct(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        source = ArtifactSource(kind="local_path", local_path=str(path))
        identity = ArtifactIdentityResolver.resolve(path, source)
        assert identity.file_size_bytes == path.stat().st_size

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.safetensors"
        source = ArtifactSource(kind="local_path", local_path=str(path))
        with pytest.raises(FileNotFoundError):
            ArtifactIdentityResolver.resolve(path, source)

    def test_deterministic_across_calls(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        source = ArtifactSource(kind="local_path", local_path=str(path))
        i1 = ArtifactIdentityResolver.resolve(path, source)
        i2 = ArtifactIdentityResolver.resolve(path, source)
        assert i1.content_hash == i2.content_hash
        assert i1.logical_id == i2.logical_id

    def test_hf_hub_logical_id_differs_from_local(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        local_source = ArtifactSource(kind="local_path", local_path=str(path))
        hf_source = ArtifactSource(
            kind="hf_hub", hf_repo_id="user/repo", hf_revision="abc123", local_path=str(path)
        )
        local_id = ArtifactIdentityResolver.resolve(path, local_source)
        hf_id = ArtifactIdentityResolver.resolve(path, hf_source)
        # Content hash is the same (same file bytes)
        assert local_id.content_hash == hf_id.content_hash
        # But logical_id differs (one is path-based, other is repo-based)
        assert local_id.logical_id != hf_id.logical_id


class TestIdentitySchemaContract:
    def test_round_trip_serialization(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        source = ArtifactSource(kind="local_path", local_path=str(path))
        identity = ArtifactIdentityResolver.resolve(path, source)
        json_str = identity.model_dump_json()
        restored = AdapterArtifactIdentity.model_validate_json(json_str)
        assert restored.content_hash == identity.content_hash
        assert restored.logical_id == identity.logical_id

    def test_extra_fields_ignored(self, tmp_path: Path) -> None:
        path = _make_adapter(tmp_path)
        source = ArtifactSource(kind="local_path", local_path=str(path))
        identity = ArtifactIdentityResolver.resolve(path, source)
        data = identity.model_dump()
        data["future_field_from_newer_writer"] = "should be ignored"
        restored = AdapterArtifactIdentity.model_validate(data)
        assert restored.content_hash == identity.content_hash
