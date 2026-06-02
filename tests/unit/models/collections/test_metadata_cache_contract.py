from __future__ import annotations

import pytest

from vision_studio.models.collections.cache import (
    SCHEMA_VERSION,
    read_metadata_cache,
    write_metadata_cache,
)
from vision_studio.models.collections.errors import MetadataCacheError
from vision_studio.models.collections.types import CollectionMetadataCache, ModelOption


def test_metadata_cache_round_trips_schema_and_models(tmp_path):
    path = tmp_path / "cache.json"
    write_metadata_cache(
        CollectionMetadataCache(
            schema_version=SCHEMA_VERSION,
            generated_at="2026-06-02T00:00:00+00:00",
            backend_versions={"timm": "1"},
            collections={
                "timm": [ModelOption(collection_id="timm", model_id="resnet18")]
            },
        ),
        path,
    )

    cache = read_metadata_cache(path)

    assert cache.schema_version == SCHEMA_VERSION
    assert cache.backend_versions == {"timm": "1"}
    assert cache.collections["timm"][0].model_id == "resnet18"


def test_missing_cache_suggests_refresh(tmp_path):
    with pytest.raises(MetadataCacheError, match="refresh"):
        read_metadata_cache(tmp_path / "missing.json")


def test_corrupt_cache_suggests_refresh(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(MetadataCacheError, match="Refresh"):
        read_metadata_cache(path)
