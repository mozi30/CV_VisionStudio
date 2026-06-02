from __future__ import annotations

from tests.unit.models.collections.test_metadata_refresh import InstalledAdapter
from vision_studio.models.collections import (
    ModelCollectionRegistry,
    list_available_models,
    refresh_collection_metadata,
)


def test_refresh_then_cached_listing_through_public_sdk_helpers(tmp_path):
    cache_path = tmp_path / "cache.json"
    registry = ModelCollectionRegistry()
    registry.register(InstalledAdapter())

    refresh_collection_metadata(cache_path=cache_path, registry=registry)
    models = list_available_models("timm", cache_path=cache_path)

    assert [model.model_id for model in models] == ["resnet18"]
