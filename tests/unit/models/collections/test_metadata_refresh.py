from __future__ import annotations

from vision_studio.models.collections import (
    ModelCollectionRegistry,
    list_available_models,
    refresh_collection_metadata,
)
from vision_studio.models.collections.base import ModelCollectionAdapter
from vision_studio.models.collections.errors import MissingBackendError
from vision_studio.models.collections.types import ModelOption


class InstalledAdapter(ModelCollectionAdapter):
    collection_id = "timm"
    display_name = "Fake timm"
    backend_package = "fake_timm"

    def backend_version(self):
        return "1.0"

    def list_models(self):
        return [ModelOption(collection_id="timm", model_id="resnet18")]

    def list_weights(self, model_id):
        return []

    def create_model(self, selection, weight_id):
        return object()


class MissingAdapter(InstalledAdapter):
    collection_id = "mmpretrain"

    def list_models(self):
        raise MissingBackendError("missing mmpretrain")


def test_refresh_skips_missing_backend_and_lists_from_cache(tmp_path):
    registry = ModelCollectionRegistry()
    registry.register(InstalledAdapter())
    registry.register(MissingAdapter())
    cache_path = tmp_path / "cache.json"

    cache = refresh_collection_metadata(cache_path=cache_path, registry=registry)
    models = list_available_models(cache_path=cache_path)

    assert cache.backend_versions["timm"] == "1.0"
    assert cache.backend_versions["mmpretrain"] is None
    assert [model.model_id for model in models] == ["resnet18"]
