from __future__ import annotations

from vision_studio.models.collections import (
    ModelCollectionRegistry,
    validate_cache_exposure,
)
from vision_studio.models.collections.base import ModelCollectionAdapter
from vision_studio.models.collections.cache import SCHEMA_VERSION, write_metadata_cache
from vision_studio.models.collections.registry import require_unambiguous_model
from vision_studio.models.collections.types import CollectionMetadataCache, ModelOption


class CountingAdapter(ModelCollectionAdapter):
    collection_id = "timm"
    display_name = "Fake timm"
    backend_package = "fake_timm"

    def list_models(self):
        return [
            ModelOption(collection_id="timm", model_id="a"),
            ModelOption(collection_id="timm", model_id="b"),
        ]

    def list_weights(self, model_id):
        return []

    def create_model(self, selection, weight_id):
        return object()


def test_cache_exposure_compares_cached_count_to_backend_count(tmp_path):
    cache_path = tmp_path / "cache.json"
    write_metadata_cache(
        CollectionMetadataCache(
            schema_version=SCHEMA_VERSION,
            generated_at="2026-06-02T00:00:00+00:00",
            backend_versions={"timm": "1"},
            collections={
                "timm": [
                    ModelOption(collection_id="timm", model_id="a"),
                    ModelOption(collection_id="timm", model_id="b"),
                ]
            },
        ),
        cache_path,
    )
    registry = ModelCollectionRegistry()
    registry.register(CountingAdapter())

    evidence = validate_cache_exposure(cache_path=cache_path, registry=registry)

    assert evidence["timm"]["cached_count"] == 2
    assert evidence["timm"]["backend_count"] == 2
    assert evidence["timm"]["passed"] is True


def test_require_unambiguous_model_returns_single_collection_match(tmp_path):
    cache_path = tmp_path / "cache.json"
    write_metadata_cache(
        CollectionMetadataCache(
            schema_version=SCHEMA_VERSION,
            generated_at="2026-06-02T00:00:00+00:00",
            backend_versions={"timm": "1"},
            collections={"timm": [ModelOption(collection_id="timm", model_id="a")]},
        ),
        cache_path,
    )

    assert require_unambiguous_model("a", cache_path=cache_path).collection_id == "timm"
