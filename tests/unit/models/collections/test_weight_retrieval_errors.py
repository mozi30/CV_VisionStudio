from __future__ import annotations

import pytest

from vision_studio.models.collections import (
    ModelCollectionRegistry,
    ModelSelection,
    WeightRetrievalError,
    select_model_collection,
)
from vision_studio.models.collections.base import ModelCollectionAdapter
from vision_studio.models.collections.cache import SCHEMA_VERSION, write_metadata_cache
from vision_studio.models.collections.types import (
    CollectionMetadataCache,
    ModelOption,
    WeightOption,
)


class FailingAdapter(ModelCollectionAdapter):
    collection_id = "timm"
    display_name = "Fake timm"
    backend_package = "fake_timm"

    def list_models(self):
        return []

    def list_weights(self, model_id):
        return []

    def create_model(self, selection, weight_id):
        raise RuntimeError("download failed")


def test_runtime_weight_retrieval_failure_raises_typed_error(tmp_path):
    cache_path = tmp_path / "cache.json"
    write_metadata_cache(
        CollectionMetadataCache(
            schema_version=SCHEMA_VERSION,
            generated_at="2026-06-02T00:00:00+00:00",
            backend_versions={"timm": "test"},
            collections={
                "timm": [
                    ModelOption(
                        collection_id="timm",
                        model_id="resnet18",
                        available_weights=(
                            WeightOption(
                                collection_id="timm",
                                model_id="resnet18",
                                weight_id="default",
                                is_default=True,
                            ),
                        ),
                        default_weight_id="default",
                    )
                ]
            },
        ),
        cache_path,
    )
    registry = ModelCollectionRegistry()
    registry.register(FailingAdapter())

    with pytest.raises(WeightRetrievalError, match="download failed"):
        select_model_collection(
            ModelSelection("timm", "resnet18"),
            cache_path=cache_path,
            registry=registry,
        )
