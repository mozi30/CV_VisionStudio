from __future__ import annotations

from vision_studio.models.collections import (
    ModelCollectionRegistry,
    ModelSelection,
    WeightMode,
    select_model_collection,
)
from vision_studio.models.collections.base import ModelCollectionAdapter
from vision_studio.models.collections.cache import SCHEMA_VERSION, write_metadata_cache
from vision_studio.models.collections.types import (
    CollectionMetadataCache,
    ModelOption,
    WeightOption,
)


class FakeAdapter(ModelCollectionAdapter):
    collection_id = "timm"
    display_name = "Fake timm"
    backend_package = "fake_timm"

    def list_models(self):
        return []

    def list_weights(self, model_id):
        return []

    def create_model(self, selection, weight_id):
        return {"model_id": selection.model_id, "weight_id": weight_id}


def _cache(tmp_path):
    path = tmp_path / "cache.json"
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
                                weight_id="imagenet",
                                source={"backend": "timm"},
                            ),
                            WeightOption(
                                collection_id="timm",
                                model_id="resnet18",
                                weight_id="default",
                                is_default=True,
                                source={"backend": "timm"},
                            ),
                        ),
                        default_weight_id="default",
                    )
                ]
            },
        ),
        path,
    )
    return path


def _registry():
    registry = ModelCollectionRegistry()
    registry.register(FakeAdapter())
    return registry


def test_selects_explicit_weight(tmp_path):
    result = select_model_collection(
        ModelSelection("timm", "resnet18", WeightMode.EXPLICIT, "imagenet"),
        cache_path=_cache(tmp_path),
        registry=_registry(),
    )

    assert result.model == {"model_id": "resnet18", "weight_id": "imagenet"}
    assert result.weight_id == "imagenet"
    assert result.source == {
        "collection": "timm",
        "model": "resnet18",
        "weight": "imagenet",
    }


def test_selects_default_weight(tmp_path):
    result = select_model_collection(
        ModelSelection("timm", "resnet18"),
        cache_path=_cache(tmp_path),
        registry=_registry(),
    )

    assert result.weight_id == "default"
    assert result.weight_mode == WeightMode.DEFAULT


def test_selects_no_pretrained_weight(tmp_path):
    result = select_model_collection(
        ModelSelection("timm", "resnet18", WeightMode.NONE),
        cache_path=_cache(tmp_path),
        registry=_registry(),
    )

    assert result.weight_id is None
    assert result.model["weight_id"] is None
