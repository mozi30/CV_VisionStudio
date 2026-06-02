from __future__ import annotations

import pytest

from vision_studio.models.collections.base import ModelCollectionAdapter
from vision_studio.models.collections.cache import SCHEMA_VERSION, write_metadata_cache
from vision_studio.models.collections.errors import (
    MissingBackendError,
    UnconfirmedWeightError,
    WeightIntegrityError,
)
from vision_studio.models.collections.types import (
    CollectionMetadataCache,
    ModelOption,
    WeightMode,
    WeightOption,
)


class MissingSelectedBackendAdapter(ModelCollectionAdapter):
    collection_id = "timm"
    display_name = "Fake timm"
    backend_package = "fake_timm"

    def list_models(self):
        return []

    def list_weights(self, model_id):
        return []

    def create_model(self, selection, weight_id):
        raise MissingBackendError("install fake_timm")


def _cache(tmp_path, weights):
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
                        available_weights=tuple(weights),
                        default_weight_id=weights[0].weight_id if weights else None,
                    )
                ]
            },
        ),
        path,
    )
    return path


def test_selected_missing_backend_error_is_actionable(tmp_path):
    from vision_studio.models.collections import ModelCollectionRegistry, ModelSelection
    from vision_studio.models.collections.registry import select_model_collection

    registry = ModelCollectionRegistry()
    registry.register(MissingSelectedBackendAdapter())

    with pytest.raises(MissingBackendError, match="fake_timm"):
        select_model_collection(
            ModelSelection("timm", "resnet18", WeightMode.NONE),
            cache_path=_cache(tmp_path, []),
            registry=registry,
        )


def test_unverifiable_weight_requires_confirmation(tmp_path):
    from vision_studio.models.collections import ModelSelection
    from vision_studio.models.collections.registry import validate_selection

    cache_path = _cache(
        tmp_path,
        [
            WeightOption(
                collection_id="timm",
                model_id="resnet18",
                weight_id="default",
                is_default=True,
                verified=False,
                requires_confirmation=True,
            )
        ],
    )

    with pytest.raises(UnconfirmedWeightError):
        validate_selection(ModelSelection("timm", "resnet18"), cache_path=cache_path)


def test_failed_integrity_blocks_loading(tmp_path):
    from vision_studio.models.collections import ModelSelection
    from vision_studio.models.collections.registry import validate_selection

    cache_path = _cache(
        tmp_path,
        [
            WeightOption(
                collection_id="timm",
                model_id="resnet18",
                weight_id="default",
                is_default=True,
                integrity={"status": "failed"},
            )
        ],
    )

    with pytest.raises(WeightIntegrityError):
        validate_selection(
            ModelSelection("timm", "resnet18", allow_unverified_weights=True),
            cache_path=cache_path,
        )
