from __future__ import annotations

import pytest

from vision_studio.models.collections import ModelCollectionRegistry, ModelSelection
from vision_studio.models.collections.base import ModelCollectionAdapter
from vision_studio.models.collections.cache import SCHEMA_VERSION, write_metadata_cache
from vision_studio.models.collections.errors import (
    AmbiguousModelError,
    InvalidCollectionError,
    InvalidModelError,
    InvalidWeightError,
    MissingBackendError,
    UnconfirmedWeightError,
    WeightIntegrityError,
)
from vision_studio.models.collections.registry import (
    require_unambiguous_model,
    select_model_collection,
    validate_selection,
)
from vision_studio.models.collections.types import (
    CollectionMetadataCache,
    ModelOption,
    WeightMode,
    WeightOption,
)


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


def test_invalid_collection_reports_valid_options():
    registry = ModelCollectionRegistry()

    with pytest.raises(InvalidCollectionError, match="Valid collections"):
        registry.get_adapter("missing")


def test_invalid_model_reports_valid_examples(tmp_path):
    path = _cache(tmp_path, [])

    with pytest.raises(InvalidModelError, match="resnet18"):
        validate_selection(ModelSelection("timm", "missing"), cache_path=path)


def test_invalid_weight_reports_valid_weights(tmp_path):
    path = _cache(
        tmp_path,
        [
            WeightOption(
                collection_id="timm",
                model_id="resnet18",
                weight_id="imagenet",
            )
        ],
    )

    with pytest.raises(InvalidWeightError, match="imagenet"):
        validate_selection(
            ModelSelection("timm", "resnet18", WeightMode.EXPLICIT, "bad"),
            cache_path=path,
        )


def test_unconfirmed_unverifiable_weight_is_blocked(tmp_path):
    path = _cache(
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
        validate_selection(ModelSelection("timm", "resnet18"), cache_path=path)


def test_failed_integrity_is_fatal(tmp_path):
    path = _cache(
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
            cache_path=path,
        )


def test_ambiguous_model_name_requires_collection(tmp_path):
    cache_path = tmp_path / "cache.json"
    write_metadata_cache(
        CollectionMetadataCache(
            schema_version=SCHEMA_VERSION,
            generated_at="2026-06-02T00:00:00+00:00",
            backend_versions={"timm": "1", "torchvision": "1"},
            collections={
                "timm": [ModelOption(collection_id="timm", model_id="resnet18")],
                "torchvision": [
                    ModelOption(collection_id="torchvision", model_id="resnet18")
                ],
            },
        ),
        cache_path,
    )

    with pytest.raises(AmbiguousModelError, match="timm, torchvision"):
        require_unambiguous_model("resnet18", cache_path=cache_path)


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


def test_selected_missing_backend_fails_with_dependency_error(tmp_path):
    path = _cache(tmp_path, [])
    registry = ModelCollectionRegistry()
    registry.register(MissingSelectedBackendAdapter())

    with pytest.raises(MissingBackendError, match="fake_timm"):
        select_model_collection(
            ModelSelection("timm", "resnet18", WeightMode.NONE),
            cache_path=path,
            registry=registry,
        )
