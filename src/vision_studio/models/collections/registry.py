from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from .base import ModelCollectionAdapter
from .cache import (
    DEFAULT_CACHE_PATH,
    SCHEMA_VERSION,
    read_metadata_cache,
    utc_now_iso,
    write_metadata_cache,
)
from .errors import (
    AmbiguousModelError,
    InvalidCollectionError,
    InvalidModelError,
    InvalidWeightError,
    MetadataCacheError,
    MissingBackendError,
    UnconfirmedWeightError,
    WeightIntegrityError,
    WeightRetrievalError,
)
from .types import (
    CollectionId,
    CollectionMetadataCache,
    ModelOption,
    ModelSelection,
    ModelSelectionResult,
    WeightMode,
)


class ModelCollectionRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ModelCollectionAdapter] = {}

    def register(self, adapter: ModelCollectionAdapter) -> None:
        self._adapters[adapter.collection_id] = adapter

    @property
    def collection_ids(self) -> tuple[str, ...]:
        return tuple(self._adapters)

    def get_adapter(self, collection_id: str | CollectionId) -> ModelCollectionAdapter:
        key = str(collection_id)
        try:
            return self._adapters[key]
        except KeyError as exc:
            valid = ", ".join(self.collection_ids) or "none"
            raise InvalidCollectionError(
                f"Unsupported model collection '{key}'. Valid collections: {valid}."
            ) from exc

    def find_ambiguous_models(self, model_id: str, models: Iterable[ModelOption]) -> set[str]:
        return {model.collection_id for model in models if model.model_id == model_id}


def create_default_registry() -> ModelCollectionRegistry:
    from .mmpretrain import MMPreTrainAdapter
    from .timm import TimmAdapter
    from .torchvision import TorchVisionAdapter

    registry = ModelCollectionRegistry()
    registry.register(TimmAdapter())
    registry.register(MMPreTrainAdapter())
    registry.register(TorchVisionAdapter())
    return registry


def _models_for_collection(
    collection_id: str,
    cache_path: str | Path,
) -> list[ModelOption]:
    cache = read_metadata_cache(cache_path)
    try:
        return list(cache.collections[collection_id])
    except KeyError as exc:
        raise InvalidCollectionError(
            f"Collection '{collection_id}' is not present in the metadata cache. "
            "Refresh metadata for installed backends."
        ) from exc


def list_available_models(
    collection_id: str | CollectionId | None = None,
    *,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> list[ModelOption]:
    cache = read_metadata_cache(cache_path)
    if collection_id is None:
        return [
            model
            for models in cache.collections.values()
            for model in models
        ]
    return list(cache.collections.get(str(collection_id), []))


def require_unambiguous_model(
    model_id: str,
    *,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> ModelOption:
    matches = [
        model
        for model in list_available_models(cache_path=cache_path)
        if model.model_id == model_id
    ]
    if not matches:
        raise InvalidModelError(f"Model '{model_id}' is not available in the cache.")
    collections = {model.collection_id for model in matches}
    if len(collections) > 1:
        valid = ", ".join(sorted(collections))
        raise AmbiguousModelError(
            f"Model '{model_id}' exists in multiple collections. "
            f"Choose a collection explicitly: {valid}."
        )
    return matches[0]


def validate_selection(
    selection: ModelSelection,
    *,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> tuple[ModelOption, str | None, tuple[str, ...]]:
    collection_id = selection.collection
    models = _models_for_collection(collection_id, cache_path)
    model = next((item for item in models if item.model_id == selection.model_id), None)
    if model is None:
        valid = ", ".join(item.model_id for item in models[:10]) or "none"
        raise InvalidModelError(
            f"Model '{selection.model_id}' is not available in collection "
            f"'{collection_id}'. Valid examples: {valid}."
        )

    warnings: list[str] = []
    weight_id: str | None
    if selection.mode == WeightMode.NONE:
        if selection.weight_id is not None:
            raise InvalidWeightError("No-pretrained-weight mode does not accept weight_id.")
        return model, None, ()

    if selection.mode == WeightMode.DEFAULT:
        if selection.weight_id is not None:
            raise InvalidWeightError(
                "Default pretrained mode does not accept explicit weight_id."
            )
        if model.default_weight_id is None:
            raise InvalidWeightError(
                f"Model '{model.model_id}' has no default pretrained weights. "
                "Choose an explicit weight or no pretrained weights."
            )
        weight_id = model.default_weight_id
    else:
        if selection.weight_id is None:
            raise InvalidWeightError("Explicit weight mode requires weight_id.")
        weight_id = selection.weight_id

    weight = next(
        (item for item in model.available_weights if item.weight_id == weight_id),
        None,
    )
    if weight is None:
        valid = ", ".join(item.weight_id for item in model.available_weights) or "none"
        raise InvalidWeightError(
            f"Weight '{weight_id}' is not compatible with model '{model.model_id}' "
            f"in collection '{collection_id}'. Valid weights: {valid}."
        )
    if weight.integrity.get("status") == "failed":
        raise WeightIntegrityError(
            f"Integrity validation failed for {collection_id}/{model.model_id}:{weight_id}."
        )
    if (weight.requires_confirmation or not weight.verified) and (
        not selection.allow_unverified_weights
    ):
        raise UnconfirmedWeightError(
            f"Weight '{weight_id}' for {collection_id}/{model.model_id} cannot be "
            "verified and requires explicit confirmation before loading."
        )
    if weight.requires_confirmation or not weight.verified:
        warnings.append(
            f"Unverifiable approved-collection weights selected for "
            f"{collection_id}/{model.model_id}:{weight_id}; loading proceeds after "
            "explicit confirmation."
        )
    return model, weight_id, tuple(warnings)


def select_model_collection(
    selection: ModelSelection,
    *,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    registry: ModelCollectionRegistry | None = None,
) -> ModelSelectionResult:
    active_registry = registry or create_default_registry()
    adapter = active_registry.get_adapter(selection.collection)
    model_option, weight_id, warnings = validate_selection(
        selection,
        cache_path=cache_path,
    )
    try:
        model = adapter.create_model(selection, weight_id)
    except MissingBackendError:
        raise
    except WeightRetrievalError:
        raise
    except Exception as exc:
        if weight_id is not None:
            raise WeightRetrievalError(
                f"Could not retrieve or apply pretrained weights for "
                f"{selection.collection}/{selection.model_id}:{weight_id}. "
                f"Reason: {exc}"
            ) from exc
        raise

    return ModelSelectionResult(
        model=model,
        collection_id=model_option.collection_id,
        model_id=model_option.model_id,
        weight_id=weight_id,
        weight_mode=selection.mode,
        source={
            "collection": model_option.collection_id,
            "model": model_option.model_id,
            "weight": weight_id,
        },
        warnings=warnings,
    )


def validate_cache_exposure(
    *,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    registry: ModelCollectionRegistry | None = None,
    minimum_ratio: float = 0.9,
) -> dict[str, dict[str, float | int | str | None | bool]]:
    active_registry = registry or create_default_registry()
    cache = read_metadata_cache(cache_path)
    evidence: dict[str, dict[str, float | int | str | None | bool]] = {}
    for collection_id, cached_models in cache.collections.items():
        adapter = active_registry.get_adapter(collection_id)
        backend_count = len(adapter.list_models())
        cached_count = len(cached_models)
        ratio = 1.0 if backend_count == 0 else cached_count / backend_count
        evidence[collection_id] = {
            "backend_count": backend_count,
            "cached_count": cached_count,
            "ratio": ratio,
            "passed": ratio >= minimum_ratio,
            "backend_version": cache.backend_versions.get(collection_id),
            "generated_at": cache.generated_at,
        }
    return evidence


def refresh_collection_metadata(
    *,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    registry: ModelCollectionRegistry | None = None,
    collection_ids: Iterable[str | CollectionId] | None = None,
) -> CollectionMetadataCache:
    active_registry = registry or create_default_registry()
    selected_ids = (
        tuple(str(item) for item in collection_ids)
        if collection_ids is not None
        else active_registry.collection_ids
    )
    collections: dict[str, list[ModelOption]] = {}
    backend_versions: dict[str, str | None] = {}
    failures: list[str] = []
    generated_at = utc_now_iso()

    for collection_id in selected_ids:
        adapter = active_registry.get_adapter(collection_id)
        try:
            backend_versions[collection_id] = adapter.backend_version()
            collections[collection_id] = [
                replace(model, discovered_at=generated_at)
                for model in adapter.list_models()
            ]
        except MissingBackendError as exc:
            backend_versions[collection_id] = None
            failures.append(str(exc))
        except Exception as exc:
            failures.append(
                f"Could not refresh collection '{collection_id}'. Reason: {exc}"
            )

    if not collections:
        detail = " ".join(failures) if failures else "No supported backends refreshed."
        raise MetadataCacheError(detail)

    cache = CollectionMetadataCache(
        schema_version=SCHEMA_VERSION,
        generated_at=generated_at,
        backend_versions=backend_versions,
        collections=collections,
    )
    return write_metadata_cache(cache, cache_path)
