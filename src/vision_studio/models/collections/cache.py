from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import MetadataCacheError
from .types import CollectionMetadataCache, ModelOption, WeightOption

SCHEMA_VERSION = 1
DEFAULT_CACHE_PATH = Path.home() / ".cache" / "vision_studio" / "model_collections.json"


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _weight_from_dict(data: dict[str, Any]) -> WeightOption:
    return WeightOption(
        collection_id=data["collection_id"],
        model_id=data["model_id"],
        weight_id=data["weight_id"],
        is_default=bool(data.get("is_default", False)),
        source=dict(data.get("source", {})),
        integrity=dict(data.get("integrity", {})),
        verified=bool(data.get("verified", True)),
        requires_confirmation=bool(data.get("requires_confirmation", False)),
    )


def model_from_dict(data: dict[str, Any]) -> ModelOption:
    return ModelOption(
        collection_id=data["collection_id"],
        model_id=data["model_id"],
        display_name=data.get("display_name"),
        available_weights=tuple(
            _weight_from_dict(item) for item in data.get("available_weights", [])
        ),
        default_weight_id=data.get("default_weight_id"),
        metadata=dict(data.get("metadata", {})),
        discovered_at=data.get("discovered_at"),
    )


def model_to_dict(model: ModelOption) -> dict[str, Any]:
    data = asdict(model)
    data["available_weights"] = [asdict(weight) for weight in model.available_weights]
    return data


def read_metadata_cache(path: str | Path = DEFAULT_CACHE_PATH) -> CollectionMetadataCache:
    cache_path = Path(path)
    if not cache_path.exists():
        raise MetadataCacheError(
            f"Model collection metadata cache is missing at {cache_path}. "
            "Run a metadata refresh before listing models."
        )
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MetadataCacheError(
            "Model collection metadata cache is corrupt. Refresh metadata to rebuild it."
        ) from exc

    if data.get("schema_version") != SCHEMA_VERSION:
        raise MetadataCacheError(
            "Model collection metadata cache schema is incompatible. "
            "Refresh metadata to rebuild it."
        )

    collections = {
        collection_id: [model_from_dict(item) for item in models]
        for collection_id, models in dict(data.get("collections", {})).items()
    }
    return CollectionMetadataCache(
        schema_version=SCHEMA_VERSION,
        generated_at=str(data.get("generated_at", "")),
        backend_versions=dict(data.get("backend_versions", {})),
        collections=collections,
        source=cache_path,
    )


def write_metadata_cache(
    cache: CollectionMetadataCache,
    path: str | Path = DEFAULT_CACHE_PATH,
) -> CollectionMetadataCache:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": cache.generated_at,
        "backend_versions": cache.backend_versions,
        "collections": {
            collection_id: [model_to_dict(model) for model in models]
            for collection_id, models in cache.collections.items()
        },
    }
    cache_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return CollectionMetadataCache(
        schema_version=cache.schema_version,
        generated_at=cache.generated_at,
        backend_versions=cache.backend_versions,
        collections=cache.collections,
        source=cache_path,
    )
