from __future__ import annotations

import builtins

from vision_studio.models.collections import list_available_models
from vision_studio.models.collections.cache import SCHEMA_VERSION, write_metadata_cache
from vision_studio.models.collections.types import CollectionMetadataCache, ModelOption


def test_cached_listing_does_not_import_backend_catalogs(tmp_path, monkeypatch):
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
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"timm", "mmpretrain"}:
            raise AssertionError(f"unexpected live backend import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert list_available_models(cache_path=cache_path)[0].model_id == "a"
