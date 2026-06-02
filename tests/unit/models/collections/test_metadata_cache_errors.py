from __future__ import annotations

import pytest

from vision_studio.models.collections.cache import read_metadata_cache
from vision_studio.models.collections.errors import MetadataCacheError


def test_missing_cache_reports_refresh_guidance(tmp_path):
    with pytest.raises(MetadataCacheError, match="refresh"):
        read_metadata_cache(tmp_path / "missing.json")


def test_corrupt_cache_reports_refresh_guidance(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(MetadataCacheError, match="Refresh"):
        read_metadata_cache(cache_path)
