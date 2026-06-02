from __future__ import annotations

from tests.unit.models.collections.test_model_selection_contract import (
    _cache,
    _registry,
)
from vision_studio.models.collections import ModelSelection, WeightMode
from vision_studio.models.collections.registry import select_model_collection


def test_public_sdk_selection_workflow_supports_all_weight_modes(tmp_path):
    cache_path = _cache(tmp_path)
    registry = _registry()

    explicit = select_model_collection(
        ModelSelection("timm", "resnet18", WeightMode.EXPLICIT, "imagenet"),
        cache_path=cache_path,
        registry=registry,
    )
    default = select_model_collection(
        ModelSelection("timm", "resnet18"),
        cache_path=cache_path,
        registry=registry,
    )
    none = select_model_collection(
        ModelSelection("timm", "resnet18", WeightMode.NONE),
        cache_path=cache_path,
        registry=registry,
    )

    assert explicit.weight_id == "imagenet"
    assert default.weight_id == "default"
    assert none.weight_id is None
