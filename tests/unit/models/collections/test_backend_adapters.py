from __future__ import annotations

import sys
import types

from vision_studio.models.collections.mmpretrain import MMPreTrainAdapter
from vision_studio.models.collections.timm import TimmAdapter
from vision_studio.models.collections.torchvision import TorchVisionAdapter


def test_timm_adapter_lists_and_creates_with_fake_backend(monkeypatch):
    fake = types.SimpleNamespace()
    fake.list_models = lambda pretrained=False: ["resnet18"] if pretrained else ["resnet18"]
    fake.create_model = lambda model_id, pretrained=False: {
        "model_id": model_id,
        "pretrained": pretrained,
    }
    monkeypatch.setitem(sys.modules, "timm", fake)

    adapter = TimmAdapter()
    models = adapter.list_models()
    created = adapter.create_model(
        types.SimpleNamespace(model_id="resnet18", mode="default"),
        "default",
    )

    assert models[0].default_weight_id == "default"
    assert created == {"model_id": "resnet18", "pretrained": True}


def test_mmpretrain_adapter_lists_and_creates_with_fake_backend(monkeypatch):
    fake_apis = types.SimpleNamespace()
    fake_apis.list_models = lambda: ["resnet50"]
    fake_apis.get_model = lambda model_id, pretrained=False: {
        "model_id": model_id,
        "pretrained": pretrained,
    }
    monkeypatch.setitem(sys.modules, "mmpretrain", types.SimpleNamespace(apis=fake_apis))

    adapter = MMPreTrainAdapter()
    models = adapter.list_models()
    created = adapter.create_model(
        types.SimpleNamespace(model_id="resnet50", mode="default"),
        "default",
    )

    assert models[0].model_id == "resnet50"
    assert created == {"model_id": "resnet50", "pretrained": True}


def test_torchvision_adapter_lists_and_creates_with_fake_backend(monkeypatch):
    class FakeWeight:
        name = "IMAGENET1K_V1"
        url = "https://example.test/weight.pth"

    class FakeWeights:
        IMAGENET1K_V1 = FakeWeight
        DEFAULT = FakeWeight

        def __iter__(self):
            return iter([FakeWeight])

    fake_models = types.SimpleNamespace()
    fake_models.list_models = lambda: ["resnet18"]
    fake_models.get_model_weights = lambda model_id: FakeWeights()
    fake_models.get_model = lambda model_id, weights=None: {
        "model_id": model_id,
        "weights": weights,
    }
    monkeypatch.setitem(
        sys.modules,
        "torchvision",
        types.SimpleNamespace(models=fake_models),
    )

    adapter = TorchVisionAdapter()
    models = adapter.list_models()
    created = adapter.create_model(
        types.SimpleNamespace(model_id="resnet18", mode="default"),
        "IMAGENET1K_V1",
    )

    assert models[0].default_weight_id == "IMAGENET1K_V1"
    assert created["weights"] is FakeWeight
