from __future__ import annotations

from vision_studio.models import ImageClassifier


def test_custom_model_construction_does_not_require_collection_cache():
    model = ImageClassifier(in_channels=1, num_classes=3)

    assert model.in_channels == 1
    assert model.num_classes == 3
