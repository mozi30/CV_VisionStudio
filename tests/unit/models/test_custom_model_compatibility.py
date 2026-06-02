from __future__ import annotations

import torch

from vision_studio.models import ImageClassifier


def test_custom_image_classifier_behavior_stays_intact():
    model = ImageClassifier(in_channels=3, num_classes=2)
    inputs = torch.randn(4, 3, 8, 8)

    logits = model(inputs)
    output = model.postprocess(logits)

    assert logits.shape == (4, 2)
    assert output["labels"].shape == (4,)
    assert model.get_config() == {"in_channels": 3, "num_classes": 2}
