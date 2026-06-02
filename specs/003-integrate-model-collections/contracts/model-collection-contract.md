# Contract: Model Collection Selection

## Purpose

Define externally visible behavior for selecting baseline models and pretrained weights from approved model collections.

## Supported Collections

1. timm
2. MMPreTrain
3. torchvision

## Identifier Behavior

1. Config and CLI callers may provide collection, model, and weight identifiers as strings.
2. SDK callers may use enum-friendly constants for approved collection names and weight modes.
3. A collection identifier is required when a model name is ambiguous across collections.
4. User-facing selection output must include collection source and selected weight source.

## Weight Selection Modes

1. Explicit weight: user selects a concrete weight variant for the selected model.
2. Default pretrained weight: user selects only a model and the collection exposes default pretrained weights.
3. No pretrained weight: user explicitly requests model initialization without pretrained weights.

## Required Behavior

1. Validate that the selected collection is approved.
2. Validate that the selected model exists in the selected collection.
3. Validate that explicit weight variants belong to the selected model and collection.
4. Apply default pretrained weights when model-only selection is used and default weights are available.
5. If model-only selection is used and no default weights exist, require the user to proceed without pretrained weights or select an explicit weight.
6. Preserve existing custom `BaseModel` construction behavior outside collection selection.
7. Do not validate dataset compatibility, task compatibility, or training/evaluation suitability in this feature.

## Failure Behavior

- Invalid collection, model, or weight identifiers fail with a clear message and valid alternatives when available.
- Ambiguous model names without collection fail before initialization.
- Missing selected backend fails with dependency installation guidance.
- Failed model creation fails without reporting a successful selection.

## Compatibility

- Existing custom model definitions remain unchanged.
- Training, evaluation, and inference workflows receive ordinary model objects plus source metadata; they remain responsible for downstream task/data checks.
