# Contract: Dependency and Weight Security

## Purpose

Define dependency isolation, safe diagnostics, and pretrained weight trust behavior for approved model collections.

## Optional Backend Behavior

1. Collection adapters lazy-import backend packages when they are not already required by other project functionality.
2. timm and MMPreTrain are optional collection backends required only when selected.
3. torchvision may remain installed or required for existing non-collection project functionality, but torchvision baseline-model collection behavior remains isolated behind collection selection or refresh.
4. Unselected missing optional backends do not affect package import, cached listing, custom model construction, training, evaluation, or inference.
5. Selecting a missing optional backend fails with a clear dependency error.
6. Dependency errors identify the missing backend and describe how to install it or choose another collection.

## Approved Source Behavior

1. Only timm, MMPreTrain, and torchvision pretrained weights are approved for this feature.
2. Selection output must identify collection source and weight source before loading.
3. Backend-provided trusted metadata, hashes, or checksums are used when available.
4. Failed available integrity validation blocks loading.
5. Approved-collection weights that cannot be verified or lack integrity metadata require a warning and explicit confirmation before loading.

## User-Facing Warning Behavior

Warnings for unverifiable approved-collection weights must include:

1. The selected collection.
2. The selected model.
3. The selected weight identifier.
4. The reason verification is unavailable.
5. The fact that loading proceeds only after confirmation.

Warnings must not include secrets, credentials, private dataset paths, or full internal stack traces.

## Error Categories

- `InvalidCollectionError`: Collection is unsupported.
- `MissingBackendError`: Selected backend package is not installed.
- `InvalidModelError`: Model identifier is absent from selected collection.
- `AmbiguousModelError`: Model identifier exists in multiple collections and collection is missing.
- `InvalidWeightError`: Weight identifier is absent or incompatible for selected model.
- `WeightRetrievalError`: Backend cannot retrieve selected pretrained weights.
- `WeightIntegrityError`: Available integrity validation fails.
- `UnconfirmedWeightError`: Unverifiable weights were selected without confirmation.
- `MetadataCacheError`: Cache is missing, corrupt, or incompatible.

## Dependency Review Requirements

Before release, each optional backend must have documented:

1. Version range.
2. License compatibility.
3. Maintenance status.
4. Vulnerability/dependency scan result or waiver.
5. Installation guidance.
6. Removal or replacement strategy.
