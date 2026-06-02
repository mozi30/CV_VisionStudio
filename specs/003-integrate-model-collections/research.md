# Research: Integrate Model Collections

## Adapter Architecture

Decision: Implement a shared `ModelCollectionAdapter` interface with concrete adapters for timm, MMPreTrain, and torchvision.

Rationale: The three backends expose different discovery and model-construction APIs. A local adapter interface keeps backend differences out of user-facing selection, cache refresh, error handling, and tests.

Alternatives considered: Direct conditionals in one factory function were rejected because optional dependencies and backend-specific metadata would make one function hard to test. A plugin system was rejected for the first implementation because only three approved collections are in scope.

## Backend API Usage

Decision: Use each backend's official model listing and construction entry points inside its adapter: timm `list_models`/`create_model`, torchvision model listing and weight helpers/enums, and MMPreTrain `list_models`/`get_model`.

Rationale: The official APIs provide the most stable and reviewable path for catalog discovery and model creation. Adapters should translate those APIs into Vision Studio metadata and construction outputs without exposing backend-specific call signatures directly to users.

Alternatives considered: Scraping documentation/model zoo pages was rejected because it is brittle and disconnected from installed package versions. Shipping only a static manifest was rejected because it would hide models available in installed backend versions until a Vision Studio release.

Sources:

- timm models reference: https://huggingface.co/docs/timm/en/reference/models
- torchvision models and pretrained weights: https://docs.pytorch.org/vision/master/models.html
- MMPreTrain APIs: https://mmpretrain.readthedocs.io/en/latest/api/apis.html

## Optional Dependencies and Lazy Imports

Decision: Treat timm and MMPreTrain as optional/lazy collection backends. Keep torchvision baseline-model collection behavior isolated behind collection selection or refresh, while allowing torchvision to remain a required dependency for existing non-collection project functionality.

Rationale: Missing unselected optional backends must not break startup, model listing from available cache data, or existing custom model flows. Lazy imports also make tests for missing timm/MMPreTrain backends deterministic. torchvision is already used elsewhere in the project, so its dependency status can be required outside the model-collection feature while its baseline-model selection behavior remains isolated.

Alternatives considered: Requiring timm and MMPreTrain for all users was rejected because it increases install weight and conflicts with the spec. Treating torchvision as fully optional was rejected because the project may need it for existing non-collection functionality. Letting torchvision collection behavior load eagerly was rejected because collection functionality should not affect custom model flows.

## Metadata Cache

Decision: Store discovered model and weight metadata in a local JSON-compatible cache and use it for normal listing requests. Provide an explicit refresh operation that inspects installed supported backends and rebuilds cache records.

Rationale: Cached listing supports the 2-second listing goal and avoids import/catalog-inspection overhead during normal user selection. Explicit refresh gives users control when backend versions change.

Alternatives considered: Live backend inspection on every listing was rejected for performance and missing-backend fragility. A release-only static manifest was rejected because it would not reflect the user's installed backend versions.

## Identifier Strategy

Decision: Accept string identifiers for config/CLI and expose enum-friendly SDK values for collection names and known weight-selection modes.

Rationale: Strings preserve compatibility with config files and command usage while enum-friendly SDK values reduce typo risk for common constants. Backend model names are too dynamic for static enums to be the only representation.

Alternatives considered: Enums-only identifiers were rejected because backend catalogs change by installed version. Strings-only identifiers were rejected because SDK users benefit from explicit constants for collections and selection modes.

## Weight Selection Semantics

Decision: Support three weight modes: explicit weight identifier, default pretrained weights when available, and no pretrained weights.

Rationale: This matches user intent while keeping behavior explicit and testable. If default weights are unavailable, users must choose no pretrained weights or an explicit variant instead of receiving silent behavior.

Alternatives considered: Always requiring explicit weight identifiers was rejected because model-only selection should use default pretrained weights when available. Falling back silently from default weights to no weights was rejected because it can change experiment baselines without notice.

## Compatibility Scope

Decision: Validate only collection/model/weight compatibility in the model collection feature.

Rationale: Dataset compatibility, task compatibility, and training/evaluation suitability require downstream context and should remain in training/evaluation validation. This feature should return a selected/initialized model and source metadata, not assert end-to-end training readiness.

Alternatives considered: Validating dataset/task compatibility during selection was rejected because model selection may happen before dataset or evaluator configuration exists.

## Weight Trust and Integrity

Decision: Load weights only from approved collections, warn and require user confirmation for approved-collection weights that are unverifiable, and fail when available integrity validation fails.

Rationale: The spec accepts unverifiable approved-collection weights with warning/confirmation, while still requiring fatal behavior for known failed integrity checks. This makes risk acceptance explicit rather than silent.

Alternatives considered: Rejecting all unverifiable weights was rejected by clarification. Trusting URLs without warning was rejected because it hides a trust boundary from users.

## Runtime Weight Retrieval Failure

Decision: Treat pretrained weight retrieval failure as a runtime exception that surfaces the failure reason and prevents successful model-load reporting.

Rationale: Weight artifacts may fail to download or load only when the backend attempts retrieval. A typed exception keeps failure behavior explicit and testable while allowing users to retry or choose different weights.

Alternatives considered: Warning and falling back to no pretrained weights was rejected because it silently changes experiment baselines. Swallowing backend retrieval errors was rejected because it would violate error-handling requirements.

## Release Cache Exposure Validation

Decision: At release validation, compare refreshed cache model counts against each installed backend's model listing and require at least 90% exposure.

Rationale: The spec requires broad model exposure at release. Comparing cache counts against backend listings gives measurable evidence that the refresh/cache layer is not dropping most available models.

Alternatives considered: Relying only on unit tests for individual adapters was rejected because it does not validate catalog coverage. Requiring 100% exposure was rejected because backend APIs may expose aliases, unsupported variants, or entries that cannot be safely surfaced.
