# Feature Specification: Integrate Model Collections

**Feature Branch**: `003-integrate-model-collections`

**Created**: June 1, 2026

**Status**: Draft

**Input**: User description: "new spec so 003. For this new feature intergration for common model collections should be done so next to defining you own model many baseline models with there corresponding pretrained weights can be used. For this timm MMPreTrain and torchvision should be able to used. Eather by string name or Enum what fits better to the current interface fo the collections fits better should be able to choose models and weifghts for these mdoels."

## Clarifications

### Session 2026-06-01

- Q: How should collection/model/weight identifiers be represented? → A: Support both: strings in config/CLI, enums in SDK.

### Session 2026-06-02

- Q: How should timm, MMPreTrain, and torchvision dependencies be required? → A: Each collection backend is optional and only required when that collection is selected.
- Q: How should model and weight listings be discovered? → A: Cache discovered model and weight metadata locally, with an explicit refresh operation.
- Q: How should pretrained weight selection behave during model selection? → A: Users can select explicit weights, use default pretrained weights when available, or choose no pretrained weights.
- Q: What compatibility validation belongs to model collection selection? → A: Validate only collection/model/weight compatibility; dataset and task compatibility are deferred to training or evaluation workflows.
- Q: How should unverifiable pretrained weights be handled? → A: Allow unverifiable weights only after warning the user before loading.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Choose baseline model and weights (Priority: P1)

A practitioner selects a baseline model from a supported collection and chooses pretrained weights to start an experiment quickly.

**Why this priority**: This is the primary user value: faster setup using trusted, prebuilt models.

**Independent Test**: Can be fully tested by selecting a collection, model, and weights and confirming the model is ready for use.

**Acceptance Scenarios**:

1. **Given** a supported collection is available, **When** the user selects a model and a weight variant, **Then** the system provides the selected model with the chosen weights applied.
2. **Given** a supported collection is available, **When** the user selects only a model and default pretrained weights are available, **Then** the system provides the selected model with the collection's default pretrained weights applied.
3. **Given** a model is selected with no pretrained weights, **When** the user confirms the selection, **Then** the system provides the model without weights and without errors.

---

### User Story 2 - Discover available models and weights (Priority: P2)

A practitioner needs to browse what models and weight variants are available in each collection before choosing.

**Why this priority**: Discoverability prevents trial-and-error and improves adoption of baseline collections.

**Independent Test**: Can be tested by requesting a list of available models and weights from cached metadata, refreshing the cache, and confirming updated metadata is used.

**Acceptance Scenarios**:

1. **Given** cached collection metadata exists, **When** the user requests available options, **Then** each model includes its available weight variants and collection source.
2. **Given** installed backend catalogs may have changed, **When** the user requests a metadata refresh, **Then** the system updates the local model and weight metadata used for future listings.

---

### User Story 3 - Keep custom model flow intact (Priority: P3)

A practitioner continues to use a custom model definition while optionally mixing in baseline collection models without changing their current workflow.

**Why this priority**: Backward compatibility protects existing users and reduces migration effort.

**Independent Test**: Can be tested by defining a custom model exactly as before and verifying no changes are required.

**Acceptance Scenarios**:

1. **Given** the user defines a custom model, **When** the system initializes it, **Then** it behaves as it did before the baseline collections were added.

---

### Edge Cases

- What happens when a model name exists in multiple collections?
- How does the system handle a request for weights that do not exist for the selected model?
- What happens if a weight artifact cannot be retrieved at request time?
- What happens if a weight artifact source cannot be verified or lacks integrity metadata?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support baseline model collections from timm, MMPreTrain, and torchvision as selectable sources.
- **FR-002**: Users MUST be able to specify collection/model/weight identifiers as strings in config/CLI, while SDK users MAY use enums for collections and weights.
- **FR-003**: System MUST provide a way to list available models for each supported collection along with their available weight variants.
- **FR-004**: System MUST allow users to select a specific pretrained weight variant, select only a model and use default pretrained weights when available, or explicitly choose no pretrained weights.
- **FR-005**: System MUST validate that a chosen weight variant is compatible with the selected model and collection.
- **FR-006**: System MUST require a collection selection when model names are ambiguous across collections.
- **FR-007**: System MUST preserve existing custom model definition behavior without breaking changes.
- **FR-008**: System MUST surface the collection source for a selected model and its weights in the user-facing selection output.
- **FR-009**: System MUST preserve startup, listing, and custom model behavior when optional collection backends that are not selected are not installed.
- **FR-010**: System MUST cache discovered model and weight metadata locally for fast listing.
- **FR-011**: System MUST provide an explicit refresh operation that rebuilds cached model and weight metadata from installed supported backends.
- **FR-012**: When a user selects only a model and the selected collection exposes default pretrained weights for that model, the system MUST apply those default pretrained weights.
- **FR-013**: When a user selects only a model and no default pretrained weights are available, the system MUST surface that no default weights exist and require the user to proceed without pretrained weights or select an explicit weight variant.
- **FR-014**: Model collection selection MUST NOT validate dataset compatibility, task compatibility, or training/evaluation suitability; those checks remain the responsibility of downstream training or evaluation workflows.

### Security Requirements *(include if feature touches trust boundaries or sensitive data)*

- **SEC-001**: System MUST only load pretrained weights from the approved collections and clearly indicate the source to the user.
- **SEC-002**: System MUST warn the user before loading approved-collection weights whose source cannot be verified or lacks integrity metadata.
- **SEC-003**: System MUST refuse to load weights when integrity validation is available and fails.

### Dependency Requirements *(include if dependencies change)*

- **DEP-001**: timm and MMPreTrain MUST be optional collection backends required only when selected; torchvision MAY remain a required project dependency for existing non-collection functionality, but torchvision baseline-model collection behavior MUST remain isolated behind the collection selection flow.
- **DEP-002**: Optional collection backend dependencies MUST be documented with version ranges, license compatibility, installation guidance, and a removal strategy if a dependency is deprecated.

### Error Handling Requirements *(mandatory)*

- **ERR-001**: When a collection, model, or weight is invalid or incompatible within the selected collection, the system MUST return a clear error message and list valid alternatives.
- **ERR-002**: When pretrained weights cannot be retrieved at runtime, the system MUST raise a clear weight-retrieval exception, surface the failure reason, and allow the user to retry or select different weights without reporting the model as successfully loaded.
- **ERR-003**: When a selected optional collection backend is not installed, the system MUST fail with a clear dependency error that identifies the missing backend and how to install or choose another collection.
- **ERR-004**: When unverifiable approved-collection weights are selected, the system MUST present a warning before loading and continue only if the user confirms.

### AI and Release Requirements *(mandatory if AI assists or feature is releasable)*

- **AI-001**: Any AI-assisted specification, planning, task generation, implementation, or review artifacts for this feature MUST be human-reviewed before release; if automated runtime recommendations are introduced for model or weight selection, they MUST be clearly labeled and reviewed before release.
- **REL-001**: Release MUST include tests that cover model selection, weight compatibility checks, runtime weight-retrieval exceptions, cache exposure validation against backend model listings, and failure modes, plus updated README/user documentation.

### Non-Functional Requirements *(mandatory)*

- **NFR-001**: Listing models and weights for a collection MUST complete in under 2 seconds for a typical local environment.
- **NFR-002**: Adding baseline collections MUST not increase the time to initialize an unchanged custom model by more than 5%.
- **NFR-003**: Cached model and weight metadata MUST be used for normal listing requests so discovery performance does not depend on live backend catalog inspection.

### Key Entities *(include if feature involves data)*

- **ModelCollection**: Represents a collection source and its available models.
- **ModelOption**: Represents a specific model identifier tied to a collection, including available weight variants.
- **WeightOption**: Represents a pretrained weight variant with model compatibility and source metadata.
- **Collection Metadata Cache**: Local metadata store containing discovered model, weight, compatibility, and source information for supported installed collection backends.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can select a baseline model and weights in three steps or fewer during a usability test.
- **SC-002**: 95% of model listing requests return results within 2 seconds.
- **SC-003**: 90% of users complete model selection on the first attempt without errors.
- **SC-004**: The release exposes at least 90% of models listed by each supported collection at the time of release.

## Supplemental Governance Checklist *(mandatory)*

| Gate | Applicability | Planned Evidence |
|------|---------------|------------------|
| Security reviewed | Yes - external artifacts are involved | Threat review and dependency audit notes |
| Dependencies justified | Yes - adds model collections | Dependency review with version and license notes |
| Errors specified and tested | Yes - selection and retrieval errors | Test plan covering invalid selections and retrieval failures |
| AI output reviewed | Yes - planning or implementation artifacts may be AI-assisted, while runtime AI recommendations are not in scope | Human review of AI-assisted artifacts before implementation or release |
| Release gates passed | Yes - user-facing change | Release checklist with tests and docs updated |

## Assumptions

- Users already have access to the existing model-collection interface in the product.
- Network access is available when pretrained weights need to be retrieved.
- Training workflows remain out of scope; this feature focuses on model selection and initialization.
- Licensing and redistribution policies for external weights are handled by existing project governance.
- Cached collection metadata may become stale until the user runs the explicit refresh operation.
