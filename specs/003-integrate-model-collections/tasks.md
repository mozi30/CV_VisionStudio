# Tasks: Integrate Model Collections

**Input**: Design documents from `specs/003-integrate-model-collections/`

**Prerequisites**: [`plan.md`](plan.md), [`spec.md`](spec.md), [`research.md`](research.md), [`data-model.md`](data-model.md), [`contracts/`](contracts/), [`quickstart.md`](quickstart.md)

**Governance**: Implementation tasks include security, dependency, error-handling, AI-review, and release-gate work required by the supplemental constitution.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the model-collection package and test structure without changing existing model behavior.

- [X] T001 Create model collection package files in src/vision_studio/models/collections/__init__.py, src/vision_studio/models/collections/base.py, src/vision_studio/models/collections/types.py, src/vision_studio/models/collections/errors.py, src/vision_studio/models/collections/registry.py, src/vision_studio/models/collections/cache.py, src/vision_studio/models/collections/timm.py, src/vision_studio/models/collections/mmpretrain.py, and src/vision_studio/models/collections/torchvision.py
- [X] T002 Create model collection test package files in tests/unit/models/collections/__init__.py and tests/integration/models/collections/__init__.py
- [X] T003 [P] Add dependency notes for timm/MMPreTrain optional collection extras and torchvision required-elsewhere behavior in pyproject.toml
- [X] T004 [P] Add model collection documentation placeholder for installation, refresh, and selection examples in specs/003-integrate-model-collections/quickstart.md
- [X] T005 [P] Add release evidence checklist placeholders for dependency review, security review, tests, lint, docs, and owner acceptance in specs/003-integrate-model-collections/quickstart.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Define shared types, errors, adapter contract, registry, and dependency boundaries that all stories require.

**Critical**: No user story implementation should begin until this phase is complete.

- [X] T006 Define CollectionId, WeightMode, ModelCollection, ModelOption, WeightOption, ModelSelection, ModelSelectionResult, and CollectionMetadataCache dataclasses/enums in src/vision_studio/models/collections/types.py
- [X] T007 Define InvalidCollectionError, MissingBackendError, InvalidModelError, AmbiguousModelError, InvalidWeightError, WeightRetrievalError, WeightIntegrityError, UnconfirmedWeightError, and MetadataCacheError in src/vision_studio/models/collections/errors.py
- [X] T008 Define ModelCollectionAdapter abstract interface with lazy backend import hooks, list_models, list_weights, and create_model methods in src/vision_studio/models/collections/base.py
- [X] T009 Implement safe backend import helper and backend version detection in src/vision_studio/models/collections/base.py
- [X] T010 Implement adapter registry with approved collection registration, lookup, ambiguous model-name detection, and missing-backend-safe diagnostics in src/vision_studio/models/collections/registry.py
- [X] T011 Implement selection validation helpers for collection/model/weight identifiers, default/no-weight mode validation, source metadata, and dataset/task non-validation guardrails in src/vision_studio/models/collections/registry.py
- [X] T012 Implement warning and confirmation helper for unverifiable approved-collection weights in src/vision_studio/models/collections/registry.py
- [X] T013 [P] Add public collection package exports for types, errors, registry helpers, and adapters in src/vision_studio/models/collections/__init__.py
- [X] T014 [P] Add top-level model package exports for collection selection and listing helpers in src/vision_studio/models/__init__.py

**Checkpoint**: Foundation ready; user story implementation can now begin.

---

## Phase 3: User Story 1 - Choose baseline model and weights (Priority: P1) MVP

**Goal**: A practitioner can select a supported collection model with explicit weights, default pretrained weights, or no pretrained weights, and receive an initialized model plus source metadata.

**Independent Test**: Select a collection, model, and weight behavior and confirm the returned model/result reflects explicit, default, and no-weight modes without invoking training.

### Tests for User Story 1

- [X] T015 [P] [US1] Add contract tests for explicit, default, and no-pretrained-weight selection modes in tests/unit/models/collections/test_model_selection_contract.py
- [X] T016 [P] [US1] Add error-path tests for invalid collection, invalid model, ambiguous model name, invalid weight, and incompatible weight alternatives in tests/unit/models/collections/test_model_selection_errors.py
- [X] T017 [P] [US1] Add dependency/security tests for missing selected backend, unverifiable weight confirmation, failed integrity validation, and safe warning text in tests/unit/models/collections/test_dependency_and_security.py
- [X] T018 [P] [US1] Add runtime weight-retrieval failure test that expects WeightRetrievalError and verifies no successful model load is reported in tests/unit/models/collections/test_weight_retrieval_errors.py
- [X] T019 [P] [US1] Add adapter unit tests with lightweight fake backend modules for timm, MMPreTrain, and torchvision model creation in tests/unit/models/collections/test_backend_adapters.py
- [X] T020 [P] [US1] Add integration test for selecting explicit/default/no pretrained weights through the public SDK helpers in tests/integration/models/collections/test_model_selection_workflow.py

### Implementation for User Story 1

- [X] T021 [P] [US1] Implement timm adapter list/create behavior using lazy timm list_models and create_model calls in src/vision_studio/models/collections/timm.py
- [X] T022 [P] [US1] Implement MMPreTrain adapter list/create behavior using lazy mmpretrain list_models and get_model calls in src/vision_studio/models/collections/mmpretrain.py
- [X] T023 [P] [US1] Implement torchvision adapter list/create behavior using lazy torchvision model and weight helper calls in src/vision_studio/models/collections/torchvision.py
- [X] T024 [US1] Implement select_model_collection entrypoint that validates ModelSelection, resolves default/explicit/no-weight behavior, and returns ModelSelectionResult in src/vision_studio/models/collections/registry.py
- [X] T025 [US1] Implement source metadata propagation for selected collection, model, and weight in src/vision_studio/models/collections/registry.py
- [X] T026 [US1] Implement approved-collection unverifiable weight warning and confirmation flow plus failed-integrity blocking in src/vision_studio/models/collections/registry.py
- [X] T027 [US1] Implement runtime pretrained weight retrieval failure handling that raises WeightRetrievalError and prevents successful-load reporting in src/vision_studio/models/collections/registry.py
- [X] T028 [US1] Wire registered timm, MMPreTrain, and torchvision adapters into package initialization without importing backend packages eagerly in src/vision_studio/models/collections/__init__.py
- [X] T029 [US1] Add SDK usage examples for explicit, default, and no-pretrained-weight selection in specs/003-integrate-model-collections/quickstart.md

**Checkpoint**: User Story 1 is fully functional and testable independently as the MVP.

---

## Phase 4: User Story 2 - Discover available models and weights (Priority: P2)

**Goal**: A practitioner can list cached model and weight options for supported collections and explicitly refresh metadata when backend catalogs change.

**Independent Test**: Request cached listings, refresh metadata from installed backends, and verify listing results include model identifiers, weight variants, collection source, and refresh metadata.

### Tests for User Story 2

- [X] T030 [P] [US2] Add metadata cache contract tests for schema version, generated_at, backend_versions, models, weights, default markers, and safe source fields in tests/unit/models/collections/test_metadata_cache_contract.py
- [X] T031 [P] [US2] Add cache error tests for missing cache, corrupt cache, stale cache visibility, and refresh suggestion diagnostics in tests/unit/models/collections/test_metadata_cache_errors.py
- [X] T032 [P] [US2] Add refresh tests for installed backend refresh, missing optional backend skip, per-backend refresh failure reporting, and retained last valid cache behavior in tests/unit/models/collections/test_metadata_refresh.py
- [X] T033 [P] [US2] Add performance-oriented cached-listing test that verifies normal listing does not import live backend catalogs in tests/unit/models/collections/test_cached_listing_performance.py
- [X] T034 [P] [US2] Add integration test for refresh then cached listing through public SDK helpers in tests/integration/models/collections/test_metadata_listing_workflow.py

### Implementation for User Story 2

- [X] T035 [P] [US2] Implement JSON cache read/write, schema validation, corrupt-cache diagnostics, and stale timestamp handling in src/vision_studio/models/collections/cache.py
- [X] T036 [US2] Implement list_available_models cached listing helper that returns ModelOption and WeightOption records without importing backend catalogs in src/vision_studio/models/collections/registry.py
- [X] T037 [US2] Implement refresh_collection_metadata helper that inspects installed adapters, records backend versions, skips missing optional backends, and writes CollectionMetadataCache in src/vision_studio/models/collections/cache.py
- [X] T038 [US2] Implement cache refresh orchestration and per-backend failure reporting in src/vision_studio/models/collections/registry.py
- [X] T039 [US2] Add CLI entrypoints for listing cached models and refreshing metadata in src/vision_studio/main.py
- [X] T040 [US2] Add user-facing examples for cached listing and explicit refresh in specs/003-integrate-model-collections/quickstart.md

**Checkpoint**: User Stories 1 and 2 both work independently.

---

## Phase 5: User Story 3 - Keep custom model flow intact (Priority: P3)

**Goal**: Existing custom model definitions continue to initialize and operate without requiring collection backends or changing the current workflow.

**Independent Test**: Define or instantiate a custom model exactly as before and verify no collection backend import or configuration is required.

### Tests for User Story 3

- [X] T041 [P] [US3] Add custom ImageClassifier compatibility test proving existing construction, config, forward, and postprocess behavior remain unchanged in tests/unit/models/test_custom_model_compatibility.py
- [X] T042 [P] [US3] Add import isolation test proving vision_studio.models and custom model imports do not import timm or MMPreTrain when collection APIs are unused in tests/unit/models/collections/test_lazy_import_isolation.py
- [X] T043 [P] [US3] Add integration test proving existing training/evaluation smoke flow can instantiate a custom model without collection metadata cache or optional backends in tests/integration/models/collections/test_custom_model_flow_intact.py

### Implementation for User Story 3

- [X] T044 [US3] Refine src/vision_studio/models/__init__.py exports so collection helpers are available while BaseModel, InputSpec, OutputSpec, and ImageClassifier remain backward compatible
- [X] T045 [US3] Ensure collection package imports do not import timm, MMPreTrain, or torchvision adapters until collection selection or refresh is invoked in src/vision_studio/models/collections/__init__.py
- [X] T046 [US3] Add missing-backend-safe import and startup behavior documentation in specs/003-integrate-model-collections/quickstart.md

**Checkpoint**: All user stories are independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final verification, documentation, dependency review, and release readiness across all user stories.

- [ ] T047 [P] Update dependency review evidence for timm/MMPreTrain optional collection backends and torchvision required-elsewhere behavior, including version ranges, licenses, maintenance status, vulnerability status, installation guidance, and removal strategy in specs/003-integrate-model-collections/quickstart.md
- [ ] T048 [P] Update security review evidence for approved collections, unverifiable weight warnings, failed integrity blocking, and safe diagnostics in specs/003-integrate-model-collections/quickstart.md
- [X] T049 [P] Update user documentation examples for string config/CLI identifiers and SDK enum-friendly constants in specs/003-integrate-model-collections/quickstart.md
- [X] T050 Run feature-scope tests with python -m pytest tests/unit/models tests/integration/models and record results in specs/003-integrate-model-collections/quickstart.md
- [X] T051 Run full test suite with python -m pytest tests/unit tests/integration and record results in specs/003-integrate-model-collections/quickstart.md
- [ ] T052 Run lint and formatting checks with python -m ruff check src tests and python -m black --check src tests and record results in specs/003-integrate-model-collections/quickstart.md
- [ ] T053 Verify model listing performance target and custom model initialization regression target, then record evidence in specs/003-integrate-model-collections/quickstart.md
- [ ] T054 Complete AI-generated artifact review and human owner acceptance evidence in specs/003-integrate-model-collections/quickstart.md
- [ ] T055 Compare refreshed cache model counts against installed backend list_models counts and record 90% exposure validation evidence in specs/003-integrate-model-collections/quickstart.md
- [X] T056 Update README.md with baseline model collection functionality, selection modes, metadata refresh, optional backend guidance, and runtime weight-retrieval exception behavior

---

## Dependencies & Execution Order

### Phase Dependencies

- Phase 1 Setup has no dependencies and can start immediately.
- Phase 2 Foundational depends on Phase 1 and blocks all user stories.
- Phase 3 User Story 1 depends on Phase 2 and is the MVP.
- Phase 4 User Story 2 depends on Phase 2; it can run after or beside US1, but selection demos are richer once US1 exists.
- Phase 5 User Story 3 depends on Phase 2; it can run beside US1 and US2.
- Phase 6 Polish depends on whichever user stories are selected for release.

### User Story Dependencies

- US1 Choose baseline model and weights: independent after foundation.
- US2 Discover available models and weights: independent after foundation, uses the shared adapter and cache contracts.
- US3 Keep custom model flow intact: independent after foundation, validates backward compatibility.

### Dependency Graph

```text
Setup -> Foundation -> US1 -> Polish
                  |-> US2 -> Polish
                  |-> US3 -> Polish
```

## Parallel Opportunities

- Setup tasks T003, T004, and T005 can run in parallel after T001 and T002.
- Foundational exports T013 and T014 can run in parallel after shared types/errors are defined.
- US1 tests T015-T020 can run in parallel, then adapters T021-T023 can run in parallel.
- US2 tests T030-T034 can run in parallel, then cache implementation T035 can proceed beside registry listing T036 after shared cache types exist.
- US3 tests T041-T043 can run in parallel with US1/US2 once the foundation is complete.
- Polish evidence tasks T047-T049 can run in parallel before final verification tasks T050-T056.

## Parallel Example: User Story 1

```bash
Task: "T015 [US1] Add contract tests in tests/unit/models/collections/test_model_selection_contract.py"
Task: "T016 [US1] Add error-path tests in tests/unit/models/collections/test_model_selection_errors.py"
Task: "T017 [US1] Add dependency/security tests in tests/unit/models/collections/test_dependency_and_security.py"
Task: "T018 [US1] Add runtime retrieval failure tests in tests/unit/models/collections/test_weight_retrieval_errors.py"
Task: "T019 [US1] Add adapter tests in tests/unit/models/collections/test_backend_adapters.py"
Task: "T020 [US1] Add integration workflow test in tests/integration/models/collections/test_model_selection_workflow.py"
```

## Parallel Example: User Story 2

```bash
Task: "T030 [US2] Add metadata cache contract tests in tests/unit/models/collections/test_metadata_cache_contract.py"
Task: "T031 [US2] Add cache error tests in tests/unit/models/collections/test_metadata_cache_errors.py"
Task: "T032 [US2] Add refresh tests in tests/unit/models/collections/test_metadata_refresh.py"
Task: "T033 [US2] Add cached-listing performance test in tests/unit/models/collections/test_cached_listing_performance.py"
Task: "T034 [US2] Add integration workflow test in tests/integration/models/collections/test_metadata_listing_workflow.py"
```

## Parallel Example: User Story 3

```bash
Task: "T041 [US3] Add custom model compatibility test in tests/unit/models/test_custom_model_compatibility.py"
Task: "T042 [US3] Add import isolation test in tests/unit/models/collections/test_lazy_import_isolation.py"
Task: "T043 [US3] Add custom flow integration test in tests/integration/models/collections/test_custom_model_flow_intact.py"
```

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 Setup.
2. Complete Phase 2 Foundational.
3. Complete Phase 3 User Story 1.
4. Stop and validate explicit/default/no-weight selection independently.
5. Demo baseline model initialization without training workflow changes.

### Incremental Delivery

1. Foundation ready.
2. Add US1 for baseline model selection.
3. Add US2 for cache-backed discovery and refresh.
4. Add US3 for backward-compatibility proof.
5. Complete polish, reviews, and release-gate evidence.

### Test-First Order

1. Write user-story tests before implementing each story.
2. Implement shared types/errors/adapters before public registry behavior.
3. Implement adapters and selection before CLI/listing examples.
4. Run feature-scope tests after each story.
5. Run full suite and lint/format checks before owner acceptance.
