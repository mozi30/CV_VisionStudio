# Tasks: Ensemble Dataset Evaluation

**Input**: Design documents from `specs/004-ensemble-evaluation/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/ensemble-evaluation-contract.md, quickstart.md

**Governance**: Implementation tasks include required error-handling, AI-review, and release-gate work from the supplemental constitution. No new dependency or security-sensitive behavior is planned.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it touches a different file or does not depend on incomplete implementation
- **[Story]**: User story label for story phases only
- All tasks include exact file paths

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare shared test locations and confirm existing evaluator/inference contracts before user story implementation.

- [X] T001 Inspect existing `LoopEvaluator.evaluate()` and `SimpleInference.run_ensemble()` behavior in `src/vision_studio/evaluate/evaluator.py` and `src/vision_studio/inference/simple.py`
- [X] T002 [P] Create the integration test package path for ensemble evaluator workflow in `tests/integration/evaluator/__init__.py`
- [X] T003 [P] Create the unit test module scaffold for ensemble evaluator behavior in `tests/unit/evaluator/test_ensemble_evaluator.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared aggregation and validation support that MUST be complete before user stories can be implemented safely.

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T004 Add reusable ensemble evaluation configuration import or alias strategy using existing `EnsembleConfig` in `src/vision_studio/evaluate/evaluator.py`
- [X] T005 Add private tensor target movement helper for evaluator batches in `src/vision_studio/evaluate/evaluator.py`
- [X] T006 Add private model state capture and restoration helpers for one or more models in `src/vision_studio/evaluate/evaluator.py`
- [X] T007 Add private ensemble configuration validation for empty model lists, unsupported modes, unsupported policies, and weighted mode requirements in `src/vision_studio/evaluate/evaluator.py`
- [X] T008 Add private classification-style aggregation helper aligned with soft, hard, weighted, tie-policy, class-count, and shape behavior from `src/vision_studio/inference/simple.py`
- [X] T009 Add private failed-model bookkeeping helper that de-duplicates and sorts failed model indexes in `src/vision_studio/evaluate/evaluator.py`

**Checkpoint**: Foundation ready - user story implementation can now begin.

---

## Phase 3: User Story 1 - Evaluate an Ensemble on a Dataset (Priority: P1) MVP

**Goal**: Users can evaluate multiple compatible classification models over a dataset and receive one consolidated metric result based on aggregated ensemble predictions.

**Independent Test**: Provide two compatible classification models, an evaluation dataset, and classification metrics; run ensemble evaluation; verify consolidated metrics, default soft voting, weighted aggregation, and mean successful per-model loss.

### Tests for User Story 1

- [X] T010 [P] [US1] Add unit test for default soft-voting ensemble dataset evaluation metrics and metadata in `tests/unit/evaluator/test_ensemble_evaluator.py`
- [X] T011 [P] [US1] Add unit test for weighted ensemble dataset evaluation and weight validation in `tests/unit/evaluator/test_ensemble_evaluator.py`
- [X] T012 [P] [US1] Add unit test proving ensemble loss equals mean successful per-model loss per batch in `tests/unit/evaluator/test_ensemble_evaluator.py`
- [X] T013 [P] [US1] Add integration test for a full ensemble evaluation workflow over multiple dataset batches in `tests/integration/evaluator/test_ensemble_evaluation_workflow.py`

### Implementation for User Story 1

- [X] T014 [US1] Implement `LoopEvaluator.evaluate_ensemble(models, dataset, config=None)` public entry point in `src/vision_studio/evaluate/evaluator.py`
- [X] T015 [US1] Implement per-batch ensemble model execution, loss computation, postprocessing, and aggregation in `src/vision_studio/evaluate/evaluator.py`
- [X] T016 [US1] Update metrics with aggregated predictions, moved targets, and mean successful per-model loss in `src/vision_studio/evaluate/evaluator.py`
- [X] T017 [US1] Return flat result with metric keys plus `status`, `aggregation_metadata`, and `failed_models` in `src/vision_studio/evaluate/evaluator.py`
- [X] T018 [US1] Log final ensemble evaluation loss through the configured reporter in `src/vision_studio/evaluate/evaluator.py`
- [X] T019 [US1] Export any new public evaluator symbols if required by package imports in `src/vision_studio/evaluate/__init__.py`

**Checkpoint**: User Story 1 is fully functional and independently testable as the MVP.

---

## Phase 4: User Story 2 - Audit Ensemble Evaluation Behavior (Priority: P2)

**Goal**: Ensemble evaluation results expose aggregation behavior and model failure status so users can audit how metrics were produced.

**Independent Test**: Run ensemble evaluation with each supported aggregation mode and failure policy; verify metadata includes mode, tie policy, failure policy, model counts, successful model count, failed model indexes, and status.

### Tests for User Story 2

- [X] T020 [P] [US2] Add unit test for hard-voting tie policy metadata and labels in `tests/unit/evaluator/test_ensemble_evaluator.py`
- [X] T021 [P] [US2] Add unit test for `continue-with-warning` model failures with completed-with-warnings status in `tests/unit/evaluator/test_ensemble_evaluator.py`
- [X] T022 [P] [US2] Add unit test for all-model failure under `continue-with-warning` in `tests/unit/evaluator/test_ensemble_evaluator.py`
- [X] T023 [P] [US2] Add unit test for incompatible class counts or prediction shapes in `tests/unit/evaluator/test_ensemble_evaluator.py`

### Implementation for User Story 2

- [X] T024 [US2] Ensure hard-voting tie-policy aggregation returns metric-compatible predictions in `src/vision_studio/evaluate/evaluator.py`
- [X] T025 [US2] Implement `fail-fast` and `continue-with-warning` behavior for per-model batch failures in `src/vision_studio/evaluate/evaluator.py`
- [X] T026 [US2] Populate aggregation metadata with mode, tie policy, failure policy, failed models, model count, and successful model count in `src/vision_studio/evaluate/evaluator.py`
- [X] T027 [US2] Raise clear configuration errors for incompatible outputs and all-model failure in `src/vision_studio/evaluate/evaluator.py`
- [X] T028 [US2] Preserve `prepare_ensemble_handoff()` compatibility with the flat ensemble evaluation result in `src/vision_studio/evaluate/evaluator.py`

**Checkpoint**: User Story 2 audit and failure-policy behavior is independently testable.

---

## Phase 5: User Story 3 - Preserve Single-Model Evaluation Behavior (Priority: P3)

**Goal**: Existing single-model evaluation behavior remains unchanged while ensemble evaluation restores every model's original state after success or failure.

**Independent Test**: Run current single-model evaluator tests and new state-restoration tests; verify existing result shape and training-mode restoration are unchanged.

### Tests for User Story 3

- [X] T029 [P] [US3] Add regression test proving `LoopEvaluator.evaluate()` single-model result shape and metric updates remain unchanged in `tests/unit/evaluator/test_evaluator.py`
- [X] T030 [P] [US3] Add unit test proving ensemble evaluation restores all model training states after success in `tests/unit/evaluator/test_ensemble_evaluator.py`
- [X] T031 [P] [US3] Add unit test proving ensemble evaluation restores all model training states after failure in `tests/unit/evaluator/test_ensemble_evaluator.py`

### Implementation for User Story 3

- [X] T032 [US3] Refactor `LoopEvaluator.evaluate()` only as needed to share target movement without changing behavior in `src/vision_studio/evaluate/evaluator.py`
- [X] T033 [US3] Wrap ensemble evaluation state changes in cleanup logic that restores each model after success or failure in `src/vision_studio/evaluate/evaluator.py`
- [X] T034 [US3] Verify reporter start, log, and finish behavior remains consistent for single-model and ensemble evaluation in `src/vision_studio/evaluate/evaluator.py`

**Checkpoint**: Existing single-model behavior and new ensemble state cleanup are independently testable.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, examples, release checks, and review evidence across all stories.

- [X] T035 [P] Update ensemble evaluation usage notes in `specs/004-ensemble-evaluation/quickstart.md`
- [X] T036 [P] Add or update notebook/example usage for ensemble evaluation in `notebooks/ensamble.ipynb`
- [X] T037 [P] Add or update full pipeline example coverage for ensemble evaluation in `examples/collection_model_full_pipeline.py`
- [X] T038 Run focused evaluator tests with `pytest tests/unit/evaluator/test_evaluator.py tests/unit/evaluator/test_ensemble_evaluator.py`
- [X] T039 Run integration workflow test with `pytest tests/integration/evaluator/test_ensemble_evaluation_workflow.py`
- [X] T040 Run related ensemble inference regression tests with `pytest tests/unit/inference/test_ensemble_soft_voting.py tests/unit/inference/test_ensemble_hard_voting.py tests/unit/inference/test_ensemble_weighted.py tests/unit/inference/test_failure_policies.py`
- [X] T041 Run project linting and formatting checks according to repository configuration in `pyproject.toml`
- [X] T042 Record AI-review, dependency no-change review, error-path coverage, and human-owner acceptance notes in `specs/004-ensemble-evaluation/quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - blocks all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational completion - MVP
- **User Story 2 (Phase 4)**: Depends on Foundational completion; can be developed after or alongside US1, but final metadata behavior should be validated with US1 result shape
- **User Story 3 (Phase 5)**: Depends on Foundational completion; can be developed after or alongside US1
- **Polish (Phase 6)**: Depends on implemented user stories selected for release

### User Story Dependencies

- **US1 Evaluate an Ensemble on a Dataset**: No dependency on other user stories after foundation; recommended MVP
- **US2 Audit Ensemble Evaluation Behavior**: Independent after foundation, but benefits from US1's public entry point
- **US3 Preserve Single-Model Evaluation Behavior**: Independent after foundation, focused on regression and cleanup

### Within Each User Story

- Tests should be written before implementation tasks for the story
- Validation helpers should exist before public entry point implementation
- Error behavior should be implemented before claiming story completion
- Release evidence should be recorded only after test and review tasks pass

---

## Parallel Execution Examples

### User Story 1

```text
Task: T010 Add unit test for default soft-voting ensemble dataset evaluation metrics and metadata in tests/unit/evaluator/test_ensemble_evaluator.py
Task: T013 Add integration test for a full ensemble evaluation workflow over multiple dataset batches in tests/integration/evaluator/test_ensemble_evaluation_workflow.py
```

### User Story 2

```text
Task: T020 Add unit test for hard-voting tie policy metadata and labels in tests/unit/evaluator/test_ensemble_evaluator.py
Task: T023 Add unit test for incompatible class counts or prediction shapes in tests/unit/evaluator/test_ensemble_evaluator.py
```

### User Story 3

```text
Task: T029 Add regression test proving LoopEvaluator.evaluate() single-model result shape and metric updates remain unchanged in tests/unit/evaluator/test_evaluator.py
Task: T031 Add unit test proving ensemble evaluation restores all model training states after failure in tests/unit/evaluator/test_ensemble_evaluator.py
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational helpers.
3. Complete Phase 3 tests and implementation for `evaluate_ensemble()`.
4. Run focused unit and integration tests for US1.
5. Demonstrate dataset-level ensemble evaluation returning flat metrics and audit fields.

### Incremental Delivery

1. Deliver US1 for basic ensemble dataset evaluation.
2. Add US2 for audit metadata, hard-vote, failure policy, and incompatible-output coverage.
3. Add US3 for regression protection and robust state restoration.
4. Complete polish tasks and release gates.

### Parallel Team Strategy

After Phase 2, one developer can implement US1's public evaluation path, another can harden US2 failure-policy tests and metadata assertions, and another can focus on US3 regression/state cleanup. Merge after focused story tests pass.
