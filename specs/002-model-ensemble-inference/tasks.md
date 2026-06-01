# Tasks: Unified Inference, Ensemble, and Realtime Webcam Runs

**Input**: Design documents from `specs/002-model-ensemble-inference/`

**Prerequisites**: [`plan.md`](plan.md), [`spec.md`](spec.md), [`research.md`](research.md), [`data-model.md`](data-model.md), [`contracts/`](contracts/), [`quickstart.md`](quickstart.md)

**Governance**: Implementation tasks include applicable security, dependency, error-handling, AI-review, and release-gate work required by the supplemental constitution.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare inference test layout and shared typing/config support.

- [x] T001 Create integration test package for inference in `tests/integration/inference/__init__.py`
- [x] T002 Create unit inference package initializer in `tests/unit/inference/__init__.py`
- [x] T003 [P] Add/update unified inference exports in `src/vision_studio/inference/__init__.py`
- [x] T004 [P] Add shared type aliases for aggregation/failure/tie/realtime settings in `src/vision_studio/types.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the unified inference foundation required by all stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T005 Refine inference base abstraction contract in `src/vision_studio/inference/base.py`
- [x] T006 Implement unified inference orchestration skeleton in `src/vision_studio/inference/simple.py`
- [x] T007 [P] Implement centralized run-configuration validation helpers in `src/vision_studio/inference/simple.py`
- [x] T008 [P] Add explicit error classes/messages for inference failures in `src/vision_studio/inference/base.py`
- [x] T009 Integrate reporter-strategy invocation points in unified orchestration in `src/vision_studio/inference/simple.py`
- [x] T010 [P] Add foundational unit tests for config and error contracts in `tests/unit/inference/test_inference_contract.py`
- [x] T011 Document trust-boundary and logging assumptions for this scope in `specs/002-model-ensemble-inference/quickstart.md`
- [x] T012 [P] Record dependency and release-gate evidence notes in `specs/002-model-ensemble-inference/research.md`

**Checkpoint**: Foundation complete; user-story work can proceed.

---

## Phase 3: User Story 1 - Run ensemble evaluation with soft voting (Priority: P1) 🎯 MVP

**Goal**: Deliver ensemble-capable inference/evaluation baseline with strict compatibility validation and soft-voting default.

**Independent Test**: Run inference/evaluation with 2+ compatible models and verify consolidated output; empty list and class-count mismatch fail before aggregation.

### Tests for User Story 1

- [x] T013 [P] [US1] Add empty-model-list rejection tests in `tests/unit/inference/test_ensemble_validation.py`
- [x] T014 [P] [US1] Add class-count-mismatch rejection tests in `tests/unit/inference/test_ensemble_validation.py`
- [x] T015 [P] [US1] Add default soft-voting behavior tests in `tests/unit/inference/test_ensemble_soft_voting.py`
- [x] T016 [P] [US1] Add integration test for multi-model soft-voting run in `tests/integration/inference/test_ensemble_soft_voting_workflow.py`

### Implementation for User Story 1

- [x] T017 [US1] Implement strict model-list and class-count validation in `src/vision_studio/inference/simple.py`
- [x] T018 [US1] Implement soft-voting aggregation path in `src/vision_studio/inference/simple.py`
- [x] T019 [US1] Implement consolidated ensemble result assembly in `src/vision_studio/inference/simple.py`
- [x] T020 [US1] Add aggregation metadata emission for mode/default policy details in `src/vision_studio/inference/simple.py`
- [x] T021 [US1] Align evaluator handoff path for ensemble evaluation outputs in `src/vision_studio/evaluate/evaluator.py`

**Checkpoint**: User Story 1 independently functional and testable.

---

## Phase 4: User Story 2 - Choose ensemble aggregation mode (Priority: P2)

**Goal**: Add hard-voting and weighted-averaging modes with deterministic/default policy behavior.

**Independent Test**: Run the same model list under soft/hard/weighted modes and verify distinct, policy-conformant behavior with explicit invalid-config failures.

### Tests for User Story 2

- [x] T022 [P] [US2] Add hard-voting majority/tie tests in `tests/unit/inference/test_ensemble_hard_voting.py`
- [x] T023 [P] [US2] Add default tie-policy and random-policy tests in `tests/unit/inference/test_ensemble_hard_voting.py`
- [x] T024 [P] [US2] Add weighted-averaging valid/invalid weight tests in `tests/unit/inference/test_ensemble_weighted.py`
- [x] T025 [P] [US2] Add integration test comparing aggregation modes in `tests/integration/inference/test_aggregation_modes_workflow.py`

### Implementation for User Story 2

- [x] T026 [US2] Implement hard-voting aggregation logic in `src/vision_studio/inference/simple.py`
- [x] T027 [US2] Implement weighted-averaging logic and weight validation in `src/vision_studio/inference/simple.py`
- [x] T028 [US2] Implement tie-policy defaulting and invalid-policy errors in `src/vision_studio/inference/simple.py`
- [x] T029 [US2] Extend output metadata with selected mode and policies in `src/vision_studio/inference/simple.py`
- [x] T030 [US2] Update ensemble contract wording for implemented mode semantics in `specs/002-model-ensemble-inference/contracts/ensemble-aggregation-contract.md`

**Checkpoint**: User Stories 1 and 2 independently functional and testable.

---

## Phase 5: User Story 3 - Streamlined unified inference workflow with reporter pattern (Priority: P3)

**Goal**: Ensure one unified inference implementation handles prediction-only defaults, image/video modes, and failure-policy behavior with reporter-mode invariance.

**Independent Test**: Verify single-model and ensemble runs use unified flow; fail-fast/continue-with-warning behaviors are correct; reporter mode changes output channel only.

### Tests for User Story 3

- [x] T031 [P] [US3] Add fail-fast policy tests in `tests/unit/inference/test_failure_policies.py`
- [x] T032 [P] [US3] Add continue-with-warning and at-least-one-success tests in `tests/unit/inference/test_failure_policies.py`
- [x] T033 [P] [US3] Add completed_with_warnings status/metadata tests in `tests/unit/inference/test_failure_policies.py`
- [x] T034 [P] [US3] Add prediction-only default (no metrics) tests in `tests/unit/inference/test_prediction_only_mode.py`
- [x] T035 [P] [US3] Add integration test for unified image/video-path flow in `tests/integration/inference/test_unified_image_video_workflow.py`

### Implementation for User Story 3

- [x] T036 [US3] Implement prediction-only default path without metric computation in `src/vision_studio/inference/simple.py`
- [x] T037 [US3] Implement model-failure policy handling in unified flow in `src/vision_studio/inference/simple.py`
- [x] T038 [US3] Implement completed_with_warnings status and failed-model metadata emission in `src/vision_studio/inference/simple.py`
- [x] T039 [US3] Ensure reporter strategy selection does not alter orchestration flow in `src/vision_studio/inference/simple.py` and `src/vision_studio/reporting/base.py`
- [x] T040 [US3] Adapt legacy split inference entrypoints to route through unified implementation in `src/vision_studio/inference/classification.py` and `src/vision_studio/inference/wandb.py`

**Checkpoint**: User Stories 1–3 independently functional and testable.

---

## Phase 6: User Story 4 - Realtime webcam inference with optional frame skipping and annotated recording (Priority: P4)

**Goal**: Add realtime webcam run mode with optional frame skipping and saved annotated output video.

**Independent Test**: Start webcam inference, verify continuous predictions; enable frame skipping and verify sampling; enable recording and verify annotated output video artifact.

### Tests for User Story 4

- [x] T041 [P] [US4] Add webcam source availability/error tests in `tests/unit/inference/test_webcam_source_handling.py`
- [x] T042 [P] [US4] Add frame-skipping behavior tests in `tests/unit/inference/test_webcam_frame_skipping.py`
- [x] T043 [P] [US4] Add annotated-output recording tests in `tests/unit/inference/test_webcam_recording.py`
- [x] T044 [P] [US4] Add integration test for realtime webcam loop in `tests/integration/inference/test_webcam_realtime_workflow.py`

### Implementation for User Story 4

- [x] T045 [US4] Implement webcam input-source mode in unified inference service in `src/vision_studio/inference/simple.py`
- [x] T046 [US4] Implement optional frame-skipping in realtime loop in `src/vision_studio/inference/simple.py`
- [x] T047 [US4] Implement annotated output-video recording toggle and artifact handling in `src/vision_studio/inference/simple.py`
- [x] T048 [US4] Implement webcam-open failure diagnostics in `src/vision_studio/inference/base.py`
- [x] T049 [US4] Update inference contract for realtime webcam guarantees in `specs/002-model-ensemble-inference/contracts/inference-contract.md`

**Checkpoint**: User Story 4 independently functional and testable.

---

## Final Phase: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, documentation, and release readiness.

- [x] T050 [P] Update quickstart validation steps/results with webcam scenarios in `specs/002-model-ensemble-inference/quickstart.md`
- [x] T051 [P] Update data model and contract wording for final terminology consistency in `specs/002-model-ensemble-inference/data-model.md` and `specs/002-model-ensemble-inference/contracts/reporting-contract.md`
- [x] T052 [P] Record AI-generated artifact review evidence in `specs/002-model-ensemble-inference/tasks.md`
- [x] T053 Run pytest suite and record results in `specs/002-model-ensemble-inference/quickstart.md`
- [x] T054 Run lint/format checks and record results in `specs/002-model-ensemble-inference/quickstart.md`
- [x] T055 Record release notes, rollback guidance, and owner acceptance evidence in `specs/002-model-ensemble-inference/quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies.
- **Phase 2 (Foundational)**: Depends on Phase 1; blocks all user stories.
- **Phase 3 (US1)**: Depends on Phase 2.
- **Phase 4 (US2)**: Depends on Phase 2 and reuses US1 aggregation baseline.
- **Phase 5 (US3)**: Depends on Phase 2 and integrates with US1/US2 behavior.
- **Phase 6 (US4)**: Depends on Phase 2 and reuses unified flow from US3.
- **Final Phase**: Depends on completion of desired user stories.

### User Story Dependencies

- **US1 (P1)**: Independent after foundational tasks.
- **US2 (P2)**: Independent after foundational tasks; can reuse US1 scaffolding.
- **US3 (P3)**: Independent after foundational tasks; validates architecture-wide orchestration behavior.
- **US4 (P4)**: Independent after foundational tasks; builds on unified inference path and stream handling.

### Parallel Opportunities

- Tasks marked [P] in Setup and Foundational phases are parallelizable.
- Test tasks marked [P] within each user story are parallelizable.
- Documentation and contract updates marked [P] in final phase are parallelizable.

---

## Parallel Example: User Story 4

```bash
Task: "T041 [P] [US4] Add webcam source availability/error tests in tests/unit/inference/test_webcam_source_handling.py"
Task: "T042 [P] [US4] Add frame-skipping behavior tests in tests/unit/inference/test_webcam_frame_skipping.py"
Task: "T043 [P] [US4] Add annotated-output recording tests in tests/unit/inference/test_webcam_recording.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 and Phase 2.
2. Complete Phase 3 (US1).
3. Validate US1 independently before expanding scope.

### Incremental Delivery

1. Deliver US1 ensemble baseline.
2. Add US2 aggregation modes.
3. Add US3 unified prediction-only and reporter-invariant behavior.
4. Add US4 realtime webcam features.
5. Finish final verification and release readiness.

---

## AI-Generated Artifact Review Evidence

- Human review completed for AI-assisted changes in inference orchestration, ensemble aggregation logic, webcam flow, and associated tests.
- Verification evidence recorded in [`quickstart.md`](quickstart.md) including UV pytest and lint/format command outcomes.
- Contract and data-model updates reviewed for consistency with implemented runtime behavior.
