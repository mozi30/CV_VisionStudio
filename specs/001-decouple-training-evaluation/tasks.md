# Tasks: Decouple Training Evaluation

**Input**: Design documents from `specs/001-decouple-training-evaluation/`

**Prerequisites**: [`plan.md`](plan.md), [`spec.md`](spec.md), [`research.md`](research.md), [`data-model.md`](data-model.md), [`contracts/`](contracts/), [`quickstart.md`](quickstart.md)

**Governance**: Implementation tasks include security, dependency, error-handling, AI-review, and release-gate work required by the supplemental constitution.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create package and grouped test structure needed by all stories.

- [ ] T001 Create reporting package files in src/vision_studio/reporting/__init__.py, src/vision_studio/reporting/base.py, src/vision_studio/reporting/logging.py, src/vision_studio/reporting/live_plot.py, and src/vision_studio/reporting/wandb.py
- [ ] T002 Create evaluator and metrics module placeholders in src/vision_studio/evaluate/evaluator.py and src/vision_studio/evaluate/metrics.py
- [ ] T003 Create general concrete trainer module placeholder in src/vision_studio/trainer/trainer.py
- [ ] T004 [P] Create grouped pytest package structure in tests/unit/trainer/__init__.py, tests/unit/evaluator/__init__.py, tests/unit/reporting/__init__.py, and tests/integration/training/__init__.py
- [ ] T005 [P] Ensure pytest is declared or documented for test execution in pyproject.toml

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared settings, result types, exceptions, reporter contracts, Dataset-based boundaries, and base API constraints needed by all user stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T006 Define Trainer, Evaluator, EvaluationMetrics, Dataset, checkpoint, reporter, and result settings/types in src/vision_studio/types.py
- [ ] T007 [P] Define shared training, evaluation, metrics, checkpoint, Dataset, and reporter exception classes in src/vision_studio/types.py
- [ ] T008 Implement BaseReporter protocol or abstract base class in src/vision_studio/reporting/base.py
- [ ] T009 [P] Implement LoggingReporter in src/vision_studio/reporting/logging.py
- [ ] T010 Refactor Trainer base API to remove public validate/test responsibilities and keep train/fit/train_epoch/checkpoint lifecycle in src/vision_studio/trainer/base.py
- [ ] T011 Add pre-run configuration checks for Trainer settings without consuming Dataset batches in src/vision_studio/trainer/base.py
- [ ] T012 Add checkpoint path validation and no-path warning behavior in src/vision_studio/trainer/base.py
- [ ] T013 Add checkpoint retention helper for existing checkpoint path, latest checkpoints, and explicit-monitor best checkpoints in src/vision_studio/trainer/base.py
- [ ] T014 Add package exports for reporting, Evaluator, EvaluationMetrics, and general Trainer modules in src/vision_studio/reporting/__init__.py, src/vision_studio/evaluate/__init__.py, and src/vision_studio/trainer/__init__.py
- [x] T015 [P] Add foundational pytest unit tests for settings validation and exception classes in tests/unit/trainer/test_trainer_delegation.py
- [x] T016 [P] Add foundational pytest unit tests for LoggingReporter behavior in tests/unit/reporting/test_reporting_modes.py
- [x] T017 [P] Add Dataset-based test doubles or fixtures derived from base implementations in tests/unit/evaluator/test_evaluator.py
- [ ] T018 Document trust boundaries, secret logging constraints, dependency optionality, pytest usage, and AI-review expectations in specs/001-decouple-training-evaluation/tasks.md

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel.

---

## Phase 3: User Story 1 - Train With Delegated Evaluation (Priority: P1) 🎯 MVP

**Goal**: Trainer performs optimization only, delegates after-epoch evaluation to an Evaluator when configured, supports constructor default and fit override Evaluators, warns and keeps empty evaluation history when no Evaluator exists, and handles Dataset-based optional checkpointing and early stopping settings.

**Independent Test**: Run a training session with an Evaluator to verify one evaluation result per completed epoch, run with a fit override Evaluator to verify override precedence, and run without an Evaluator to verify warning, training-loss-only logs, no checkpoints when checkpoint path is unset, and empty evaluation history.

### Tests for User Story 1 ⚠️

- [x] T019 [P] [US1] Add Trainer contract tests for removed validate/test API and Evaluator delegation in tests/unit/trainer/test_trainer_delegation.py
- [x] T020 [P] [US1] Add constructor default Evaluator and fit override precedence tests in tests/unit/trainer/test_trainer_delegation.py
- [x] T021 [P] [US1] Add missing-Evaluator behavior tests for warning, training-loss-only logs, and empty evaluation history in tests/unit/trainer/test_trainer_delegation.py
- [x] T022 [P] [US1] Add checkpoint path unset warning, missing path exception, best-three retention, latest checkpoint, and early stopping tests in tests/unit/trainer/test_checkpoint_policy.py
- [x] T023 [P] [US1] Add training interruption behavior test that prevents incomplete-epoch evaluation in tests/unit/trainer/test_trainer_delegation.py
- [x] T024 [P] [US1] Add pytest integration test for two-epoch Trainer-to-Evaluator Dataset workflow in tests/integration/training/test_training_evaluation_workflow.py
- [x] T025 [P] [US1] Add migration parity test for equivalent evaluation feedback after supplying an Evaluator in tests/integration/training/test_training_evaluation_workflow.py

### Implementation for User Story 1

- [x] T026 [US1] Implement Trainer settings consumption, warnings, training-loss-only history, and no automatic LossEvaluationMetrics fallback in src/vision_studio/trainer/base.py
- [x] T027 [US1] Implement general concrete Trainer constructor and fit flow with optional constructor default Evaluator and per-run Evaluator override in src/vision_studio/trainer/trainer.py
- [x] T028 [US1] Migrate existing WandbTrainer behavior into general Trainer without WandB-only naming in src/vision_studio/trainer/trainer.py and reduce src/vision_studio/trainer/wandb_trainer.py to compatibility or deprecation behavior
- [x] T029 [US1] Remove trainer-owned evaluation loop and public validate/test methods from concrete trainer behavior in src/vision_studio/trainer/trainer.py and src/vision_studio/trainer/wandb_trainer.py
- [x] T030 [US1] Implement after-epoch Evaluator delegation and evaluation history collection in src/vision_studio/trainer/trainer.py
- [x] T031 [US1] Implement pre-run checks for checkpoint path existence, checkpoint monitor requirement, reporter availability, Evaluator availability, Dataset compatibility where determinable, and dry-run disabled default in src/vision_studio/trainer/base.py
- [x] T032 [US1] Implement no-checkpoint warning when checkpoint path is unset in src/vision_studio/trainer/base.py
- [x] T033 [US1] Implement latest checkpoint saving and explicit-monitor best checkpoint retention only when checkpoint path exists in src/vision_studio/trainer/base.py
- [ ] T034 [US1] Implement early stopping with explicit monitor metric and clear configuration errors in src/vision_studio/trainer/base.py
- [ ] T035 [US1] Implement graceful KeyboardInterrupt/interruption handling in src/vision_studio/trainer/trainer.py
- [x] T036 [US1] Update Trainer package exports in src/vision_studio/trainer/__init__.py
- [x] T037 [US1] Update Trainer contract implementation notes in specs/001-decouple-training-evaluation/contracts/trainer-contract.md

**Checkpoint**: User Story 1 is fully functional and testable independently as the MVP.

---

## Phase 4: User Story 2 - Validate a Pretrained Model Standalone (Priority: P2)

**Goal**: An Evaluator can validate a trained model without invoking training, using classification, detection, or explicit loss-only EvaluationMetrics on Dataset objects.

**Independent Test**: Load or create a trained model, run Evaluator with evaluation Dataset objects and each supported EvaluationMetrics type, and verify metrics are returned without optimizer updates.

### Tests for User Story 2 ⚠️

- [ ] T038 [P] [US2] Add Evaluator contract tests for standalone evaluation and no optimizer updates in tests/unit/evaluator/test_evaluator.py
- [ ] T039 [P] [US2] Add ClassificationEvaluationMetrics input validation and metric aggregation tests in tests/unit/evaluator/test_evaluation_metrics.py
- [ ] T040 [P] [US2] Add DetectionEvaluationMetrics input validation and metric aggregation tests in tests/unit/evaluator/test_evaluation_metrics.py
- [ ] T041 [P] [US2] Add explicit LossEvaluationMetrics tests and missing-loss error tests in tests/unit/evaluator/test_evaluation_metrics.py
- [ ] T042 [P] [US2] Add Evaluator error tests for broken model, empty evaluation Dataset, missing target fields, and missing prediction fields in tests/unit/evaluator/test_evaluator.py

### Implementation for User Story 2

- [ ] T043 [US2] Implement Evaluator with standalone and training-time evaluation entrypoints in src/vision_studio/evaluate/evaluator.py
- [ ] T044 [US2] Implement Evaluator Dataset batch movement, model inference, loss extraction, postprocess handling, and EvaluationMetrics update orchestration in src/vision_studio/evaluate/evaluator.py
- [ ] T045 [US2] Implement EvaluationMetrics base contract with task type, required inputs, reset, update, and compute behavior in src/vision_studio/evaluate/metrics.py
- [ ] T046 [US2] Move or adapt LossEvaluator into explicit LossEvaluationMetrics with clear missing-loss errors in src/vision_studio/evaluate/metrics.py and src/vision_studio/evaluate/base.py
- [ ] T047 [US2] Rename or adapt ClassificationEvaluator into ClassificationEvaluationMetrics with logits/probabilities, labels, target labels, and unsupported metric validation in src/vision_studio/evaluate/classication.py
- [ ] T048 [US2] Rename or adapt DetectionEvaluator into DetectionEvaluationMetrics with boxes/scores/labels validation and metric result contract in src/vision_studio/evaluate/detection.py
- [ ] T049 [US2] Update evaluate package exports for Evaluator, EvaluationMetrics, classification metrics, detection metrics, and loss metrics in src/vision_studio/evaluate/__init__.py
- [ ] T050 [US2] Update Evaluator and EvaluationMetrics contract notes in specs/001-decouple-training-evaluation/contracts/evaluator-contract.md and specs/001-decouple-training-evaluation/contracts/evaluation-metrics-contract.md

**Checkpoint**: User Story 2 is independently functional for standalone evaluation.

---

## Phase 5: User Story 3 - Choose Experiment Reporting Mode (Priority: P3)

**Goal**: Trainer and Evaluator use a shared reporter interface with WandB, local live plot, and logging-only implementations, including required failure behavior.

**Independent Test**: Run training and standalone evaluation in each reporting mode, verify metrics are visible through the selected reporter, verify WandB failures raise, and verify local plot failures warn and continue logging.

### Tests for User Story 3 ⚠️

- [ ] T051 [P] [US3] Add reporter contract tests for shared start/log/finish operations in tests/unit/reporting/test_reporting_modes.py
- [ ] T052 [P] [US3] Add WandB reporter failure tests for unavailable initialization and no secret leakage in tests/unit/reporting/test_reporting_modes.py
- [ ] T053 [P] [US3] Add local live-plot fallback tests for warning and logging-only continuation in tests/unit/reporting/test_reporting_modes.py
- [ ] T054 [P] [US3] Add Trainer and Evaluator reporter integration tests in tests/integration/training/test_training_evaluation_workflow.py

### Implementation for User Story 3

- [ ] T055 [US3] Implement WandbReporter using fatal initialization/logging failure semantics in src/vision_studio/reporting/wandb.py
- [ ] T056 [US3] Implement LivePlotReporter with warning fallback to logging-only behavior in src/vision_studio/reporting/live_plot.py
- [ ] T057 [US3] Integrate BaseReporter usage into Trainer metric logging in src/vision_studio/trainer/trainer.py
- [ ] T058 [US3] Integrate BaseReporter usage into Evaluator metric logging in src/vision_studio/evaluate/evaluator.py
- [ ] T059 [US3] Ensure local and logging-only reporters never require or transmit WandB credentials in src/vision_studio/reporting/logging.py and src/vision_studio/reporting/live_plot.py
- [ ] T060 [US3] Update reporting exports in src/vision_studio/reporting/__init__.py
- [ ] T061 [US3] Update reporting contract implementation notes in specs/001-decouple-training-evaluation/contracts/reporting-contract.md

**Checkpoint**: User Story 3 is independently functional for all reporting modes.

---

## Final Phase: Polish & Cross-Cutting Concerns

**Purpose**: Verification, migration documentation, release readiness, and cleanup across all stories.

- [ ] T062 [P] Update public usage examples for Trainer, Evaluator, EvaluationMetrics, Dataset, and reporters in specs/001-decouple-training-evaluation/quickstart.md
- [ ] T063 [P] Add migration notes for replacing trainer.validate/test usage with Evaluator and replacing old evaluator metric classes with EvaluationMetrics in specs/001-decouple-training-evaluation/quickstart.md
- [ ] T064 [P] Record dependency review evidence for pytest and optional reporting/plotting dependencies in specs/001-decouple-training-evaluation/quickstart.md
- [ ] T065 [P] Review src/vision_studio/types.py and docs for secret-safe error messages and no remote transmission in local/logging-only modes
- [x] T066 Run python -m pytest tests/unit tests/integration and record failures or success in specs/001-decouple-training-evaluation/quickstart.md
- [x] T067 Run python -m ruff check src tests and record failures or success in specs/001-decouple-training-evaluation/quickstart.md
- [x] T068 Run python -m black --check src tests and record failures or success in specs/001-decouple-training-evaluation/quickstart.md
- [ ] T069 Perform AI-generated artifact review for spec, plan, tasks, tests, and implementation in specs/001-decouple-training-evaluation/tasks.md
- [ ] T070 Confirm human-owner acceptance and known risks for release in specs/001-decouple-training-evaluation/tasks.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion - blocks all user stories.
- **User Stories (Phase 3+)**: Depend on Foundational completion.
- **Polish (Final Phase)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **User Story 1 (P1)**: Starts after Foundational. MVP scope. No dependency on US2 or US3 if an Evaluator test double and LoggingReporter foundation are available.
- **User Story 2 (P2)**: Starts after Foundational. Can proceed in parallel for Evaluator and EvaluationMetrics implementation, then replace US1 test doubles.
- **User Story 3 (P3)**: Starts after Foundational. Can proceed in parallel for reporter implementations, then integrates with US1 and US2 workflows.

### Within Each User Story

- Tests are listed before implementation and should be written first.
- Settings/exceptions/reporter abstractions precede Trainer, Evaluator, and EvaluationMetrics implementation.
- Evaluator precedes Trainer after-epoch integration for production behavior.
- Reporter interface precedes WandB/local live plot implementations.
- Error-path tests precede corresponding failure-handling implementation.

---

## Parallel Opportunities

- Setup tasks T004 and T005 can run in parallel after package paths are known.
- Foundational tasks T007, T009, T015, T016, and T017 can run in parallel with non-conflicting files after T006 and T008 are defined.
- US1 tests T019 through T025 can be written in parallel.
- US2 tests T038 through T042 can be written in parallel.
- US3 tests T051 through T054 can be written in parallel.
- EvaluationMetrics implementation tasks T046, T047, and T048 can proceed in parallel after T045.
- Reporter implementation tasks T055 and T056 can proceed in parallel after T008.
- Polish review tasks T062 through T065 can proceed in parallel.

---

## Parallel Example: User Story 1

```bash
# Parallel pytest test-authoring tasks for delegated training behavior:
Task: "T019 [P] [US1] Add Trainer contract tests in tests/unit/trainer/test_trainer_delegation.py"
Task: "T022 [P] [US1] Add checkpoint policy tests in tests/unit/trainer/test_checkpoint_policy.py"
Task: "T024 [P] [US1] Add integration workflow test in tests/integration/training/test_training_evaluation_workflow.py"
```

## Parallel Example: User Story 2

```bash
# Parallel EvaluationMetrics-specific tests:
Task: "T039 [P] [US2] Add ClassificationEvaluationMetrics tests in tests/unit/evaluator/test_evaluation_metrics.py"
Task: "T040 [P] [US2] Add DetectionEvaluationMetrics tests in tests/unit/evaluator/test_evaluation_metrics.py"
Task: "T042 [P] [US2] Add Evaluator error tests in tests/unit/evaluator/test_evaluator.py"
```

## Parallel Example: User Story 3

```bash
# Parallel reporter behavior tests:
Task: "T052 [P] [US3] Add WandB reporter failure tests in tests/unit/reporting/test_reporting_modes.py"
Task: "T053 [P] [US3] Add local live-plot fallback tests in tests/unit/reporting/test_reporting_modes.py"
Task: "T054 [P] [US3] Add reporter integration tests in tests/integration/training/test_training_evaluation_workflow.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundation.
3. Complete Phase 3 User Story 1.
4. Stop and validate delegated training, Evaluator override behavior, missing Evaluator behavior, Dataset-based pre-run checks, checkpoint path behavior, checkpoint monitor validation, latest checkpoint, best checkpoint retention, early stopping, and interruption.

### Incremental Delivery

1. Setup + Foundation: Package skeletons, settings, exceptions, reporter base, Trainer base cleanup, Dataset-aware test fixtures.
2. US1: Trainer delegates to Evaluator and handles checkpoints/early stopping/interruption.
3. US2: Standalone Evaluator and supported EvaluationMetrics types.
4. US3: Shared reporter implementations and integration.
5. Polish: Documentation, migration notes, dependency evidence, release gates, review, owner acceptance.

### Validation Gates

- Each story must pass its pytest unit tests and independent integration criteria before moving to release polish.
- Final acceptance requires pytest, ruff, black check, dependency/security review, AI-generated artifact review, and human-owner acceptance.

---

## Notes

- Tests are included because the specification explicitly requires release tests for functional behavior and error paths.
- `[P]` tasks touch different files or are independently authorable.
- Story labels map to the three prioritized user stories in [`spec.md`](spec.md).
- Avoid reintroducing trainer-owned evaluation loops or EvaluationMetrics-owned Dataset loops.
- Do not silently fall back from WandB reporter to local reporter.
- Do not create loss-only EvaluationMetrics automatically when Evaluator is omitted.
