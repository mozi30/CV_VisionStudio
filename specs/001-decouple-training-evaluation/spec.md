# Feature Specification: Decouple Training Evaluation

**Feature Branch**: `[001-decouple-training-evaluation]`

**Created**: 2026-05-26

**Status**: Draft

**Input**: User description: "The trainer setup must change. Aktuell gibt es eine Trainer classe dessen hauptaufgabe es ist modele zu trainieren und evaluieren. Ich würde gerne training und evaluation entkoppeln. Somit soll der Trainer nur fürs training verantwortlich sein und er bekommt für die evaluation nach jeder EPoche einen Evaluator der die Aufgabe übernimmt. Der Evaluator funktioniert auch standalone wenn man ein schon trainiertes model validieren will. Es soll wie auch jetzt schon eine BAseclass geben und eine klasse die davon derived wird. DIe neue Derived class verwendet Wandb, dies kann umgeschalten werden zu localem training mit lifeplot. Somit kann Wandb verwendet werden muss aber nicht. Das selbe gilt für den Evaluator"

## Clarifications

### Session 2026-05-26

- Q: What should happen when an evaluator is missing during training? → A: Warn, continue training, and log only loss.
- Q: How much validation should happen before long-running training starts? → A: Validate as much as possible before the run to avoid late failures after hours of training.
- Q: What checkpoint retention and early stopping behavior is required? → A: Save the best three checkpoints plus the last checkpoint by default, with checkpoint count and early stopping configurable in trainer settings.
- Q: How should invalid models, wrong metrics, unsupported tasks, reporting failures, local plot failures, and interruption be handled? → A: Broken models, wrong metrics, unsupported tasks, and failed remote reporting throw exceptions; local plot failures warn and continue with logging only; interrupted training stops.
- Q: Where should the full validation loop live after removing validation from the trainer? → A: Create a separate validation service that uses evaluators, leaving both trainer and evaluator smaller.
- Q: How should loss-only training be represented when no evaluator is configured? → A: No evaluator means no validation at all; training logs only training loss and validation history stays empty.
- Q: Which metric ranks best checkpoints by default? → A: Users must configure the monitor metric explicitly before best checkpointing works.
- Q: Which task types must the first decoupled evaluation implementation support? → A: Support classification, detection, and loss-only in the first implementation.
- Q: How should remote, local live plot, and logging-only reporting be structured? → A: Create a shared reporter interface with WandB, local live plot, and logging-only implementations used by both trainer and validation service.
- Q: How should pre-run validation treat training and validation loaders? → A: Make dry-run batch validation optional through trainer settings; default to no batch consumption.
- Q: Where should the Evaluator be supplied for training-time validation? → A: Support both a constructor default Evaluator and a train/fit override per run.
- Q: How should remaining validation-service terminology and checkpoint defaults be resolved? → A: Replace validation-service terminology with Evaluator, rename current evaluator metric classes to EvaluationMetrics, keep checkpoint saving enabled by default when a valid existing path is configured, warn and disable checkpoint saving only when no path is configured, use Dataset objects for training/evaluation data, and use pytest under tests/unit or tests/integration with mocks enabled by base implementations.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train With Delegated Evaluation (Priority: P1)

As a model developer, I want the training workflow to focus only on training and delegate validation after each epoch to an Evaluator, so that training and evaluation loop responsibilities are clearly separated while preserving epoch-by-epoch feedback.

**Why this priority**: This is the core requested change and enables cleaner ownership, easier testing, and independent evolution of training and evaluation behavior.

**Independent Test**: Can be fully tested by running a training session with an Evaluator configured and verifying that training progresses while validation is performed by the Evaluator after each epoch.

**Acceptance Scenarios**:

1. **Given** a training workflow configured with an Evaluator and evaluation Dataset, **When** one epoch finishes, **Then** the Evaluator performs evaluation and returns evaluation results to the training workflow.
2. **Given** a training workflow configured with multiple epochs, **When** the workflow completes, **Then** evaluation has occurred once after each completed epoch and the trainer has not directly performed validation calculations.
3. **Given** a training workflow configured without an evaluator, **When** training starts or an epoch finishes, **Then** training continues using training loss only, emits a warning, logs only training loss, and leaves validation history empty.
4. **Given** a training workflow configured with checkpointing and early stopping settings including an explicit monitor metric, **When** training progresses across epochs, **Then** the best checkpoints, the latest checkpoint, and early stopping decisions follow those settings.

---

### User Story 2 - Validate a Pretrained Model Standalone (Priority: P2)

As a model developer, I want to run the Evaluator without starting a training session, so that I can validate an already trained model on a chosen evaluation Dataset.

**Why this priority**: Standalone evaluation is explicitly requested and makes validation reusable outside the training lifecycle.

**Independent Test**: Can be fully tested by loading an already trained model, running the Evaluator with an evaluation Dataset, and verifying that evaluation results are produced without invoking training.

**Acceptance Scenarios**:

1. **Given** an already trained model and evaluation Dataset, **When** standalone evaluation is started, **Then** the Evaluator returns evaluation metrics without performing any training updates.
2. **Given** an Evaluator receives invalid, broken, or incompatible model input, **When** standalone evaluation is started, **Then** it raises an exception with a clear diagnostic and no partial result is presented as successful.

---

### User Story 3 - Choose Experiment Reporting Mode (Priority: P3)

As a model developer, I want both training and evaluation to support either remote experiment reporting or local live plotting, so that I can work with remote tracking when available and still run locally without that service.

**Why this priority**: Optional reporting support keeps existing experiment workflows available while avoiding a hard dependency on remote tracking.

**Independent Test**: Can be fully tested by running training and standalone evaluation in each reporting mode and verifying that metrics are visible through the selected reporting destination.

**Acceptance Scenarios**:

1. **Given** remote reporting is selected and available, **When** training and evaluation produce metrics, **Then** the selected metrics are sent to the remote reporting destination.
2. **Given** local reporting is selected, **When** training and evaluation produce metrics, **Then** the selected metrics are visible through local live plots and no remote reporting account is required.
3. **Given** remote reporting is selected but unavailable, **When** training or evaluation starts, **Then** the run fails with an exception before long-running work begins whenever the failure can be detected up front.
4. **Given** local live plotting is selected but plotting output is unavailable, **When** training or evaluation produces metrics, **Then** the system emits a warning, continues the run, and records metrics through logging only.

### Edge Cases

- Evaluation is requested after an epoch but no Evaluator is provided.
- Evaluation Dataset is missing, empty, or incompatible with the model task.
- The standalone Evaluator is given a broken model, a model that has not been trained, or a model that cannot produce predictions for the evaluation Dataset.
- Remote reporting credentials or connectivity are unavailable while remote reporting is selected.
- Local live plotting cannot open a display or write its local output.
- EvaluationMetrics differ by task type, and unsupported task metrics are requested.
- Training is interrupted before an epoch completes, so training stops and no after-epoch evaluation is recorded for that incomplete epoch.
- Checkpoint storage cannot save one of the retained best checkpoints or the latest checkpoint.
- Best checkpointing is enabled but no monitor metric is configured.
- Checkpoint saving is enabled by default but no checkpoint path is configured.
- A checkpoint path is configured but does not already exist.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST separate training responsibility from evaluation responsibility so that the trainer no longer performs validation metric calculation directly.
- **FR-002**: The system MUST allow a Trainer to receive an Evaluator for after-epoch evaluation.
- **FR-003**: The system MUST invoke the provided Evaluator after each successfully completed epoch when evaluation is configured.
- **FR-004**: The Evaluator MUST own the full evaluation loop and be runnable independently for validating an already trained model without starting a training process.
- **FR-005**: The training component MUST expose a base abstraction and at least one concrete training implementation.
- **FR-006**: The evaluation component MUST expose a base abstraction and at least one concrete evaluation implementation.
- **FR-007**: Concrete training and evaluation workflows MUST support remote experiment reporting as an optional mode.
- **FR-008**: Concrete training and evaluation workflows MUST support local live metric visualization as an optional mode.
- **FR-009**: Users MUST be able to choose reporting mode for training independently from reporting mode for standalone evaluation.
- **FR-010**: Training results MUST include enough information to identify per-epoch training outcomes and any Evaluator-produced evaluation outcomes.
- **FR-011**: Standalone evaluation results MUST include enough information to identify the evaluated model, evaluation Dataset context, selected EvaluationMetrics, and evaluation outcome.
- **FR-012**: The system MUST preserve a migration path for existing users who currently run training with evaluation, so the new workflow can reproduce equivalent evaluation feedback when an Evaluator is supplied.
- **FR-013**: If no evaluator is configured for training, the trainer MUST continue training, emit a warning, log only training loss, perform no validation, and leave validation history empty.
- **FR-014**: The Trainer MUST perform all pre-run checks that can reasonably be completed before the run starts without consuming data batches by default, including Evaluator presence, model compatibility where determinable, evaluation Dataset availability, metric-task compatibility, reporting mode availability, and checkpoint configuration.
- **FR-015**: The Trainer MUST save the latest checkpoint by default when checkpoint saving is enabled and a configured checkpoint path exists, and MUST save the best three checkpoints only when the user configures an explicit monitor metric for best checkpoint ranking.
- **FR-016**: Users MUST be able to configure checkpoint retention settings and early stopping behavior through trainer settings.
- **FR-017**: If training is interrupted, the trainer MUST stop the run and avoid recording evaluation for incomplete epochs.
- **FR-018**: The current metric-aggregation evaluator concept MUST be renamed to EvaluationMetrics or a similar name so the Evaluator can own the full evaluation loop.
- **FR-019**: The trainer base MUST NOT expose validation or test responsibilities after decoupling; it may call the configured Evaluator when after-epoch validation is configured.
- **FR-020**: The system MUST NOT automatically create a loss-only evaluator when the user omits an evaluator.
- **FR-021**: If best checkpointing is requested without an explicit monitor metric, the trainer MUST fail pre-run validation with a clear configuration error.
- **FR-022**: The first implementation MUST support classification evaluation, detection evaluation, and explicit loss-only evaluation.
- **FR-023**: Classification, detection, and loss-only EvaluationMetrics MUST each define their required prediction and target inputs so unsupported or mismatched inputs fail with clear exceptions.
- **FR-024**: The system MUST provide a shared reporter interface with remote experiment reporting, local live plotting, and logging-only implementations usable by both the Trainer and the Evaluator.
- **FR-025**: Trainer and Evaluator reporting MUST use the shared reporter interface instead of duplicating remote or local plotting behavior inside each workflow.
- **FR-026**: Users MUST be able to enable optional dry-run batch validation through trainer settings; when disabled by default, pre-run validation MUST NOT consume training or validation loader batches.
- **FR-027**: The Trainer MUST support both a default Evaluator supplied during construction and an Evaluator override supplied to train/fit for a specific run.
- **FR-028**: The concrete Trainer implementation MUST have a general name that does not imply WandB-only behavior, because it can run with WandB, local live plotting, or logging-only reporting.
- **FR-029**: If checkpoint path is set, the Trainer MUST verify that the path exists before training; if it does not exist, training MUST fail before the run starts.
- **FR-030**: If checkpoint path is not set, training MUST continue and emit a warning that no checkpoints will be saved.
- **FR-031**: Training and evaluation inputs MUST use Dataset objects rather than unstructured data inputs.
- **FR-032**: Automated tests MUST use pytest and be placed under tests/unit or tests/integration according to their scope.
- **FR-033**: Tests SHOULD use mocks or lightweight doubles created from the available base implementations for Trainer, Evaluator, EvaluationMetrics, Dataset, and Reporter behavior.

### Security Requirements *(include if feature touches trust boundaries or sensitive data)*

- **SEC-001**: Remote reporting credentials and access tokens MUST NOT be printed in logs, plots, errors, or result summaries.
- **SEC-002**: When remote reporting is disabled, training and evaluation MUST NOT transmit experiment metrics or model metadata to remote reporting destinations.

### Dependency Requirements *(include if dependencies change)*

- **DEP-001**: Any reporting or plotting dependency used by shared reporter implementations MUST be optional for users who do not select that reporting mode, must have an acceptable license, and must have a documented removal or replacement path.

### Error Handling Requirements *(mandatory)*

- **ERR-001**: If after-epoch evaluation cannot run, the system MUST report a clear reason and avoid marking missing or failed evaluation results as successful.
- **ERR-002**: If standalone evaluation receives invalid model, Dataset, or EvaluationMetrics configuration, the Evaluator MUST fail safely with a clear diagnostic that identifies the invalid input category.
- **ERR-003**: If remote reporting is selected and unavailable, training or evaluation MUST raise an exception, preferably before long-running work begins.
- **ERR-004**: If local live plotting is selected and plotting is unavailable, training or evaluation MUST emit a warning, continue the run, and log metrics without live plots.
- **ERR-005**: If the Evaluator receives a broken model or EvaluationMetrics receive a wrong metric or unsupported task, evaluation MUST raise an exception with a clear diagnostic.
- **ERR-006**: If training is interrupted, the trainer MUST stop gracefully and avoid presenting the interrupted epoch as completed.
- **ERR-007**: If best checkpointing is enabled without a monitor metric, the trainer MUST raise a clear pre-run configuration exception.
- **ERR-008**: If a configured checkpoint path does not exist, the trainer MUST raise a clear pre-run configuration exception.

### AI and Release Requirements *(mandatory if AI assists or feature is releasable)*

- **AI-001**: Any AI-assisted changes to training or evaluation behavior MUST be human-reviewed for correctness, responsibility separation, and metric validity before release.
- **REL-001**: Release readiness MUST include pytest tests under tests/unit or tests/integration for delegated after-epoch evaluation, standalone evaluation, classification evaluation, detection evaluation, explicit loss-only evaluation, reporting mode selection, missing Evaluator behavior, invalid Dataset data, broken model handling, wrong metrics, unsupported tasks, checkpoint path behavior, checkpoint retention, early stopping configuration, unavailable reporting destination handling, local plot fallback, pre-run checks, and training interruption.

### Non-Functional Requirements *(mandatory)*

- **NFR-001**: The separation of responsibilities MUST improve maintainability by allowing trainer and evaluator behavior to be tested independently.
- **NFR-002**: After-epoch evaluation MUST not change the training update behavior for the epoch that just completed.
- **NFR-003**: Optional remote reporting MUST not be required for local training or standalone evaluation.
- **NFR-004**: Result reporting MUST be reproducible enough for users to compare training-time validation and standalone validation for the same model and data.
- **NFR-005**: Pre-run validation MUST catch configuration and availability problems that can be detected before training starts to reduce the risk of late failures during long-running training.
- **NFR-006**: Default pre-run validation MUST avoid data-loader side effects by not consuming batches unless dry-run batch validation is explicitly enabled.

### Key Entities *(include if feature involves data)*

- **Trainer**: Responsible for model training progress, epoch lifecycle, training outcomes, and delegation to an Evaluator when validation is configured.
- **Evaluator**: Responsible for the full evaluation loop over a model and validation data, during training-time or standalone validation.
- **EvaluationMetrics**: Responsible for metric aggregation and computation from predictions, targets, and loss supplied by the Evaluator.
- **Model**: The trainable or already trained artifact being optimized or validated.
- **Evaluation Dataset**: The Dataset object used by the Evaluator to measure model performance.
- **Evaluation Result**: The metrics and context produced by an evaluation run.
- **Reporting Mode**: The selected destination for training and evaluation metrics, either remote experiment reporting or local live visualization.
- **Reporter**: Shared interface used by the Trainer and Evaluator to publish metrics through remote reporting, local live plotting, or logging-only behavior.
- **Checkpoint**: A saved model state retained because it is among the best configured results or because it represents the latest completed training state.
- **Trainer Settings**: User-controlled training options including checkpoint retention and early stopping behavior.
- **Dry-Run Batch Validation Setting**: User-controlled option that allows pre-run validation to consume a sample batch for model, loss, evaluator, and metric compatibility checks.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of configured training runs with evaluation produce one Evaluator-generated evaluation result for each successfully completed epoch.
- **SC-002**: A trained model can be validated through standalone evaluation without invoking training in 100% of supported validation scenarios.
- **SC-003**: Existing training workflows that require epoch evaluation can reproduce equivalent evaluation metrics after migrating by supplying an Evaluator.
- **SC-004**: Users can run both training and standalone evaluation without remote reporting credentials when local reporting is selected.
- **SC-005**: At least 90% of responsibility-specific tests can exercise trainer behavior without requiring validation metric calculations and evaluator behavior without requiring model training.
- **SC-006**: 100% of training runs without an evaluator continue with a warning, produce training-loss-only logs, and produce no validation history entries.
- **SC-007**: 100% of detected broken model, wrong metric, unsupported task, and unavailable remote reporting cases fail with exceptions before successful completion is reported.
- **SC-008**: 100% of completed training runs retain the latest checkpoint by default and retain the configured number of best checkpoints only when an explicit monitor metric is configured, with a default best-checkpoint retention count of three after configuration.
- **SC-009**: 100% of training runs with no checkpoint path configured continue with a warning and save no checkpoints, while 100% of runs with a non-existing configured checkpoint path fail before training starts.
- **SC-010**: 100% of automated tests for this feature are discoverable by pytest under tests/unit or tests/integration.

## Supplemental Governance Checklist *(mandatory)*

| Gate | Applicability | Planned Evidence |
|------|---------------|------------------|
| Security reviewed | Yes, remote reporting may handle credentials and experiment metadata | Review secret handling and disabled-remote-reporting behavior |
| Dependencies justified | Yes, optional reporting and plotting capabilities may depend on external packages | Dependency review for optionality, license, version policy, and removal path |
| Errors specified and tested | Yes, evaluator, data, and reporting failures are core feature paths | Tests for missing evaluator, invalid validation data, invalid model, and unavailable reporting |
| AI output reviewed | Yes, specification and later implementation may be AI-assisted | Human review of generated artifacts and behavioral tests |
| Release gates passed | Yes, this is a releasable workflow refactor | Test, lint, documentation, migration notes, and owner approval evidence |

## Assumptions

- Existing users are model developers or experiment authors who already configure training, Dataset objects, and metrics.
- The first implementation supports classification, detection, and explicit loss-only evaluation; other task types are out of scope unless added by a later feature.
- Epoch-based training remains the primary training lifecycle for this feature.
- Remote experiment reporting refers to the currently used remote tracking workflow, and local live visualization refers to the currently used local plotting workflow.
- The initial migration goal is behavioral parity for existing evaluation feedback, not new metric definitions.
- If no evaluator is provided, the trainer runs training-only workflows based on loss, warns the user, does not create a loss-only evaluator automatically, and must not silently pretend that validation occurred.
