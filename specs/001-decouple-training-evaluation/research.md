# Research: Decouple Training Evaluation

## Decision: Trainer owns training loop and always trains from loss

**Rationale**: Training should be responsible only for optimization and epoch lifecycle. Loss is always used for training, regardless of whether evaluation is configured.

**Alternatives considered**:

- Keep validation logic inside trainer: rejected because it preserves coupling.
- Let Evaluator influence training loss calculation: rejected because Evaluator is for evaluation only.

## Decision: Evaluator owns the full evaluation loop

**Rationale**: The clarified target states the Evaluator is the evaluation service. It must run after epochs and standalone for already trained models. This removes the separate validation service concept.

**Alternatives considered**:

- Separate validation service plus evaluator metrics: rejected because the user clarified that the Evaluator should be the evaluation service.
- Put full loop in metric classes: rejected because metric classes should remain focused on aggregation.

## Decision: Existing evaluator metric accumulators become EvaluationMetrics

**Rationale**: Current classes such as classification evaluator aggregate metrics. To let Evaluator mean the full loop, those classes should be renamed conceptually to EvaluationMetrics.

**Alternatives considered**:

- Keep current naming: rejected because it conflicts with Evaluator as loop owner.
- Use Validator naming: rejected to keep user-selected Evaluator terminology.

## Decision: Dataset objects are the training and evaluation data abstraction

**Rationale**: The project already has Dataset objects and the user clarified they should be used. This keeps Trainer and Evaluator APIs aligned with existing project structure.

**Alternatives considered**:

- Accept arbitrary iterables only: rejected because it weakens alignment with project data abstractions.
- Introduce new data container types: rejected because Dataset objects already exist.

## Decision: Pytest tests live under grouped unit and integration folders

**Rationale**: The user clarified pytest should be used and planning artifacts should describe grouped test folders such as `tests/unit/evaluator`, `tests/unit/trainer`, and `tests/integration/training`. Base implementations make mocks and lightweight doubles practical.

**Alternatives considered**:

- Notebook-based validation only: rejected because release gates require repeatable tests.
- Colocated tests beside source files: rejected because the requested layout is grouped under `tests/unit/...` and `tests/integration/...`.

## Decision: Trainer accepts Evaluator both in constructor and fit override

**Rationale**: Constructor default supports stable Trainer configuration; fit/train override supports per-run evaluation choices without rebuilding the Trainer.

**Alternatives considered**:

- Constructor only: rejected because it is less flexible for repeated experiments.
- Fit only: rejected because it makes default training setup less convenient.

## Decision: Concrete Trainer receives general naming

**Rationale**: The current concrete trainer is named for WandB, but the new implementation can use WandB, local live plotting, or logging-only mode.

**Alternatives considered**:

- Keep WandbTrainer: rejected because WandB is optional.
- Create multiple trainer classes per reporter: rejected because reporting is separated through reporter implementations.

## Decision: Missing Evaluator means no evaluation

**Rationale**: Training still runs from model loss, emits a warning, logs only training loss, and leaves evaluation history empty.

**Alternatives considered**:

- Automatically create loss-only metrics: rejected because missing Evaluator should mean no evaluation.
- Raise exception for missing Evaluator: rejected because training-only workflows must still work.

## Decision: Checkpoint saving is default only when a valid path is configured

**Rationale**: Users should get checkpoint persistence by default when they provide a valid existing path. If no path is configured, training should continue with an explicit warning and no checkpoint saving.

**Alternatives considered**:

- Fail if checkpoint path is unset: rejected because training should continue.
- Silently skip checkpoints: rejected because users need visibility.
- Create missing directories automatically: rejected because the user required existing path validation.

## Decision: Shared reporter interface for Trainer and Evaluator

**Rationale**: Reporting should not be duplicated in Trainer and Evaluator. A shared reporter interface keeps WandB, local live plot, and logging-only behavior consistent.

**Alternatives considered**:

- Keep reporting inside each workflow class: rejected because it duplicates behavior.
- Use delivery and visualise modules directly: rejected because they do not define a metrics reporting contract.

## Decision: Dry-run batch validation is optional and disabled by default

**Rationale**: Pre-run validation should catch deterministic configuration problems without consuming Dataset batches by default. Users can opt into dry-run validation when stronger early compatibility checks are desired.

**Alternatives considered**:

- Always run one dry-run batch: rejected because loaders may be stateful.
- Never support dry-run validation: rejected because the user wants as much early validation as possible.
