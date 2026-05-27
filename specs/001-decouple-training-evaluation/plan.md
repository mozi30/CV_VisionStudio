# Implementation Plan: Decouple Training Evaluation

**Branch**: `[001-add-multi-model-ensamble]` | **Date**: 2026-05-26 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-decouple-training-evaluation/spec.md`

## Summary

Refactor training and evaluation so the `Trainer` owns the training loop and always trains from loss, while the `Evaluator` owns the full evaluation loop for both after-epoch evaluation and standalone evaluation. Existing metric accumulator classes currently named as evaluators will become `EvaluationMetrics`. Training and evaluation operate on project `Dataset` objects. A general concrete trainer implementation replaces WandB-specific naming because WandB is optional. Checkpointing saves by default when a configured checkpoint path exists; no checkpoint path warns and disables checkpoint saving, while a configured non-existing path fails before training starts.

## Technical Context

**Language/Version**: Python >=3.10, compatible with the active Python 3.12 development environment.

**Primary Dependencies**: PyTorch, torchvision, tqdm, matplotlib, seaborn, wandb, pytest, standard-library logging/warnings/pathlib/dataclasses; no required new runtime dependency planned except ensuring pytest is available for tests.

**Storage**: Local filesystem checkpoint files when an existing checkpoint path is configured. Remote experiment metadata only when WandB reporting is explicitly selected.

**Testing**: pytest tests under grouped folders such as `tests/unit/evaluator`, `tests/unit/trainer`, and `tests/integration/training`; release gates should run pytest, ruff, black check, and practical import/type smoke checks.

**Target Platform**: Linux development environment and Python package/CLI consumers.

**Project Type**: Python computer-vision library/package with CLI entry point and notebook usage.

**Performance Goals**: Default pre-run checks must not consume Dataset batches; optional dry-run batch validation may add one configured sample pass. Training without Evaluator must avoid evaluation iteration entirely. After-epoch evaluation must add no extra training-forward passes beyond Evaluator evaluation passes.

**Constraints**: Trainer base exposes train/fit behavior only and no public validation/test API. Evaluator owns evaluation loop. EvaluationMetrics aggregate metrics only. Remote reporting failure is fatal; local live-plot failure is warning-only. Best checkpointing requires explicit monitor metric and configured existing checkpoint path. Dry-run batch validation is optional and disabled by default. Tests should use mocks or lightweight doubles from base implementations where useful.

**Scale/Scope**: First implementation supports classification, detection, and explicit loss-only EvaluationMetrics. Other task types are out of scope. Existing Dataset/model/dataloader conventions are preserved where possible.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Security Constitution**: PASS. Trust boundary is WandB remote reporting. Credentials and tokens must remain external to source control and must not be logged. Local/logging-only modes must not transmit data remotely. Reporting exceptions must not expose secrets.

**Dependency Management Constitution**: PASS. The plan uses existing declared dependencies for PyTorch training, WandB reporting, and matplotlib-based local plotting. Pytest is used for test execution and must be declared or documented as a development test dependency. Optional reporting implementations isolate dependency usage behind selected reporter mode.

**Error Handling Constitution**: PASS. Required failures are explicit: missing Evaluator warning with no evaluation; broken model, wrong metric, unsupported task, invalid Dataset data, failed remote reporting, missing best-checkpoint monitor, non-existing checkpoint path, and checkpoint save failure raise exceptions; absent checkpoint path warns and continues without checkpoints; local plot failure warns and continues logging; interruption stops gracefully.

**AI Usage Constitution**: PASS. AI-generated plan and later implementation must be reviewed. External facts about dependency behavior must be verified. Pytest tests must be written before implementation in the Spec Kit workflow.

**Release Gates Constitution**: PASS. Release requires pytest tests for trainer-only training, after-epoch Evaluator delegation, standalone Evaluator validation, classification/detection/loss-only EvaluationMetrics, shared reporting modes, missing Evaluator, dry-run validation setting, Dataset inputs, checkpoint path validation, checkpoint monitor validation, checkpoint retention, early stopping, local plot fallback, remote failure, invalid inputs, and interruption.

**Unified Acceptance Rule**: PASS. The planned work can satisfy Security reviewed → Dependencies justified → Errors specified and tested → AI output reviewed → Release gates passed → Human owner accepted.

## Project Structure

### Documentation (this feature)

```text
specs/001-decouple-training-evaluation/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── evaluator-contract.md
│   ├── evaluation-metrics-contract.md
│   ├── reporting-contract.md
│   └── trainer-contract.md
└── tasks.md
```

### Source Code (repository root)

```text
src/vision_studio/
├── trainer/
│   ├── __init__.py
│   ├── base.py
│   └── trainer.py
├── evaluate/
│   ├── __init__.py
│   ├── evaluator.py
│   ├── metrics.py
│   ├── classication.py
│   ├── detection.py
│   └── utils.py
├── reporting/
│   ├── __init__.py
│   ├── base.py
│   ├── logging.py
│   ├── live_plot.py
│   └── wandb.py
├── dataset/
│   └── base.py
├── visualise/
│   └── utils.py
└── types.py

tests/
├── unit/
│   ├── evaluator/
│   │   ├── test_evaluator.py
│   │   └── test_evaluation_metrics.py
│   ├── trainer/
│   │   ├── test_trainer_delegation.py
│   │   └── test_checkpoint_policy.py
│   └── reporting/
│       └── test_reporting_modes.py
└── integration/
    └── training/
        └── test_training_evaluation_workflow.py
```

**Structure Decision**: Keep the existing single Python package structure under `src/vision_studio`. Do not add a separate validation package: the `Evaluator` is the evaluation service. Add a focused `reporting` package because reporting must be shared by Trainer and Evaluator. Use existing Dataset objects as the training/evaluation data abstraction. Rename/generalize the concrete trainer away from `WandbTrainer` because WandB is optional.

## Complexity Tracking

No constitution violations or complexity waivers required.

## Phase 0: Research Summary

Detailed decisions are recorded in [research.md](research.md). Key decisions:

- Use `Evaluator` for the full evaluation loop.
- Rename existing metric accumulator evaluators to `EvaluationMetrics`.
- Use Dataset objects for training and evaluation data inputs.
- Use pytest under grouped folders such as `tests/unit/evaluator`, `tests/unit/trainer`, and `tests/integration/training`, with mocks or doubles based on base implementations.
- Support constructor default Evaluator plus train/fit override per run.
- Generalize concrete trainer naming away from WandB-specific naming.
- Treat missing Evaluator as no evaluation: warning, training-loss-only logging, empty evaluation history.
- Enable checkpoint saving by default when a configured checkpoint path exists.
- Warn and train without checkpoints when checkpoint path is unset.
- Fail before training when a configured checkpoint path does not exist.
- Use a shared reporter interface across Trainer and Evaluator.
- Keep dry-run batch validation optional and disabled by default.

## Phase 1: Design Summary

Detailed entities are recorded in [data-model.md](data-model.md). Public behavior contracts are documented in [contracts](contracts/). Quickstart validation steps are documented in [quickstart.md](quickstart.md).

## Post-Design Constitution Check

**Security Constitution**: PASS. Contracts explicitly require no remote transmission in local/logging-only modes and no secret logging.

**Dependency Management Constitution**: PASS. No new runtime dependency is planned; pytest is documented for tests and optional reporting dependency usage is isolated behind reporter mode.

**Error Handling Constitution**: PASS. Contracts and quickstart include fatal exceptions, warning-only local plot fallback, missing Evaluator behavior, checkpoint path validation, checkpoint monitor validation, Dataset input validation, and interrupted training behavior.

**AI Usage Constitution**: PASS. Generated artifacts remain draft planning material pending human review and test-first implementation.

**Release Gates Constitution**: PASS. Quickstart and contracts identify verification commands and required behavior tests.

**Unified Acceptance Rule**: PASS. No unresolved gate violations remain.
