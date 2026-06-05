# Implementation Plan: Ensemble Dataset Evaluation

**Branch**: `[004-ensamble-evaluator]` | **Date**: 2026-06-02 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/004-ensemble-evaluation/spec.md`

## Summary

Add dataset-level ensemble evaluation to `LoopEvaluator` while preserving existing single-model evaluation behavior. The implementation will introduce an ensemble evaluation entry point that accepts a model list, dataset, and ensemble configuration; evaluates each model on every batch; aggregates predictions using the same soft, hard, weighted, tie-policy, and failure-policy behavior as ensemble inference; updates existing evaluation metrics with aggregated predictions; reports loss as the mean successful per-model loss per batch; and returns a flat metric result with ensemble audit fields.

## Technical Context

**Language/Version**: Python >=3.10, compatible with the active Python 3.12 development environment.

**Primary Dependencies**: Existing PyTorch and project-local `vision_studio` evaluation, inference, reporting, and type modules. No new runtime dependencies.

**Storage**: N/A. The feature is in-memory evaluation behavior and does not add persistence.

**Testing**: pytest unit tests under `tests/unit/evaluator` and integration tests under `tests/integration/evaluator` or the nearest existing integration test layout.

**Target Platform**: Linux development environment and Python package/notebook consumers.

**Project Type**: Python computer-vision library/package with notebook, example, and CLI-oriented usage.

**Performance Goals**: Ensemble evaluation performs one forward and loss computation per participating successful model per dataset batch plus aggregation overhead; aggregation overhead remains small relative to model inference for normal tensor batch sizes.

**Constraints**: Preserve current `LoopEvaluator.evaluate()` behavior for single-model evaluation; use the same ensemble aggregation properties as `SimpleInference.run_ensemble`; return flat results; report loss as mean successful per-model loss per batch; restore model training/evaluation state after success or failure; do not add dependencies.

**Scale/Scope**: Support classification-style ensemble outputs for this increment, matching current ensemble inference and `ClassificationEvaluationMetrics`. Detection and other task-specific ensemble metric behavior remain out of scope unless introduced by a later feature.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Security Constitution**: PASS. The feature does not add new credentials, network access, artifact loading, authentication, authorization, or external trust boundaries. Inputs are existing in-process models, datasets, targets, and configuration values. Error messages must remain safe and avoid exposing stack traces in user-facing diagnostics.

**Dependency Management Constitution**: PASS. No new dependency is planned. The implementation should reuse existing PyTorch tensors, existing evaluator metrics, and existing ensemble configuration concepts.

**Error Handling Constitution**: PASS. Expected failure modes are specified: empty model list, invalid aggregation settings, incompatible prediction shapes/class counts, model batch failure under fail-fast, all models failing under continue-with-warning, and metric incompatibility. Tests must cover these paths and prove no partial result is reported as successful.

**AI Usage Constitution**: PASS. AI is used as Builder/Summarizer for spec and plan artifacts. Implementation and tests require human review, especially for metric semantics and compatibility with existing inference behavior.

**Release Gates Constitution**: PASS. Release gates include pytest coverage for ensemble modes, invalid configs, failure policies, metadata, loss semantics, training-state restoration, unchanged single-model evaluation, integration workflow, and lint/format checks where configured.

**Unified Acceptance Rule**: PASS. The planned work can satisfy Security reviewed -> Dependencies justified -> Errors specified and tested -> AI output reviewed -> Release gates passed -> Human owner accepted.

## Project Structure

### Documentation (this feature)

```text
specs/004-ensemble-evaluation/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── ensemble-evaluation-contract.md
└── tasks.md
```

### Source Code (repository root)

```text
src/vision_studio/
├── evaluate/
│   ├── evaluator.py
│   ├── classication.py
│   └── metrics.py
├── inference/
│   └── simple.py
└── types.py

tests/
├── unit/
│   └── evaluator/
│       ├── test_evaluator.py
│       └── test_ensemble_evaluator.py
└── integration/
    └── evaluator/
        └── test_ensemble_evaluation_workflow.py
```

**Structure Decision**: Extend the existing `src/vision_studio/evaluate` package and keep ensemble dataset evaluation inside `LoopEvaluator`, as requested. Reuse or align with `src/vision_studio/inference/simple.py` aggregation behavior rather than creating a separate ensemble subsystem.

## Complexity Tracking

No constitution violations or waivers are required.

## Phase 0: Research Summary

Detailed decisions are recorded in [research.md](research.md). Key decisions:

- Add an explicit ensemble evaluation entry point on `LoopEvaluator` instead of changing the existing single-model `evaluate()` signature.
- Reuse or extract the existing ensemble configuration and aggregation rules so evaluator and inference behavior stay aligned.
- Compute non-loss metrics from aggregated ensemble predictions and report loss as the mean successful per-model loss per batch.
- Return a flat result with metric keys plus `status`, `aggregation_metadata`, and `failed_models`.
- Restore each model's original training state in a `finally`-style cleanup path after success or failure.
- Keep scope to classification-style outputs for this increment.

## Phase 1: Design Summary

Entities are in [data-model.md](data-model.md), the public behavior contract is in [contracts/ensemble-evaluation-contract.md](contracts/ensemble-evaluation-contract.md), and validation steps are in [quickstart.md](quickstart.md).

## Post-Design Constitution Check

**Security Constitution**: PASS. Design introduces no new trust boundary and specifies safe invalid-state handling.

**Dependency Management Constitution**: PASS. Design uses existing project modules and adds no runtime dependency.

**Error Handling Constitution**: PASS. Contract and quickstart cover invalid config, incompatible model outputs, per-model failures, all-model failures, metric incompatibility, and cleanup expectations.

**AI Usage Constitution**: PASS. AI-generated artifacts remain draft planning material requiring human review.

**Release Gates Constitution**: PASS. Quickstart lists targeted pytest commands and behavior checks required before acceptance.

**Unified Acceptance Rule**: PASS. No unresolved gate violations remain.
