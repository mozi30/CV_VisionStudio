# Quickstart: Ensemble Dataset Evaluation

## Intended User Flow

1. Create or load two or more compatible classification models.
2. Create an evaluation dataset or data loader that yields existing evaluator batches.
3. Create classification evaluation metrics.
4. Create a `LoopEvaluator`.
5. Run ensemble evaluation with the model list and optional ensemble configuration.
6. Inspect flat metric values plus `status`, `aggregation_metadata`, and `failed_models`.

## Example Shape

```python
from vision_studio.evaluate import ClassificationEvaluationMetrics, LoopEvaluator
from vision_studio.inference.simple import EnsembleConfig

metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
evaluator = LoopEvaluator(metrics=metrics)

result = evaluator.evaluate_ensemble(
    models=[model_a, model_b],
    dataset=eval_loader,
    config=EnsembleConfig(mode="soft"),
)

print(result["loss"])
print(result["accuracy"])
print(result["status"])
print(result["aggregation_metadata"]["mode"])
```

## Validation Commands

Run focused tests:

```bash
uv run pytest tests/unit/evaluator/test_evaluator.py tests/unit/evaluator/test_ensemble_evaluator.py
```

Run integration coverage for the workflow:

```bash
uv run pytest tests/integration/evaluator/test_ensemble_evaluation_workflow.py
```

Run related existing ensemble inference tests:

```bash
uv run pytest tests/unit/inference/test_ensemble_soft_voting.py tests/unit/inference/test_ensemble_hard_voting.py tests/unit/inference/test_ensemble_weighted.py tests/unit/inference/test_failure_policies.py
```

## Required Checks

- Default ensemble evaluation uses soft voting.
- Soft, hard, and weighted aggregation modes produce expected metrics.
- Weighted mode validates weight count and presence.
- Incompatible class counts or prediction shapes fail before successful metric output.
- `fail-fast` stops on the first model failure.
- `continue-with-warning` returns warnings metadata when at least one model succeeds.
- All-model failure under continue-with-warning fails the run.
- Loss equals the mean successful per-model loss per batch.
- Result is flat and includes metric keys plus `status`, `aggregation_metadata`, and `failed_models`.
- Single-model `evaluate()` behavior remains unchanged.
- Model training/evaluation states are restored after success and failure.

## Release Evidence Notes

- AI-review required: confirm implementation matches `spec.md`, `plan.md`, and `contracts/ensemble-evaluation-contract.md`.
- Dependency review: no new runtime dependency is introduced; implementation uses existing PyTorch and Vision Studio modules.
- Error-path coverage: unit tests cover invalid weights, incompatible output classes, all-model failure, continue-with-warning, fail-fast restoration, and flat metadata shape.
- Test evidence: `env PYTHONPATH=. uv run pytest` passed with 92 tests.
- Feature lint evidence: `uv run ruff check src/vision_studio/evaluate/evaluator.py tests/unit/evaluator/test_evaluator.py tests/unit/evaluator/test_ensemble_evaluator.py tests/integration/evaluator/test_ensemble_evaluation_workflow.py examples/collection_model_full_pipeline.py` passed.
- Feature format evidence: `uv run black --check src/vision_studio/evaluate/evaluator.py tests/unit/evaluator/test_evaluator.py tests/unit/evaluator/test_ensemble_evaluator.py tests/integration/evaluator/test_ensemble_evaluation_workflow.py examples/collection_model_full_pipeline.py` passed.
- Repo-wide style status: `uv run ruff check .` and `uv run black --check .` currently fail on pre-existing notebook lint issues and model-collection formatting outside this feature's touched Python files.
- Human-owner acceptance: pending final owner review after tests, formatting, and linting pass.
