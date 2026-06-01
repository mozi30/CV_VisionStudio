# Quickstart: Unified Inference and Ensemble Evaluation

## Goal

Validate that one unified inference implementation supports single-model and ensemble workflows while reporter mode remains orthogonal to inference control flow.

## Prerequisites

- Python environment with project dependencies installed.
- Pytest available for test execution.
- At least two compatible trained models for ensemble checks.
- Dataset fixtures for supported inference/evaluation tasks.

## Trust Boundary & Logging Assumptions

- **Trust boundaries**:
  - Model artifact paths, image/video/webcam sources, and run-configuration values are treated as untrusted inputs until validated by inference configuration checks.
  - External integrations (for example reporter backends such as wandb) are treated as external trust boundaries; failures must be surfaced as safe diagnostics and must not alter inference orchestration semantics.
- **Validation assumptions**:
  - Empty model lists, incompatible class counts, invalid aggregation settings, invalid policy values, and unreadable/unopenable media sources fail before or at boundary entry points with explicit, actionable errors.
  - In continue-with-warning mode, at least one model must succeed; otherwise the run fails with a clear diagnostic.
- **Logging assumptions**:
  - Logs include run status, selected modes/policies, and high-level failure summaries needed for debugging and release evidence.
  - Logs must not contain secrets, credentials, tokens, full stack traces in user-facing paths, or private dataset contents.
  - Model failures are summarized with safe diagnostics and stable model identifiers only.
- **Operational assumptions**:
  - Webcam and video errors are expected operational failures and should be logged distinctly from programming/invariant failures.
  - Reporter mode changes destination/format only; it must not change decision logic, validation behavior, or failure-policy execution.

## Validation Steps

1. Run single-model inference and verify compatibility with existing output expectations.
2. Run ensemble inference/evaluation in soft-voting mode and verify aggregated outputs and metadata.
3. Run ensemble in hard-voting mode with tie-policy defaults and explicit random policy.
4. Run weighted averaging with valid and invalid weights to verify success and failure paths.
5. Validate strict class-count mismatch failure before aggregation.
6. Validate fail-fast policy stops on first model failure.
7. Validate continue-with-warning policy completes only when at least one model succeeds and marks status `completed_with_warnings`.
8. Run with different reporter modes and confirm inference control flow remains unchanged while output channels differ.
9. Run default image inference and verify no evaluation metrics are computed.
10. Run continuous video-path inference and verify ordered outputs until stream completion.
11. Run realtime webcam inference with default settings and verify continuous predictions.
12. Run webcam inference with frame skipping enabled and verify sampling behavior is applied.
13. Run webcam inference with annotated recording enabled and verify output video artifact is created.

## Suggested Verification Commands

- `python -m pytest tests/unit tests/integration`
- `python -m ruff check src tests`
- `python -m black --check src tests`

## Expected Outcomes

- Exactly one concrete inference implementation is exercised across supported use cases.
- Ensemble metadata includes aggregation mode, tie policy, failure policy, and failed model identifiers when applicable.
- Error paths are explicit, safe, and reproducible.

## Validation Results (Final Phase)

- Test suite command (UV environment):
  - `uv run pytest tests/unit tests/integration`
  - Result: **PASS** (`48 passed`)
- Lint/format commands (UV environment):
  - `uv run ruff check src tests && uv run black --check src tests`
  - Result: **FAIL** at repository scope due pre-existing Ruff baseline outside this feature scope.
- Feature-scope lint/format verification:
  - `uv run ruff check src/vision_studio/inference src/vision_studio/evaluate/evaluator.py tests/unit/inference tests/integration/inference tests/unit/evaluator/test_evaluator.py`
  - `uv run black --check src/vision_studio/inference src/vision_studio/evaluate/evaluator.py tests/unit/inference tests/integration/inference tests/unit/evaluator/test_evaluator.py`
  - Result: **PASS** for files created/modified in this feature stream.

## Release Notes

- Unified inference is now the canonical interface for single-model, ensemble, image, video, and webcam inference.
- Ensemble support includes soft/hard/weighted aggregation with explicit policy metadata.
- Realtime webcam mode supports source validation, frame skipping, and optional recording artifact output.
- Legacy classification and wandb inference entrypoints are routed through unified inference behavior.

## Rollback Guidance

- Revert the feature branch changes affecting:
  - `src/vision_studio/inference/`
  - `src/vision_studio/evaluate/evaluator.py`
  - `tests/unit/inference/`
  - `tests/integration/inference/`
  - `specs/002-model-ensemble-inference/`
- Re-run `uv run pytest tests/unit tests/integration` to verify baseline behavior after rollback.
- If partial rollback is required, prioritize restoring `src/vision_studio/inference/simple.py`, `src/vision_studio/inference/base.py`, and legacy adapters (`classification.py`, `wandb.py`) together to avoid interface drift.

## Owner Acceptance Evidence

- Implementation accepted against task plan through completion of Phases 1–6 and final-phase documentation/evidence updates.
- Human review gate acknowledged for AI-assisted artifacts; feature-scope Ruff/Black and full pytest evidence recorded above.
