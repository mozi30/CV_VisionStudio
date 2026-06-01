# Implementation Plan: Unified Inference, Ensemble, and Realtime Webcam Runs

**Branch**: `[002-add-model-ensamble]` | **Date**: 2026-05-27 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/002-model-ensemble-inference/spec.md`

## Summary

Design a unified inference service with one base abstraction and one concrete implementation for single-model and ensemble inference/evaluation. Support soft voting, hard voting, weighted averaging, strict class-count validation, deterministic/default policies, and partial-failure handling. Extend inference to prediction-only defaults, image mode, continuous video-path mode, and realtime webcam mode with optional frame skipping and annotated video output recording. Keep reporting channel selection behind reporter strategy without changing inference orchestration.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python >=3.10 (active environment Python 3.12 compatible)

**Primary Dependencies**: PyTorch, torchvision, numpy, pytest; optional reporting dependencies already present (wandb, matplotlib)

**Storage**: Local filesystem for model artifacts and optional annotated output video files

**Testing**: pytest unit and integration tests; integration coverage under `tests/integration/inference`

**Target Platform**: Linux development and package/CLI users with local image/video/webcam input sources

**Project Type**: Python computer-vision library/package

**Performance Goals**: Realtime webcam mode should maintain continuous inference with configurable frame-skipping to reduce processing load; deterministic modes remain reproducible.

**Constraints**: One concrete inference implementation only; inference defaults to prediction-only (no metric computation); class-count mismatch must fail; webcam unavailability must fail with clear diagnostics.

**Scale/Scope**: Single-image, video-path stream, and realtime webcam streams; single-model and ensemble flows for supported tasks; no new private-data security controls in scope.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Security Constitution**: PASS. No new private-data handling model is introduced for this feature scope. Existing safe logging and input-validation baseline remains required.

**Dependency Management Constitution**: PASS. No new mandatory runtime dependency required.

**Error Handling Constitution**: PASS. Explicit failures include empty model list, class-count mismatch, invalid mode/policy/weights, video stream unreadable, webcam unavailable, and zero-success continuation mode.

**AI Usage Constitution**: PASS. AI-generated outputs require human review before acceptance.

**Release Gates Constitution**: PASS. Release gates include pytest + lint/format checks, evidence for realtime inference behavior, and owner approval.

**Unified Acceptance Rule**: PASS. No unresolved gate violations.

## Project Structure

### Documentation (this feature)

```text
specs/002-model-ensemble-inference/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── inference-contract.md
│   ├── ensemble-aggregation-contract.md
│   └── reporting-contract.md
└── tasks.md
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
src/vision_studio/
├── inference/
│   ├── __init__.py
│   ├── base.py
│   ├── simple.py
│   ├── classification.py
│   └── wandb.py
├── evaluate/
│   ├── __init__.py
│   ├── evaluator.py
│   └── metrics.py
└── reporting/
    ├── __init__.py
    ├── base.py
    ├── logging.py
    ├── live_plot.py
    └── wandb.py

tests/
├── unit/
│   ├── inference/
│   └── reporting/
└── integration/
    └── inference/
```

**Structure Decision**: Reuse existing package layout and keep a single unified inference implementation entry path. Integration tests for this feature are under `tests/integration/inference`.

## Complexity Tracking

No constitution waivers required.

## Phase 0: Research Summary

Detailed decisions are in [research.md](research.md): unified inference architecture, ensemble policies/defaults, prediction-only default inference, realtime webcam with optional frame skipping, and annotated output recording.

## Phase 1: Design Summary

Entities are in [data-model.md](data-model.md); contracts are in [contracts](contracts/); scenario validation is in [quickstart.md](quickstart.md).

## Post-Design Constitution Check

**Security Constitution**: PASS.

**Dependency Management Constitution**: PASS.

**Error Handling Constitution**: PASS.

**AI Usage Constitution**: PASS.

**Release Gates Constitution**: PASS.

**Unified Acceptance Rule**: PASS.
