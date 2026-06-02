# Implementation Plan: Integrate Model Collections

**Branch**: `[003-add-model-collections]` | **Date**: 2026-06-02 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/003-integrate-model-collections/spec.md`

## Summary

Add a model-collection selection layer for baseline models from timm, MMPreTrain, and torchvision while preserving the existing custom `BaseModel` flow. The implementation will introduce collection adapters, cached model/weight metadata with explicit refresh, string identifiers for config/CLI, enum-friendly SDK values, and clear selection behavior for explicit weights, default pretrained weights, and no pretrained weights. Collection/model/weight compatibility is validated in this feature; dataset and task suitability remain downstream training/evaluation responsibilities.

## Technical Context

**Language/Version**: Python >=3.10, compatible with the active Python 3.12 development environment.

**Primary Dependencies**: Existing PyTorch package context; torchvision may remain a required project dependency for existing non-collection functionality; new optional collection backends for timm and MMPreTrain; standard-library `dataclasses`, `enum`, `importlib`, `json`, `pathlib`, and `warnings`; pytest for tests.

**Storage**: Local filesystem JSON metadata cache for discovered collection/model/weight records. Existing backend download/cache mechanisms remain responsible for pretrained weight artifacts.

**Testing**: pytest unit tests under `tests/unit/models` and integration tests under `tests/integration/models`; release gates should run pytest, ruff, black check, import smoke checks for missing optional backends, runtime weight-retrieval exception tests, and refreshed-cache count comparison against backend model listings.

**Target Platform**: Linux development environment and Python package/CLI consumers.

**Project Type**: Python computer-vision library/package with CLI entry point and notebook usage.

**Performance Goals**: 95% of normal model/weight listing requests complete within 2 seconds by reading cached metadata; unchanged custom model initialization increases by no more than 5%; release validation confirms refreshed cache exposes at least 90% of models listed by each installed supported backend.

**Constraints**: timm and MMPreTrain collection adapters must be lazy/optional for collection selection; torchvision baseline-model collection behavior must stay isolated behind the collection selection flow even if torchvision remains installed for other project functionality. Missing unselected optional backends must not affect startup, custom models, or other collections. Explicit metadata refresh is the only path that inspects installed backend catalogs. Selection validates collection/model/weight compatibility only. Runtime pretrained weight retrieval failure raises a clear exception. Unverifiable approved-collection weights require user confirmation before loading; failed integrity validation remains fatal.

**Scale/Scope**: Support collection discovery and initialization for baseline model collections from timm, MMPreTrain, and torchvision. Training workflows, dataset/task compatibility validation, new metric definitions, and model recommendations are out of scope. User documentation includes README updates for the new functionality.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Security Constitution**: PASS. Trust boundaries are external pretrained weight metadata/artifacts, optional backend imports, and user-provided collection/model/weight identifiers. Only approved collections are in scope; failed integrity validation is fatal; unverifiable weights require an explicit user warning/confirmation and human-owner risk acceptance before release.

**Dependency Management Constitution**: PASS. timm and MMPreTrain are optional new backend dependencies. The collection layer uses lazy imports so missing optional backends fail only when selected. Dependency documentation must include version ranges, licenses, maintenance status, vulnerability review, installation guidance, and removal strategy. torchvision is already present for other project functionality, but torchvision baseline-model collection behavior must still be isolated behind the collection selection flow.

**Error Handling Constitution**: PASS. Required failures include invalid collection/model/weight identifiers, ambiguous model names without collection, incompatible weight/model selection, missing selected backend, runtime weight retrieval exceptions, failed integrity validation, and unconfirmed unverifiable weights. Safe messages must list valid alternatives where practical and must not expose sensitive paths or stack traces in user-facing output.

**AI Usage Constitution**: PASS. AI-generated artifacts remain draft planning material. External API facts about timm, MMPreTrain, and torchvision were checked against official documentation and require human review before implementation acceptance.

**Release Gates Constitution**: PASS. Release requires pytest coverage for selection modes, cached listing, refresh, optional backend missing behavior, custom model compatibility, invalid alternatives, runtime retrieval exceptions, integrity/fingerprint warning paths, refreshed-cache model-count validation, README/docs updates, dependency review, linting/formatting, and owner approval.

**Unified Acceptance Rule**: PASS. The planned work can satisfy Security reviewed -> Dependencies justified -> Errors specified and tested -> AI output reviewed -> Release gates passed -> Human owner accepted.

## Project Structure

### Documentation (this feature)

```text
specs/003-integrate-model-collections/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── model-collection-contract.md
│   ├── metadata-cache-contract.md
│   └── dependency-and-security-contract.md
└── tasks.md
```

### Source Code (repository root)

```text
src/vision_studio/
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── image_classifier.py
│   └── collections/
│       ├── __init__.py
│       ├── base.py
│       ├── registry.py
│       ├── cache.py
│       ├── types.py
│       ├── timm.py
│       ├── mmpretrain.py
│       └── torchvision.py
├── types.py
└── main.py

tests/
├── unit/
│   └── models/
│       └── collections/
└── integration/
    └── models/
        └── collections/
```

**Structure Decision**: Reuse the existing `src/vision_studio/models` package and add a focused `models/collections` subpackage. Keep baseline collection behavior separate from custom `BaseModel` implementations so existing model construction and training/evaluation code paths remain unchanged.

## Complexity Tracking

No constitution waivers required. The accepted unverifiable-weight behavior is documented as an explicit user-confirmed risk path, not a silent security exception.

## Phase 0: Research Summary

Detailed decisions are recorded in [research.md](research.md). Key decisions:

- Use one adapter interface per collection backend.
- Use cached metadata for normal listing and an explicit refresh operation for backend catalog inspection.
- Support strings for config/CLI and enum-friendly SDK constants without requiring dynamic enums for every backend model.
- Support three selection modes: explicit weight, default pretrained weight when available, and no pretrained weights.
- Validate collection/model/weight compatibility only; defer dataset/task compatibility to training or evaluation workflows.
- Lazy-load timm and MMPreTrain optional backends and fail selected missing backends with clear dependency diagnostics.
- Keep torchvision collection behavior isolated even though torchvision may remain required elsewhere in the project.
- Treat runtime pretrained weight retrieval failure as an exception that prevents successful-load reporting.
- Treat unverifiable approved-collection weights as a user-confirmed warning path; failed available integrity validation remains fatal.

## Phase 1: Design Summary

Entities are in [data-model.md](data-model.md); public contracts are in [contracts](contracts/); validation and release checks are in [quickstart.md](quickstart.md).

## Post-Design Constitution Check

**Security Constitution**: PASS. Contracts require source disclosure, warning/confirmation for unverifiable approved-collection weights, and fatal handling for failed integrity checks.

**Dependency Management Constitution**: PASS. Contracts isolate optional backend imports and document dependency review requirements, while allowing torchvision to remain required for non-collection functionality.

**Error Handling Constitution**: PASS. Contracts and quickstart cover invalid identifiers, ambiguity, missing optional backends, stale cache, failed refresh, runtime weight-retrieval exceptions, unverifiable weights, and failed integrity validation.

**AI Usage Constitution**: PASS. Generated artifacts cite official backend documentation in research and remain subject to human review.

**Release Gates Constitution**: PASS. Quickstart lists pytest, lint/format, dependency review, cache-count validation, README/docs, and owner acceptance evidence.

**Unified Acceptance Rule**: PASS. No unresolved gate violations remain.
