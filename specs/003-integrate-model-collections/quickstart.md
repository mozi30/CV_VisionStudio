# Quickstart: Integrate Model Collections

## Goal

Validate that baseline models from timm, MMPreTrain, and torchvision can be discovered, selected, and initialized without breaking existing custom model construction.

## Prerequisites

- Python environment with Vision Studio installed in editable or package mode.
- Pytest available for test execution.
- Optional backend packages installed only for the collections being tested; torchvision may also be installed for existing non-collection project functionality.
- Network or backend cache availability when pretrained weights must be retrieved.

## Trust Boundary & Logging Assumptions

- External pretrained weights and backend metadata are trust boundaries.
- Collection/model/weight identifiers are user-controlled input and must be validated.
- Missing optional backends are expected configuration failures, not internal crashes.
- User-facing diagnostics must not include secrets, credentials, private dataset paths, or raw stack traces.
- Unverifiable approved-collection weights may load only after a warning and explicit confirmation.
- Failed available integrity validation must block loading.

## Validation Steps

1. List cached model metadata for each supported collection and verify collection source plus weight variants are shown.
2. Refresh metadata with one backend installed and verify the cache records backend version and refresh time.
3. Refresh metadata with one optional backend missing and verify other installed backends still refresh.
4. Select a collection, model, and explicit weight variant; verify the model is initialized with that weight source.
5. Select only a model with default pretrained weights available; verify default weights are applied.
6. Select only a model with no default pretrained weights; verify the user must choose explicit no-weight behavior or an explicit weight.
7. Select a model with explicit no-pretrained-weights behavior; verify no weight retrieval is attempted.
8. Select an invalid model and verify the error lists valid alternatives when available.
9. Select a weight that is incompatible with the model and verify a clear validation error.
10. Select a model name that exists in multiple collections without collection and verify ambiguity failure.
11. Select a missing optional backend and verify dependency installation guidance.
12. Simulate weight retrieval failure and verify a runtime weight-retrieval exception is raised, the failure reason is surfaced, and no successful model load is reported.
13. Simulate unverifiable approved-collection weights and verify warning plus confirmation behavior.
14. Simulate failed available integrity validation and verify loading is refused.
15. Compare refreshed cache model counts against each installed backend's model listing and verify the release exposes at least 90% of backend-listed models.
16. Construct an existing custom `ImageClassifier` or `BaseModel` subclass and verify behavior is unchanged.

## Suggested Verification Commands

- `python -m pytest tests/unit/models tests/integration/models`
- `python -m pytest tests/unit tests/integration`
- `python -m ruff check src tests`
- `python -m black --check src tests`
- `vision-studio model-collections refresh --collection timm`
- `vision-studio model-collections list --collection timm`
- `vision-studio model-collections validate-cache`

## SDK Examples

```python
from vision_studio.models import (
    CollectionId,
    ModelSelection,
    WeightMode,
    list_available_models,
    refresh_collection_metadata,
    select_model_collection,
)

refresh_collection_metadata(collection_ids=[CollectionId.TIMM])
models = list_available_models(CollectionId.TIMM)

explicit = select_model_collection(
    ModelSelection(
        collection_id=CollectionId.TIMM,
        model_id="resnet18",
        weight_mode=WeightMode.EXPLICIT,
        weight_id="imagenet",
    )
)

default = select_model_collection(
    ModelSelection(
        collection_id=CollectionId.TIMM,
        model_id="resnet18",
        weight_mode=WeightMode.DEFAULT,
    )
)

no_weights = select_model_collection(
    ModelSelection(
        collection_id=CollectionId.TIMM,
        model_id="resnet18",
        weight_mode=WeightMode.NONE,
    )
)
```

## Expected Outcomes

- Normal listing uses cached metadata and meets the 2-second target.
- Collection adapters are lazy and optional.
- Selected missing backends fail only when selected or refreshed.
- Selection result metadata identifies collection, model, and weight source.
- Existing custom model flows do not import or require timm/MMPreTrain collection backends.

## Documentation Updates

- Add user-facing examples for string identifiers in config/CLI.
- Add SDK examples for collection constants and weight modes.
- Document how to refresh model metadata.
- Document optional backend installation commands and version ranges.
- Document security behavior for unverifiable and failed-integrity weights.
- Update README.md with the new baseline model collection functionality, selection modes, metadata refresh behavior, optional backend guidance, and runtime weight-retrieval exception behavior.

## Release Evidence To Record

- Pytest results for unit and integration model-collection tests: `.venv/bin/python -m pytest tests/unit/models tests/integration/models` passed with 31 tests on 2026-06-02.
- Full pytest results: `.venv/bin/python -m pytest tests/unit tests/integration` passed with 79 tests on 2026-06-02.
- Lint results: `.venv/bin/python -m ruff check src tests` passed on 2026-06-02.
- Format results: `.venv/bin/python -m black --check ...` could not complete in this environment because Black repeatedly hung after a Python 3.15 parser safety warning under Python 3.12; re-run with a compatible Black/Python target before release.
- Dependency review for timm, MMPreTrain, and torchvision collection backend usage: pending.
- Security review for pretrained weight trust and warning behavior: pending.
- Cache exposure validation against installed backend model counts: pending.
- Documentation review for install, refresh, and selection examples: pending.
- README review for new model collection behavior: pending.
- Human owner acceptance: pending.
