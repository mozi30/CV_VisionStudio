# Data Model: Integrate Model Collections

## ModelCollection

**Purpose**: Represents an approved model collection source.

**Key fields / attributes**:

- `id`: Stable collection identifier (`timm`, `mmpretrain`, `torchvision`).
- `display_name`: Human-readable collection name.
- `backend_package`: Python package imported by the adapter when selected or refreshed.
- `installed`: Whether the backend package is importable in the current environment.
- `version`: Installed backend version when available.
- `adapter`: Adapter object that lists metadata and creates models.

**Validation rules**:

- `id` must be one of the approved collections.
- Missing backend package is allowed for unselected collections.
- Selecting a collection with missing backend package fails with a dependency diagnostic.

## ModelCollectionAdapter

**Purpose**: Backend-specific bridge that translates timm, MMPreTrain, or torchvision APIs into Vision Studio collection behavior.

**Key fields / attributes**:

- `collection_id`: Approved collection identifier handled by the adapter.
- `backend_module`: Lazily imported backend module.
- `list_models()`: Discovers backend model identifiers.
- `list_weights(model_id)`: Discovers weight variants for one model when available.
- `create_model(selection)`: Creates a model from a validated selection.

**Validation rules**:

- Must not import the backend until selected or explicitly refreshed.
- Must return safe user-facing errors for missing package, invalid model, invalid weight, and failed creation.
- Must include collection source metadata for created models or returned selection results.

## ModelOption

**Purpose**: Represents a specific model identifier within one collection.

**Key fields / attributes**:

- `collection_id`: Source collection identifier.
- `model_id`: Canonical backend model name.
- `display_name`: Optional human-readable name.
- `available_weights`: List of `WeightOption` values for the model.
- `default_weight_id`: Default pretrained weight identifier when the backend exposes one.
- `metadata`: Backend-supplied metadata that is safe to expose.
- `discovered_at`: Metadata refresh timestamp.

**Validation rules**:

- `model_id` must be unique within a collection.
- Ambiguous model names across collections require explicit collection selection.
- Normal listing reads cached model options instead of live backend catalogs.

## WeightOption

**Purpose**: Represents a pretrained weight variant for a model.

**Key fields / attributes**:

- `collection_id`: Source collection identifier.
- `model_id`: Compatible model identifier.
- `weight_id`: Canonical backend weight identifier.
- `is_default`: Whether this is the default pretrained weight for the model.
- `source`: Backend/source metadata shown to users.
- `integrity`: Hash/checksum/trusted metadata when available.
- `verified`: Whether source/integrity verification is available and passed.
- `requires_confirmation`: Whether loading requires an unverifiable-weight warning.

**Validation rules**:

- A weight option must belong to the selected model and collection.
- Failed available integrity validation blocks loading.
- Unverifiable approved-collection weights require warning and user confirmation before loading.

## ModelSelection

**Purpose**: User input that selects a collection model and weight behavior.

**Key fields / attributes**:

- `collection_id`: Required collection identifier.
- `model_id`: Required model identifier.
- `weight_mode`: One of `explicit`, `default`, or `none`.
- `weight_id`: Required only for explicit weight selection.
- `allow_unverified_weights`: Confirmation flag for unverifiable approved-collection weights.

**Validation rules**:

- `collection_id` and `model_id` are required.
- `weight_id` is required for explicit weight selection and forbidden for no-weight selection.
- Model-only selection uses default pretrained weights when available.
- If no default pretrained weights exist, users must choose no pretrained weights or an explicit weight variant.
- Dataset/task suitability is not validated here.

**State transitions**:

- `created` -> `validated` -> (`ready_to_create` | `requires_confirmation` | `failed`)
- `requires_confirmation` -> (`ready_to_create` | `failed`)

## ModelSelectionResult

**Purpose**: Output returned after a selection is validated and optionally initialized.

**Key fields / attributes**:

- `model`: Created model object when initialization succeeds.
- `collection_id`: Source collection identifier.
- `model_id`: Selected model identifier.
- `weight_id`: Applied weight identifier, or `None`.
- `weight_mode`: Applied weight mode.
- `source`: User-facing source information.
- `warnings`: Non-fatal warnings, including unverifiable approved-collection weights.

**Validation rules**:

- Must never report failed or skipped weight loading as successful.
- Must include source metadata for the model and weights.
- Must preserve existing custom model behavior outside collection selection.

## CollectionMetadataCache

**Purpose**: Local metadata store used for fast listing.

**Key fields / attributes**:

- `schema_version`: Cache schema version.
- `generated_at`: Timestamp of last successful refresh.
- `backend_versions`: Installed backend versions used for refresh.
- `collections`: Cached `ModelCollection`, `ModelOption`, and `WeightOption` metadata.
- `source`: Cache file path or storage provider.

**Validation rules**:

- Normal listing uses cache data.
- Refresh rebuilds metadata from installed supported backends.
- Corrupt cache fails with a clear diagnostic and can be rebuilt with refresh.
- Stale cache is allowed and must be documented as stale until refresh.

**State transitions**:

- `missing` -> `refreshing` -> `ready`
- `ready` -> `refreshing` -> (`ready` | `stale`)
- `ready` -> `corrupt` -> `refreshing`
