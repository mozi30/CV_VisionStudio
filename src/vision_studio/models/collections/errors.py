from __future__ import annotations


class ModelCollectionError(Exception):
    """Base error for model collection failures."""


class InvalidCollectionError(ModelCollectionError):
    """Raised when a collection identifier is unsupported."""


class MissingBackendError(ModelCollectionError):
    """Raised when the selected backend package is not installed."""


class InvalidModelError(ModelCollectionError):
    """Raised when a model identifier is unavailable for a collection."""


class AmbiguousModelError(ModelCollectionError):
    """Raised when a model name exists in multiple collections."""


class InvalidWeightError(ModelCollectionError):
    """Raised when a weight identifier is absent or incompatible."""


class WeightRetrievalError(ModelCollectionError):
    """Raised when a backend cannot retrieve selected pretrained weights."""


class WeightIntegrityError(ModelCollectionError):
    """Raised when available integrity metadata fails validation."""


class UnconfirmedWeightError(ModelCollectionError):
    """Raised when unverifiable weights need explicit user confirmation."""


class MetadataCacheError(ModelCollectionError):
    """Raised when metadata cache data is missing, corrupt, or incompatible."""
