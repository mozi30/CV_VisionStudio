from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class CollectionId(StrEnum):
    TIMM = "timm"
    MMPRETRAIN = "mmpretrain"
    TORCHVISION = "torchvision"


class WeightMode(StrEnum):
    EXPLICIT = "explicit"
    DEFAULT = "default"
    NONE = "none"


@dataclass(frozen=True)
class WeightOption:
    collection_id: str
    model_id: str
    weight_id: str
    is_default: bool = False
    source: dict[str, Any] = field(default_factory=dict)
    integrity: dict[str, Any] = field(default_factory=dict)
    verified: bool = True
    requires_confirmation: bool = False


@dataclass(frozen=True)
class ModelOption:
    collection_id: str
    model_id: str
    display_name: str | None = None
    available_weights: tuple[WeightOption, ...] = ()
    default_weight_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    discovered_at: str | None = None


@dataclass(frozen=True)
class ModelCollection:
    id: str
    display_name: str
    backend_package: str
    installed: bool = False
    version: str | None = None


@dataclass(frozen=True)
class ModelSelection:
    collection_id: str | CollectionId
    model_id: str
    weight_mode: str | WeightMode = WeightMode.DEFAULT
    weight_id: str | None = None
    allow_unverified_weights: bool = False

    @property
    def collection(self) -> str:
        return str(self.collection_id)

    @property
    def mode(self) -> WeightMode:
        return WeightMode(str(self.weight_mode))


@dataclass(frozen=True)
class ModelSelectionResult:
    model: Any
    collection_id: str
    model_id: str
    weight_id: str | None
    weight_mode: WeightMode
    source: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CollectionMetadataCache:
    schema_version: int
    generated_at: str
    backend_versions: dict[str, str | None]
    collections: dict[str, list[ModelOption]]
    source: Path | None = None
