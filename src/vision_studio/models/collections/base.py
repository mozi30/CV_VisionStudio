from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module, metadata
from types import ModuleType

from .errors import MissingBackendError
from .types import ModelOption, ModelSelection, WeightOption


class ModelCollectionAdapter(ABC):
    collection_id: str
    display_name: str
    backend_package: str

    def import_backend(self) -> ModuleType:
        try:
            return import_module(self.backend_package)
        except ModuleNotFoundError as exc:
            if exc.name == self.backend_package:
                raise MissingBackendError(
                    f"Backend '{self.backend_package}' is required for "
                    f"collection '{self.collection_id}'. Install it or choose "
                    "another collection."
                ) from exc
            raise

    def backend_version(self) -> str | None:
        try:
            return metadata.version(self.backend_package)
        except metadata.PackageNotFoundError:
            return None

    @abstractmethod
    def list_models(self) -> list[ModelOption]:
        raise NotImplementedError

    @abstractmethod
    def list_weights(self, model_id: str) -> list[WeightOption]:
        raise NotImplementedError

    @abstractmethod
    def create_model(self, selection: ModelSelection, weight_id: str | None) -> object:
        raise NotImplementedError


def safe_backend_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None
