from __future__ import annotations

from .base import ModelCollectionAdapter
from .errors import InvalidModelError
from .types import CollectionId, ModelOption, ModelSelection, WeightMode, WeightOption


class TorchVisionAdapter(ModelCollectionAdapter):
    collection_id = CollectionId.TORCHVISION.value
    display_name = "torchvision"
    backend_package = "torchvision"

    def _models_module(self) -> object:
        backend = self.import_backend()
        return backend.models

    def list_models(self) -> list[ModelOption]:
        models = self._models_module()
        options: list[ModelOption] = []
        for model_id in list(models.list_models()):
            weights: list[WeightOption] = []
            default_weight_id: str | None = None
            try:
                enum = models.get_model_weights(model_id)
                for item in enum:
                    weight_id = item.name
                    is_default = item == enum.DEFAULT
                    if is_default:
                        default_weight_id = weight_id
                    weights.append(
                        WeightOption(
                            collection_id=self.collection_id,
                            model_id=model_id,
                            weight_id=weight_id,
                            is_default=is_default,
                            source={
                                "backend": self.display_name,
                                "url": getattr(item, "url", None),
                            },
                            verified=bool(getattr(item, "url", None)),
                            requires_confirmation=not bool(getattr(item, "url", None)),
                        )
                    )
            except Exception:
                weights = []
            options.append(
                ModelOption(
                    collection_id=self.collection_id,
                    model_id=model_id,
                    display_name=model_id,
                    available_weights=tuple(weights),
                    default_weight_id=default_weight_id,
                    metadata={"backend": self.display_name},
                )
            )
        return options

    def list_weights(self, model_id: str) -> list[WeightOption]:
        model = next((item for item in self.list_models() if item.model_id == model_id), None)
        if model is None:
            raise InvalidModelError(f"Model '{model_id}' is not available in torchvision.")
        return list(model.available_weights)

    def create_model(self, selection: ModelSelection, weight_id: str | None) -> object:
        models = self._models_module()
        weights = None
        if selection.mode != WeightMode.NONE and weight_id is not None:
            enum = models.get_model_weights(selection.model_id)
            weights = getattr(enum, weight_id)
        return models.get_model(selection.model_id, weights=weights)
