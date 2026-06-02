from __future__ import annotations

from .base import ModelCollectionAdapter
from .errors import InvalidModelError
from .types import CollectionId, ModelOption, ModelSelection, WeightMode, WeightOption


class TimmAdapter(ModelCollectionAdapter):
    collection_id = CollectionId.TIMM.value
    display_name = "timm"
    backend_package = "timm"

    def list_models(self) -> list[ModelOption]:
        timm = self.import_backend()
        model_ids = list(timm.list_models())
        pretrained_ids = set(timm.list_models(pretrained=True))
        options: list[ModelOption] = []
        for model_id in model_ids:
            weights: tuple[WeightOption, ...] = ()
            default_weight_id = None
            if model_id in pretrained_ids:
                default_weight_id = "default"
                weights = (
                    WeightOption(
                        collection_id=self.collection_id,
                        model_id=model_id,
                        weight_id="default",
                        is_default=True,
                        source={"backend": self.display_name},
                        verified=False,
                        requires_confirmation=True,
                    ),
                )
            options.append(
                ModelOption(
                    collection_id=self.collection_id,
                    model_id=model_id,
                    display_name=model_id,
                    available_weights=weights,
                    default_weight_id=default_weight_id,
                    metadata={"backend": self.display_name},
                )
            )
        return options

    def list_weights(self, model_id: str) -> list[WeightOption]:
        model = next((item for item in self.list_models() if item.model_id == model_id), None)
        if model is None:
            raise InvalidModelError(f"Model '{model_id}' is not available in timm.")
        return list(model.available_weights)

    def create_model(self, selection: ModelSelection, weight_id: str | None) -> object:
        timm = self.import_backend()
        pretrained = selection.mode != WeightMode.NONE and weight_id is not None
        return timm.create_model(selection.model_id, pretrained=pretrained)
