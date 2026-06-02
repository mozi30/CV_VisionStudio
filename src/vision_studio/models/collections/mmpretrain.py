from __future__ import annotations

from .base import ModelCollectionAdapter
from .errors import InvalidModelError
from .types import CollectionId, ModelOption, ModelSelection, WeightMode, WeightOption


class MMPreTrainAdapter(ModelCollectionAdapter):
    collection_id = CollectionId.MMPRETRAIN.value
    display_name = "MMPreTrain"
    backend_package = "mmpretrain"

    def _apis(self) -> object:
        backend = self.import_backend()
        return getattr(backend, "apis", backend)

    def list_models(self) -> list[ModelOption]:
        apis = self._apis()
        model_ids = list(apis.list_models())
        return [
            ModelOption(
                collection_id=self.collection_id,
                model_id=model_id,
                display_name=model_id,
                available_weights=(
                    WeightOption(
                        collection_id=self.collection_id,
                        model_id=model_id,
                        weight_id="default",
                        is_default=True,
                        source={"backend": self.display_name},
                        verified=False,
                        requires_confirmation=True,
                    ),
                ),
                default_weight_id="default",
                metadata={"backend": self.display_name},
            )
            for model_id in model_ids
        ]

    def list_weights(self, model_id: str) -> list[WeightOption]:
        model = next((item for item in self.list_models() if item.model_id == model_id), None)
        if model is None:
            raise InvalidModelError(f"Model '{model_id}' is not available in MMPreTrain.")
        return list(model.available_weights)

    def create_model(self, selection: ModelSelection, weight_id: str | None) -> object:
        apis = self._apis()
        pretrained = selection.mode != WeightMode.NONE and weight_id is not None
        return apis.get_model(selection.model_id, pretrained=pretrained)
