from abc import ABC, abstractmethod
from typing import Dict


class BaseEvaluator(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def update(self, predictions, targets) -> None:
        pass

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        pass
