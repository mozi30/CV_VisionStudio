from .base import Inference
from .classification import ClassificationInference
from .simple import SimpleInference
from .wandb import WandbInference

__all__ = ["ClassificationInference", "Inference", "SimpleInference", "WandbInference"]
