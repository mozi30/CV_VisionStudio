# Vision Studio

Vision Studio is a Python computer-vision toolkit for building end-to-end
experiments with PyTorch. It includes dataset wrappers, augmentation pipelines,
model interfaces, training loops, evaluation helpers, inference utilities, and
reporting integrations.

The package is designed around a strict model contract: `forward()` returns raw
outputs, task-specific postprocessing happens in `postprocess()`, and training
uses explicit loss outputs. This keeps custom models predictable while still
allowing common baseline models to be pulled from supported collections.

## Features

- Custom model interface through `BaseModel`.
- Built-in `ImageClassifier` example model.
- Dataset helpers for image classification, COCO, ImageNet-style, and MNIST
  workflows.
- Simple and balanced data loader helpers.
- Augmentation and preprocessing transforms for common image pipelines.
- Training through `VisionTrainer` and optional W&B-aware trainer/reporting.
- Evaluation and inference utilities.
- Baseline model collections from `timm`, `MMPreTrain`, and `torchvision`.
- Cached model metadata listing and explicit collection refresh commands.

## Installation

For local development, install the package in editable mode:

```bash
pip install -e .
```

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Optional model-collection backends can be installed only when needed:

```bash
pip install -e ".[timm]"
pip install -e ".[mmpretrain]"
pip install -e ".[model-collections]"
```

`torchvision` is a normal project dependency because it is also used by existing
non-collection functionality.

## Package Overview

```text
vision_studio/
├── augmentation/    Image augmentations and preprocessing helpers
├── data_loader/     Simple and balanced data loader wrappers
├── dataset/         Dataset abstractions and common dataset types
├── evaluate/        Evaluators and metrics
├── inference/       Inference interfaces and simple prediction helpers
├── models/          Custom models and baseline model collections
├── reporting/       Logging, live plot, and W&B reporting helpers
├── trainer/         Training loops and checkpoint policies
├── transforms/      Transform primitives
└── visualise/       Visualization utilities
```

## Basic Custom Model Usage

```python
import torch

from vision_studio.models import ImageClassifier

model = ImageClassifier(in_channels=3, num_classes=10)
inputs = torch.randn(4, 3, 32, 32)

logits = model(inputs)
outputs = model.postprocess(logits)

print(outputs["labels"])
```

Custom models should inherit from `BaseModel` and implement:

- `forward()`
- `postprocess()`
- `compute_loss()`

`input_spec` and `output_spec` are available as optional metadata hooks. They
default to empty dictionaries, so simple custom models do not need to implement
them unless downstream tooling should inspect expected shapes or dtypes.

## Training Workflow

```python
import torch
from torch.optim import Adam

from vision_studio.data_loader import SimpleDataLoader
from vision_studio.evaluate import ClassificationEvaluationMetrics, LoopEvaluator
from vision_studio.models import ImageClassifier
from vision_studio.trainer import VisionTrainer
from vision_studio.types import TrainerSettings

model = ImageClassifier(in_channels=3, num_classes=10)
optimizer = Adam(model.parameters(), lr=1e-3)
metrics = ClassificationEvaluationMetrics(num_classes=10, topk=(1, 5))
evaluator = LoopEvaluator(metrics=metrics, device="cpu")
trainer = VisionTrainer(
    optimizer=optimizer,
    evaluator=evaluator,
    device="cpu",
    settings=TrainerSettings(
        epochs=10,
        checkpoint_path=None,
        best_checkpoint_count=0,
    ),
)

# Any PyTorch-style dataset that returns (image, target) can be wrapped.
# Images may be PIL images, CHW tensors, NumPy HWC arrays, or NumPy CHW arrays.
dataset = [
    (torch.zeros(3, 32, 32), {"label": 0}),
    (torch.ones(3, 32, 32), {"label": 1}),
]
loader = SimpleDataLoader(dataset, batch_size=2)

result = trainer.fit(model, loader, loader)
print(result["history"]["evaluation"][-1])
```

The trainer expects model methods to follow the `BaseModel` contract, so training
and validation steps can call `forward()` and `compute_loss()` consistently.

## Datasets and Data Loading

Common public exports include:

```python
from vision_studio.dataset import (
    CocoDataset,
    Dataset,
    ImageClassificationDataset,
    ImageNetClassificationDataset,
    MnistDataset,
)
from vision_studio.data_loader import BalancedDataLoader, SimpleDataLoader
```

Typical classification data is organized as image files plus an annotation file.
The exact dataset class decides the expected annotation format.

`SimpleDataLoader` accepts common PyTorch dataset return shapes:

- `(image, label)` for classification; scalar labels become `{"label": ...}`.
- `(image, target_dict)` for task-specific targets.
- `{"image": image, ...}`, `{"input": image, ...}`, or `{"inputs": image, ...}`.

Input conversion rules:

- PIL images are converted with torchvision's `to_tensor()`.
- Tensor inputs are kept as-is.
- NumPy integer arrays are converted to float tensors in `[0, 1]`.
- NumPy HWC images are converted to CHW tensors.
- NumPy CHW tensors are preserved.

For ambiguous NumPy arrays, set the layout explicitly:

```python
loader_hwc = SimpleDataLoader(dataset, batch_size=32, numpy_layout="hwc")
loader_chw = SimpleDataLoader(dataset, batch_size=32, numpy_layout="chw")
loader_auto = SimpleDataLoader(dataset, batch_size=32, numpy_layout="auto")
```

`BalancedDataLoader` inherits the same input conversion behavior, including the
`numpy_layout` option, while sampling labels with class-balanced weights.

Target conventions used by the built-in workflows:

- Classification: `{"label": Tensor[batch]}` or `{"labels": Tensor[batch]}`.
- Detection: `{"boxes": Tensor[num_objects, 4], "labels": Tensor[num_objects]}`.
  Variable-length detection targets remain as per-image lists after collation.
- Segmentation: `{"mask": Tensor[height, width]}` for class-id masks.

Nested tensors inside dictionaries, lists, and tuples are moved to the selected
device by the trainer and evaluator.

## Evaluation and Metrics

`LoopEvaluator` requires a metrics object that implements the
`EvaluationMetrics` interface:

```python
from vision_studio.evaluate import EvaluationMetrics


class MyMetrics(EvaluationMetrics):
    def reset(self) -> None:
        ...

    def update(self, predictions, targets, loss) -> None:
        ...

    def compute(self) -> dict:
        return {"loss": 0.0}
```

Built-in metric implementations include `ClassificationEvaluationMetrics`,
`DetectionEvaluationMetrics`, `ConfusionMatrixEvaluationMetrics`, and
`LossEvaluationMetrics`.

Reporter ownership is scoped by workflow:

- `VisionTrainer.fit()` owns the reporting session from the beginning of
  training until the end.
- Standalone `LoopEvaluator.evaluate()` owns its own reporting session.
- Evaluations run from inside `VisionTrainer.fit()` do not start or finish a
  second reporter session.

## Augmentation and Preprocessing

```python
from vision_studio.augmentation import HorizontalFlip, Resize
from vision_studio.transforms import ImageToArray, Normalize, ToTensor

transforms = [
    ImageToArray(),
    HorizontalFlip(),
    Resize(32, 32),
    ToTensor(),
    Normalize(),
]
```

The augmentation package includes geometric transforms, color transforms, noise
and blur transforms, cutout/mixup/cutmix helpers, and preprocessing filters such
as grayscale, histogram equalization, Sobel, Canny, and custom filters.

## Inference

```python
from vision_studio.inference import SimpleInference

inference = SimpleInference(device="cpu")

# results = inference.predict(model, data_loader)
```

Inference utilities are separate from model definitions so the same model can be
trained, evaluated, and served through different workflows.

## Baseline Model Collections

Vision Studio supports baseline model selection from approved collections next
to custom `BaseModel` implementations:

- `timm`
- `mmpretrain`
- `torchvision`

Config and CLI callers use string identifiers. SDK callers can use
`CollectionId`, `WeightMode`, and `ModelSelection` constants from
`vision_studio.models`.

```python
from vision_studio.models import (
    CollectionId,
    ModelSelection,
    WeightMode,
    select_model_collection,
)

result = select_model_collection(
    ModelSelection(
        collection_id=CollectionId.TIMM,
        model_id="resnet18",
        weight_mode=WeightMode.DEFAULT,
    )
)
model = result.model
```

Supported pretrained-weight modes:

- `WeightMode.EXPLICIT`: load a named weight variant.
- `WeightMode.DEFAULT`: load the backend default when one exists.
- `WeightMode.NONE`: create the model without pretrained weights.

Runtime pretrained weight download/load failures raise `WeightRetrievalError`.
The selection result is not reported as successful when weights cannot be
retrieved.

## Metadata Refresh and Listing

Normal model listing reads local cached metadata for speed. Refresh the cache
explicitly when installed backend catalogs change:

```bash
vision-studio model-collections refresh --collection timm
vision-studio model-collections list --collection timm
vision-studio model-collections validate-cache
```

Release validation compares refreshed cache model counts against installed
backend `list_models` counts and expects at least 90% exposure for each installed
supported backend unless a waiver is recorded.

## Optional Backends

`torchvision` remains a project dependency for existing non-collection
functionality. Its baseline collection behavior is still isolated behind the
collection selection or refresh flow.

`timm` and `MMPreTrain` are optional collection backends. Install only the
collections you need:

```bash
pip install "vision-studio[timm]"
pip install "vision-studio[mmpretrain]"
pip install "vision-studio[model-collections]"
```

## CLI

Running the command without subcommands prints a small image-classification demo:

```bash
vision-studio
```

Model collection commands:

```bash
vision-studio model-collections refresh
vision-studio model-collections refresh --collection timm
vision-studio model-collections list
vision-studio model-collections list --collection torchvision
vision-studio model-collections validate-cache
```

## Development

Run the unit and integration suite:

```bash
.venv/bin/python -m pytest tests/unit tests/integration
```

Run linting:

```bash
.venv/bin/python -m ruff check src tests
```

Run formatting checks when Black is available in a compatible environment:

```bash
.venv/bin/python -m black --check src tests
```

## Current Status

The package is early-stage (`0.0.1`). APIs may still evolve while feature specs
are implemented. The model-collection feature currently focuses on selection,
metadata caching, optional backend isolation, and runtime error behavior; dataset
or task suitability checks remain the responsibility of training and evaluation
workflows.
