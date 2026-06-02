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

- `input_spec`
- `output_spec`
- `forward()`
- `postprocess()`
- `compute_loss()`

## Training Workflow

```python
from torch.optim import Adam

from vision_studio.models import ImageClassifier
from vision_studio.trainer import VisionTrainer

model = ImageClassifier(in_channels=3, num_classes=10)
optimizer = Adam(model.parameters(), lr=1e-3)
trainer = VisionTrainer(optimizer=optimizer, device="cpu")

# trainer.fit(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     num_epochs=10,
# )
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
