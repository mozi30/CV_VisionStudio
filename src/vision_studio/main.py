from __future__ import annotations

import argparse
import json
import sys

import torch
from torch.optim import Adam
from torchvision.transforms import Compose

from vision_studio.augmentation import HorizontalFlip, Resize
from vision_studio.inference import SimpleInference
from vision_studio.models import ImageClassifier
from vision_studio.trainer import VisionTrainer
from vision_studio.transforms import ImageToArray, Normalize, ToTensor


def _model_collections_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vision-studio")
    subparsers = parser.add_subparsers(dest="command")
    collections = subparsers.add_parser("model-collections")
    collection_commands = collections.add_subparsers(dest="collection_command")

    list_command = collection_commands.add_parser("list")
    list_command.add_argument("--collection", default=None)
    list_command.add_argument("--cache-path", default=None)

    refresh_command = collection_commands.add_parser("refresh")
    refresh_command.add_argument("--collection", action="append", dest="collections")
    refresh_command.add_argument("--cache-path", default=None)

    validate_command = collection_commands.add_parser("validate-cache")
    validate_command.add_argument("--cache-path", default=None)
    validate_command.add_argument("--minimum-ratio", type=float, default=0.9)
    return parser


def _run_model_collections_cli(argv: list[str]) -> bool:
    if not argv or argv[0] != "model-collections":
        return False

    from vision_studio.models.collections import (
        list_available_models,
        refresh_collection_metadata,
        validate_cache_exposure,
    )

    parser = _model_collections_parser()
    args = parser.parse_args(argv)
    if args.collection_command == "list":
        options = {
            "collection_id": args.collection,
        }
        if args.cache_path:
            options["cache_path"] = args.cache_path
        models = list_available_models(**options)
        print(
            json.dumps(
                [
                    {
                        "collection_id": model.collection_id,
                        "model_id": model.model_id,
                        "weights": [
                            {
                                "weight_id": weight.weight_id,
                                "is_default": weight.is_default,
                            }
                            for weight in model.available_weights
                        ],
                    }
                    for model in models
                ],
                indent=2,
            )
        )
        return True
    if args.collection_command == "refresh":
        options = {}
        if args.cache_path:
            options["cache_path"] = args.cache_path
        if args.collections:
            options["collection_ids"] = args.collections
        cache = refresh_collection_metadata(**options)
        print(
            json.dumps(
                {
                    "generated_at": cache.generated_at,
                    "backend_versions": cache.backend_versions,
                    "collections": {
                        collection_id: len(models)
                        for collection_id, models in cache.collections.items()
                    },
                },
                indent=2,
            )
        )
        return True
    if args.collection_command == "validate-cache":
        options = {"minimum_ratio": args.minimum_ratio}
        if args.cache_path:
            options["cache_path"] = args.cache_path
        print(json.dumps(validate_cache_exposure(**options), indent=2))
        return True

    parser.error("Choose one of: list, refresh, validate-cache.")
    return True


def main() -> None:
    """Example usage of VisionStudio for image classification tasks."""
    if _run_model_collections_cli(sys.argv[1:]):
        return

    print("Vision Studio - Image Classification Example")

    # Configuration
    num_classes = 10  # CIFAR-10 style
    image_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Create example dataset and dataloaders
    # Note: In a real scenario, you would have actual image files
    # Expected annotation file format:
    # {
    #     "images": [{"id": 0, "file_name": "image_0.jpg"}, ...],
    #     "categories": [{"id": 0, "name": "class_0"}, ...],
    #     "annotations": [{"image_id": 0, "category_id": 0}, ...]
    # }

    # Create augmentation pipeline for classification
    _train_transforms = Compose(
        [
            ImageToArray(),  # Convert PIL to numpy if needed
            HorizontalFlip(),  # Random horizontal flip
            Resize(image_size, image_size),  # Resize to model input size
            ToTensor(),  # Convert to tensor (0-1 range)
            Normalize(),  # Normalize using ImageNet stats
        ]
    )

    _val_transforms = Compose(
        [
            ImageToArray(),
            Resize(image_size, image_size),
            ToTensor(),
            Normalize(),
        ]
    )

    # Initialize model
    model = ImageClassifier(in_channels=3, num_classes=num_classes)
    print(f"Model created with {num_classes} classes")

    # Create optimizer
    optimizer = Adam(model.parameters(), lr=0.001)

    # Create trainer
    trainer = VisionTrainer(optimizer=optimizer, device=device)

    # Create inference engine
    inference = SimpleInference(device=device)

    print("\nVisionStudio Classification Pipeline initialized!")
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Trainer: {trainer.__class__.__name__}")
    print(f"Inference: {inference.__class__.__name__}")

    # Example of model configuration
    print(f"\nModel config: {model.get_config()}")

    # Print usage instructions
    print("\n=== Usage Instructions ===")
    print("""
To use VisionStudio for your classification task:

1. Prepare your dataset with the following structure:
   ```
   dataset/
   ├── images/
   │   ├── image_0.jpg
   │   ├── image_1.jpg
   │   └── ...
   └── annotations.json
   ```

2. Create dataset and dataloaders:
   ```python
   from vision_studio.dataset import ImageClassificationDataset
   from vision_studio.data_loader import SimpleDataLoader

   dataset = ImageClassificationDataset(
       images_dir='dataset/images',
       annotation_file='dataset/annotations.json'
   )
   
   train_loader = SimpleDataLoader(dataset, batch_size=32, shuffle=True)
   ```

3. Train the model:
   ```python
   history = trainer.fit(
       model=model,
       train_loader=train_loader,
       val_loader=val_loader,
       num_epochs=10
   )
   ```

4. Run inference:
   ```python
   results = inference.predict(model, test_loader)
   print(results['metrics'])  # accuracy, precision, recall, f1
   ```

5. Save/load models:
   ```python
   model.save_weights('path/to/model.pt')
   model.load_weights('path/to/model.pt')
   ```
    """)


if __name__ == "__main__":
    main()
