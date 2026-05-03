from __future__ import annotations

import torch
from torch.optim import Adam
from torchvision.transforms import Compose

from vision_studio.augmentation import HorizontalFlip, Resize
from vision_studio.inference import ClassificationInference
from vision_studio.models import ImageClassifier
from vision_studio.trainer import ClassificationTrainer
from vision_studio.transforms import ImageToArray, Normalize, ToTensor


def main() -> None:
    """Example usage of VisionStudio for image classification tasks."""
    print("Vision Studio - Image Classification Example")

    # Configuration
    num_classes = 10  # CIFAR-10 style
    image_size = 32
    batch_size = 32
    num_epochs = 5
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
    train_transforms = Compose(
        [
            ImageToArray(),  # Convert PIL to numpy if needed
            HorizontalFlip(),  # Random horizontal flip
            Resize(image_size, image_size),  # Resize to model input size
            ToTensor(),  # Convert to tensor (0-1 range)
            Normalize(),  # Normalize using ImageNet stats
        ]
    )

    val_transforms = Compose(
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
    trainer = ClassificationTrainer(optimizer=optimizer, device=device)

    # Create inference engine
    inference = ClassificationInference(device=device)

    print("\nVisionStudio Classification Pipeline initialized!")
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Trainer: {trainer.__class__.__name__}")
    print(f"Inference: {inference.__class__.__name__}")

    # Example of model configuration
    print(f"\nModel config: {model.get_config()}")

    # Print usage instructions
    print("\n=== Usage Instructions ===")
    print(
        """
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
    """
    )


if __name__ == "__main__":
    main()
