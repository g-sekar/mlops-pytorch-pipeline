"""
model.py - PyTorch CNN model definition for CIFAR-10 classification.

Supports ResNet-18 (fine-tuned from torchvision) and a custom lightweight CNN.
"""

import torch
import torch.nn as nn
from torchvision import models


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class LightweightCNN(nn.Module):
    """
    A simple custom CNN for CIFAR-10 (32x32 images, 10 classes).
    Useful for quick experiments without pretrained weights.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 16x16
            nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 8x8
            nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 4x4
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(
    architecture: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = True,
) -> nn.Module:
    """
    Factory function to create a model by architecture name.

    Args:
        architecture: One of 'resnet18', 'resnet34', 'resnet50', 'cnn'.
        num_classes:  Number of output classes (default: 10 for CIFAR-10).
        pretrained:   Whether to load ImageNet pretrained weights (for ResNet).

    Returns:
        A PyTorch nn.Module ready for training.
    """
    architecture = architecture.lower()

    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        # Replace the final FC layer to match num_classes
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif architecture == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif architecture == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif architecture == "cnn":
        return LightweightCNN(num_classes=num_classes)

    else:
        raise ValueError(
            f"Unknown architecture: '{architecture}'. "
            f"Choose from: resnet18, resnet34, resnet50, cnn."
        )


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    for arch in ["resnet18", "cnn"]:
        m = get_model(architecture=arch, num_classes=10, pretrained=False)
        dummy = torch.randn(4, 3, 32, 32)
        out = m(dummy)
        print(f"{arch}: output shape={out.shape}, params={count_parameters(m):,}")
