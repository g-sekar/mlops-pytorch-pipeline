"""
dataset.py - CIFAR-10 data loading with torchvision.

Provides transforms and DataLoader setup for training and validation.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-10 channel-wise mean and std (computed over the training set)
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Return data augmentation + normalization transforms.

    Training transforms include random horizontal flip and random crop
    for regularization. Validation transforms only normalize.

    Args:
        train: If True, return training transforms; else validation transforms.

    Returns:
        A torchvision.transforms.Compose pipeline.
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 train and validation DataLoaders.

    Downloads the dataset automatically if not present at data_dir.

    Args:
        data_dir:    Root directory where CIFAR-10 data is stored/downloaded.
        batch_size:  Number of samples per batch.
        num_workers: Number of subprocesses for data loading.
        pin_memory:  If True, pin memory for faster GPU transfer.

    Returns:
        A tuple of (train_loader, val_loader).
    """
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader


def get_single_image_transform() -> transforms.Compose:
    """
    Return the inference-time transform for a single PIL image.
    Used by the serving endpoint to preprocess uploaded images.
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


if __name__ == "__main__":
    # Quick sanity check
    train_loader, val_loader = get_dataloaders(data_dir="./data", batch_size=64)
    images, labels = next(iter(train_loader))
    print(f"Train batch: images={images.shape}, labels={labels.shape}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
