"""
train.py - Training loop for CIFAR-10 image classification.

Reads hyperparameters from configs/training_config.yaml,
logs metrics as JSON lines to stdout, saves checkpoints,
and supports early stopping.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Allow running from repo root or from src/
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloaders  # noqa: E402
from model import get_model, count_parameters  # noqa: E402


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def log(data: dict) -> None:
    """Print a JSON-lines log entry to stdout (structured logging)."""
    print(json.dumps(data), flush=True)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> tuple[float, float]:
    """
    Run one full training epoch.

    Returns:
        (avg_loss, accuracy) over the entire training set.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on a validation/test DataLoader.

    Returns:
        (avg_loss, accuracy) over the entire validation set.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Load configuration
    # ------------------------------------------------------------------ #
    config_path = Path("/app/configs/training_config.yaml")
    if not config_path.exists():
        config_path = Path("configs/training_config.yaml")
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"

    config = load_config(str(config_path))
    log({"event": "config_loaded", "path": str(config_path)})

    # ------------------------------------------------------------------ #
    # 2. Device setup
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log({"event": "device_selected", "device": str(device)})

    # ------------------------------------------------------------------ #
    # 3. Model
    # ------------------------------------------------------------------ #
    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"].get("pretrained", True),
    ).to(device)
    log({
        "event": "model_created",
        "architecture": config["model"]["architecture"],
        "trainable_params": count_parameters(model),
    })

    # ------------------------------------------------------------------ #
    # 4. Data
    # ------------------------------------------------------------------ #
    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"].get("num_workers", 2),
    )
    log({
        "event": "data_loaded",
        "dataset": config["data"]["dataset"],
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
    })

    # ------------------------------------------------------------------ #
    # 5. Optimizer, scheduler, loss
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
    )
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------ #
    # 6. Training loop with early stopping
    # ------------------------------------------------------------------ #
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / config["output"]["model_name"]

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        log({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "lr": round(scheduler.get_last_lr()[0], 6),
        })

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "config": config,
                },
                best_checkpoint_path,
            )
            log({"event": "checkpoint_saved", "path": str(best_checkpoint_path)})
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log({"event": "early_stopping", "epoch": epoch + 1, "patience": patience})
                break

    log({
        "event": "training_complete",
        "best_val_loss": round(best_val_loss, 4),
        "checkpoint": str(best_checkpoint_path),
    })


if __name__ == "__main__":
    main()
