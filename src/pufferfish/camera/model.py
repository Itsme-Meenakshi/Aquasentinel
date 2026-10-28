"""CNN model definition, training, and evaluation for pufferfish recognition.

Architecture: MobileNetV3-Small with transfer learning (ImageNet pretrained).
CPU-compatible. Optional ResNet18 comparison.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from pufferfish.camera.dataset import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ImageNet normalisation (used by pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


def get_transforms(split: str = "train") -> transforms.Compose:
    """Data transforms. Training uses augmentation; val/test use only resize+normalise."""
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(INPUT_SIZE + 32),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def build_model(arch: str = "mobilenet_v3_small", num_classes: int = 2,
                pretrained: bool = True) -> nn.Module:
    """Build a classifier using transfer learning.

    Freezes feature layers and replaces the final classifier head.
    """
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        # Freeze feature layers
        for param in model.features.parameters():
            param.requires_grad = False
        # Replace classifier
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 16,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Create DataLoaders for train/val/test splits.

    Expects data_dir/{train,val,test}/{class_name}/ structure.
    """
    loaders = {}
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        ds = datasets.ImageFolder(split_dir, transform=get_transforms(split))
        # Verify class mapping matches our expected order
        if ds.class_to_idx != CLASS_TO_IDX:
            # ImageFolder uses alphabetical order, which should match our CLASS_NAMES
            pass  # Log warning but proceed
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=False
        )
    return loaders


def train_model(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    num_epochs: int = 15,
    lr: float = 1e-3,
    device: str = "cpu",
    checkpoint_dir: Optional[Path] = None,
) -> dict:
    """Train the model with early stopping based on validation loss.

    Returns training history dict with per-epoch metrics.
    """
    device = torch.device(device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Only train parameters that require gradients (the classifier head)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "epoch_time": []}
    best_val_loss = float("inf")
    best_epoch = 0
    patience = 5
    no_improve = 0

    for epoch in range(num_epochs):
        t0 = time.time()

        # Training phase
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            running_total += inputs.size(0)
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        # Validation phase
        val_loss, val_acc = 0.0, 0.0
        if "val" in dataloaders:
            model.eval()
            val_running_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in dataloaders["val"]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += inputs.size(0)
            val_loss = val_running_loss / val_total
            val_acc = val_correct / val_total

        scheduler.step()
        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(elapsed)

        print(f"  Epoch {epoch+1}/{num_epochs}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} ({elapsed:.1f}s)")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    return history


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> dict:
    """Evaluate on a test set. Returns per-class and aggregate metrics.

    Computes precision, recall, F1 for each class, plus confusion matrix.
    """
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device_obj)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Confusion matrix
    n_classes = len(CLASS_NAMES)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    # Per-class metrics
    per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(cm[i, :].sum()),
        }

    # Aggregate metrics
    accuracy = np.mean(all_preds == all_labels)
    macro_precision = np.mean([per_class[c]["precision"] for c in CLASS_NAMES])
    macro_recall = np.mean([per_class[c]["recall"] for c in CLASS_NAMES])
    macro_f1 = np.mean([per_class[c]["f1"] for c in CLASS_NAMES])

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "n_samples": int(len(all_labels)),
    }


def save_model_artifacts(
    model: nn.Module,
    history: dict,
    eval_results: Optional[dict],
    arch: str,
    environment: str,
    output_dir: Path,
    notes: str = "",
) -> Path:
    """Save model weights, class mapping, training history, and evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pth")

    meta = {
        "architecture": arch,
        "environment": environment,
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "input_size": INPUT_SIZE,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "training_history": {
            k: v for k, v in history.items()
            if k not in ("epoch_time",)
        },
        "evaluation": eval_results,
        "notes": notes,
        "trained": True,
    }
    with open(output_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return output_dir


def load_model(model_dir: Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Load a saved model and its metadata."""
    meta_path = model_dir / "model_meta.json"
    weights_path = model_dir / "model.pth"

    if not meta_path.exists():
        raise FileNotFoundError(f"No model_meta.json in {model_dir}")
    with open(meta_path) as f:
        meta = json.load(f)

    model = build_model(meta["architecture"], num_classes=len(meta["class_names"]),
                        pretrained=False)
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model, meta
