"""Dataset loading, validation, and splitting for the camera recognition module.

Design choices:
- Boat/deck and underwater datasets are kept separate (different directories, separate splits).
- Raw and processed data are in different directories.
- Splits are by source image to prevent data leakage (no augmented copy can appear
  in both train and test).
- Reproducible seeds throughout.
"""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Class labels: fixed order for all models
CLASS_NAMES = ["other_fish", "pufferfish"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
ENVIRONMENTS = ("boat_deck", "underwater")


def validate_image(path: Path) -> tuple[bool, str]:
    """Check that a file is a valid, loadable image."""
    if not path.exists():
        return False, "file not found"
    if path.suffix.lower() not in VALID_EXTENSIONS:
        return False, f"unsupported extension: {path.suffix}"
    try:
        img = Image.open(path)
        img.verify()
        return True, "ok"
    except Exception as e:
        return False, f"corrupt or unreadable: {e}"


def file_hash(path: Path) -> str:
    """SHA-256 hash of an image file (first 16 hex chars) for duplicate detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def scan_directory(data_dir: Path) -> dict:
    """Scan a class-structured image directory and report statistics.

    Expected structure:
        data_dir/
            pufferfish/
                img1.jpg
                img2.jpg
            other_fish/
                img3.jpg
                img4.jpg

    Returns a dict with counts, file lists, and any validation issues.
    """
    report = {"directory": str(data_dir), "classes": {}, "total_images": 0,
              "valid_images": 0, "invalid_images": [], "duplicates": []}
    seen_hashes: dict[str, Path] = {}

    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            report["classes"][class_name] = {"count": 0, "missing_dir": True}
            continue
        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )
        valid, invalid = [], []
        for f in files:
            ok, msg = validate_image(f)
            if ok:
                h = file_hash(f)
                if h in seen_hashes:
                    report["duplicates"].append(
                        {"file": str(f), "duplicate_of": str(seen_hashes[h])}
                    )
                else:
                    seen_hashes[h] = f
                valid.append(f)
            else:
                invalid.append({"file": str(f), "reason": msg})
        report["classes"][class_name] = {"count": len(valid)}
        report["total_images"] += len(valid) + len(invalid)
        report["valid_images"] += len(valid)
        report["invalid_images"].extend(invalid)
    return report


def split_dataset(
    data_dir: Path,
    output_dir: Path,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    """Split images into train/val/test sets by copying to output_dir/{split}/{class}/.

    Splits by source image (shuffle then partition). Returns split counts.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    splits = {"train": {}, "val": {}, "test": {}}

    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            for sp in splits:
                splits[sp][class_name] = 0
            continue
        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )
        rng.shuffle(files)
        n = len(files)
        n_train = max(1, int(n * train_frac)) if n > 0 else 0
        n_val = max(1, int(n * val_frac)) if n > 2 else 0
        n_test = n - n_train - n_val

        assignments = (
            [("train", files[:n_train])]
            + [("val", files[n_train : n_train + n_val])]
            + [("test", files[n_train + n_val :])]
        )
        for split_name, split_files in assignments:
            dest = output_dir / split_name / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dest / f.name)
            splits[split_name][class_name] = len(split_files)
    return splits


def save_manifest(data_dir: Path, manifest: dict, filename: str = "data_manifest.json"):
    """Save a source and licence manifest to the data directory."""
    out = data_dir / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return out


def class_balance_report(data_dir: Path) -> dict:
    """Return class counts and imbalance ratio for a split directory."""
    counts = {}
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if class_dir.exists():
            counts[class_name] = len([
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
            ])
        else:
            counts[class_name] = 0
    total = sum(counts.values())
    minority = min(counts.values()) if counts else 0
    majority = max(counts.values()) if counts else 0
    return {
        "counts": counts,
        "total": total,
        "imbalance_ratio": round(majority / minority, 2) if minority > 0 else float("inf"),
    }
