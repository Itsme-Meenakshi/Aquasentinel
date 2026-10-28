"""Single-image inference for the camera recognition module."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pufferfish.camera.dataset import CLASS_NAMES, IDX_TO_CLASS
from pufferfish.camera.model import (
    IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, build_model, load_model
)


class PufferfishClassifier:
    """Wraps a trained model for single-image or batch inference.

    Designed for CPU use. Returns class name and confidence score.
    """

    def __init__(self, model_dir: Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.meta = load_model(model_dir, device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE + 32),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.trained = self.meta.get("trained", False)
        self.class_names = self.meta.get("class_names", CLASS_NAMES)

    @property
    def is_trained(self) -> bool:
        return self.trained

    def predict_image(self, image_path: Path | str) -> dict:
        """Classify a single image.

        Returns dict with: class_name, confidence, probabilities, is_pufferfish.
        """
        image_path = Path(image_path)
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        return {
            "image": str(image_path),
            "class_name": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                self.class_names[i]: float(probs[i])
                for i in range(len(self.class_names))
            },
            "is_pufferfish": self.class_names[pred_idx] == "pufferfish",
        }

    def predict_batch(self, image_paths: list[Path | str]) -> list[dict]:
        """Classify multiple images. Returns a list of prediction dicts."""
        results = []
        for p in image_paths:
            try:
                results.append(self.predict_image(p))
            except Exception as e:
                results.append({
                    "image": str(p),
                    "error": str(e),
                    "class_name": None,
                    "confidence": 0.0,
                    "is_pufferfish": False,
                })
        return results

    def status(self) -> dict:
        """Return model status information."""
        return {
            "trained": self.trained,
            "architecture": self.meta.get("architecture", "unknown"),
            "environment": self.meta.get("environment", "unknown"),
            "class_names": self.class_names,
            "evaluation": self.meta.get("evaluation"),
            "notes": self.meta.get("notes", ""),
        }
