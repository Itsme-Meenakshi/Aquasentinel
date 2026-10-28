"""Unit tests for src/pufferfish/camera/ — dataset, model, inference, video.

These tests run entirely on CPU with stub/random weights (no real training data).
They verify the code structure, shapes, data types, and API contracts.
"""
from __future__ import annotations
import csv
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


# ── helpers ───────────────────────────────────────────────────────────────────

def make_dummy_image(path: Path, size: tuple[int, int] = (128, 128), color=(100, 150, 200)):
    """Create a small dummy RGB image at path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path)
    return path


def make_dummy_dataset(root: Path, n_puffer: int = 8, n_other: int = 12) -> Path:
    """Create a minimal class-structured image directory."""
    for i in range(n_puffer):
        make_dummy_image(root / "pufferfish" / f"puffer_{i:03d}.jpg",
                         color=(255, 200, i * 10 % 255))
    for i in range(n_other):
        make_dummy_image(root / "other_fish" / f"other_{i:03d}.jpg",
                         color=(i * 15 % 255, 100, 50))
    return root


def make_dummy_model_dir(out_dir: Path, arch: str = "mobilenet_v3_small") -> Path:
    """Create a minimal stub model directory."""
    import torch
    from pufferfish.camera.model import (
        build_model, CLASS_NAMES, CLASS_TO_IDX, INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(arch, num_classes=2, pretrained=False)
    torch.save(model.state_dict(), out_dir / "model.pth")
    meta = {
        "architecture": arch,
        "environment": "boat_deck",
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "input_size": INPUT_SIZE,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "training_history": None,
        "evaluation": None,
        "notes": "test stub",
        "trained": False,
    }
    with open(out_dir / "model_meta.json", "w") as f:
        json.dump(meta, f)
    return out_dir


# ═════════════════════════════════════════════════════════════════════════════
# Dataset tests
# ═════════════════════════════════════════════════════════════════════════════

class TestDataset:
    def test_class_names_order(self):
        from pufferfish.camera.dataset import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS
        assert CLASS_NAMES == ["other_fish", "pufferfish"]
        assert CLASS_TO_IDX["pufferfish"] == 1
        assert IDX_TO_CLASS[0] == "other_fish"

    def test_validate_image_ok(self, tmp_path):
        from pufferfish.camera.dataset import validate_image
        p = make_dummy_image(tmp_path / "test.jpg")
        ok, msg = validate_image(p)
        assert ok, msg

    def test_validate_image_bad_extension(self, tmp_path):
        from pufferfish.camera.dataset import validate_image
        bad = tmp_path / "file.txt"
        bad.write_text("not an image")
        ok, msg = validate_image(bad)
        assert not ok
        assert "unsupported" in msg

    def test_validate_image_missing(self, tmp_path):
        from pufferfish.camera.dataset import validate_image
        ok, msg = validate_image(tmp_path / "ghost.jpg")
        assert not ok

    def test_file_hash_deterministic(self, tmp_path):
        from pufferfish.camera.dataset import file_hash
        p = make_dummy_image(tmp_path / "img.jpg")
        h1 = file_hash(p)
        h2 = file_hash(p)
        assert h1 == h2
        assert len(h1) == 16

    def test_scan_directory(self, tmp_path):
        from pufferfish.camera.dataset import scan_directory
        make_dummy_dataset(tmp_path / "raw", n_puffer=5, n_other=10)
        report = scan_directory(tmp_path / "raw")
        assert report["valid_images"] == 15
        assert report["classes"]["pufferfish"]["count"] == 5
        assert report["classes"]["other_fish"]["count"] == 10
        assert report["invalid_images"] == []

    def test_scan_directory_missing_class(self, tmp_path):
        from pufferfish.camera.dataset import scan_directory
        (tmp_path / "pufferfish").mkdir()
        report = scan_directory(tmp_path)
        assert report["classes"]["other_fish"].get("missing_dir", False)

    def test_split_dataset_proportions(self, tmp_path):
        from pufferfish.camera.dataset import split_dataset
        raw = tmp_path / "raw"
        make_dummy_dataset(raw, n_puffer=20, n_other=20)
        out = tmp_path / "split"
        splits = split_dataset(raw, out, train_frac=0.7, val_frac=0.15, test_frac=0.15)
        # Each class: 20 images → 14 train, 3 val, 3 test (approx)
        for cls in ("pufferfish", "other_fish"):
            total = splits["train"][cls] + splits["val"][cls] + splits["test"][cls]
            assert total == 20, f"Total mismatch for {cls}: {total}"

    def test_split_dataset_files_exist(self, tmp_path):
        from pufferfish.camera.dataset import split_dataset
        raw = tmp_path / "raw"
        make_dummy_dataset(raw, n_puffer=10, n_other=10)
        out = tmp_path / "split"
        split_dataset(raw, out)
        for split in ("train", "val", "test"):
            for cls in ("pufferfish", "other_fish"):
                d = out / split / cls
                assert d.exists(), f"{d} missing"

    def test_class_balance_report(self, tmp_path):
        from pufferfish.camera.dataset import class_balance_report
        make_dummy_dataset(tmp_path, n_puffer=5, n_other=20)
        bal = class_balance_report(tmp_path)
        assert bal["counts"]["pufferfish"] == 5
        assert bal["counts"]["other_fish"] == 20
        assert bal["imbalance_ratio"] == 4.0
        assert bal["total"] == 25


# ═════════════════════════════════════════════════════════════════════════════
# Model tests
# ═════════════════════════════════════════════════════════════════════════════

class TestModel:
    def test_build_mobilenet_v3_small(self):
        from pufferfish.camera.model import build_model
        model = build_model("mobilenet_v3_small", num_classes=2, pretrained=False)
        # Check that only classifier parameters require grad
        feature_params_trainable = [
            p.requires_grad for p in model.features.parameters()
        ]
        assert all(not t for t in feature_params_trainable), \
            "Feature layers should be frozen"

    def test_build_resnet18(self):
        from pufferfish.camera.model import build_model
        model = build_model("resnet18", num_classes=2, pretrained=False)
        assert model.fc.out_features == 2

    def test_build_unsupported_arch(self):
        from pufferfish.camera.model import build_model
        with pytest.raises(ValueError, match="Unsupported"):
            build_model("vgg16", num_classes=2)

    def test_model_forward_pass(self):
        import torch
        from pufferfish.camera.model import build_model, INPUT_SIZE
        model = build_model("mobilenet_v3_small", num_classes=2, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"

    def test_get_transforms_shapes(self):
        import torch
        from pufferfish.camera.model import get_transforms, INPUT_SIZE
        for split in ("train", "val", "test"):
            tfm = get_transforms(split)
            img = Image.new("RGB", (256, 256), color=(100, 100, 100))
            t = tfm(img)
            assert t.shape == (3, INPUT_SIZE, INPUT_SIZE), \
                f"Shape mismatch for {split}: {t.shape}"

    def test_save_and_load_model(self, tmp_path):
        import torch
        from pufferfish.camera.model import (
            build_model, save_model_artifacts, load_model, CLASS_NAMES
        )
        model = build_model("mobilenet_v3_small", num_classes=2, pretrained=False)
        out_dir = tmp_path / "model_out"
        save_model_artifacts(model, history={"best_epoch": 0}, eval_results=None,
                             arch="mobilenet_v3_small", environment="boat_deck",
                             output_dir=out_dir, notes="test")
        assert (out_dir / "model.pth").exists()
        assert (out_dir / "model_meta.json").exists()

        loaded_model, meta = load_model(out_dir, device="cpu")
        assert meta["architecture"] == "mobilenet_v3_small"
        assert meta["class_names"] == CLASS_NAMES

    def test_evaluate_model_structure(self, tmp_path):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from pufferfish.camera.model import build_model, evaluate_model, CLASS_NAMES
        model = build_model("mobilenet_v3_small", num_classes=2, pretrained=False)
        model.eval()
        # Fake a dataloader
        n = 10
        x = torch.randn(n, 3, 224, 224)
        y = torch.randint(0, 2, (n,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        result = evaluate_model(model, loader, device="cpu")
        assert "accuracy" in result
        assert "macro_f1" in result
        assert "confusion_matrix" in result
        assert len(result["class_names"]) == 2
        assert result["n_samples"] == n


# ═════════════════════════════════════════════════════════════════════════════
# Inference tests
# ═════════════════════════════════════════════════════════════════════════════

class TestInference:
    def test_classifier_loads(self, tmp_path):
        from pufferfish.camera.inference import PufferfishClassifier
        model_dir = make_dummy_model_dir(tmp_path / "model")
        clf = PufferfishClassifier(model_dir, device="cpu")
        assert clf is not None
        assert not clf.is_trained

    def test_predict_image_returns_correct_keys(self, tmp_path):
        from pufferfish.camera.inference import PufferfishClassifier
        model_dir = make_dummy_model_dir(tmp_path / "model")
        clf = PufferfishClassifier(model_dir, device="cpu")
        img_path = make_dummy_image(tmp_path / "fish.jpg")
        result = clf.predict_image(img_path)
        assert "class_name" in result
        assert "confidence" in result
        assert "is_pufferfish" in result
        assert "probabilities" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["class_name"] in ("pufferfish", "other_fish")
        assert isinstance(result["is_pufferfish"], bool)

    def test_predict_batch(self, tmp_path):
        from pufferfish.camera.inference import PufferfishClassifier
        model_dir = make_dummy_model_dir(tmp_path / "model")
        clf = PufferfishClassifier(model_dir, device="cpu")
        paths = [make_dummy_image(tmp_path / f"fish_{i}.jpg") for i in range(5)]
        results = clf.predict_batch(paths)
        assert len(results) == 5
        for r in results:
            assert "class_name" in r

    def test_predict_missing_file_handled(self, tmp_path):
        from pufferfish.camera.inference import PufferfishClassifier
        model_dir = make_dummy_model_dir(tmp_path / "model")
        clf = PufferfishClassifier(model_dir, device="cpu")
        results = clf.predict_batch([tmp_path / "nonexistent.jpg"])
        assert results[0].get("error") is not None

    def test_status_dict(self, tmp_path):
        from pufferfish.camera.inference import PufferfishClassifier
        model_dir = make_dummy_model_dir(tmp_path / "model")
        clf = PufferfishClassifier(model_dir, device="cpu")
        s = clf.status()
        assert s["architecture"] == "mobilenet_v3_small"
        assert s["environment"] == "boat_deck"
        assert not s["trained"]


# ═════════════════════════════════════════════════════════════════════════════
# Video tests
# ═════════════════════════════════════════════════════════════════════════════

class TestVideo:
    def _make_dummy_video(self, path: Path, n_frames: int = 30, fps: int = 10) -> Path:
        """Create a minimal synthetic video using OpenCV."""
        import cv2
        path.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (128, 128),
        )
        for i in range(n_frames):
            frame = np.full((128, 128, 3), (i * 8 % 255, 100, 50), dtype=np.uint8)
            out.write(frame)
        out.release()
        return path

    def test_get_video_info(self, tmp_path):
        from pufferfish.camera.video import get_video_info
        vid = self._make_dummy_video(tmp_path / "test.mp4", n_frames=30, fps=10)
        info = get_video_info(vid)
        assert "fps" in info
        assert "frame_count" in info
        assert info.get("error") is None

    def test_sample_video_frames(self, tmp_path):
        from pufferfish.camera.video import sample_video_frames
        vid = self._make_dummy_video(tmp_path / "test.mp4", n_frames=60, fps=10)
        frames = sample_video_frames(vid, sample_interval_sec=0.5, max_frames=50)
        assert len(frames) > 0
        for frame_num, ts, frame_bgr in frames:
            assert isinstance(frame_num, int)
            assert ts >= 0.0
            assert frame_bgr.shape[2] == 3  # BGR channels

    def test_classify_video_returns_predictions(self, tmp_path):
        from pufferfish.camera.inference import PufferfishClassifier
        from pufferfish.camera.video import classify_video
        model_dir = make_dummy_model_dir(tmp_path / "model")
        clf = PufferfishClassifier(model_dir, device="cpu")
        vid = self._make_dummy_video(tmp_path / "clip.mp4", n_frames=30, fps=10)
        preds = classify_video(clf, vid, sample_interval_sec=0.5, max_frames=20,
                               temp_dir=tmp_path / "frames")
        assert len(preds) > 0
        for p in preds:
            assert p.class_name in ("pufferfish", "other_fish")
            assert 0.0 <= p.confidence <= 1.0

    def test_group_candidate_events_empty(self):
        from pufferfish.camera.video import group_candidate_events, FramePrediction
        preds = [
            FramePrediction(i*10, float(i), "other_fish", 0.9, False)
            for i in range(10)
        ]
        events = group_candidate_events(preds)
        assert events == []

    def test_group_candidate_events_grouped(self):
        from pufferfish.camera.video import group_candidate_events, FramePrediction
        preds = [
            FramePrediction(0,  0.0, "pufferfish", 0.9, True),
            FramePrediction(10, 1.0, "pufferfish", 0.85, True),
            FramePrediction(20, 2.0, "pufferfish", 0.8, True),
            FramePrediction(30, 15.0, "other_fish", 0.6, False),  # gap
            FramePrediction(40, 16.0, "pufferfish", 0.75, True),
            FramePrediction(50, 17.0, "pufferfish", 0.7, True),
        ]
        events = group_candidate_events(preds, gap_tolerance_sec=3.0, min_positive_frames=2)
        assert len(events) == 2
        assert events[0].n_positive_frames == 3
        assert events[1].n_positive_frames == 2

    def test_export_predictions_csv(self, tmp_path):
        from pufferfish.camera.video import FramePrediction, export_predictions_csv
        preds = [
            FramePrediction(0, 0.0, "pufferfish", 0.9, True),
            FramePrediction(10, 1.0, "other_fish", 0.8, False),
        ]
        out = tmp_path / "preds.csv"
        export_predictions_csv(preds, out)
        assert out.exists()
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["class_name"] == "pufferfish"

    def test_export_events_csv(self, tmp_path):
        from pufferfish.camera.video import CandidateEvent, export_events_csv
        events = [
            CandidateEvent(1, 0, 20, 0.0, 2.0, 3, 0.9, 0.85, "accepted", "test note")
        ]
        out = tmp_path / "events.csv"
        export_events_csv(events, out)
        assert out.exists()
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["review_status"] == "accepted"
        assert rows[0]["reviewer_notes"] == "test note"
