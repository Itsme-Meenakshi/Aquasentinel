"""Video inference for pufferfish detection.

Samples frames from a video file, classifies each with the trained model,
groups consecutive positive frames into candidate events, and exports results.
Does NOT claim bounding boxes, tracking, or individual fish detection.
Does NOT invent metadata (location, date, depth, gear) not present in the video.
"""
from __future__ import annotations
import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pufferfish.camera.inference import PufferfishClassifier


@dataclass
class FramePrediction:
    """A single frame prediction."""
    frame_number: int
    timestamp_sec: float
    class_name: str
    confidence: float
    is_pufferfish: bool


@dataclass
class CandidateEvent:
    """A group of consecutive positive frames forming a candidate detection event."""
    event_id: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    n_positive_frames: int
    max_confidence: float
    mean_confidence: float
    review_status: str = "unreviewed"  # unreviewed | accepted | rejected | uncertain
    reviewer_notes: str = ""


def sample_video_frames(
    video_path: Path | str,
    sample_interval_sec: float = 1.0,
    max_frames: int = 500,
) -> list[tuple[int, float, np.ndarray]]:
    """Extract frames from a video at regular intervals.

    Returns list of (frame_number, timestamp_sec, frame_bgr).
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = max(1, int(fps * sample_interval_sec))

    frames = []
    frame_idx = 0
    while frame_idx < total_frames and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / fps
        frames.append((frame_idx, timestamp, frame))
        frame_idx += interval_frames

    cap.release()
    return frames


def classify_video(
    classifier: PufferfishClassifier,
    video_path: Path | str,
    sample_interval_sec: float = 1.0,
    max_frames: int = 500,
    temp_dir: Optional[Path] = None,
) -> list[FramePrediction]:
    """Classify sampled frames from a video.

    Saves each frame to a temp file and runs single-image inference.
    Returns sorted list of FramePrediction objects.
    """
    import tempfile
    video_path = Path(video_path)
    temp_dir = temp_dir or Path(tempfile.mkdtemp(prefix="aqua_video_"))
    temp_dir.mkdir(parents=True, exist_ok=True)

    frames = sample_video_frames(video_path, sample_interval_sec, max_frames)
    predictions = []

    for frame_number, timestamp, frame_bgr in frames:
        # Save frame as temp image
        frame_path = temp_dir / f"frame_{frame_number:06d}.jpg"
        cv2.imwrite(str(frame_path), frame_bgr)

        # Classify
        result = classifier.predict_image(frame_path)
        predictions.append(FramePrediction(
            frame_number=frame_number,
            timestamp_sec=round(timestamp, 2),
            class_name=result["class_name"],
            confidence=result["confidence"],
            is_pufferfish=result["is_pufferfish"],
        ))

        # Clean up temp file
        frame_path.unlink(missing_ok=True)

    return sorted(predictions, key=lambda p: p.frame_number)


def group_candidate_events(
    predictions: list[FramePrediction],
    gap_tolerance_sec: float = 3.0,
    min_positive_frames: int = 2,
) -> list[CandidateEvent]:
    """Group consecutive positive frames into candidate events.

    Two positive frames within gap_tolerance_sec of each other are considered
    part of the same event. Events with fewer than min_positive_frames are discarded.
    """
    positive_preds = [p for p in predictions if p.is_pufferfish]
    if not positive_preds:
        return []

    # Sort by timestamp
    positive_preds.sort(key=lambda p: p.timestamp_sec)

    events: list[CandidateEvent] = []
    current_group: list[FramePrediction] = [positive_preds[0]]

    for i in range(1, len(positive_preds)):
        if positive_preds[i].timestamp_sec - current_group[-1].timestamp_sec <= gap_tolerance_sec:
            current_group.append(positive_preds[i])
        else:
            # Close current group
            if len(current_group) >= min_positive_frames:
                confs = [p.confidence for p in current_group]
                events.append(CandidateEvent(
                    event_id=len(events) + 1,
                    start_frame=current_group[0].frame_number,
                    end_frame=current_group[-1].frame_number,
                    start_time_sec=current_group[0].timestamp_sec,
                    end_time_sec=current_group[-1].timestamp_sec,
                    n_positive_frames=len(current_group),
                    max_confidence=max(confs),
                    mean_confidence=round(float(np.mean(confs)), 4),
                ))
            current_group = [positive_preds[i]]

    # Final group
    if len(current_group) >= min_positive_frames:
        confs = [p.confidence for p in current_group]
        events.append(CandidateEvent(
            event_id=len(events) + 1,
            start_frame=current_group[0].frame_number,
            end_frame=current_group[-1].frame_number,
            start_time_sec=current_group[0].timestamp_sec,
            end_time_sec=current_group[-1].timestamp_sec,
            n_positive_frames=len(current_group),
            max_confidence=max(confs),
            mean_confidence=round(float(np.mean(confs)), 4),
        ))

    return events


def export_predictions_csv(
    predictions: list[FramePrediction],
    output_path: Path,
) -> Path:
    """Export frame-level predictions to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_number", "timestamp_sec", "class_name", "confidence", "is_pufferfish"
        ])
        writer.writeheader()
        for p in predictions:
            writer.writerow(asdict(p))
    return output_path


def export_events_csv(
    events: list[CandidateEvent],
    output_path: Path,
) -> Path:
    """Export candidate events to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "event_id", "start_frame", "end_frame", "start_time_sec", "end_time_sec",
            "n_positive_frames", "max_confidence", "mean_confidence",
            "review_status", "reviewer_notes"
        ])
        writer.writeheader()
        for e in events:
            writer.writerow(asdict(e))
    return output_path


def get_video_info(video_path: Path | str) -> dict:
    """Get basic video metadata (duration, fps, frame count, resolution)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Cannot open: {video_path}"}
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_sec"] = round(info["frame_count"] / info["fps"], 2) if info["fps"] > 0 else 0
    cap.release()
    return info
