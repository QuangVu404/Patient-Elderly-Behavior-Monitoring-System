from __future__ import annotations

import time

import numpy as np

from .config import DetectorConfig
from .types import TrackedPose


class UltralyticsPoseTracker:
    """YOLO pose detector with ByteTrack state persisted across consecutive frames."""

    def __init__(self, config: DetectorConfig):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install runtime dependencies: pip install -e .") from exc
        self.config = config
        self.model = YOLO(config.model)

    def __call__(self, frame: np.ndarray, timestamp: float | None = None) -> list[TrackedPose]:
        timestamp = time.monotonic() if timestamp is None else timestamp
        result = self.model.track(
            frame,
            persist=True,
            tracker=self.config.tracker,
            device=self.config.device,
            imgsz=self.config.image_size,
            conf=self.config.confidence,
            iou=self.config.iou,
            classes=[0],
            verbose=False,
        )[0]
        if result.boxes is None or result.keypoints is None or result.boxes.id is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy()
        frame_shape = (frame.shape[0], frame.shape[1])
        return [
            TrackedPose(int(track_id), box, points, float(score), timestamp, frame_shape)
            for box, track_id, score, points in zip(boxes, ids, scores, keypoints, strict=True)
        ]

