from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from .types import TrackedPose

KEYPOINT_COUNT = 17
FEATURE_DIM = KEYPOINT_COUNT * 5 + 4


def pose_to_feature(pose: TrackedPose, previous: TrackedPose | None, min_conf: float) -> np.ndarray:
    """Convert pixels into translation/scale-invariant pose and motion features."""
    box = pose.bbox.astype(np.float32)
    width = max(float(box[2] - box[0]), 1.0)
    height = max(float(box[3] - box[1]), 1.0)
    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)

    keypoints = pose.keypoints.astype(np.float32).copy()
    visible = keypoints[:, 2] >= min_conf
    normalized = (keypoints[:, :2] - center) / height
    normalized[~visible] = 0.0

    velocity = np.zeros((KEYPOINT_COUNT, 2), dtype=np.float32)
    if previous is not None:
        prev_box = previous.bbox.astype(np.float32)
        prev_height = max(float(prev_box[3] - prev_box[1]), 1.0)
        both_visible = visible & (previous.keypoints[:, 2] >= min_conf)
        velocity[both_visible] = (
            keypoints[both_visible, :2] - previous.keypoints[both_visible, :2]
        ) / prev_height

    h, w = pose.frame_shape
    geometry = np.array(
        [width / height, center[0] / w, center[1] / h, (width * height) / (w * h)],
        dtype=np.float32,
    )
    return np.concatenate(
        [normalized.reshape(-1), keypoints[:, 2], velocity.reshape(-1), geometry]
    ).astype(np.float32)


@dataclass(slots=True)
class TrackWindow:
    features: np.ndarray
    timestamps: np.ndarray
    valid_length: int


class SequenceStore:
    def __init__(self, length: int, minimum_frames: int, min_conf: float, stale_seconds: float):
        self.length = length
        self.minimum_frames = minimum_frames
        self.min_conf = min_conf
        self.stale_seconds = stale_seconds
        self._poses: dict[int, deque[TrackedPose]] = defaultdict(lambda: deque(maxlen=length))
        self._features: dict[int, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=length))

    def update(self, pose: TrackedPose) -> TrackWindow | None:
        history = self._poses[pose.track_id]
        previous = history[-1] if history else None
        feature = pose_to_feature(pose, previous, self.min_conf)
        history.append(pose)
        self._features[pose.track_id].append(feature)
        self.prune(pose.timestamp)
        if len(history) < self.minimum_frames:
            return None
        values = np.stack(self._features[pose.track_id])
        times = np.array([item.timestamp for item in history], dtype=np.float64)
        if len(values) < self.length:
            pad = self.length - len(values)
            values = np.pad(values, ((pad, 0), (0, 0)), mode="edge")
            times = np.pad(times, (pad, 0), mode="edge")
        return TrackWindow(values, times, len(history))

    def prune(self, now: float) -> None:
        stale = [track_id for track_id, poses in self._poses.items() if now - poses[-1].timestamp > self.stale_seconds]
        for track_id in stale:
            self._poses.pop(track_id, None)
            self._features.pop(track_id, None)
