from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(slots=True)
class TrackedPose:
    track_id: int
    bbox: np.ndarray  # xyxy, pixels
    keypoints: np.ndarray  # [17, 3], x/y pixels + confidence
    confidence: float
    timestamp: float
    frame_shape: tuple[int, int]  # height, width


@dataclass(slots=True)
class RiskEvent:
    kind: Literal["fall", "immobility"]
    track_id: int
    probability: float
    timestamp: float
    message: str

