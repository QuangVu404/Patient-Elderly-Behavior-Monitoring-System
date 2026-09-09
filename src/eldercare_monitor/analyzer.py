from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .classifier import FallClassifier
from .config import AnalysisConfig
from .features import FEATURE_DIM, TrackWindow
from .types import RiskEvent


@dataclass(slots=True)
class _TrackRiskState:
    consecutive_fall: int = 0
    last_event_at: float = float("-inf")
    still_since: float | None = None
    last_seen: float = float("-inf")


class RiskAnalyzer:
    def __init__(self, config: AnalysisConfig, classifier: FallClassifier):
        self.config = config
        self.classifier = classifier
        self.states: dict[int, _TrackRiskState] = defaultdict(_TrackRiskState)

    def update(self, track_id: int, window: TrackWindow, now: float) -> list[RiskEvent]:
        state = self.states[track_id]
        state.last_seen = now
        probability = self.classifier.predict(window.features)
        state.consecutive_fall = state.consecutive_fall + 1 if probability >= self.config.fall_threshold else 0
        events: list[RiskEvent] = []
        cooled_down = now - state.last_event_at >= self.config.cooldown_seconds

        if state.consecutive_fall >= self.config.confirm_frames and cooled_down:
            events.append(RiskEvent("fall", track_id, probability, now, "Phat hien nguy co nga"))
            state.last_event_at = now
            state.consecutive_fall = 0
            cooled_down = False

        valid = window.features[-min(window.valid_length, 5) :]
        velocity_start = 17 * 3
        velocity = valid[:, velocity_start : velocity_start + 17 * 2].reshape(-1, 17, 2)
        confidences = valid[:, 17 * 2 : 17 * 3]
        motion = float(np.median(np.linalg.norm(velocity, axis=2)))
        aspect = float(np.median(valid[:, FEATURE_DIM - 4]))
        visible_keypoints = float(np.median(np.sum(confidences >= 0.25, axis=1)))
        is_still_and_horizontal = (
            motion < self.config.immobility_motion_threshold
            and aspect >= self.config.horizontal_aspect_ratio
            and visible_keypoints >= 5
        )
        if is_still_and_horizontal:
            state.still_since = now if state.still_since is None else state.still_since
        else:
            state.still_since = None
        if (
            state.still_since is not None
            and now - state.still_since >= self.config.immobility_seconds
            and cooled_down
        ):
            events.append(RiskEvent("immobility", track_id, 1.0 - motion, now, "Nam bat dong keo dai"))
            state.last_event_at = now
            state.still_since = now
        return events

    def prune(self, now: float, stale_seconds: float) -> None:
        stale = [
            track_id
            for track_id, state in self.states.items()
            if now - state.last_seen > stale_seconds
        ]
        for track_id in stale:
            self.states.pop(track_id, None)
