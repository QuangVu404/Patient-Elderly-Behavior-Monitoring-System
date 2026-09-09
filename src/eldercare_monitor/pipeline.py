from __future__ import annotations

import time
from collections import defaultdict, deque

import cv2
import numpy as np

from .analyzer import RiskAnalyzer
from .classifier import HeuristicFallClassifier, OpenVinoFallClassifier
from .config import AppConfig
from .detector import UltralyticsPoseTracker
from .events import EventSink
from .features import SequenceStore
from .types import RiskEvent, TrackedPose


class MonitorPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.tracker = UltralyticsPoseTracker(config.detector)
        self.store = SequenceStore(
            config.sequence.length,
            config.sequence.minimum_frames,
            config.sequence.keypoint_confidence,
            config.sequence.stale_track_seconds,
        )
        classifier = (
            OpenVinoFallClassifier(config.analysis.classifier_model)
            if config.analysis.classifier_model
            else HeuristicFallClassifier()
        )
        self.analyzer = RiskAnalyzer(config.analysis, classifier)
        self.sink = EventSink(config.output)
        self.recent_events: dict[int, deque[RiskEvent]] = defaultdict(lambda: deque(maxlen=1))

    def run(self) -> None:
        cap = cv2.VideoCapture(self.config.video.source)
        if not cap.isOpened():
            raise RuntimeError(f"Khong mo duoc nguon video: {self.config.video.source}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.video.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.video.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame_index, fps, last_tick = 0, 0.0, time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % self.config.video.process_every_n_frames:
                    continue
                now = time.monotonic()
                poses = self.tracker(frame, now)
                self.analyzer.prune(now, self.config.sequence.stale_track_seconds)
                for pose in poses:
                    window = self.store.update(pose)
                    if window is not None:
                        for event in self.analyzer.update(pose.track_id, window, now):
                            self.recent_events[pose.track_id].append(event)
                            self.sink.emit(event, frame)
                elapsed = time.perf_counter() - last_tick
                instant_fps = 1.0 / max(elapsed, 1e-6)
                fps = instant_fps if fps == 0 else 0.90 * fps + 0.10 * instant_fps
                last_tick = time.perf_counter()
                annotated = self._draw(frame, poses, fps)
                if self.config.video.display:
                    cv2.imshow("Eldercare Monitor - q to quit", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _draw(self, frame: np.ndarray, poses: list[TrackedPose], fps: float) -> np.ndarray:
        output = frame.copy()
        for pose in poses:
            x1, y1, x2, y2 = pose.bbox.astype(int)
            event = self.recent_events[pose.track_id][-1] if self.recent_events[pose.track_id] else None
            alert = event is not None and pose.timestamp - event.timestamp < 3.0
            color = (0, 0, 255) if alert else (40, 210, 40)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"ID {pose.track_id}" + (f" | {event.kind.upper()}" if alert else "")
            cv2.putText(output, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            for x, y, confidence in pose.keypoints:
                if confidence >= self.config.sequence.keypoint_confidence:
                    cv2.circle(output, (int(x), int(y)), 2, color, -1)
        cv2.putText(output, f"FPS {fps:.1f}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)
        return output
