from __future__ import annotations

import time
from collections import defaultdict, deque

import cv2
import numpy as np

from .analyzer import RiskAnalyzer
from .classifier import HeuristicFallClassifier, OpenVinoFallClassifier, TorchFallClassifier
from .config import AppConfig
from .detector import UltralyticsPoseTracker
from .events import EventSink
from .features import SequenceStore
from .types import RiskEvent, TrackedPose


class MonitorPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.tracker = UltralyticsPoseTracker(config.detector)
        # The learned classifier is trained only on complete fixed-length windows.
        # Keep the shorter bootstrap path exclusively for the heuristic classifier.
        minimum_frames = (
            config.sequence.length
            if config.analysis.classifier_model
            else config.sequence.minimum_frames
        )
        self.store = SequenceStore(
            config.sequence.length,
            minimum_frames,
            config.sequence.keypoint_confidence,
            config.sequence.stale_track_seconds,
        )
        if not config.analysis.classifier_model:
            classifier = HeuristicFallClassifier()
        elif config.analysis.classifier_model.lower().endswith(".xml"):
            classifier = OpenVinoFallClassifier(config.analysis.classifier_model)
        else:
            classifier = TorchFallClassifier(config.analysis.classifier_model)
        self.analyzer = RiskAnalyzer(config.analysis, classifier)
        self.sink = EventSink(config.output)
        self.recent_events: dict[int, deque[RiskEvent]] = defaultdict(lambda: deque(maxlen=1))
        self._last_sample_at: float | None = None

    def process_frame(
        self,
        frame: np.ndarray,
        now: float | None = None,
        fps: float = 0.0,
    ) -> tuple[np.ndarray, list[RiskEvent]]:
        """Process one frame for CLI or UI integrations."""
        now = time.monotonic() if now is None else now
        sample_interval = 1.0 / max(self.config.sequence.sample_fps, 1e-6)
        if (
            self._last_sample_at is not None
            and now >= self._last_sample_at
            and now - self._last_sample_at < sample_interval * 0.95
        ):
            return frame.copy(), []
        self._last_sample_at = now
        poses = self.tracker(frame, now)
        self.analyzer.prune(now, self.config.sequence.stale_track_seconds)
        events: list[RiskEvent] = []
        for pose in poses:
            window = self.store.update(pose)
            if window is not None:
                detected = self.analyzer.update(pose.track_id, window, now)
                for event in detected:
                    self.recent_events[pose.track_id].append(event)
                    self.sink.emit(event, frame)
                events.extend(detected)
        return self._draw(frame, poses, fps), events

    def run(self) -> None:
        cap = cv2.VideoCapture(self.config.video.source)
        if not cap.isOpened():
            raise RuntimeError(f"Khong mo duoc nguon video: {self.config.video.source}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.video.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.video.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame_index, fps, last_tick = 0, 0.0, time.perf_counter()
        source_fps = cap.get(cv2.CAP_PROP_FPS) or self.config.sequence.sample_fps
        live_source = isinstance(self.config.video.source, int) or str(
            self.config.video.source
        ).lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % self.config.video.process_every_n_frames:
                    continue
                media_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                now = (
                    time.monotonic()
                    if live_source
                    else media_time if media_time > 0 else (frame_index - 1) / source_fps
                )
                elapsed = time.perf_counter() - last_tick
                instant_fps = 1.0 / max(elapsed, 1e-6)
                fps = instant_fps if fps == 0 else 0.90 * fps + 0.10 * instant_fps
                last_tick = time.perf_counter()
                annotated, _ = self.process_frame(frame, now, fps)
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
