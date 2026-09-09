from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .config import OutputConfig
from .types import RiskEvent


class EventSink:
    def __init__(self, config: OutputConfig):
        self.config = config
        self.path = Path(config.events_jsonl)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots = Path(config.snapshots_dir)
        if config.save_snapshots:
            self.snapshots.mkdir(parents=True, exist_ok=True)

    def emit(self, event: RiskEvent, frame: np.ndarray) -> None:
        payload = asdict(event)
        payload["recorded_at"] = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        if self.config.save_snapshots:
            filename = f"{event.kind}_id{event.track_id}_{int(event.timestamp * 1000)}.jpg"
            cv2.imwrite(str(self.snapshots / filename), frame)
        print(f"[ALERT] {event.kind} id={event.track_id} p={event.probability:.2f}")

