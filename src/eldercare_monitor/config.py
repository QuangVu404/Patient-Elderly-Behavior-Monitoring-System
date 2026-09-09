from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class VideoConfig:
    source: int | str = 0
    width: int = 640
    height: int = 384
    process_every_n_frames: int = 1
    display: bool = True


@dataclass(slots=True)
class DetectorConfig:
    model: str = "yolo8n-pose.pt"
    device: str = "cpu"
    image_size: int = 416
    confidence: float = 0.30
    iou: float = 0.50
    tracker: str = "bytetrack.yaml"


@dataclass(slots=True)
class SequenceConfig:
    length: int = 32
    minimum_frames: int = 12
    keypoint_confidence: float = 0.25
    stale_track_seconds: float = 2.0


@dataclass(slots=True)
class AnalysisConfig:
    classifier_model: str | None = None
    fall_threshold: float = 0.72
    confirm_frames: int = 3
    cooldown_seconds: float = 15.0
    immobility_seconds: float = 30.0
    immobility_motion_threshold: float = 0.018
    horizontal_aspect_ratio: float = 0.95


@dataclass(slots=True)
class OutputConfig:
    events_jsonl: str = "artifacts/events.jsonl"
    snapshots_dir: str = "artifacts/snapshots"
    save_snapshots: bool = False


@dataclass(slots=True)
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _coerce_source(value: Any) -> int | str:
    if isinstance(value, int):
        return value
    value = str(value)
    return int(value) if value.isdigit() else value


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    video = VideoConfig(**raw.get("video", {}))
    video.source = _coerce_source(video.source)
    return AppConfig(
        video=video,
        detector=DetectorConfig(**raw.get("detector", {})),
        sequence=SequenceConfig(**raw.get("sequence", {})),
        analysis=AnalysisConfig(**raw.get("analysis", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
