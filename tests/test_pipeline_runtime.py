from __future__ import annotations

import numpy as np

import eldercare_monitor.pipeline as pipeline_module
from eldercare_monitor.config import AppConfig


def test_learned_classifier_waits_for_a_full_training_window(monkeypatch) -> None:
    config = AppConfig()
    config.analysis.classifier_model = "model.xml"
    monkeypatch.setattr(pipeline_module, "UltralyticsPoseTracker", lambda _: object())
    monkeypatch.setattr(pipeline_module, "OpenVinoFallClassifier", lambda _: object())
    monkeypatch.setattr(pipeline_module, "RiskAnalyzer", lambda *_: object())
    monkeypatch.setattr(pipeline_module, "EventSink", lambda *_: object())

    pipeline = pipeline_module.MonitorPipeline(config)

    assert pipeline.store.minimum_frames == config.sequence.length


def test_process_frame_skips_frames_above_the_training_sample_rate() -> None:
    config = AppConfig()
    config.sequence.sample_fps = 10.0
    pipeline = pipeline_module.MonitorPipeline.__new__(pipeline_module.MonitorPipeline)
    pipeline.config = config
    pipeline._last_sample_at = 1.0
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    output, events = pipeline.process_frame(frame, now=1.01)

    assert not events
    assert np.array_equal(output, frame)
