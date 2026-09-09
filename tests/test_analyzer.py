import numpy as np

from eldercare_monitor.analyzer import RiskAnalyzer
from eldercare_monitor.config import AnalysisConfig
from eldercare_monitor.features import FEATURE_DIM, TrackWindow


class FixedClassifier:
    def __init__(self, value):
        self.value = value

    def predict(self, features):
        return self.value


def window(aspect=0.4, motion=0.1):
    values = np.zeros((12, FEATURE_DIM), dtype=np.float32)
    values[:, 17 * 2 : 17 * 3] = 0.9
    values[:, -4] = aspect
    values[:, 17 * 3 : 17 * 5] = motion
    return TrackWindow(values, np.arange(12, dtype=np.float64), 12)


def test_fall_requires_consecutive_confirmations_and_cooldown():
    config = AnalysisConfig(fall_threshold=0.7, confirm_frames=2, cooldown_seconds=10)
    analyzer = RiskAnalyzer(config, FixedClassifier(0.9))
    assert analyzer.update(7, window(), 1.0) == []
    events = analyzer.update(7, window(), 2.0)
    assert len(events) == 1 and events[0].kind == "fall"
    assert analyzer.update(7, window(), 3.0) == []


def test_immobility_uses_wall_clock_not_short_model_window():
    config = AnalysisConfig(
        fall_threshold=1.0,
        immobility_seconds=5.0,
        immobility_motion_threshold=0.02,
        horizontal_aspect_ratio=0.9,
    )
    analyzer = RiskAnalyzer(config, FixedClassifier(0.0))
    still = window(aspect=1.2, motion=0.0)
    assert analyzer.update(1, still, 10.0) == []
    events = analyzer.update(1, still, 15.1)
    assert len(events) == 1 and events[0].kind == "immobility"
