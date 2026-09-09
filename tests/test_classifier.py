import numpy as np

from eldercare_monitor.classifier import HeuristicFallClassifier
from eldercare_monitor.features import FEATURE_DIM


def sequence(aspect_start, aspect_end, center_start, center_end, visible=True):
    values = np.zeros((32, FEATURE_DIM), dtype=np.float32)
    values[:, -4] = np.linspace(aspect_start, aspect_end, 32)
    values[:, -2] = np.linspace(center_start, center_end, 32)
    if visible:
        values[:, 17 * 2 : 17 * 3] = 0.9
    return values


def test_heuristic_separates_clear_fall_from_standing():
    classifier = HeuristicFallClassifier()
    standing = classifier.predict(sequence(0.4, 0.4, 0.45, 0.45))
    fall = classifier.predict(sequence(0.4, 1.25, 0.40, 0.72))
    assert standing < 0.1
    assert fall > 0.72


def test_heuristic_rejects_pose_with_no_visible_keypoints():
    classifier = HeuristicFallClassifier()
    score = classifier.predict(sequence(0.4, 1.25, 0.40, 0.72, visible=False))
    assert score == 0.0

