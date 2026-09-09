import numpy as np

from eldercare_monitor.features import FEATURE_DIM, SequenceStore, pose_to_feature
from eldercare_monitor.types import TrackedPose


def pose(track_id=1, x=100.0, y=100.0, width=40.0, height=100.0, timestamp=0.0):
    points = np.zeros((17, 3), dtype=np.float32)
    points[:, 0] = x + width / 2
    points[:, 1] = y + height / 2
    points[:, 2] = 0.9
    return TrackedPose(
        track_id,
        np.array([x, y, x + width, y + height], dtype=np.float32),
        points,
        0.9,
        timestamp,
        (480, 640),
    )


def test_pose_feature_shape_and_geometry():
    feature = pose_to_feature(pose(), None, 0.25)
    assert feature.shape == (FEATURE_DIM,)
    assert np.isclose(feature[-4], 0.4)


def test_sequence_is_left_padded_and_track_isolated():
    store = SequenceStore(length=4, minimum_frames=2, min_conf=0.25, stale_seconds=2.0)
    assert store.update(pose(timestamp=0.0)) is None
    window = store.update(pose(timestamp=0.1, x=102.0))
    assert window is not None
    assert window.features.shape == (4, FEATURE_DIM)
    assert window.valid_length == 2
    assert store.update(pose(track_id=2, timestamp=0.1)) is None

