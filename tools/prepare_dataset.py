from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from eldercare_monitor.config import DetectorConfig
from eldercare_monitor.detector import UltralyticsPoseTracker
from eldercare_monitor.features import FEATURE_DIM, SequenceStore


def iter_rows(manifest: Path):
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def extract_row(row: dict[str, str], args: argparse.Namespace) -> list[np.ndarray]:
    video_path = Path(row["video_path"])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start = float(row.get("start_sec") or 0)
    end = float(row.get("end_sec") or float("inf"))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    sample_step = max(round(fps / args.sample_fps), 1)
    tracker = UltralyticsPoseTracker(
        DetectorConfig(model=args.model, image_size=args.image_size, device="cpu")
    )
    store = SequenceStore(args.sequence_length, args.minimum_frames, 0.25, 2.0)
    windows: list[np.ndarray] = []
    sampled = 0
    frame_index = round(start * fps)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame_index / fps > end:
                break
            if frame_index % sample_step == 0:
                timestamp = frame_index / fps
                poses = tracker(frame, timestamp)
                # In multi-person clips, each stable ByteTrack ID becomes its own sample stream.
                for pose in poses:
                    window = store.update(pose)
                    if window is not None and sampled % args.window_stride == 0:
                        windows.append(window.features.copy())
                sampled += 1
            frame_index += 1
    finally:
        cap.release()
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporal pose windows from a CSV manifest")
    parser.add_argument("manifest")
    parser.add_argument("--output", default="data/processed/sequences.npz")
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--minimum-frames", type=int, default=12)
    parser.add_argument("--window-stride", type=int, default=4)
    args = parser.parse_args()

    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    all_groups: list[str] = []
    for row in iter_rows(Path(args.manifest)):
        label = int(row["label"])
        group = row.get("subject") or Path(row["video_path"]).stem
        windows = extract_row(row, args)
        all_x.extend(windows)
        all_y.extend([label] * len(windows))
        all_groups.extend([group] * len(windows))
        print(f"video={row['video_path']} label={label} windows={len(windows)}")
    if not all_x:
        raise RuntimeError("No pose windows extracted; check paths, intervals, and detections")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=np.stack(all_x).astype(np.float32),
        y=np.asarray(all_y, dtype=np.int64),
        groups=np.asarray(all_groups),
    )
    print(f"saved={output} samples={len(all_x)} shape=(*,{args.sequence_length},{FEATURE_DIM})")


if __name__ == "__main__":
    main()
