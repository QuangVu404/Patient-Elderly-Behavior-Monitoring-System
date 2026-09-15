from __future__ import annotations

import argparse
import statistics
import time

import cv2

from eldercare_monitor.config import DetectorConfig
from eldercare_monitor.detector import UltralyticsPoseTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure end-to-end detector/tracker latency")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    tracker = UltralyticsPoseTracker(
        DetectorConfig(model=args.model, image_size=args.image_size, device="cpu")
    )
    latencies: list[float] = []
    try:
        for index in range(args.frames + args.warmup):
            ok, frame = cap.read()
            if not ok:
                break
            started = time.perf_counter()
            tracker(frame)
            elapsed = time.perf_counter() - started
            if index >= args.warmup:
                latencies.append(elapsed)
    finally:
        cap.release()
    if not latencies:
        raise RuntimeError("No frames measured")
    ordered = sorted(latencies)
    p95 = ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)]
    print(
        f"frames={len(latencies)} mean_ms={statistics.mean(latencies) * 1000:.1f} "
        f"p95_ms={p95 * 1000:.1f} fps={1 / statistics.mean(latencies):.2f}"
    )


if __name__ == "__main__":
    main()
