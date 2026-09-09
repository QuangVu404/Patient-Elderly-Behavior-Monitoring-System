from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import MonitorPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-first behavior monitor")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", help="Override camera index, video path, or RTSP URL")
    parser.add_argument("--model", help="Override YOLO .pt or OpenVINO model directory")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.source is not None:
        config.video.source = int(args.source) if args.source.isdigit() else args.source
    if args.model:
        config.detector.model = args.model
    if args.headless:
        config.video.display = False
    MonitorPipeline(config).run()


if __name__ == "__main__":
    main()
