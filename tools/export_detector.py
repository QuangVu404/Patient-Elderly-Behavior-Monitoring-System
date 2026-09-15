from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO pose detector to OpenVINO")
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--data", help="Calibration dataset YAML required for meaningful INT8 calibration")
    args = parser.parse_args()
    from ultralytics import YOLO

    kwargs = {"format": "openvino", "imgsz": args.image_size, "half": not args.int8}
    if args.int8:
        if not args.data:
            raise ValueError("--data is required with --int8")
        kwargs.update({"int8": True, "data": args.data})
    result = YOLO(args.model).export(**kwargs)
    print(f"saved={result}")


if __name__ == "__main__":
    main()
