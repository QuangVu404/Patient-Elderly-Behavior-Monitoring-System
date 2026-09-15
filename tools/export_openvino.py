from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from eldercare_monitor.model import build_temporal_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and optionally INT8-quantize temporal model")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="artifacts/temporal_openvino/model.xml")
    parser.add_argument("--calibration", help="NPZ dataset; enables NNCF INT8 PTQ")
    args = parser.parse_args()

    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError("Install export dependencies: pip install -e .[train,openvino]") from exc

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_temporal_model(
        checkpoint.get("model_name", "TemporalAttention1D"),
        checkpoint["input_dim"], checkpoint["hidden_dim"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    example = torch.zeros(1, checkpoint["sequence_length"], checkpoint["input_dim"])
    ov_model = ov.convert_model(model, example_input=example)

    if args.calibration:
        try:
            import nncf
        except ImportError as exc:
            raise RuntimeError("INT8 export requires NNCF") from exc
        samples = np.load(args.calibration, allow_pickle=False)["X"].astype(np.float32)
        dataset = nncf.Dataset(samples[: min(len(samples), 300)], lambda item: item[None])
        ov_model = nncf.quantize(ov_model, dataset)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, output, compress_to_fp16=not bool(args.calibration))
    print(f"saved={output} precision={'INT8' if args.calibration else 'FP16'}")


if __name__ == "__main__":
    main()
