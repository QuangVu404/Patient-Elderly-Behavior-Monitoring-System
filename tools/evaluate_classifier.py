from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from eldercare_monitor.model import TemporalAttention1D


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen checkpoint on an independent NPZ set")
    parser.add_argument("checkpoint")
    parser.add_argument("dataset")
    parser.add_argument("--threshold", type=float, help="Default: checkpoint recommendation or 0.5")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TemporalAttention1D(checkpoint["input_dim"], checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    dataset = np.load(args.dataset, allow_pickle=False)
    x, labels = dataset["X"].astype(np.float32), dataset["y"].astype(np.int64)
    threshold = args.threshold
    if threshold is None:
        threshold = float(checkpoint.get("metrics", {}).get("recommended_threshold", 0.5))
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(x)), dim=1)[:, 1].numpy()
    predictions = probabilities >= threshold
    tp = int(np.sum((labels == 1) & predictions))
    fp = int(np.sum((labels == 0) & predictions))
    fn = int(np.sum((labels == 1) & ~predictions))
    tn = int(np.sum((labels == 0) & ~predictions))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {
        "samples": len(labels),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "false_positive_rate": fp / max(fp + tn, 1),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
