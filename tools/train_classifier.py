from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eldercare_monitor.features import FEATURE_DIM
from eldercare_monitor.model import MultiScaleTemporalGRU


def group_split(groups: np.ndarray, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = list(np.unique(groups))
    if len(unique) < 2:
        raise ValueError("At least two independent subject/video groups are required")
    random.Random(seed).shuffle(unique)
    count = max(1, round(len(unique) * validation_fraction))
    validation_groups = set(unique[:count])
    validation = np.array([group in validation_groups for group in groups])
    return ~validation, validation


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "accuracy": (tp + tn) / max(len(labels), 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "false_positive_rate": fp / max(fp + tn, 1),
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = np.linspace(0.30, 0.95, 66)
    scored = [(float(value), metrics(labels, (probabilities >= value).astype(np.int64))) for value in candidates]
    # F1 first; higher threshold breaks ties to reduce false alarms.
    return max(scored, key=lambda item: (item[1]["f1"], item[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="NPZ containing X [N,T,F], y [N], groups [N]")
    parser.add_argument("--output", default="artifacts/temporal_attention.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data = np.load(args.dataset, allow_pickle=False)
    x, y = data["X"].astype(np.float32), data["y"].astype(np.int64)
    groups = data["groups"] if "groups" in data else np.arange(len(y))
    if x.ndim != 3 or x.shape[2] != FEATURE_DIM:
        raise ValueError(f"Expected X [N,T,{FEATURE_DIM}], received {x.shape}")
    train_mask, val_mask = group_split(groups, 0.2, args.seed)
    if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[val_mask])) < 2:
        raise ValueError("Train and validation partitions must each contain both labels; add more groups")
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x[train_mask]), torch.from_numpy(y[train_mask])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_x, val_y = torch.from_numpy(x[val_mask]), torch.from_numpy(y[val_mask])
    model = MultiScaleTemporalGRU(FEATURE_DIM, args.hidden_dim)
    class_counts = np.bincount(y[train_mask], minlength=2)
    weights = len(y[train_mask]) / np.maximum(class_counts * 2, 1)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32), label_smoothing=0.05
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )

    best_f1, best_state, best_metrics = -1.0, None, {}
    for epoch in range(args.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            predictions = model(val_x).argmax(1).numpy()
        current = metrics(val_y.numpy(), predictions)
        if current["f1"] > best_f1:
            best_f1 = current["f1"]
            best_metrics = current
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(f"epoch={epoch + 1:03d} val_f1={current['f1']:.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(val_x), dim=1)[:, 1].numpy()
    recommended_threshold, best_metrics = select_threshold(val_y.numpy(), probabilities)
    best_metrics["recommended_threshold"] = recommended_threshold
    torch.save(
        {
            "state_dict": best_state,
            "model_name": "MultiScaleTemporalGRU",
            "input_dim": FEATURE_DIM,
            "hidden_dim": args.hidden_dim,
            "sequence_length": x.shape[1],
            "metrics": best_metrics,
        },
        output,
    )
    output.with_suffix(".metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    print(f"saved={output} metrics={best_metrics}")


if __name__ == "__main__":
    main()
