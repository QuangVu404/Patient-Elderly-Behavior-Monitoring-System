from __future__ import annotations

from typing import Protocol

import numpy as np

from .features import FEATURE_DIM


class FallClassifier(Protocol):
    def predict(self, features: np.ndarray) -> float: ...


class HeuristicFallClassifier:
    """Conservative bootstrap baseline used until a trained model is supplied."""

    def predict(self, features: np.ndarray) -> float:
        geometry = features[:, FEATURE_DIM - 4 :]
        confidences = features[:, 17 * 2 : 17 * 3]
        aspect = geometry[:, 0]
        center_y = geometry[:, 2]
        split = max(len(features) // 3, 1)
        initial_aspect = float(np.median(aspect[:split]))
        posture_change = max(float(aspect[-1] - initial_aspect), 0.0)
        vertical_drop = max(float(center_y[-1] - np.min(center_y[: max(len(features) - 2, 1)])), 0.0)
        horizontal = np.clip((float(aspect[-1]) - 0.75) / 0.55, 0.0, 1.0)
        dynamic = np.clip(vertical_drop / 0.22, 0.0, 1.0)
        rotation = np.clip(posture_change / 0.55, 0.0, 1.0)
        visibility = np.clip(np.median(np.sum(confidences[-5:] >= 0.25, axis=1)) / 5.0, 0.0, 1.0)
        score = 0.45 * horizontal + 0.35 * dynamic + 0.20 * rotation
        return float(np.clip(score * visibility, 0.0, 1.0))


class OpenVinoFallClassifier:
    def __init__(self, model_path: str, device: str = "CPU"):
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError("Install OpenVINO support: pip install -e .[openvino]") from exc
        core = ov.Core()
        self.compiled = core.compile_model(model_path, device)
        self.input = self.compiled.input(0)
        self.output = self.compiled.output(0)

    def predict(self, features: np.ndarray) -> float:
        logits = np.asarray(self.compiled({self.input: features[None].astype(np.float32)})[self.output])[0]
        logits = logits - np.max(logits)
        probability = np.exp(logits) / np.exp(logits).sum()
        return float(probability[1])


class TorchFallClassifier:
    """PyTorch runtime fallback for a stage-07 temporal checkpoint."""

    def __init__(self, model_path: str, device: str = "cpu"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install PyTorch support: pip install -e .[train]") from exc

        from .model import build_temporal_model

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        required = {"state_dict", "model_name", "input_dim", "hidden_dim"}
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"Temporal checkpoint is missing fields: {sorted(missing)}")
        self.torch = torch
        self.device = device
        self.model = build_temporal_model(
            str(checkpoint["model_name"]),
            int(checkpoint["input_dim"]),
            int(checkpoint["hidden_dim"]),
        ).to(device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict(self, features: np.ndarray) -> float:
        tensor = self.torch.from_numpy(features[None].astype(np.float32)).to(self.device)
        with self.torch.inference_mode():
            probability = self.torch.softmax(self.model(tensor), dim=1)[0, 1]
        return float(probability.cpu())


def load_fall_classifier(model_path: str) -> FallClassifier:
    suffix = str(model_path).lower()
    if suffix.endswith(".xml"):
        return OpenVinoFallClassifier(model_path)
    if suffix.endswith((".pt", ".pth")):
        return TorchFallClassifier(model_path)
    raise ValueError(f"Unsupported classifier model format: {model_path}")
