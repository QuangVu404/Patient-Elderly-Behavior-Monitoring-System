from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - only imported by training/export tools
    raise RuntimeError("Install training dependencies: pip install -e .[train]") from exc


class TemporalAttention1D(nn.Module):
    """Small temporal convolution encoder with learned attention pooling."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, classes: int = 2, dropout: float = 0.15):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.projection(features)
        encoded = encoded + self.temporal(encoded.transpose(1, 2)).transpose(1, 2)
        weights = torch.softmax(self.attention(encoded), dim=1)
        pooled = torch.sum(encoded * weights, dim=1)
        return self.classifier(pooled)

