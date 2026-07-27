import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.network(inputs))


class TransitFusionNet(nn.Module):
    """Fuse phase-folded morphology with astrophysical diagnostics."""

    def __init__(self, feature_count: int = 8, channels: int = 48) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            ResidualBlock1D(channels, dilation=1),
            ResidualBlock1D(channels, dilation=2),
            ResidualBlock1D(channels, dilation=4),
            nn.Conv1d(channels, channels * 2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            ResidualBlock1D(channels * 2, dilation=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_count, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(channels * 2 + 32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, phase_flux: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        temporal = self.pool(self.temporal(self.stem(phase_flux))).squeeze(-1)
        diagnostics = self.feature_encoder(features)
        return self.classifier(torch.cat([temporal, diagnostics], dim=1)).squeeze(1)
