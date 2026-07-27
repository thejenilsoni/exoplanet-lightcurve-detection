from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset


class TransitDataset(Dataset[dict[str, torch.Tensor]]):
    """NPZ-backed dataset with a deterministic synthetic fallback.

    Real files contain phase_flux [L], features [8], and scalar label. When no
    root is supplied, physically motivated transit and false-positive examples
    are generated for pipeline smoke tests and pretraining.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        samples: int = 12000,
        length: int = 256,
        seed: int = 2026,
    ) -> None:
        self.files = sorted(Path(root).glob("*.npz")) if root else []
        self.samples = len(self.files) if self.files else samples
        self.length = length
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.files:
            with np.load(self.files[index]) as sample:
                phase_flux = sample["phase_flux"].astype(np.float32)
                features = sample["features"].astype(np.float32)
                label = np.float32(sample["label"])
        else:
            phase_flux, features, label = synthetic_example(
                np.random.default_rng(self.seed + index),
                self.length,
            )
        return {
            "phase_flux": torch.from_numpy(phase_flux.reshape(1, -1)),
            "features": torch.from_numpy(features),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def synthetic_example(
    rng: np.random.Generator,
    length: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.float32]:
    phase = np.linspace(-0.5, 0.5, length, dtype=np.float32)
    white_noise = rng.normal(0, 1, length)
    red_noise = gaussian_filter1d(rng.normal(0, 1, length), sigma=rng.uniform(2, 9))
    signal = white_noise * rng.uniform(0.55, 1.0) + red_noise * rng.uniform(0.2, 0.75)
    signal = (signal - np.median(signal)) / (np.std(signal) + 1e-6)

    positive = bool(rng.random() < 0.5)
    period = float(np.exp(rng.uniform(np.log(0.5), np.log(24))))
    transit_count = int(rng.integers(3, 13))

    if positive:
        half_width = rng.uniform(0.012, 0.075)
        ingress = max(half_width * rng.uniform(0.12, 0.35), 0.004)
        depth_sigma = rng.uniform(3.2, 13.0)
        distance = np.abs(phase)
        profile = np.clip((half_width - distance) / ingress, 0, 1)
        profile[distance < half_width - ingress] = 1
        signal -= profile * depth_sigma
        duration_fraction = half_width * 2
        depth_ppm = float(depth_sigma * rng.uniform(350, 1100))
        snr = float(depth_sigma * np.sqrt(transit_count) * rng.uniform(0.75, 1.05))
        odd_even = float(rng.uniform(0, 0.16))
        secondary_ratio = float(rng.uniform(0, 0.22))
        label = np.float32(1)
    else:
        kind = int(rng.integers(0, 4))
        half_width = rng.uniform(0.02, 0.16)
        depth_sigma = rng.uniform(1.0, 9.0)
        if kind == 0:
            signal += np.sin(phase * np.pi * rng.uniform(3, 12)) * depth_sigma * 0.45
        elif kind == 1:
            primary = np.exp(-0.5 * (phase / half_width) ** 2)
            secondary = np.exp(-0.5 * ((np.abs(phase) - 0.48) / half_width) ** 2)
            signal -= depth_sigma * primary + depth_sigma * rng.uniform(0.35, 0.85) * secondary
        elif kind == 2:
            location = rng.uniform(-0.42, 0.42)
            signal -= depth_sigma * np.exp(-0.5 * ((phase - location) / 0.008) ** 2)
        duration_fraction = half_width * 2
        depth_ppm = float(depth_sigma * rng.uniform(250, 1400))
        snr = float(depth_sigma * np.sqrt(transit_count) * rng.uniform(0.35, 0.95))
        odd_even = float(rng.uniform(0.18, 1.5))
        secondary_ratio = float(rng.uniform(0.2, 1.4))
        label = np.float32(0)

    features = np.array(
        [
            np.log1p(period) / 4,
            duration_fraction,
            np.log1p(depth_ppm) / 12,
            min(snr, 40) / 40,
            min(odd_even, 2) / 2,
            min(secondary_ratio, 2) / 2,
            min(transit_count, 15) / 15,
            min(1 / max(duration_fraction, 1e-4), 1000) / 1000,
        ],
        dtype=np.float32,
    )
    return signal.astype(np.float32), features, label
