from pathlib import Path
from typing import Literal

import numpy as np

from app.services.pipeline import TransitCandidate, heuristic_probability


class CandidateScorer:
    """Optional morphology-and-diagnostics model with a transparent fallback."""

    def __init__(self, model_path: str | None = None) -> None:
        self.mode: Literal["learned", "baseline"] = "baseline"
        self.name = "astrophysical-vetting-baseline"
        self._model = None
        self._torch = None
        if not model_path:
            return
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Candidate model not found: {model_path}")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install the API with the 'ml' extra to use MODEL_PATH.") from exc
        self._torch = torch
        self._model = torch.jit.load(str(path), map_location="cpu").eval()
        self.mode = "learned"
        self.name = "transit-fusion-1d"

    @staticmethod
    def features(candidate: TransitCandidate) -> np.ndarray:
        return np.array(
            [
                np.log1p(candidate.period_days) / 4,
                candidate.duration_days / candidate.period_days,
                np.log1p(candidate.depth * 1e6) / 12,
                min(candidate.signal_to_noise, 40) / 40,
                min(candidate.odd_even_mismatch, 2) / 2,
                min(candidate.secondary_depth / max(candidate.depth, 1e-8), 2) / 2,
                min(candidate.transit_count, 15) / 15,
                min(candidate.period_days / max(candidate.duration_days, 1e-6), 1000) / 1000,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def phase_window(candidate: TransitCandidate, length: int = 256) -> np.ndarray:
        order = np.argsort(candidate.phase)
        phase = candidate.phase[order]
        flux = candidate.phase_flux[order]
        grid = np.linspace(-0.5, 0.5, length, dtype=np.float32)
        interpolated = np.interp(grid, phase, flux)
        median = float(np.median(interpolated))
        scale = max(float(np.median(np.abs(interpolated - median))) * 1.4826, 1e-5)
        return ((interpolated - median) / scale).astype(np.float32)

    def score(self, candidate: TransitCandidate) -> float:
        if self._model is None:
            return heuristic_probability(candidate)
        features = self.features(candidate)
        window = self.phase_window(candidate)
        with self._torch.inference_mode():
            output = self._model(
                self._torch.from_numpy(window).reshape(1, 1, -1),
                self._torch.from_numpy(features).reshape(1, -1),
            )
            probability = self._torch.sigmoid(output.reshape(-1)[0]).item()
        return float(np.clip(probability, 0.0, 1.0))
