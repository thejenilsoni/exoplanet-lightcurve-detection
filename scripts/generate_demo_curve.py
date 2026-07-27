"""Generate a deterministic noisy transit light curve for the analysis console."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo-light-curve.csv"))
    parser.add_argument("--period", type=float, default=2.4704)
    parser.add_argument("--depth", type=float, default=0.0087)
    parser.add_argument("--days", type=float, default=27.0)
    parser.add_argument("--cadence-minutes", type=float, default=20.0)
    return parser.parse_args()


def generate(period: float, depth: float, days: float, cadence_minutes: float) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    cadence_days = cadence_minutes / (24 * 60)
    time = np.arange(0, days, cadence_days)
    phase = ((time - 0.48 + period / 2) % period) / period - 0.5
    transit = np.abs(phase) < 0.022
    trend = 1 + 0.0018 * np.sin(time / 4.7) + 0.0007 * np.cos(time / 1.8)
    noise = rng.normal(0, 0.00085, time.size)
    flux = trend + noise
    flux[transit] -= depth
    quality = np.zeros(time.size, dtype=int)
    quality[rng.choice(time.size, size=max(1, time.size // 180), replace=False)] = 1
    return pd.DataFrame(
        {
            "time": time,
            "flux": flux,
            "flux_err": np.full(time.size, 0.00085),
            "quality": quality,
        }
    )


if __name__ == "__main__":
    arguments = parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    generate(
        arguments.period,
        arguments.depth,
        arguments.days,
        arguments.cadence_minutes,
    ).to_csv(arguments.output, index=False)
    print(f"Wrote demo light curve to {arguments.output}")
