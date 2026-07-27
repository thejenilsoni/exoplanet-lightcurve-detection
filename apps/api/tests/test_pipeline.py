import numpy as np
from app.services.pipeline import (
    LightCurve,
    heuristic_probability,
    preprocess,
    search_transits,
)


def synthetic_curve(period: float = 2.75, points: int = 2400) -> LightCurve:
    rng = np.random.default_rng(42)
    time = np.linspace(0, 27, points)
    phase = ((time - 0.42 + period / 2) % period) / period - 0.5
    flux = 1 + rng.normal(0, 0.0008, points) + np.sin(time / 8) * 0.0015
    flux[np.abs(phase) < 0.025] -= 0.009
    return LightCurve(
        time=time,
        flux=flux,
        flux_error=np.full(points, 0.0008),
        quality=np.zeros(points, dtype=np.int64),
        target_name="synthetic",
        mission="test",
    )


def test_pipeline_recovers_injected_period() -> None:
    processed = preprocess(synthetic_curve())
    candidate = search_transits(processed)

    recovered = candidate.period_days
    assert min(abs(recovered - 2.75), abs(recovered * 2 - 2.75), abs(recovered - 5.5)) < 0.08
    assert candidate.depth > 0.004
    assert candidate.signal_to_noise > 7
    assert candidate.transit_count >= 3
    assert heuristic_probability(candidate) > 0.65


def test_preprocessing_rejects_short_curves() -> None:
    curve = synthetic_curve(points=80)
    try:
        preprocess(curve)
    except ValueError as error:
        assert "120" in str(error)
    else:
        raise AssertionError("Expected a short-curve validation error")
