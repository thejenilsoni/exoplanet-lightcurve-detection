import io

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


def csv_payload() -> bytes:
    rng = np.random.default_rng(18)
    time = np.linspace(0, 21, 1800)
    period = 3.12
    phase = ((time - 0.65 + period / 2) % period) / period - 0.5
    flux = 1 + rng.normal(0, 0.0007, time.size)
    flux[np.abs(phase) < 0.022] -= 0.0075
    frame = pd.DataFrame(
        {
            "time": time,
            "flux": flux,
            "flux_err": np.full(time.size, 0.0007),
            "quality": np.zeros(time.size, dtype=int),
        }
    )
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def test_health_and_analysis_contract() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        response = client.post(
            "/v1/analyze",
            files={"light_curve": ("synthetic.csv", csv_payload(), "text/csv")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["requestId"]
    assert payload["targetName"] == "synthetic"
    assert payload["metrics"]["signalToNoise"] > 5
    assert len(payload["phase"]) == len(payload["phaseFlux"])
    assert len(payload["periods"]) == len(payload["power"])
    assert payload["mode"] in {"baseline", "learned"}
