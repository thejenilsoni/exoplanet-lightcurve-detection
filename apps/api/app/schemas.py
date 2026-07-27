from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CandidateMetrics(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    period_days: float = Field(alias="periodDays")
    duration_hours: float = Field(alias="durationHours")
    depth_ppm: float = Field(alias="depthPpm")
    signal_to_noise: float = Field(alias="signalToNoise")
    odd_even_mismatch: float = Field(alias="oddEvenMismatch")
    secondary_depth_ppm: float = Field(alias="secondaryDepthPpm")


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    request_id: str = Field(alias="requestId")
    target_name: str = Field(alias="targetName")
    mission: str
    disposition: Literal["high-interest", "review", "low-interest"]
    probability: float
    model: str
    mode: Literal["learned", "baseline"]
    metrics: CandidateMetrics
    time: list[float]
    normalized_flux: list[float] = Field(alias="normalizedFlux")
    trend: list[float]
    phase: list[float]
    phase_flux: list[float] = Field(alias="phaseFlux")
    periods: list[float]
    power: list[float]
    flags: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    model_ready: bool
    inference_mode: Literal["learned", "baseline"]
