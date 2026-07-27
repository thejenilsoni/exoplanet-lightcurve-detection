import time
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import AnalysisResponse, CandidateMetrics, HealthResponse
from app.services.pipeline import (
    diagnostic_flags,
    downsample,
    preprocess,
    read_light_curve,
    search_transits,
)
from app.services.scoring import CandidateScorer


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.scorer = CandidateScorer(settings.model_path)
    yield


app = FastAPI(
    title="Exoplanet Light-Curve Detection API",
    version="0.1.0",
    description=(
        "Robust preprocessing, periodic transit search, false-positive diagnostics, "
        "and candidate ranking for Kepler and TESS light curves."
    ),
    lifespan=lifespan,
)
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": app.title, "health": "/health", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request) -> HealthResponse:
    scorer = request.app.state.scorer
    return HealthResponse(
        status="ok",
        service="exoplanet-detection",
        model_ready=True,
        inference_mode=scorer.mode,
    )


@app.post("/v1/analyze", response_model=AnalysisResponse, tags=["detection"])
async def analyze(
    request: Request,
    light_curve: Annotated[UploadFile, File(description="CSV or FITS light curve")],
) -> AnalysisResponse:
    started = time.perf_counter()
    payload = await light_curve.read()
    maximum = request.app.state.settings.max_upload_mb * 1024 * 1024
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded light curve is empty.")
    if len(payload) > maximum:
        raise HTTPException(status_code=413, detail="The uploaded file exceeds the size limit.")

    try:
        source = read_light_curve(payload, light_curve.filename or "light-curve.csv")
        processed = preprocess(source)
        candidate = search_transits(processed)
        probability = request.app.state.scorer.score(candidate)
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if probability >= 0.8:
        disposition = "high-interest"
    elif probability >= 0.45:
        disposition = "review"
    else:
        disposition = "low-interest"

    indexes = np.linspace(
        0,
        processed.time.size - 1,
        min(900, processed.time.size),
    ).astype(int)
    _ = time.perf_counter() - started
    return AnalysisResponse(
        requestId=str(uuid4()),
        targetName=source.target_name,
        mission=source.mission,
        disposition=disposition,
        probability=round(probability, 5),
        model=request.app.state.scorer.name,
        mode=request.app.state.scorer.mode,
        metrics=CandidateMetrics(
            periodDays=round(candidate.period_days, 6),
            durationHours=round(candidate.duration_days * 24, 4),
            depthPpm=round(candidate.depth * 1e6, 2),
            signalToNoise=round(candidate.signal_to_noise, 3),
            oddEvenMismatch=round(candidate.odd_even_mismatch, 4),
            secondaryDepthPpm=round(candidate.secondary_depth * 1e6, 2),
        ),
        time=processed.time[indexes].round(7).tolist(),
        normalizedFlux=processed.normalized_flux[indexes].round(8).tolist(),
        trend=processed.trend[indexes].round(8).tolist(),
        phase=candidate.phase.round(7).tolist(),
        phaseFlux=candidate.phase_flux.round(8).tolist(),
        periods=downsample(candidate.periods, 700).round(7).tolist(),
        power=downsample(candidate.power, 700).round(7).tolist(),
        flags=diagnostic_flags(candidate),
    )
