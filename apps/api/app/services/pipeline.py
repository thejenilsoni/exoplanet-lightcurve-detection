import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from scipy.ndimage import median_filter
from scipy.special import expit


@dataclass(slots=True)
class LightCurve:
    time: np.ndarray
    flux: np.ndarray
    flux_error: np.ndarray
    quality: np.ndarray
    target_name: str
    mission: str


@dataclass(slots=True)
class ProcessedCurve:
    time: np.ndarray
    raw_normalized_flux: np.ndarray
    normalized_flux: np.ndarray
    trend: np.ndarray
    flux_error: np.ndarray


@dataclass(slots=True)
class TransitCandidate:
    period_days: float
    duration_days: float
    transit_time: float
    depth: float
    signal_to_noise: float
    odd_even_mismatch: float
    secondary_depth: float
    transit_count: int
    periods: np.ndarray
    power: np.ndarray
    phase: np.ndarray
    phase_flux: np.ndarray


def read_light_curve(payload: bytes, filename: str) -> LightCurve:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return _read_csv(payload, filename)
    if suffix in {".fits", ".fit"}:
        return _read_fits(payload, filename)
    raise ValueError("Unsupported format. Upload a CSV, FITS, or FIT light-curve file.")


def _read_csv(payload: bytes, filename: str) -> LightCurve:
    frame = pd.read_csv(io.BytesIO(payload))
    columns = {str(column).strip().lower(): column for column in frame.columns}
    time_column = _first_column(columns, ["time", "bjd", "btjd", "jd"])
    flux_column = _first_column(
        columns,
        ["pdcsap_flux", "sap_flux", "normalized_flux", "flux"],
    )
    if time_column is None or flux_column is None:
        raise ValueError("CSV must contain time and flux columns.")

    error_column = _first_column(columns, ["pdcsap_flux_err", "sap_flux_err", "flux_err", "error"])
    quality_column = _first_column(columns, ["quality", "sap_quality"])
    flux = frame[flux_column].to_numpy(dtype=np.float64)
    error = (
        frame[error_column].to_numpy(dtype=np.float64)
        if error_column is not None
        else np.full_like(flux, np.nan)
    )
    quality = (
        frame[quality_column].fillna(0).to_numpy(dtype=np.int64)
        if quality_column is not None
        else np.zeros(flux.size, dtype=np.int64)
    )
    return LightCurve(
        time=frame[time_column].to_numpy(dtype=np.float64),
        flux=flux,
        flux_error=error,
        quality=quality,
        target_name=Path(filename).stem,
        mission="Uploaded CSV",
    )


def _read_fits(payload: bytes, filename: str) -> LightCurve:
    with fits.open(io.BytesIO(payload), memmap=False) as hdus:
        table_hdu = next(
            (hdu for hdu in hdus if getattr(hdu, "data", None) is not None and hasattr(hdu.data, "names")),
            None,
        )
        if table_hdu is None:
            raise ValueError("FITS file does not contain a binary light-curve table.")
        names = {name.upper(): name for name in table_hdu.data.names}
        flux_name = next(
            (names[name] for name in ("PDCSAP_FLUX", "SAP_FLUX", "FLUX") if name in names),
            None,
        )
        if "TIME" not in names or flux_name is None:
            raise ValueError("FITS table must contain TIME and a supported flux column.")
        error_name = next(
            (names[name] for name in ("PDCSAP_FLUX_ERR", "SAP_FLUX_ERR", "FLUX_ERR") if name in names),
            None,
        )
        quality_name = next((names[name] for name in ("QUALITY", "SAP_QUALITY") if name in names), None)
        header = table_hdu.header
        primary = hdus[0].header
        flux = np.asarray(table_hdu.data[flux_name], dtype=np.float64)
        error = (
            np.asarray(table_hdu.data[error_name], dtype=np.float64)
            if error_name
            else np.full_like(flux, np.nan)
        )
        quality = (
            np.asarray(table_hdu.data[quality_name], dtype=np.int64)
            if quality_name
            else np.zeros(flux.size, dtype=np.int64)
        )
        time_values = np.asarray(table_hdu.data[names["TIME"]], dtype=np.float64)
        target = str(primary.get("OBJECT") or primary.get("TICID") or Path(filename).stem)
        mission = str(primary.get("MISSION") or header.get("TELESCOP") or "Uploaded FITS")
    return LightCurve(
        time=time_values,
        flux=flux,
        flux_error=error,
        quality=quality,
        target_name=target,
        mission=mission,
    )


def _first_column(columns: dict[str, object], candidates: list[str]) -> object | None:
    return next((columns[name] for name in candidates if name in columns), None)


def preprocess(curve: LightCurve) -> ProcessedCurve:
    valid = (
        np.isfinite(curve.time)
        & np.isfinite(curve.flux)
        & (curve.quality == 0)
    )
    time = curve.time[valid]
    flux = curve.flux[valid]
    error = curve.flux_error[valid]
    if time.size < 120:
        raise ValueError("At least 120 valid cadences are required for transit detection.")

    order = np.argsort(time)
    time, flux, error = time[order], flux[order], error[order]
    unique = np.concatenate(([True], np.diff(time) > 0))
    time, flux, error = time[unique], flux[unique], error[unique]

    median = float(np.median(flux))
    if median == 0:
        raise ValueError("Flux median is zero; normalization is undefined.")
    normalized = flux / median
    mad = float(np.median(np.abs(normalized - np.median(normalized))))
    robust_sigma = max(1.4826 * mad, 1e-8)
    clipped = np.abs(normalized - np.median(normalized)) < 8 * robust_sigma
    time, normalized, error = time[clipped], normalized[clipped], error[clipped] / abs(median)

    window = max(51, (time.size // 18) | 1)
    window = min(window, time.size - 1 if time.size % 2 == 0 else time.size)
    if window % 2 == 0:
        window -= 1
    trend = median_filter(normalized, size=max(window, 3), mode="nearest")
    trend = np.where(np.abs(trend) < 1e-8, 1.0, trend)
    detrended = normalized / trend
    detrended /= np.median(detrended)

    finite_error = np.isfinite(error) & (error > 0)
    fallback_error = max(float(np.median(np.abs(np.diff(detrended)))) / 0.954, 1e-5)
    error = np.where(finite_error, error, fallback_error)

    return ProcessedCurve(
        time=time,
        raw_normalized_flux=normalized,
        normalized_flux=detrended,
        trend=trend,
        flux_error=error,
    )


def search_transits(curve: ProcessedCurve) -> TransitCandidate:
    baseline = float(curve.time[-1] - curve.time[0])
    max_period = min(30.0, baseline / 2.0)
    if max_period <= 0.6:
        raise ValueError("Time baseline is too short for a reliable period search.")

    durations = np.linspace(0.75 / 24.0, min(10 / 24.0, max_period * 0.12), 16)
    model = BoxLeastSquares(curve.time, curve.normalized_flux, dy=curve.flux_error)
    results = model.autopower(
        durations,
        minimum_period=0.4,
        maximum_period=max_period,
        frequency_factor=1.2,
        objective="snr",
    )
    best = int(np.nanargmax(results.power))
    period = float(results.period[best])
    duration = float(results.duration[best])
    transit_time = float(results.transit_time[best])
    depth = float(max(results.depth[best], 0.0))
    snr = float(max(results.depth_snr[best], 0.0))

    mask = model.transit_mask(curve.time, period, duration, transit_time)
    event_numbers = np.floor((curve.time - transit_time + period / 2) / period).astype(int)
    odd = mask & (event_numbers % 2 != 0)
    even = mask & (event_numbers % 2 == 0)
    odd_depth = 1 - float(np.median(curve.normalized_flux[odd])) if odd.any() else depth
    even_depth = 1 - float(np.median(curve.normalized_flux[even])) if even.any() else depth
    odd_even = abs(odd_depth - even_depth) / max(depth, 1e-8)

    phase = ((curve.time - transit_time + period / 2) % period) / period - 0.5
    secondary_mask = np.abs(np.abs(phase) - 0.5) < duration / period / 2
    secondary_depth = (
        max(0.0, 1 - float(np.median(curve.normalized_flux[secondary_mask])))
        if secondary_mask.any()
        else 0.0
    )
    transit_count = int(np.unique(event_numbers[mask]).size)

    phase_order = np.argsort(phase)
    period_indexes = np.linspace(0, len(results.period) - 1, min(700, len(results.period))).astype(int)
    phase_indexes = phase_order[
        np.linspace(0, phase_order.size - 1, min(900, phase_order.size)).astype(int)
    ]
    return TransitCandidate(
        period_days=period,
        duration_days=duration,
        transit_time=transit_time,
        depth=depth,
        signal_to_noise=snr,
        odd_even_mismatch=float(odd_even),
        secondary_depth=float(secondary_depth),
        transit_count=transit_count,
        periods=np.asarray(results.period[period_indexes]),
        power=np.asarray(results.power[period_indexes]),
        phase=phase[phase_indexes],
        phase_flux=curve.normalized_flux[phase_indexes],
    )


def heuristic_probability(candidate: TransitCandidate) -> float:
    duration_fraction = candidate.duration_days / candidate.period_days
    secondary_ratio = candidate.secondary_depth / max(candidate.depth, 1e-8)
    logit = (
        -3.25
        + 0.31 * min(candidate.signal_to_noise, 25)
        + 0.16 * min(candidate.transit_count, 10)
        - 1.8 * min(candidate.odd_even_mismatch, 2)
        - 1.5 * min(secondary_ratio, 2)
        - 8.0 * max(duration_fraction - 0.12, 0)
    )
    return float(np.clip(expit(logit), 0.01, 0.99))


def diagnostic_flags(candidate: TransitCandidate) -> list[str]:
    flags = []
    flags.append("Period stable" if candidate.transit_count >= 3 else "Few observed transits")
    flags.append(
        "Odd-even consistent"
        if candidate.odd_even_mismatch < 0.25
        else "Odd-even mismatch requires review"
    )
    secondary_ratio = candidate.secondary_depth / max(candidate.depth, 1e-8)
    flags.append(
        "No significant secondary"
        if secondary_ratio < 0.35
        else "Possible secondary eclipse"
    )
    duration_fraction = candidate.duration_days / candidate.period_days
    flags.append(
        "Transit shape plausible"
        if 0.002 < duration_fraction < 0.15
        else "Duration ratio requires review"
    )
    return flags


def downsample(values: np.ndarray, maximum: int = 900) -> np.ndarray:
    if values.size <= maximum:
        return values
    indexes = np.linspace(0, values.size - 1, maximum).astype(int)
    return values[indexes]
