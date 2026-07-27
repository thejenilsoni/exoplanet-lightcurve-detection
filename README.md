# Exoplanet Light-Curve Detection

An end-to-end scientific machine-learning system for detecting and ranking exoplanet transit candidates in noisy Kepler and TESS light curves.

The project combines robust time-series preprocessing, Box Least Squares transit search, engineered astrophysical diagnostics, a one-dimensional neural classifier, explainable candidate scoring, and a Next.js analysis console.

## Capabilities

- CSV and FITS light-curve ingestion
- Quality filtering, sigma clipping, normalization, and detrending
- Box Least Squares period search
- Transit depth, duration, signal-to-noise, odd-even, and secondary-eclipse diagnostics
- AI candidate classification with a calibrated heuristic fallback
- Phase-folded light-curve and periodogram outputs
- Next.js candidate-analysis workspace
- FastAPI inference service
- Training, evaluation, tests, Docker, and CI workflows

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

Open `http://localhost:3000` for the analysis console and `http://localhost:8000/docs` for the API explorer.

## Scientific status

The repository includes a fully testable detection baseline and the complete model-training path. A trained neural checkpoint is intentionally not bundled. Candidate scores are screening evidence and do not constitute astronomical confirmation; follow-up validation remains necessary.

See `docs/architecture.md` and `docs/methodology.md` for the system design and evaluation protocol.

## License

MIT
