# Data acquisition and preparation

## Mission products

MAST publishes TESS light-curve FITS files, target-pixel files, full-frame
images, cotrending basis vectors, and data-validation products. The runtime
accepts SPOC-style FITS light curves or a portable CSV representation.

Lightkurve can search MAST for Kepler, K2, and TESS observations:

```python
import lightkurve as lk

search = lk.search_lightcurve(
    "TIC 307210830",
    mission="TESS",
    author="SPOC",
)
collection = search.download_all()
```

The data acquisition step requires network access. Runtime inference does not.

References:

- [MAST TESS archive](https://archive.stsci.edu/missions-and-data/tess)
- [Lightkurve documentation](https://lightkurve.github.io/lightkurve)
- [Reading mission data products](https://lightkurve.github.io/lightkurve/reference/io.html)

## CSV contract

Column names are case-insensitive.

| Meaning | Accepted names | Required |
|---|---|---:|
| Time | `time`, `bjd`, `btjd`, `jd` | yes |
| Flux | `pdcsap_flux`, `sap_flux`, `normalized_flux`, `flux` | yes |
| Error | `pdcsap_flux_err`, `sap_flux_err`, `flux_err`, `error` | no |
| Quality | `quality`, `sap_quality` | no |

Generate a deterministic demonstration curve:

```bash
python scripts/generate_demo_curve.py
```

Then upload `artifacts/demo-light-curve.csv` through the console or API.

## FITS contract

The first compatible binary table must contain `TIME` and one of
`PDCSAP_FLUX`, `SAP_FLUX`, or `FLUX`. Error and quality columns are used
when available.

## Classifier sample contract

Real training examples are compressed NumPy files:

```text
candidate_00001.npz
  phase_flux  float32 [256]
  features    float32 [8]
  label       float32 scalar, 0 or 1
```

The eight features follow the exact normalization in
`CandidateScorer.features`. Dataset generation must use the same versioned
feature definition as inference.

## Training

Synthetic smoke-test training:

```bash
python -m ml.train --samples 12000 --epochs 35
```

Mission-labelled training:

```bash
python -m ml.train   --data data/processed/train   --output checkpoints   --epochs 35
```

The best checkpoint is exported as
`checkpoints/transit-fusion.ts`. Configure `MODEL_PATH` to enable learned
inference.

## Leakage prevention

- Keep all sectors or quarters for one target in exactly one split.
- Separate injected copies of a source light curve with their source target.
- Record mission, sector, cadence, target ID, label provenance, and disposition.
- Preserve a fully held-out test set until model and threshold selection finish.
- Treat candidate catalogs as noisy labels unless their dispositions are final.
