# Detection methodology

## Scientific objective

A planetary transit produces a small, repeated reduction in a star's measured
brightness. Real light curves also contain stellar variability, spacecraft
systematics, data gaps, cosmic-ray effects, eclipsing binaries, and random
noise. This system therefore treats detection as a two-stage problem:

1. retrieve periodic box-like events with a high-recall physical search;
2. rank and vet those events with morphology and astrophysical diagnostics.

NASA's TESS Science Processing Operations Center already generates calibrated
light curves and searches for periodic transit events. This repository is an
independent, reproducible candidate-analysis implementation rather than a
replacement for mission validation products.

Primary references:

- [TESS documentation and SPOC pipeline](https://heasarc.gsfc.nasa.gov/docs/tess/documentation.html)
- [MAST TESS data products](https://archive.stsci.edu/missions-and-data/tess/data-products)
- [Lightkurve search API](https://lightkurve.github.io/lightkurve/reference/api/lightkurve.search_lightcurve.html)

## Preprocessing

The pipeline applies:

- finite-value and mission-quality filtering;
- chronological sorting and duplicate-cadence removal;
- median normalization;
- robust MAD-based outlier clipping;
- long-window median detrending;
- uncertainty fallback derived from point-to-point scatter.

Detrending can suppress long or shallow transits if its window is too short.
Production experiments should tune the window per cadence and expected
duration, and compare against mission-provided PDCSAP flux.

## Box Least Squares

For normalized flux (f(t)), Box Least Squares evaluates trial periods and
durations using a periodic box-shaped transit model. The current search spans
periods from 0.4 days to the smaller of 30 days or half the observation
baseline. A grid of durations from 0.75 to 10 hours is evaluated subject to
the period range.

The strongest peak supplies:

- candidate period and epoch;
- transit duration and depth;
- depth signal-to-noise;
- the in-transit cadence mask;
- the phase-folded morphology.

## False-positive diagnostics

### Odd-even consistency

Alternating eclipse depths can indicate an eclipsing binary whose true period
is twice the detected value. The normalized mismatch is:

[
r_{oe} = rac{|d_{odd} - d_{even}|}{max(d, epsilon)}.
]

### Secondary eclipse

Flux near phase 0.5 is inspected for a secondary event. A large
secondary-to-primary depth ratio lowers the candidate probability.

### Duration and recurrence

Implausibly long fractional duration, too few observed events, or low S/N are
surfaced explicitly rather than hidden inside one score.

## Neural classifier

`TransitFusionNet` processes two inputs:

- a robustly scaled 256-bin phase-folded flux window;
- eight normalized physical diagnostics.

Dilated residual 1D convolutions encode local ingress, flat-bottom, V-shape,
and correlated-noise patterns. Their pooled representation is fused with the
diagnostic encoder before binary classification.

Synthetic examples support smoke testing and pretraining, but final scientific
claims require mission-derived positive and negative labels.

## Evaluation protocol

Split data by target star, not by individual sectors or windows, to prevent
the same star from appearing in training and validation.

Report:

- precision, recall, F1, and precision-recall AUC;
- false positives per target;
- injection-recovery completeness over period, radius ratio, magnitude, and S/N;
- calibration error and reliability diagrams;
- performance by mission, cadence, sector, and stellar type;
- failure cases for binaries, pulsators, flares, and instrumental artifacts.

Thresholds should be chosen for the desired follow-up budget. A high candidate
probability is not astronomical confirmation. Centroid motion, contamination,
stellar parameters, independent pipelines, and follow-up observations remain
necessary.
