# Contributing

## Setup

```bash
cp .env.example .env
make install
make dev
```

Before publishing a change:

```bash
make lint
make test
make build
```

## Scientific changes

Changes to preprocessing, search grids, features, labels, models, or score
thresholds should include:

- the hypothesis and affected failure mode;
- the target-level data split;
- before-and-after precision, recall, and injection recovery;
- probability calibration impact;
- representative false positives and false negatives.

Do not commit mission data, trained weights, credentials, or generated
artifacts. Use short imperative commit messages describing one coherent
change.
