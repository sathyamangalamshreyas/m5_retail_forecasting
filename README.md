# M5 Forecasting with Timestral

Learning project for exploring the M5 Forecasting Accuracy dataset and evaluating Timestral against statistical baselines. Nori comparison is intentionally out of scope for this project.

## Project environments

This Pixi workspace has two environments because `fg-data-profiling` currently requires `pandas <3.0`, while the forecasting environment is kept on pandas 3.x.

- **default**: forecasting, Timestral, PyTorch, StatsForecast, UtilsForecast.
- **eda**: `fg-data-profiling`, Skimpy and D-Tale, with pandas 2.2.x/2.x.

The two environments are isolated. Do not try to import packages from one environment while running the other.

## Install

From the project root:

```bash
pixi install
pixi install --environment eda
```

## Validate environments

```bash
# Default forecasting environment
pixi run check-env

# EDA environment
pixi run --environment eda check-env
```

## Start notebooks

```bash
# EDA (pandas < 3)
pixi run --environment eda eda-notebook

# Forecasting (pandas 3.x)
pixi run forecast-notebook
```

## Data layout

Place the four downloaded CSVs in `data/raw/`:

- `sales_train_evaluation.csv`
- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

`sample_submission.csv` is not required for this learning project.

Raw data, processed datasets, generated reports and forecast artifacts are excluded from Git by default. The Pixi manifest and lockfile should be committed.

## Reproducibility

Commit `pixi.toml` and `pixi.lock`. The Timestral Git dependency is declared against the upstream repository; Pixi's lockfile records the resolved revision. Refresh it deliberately rather than updating dependencies implicitly during an experiment.
