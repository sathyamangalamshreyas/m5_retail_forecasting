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

# M5 Option B patch

This patch contains:

- `scripts/data_prep_with_subset.py` — corrected project-root handling, UInt16 sales storage, row-order-preserving metadata join, and stricter `--subset-only` prerequisites.
- `notebooks/01b_m5_phase1_profile.py` — updated Phase 1 diagnostics.
- `scripts/run_ca1_timestral_baseline.py` — resumable CA_1 Timestral vs. seasonal-naive baseline.

Copy these into the matching locations in your existing project. Keep your existing Pixi manifest and lockfile unchanged.

## 1. Replace/copy the preparation script

If your current script has another filename, retain that filename or update the Pixi task accordingly. Run a subset-only extraction after the original Parquet conversion:

```bash
pixi run python scripts/data_prep_with_subset.py --subset-only --stores WI_3 --subset-name wi_3 --skip-long
pixi run python scripts/data_prep_with_subset.py --subset-only --stores TX_1 --subset-name tx_1 --skip-long
```

The script now requires all four converted Parquet inputs (`calendar`, `sell_prices`, `sales_wide`, `series_meta`) before it will run in subset-only mode. It will not silently regenerate them.

The UInt16 change fixes a data-integrity issue: the validated M5 values fit UInt16, but values above 32,767 would overflow Int16.

## 2. Rerun the Phase 1 profile

CA_1 (default):

```bash
pixi run -e eda marimo edit notebooks/01b_m5_phase1_profile.py
```

Other subsets use an environment variable; no source edit is needed:

```bash
M5_SUBSET=wi_3 pixi run -e eda marimo edit notebooks/01b_m5_phase1_profile.py
M5_SUBSET=tx_1 pixi run -e eda marimo edit notebooks/01b_m5_phase1_profile.py
```

The updated notebook:

- uses the daily price availability mask, rather than assuming an item remains live forever after its first price week;
- calculates live-adjusted ADI/CV² and zero-run diagnostics only over price-available days;
- flags unusually long zero runs relative to each contiguous live block's own sale frequency using a conservative iid Bernoulli union-bound screen and Benjamini–Hochberg adjustment; this is an investigation flag, **not a stockout diagnosis**;
- reports non-zero density sensitivity for fully live windows of 7 and 28 days rather than asserting one arbitrary readiness cutoff;
- measures named-event windows over ±7 days;
- adds HOBBIES/HOUSEHOLD as descriptive negative-control references for SNAP;
- adds a like-for-like trend for the opening d_1 live cohort;
- computes full-portfolio zero-share comparisons using training days only;
- makes the subset selectable through `M5_SUBSET`.

The annual and weekly seasonal-strength tables were already present in the source notebook; they remain in the updated notebook.

## 3. Run the CA_1 Timestral baseline

Run from the project root in the forecasting environment:

```bash
pixi run python scripts/run_ca1_timestral_baseline.py --subset ca_1 --device mps --batch-size 16
```

If MPS inference encounters an issue, retry with `--device cpu`. The script downloads the model weights on first use. Forecast batches are saved individually under `artifacts/ca1_baseline/batches/`; completed batches are reused on restart.

The evaluation is fixed to:

- context: `d_1..d_1913`;
- target: `d_1914..d_1941` (28 days);
- series: all item-store series in the selected subset (CA_1 should contain 3,049);
- context values on days without a price record: NaN (missing/not live);
- point baseline: repeat the final 7 training observations four times;
- revenue weights: subset item-store dollar sales over the final 28 training days;
- probabilistic seasonal-naive reference: empirical 7-day seasonal forecast residual quantiles from training data only.

Metrics include MAE, RMSE, WAPE, bias, sMAPE, bottom-level subset-weighted RMSSE-style score, and revenue-weighted pinball loss. **The reported RMSSE-style score is not the official M5 WRMSSE**: it covers only the bottom-level series in the selected subset, not all 42,840 hierarchy nodes and 12 levels. Do not compare it directly with the competition leaderboard score.

Outputs:

- `artifacts/ca1_baseline/metrics.csv`
- `artifacts/ca1_baseline/forecasts_and_actuals.npz`
- `artifacts/ca1_baseline/quantile_levels.npy`
- `artifacts/ca1_baseline/batches/`
- `artifacts/ca1_baseline/run_config.json`

The model inference itself must be run on your Mac against your prepared local data; it has not been executed in this container.
