#!/usr/bin/env python3
"""Phase B T1: Timestral with six causal, past-only demand-history covariates.

Reuses the original Timestral/Seasonal-Naive cache for each independent seasonal
window, then runs a second Timestral forecast conditioned on six past-only
covariate rows. It never supplies demand-class labels to Timestral.

Expected reference cache layout (from run_m5_rolling_origin.py):
  artifacts/independent_seasons/<store>/origin_dXXXX_dYYYY/forecasts_and_actuals.npz

Outputs are written separately under artifacts/phase_b_timestral_features.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

try:
    import run_ca1_timestral_baseline as base
except ImportError as exc:  # pragma: no cover - project-environment specific
    raise ImportError(
        "Could not import run_ca1_timestral_baseline.py. Place this script beside "
        "that file in scripts/ and run it from the project root."
    ) from exc

HORIZON = 28
DEFAULT_STORES = ("ca_1", "wi_3", "tx_1")
DEFAULT_ORIGINS = (
    429, 520, 612, 657, 687,
    794, 885, 977, 1022, 1052,
    1159, 1250, 1342, 1387, 1417,
    1524, 1615, 1707, 1752, 1782,
)
FEATURE_VERSION = "t1_demand_history_v1"
FEATURE_NAMES = (
    "log1p_days_since_last_sale",
    "demand_frequency_7d",
    "demand_frequency_28d",
    "log1p_sales_volume_7d",
    "log1p_sales_volume_28d",
    "log1p_mean_positive_demand_28d",
)
WINDOW_META = {
    429: ("Spring", 2012), 520: ("Summer", 2012), 612: ("Autumn", 2012),
    657: ("Thanksgiving", 2012), 687: ("Christmas_NewYear", 2012),
    794: ("Spring", 2013), 885: ("Summer", 2013), 977: ("Autumn", 2013),
    1022: ("Thanksgiving", 2013), 1052: ("Christmas_NewYear", 2013),
    1159: ("Spring", 2014), 1250: ("Summer", 2014), 1342: ("Autumn", 2014),
    1387: ("Thanksgiving", 2014), 1417: ("Christmas_NewYear", 2014),
    1524: ("Spring", 2015), 1615: ("Summer", 2015), 1707: ("Autumn", 2015),
    1752: ("Thanksgiving", 2015), 1782: ("Christmas_NewYear", 2015),
}


def _first_observed_day(mask: np.ndarray) -> np.ndarray:
    present = mask.any(axis=1)
    first = np.argmax(mask, axis=1).astype(np.int32) + 1
    return np.where(present, first, 0).astype(np.int32)


def build_demand_covariates(train: np.ndarray, train_price: np.ndarray) -> np.ndarray:
    """Build six causal covariates aligned to the historical target timeline.

    Calculations use price-live observations only. Missing price-live days are
    excluded from rolling denominators and represented as NaN in the covariate
    rows. For a live day, the feature includes that day's observed sale because
    it is known at the end of that day (the same timestamp as the target context).

    Magnitude features use log1p to reduce scale differences across M5 items.
    Days since last sale counts elapsed calendar days. Before a first sale, it
    counts elapsed days since the first price-live day, so never-sold-to-date
    series still carry a meaningful recency signal.

    Returns float32 with shape (series, 6, history_days).
    """
    if train.ndim != 2 or train_price.shape != train.shape:
        raise ValueError(f"Expected matching 2D train/price matrices; got {train.shape}, {train_price.shape}")
    n_series, n_days = train.shape
    out = np.full((n_series, len(FEATURE_NAMES), n_days), np.nan, dtype=np.float32)
    day_idx = np.arange(n_days, dtype=np.int64)

    for i in range(n_series):
        live = np.isfinite(train_price[i])
        if not live.any():
            continue
        y = np.asarray(train[i], dtype=np.float64)
        if not np.isfinite(y[live]).all():
            raise ValueError(f"Non-finite sales on price-live days for series row {i}")
        if (y[live] < 0).any():
            raise ValueError(f"Negative sales on price-live days for series row {i}")

        observed = np.where(live, y, 0.0)
        positive = live & (y > 0)
        live_f = live.astype(np.int64)
        positive_i = positive.astype(np.int64)
        c_sales = np.cumsum(observed, dtype=np.float64)
        c_live = np.cumsum(live_f, dtype=np.int64)
        c_positive = np.cumsum(positive_i, dtype=np.int64)

        def rolling_sum(cum: np.ndarray, window: int) -> np.ndarray:
            starts = np.maximum(0, day_idx - window + 1)
            before = np.zeros(n_days, dtype=cum.dtype)
            has_before = starts > 0
            before[has_before] = cum[starts[has_before] - 1]
            return cum - before

        sales_7 = rolling_sum(c_sales, 7)
        sales_28 = rolling_sum(c_sales, 28)
        live_7 = rolling_sum(c_live, 7)
        live_28 = rolling_sum(c_live, 28)
        positive_7 = rolling_sum(c_positive, 7)
        positive_28 = rolling_sum(c_positive, 28)

        frequency_7 = np.divide(
            positive_7, live_7, out=np.zeros(n_days, dtype=np.float64), where=live_7 > 0
        )
        frequency_28 = np.divide(
            positive_28, live_28, out=np.zeros(n_days, dtype=np.float64), where=live_28 > 0
        )
        first_live_idx = int(np.flatnonzero(live)[0])
        last_sale_idx = np.maximum.accumulate(np.where(positive, day_idx, -1))
        elapsed = np.where(last_sale_idx >= 0, day_idx - last_sale_idx, day_idx - first_live_idx)
        mean_positive_28 = np.divide(
            sales_28, positive_28,
            out=np.zeros(n_days, dtype=np.float64), where=positive_28 > 0,
        )

        vals = np.stack([
            np.log1p(np.maximum(elapsed, 0)),
            frequency_7,
            frequency_28,
            np.log1p(np.maximum(sales_7, 0.0)),
            np.log1p(np.maximum(sales_28, 0.0)),
            np.log1p(np.maximum(mean_positive_28, 0.0)),
        ])
        out[i][:, live] = vals[:, live].astype(np.float32, copy=False)

    if out.shape != (n_series, len(FEATURE_NAMES), n_days):
        raise RuntimeError("Demand covariate shape invariant failed")
    return out


def _load_reference(path: Path, actual: np.ndarray, n_series: int, origin: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing T0 reference cache: {path}. Run the independent-season "
            "Timestral evaluation first, using artifacts/independent_seasons."
        )
    required = {
        "actual", "seasonal_naive_point", "seasonal_naive_quantiles",
        "timestral_quantiles", "quantile_levels", "revenue_weights",
    }
    with np.load(path, allow_pickle=False) as saved:
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"Reference cache {path} is missing {sorted(missing)}")
        data = {key: np.asarray(saved[key]).copy() for key in required}
        if "forecast_start_day" in saved.files:
            saved_origin = int(np.asarray(saved["forecast_start_day"]).item())
            if saved_origin != origin:
                raise ValueError(f"{path} is d_{saved_origin}, expected d_{origin}")
    expected = (n_series, HORIZON)
    if data["actual"].shape != expected or not np.array_equal(data["actual"], actual):
        raise ValueError(f"Reference actuals do not match {path}; check subset row order")
    if data["timestral_quantiles"].shape[:2] != expected:
        raise ValueError(f"Unexpected T0 Timestral quantile shape in {path}")
    if data["seasonal_naive_point"].shape != expected:
        raise ValueError(f"Unexpected Seasonal Naive point shape in {path}")
    weights = data["revenue_weights"].astype(np.float64, copy=False)
    if weights.shape != (n_series,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(f"Invalid saved revenue weights in {path}")
    if not np.isclose(weights.sum(), 1.0, rtol=1e-5, atol=1e-7):
        raise ValueError(f"Saved revenue weights do not sum to 1 in {path}")
    return data


def _forecast_batches(
    forecaster: Any,
    context: np.ndarray,
    past_covariates: np.ndarray,
    out_dir: Path,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_series = context.shape[0]
    if past_covariates.shape != (n_series, len(FEATURE_NAMES), context.shape[1]):
        raise ValueError(
            f"Covariate shape {past_covariates.shape} does not match context {context.shape}"
        )
    batch_dir = out_dir / "t1_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    levels_path = out_dir / "t1_quantile_levels.npy"
    levels: np.ndarray | None = np.load(levels_path) if levels_path.exists() else None

    for start in range(0, n_series, batch_size):
        stop = min(start + batch_size, n_series)
        batch_path = batch_dir / f"forecast_{start:05d}_{stop:05d}.npy"
        if batch_path.exists():
            continue
        x = context[start:stop, None, :].astype(np.float32, copy=False)
        cov = past_covariates[start:stop].astype(np.float32, copy=False)
        result = forecaster.forecast(
            x,
            prediction_length=HORIZON,
            past_covariates=cov,
        )
        q = base._as_numpy(result.quantiles).astype(np.float32, copy=False)[:, 0, :, :]
        current_levels = base._as_numpy(result.quantile_levels).astype(np.float32, copy=False)
        if q.shape != (stop - start, HORIZON, current_levels.size):
            raise ValueError(f"Unexpected T1 Timestral output shape: {q.shape}")
        if levels is None:
            levels = current_levels
            np.save(levels_path, levels)
        elif not np.array_equal(levels, current_levels):
            raise ValueError("Timestral quantile levels changed between batches")
        if not np.isfinite(q).all():
            raise ValueError(f"Non-finite quantile output for batch {start}:{stop}")
        np.save(batch_path, q)
        print(f"      T1 saved series {start:,}:{stop:,} / {n_series:,}")

    if levels is None:
        raise RuntimeError("No T1 quantile batches were produced")
    q_all = np.empty((n_series, HORIZON, levels.size), dtype=np.float32)
    for start in range(0, n_series, batch_size):
        stop = min(start + batch_size, n_series)
        q_all[start:stop] = np.load(batch_dir / f"forecast_{start:05d}_{stop:05d}.npy")
    return q_all, levels


def _load_or_forecast_t1(
    cache_path: Path,
    forecaster: Any,
    context: np.ndarray,
    covariates: np.ndarray,
    actual: np.ndarray,
    origin: int,
    batch_size: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.is_file() and not force:
        with np.load(cache_path, allow_pickle=False) as saved:
            version = str(np.asarray(saved["feature_version"]).item())
            saved_origin = int(np.asarray(saved["forecast_start_day"]).item())
            saved_actual = np.asarray(saved["actual"]).copy()
            q = np.asarray(saved["timestral_t1_quantiles"]).copy()
            levels = np.asarray(saved["quantile_levels"]).copy()
            names = tuple(str(x) for x in saved["feature_names"].tolist())
        if version != FEATURE_VERSION or names != FEATURE_NAMES:
            raise ValueError(f"Feature cache {cache_path} has a different feature definition; use --force")
        if saved_origin != origin or not np.array_equal(saved_actual, actual):
            raise ValueError(f"T1 cache {cache_path} does not match this origin/actual array")
        if q.shape[:2] != actual.shape:
            raise ValueError(f"Invalid T1 quantile array shape in {cache_path}")
        print(f"    Reusing T1 forecast cache {cache_path}")
        return q, levels

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out_batches = cache_path.parent / "feature_inference"
    if force and out_batches.exists():
        shutil.rmtree(out_batches)
    q, levels = _forecast_batches(forecaster, context, covariates, out_batches, batch_size)
    np.savez_compressed(
        cache_path,
        actual=actual,
        timestral_t1_quantiles=q,
        quantile_levels=levels,
        feature_names=np.asarray(FEATURE_NAMES, dtype="U48"),
        feature_version=np.asarray(FEATURE_VERSION),
        forecast_start_day=np.asarray(origin, dtype=np.int32),
    )
    print(f"    Saved T1 forecast cache {cache_path}")
    return q, levels


def _metric_rows(
    store: str,
    origin: int,
    actual: np.ndarray,
    reference: dict[str, np.ndarray],
    t1_q: np.ndarray,
    t1_levels: np.ndarray,
    train: np.ndarray,
    train_price: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    levels = reference["quantile_levels"].astype(np.float32, copy=False)
    if not np.array_equal(levels, t1_levels):
        raise ValueError("T0 and T1 quantile levels do not match")
    t0_q = reference["timestral_quantiles"].astype(np.float32, copy=False)
    weights = reference["revenue_weights"].astype(np.float64, copy=False)
    median_idx = int(np.argmin(np.abs(levels - 0.5)))
    points = {
        "Timestral_T0": t0_q[:, :, median_idx],
        "Timestral_T1_demand_features": t1_q[:, :, median_idx],
    }
    quantiles = {"Timestral_T0": t0_q, "Timestral_T1_demand_features": t1_q}
    first_sale = _first_observed_day(train > 0)
    scales = base._wrmsse_scales(train, first_sale)
    end_day = origin + HORIZON - 1
    season, year = WINDOW_META.get(origin, ("Custom", -1))
    rows: list[dict[str, Any]] = []

    actual_total = float(actual.sum(dtype=np.float64))
    for name, point in points.items():
        overall = base._metrics(actual, point)
        wrmsse, excluded, excluded_weight = base._wrmsse_style(actual, point, scales, weights)
        qloss = base._weighted_pinball(actual, quantiles[name], levels, weights)
        forecast_total = float(point.sum(dtype=np.float64))
        agg_error = forecast_total - actual_total
        rows.append({
            "store": store.upper(),
            "forecast_start_day": origin,
            "forecast_end_day": end_day,
            "season": season,
            "season_year": year,
            "evaluation_period": "development_2012_2014" if year <= 2014 else "holdout_2015",
            "train_days": origin - 1,
            "model": name,
            "n_series": int(actual.shape[0]),
            **overall,
            "bottom_level_subset_wrmsse_style": wrmsse,
            "weighted_pinball": qloss,
            "excluded_scale_series": excluded,
            "excluded_revenue_weight": excluded_weight,
            "actual_units_total": actual_total,
            "forecast_units_total": forecast_total,
            "aggregate_error_units": agg_error,
            "aggregate_error_pct_actual": 100.0 * agg_error / actual_total if actual_total > 0 else float("nan"),
            "aggregate_forecast_to_actual_pct": 100.0 * forecast_total / actual_total if actual_total > 0 else float("nan"),
            "underforecast": bool(agg_error < 0),
        })

    # The demand classes are computed only after the forecasts and used solely
    # for explanatory diagnostics, never as covariates or input labels.
    classes, _, _, _ = base._live_adjusted_demand_class(train, train_price)
    class_rows: list[dict[str, Any]] = []
    for demand_class in ("Smooth", "Erratic", "Intermittent", "Lumpy", "NoSales", "Unclassified"):
        idx = np.flatnonzero(classes == demand_class)
        if not idx.size:
            continue
        class_actual = actual[idx]
        class_actual_total = float(class_actual.sum(dtype=np.float64))
        class_weights = weights[idx]
        for name, point in points.items():
            class_forecast_total = float(point[idx].sum(dtype=np.float64))
            class_m = base._metrics(class_actual, point[idx])
            class_wrmsse, excluded_n, excluded_w = base._wrmsse_style(
                class_actual, point[idx], scales[idx], class_weights
            )
            class_rows.append({
                "store": store.upper(), "forecast_start_day": origin,
                "season": season, "season_year": year,
                "demand_class_posthoc": demand_class, "model": name,
                "n_series": int(idx.size), "revenue_weight_share": float(class_weights.sum()),
                **class_m,
                "actual_units_total": class_actual_total,
                "forecast_units_total": class_forecast_total,
                "aggregate_error_units": class_forecast_total - class_actual_total,
                "aggregate_error_pct_actual": (
                    100.0 * (class_forecast_total - class_actual_total) / class_actual_total
                    if class_actual_total > 0 else float("nan")
                ),
                "bottom_level_subset_wrmsse_style": class_wrmsse,
                "excluded_scale_series": excluded_n,
                "excluded_revenue_weight": excluded_w,
            })
    return rows, class_rows


def _write_summaries(output: Path, metric_rows: list[dict[str, Any]], class_rows: list[dict[str, Any]]) -> None:
    metrics = pl.DataFrame(metric_rows).sort(["store", "forecast_start_day", "model"])
    metrics.write_csv(output / "phase_b_metrics.csv")
    comparisons: list[dict[str, Any]] = []
    for (store, origin), group in metrics.group_by(["store", "forecast_start_day"], maintain_order=True):
        recs = {r["model"]: r for r in group.to_dicts()}
        t0, t1 = recs["Timestral_T0"], recs["Timestral_T1_demand_features"]
        comparisons.append({
            "store": store, "forecast_start_day": origin,
            "forecast_end_day": t1["forecast_end_day"], "season": t1["season"],
            "season_year": t1["season_year"], "evaluation_period": t1["evaluation_period"],
            "mae_improvement_pct_t1_vs_t0": 100 * (t0["mae"] - t1["mae"]) / t0["mae"] if t0["mae"] else float("nan"),
            "rmse_improvement_pct_t1_vs_t0": 100 * (t0["rmse"] - t1["rmse"]) / t0["rmse"] if t0["rmse"] else float("nan"),
            "wape_improvement_pct_t1_vs_t0": 100 * (t0["wape"] - t1["wape"]) / t0["wape"] if t0["wape"] else float("nan"),
            "wrmsse_style_improvement_pct_t1_vs_t0": 100 * (
                t0["bottom_level_subset_wrmsse_style"] - t1["bottom_level_subset_wrmsse_style"]
            ) / t0["bottom_level_subset_wrmsse_style"] if t0["bottom_level_subset_wrmsse_style"] else float("nan"),
            "weighted_pinball_improvement_pct_t1_vs_t0": 100 * (
                t0["weighted_pinball"] - t1["weighted_pinball"]
            ) / t0["weighted_pinball"] if t0["weighted_pinball"] else float("nan"),
            "t0_aggregate_error_pct_actual": t0["aggregate_error_pct_actual"],
            "t1_aggregate_error_pct_actual": t1["aggregate_error_pct_actual"],
            "t0_forecast_to_actual_pct": t0["aggregate_forecast_to_actual_pct"],
            "t1_forecast_to_actual_pct": t1["aggregate_forecast_to_actual_pct"],
            "aggregate_error_change_percentage_points": (
                t1["aggregate_error_pct_actual"] - t0["aggregate_error_pct_actual"]
            ),
        })
    comp = pl.DataFrame(comparisons).sort(["store", "forecast_start_day"])
    comp.write_csv(output / "phase_b_comparison.csv")

    summary_metrics = [
        "mae", "rmse", "wape", "bias", "smape", "bottom_level_subset_wrmsse_style",
        "weighted_pinball", "aggregate_error_pct_actual", "aggregate_forecast_to_actual_pct",
    ]
    aggs: list[pl.Expr] = [pl.len().alias("n_windows"), pl.col("underforecast").sum().alias("underforecast_windows")]
    for field in summary_metrics:
        aggs.extend([pl.col(field).mean().alias(f"{field}_mean"), pl.col(field).std().alias(f"{field}_std")])
    by_period = metrics.group_by(["store", "model", "evaluation_period"]).agg(aggs)
    overall = metrics.group_by(["store", "model"]).agg(aggs).with_columns(
        pl.lit("all_20_windows").alias("evaluation_period")
    )
    pl.concat([overall, by_period], how="diagonal_relaxed").sort(
        ["store", "evaluation_period", "model"]
    ).write_csv(output / "phase_b_summary.csv")
    by_period.filter(pl.col("evaluation_period") == "development_2012_2014").sort(
        ["store", "model"]
    ).write_csv(output / "phase_b_development_summary.csv")
    by_period.filter(pl.col("evaluation_period") == "holdout_2015").sort(
        ["store", "model"]
    ).write_csv(output / "phase_b_2015_holdout_summary.csv")
    metrics.group_by(["store", "model", "season"]).agg(aggs).sort(
        ["store", "season", "model"]
    ).write_csv(output / "phase_b_season_summary.csv")
    pl.DataFrame(class_rows).sort(
        ["store", "forecast_start_day", "demand_class_posthoc", "model"]
    ).write_csv(output / "phase_b_demand_class_metrics.csv")

    class_comparisons: list[dict[str, Any]] = []
    cls = pl.DataFrame(class_rows)
    for (store, origin, demand_class), group in cls.group_by(
        ["store", "forecast_start_day", "demand_class_posthoc"], maintain_order=True
    ):
        recs = {r["model"]: r for r in group.to_dicts()}
        t0, t1 = recs["Timestral_T0"], recs["Timestral_T1_demand_features"]
        class_comparisons.append({
            "store": store, "forecast_start_day": origin, "demand_class_posthoc": demand_class,
            "mae_improvement_pct_t1_vs_t0": 100 * (t0["mae"] - t1["mae"]) / t0["mae"] if t0["mae"] else float("nan"),
            "wape_improvement_pct_t1_vs_t0": 100 * (t0["wape"] - t1["wape"]) / t0["wape"] if t0["wape"] else float("nan"),
            "t0_aggregate_error_pct_actual": t0["aggregate_error_pct_actual"],
            "t1_aggregate_error_pct_actual": t1["aggregate_error_pct_actual"],
            "aggregate_error_change_percentage_points": t1["aggregate_error_pct_actual"] - t0["aggregate_error_pct_actual"],
        })
    pl.DataFrame(class_comparisons).sort(
        ["store", "forecast_start_day", "demand_class_posthoc"]
    ).write_csv(output / "phase_b_demand_class_comparison.csv")


def run_feature_smoke_test() -> None:
    """Lightweight check of feature values, causality, and missing-day handling."""
    train = np.zeros((2, 40), dtype=np.float32)
    price = np.ones((2, 40), dtype=np.float32)
    train[0, [2, 10, 35]] = [4, 8, 2]
    train[1, [4, 20]] = [3, 6]
    price[0, 15] = np.nan
    price[1, :5] = np.nan
    cov = build_demand_covariates(train, price)
    assert cov.shape == (2, 6, 40)
    assert np.isnan(cov[0, :, 15]).all()
    assert np.isnan(cov[1, :, :5]).all()
    assert np.isfinite(cov[0, :, 35]).all()
    # At d_10, day 10 is the last sale (zero-based index 10).
    assert cov[0, 0, 10] == 0.0
    # A later input value must not alter features at earlier timestamps.
    changed = train.copy()
    changed[0, 39] = 999
    cov_changed = build_demand_covariates(changed, price)
    np.testing.assert_array_equal(cov[0, :, :39], cov_changed[0, :, :39])
    print("Feature smoke test passed: shape, missing-live-day masks, recency, and causality.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stores", nargs="+", default=list(DEFAULT_STORES))
    parser.add_argument("--origins", nargs="+", type=int, default=list(DEFAULT_ORIGINS))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--reference", type=Path, default=Path("artifacts/independent_seasons"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase_b_timestral_features"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default=base.MODEL_ID)
    parser.add_argument("--force", action="store_true", help="regenerate T1 forecasts and batch cache")
    parser.add_argument("--diagnostics-only", action="store_true", help="rebuild CSVs from existing T1 caches")
    parser.add_argument("--smoke-test-features", action="store_true")
    args = parser.parse_args()

    if args.smoke_test_features:
        run_feature_smoke_test()
        return
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if len(set(args.origins)) != len(args.origins):
        parser.error("--origins contains duplicate values")
    if not args.origins or not args.stores:
        parser.error("Specify at least one store and one origin")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "Phase B T1: Timestral + historical demand covariates",
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_semantics": {
            "days_since_last_sale": "log1p elapsed calendar days since latest positive sale; before first sale, age since first price-live day",
            "demand_frequency_7d_28d": "positive-sales days / price-live days in the trailing calendar window, including current day",
            "sales_volume_7d_28d": "sum of observed sales on price-live days in trailing calendar window, log1p transformed",
            "mean_positive_demand_28d": "mean units on positive-sales days in trailing 28 calendar days, log1p transformed; zero if no positive sales",
            "missing_days": "all six features set to NaN on days where sell-price matrix is not finite; such days are also masked in target context",
        },
        "leakage_control": "past-only feature rows use observations available by each context day; no holdout targets or future covariates are supplied",
        "models": ["Timestral_T0", "Timestral_T1_demand_features"],
        "stores": [s.upper() for s in args.stores],
        "origins": args.origins,
        "horizon": HORIZON,
        "device": args.device,
        "batch_size": args.batch_size,
    }
    (args.output / "phase_b_config.json").write_text(json.dumps(manifest, indent=2))

    all_metrics: list[dict[str, Any]] = []
    all_class_rows: list[dict[str, Any]] = []
    for raw_store in args.stores:
        store = raw_store.lower()
        subset_dir = args.processed / "subsets" / store
        required = [subset_dir / "sales.npy", subset_dir / "price.npy", subset_dir / "series.parquet"]
        missing = [p for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing prepared files for {store}: " + ", ".join(map(str, missing)))
        sales = np.load(subset_dir / "sales.npy", mmap_mode="r")
        prices = np.load(subset_dir / "price.npy", mmap_mode="r")
        series = pl.read_parquet(subset_dir / "series.parquet")
        if sales.shape != prices.shape or sales.shape[0] != series.height:
            raise ValueError(f"{store.upper()} subset arrays and metadata have inconsistent shapes")
        max_day = max(args.origins) + HORIZON - 1
        if max_day > sales.shape[1]:
            raise ValueError(f"Requested d_{max_day} exceeds {sales.shape[1]} stored sales days")

        print(f"\n{'=' * 76}\nStore {store.upper()} | {sales.shape[0]:,} series")
        forecaster = None
        for origin in args.origins:
            train_days = origin - 1
            end_day = origin + HORIZON - 1
            train = np.asarray(sales[:, :train_days], dtype=np.float32)
            actual = np.asarray(sales[:, train_days:end_day], dtype=np.float32)
            train_price = np.asarray(prices[:, :train_days], dtype=np.float32)
            reference_path = (
                args.reference / store / f"origin_d{origin:04d}_d{end_day:04d}" / "forecasts_and_actuals.npz"
            )
            reference = _load_reference(reference_path, actual, sales.shape[0], origin)
            ref_levels = reference["quantile_levels"].astype(np.float32, copy=False)
            covariates = build_demand_covariates(train, train_price)
            context = np.where(np.isfinite(train_price), train, np.nan).astype(np.float32)

            out_dir = args.output / store / f"origin_d{origin:04d}_d{end_day:04d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            t1_cache = out_dir / "timestral_t1_covariates.npz"
            if t1_cache.is_file() and not args.force:
                t1_q, t1_levels = _load_or_forecast_t1(
                    t1_cache, None, context, covariates, actual, origin, args.batch_size, False
                )
            else:
                if args.diagnostics_only:
                    raise FileNotFoundError(f"Missing T1 cache {t1_cache}; run without --diagnostics-only first")
                if forecaster is None:
                    print(f"  Loading {args.model} on {args.device}")
                    forecaster = base.TimestralForecaster.from_pretrained(args.model, device=args.device)
                t1_q, t1_levels = _load_or_forecast_t1(
                    t1_cache, forecaster, context, covariates, actual, origin, args.batch_size, args.force
                )
            rows, class_rows = _metric_rows(
                store, origin, actual, reference, t1_q, t1_levels, train, train_price
            )
            all_metrics.extend(rows)
            all_class_rows.extend(class_rows)
            pl.DataFrame(rows).write_csv(out_dir / "phase_b_metrics.csv")
            pl.DataFrame(class_rows).write_csv(out_dir / "phase_b_demand_class_metrics.csv")
            print(
                f"  d_{origin}..d_{end_day}: "
                f"T0/T1 metrics ready; T1 cache {'reused' if t1_cache.exists() else 'written'}"
            )
            del train, actual, train_price, covariates, context, reference, t1_q, t1_levels
            gc.collect()

        if forecaster is not None:
            del forecaster
            gc.collect()
        del sales, prices, series
        gc.collect()

    if not all_metrics:
        raise RuntimeError("No metrics were produced; check requested stores/origins and caches")
    _write_summaries(args.output, all_metrics, all_class_rows)
    print("\nPhase B T1 complete. Summary outputs:")
    for filename in (
        "phase_b_metrics.csv", "phase_b_comparison.csv", "phase_b_summary.csv",
        "phase_b_development_summary.csv", "phase_b_2015_holdout_summary.csv",
        "phase_b_season_summary.csv", "phase_b_demand_class_metrics.csv",
        "phase_b_demand_class_comparison.csv",
    ):
        print(f"  {args.output / filename}")


if __name__ == "__main__":
    main()
