#!/usr/bin/env python3
"""Rolling-origin Timestral vs seasonal-naive evaluation for M5 store subsets.

This script imports the metric and diagnostic helpers from the project's
`scripts/run_ca1_timestral_baseline.py`. Place this file beside that script.

Default origins are forecast *start* day IDs. With a 28-day horizon they yield:
  d_1886..d_1913, d_1893..d_1920, d_1900..d_1927,
  d_1907..d_1934, d_1914..d_1941.

For each origin, only sales/prices strictly before the forecast start are used
for model context, seasonal-naive forecasts, demand classes, revenue weights,
RMSSE scales, and seasonal-naive residual quantiles. Forecasts are cached per
store/origin so interrupted runs can resume.

Run from project root:
  pixi run python scripts/run_m5_rolling_origin.py --stores ca_1 wi_3 tx_1 --device cpu

Regenerate CSV diagnostics from cached forecast artifacts without inference:
  pixi run python scripts/run_m5_rolling_origin.py --stores ca_1 wi_3 tx_1 --diagnostics-only
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

try:
    import run_ca1_timestral_baseline as base
except ImportError as exc:  # pragma: no cover - user-facing project setup message
    raise ImportError(
        "Could not import run_ca1_timestral_baseline.py. Place this script in the "
        "same scripts/ directory as that file, then run it from the project root."
    ) from exc

HORIZON = 28
DEFAULT_ORIGINS = (1886, 1893, 1900, 1907, 1914)


def _load_npz(path: Path, actual: np.ndarray, n_series: int, origin: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing rolling-origin forecast artifact: {path}")
    with np.load(path) as saved:
        required = {
            "actual", "seasonal_naive_point", "seasonal_naive_quantiles",
            "timestral_quantiles", "quantile_levels", "revenue_weights",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"{path} is missing required arrays: {sorted(missing)}")
        artifact = {name: np.asarray(saved[name]).copy() for name in required}
        if "forecast_start_day" in saved.files:
            saved_origin = int(np.asarray(saved["forecast_start_day"]).item())
            if saved_origin != origin:
                raise ValueError(
                    f"Artifact {path} belongs to d_{saved_origin}, not requested d_{origin}"
                )
    if artifact["actual"].shape != actual.shape or not np.array_equal(artifact["actual"], actual):
        raise ValueError(
            f"Saved actuals in {path} do not match d_{origin}..d_{origin + HORIZON - 1}. "
            "Check that the prepared subset and row order have not changed."
        )
    expected = (n_series, HORIZON)
    if artifact["seasonal_naive_point"].shape != expected:
        raise ValueError(f"Unexpected seasonal-naive shape in {path}")
    if artifact["seasonal_naive_quantiles"].shape[:2] != expected:
        raise ValueError(f"Unexpected seasonal-naive quantile shape in {path}")
    if artifact["timestral_quantiles"].shape[:2] != expected:
        raise ValueError(f"Unexpected Timestral quantile shape in {path}")
    weights = artifact["revenue_weights"].astype(np.float64, copy=False)
    if weights.shape != (n_series,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(f"Invalid saved revenue weights in {path}")
    if not np.isclose(weights.sum(), 1.0, rtol=1e-5, atol=1e-7):
        raise ValueError(f"Saved revenue weights in {path} do not sum to one")
    return artifact


def _make_naive_quantiles(
    train: np.ndarray,
    train_price: np.ndarray,
    naive_point: np.ndarray,
    levels: np.ndarray,
) -> np.ndarray:
    """Estimate baseline residual quantiles using training-only weekly errors."""
    residual = train[:, 7:] - train[:, :-7]
    residual_live = np.isfinite(train_price[:, 7:]) & np.isfinite(train_price[:, :-7])
    residual = np.where(residual_live, residual, np.nan)
    naive_resid_q = np.zeros((train.shape[0], levels.size), dtype=np.float64)
    for i in range(train.shape[0]):
        valid_residuals = residual[i, np.isfinite(residual[i])]
        if valid_residuals.size:
            naive_resid_q[i] = np.quantile(valid_residuals, levels)
    return np.maximum(0.0, naive_point[:, :, None] + naive_resid_q[:, None, :]).astype(np.float32)


def _calculate_metrics(
    store: str,
    origin: int,
    actual: np.ndarray,
    naive_point: np.ndarray,
    naive_q: np.ndarray,
    timestral_q: np.ndarray,
    levels: np.ndarray,
    weights: np.ndarray,
    scales: np.ndarray,
) -> list[dict[str, Any]]:
    median_idx = int(np.argmin(np.abs(levels - 0.5)))
    ts_point = timestral_q[:, :, median_idx]
    results: list[dict[str, Any]] = []
    for model, point, quantiles in (
        ("SeasonalNaive7_point", naive_point, naive_q),
        ("Timestral_median", ts_point, timestral_q),
    ):
        overall = base._metrics(actual, point)
        wrmsse, excluded, excluded_weight = base._wrmsse_style(actual, point, scales, weights)
        pinball = base._weighted_pinball(actual, quantiles, levels, weights)
        actual_total = float(actual.sum(dtype=np.float64))
        forecast_total = float(point.sum(dtype=np.float64))
        aggregate_error = forecast_total - actual_total
        results.append({
            "store": store.upper(),
            "forecast_start_day": origin,
            "forecast_end_day": origin + HORIZON - 1,
            "train_days": origin - 1,
            "model": model,
            "n_series": int(actual.shape[0]),
            **overall,
            "bottom_level_subset_wrmsse_style": wrmsse,
            "weighted_pinball": pinball,
            "excluded_scale_series": excluded,
            "excluded_revenue_weight": excluded_weight,
            "actual_units_total": actual_total,
            "forecast_units_total": forecast_total,
            "aggregate_error_units": aggregate_error,
            "aggregate_error_pct_actual": 100.0 * aggregate_error / actual_total
                if actual_total > 0 else float("nan"),
            "aggregate_forecast_to_actual_pct": 100.0 * forecast_total / actual_total
                if actual_total > 0 else float("nan"),
        })
    return results


def _write_summary(output: Path, metric_rows: list[dict[str, Any]]) -> None:
    metrics = pl.DataFrame(metric_rows).sort(["store", "forecast_start_day", "model"])
    metrics.write_csv(output / "rolling_origin_metrics.csv")

    numeric_metrics = [
        "mae", "rmse", "wape", "bias", "smape",
        "bottom_level_subset_wrmsse_style", "weighted_pinball",
        "aggregate_error_pct_actual", "aggregate_forecast_to_actual_pct",
    ]
    summary_exprs = []
    for name in numeric_metrics:
        summary_exprs.extend([
            pl.col(name).mean().alias(f"{name}_mean"),
            pl.col(name).std().alias(f"{name}_std"),
        ])
    metrics.group_by(["store", "model"]).agg(summary_exprs).sort(["store", "model"]).write_csv(
        output / "rolling_origin_summary.csv"
    )

    # Paired Timestral improvements for each store/origin. Positive values mean
    # Timestral has lower error than Seasonal Naive; bias is shown separately.
    comparison_rows: list[dict[str, Any]] = []
    for (store, origin), group in metrics.group_by(["store", "forecast_start_day"], maintain_order=True):
        records = {row["model"]: row for row in group.to_dicts()}
        naive = records["SeasonalNaive7_point"]
        ts = records["Timestral_median"]
        comparison_rows.append({
            "store": store,
            "forecast_start_day": origin,
            "forecast_end_day": int(naive["forecast_end_day"]),
            "mae_improvement_pct": 100.0 * (naive["mae"] - ts["mae"]) / naive["mae"]
                if naive["mae"] else float("nan"),
            "rmse_improvement_pct": 100.0 * (naive["rmse"] - ts["rmse"]) / naive["rmse"]
                if naive["rmse"] else float("nan"),
            "wape_improvement_pct": 100.0 * (naive["wape"] - ts["wape"]) / naive["wape"]
                if naive["wape"] else float("nan"),
            "wrmsse_style_improvement_pct": 100.0 * (
                naive["bottom_level_subset_wrmsse_style"] - ts["bottom_level_subset_wrmsse_style"]
            ) / naive["bottom_level_subset_wrmsse_style"]
                if naive["bottom_level_subset_wrmsse_style"] else float("nan"),
            "weighted_pinball_improvement_pct": 100.0 * (
                naive["weighted_pinball"] - ts["weighted_pinball"]
            ) / naive["weighted_pinball"] if naive["weighted_pinball"] else float("nan"),
            "seasonal_naive_aggregate_error_pct_actual": naive["aggregate_error_pct_actual"],
            "timestral_aggregate_error_pct_actual": ts["aggregate_error_pct_actual"],
            "seasonal_naive_bias": naive["bias"],
            "timestral_bias": ts["bias"],
        })
    pl.DataFrame(comparison_rows).sort(["store", "forecast_start_day"]).write_csv(
        output / "rolling_origin_comparison.csv"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stores", nargs="+", default=["ca_1", "wi_3", "tx_1"],
                        help="prepared store subsets (default: ca_1 wi_3 tx_1)")
    parser.add_argument("--origins", nargs="+", type=int, default=list(DEFAULT_ORIGINS),
                        help="forecast start M5 day IDs; default: 1886 1893 1900 1907 1914")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/rolling_origin"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="Timestral device, e.g. cpu or mps")
    parser.add_argument("--model", default=base.MODEL_ID)
    parser.add_argument("--diagnostics-only", action="store_true",
                        help="rebuild diagnostics from existing per-origin forecast NPZs; no inference")
    parser.add_argument(
        "--reuse-single-origin", action=argparse.BooleanOptionalAction, default=True,
        help="reuse each store's existing d_1914..d_1941 baseline artifact (default: enabled)",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if len(set(args.origins)) != len(args.origins):
        parser.error("--origins contains duplicate day IDs")
    if any(origin <= 28 for origin in args.origins):
        parser.error("Every forecast start day must be greater than 28")
    if not args.stores:
        parser.error("Specify at least one store with --stores")

    args.output.mkdir(parents=True, exist_ok=True)
    all_metric_rows: list[dict[str, Any]] = []
    manifest = {
        "model": args.model,
        "device": args.device,
        "batch_size": args.batch_size,
        "horizon": HORIZON,
        "stores": [store.upper() for store in args.stores],
        "forecast_start_days": args.origins,
        "forecast_windows": [
            {"start_day": origin, "end_day": origin + HORIZON - 1, "train_days": origin - 1}
            for origin in args.origins
        ],
        "demand_class_method": "live-adjusted SBC using training data available before each origin",
        "first_sale_and_first_live_metadata": "recomputed from each origin's training slice to avoid future leakage",
        "zero_share": "recomputed over price-live days available before each origin",
        "revenue_weights": "normalized store weights from the 28 days immediately before each origin",
        "seasonal_naive": "last 7 training days repeated four times; residual quantiles from training-only weekly residuals",
        "note": "Overlapping forecast windows are not independent statistical observations.",
    }
    (args.output / "rolling_origin_config.json").write_text(json.dumps(manifest, indent=2))

    for raw_store in args.stores:
        store = raw_store.lower()
        subset_dir = args.processed / "subsets" / store
        required = [subset_dir / "sales.npy", subset_dir / "price.npy", subset_dir / "series.parquet"]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Store {store.upper()} is not prepared. Missing files:\n" +
                "\n".join(str(path) for path in missing)
            )

        sales = np.load(subset_dir / "sales.npy", mmap_mode="r")
        prices = np.load(subset_dir / "price.npy", mmap_mode="r")
        series = pl.read_parquet(subset_dir / "series.parquet")
        if sales.shape != prices.shape or sales.shape[0] != series.height:
            raise ValueError(
                f"{store.upper()} shape mismatch: sales={sales.shape}, prices={prices.shape}, "
                f"metadata rows={series.height}"
            )
        max_required_day = max(args.origins) + HORIZON - 1
        if max_required_day > sales.shape[1]:
            raise ValueError(
                f"{store.upper()} has {sales.shape[1]} days, but requested origins require "
                f"through d_{max_required_day}"
            )
        if "first_sale_d" not in series.columns:
            raise ValueError(f"{store.upper()} series.parquet lacks first_sale_d")

        print(f"\n{'=' * 76}\nStore {store.upper()} | {sales.shape[0]:,} series")
        forecaster = None
        for origin in args.origins:
            train_days = origin - 1
            end_day = origin + HORIZON - 1
            out_dir = args.output / store / f"origin_d{origin:04d}_d{end_day:04d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            npz_path = out_dir / "forecasts_and_actuals.npz"

            # Only pre-origin observations are passed to either model or to any
            # training-derived diagnostic/weight calculation.
            train = np.asarray(sales[:, :train_days], dtype=np.float32)
            actual = np.asarray(sales[:, train_days:end_day], dtype=np.float32)
            train_price = np.asarray(prices[:, :train_days], dtype=np.float32)
            if train.shape[1] != train_days or actual.shape[1] != HORIZON:
                raise ValueError(f"Invalid train/holdout slice for {store.upper()} d_{origin}")
            if train.shape[1] < 35:
                raise ValueError("Training history must contain at least 35 days")

            recent_train = train[:, -28:]
            recent_price = np.asarray(prices[:, train_days - 28:train_days], dtype=np.float32)
            revenue = np.nansum(recent_train * recent_price, axis=1, dtype=np.float64)
            if not np.isfinite(revenue).all() or revenue.sum() <= 0:
                raise ValueError(f"Invalid or zero training revenue weights for {store.upper()} d_{origin}")
            computed_weights = revenue / revenue.sum()

            # The already-completed single-origin baseline is exactly the
            # d_1914..d_1941 rolling-origin window. Reuse it by default so the
            # final origin does not incur duplicate Timestral inference.
            if (not npz_path.is_file() and args.reuse_single_origin and origin == 1914):
                legacy_dirs = {
                    "ca_1": "ca1_baseline",
                    "wi_3": "wi3_baseline",
                    "tx_1": "tx1_baseline",
                }
                legacy_name = legacy_dirs.get(store)
                legacy_npz = args.output.parent / legacy_name / "forecasts_and_actuals.npz"                     if legacy_name else None
                if legacy_npz is not None and legacy_npz.is_file():
                    legacy_saved = _load_npz(legacy_npz, actual, sales.shape[0], origin)
                    np.savez_compressed(
                        npz_path, **legacy_saved,
                        forecast_start_day=np.asarray(origin, dtype=np.int32),
                        train_days=np.asarray(train_days, dtype=np.int32),
                    )
                    print(f"  d_{origin}..d_{end_day}: imported existing baseline cache {legacy_npz}")

            if npz_path.is_file():
                saved = _load_npz(npz_path, actual, sales.shape[0], origin)
                naive_point = saved["seasonal_naive_point"].astype(np.float32, copy=False)
                naive_q = saved["seasonal_naive_quantiles"].astype(np.float32, copy=False)
                timestral_q = saved["timestral_quantiles"].astype(np.float32, copy=False)
                levels = saved["quantile_levels"].astype(np.float32, copy=False)
                weights = saved["revenue_weights"].astype(np.float64, copy=False)
                max_weight_diff = float(np.max(np.abs(computed_weights - weights)))
                if not np.allclose(computed_weights, weights, rtol=1e-5, atol=1e-7):
                    print(
                        f"  d_{origin}: using cached original weights; current prep differs "
                        f"(max abs diff {max_weight_diff:.3e})"
                    )
                print(f"  d_{origin}..d_{end_day}: reusing cached forecasts")
            else:
                if args.diagnostics_only:
                    raise FileNotFoundError(
                        f"--diagnostics-only requested, but missing {npz_path}. "
                        "Run once without --diagnostics-only to generate forecasts."
                    )
                context = np.where(np.isfinite(train_price), train, np.nan).astype(np.float32)
                naive_point = np.tile(train[:, -7:], (1, 4)).astype(np.float32)
                if forecaster is None:
                    print(f"  Loading {args.model} on {args.device} for {store.upper()}")
                    forecaster = base.TimestralForecaster.from_pretrained(args.model, device=args.device)
                # The helper uses the module-level horizon, which remains 28.
                timestral_q, levels = base._forecast_batches(
                    forecaster, context, out_dir, args.batch_size
                )
                naive_q = _make_naive_quantiles(train, train_price, naive_point, levels)
                weights = computed_weights.astype(np.float64, copy=False)
                np.savez_compressed(
                    npz_path,
                    actual=actual,
                    seasonal_naive_point=naive_point,
                    seasonal_naive_quantiles=naive_q,
                    timestral_quantiles=timestral_q,
                    quantile_levels=levels,
                    revenue_weights=weights,
                    forecast_start_day=np.asarray(origin, dtype=np.int32),
                    train_days=np.asarray(train_days, dtype=np.int32),
                )
                print(f"  d_{origin}..d_{end_day}: saved forecasts")

            # Derive first-sale and first-live dates from this origin's training
            # slice. The prepared metadata's original first_sale_d/first_live_d
            # can reflect events after an earlier origin, so do not reuse them.
            def first_observed_day(mask: np.ndarray) -> np.ndarray:
                present = mask.any(axis=1)
                first = np.argmax(mask, axis=1).astype(np.int32) + 1
                return np.where(present, first, 0).astype(np.int32)

            live_mask = np.isfinite(train_price)
            first_sale_asof = first_observed_day(train > 0)
            first_live_asof = first_observed_day(live_mask)
            live_day_count = live_mask.sum(axis=1)
            zero_live_day_count = ((train == 0) & live_mask).sum(axis=1)
            zero_share_asof = np.divide(
                zero_live_day_count, live_day_count,
                out=np.full(train.shape[0], np.nan, dtype=np.float64),
                where=live_day_count > 0,
            )
            origin_series = series.with_columns([
                pl.Series("first_sale_d", first_sale_asof),
                pl.Series("first_live_d", first_live_asof),
                # Replace the all-history field from prepared metadata with a
                # live-day zero share computed strictly before this origin.
                pl.Series("zero_share", zero_share_asof),
            ])
            scales = base._wrmsse_scales(train, first_sale_asof)
            origin_metrics = _calculate_metrics(
                store, origin, actual, naive_point, naive_q, timestral_q,
                levels, weights, scales,
            )
            pl.DataFrame(origin_metrics).write_csv(out_dir / "metrics.csv")
            all_metric_rows.extend(origin_metrics)

            # The helper labels calendar-relative M5 days with this value; set
            # it for the current origin so all per-day diagnostics are correct.
            base.TRAIN_DAYS = train_days
            base._write_diagnostics(
                out_dir, origin_series, train, actual, naive_point, naive_q,
                timestral_q, levels, weights, train_price,
            )
            origin_config = {
                "store": store.upper(), "forecast_start_day": origin,
                "forecast_end_day": end_day, "train_days": train_days,
                "horizon": HORIZON, "model": args.model,
                "device": args.device, "batch_size": args.batch_size,
            }
            (out_dir / "origin_config.json").write_text(json.dumps(origin_config, indent=2))

            del train, actual, train_price, recent_train, recent_price, origin_series
            del naive_point, naive_q, timestral_q, levels, weights, scales
            gc.collect()

        # Release model before moving to the next store to keep memory bounded.
        if forecaster is not None:
            del forecaster
            gc.collect()
        del sales, prices, series
        gc.collect()

    if not all_metric_rows:
        # diagnostics-only runs still append each origin's metrics above.
        raise RuntimeError("No origin metrics were produced")
    _write_summary(args.output, all_metric_rows)
    print("\nRolling-origin evaluation complete.")
    print(f"  {args.output / 'rolling_origin_metrics.csv'}")
    print(f"  {args.output / 'rolling_origin_comparison.csv'}")
    print(f"  {args.output / 'rolling_origin_summary.csv'}")


if __name__ == "__main__":
    main()
