#!/usr/bin/env python3
"""Run and diagnose the CA_1 zero-shot Timestral vs seasonal-naive baseline.

Expected inputs under data/processed/subsets/<subset>:
  sales.npy, price.npy, series.parquet

The script preserves the original resumable batch inference and baseline outputs,
then adds per-series, live-adjusted demand-class, forecast-horizon, and aggregate-bias diagnostics.
Demand classes are computed from training days only (d_1..d_1913).

Run from the project root:
  pixi run python scripts/run_ca1_timestral_baseline.py --subset ca_1 --device cpu

To regenerate diagnostics from existing forecasts without loading Timestral:
  pixi run python scripts/run_ca1_timestral_baseline.py --subset ca_1 --diagnostics-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from timestral import TimestralForecaster

TRAIN_DAYS = 1913
HORIZON = 28
MODEL_ID = "mistralai/Timestral-209M"
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49
HORIZON_BUCKETS = ((1, 7), (8, 14), (15, 21), (22, 28))


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    denom = float(np.abs(y).sum())
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "wape": float(np.abs(err).sum() / denom) if denom else float("nan"),
        "bias": float(err.mean()),
        "smape": float(np.mean(2 * np.abs(err) / np.maximum(np.abs(y) + np.abs(pred), 1e-8))),
    }


def _wrmsse_scales(train: np.ndarray, first_sale_d: np.ndarray) -> np.ndarray:
    """Training-only M5-style naive-1 scales, one per bottom-level series."""
    scales = np.full(train.shape[0], np.nan, dtype=np.float64)
    for i, first_day in enumerate(first_sale_d):
        start = max(0, int(first_day) - 1) if first_day is not None and np.isfinite(first_day) else 0
        diffs = np.diff(train[i, start:].astype(np.float64))
        if diffs.size:
            scale = float(np.mean(diffs**2))
            if scale > 0 and np.isfinite(scale):
                scales[i] = scale
    return scales


def _rmsse_per_series(y: np.ndarray, pred: np.ndarray, scales: np.ndarray) -> np.ndarray:
    mse = np.mean((pred - y) ** 2, axis=1)
    out = np.full(y.shape[0], np.nan, dtype=np.float64)
    valid = np.isfinite(scales) & (scales > 0)
    out[valid] = np.sqrt(mse[valid] / scales[valid])
    return out


def _weighted_score(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    total = float(weights[valid].sum())
    return float(np.sum(values[valid] * weights[valid]) / total) if total else float("nan")


def _win_rate(candidate: np.ndarray, reference: np.ndarray) -> float:
    valid = np.isfinite(candidate) & np.isfinite(reference)
    return float(np.mean(candidate[valid] < reference[valid])) if valid.any() else float("nan")


def _wrmsse_style(
    y: np.ndarray, pred: np.ndarray, scales: np.ndarray, weights: np.ndarray
) -> tuple[float, int, float]:
    rmsse = _rmsse_per_series(y, pred, scales)
    valid = np.isfinite(rmsse) & np.isfinite(weights) & (weights > 0)
    excluded_weight = float(weights[~valid].sum())
    return _weighted_score(rmsse, weights), int((~valid).sum()), excluded_weight


def _weighted_pinball(
    y: np.ndarray, qhat: np.ndarray, levels: np.ndarray, weights: np.ndarray
) -> float:
    # qhat: (series, horizon, quantile)
    error = y[:, :, None] - qhat
    loss = np.maximum(levels[None, None, :] * error, (levels[None, None, :] - 1) * error)
    return float(np.sum(loss * (weights / weights.sum())[:, None, None]) / (y.shape[1] * len(levels)))


def _pinball_per_series(y: np.ndarray, qhat: np.ndarray, levels: np.ndarray) -> np.ndarray:
    error = y[:, :, None] - qhat
    loss = np.maximum(levels[None, None, :] * error, (levels[None, None, :] - 1) * error)
    return loss.mean(axis=(1, 2))


def _live_adjusted_demand_class(
    train: np.ndarray, train_price: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SBC classes using price-live training days; never uses the holdout."""
    live = np.isfinite(train_price)
    positive = (train > 0) & live
    n_live = live.sum(axis=1)
    n_sale = positive.sum(axis=1)
    adi = np.divide(
        n_live, n_sale, out=np.full(train.shape[0], np.nan, dtype=np.float64), where=n_sale > 0
    )
    cv2 = np.full(train.shape[0], np.nan, dtype=np.float64)
    for i in range(train.shape[0]):
        values = train[i, positive[i]].astype(np.float64)
        if values.size and values.mean() > 0:
            cv2[i] = (values.std(ddof=0) / values.mean()) ** 2

    classes = np.full(train.shape[0], "Unclassified", dtype=object)
    no_sales = n_sale == 0
    classes[no_sales] = "NoSales"
    valid = (~no_sales) & np.isfinite(adi) & np.isfinite(cv2)
    low_adi = adi < ADI_THRESHOLD
    low_cv2 = cv2 < CV2_THRESHOLD
    classes[valid & low_adi & low_cv2] = "Smooth"
    classes[valid & low_adi & ~low_cv2] = "Erratic"
    classes[valid & ~low_adi & low_cv2] = "Intermittent"
    classes[valid & ~low_adi & ~low_cv2] = "Lumpy"
    return classes, adi, cv2, n_live


def _forecast_batches(
    forecaster: TimestralForecaster, context: np.ndarray, out_dir: Path, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    n = context.shape[0]
    batch_dir = out_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    levels_path = out_dir / "quantile_levels.npy"
    levels: np.ndarray | None = np.load(levels_path) if levels_path.exists() else None

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch_path = batch_dir / f"forecast_{start:05d}_{stop:05d}.npy"
        if batch_path.exists():
            continue
        x = context[start:stop, None, :].astype(np.float32, copy=False)
        forecast = forecaster.forecast(x, prediction_length=HORIZON)
        q = _as_numpy(forecast.quantiles).astype(np.float32, copy=False)[:, 0, :, :]
        current_levels = _as_numpy(forecast.quantile_levels).astype(np.float32, copy=False)
        if q.shape != (stop - start, HORIZON, current_levels.size):
            raise ValueError(f"Unexpected Timestral output shape: {q.shape}")
        if levels is None:
            levels = current_levels
            np.save(levels_path, levels)
        elif not np.array_equal(levels, current_levels):
            raise ValueError("Timestral quantile levels changed between batches")
        np.save(batch_path, q)
        print(f"saved series {start:,}:{stop:,} / {n:,}")

    if levels is None:
        raise RuntimeError("No forecast batches were produced")
    q_all = np.empty((n, HORIZON, levels.size), dtype=np.float32)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        q_all[start:stop] = np.load(batch_dir / f"forecast_{start:05d}_{stop:05d}.npy")
    return q_all, levels


def _write_diagnostics(
    out_dir: Path,
    series: pl.DataFrame,
    train: np.ndarray,
    actual: np.ndarray,
    naive_point: np.ndarray,
    naive_q: np.ndarray,
    timestral_q: np.ndarray,
    levels: np.ndarray,
    weights: np.ndarray,
    train_price: np.ndarray,
) -> None:
    """Write per-series, demand-class, and horizon diagnostic CSVs."""
    median_idx = int(np.argmin(np.abs(levels - 0.5)))
    ts_point = timestral_q[:, :, median_idx]
    first_sale = series["first_sale_d"].fill_null(0).to_numpy()
    scales = _wrmsse_scales(train, first_sale)
    # Use the price-availability mask from d_1..d_1913; no holdout information enters class labels.
    classes, adi, cv2, live_days = _live_adjusted_demand_class(train, train_price)

    naive_mae = np.mean(np.abs(naive_point - actual), axis=1)
    ts_mae = np.mean(np.abs(ts_point - actual), axis=1)
    naive_bias = np.mean(naive_point - actual, axis=1)
    ts_bias = np.mean(ts_point - actual, axis=1)
    actual_sum = np.abs(actual).sum(axis=1)
    naive_wape = np.divide(np.abs(naive_point - actual).sum(axis=1), actual_sum,
                           out=np.full(actual.shape[0], np.nan), where=actual_sum > 0)
    ts_wape = np.divide(np.abs(ts_point - actual).sum(axis=1), actual_sum,
                        out=np.full(actual.shape[0], np.nan), where=actual_sum > 0)
    naive_rmsse = _rmsse_per_series(actual, naive_point, scales)
    ts_rmsse = _rmsse_per_series(actual, ts_point, scales)
    naive_pinball = _pinball_per_series(actual, naive_q, levels)
    ts_pinball = _pinball_per_series(actual, timestral_q, levels)

    base_cols = series.to_dict(as_series=False)
    base_cols.update({
        "demand_class": classes.tolist(), "adi_live_adjusted": adi.tolist(),
        "cv2_live_adjusted": cv2.tolist(), "live_training_days": live_days.tolist(),
        "actual_28d_units": actual.sum(axis=1).tolist(),
        "revenue_weight": weights.tolist(),
        "seasonal_naive_mae": naive_mae.tolist(), "timestral_mae": ts_mae.tolist(),
        "mae_improvement_pct": (100 * (naive_mae - ts_mae) / np.maximum(naive_mae, 1e-12)).tolist(),
        "seasonal_naive_wape": naive_wape.tolist(), "timestral_wape": ts_wape.tolist(),
        "seasonal_naive_bias": naive_bias.tolist(), "timestral_bias": ts_bias.tolist(),
        "seasonal_naive_rmsse": naive_rmsse.tolist(), "timestral_rmsse": ts_rmsse.tolist(),
        "seasonal_naive_pinball": naive_pinball.tolist(), "timestral_pinball": ts_pinball.tolist(),
        "timestral_wins_mae": (ts_mae < naive_mae).tolist(),
        "timestral_wins_wape": [bool(a < b) if np.isfinite(a) and np.isfinite(b) else None
                                for a, b in zip(ts_wape, naive_wape)],
        "timestral_wins_rmsse": [bool(a < b) if np.isfinite(a) and np.isfinite(b) else None
                                 for a, b in zip(ts_rmsse, naive_rmsse)],
        "timestral_wins_pinball": (ts_pinball < naive_pinball).tolist(),
    })
    pl.DataFrame(base_cols).write_csv(out_dir / "per_series_diagnostics.csv")

    class_rows: list[dict[str, object]] = []
    for demand_class in ["Smooth", "Erratic", "Intermittent", "Lumpy", "NoSales", "Unclassified"]:
        idx = np.flatnonzero(classes == demand_class)
        if idx.size == 0:
            continue
        group_weights = weights[idx]
        group_weight_share = float(group_weights.sum())
        for model, point, quantiles in (
            ("SeasonalNaive7", naive_point, naive_q), ("Timestral", ts_point, timestral_q)
        ):
            m = _metrics(actual[idx], point[idx])
            per_mae = np.mean(np.abs(point[idx] - actual[idx]), axis=1)
            per_bias = np.mean(point[idx] - actual[idx], axis=1)
            per_wape = np.divide(np.abs(point[idx] - actual[idx]).sum(axis=1),
                                  np.abs(actual[idx]).sum(axis=1),
                                  out=np.full(idx.size, np.nan),
                                  where=np.abs(actual[idx]).sum(axis=1) > 0)
            rmsse = _rmsse_per_series(actual[idx], point[idx], scales[idx])
            row = {
                "demand_class": demand_class, "model": model, "n_series": int(idx.size),
                "series_share": float(idx.size / len(classes)),
                "revenue_weight_share": group_weight_share,
                **m,
                "mean_series_mae": float(np.mean(per_mae)),
                "revenue_weighted_series_mae": _weighted_score(per_mae, group_weights),
                "mean_series_wape": float(np.nanmean(per_wape)) if np.isfinite(per_wape).any() else float("nan"),
                "mean_series_bias": float(np.mean(per_bias)),
                "revenue_weighted_series_bias": _weighted_score(per_bias, group_weights),
                "bottom_level_subset_wrmsse_style": _weighted_score(rmsse, group_weights),
                "weighted_pinball_within_class": _weighted_pinball(actual[idx], quantiles[idx], levels, group_weights)
                    if group_weights.sum() > 0 else float("nan"),
                "timestral_win_rate_mae": _win_rate(ts_mae[idx], naive_mae[idx]),
                "timestral_win_rate_wape": _win_rate(ts_wape[idx], naive_wape[idx]),
                "timestral_win_rate_rmsse": _win_rate(ts_rmsse[idx], naive_rmsse[idx]),
                "timestral_win_rate_pinball": _win_rate(ts_pinball[idx], naive_pinball[idx]),
            }
            class_rows.append(row)
    pl.DataFrame(class_rows).write_csv(out_dir / "demand_class_diagnostics.csv")

    horizon_rows: list[dict[str, object]] = []
    for start_day, end_day in HORIZON_BUCKETS:
        sl = slice(start_day - 1, end_day)
        for model, point, quantiles in (
            ("SeasonalNaive7", naive_point, naive_q), ("Timestral", ts_point, timestral_q)
        ):
            y_part = actual[:, sl]
            p_part = point[:, sl]
            m = _metrics(y_part, p_part)
            rmsse = _rmsse_per_series(y_part, p_part, scales)
            per_series_mae = np.mean(np.abs(p_part - y_part), axis=1)
            per_series_bias = np.mean(p_part - y_part, axis=1)
            horizon_rows.append({
                "horizon": f"days_{start_day}_{end_day}",
                "start_day": start_day, "end_day": end_day, "model": model,
                "n_series": int(actual.shape[0]), **m,
                "mean_series_mae": float(np.mean(per_series_mae)),
                "revenue_weighted_series_mae": _weighted_score(per_series_mae, weights),
                "revenue_weighted_series_bias": _weighted_score(per_series_bias, weights),
                "bottom_level_subset_wrmsse_style": _weighted_score(rmsse, weights),
                "weighted_pinball": _weighted_pinball(actual[:, sl], quantiles[:, sl, :], levels, weights),
            })
    pl.DataFrame(horizon_rows).write_csv(out_dir / "horizon_diagnostics.csv")

    # Step 5.1: daily portfolio totals. Positive forecast_error means
    # overforecast; negative means underforecast. Cumulative errors show
    # whether daily errors offset or compound across the 28-day horizon.
    daily_rows: list[dict[str, object]] = []
    for day in range(HORIZON):
        y_day = actual[:, day]
        naive_day = naive_point[:, day]
        ts_day = ts_point[:, day]
        actual_total = float(y_day.sum())
        naive_total = float(naive_day.sum())
        ts_total = float(ts_day.sum())
        naive_error = naive_total - actual_total
        ts_error = ts_total - actual_total
        daily_rows.append({
            "horizon_day": day + 1,
            "m5_day": f"d_{TRAIN_DAYS + day + 1}",
            "actual_units": actual_total,
            "seasonal_naive_forecast_units": naive_total,
            "timestral_forecast_units": ts_total,
            "seasonal_naive_error_units": naive_error,
            "timestral_error_units": ts_error,
            "seasonal_naive_error_pct_actual": 100.0 * naive_error / actual_total if actual_total else float("nan"),
            "timestral_error_pct_actual": 100.0 * ts_error / actual_total if actual_total else float("nan"),
            "seasonal_naive_absolute_error_units": abs(naive_error),
            "timestral_absolute_error_units": abs(ts_error),
            "cumulative_actual_units": float(actual[:, :day + 1].sum()),
            "cumulative_seasonal_naive_forecast_units": float(naive_point[:, :day + 1].sum()),
            "cumulative_timestral_forecast_units": float(ts_point[:, :day + 1].sum()),
            "cumulative_seasonal_naive_error_units": float((naive_point[:, :day + 1] - actual[:, :day + 1]).sum()),
            "cumulative_timestral_error_units": float((ts_point[:, :day + 1] - actual[:, :day + 1]).sum()),
        })
    pl.DataFrame(daily_rows).write_csv(out_dir / "daily_aggregate_diagnostics.csv")

    # Keep series with no actual demand in the full holdout in a dedicated
    # file. Their percentage improvements are not meaningful when the
    # denominator (actual demand) is zero.
    per_series = pl.read_csv(out_dir / "per_series_diagnostics.csv")
    zero_actual = per_series.filter(pl.col("actual_28d_units") == 0)
    zero_actual.write_csv(out_dir / "zero_actual_28d_series.csv")

    print("\nWrote diagnostics:")
    for name in ("per_series_diagnostics.csv", "demand_class_diagnostics.csv", "horizon_diagnostics.csv",
                 "daily_aggregate_diagnostics.csv", "zero_actual_28d_series.csv"):
        print(f"  {out_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="ca_1")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/ca1_baseline"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="mps", help="mps or cpu")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--diagnostics-only", action="store_true",
                        help="use existing forecasts_and_actuals.npz; do not load Timestral")
    args = parser.parse_args()

    sub = args.processed / "subsets" / args.subset.lower()
    required = [sub / "sales.npy", sub / "price.npy", sub / "series.parquet"]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing prepared subset files:\n" + "\n".join(map(str, missing)))

    sales = np.load(sub / "sales.npy", mmap_mode="r")
    price = np.load(sub / "price.npy", mmap_mode="r")
    series = pl.read_parquet(sub / "series.parquet")
    if sales.shape != price.shape or sales.shape[0] != series.height:
        raise ValueError(f"Row/shape mismatch: sales={sales.shape}, price={price.shape}, metadata={series.height}")
    if sales.shape[1] != 1941:
        raise ValueError(f"Expected 1,941 days, found {sales.shape[1]}")
    if args.subset.lower() != "ca_1":
        print("WARNING: this script was designed around CA_1; confirm the requested subset.")

    train = np.asarray(sales[:, :TRAIN_DAYS], dtype=np.float32)
    actual = np.asarray(sales[:, TRAIN_DAYS : TRAIN_DAYS + HORIZON], dtype=np.float32)
    train_price = np.asarray(price[:, :TRAIN_DAYS], dtype=np.float32)
    context = np.where(np.isfinite(train_price), train, np.nan).astype(np.float32)
    if np.isnan(context).all(axis=1).any():
        print(f"Warning: {(np.isnan(context).all(axis=1)).sum()} series have no observed context")

    revenue = np.nansum(train[:, -28:] * np.asarray(price[:, -28:], dtype=np.float32), axis=1, dtype=np.float64)
    if revenue.sum() <= 0:
        raise ValueError("Subset has no positive revenue in the final 28 training days")
    weights = revenue / revenue.sum()
    args.output.mkdir(parents=True, exist_ok=True)
    npz_path = args.output / "forecasts_and_actuals.npz"
    if args.diagnostics_only:
        if not npz_path.is_file():
            raise FileNotFoundError(f"Cannot run diagnostics-only; missing {npz_path}")
        saved = np.load(npz_path)
        actual_saved = saved["actual"]
        if actual_saved.shape != actual.shape or not np.array_equal(actual_saved, actual):
            raise ValueError("Saved forecast actuals do not match the currently prepared subset/order")
        naive_point = saved["seasonal_naive_point"]
        naive_q = saved["seasonal_naive_quantiles"]
        timestral_q = saved["timestral_quantiles"]
        levels = saved["quantile_levels"]
        # The saved weights are the weights used for the original baseline
        # metrics. Reuse them for diagnostics so those results remain directly
        # comparable, even if a later preparation run changes prices or
        # introduces tiny floating-point differences.
        if "revenue_weights" not in saved.files:
            raise ValueError("Forecast artifact is missing saved revenue_weights")
        saved_weights = np.asarray(saved["revenue_weights"], dtype=np.float64)
        if saved_weights.shape != weights.shape:
            raise ValueError(
                f"Saved revenue-weight shape {saved_weights.shape} does not match "
                f"current subset shape {weights.shape}"
            )
        if (not np.isfinite(saved_weights).all() or (saved_weights < 0).any()
                or not np.isclose(saved_weights.sum(), 1.0, rtol=1e-5, atol=1e-7)):
            raise ValueError("Saved revenue weights are invalid or do not sum to one")
        max_weight_diff = float(np.max(np.abs(saved_weights - weights)))
        if not np.allclose(weights, saved_weights, rtol=1e-5, atol=1e-7):
            print(
                "WARNING: recomputed revenue weights differ from the original run "
                f"(max absolute difference={max_weight_diff:.3e}). "
                "Using saved weights to reproduce the original baseline weighting."
            )
        weights = saved_weights
        print(f"Loaded existing forecasts from {npz_path}")
    else:
        naive_point = np.tile(train[:, -7:], (1, 4)).astype(np.float32)
        config = {
            "subset": args.subset, "train_days": TRAIN_DAYS, "horizon": HORIZON,
            "model": args.model, "device": args.device, "batch_size": args.batch_size,
            "n_series": int(sales.shape[0]), "context_missing_values": int(np.isnan(context).sum()),
            "demand_class_method": "live-adjusted SBC; training period only",
            "adi_threshold": ADI_THRESHOLD, "cv2_threshold": CV2_THRESHOLD,
        }
        (args.output / "run_config.json").write_text(json.dumps(config, indent=2))
        print(f"Loading {args.model} on {args.device}; {sales.shape[0]:,} series")
        forecaster = TimestralForecaster.from_pretrained(args.model, device=args.device)
        timestral_q, levels = _forecast_batches(forecaster, context, args.output, args.batch_size)

        residual = train[:, 7:] - train[:, :-7]
        residual_live = np.isfinite(train_price[:, 7:]) & np.isfinite(train_price[:, :-7])
        residual = np.where(residual_live, residual, np.nan)
        naive_resid_q = np.zeros((train.shape[0], levels.size), dtype=np.float64)
        for i in range(train.shape[0]):
            _r = residual[i, np.isfinite(residual[i])]
            if _r.size:
                naive_resid_q[i] = np.quantile(_r, levels)
        naive_q = np.maximum(0.0, naive_point[:, :, None] + naive_resid_q[:, None, :])

        np.savez_compressed(
            npz_path, actual=actual, seasonal_naive_point=naive_point,
            seasonal_naive_quantiles=naive_q, timestral_quantiles=timestral_q,
            quantile_levels=levels, revenue_weights=weights,
        )

    # Validate dimensions before computing diagnostics.
    expected = (sales.shape[0], HORIZON)
    if actual.shape != expected or naive_point.shape != expected or timestral_q.shape[:2] != expected:
        raise ValueError("Forecast artifact dimensions do not match the prepared subset")
    levels = np.asarray(levels, dtype=np.float32)
    ts_median = timestral_q[:, :, int(np.argmin(np.abs(levels - 0.5)))]
    scales = _wrmsse_scales(train, series["first_sale_d"].fill_null(0).to_numpy())
    naive_metrics = _metrics(actual, naive_point)
    ts_metrics = _metrics(actual, ts_median)
    naive_wrmsse, naive_excluded, naive_excluded_weight = _wrmsse_style(actual, naive_point, scales, weights)
    ts_wrmsse, ts_excluded, ts_excluded_weight = _wrmsse_style(actual, ts_median, scales, weights)
    naive_pinball = _weighted_pinball(actual, naive_q, levels, weights)
    ts_pinball = _weighted_pinball(actual, timestral_q, levels, weights)
    summary = [
        {"model": "SeasonalNaive7_point", **naive_metrics,
         "bottom_level_subset_wrmsse_style": naive_wrmsse, "weighted_pinball": naive_pinball,
         "excluded_scale_series": naive_excluded, "excluded_revenue_weight": naive_excluded_weight},
        {"model": "Timestral_median", **ts_metrics,
         "bottom_level_subset_wrmsse_style": ts_wrmsse, "weighted_pinball": ts_pinball,
         "excluded_scale_series": ts_excluded, "excluded_revenue_weight": ts_excluded_weight},
    ]
    pl.DataFrame(summary).write_csv(args.output / "metrics.csv")
    _write_diagnostics(args.output, series, train, actual, naive_point, naive_q,
                       timestral_q, levels, weights, train_price)

    print("\nMetrics (bottom-level subset; not official 12-level M5 WRMSSE):")
    for row in summary:
        print(row)
    print(f"\nSaved results under {args.output}")


if __name__ == "__main__":
    main()
