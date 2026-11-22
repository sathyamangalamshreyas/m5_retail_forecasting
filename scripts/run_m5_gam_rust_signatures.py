#!/usr/bin/env python3
"""Extract interpretable M5 demand signatures with mgcv-rust (no R dependency).

This is a *descriptive* GAM workflow, not a forecasting model.

For each selected item-store series and historical cutoff:
  1) fit a Binomial GAM to price-live-day sale occurrence;
  2) fit a Gaussian GAM to log1p(positive daily sales);
  3) extract smooth curves and pointwise coefficient-based intervals;
  4) calculate empirical demand/weekly descriptors and residual diagnostics.

The existing 240-series manifest from the pymgcv prototype is reused by default
so v1/v2 signatures can be compared for the same items. It is not a demand label.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import re
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

DEFAULT_STORES = ("ca_1", "wi_3", "tx_1")
DEFAULT_ORIGINS = (1342, 1387)  # train through d_1341 and d_1386 respectively
DEFAULT_SAMPLE_MANIFEST = Path("artifacts/gam_prototype/gam_sample_manifest.csv")
MODEL_VERSION = "mgcv_rust_two_part_signatures_v1"
WEEKDAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def load_gam_class():
    try:
        from mgcv_rust import Gam
    except Exception as exc:
        raise RuntimeError(
            "Could not import mgcv_rust. Install it in the dedicated environment, "
            "for example: pixi add --pypi 'mgcv-rust==0.23.2' --feature gam-rust. "
            f"Original error: {exc!r}"
        ) from exc
    return Gam


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        return float(arr[0]) if arr.size else default
    except Exception:
        return default


def _float_list(value: Any) -> list[float]:
    try:
        return np.asarray(value, dtype=np.float64).reshape(-1).tolist()
    except Exception:
        return []


def _exact_lag_corr(values: np.ndarray, day_idx: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    day_idx = np.asarray(day_idx, dtype=np.int64).reshape(-1)
    if len(values) != len(day_idx) or len(values) < 3:
        return float("nan")
    lookup = {int(d): i for i, d in enumerate(day_idx)}
    left, right = [], []
    for i, day in enumerate(day_idx):
        j = lookup.get(int(day + lag))
        if j is not None:
            left.append(values[i])
            right.append(values[j])
    if len(left) < 3:
        return float("nan")
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _max_consecutive_zero_run(y: np.ndarray, day_idx: np.ndarray) -> int:
    best = current = 0
    prev_day = None
    for val, day in zip(y, day_idx):
        day = int(day)
        if val == 0:
            current = current + 1 if prev_day is not None and day == prev_day + 1 else 1
            best = max(best, current)
        else:
            current = 0
        prev_day = day
    return int(best)


def _empirical_signature(
    *, store: str, row_index: int, item_id: str, origin: int,
    y_all: np.ndarray, price_all: np.ndarray, weekday_all: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """Calculate empirical features using only data before the given origin."""
    train_end = origin - 1
    y_hist = np.asarray(y_all[:train_end], dtype=np.float64)
    price_hist = np.asarray(price_all[:train_end], dtype=np.float64)
    live_idx = np.flatnonzero(np.isfinite(price_hist)).astype(np.int64)
    y_live = y_hist[live_idx]
    dow_live = np.asarray(weekday_all[:train_end][live_idx], dtype=np.int8)
    positive_mask = y_live > 0
    positive_idx = live_idx[positive_mask]
    positive = y_live[positive_mask]
    n_live = int(len(y_live))
    n_pos = int(len(positive))
    zero_share = float(np.mean(~positive_mask)) if n_live else float("nan")
    frequency = float(n_pos / n_live) if n_live else float("nan")
    pos_mean = float(np.mean(positive)) if n_pos else float("nan")
    pos_median = float(np.median(positive)) if n_pos else float("nan")
    pos_sd = float(np.std(positive, ddof=1)) if n_pos > 1 else float("nan")
    pos_cv = pos_sd / pos_mean if np.isfinite(pos_sd) and pos_mean > 0 else float("nan")
    gaps = np.diff(positive_idx).astype(np.float64) if n_pos >= 2 else np.asarray([], dtype=float)
    gap_mean = float(np.mean(gaps)) if len(gaps) else float("nan")
    gap_sd = float(np.std(gaps, ddof=1)) if len(gaps) > 1 else float("nan")
    gap_cv = gap_sd / gap_mean if np.isfinite(gap_sd) and gap_mean > 0 else float("nan")

    weekly_rows: list[dict[str, Any]] = []
    signature: dict[str, Any] = {
        "store": store.upper(), "row_index": int(row_index), "item_id": item_id,
        "forecast_origin": int(origin), "training_end_day": int(train_end),
        "n_calendar_training_days": int(train_end), "n_price_live_days": n_live,
        "n_positive_days": n_pos, "zero_share": zero_share,
        "demand_frequency": frequency,
        "mean_positive_demand": pos_mean, "median_positive_demand": pos_median,
        "positive_demand_sd": pos_sd, "positive_demand_cv": pos_cv,
        "positive_demand_q90": float(np.quantile(positive, 0.9)) if n_pos else float("nan"),
        "positive_demand_max": float(np.max(positive)) if n_pos else float("nan"),
        "mean_interarrival_days": gap_mean, "sd_interarrival_days": gap_sd,
        "cv_interarrival_days": gap_cv,
        "median_interarrival_days": float(np.median(gaps)) if len(gaps) else float("nan"),
        "max_consecutive_zero_live_days": _max_consecutive_zero_run(y_live, live_idx),
        "sales_acf_lag1_calendar": _exact_lag_corr(y_live, live_idx, 1),
        "sales_acf_lag7_calendar": _exact_lag_corr(y_live, live_idx, 7),
        "occurrence_acf_lag7_calendar": _exact_lag_corr(positive_mask.astype(float), live_idx, 7),
        "price_live_fraction": n_live / train_end if train_end else float("nan"),
    }

    weekday_daily_means = np.full(7, np.nan, dtype=np.float64)
    weekday_freqs = np.full(7, np.nan, dtype=np.float64)
    weekday_counts = np.zeros(7, dtype=np.int64)
    for dow in range(7):
        mask = dow_live == dow
        weekday_counts[dow] = int(mask.sum())
        if mask.any():
            weekday_daily_means[dow] = float(np.mean(y_live[mask]))
            weekday_freqs[dow] = float(np.mean(positive_mask[mask]))
        weekly_rows.append({
            "store": store.upper(), "row_index": int(row_index), "item_id": item_id,
            "forecast_origin": int(origin), "weekday": dow,
            "weekday_name": WEEKDAY_NAMES[dow], "n_price_live_days": int(mask.sum()),
            "positive_sales_frequency": weekday_freqs[dow],
            "mean_sales_per_price_live_day": weekday_daily_means[dow],
        })
        signature[f"weekday_{WEEKDAY_NAMES[dow]}_mean_sales"] = weekday_daily_means[dow]
        signature[f"weekday_{WEEKDAY_NAMES[dow]}_positive_frequency"] = weekday_freqs[dow]

    weekday_mean = float(np.nanmean(weekday_daily_means[:5])) if np.isfinite(weekday_daily_means[:5]).any() else float("nan")
    weekend_mean = float(np.nanmean(weekday_daily_means[5:])) if np.isfinite(weekday_daily_means[5:]).any() else float("nan")
    signature["weekend_lift"] = weekend_mean / weekday_mean if np.isfinite(weekday_mean) and weekday_mean > 0 and np.isfinite(weekend_mean) else float("nan")
    signature["weekday_sales_profile_sd"] = float(np.nanstd(weekday_daily_means)) if np.isfinite(weekday_daily_means).any() else float("nan")
    signature["weekday_sales_profile_mean"] = float(np.nanmean(weekday_daily_means)) if np.isfinite(weekday_daily_means).any() else float("nan")
    signature["weekday_observation_min"] = int(weekday_counts.min())
    signature["weekday_observation_max"] = int(weekday_counts.max())
    return signature, weekly_rows, live_idx, y_live, dow_live


def _deviance_and_aic(
    family: str, y: np.ndarray, fitted: np.ndarray, edf_smooth: float
) -> dict[str, float]:
    """Compute transparent response-scale deviance and approximate conditional AIC.

    AIC uses fitted likelihood and effective df (smooth EDF + intercept, and
    Gaussian scale for the Gaussian model). It is reported as an approximation;
    only compare within the same family/component on comparable observations.
    """
    y = np.asarray(y, dtype=np.float64)
    fitted = np.asarray(fitted, dtype=np.float64)
    if len(y) != len(fitted) or len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(fitted).all():
        return {"deviance": float("nan"), "null_deviance": float("nan"),
                "deviance_explained": float("nan"), "aic_conditional_approx": float("nan")}
    n = len(y)
    if family == "binomial":
        mu = np.clip(fitted, 1e-10, 1 - 1e-10)
        p0 = float(np.clip(np.mean(y), 1e-10, 1 - 1e-10))
        dev = -2.0 * float(np.sum(y * np.log(mu) + (1.0 - y) * np.log1p(-mu)))
        null_dev = -2.0 * float(np.sum(y * math.log(p0) + (1.0 - y) * math.log1p(-p0)))
        loglik = -0.5 * dev
        k_eff = float(edf_smooth) + 1.0  # smooth EDF plus intercept
    else:
        resid = y - fitted
        dev = float(np.sum(resid**2))
        null_dev = float(np.sum((y - np.mean(y))**2))
        sigma2 = max(dev / n, 1e-12)
        loglik = -0.5 * n * (math.log(2.0 * math.pi * sigma2) + 1.0)
        k_eff = float(edf_smooth) + 2.0  # smooth EDF, intercept and scale
    explained = 1.0 - dev / null_dev if null_dev > 1e-12 else float("nan")
    return {
        "deviance": dev, "null_deviance": null_dev,
        "deviance_explained": explained,
        "aic_conditional_approx": -2.0 * loglik + 2.0 * k_eff,
    }


def _residual_diagnostics(
    family: str, y: np.ndarray, fitted: np.ndarray, day_idx: np.ndarray
) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    fitted = np.asarray(fitted, dtype=np.float64)
    if family == "binomial":
        mu = np.clip(fitted, 1e-8, 1 - 1e-8)
        residual = (y - mu) / np.sqrt(mu * (1 - mu))
    else:
        residual = y - fitted
        scale = float(np.std(residual, ddof=1)) if len(residual) > 1 else float("nan")
        if np.isfinite(scale) and scale > 0:
            residual = residual / scale
    return {
        "pearson_residual_mean": float(np.mean(residual)) if len(residual) else float("nan"),
        "pearson_residual_sd": float(np.std(residual, ddof=1)) if len(residual) > 1 else float("nan"),
        "residual_acf_lag1_calendar": _exact_lag_corr(residual, day_idx, 1),
        "residual_acf_lag7_calendar": _exact_lag_corr(residual, day_idx, 7),
        "residual_acf_lag28_calendar": _exact_lag_corr(residual, day_idx, 28),
    }


def _extract_model_stats(model: Any, family: str, y: np.ndarray, fitted: np.ndarray, day_idx: np.ndarray) -> dict[str, Any]:
    edf = _float_list(getattr(model, "edf_", []))
    lambdas = _float_list(getattr(model, "lambda_", []))
    basis_k = _float_list(getattr(model, "k_", []))
    basis_names = [str(x) for x in getattr(model, "bs_", [])] if getattr(model, "bs_", None) is not None else []
    dev = _deviance_and_aic(family, y, fitted, sum(edf))
    residuals = _residual_diagnostics(family, y, fitted, day_idx)
    summary = str(model.summary())
    # Optional package attributes are probed, never assumed to exist.
    pkg_aic = _float(getattr(model, "aic_", getattr(model, "aic", np.nan)))
    pkg_deviance = _float(getattr(model, "deviance_", np.nan))
    convergence = None
    for attr in ("converged_", "fit_converged_", "converged"):
        if hasattr(model, attr):
            value = getattr(model, attr)
            try:
                convergence = bool(np.asarray(value).reshape(-1)[0])
            except Exception:
                convergence = str(value)
            break
    return {
        "fit_status": "fit_completed_finite_predictions",
        "optimizer_convergence_exposed_by_package": convergence,
        "family": family, "n_observations": int(len(y)),
        "edf_json": json.dumps(edf), "lambda_json": json.dumps(lambdas),
        "k_json": json.dumps(basis_k), "basis_json": json.dumps(basis_names),
        "edf_total_smooth": float(sum(edf)),
        "edf_time": float(edf[0]) if edf else float("nan"),
        "lambda_time": float(lambdas[0]) if lambdas else float("nan"),
        "deviance_package_attr": pkg_deviance,
        "aic_package_attr": pkg_aic,
        **dev, **residuals,
        "summary_text": summary,
    }


def _fit_gam_component(
    Gam: Any, family: str, x: np.ndarray, y: np.ndarray, k: int,
    ci_samples: int, curve_points: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    frame = pd.DataFrame({"elapsed_day": x})
    model = Gam(
        family=family,
        method="REML",
        k_default=int(k),
        term_k_mapping={"elapsed_day": int(k)},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(frame, y)
        fitted = np.asarray(model.predict(frame), dtype=np.float64).reshape(-1)
    if len(fitted) != len(y) or not np.isfinite(fitted).all():
        raise ValueError("GAM fit returned non-finite or incorrectly sized training predictions")
    stats = _extract_model_stats(model, family, y, fitted, np.asarray(x, dtype=np.int64))
    stats["python_warning_count"] = len(caught)
    stats["python_warnings_json"] = json.dumps([str(w.message) for w in caught])

    grid = np.linspace(float(np.min(x)), float(np.max(x)), curve_points, dtype=np.float64)
    grid_X = pd.DataFrame({"elapsed_day": grid})
    mean, lower, upper = model.predict_ci(grid_X, level=0.95, n_samples=ci_samples)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    if not (len(mean) == len(lower) == len(upper) == len(grid)):
        raise ValueError("predict_ci returned an unexpected curve shape")
    if not (np.isfinite(mean).all() and np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("predict_ci returned non-finite curve values")

    curve_rows: list[dict[str, Any]] = []
    for xx, mm, lo, hi in zip(grid, mean, lower, upper):
        if family == "binomial":
            mm, lo, hi = np.clip([mm, lo, hi], 0.0, 1.0)
            value_scale = "probability_response_scale"
        else:
            # The Gaussian response is log1p(positive sales). expm1(mean log)
            # is a median-like typical positive magnitude, not arithmetic mean.
            # Clamp back-transformed bounds at zero because sales are positive.
            mm, lo, hi = np.maximum(np.expm1([mm, lo, hi]), 0.0)
            value_scale = "exp_of_fitted_log1p_typical_size_nonnegative"
        curve_rows.append({
            "elapsed_day": float(xx), "curve_value": float(mm),
            "lower_95": float(lo), "upper_95": float(hi),
            "value_scale": value_scale,
            "interval_type": "pointwise coefficient-based 95% confidence interval for fitted smooth mean; not predictive interval",
        })
    stats["curve_value_start"] = curve_rows[0]["curve_value"]
    stats["curve_value_end"] = curve_rows[-1]["curve_value"]
    stats["curve_change_end_minus_start"] = stats["curve_value_end"] - stats["curve_value_start"]
    duration_days = max(float(np.max(x) - np.min(x)), 1.0)
    stats["curve_change_per_365d"] = stats["curve_change_end_minus_start"] * 365.25 / duration_days
    curve_values = np.asarray([r["curve_value"] for r in curve_rows], dtype=np.float64)
    stats["curve_range_p95_p05"] = float(np.quantile(curve_values, 0.95) - np.quantile(curve_values, 0.05))
    stats["curve_mean"] = float(np.mean(curve_values))
    return stats, curve_rows


def _fit_one_series_origin(
    Gam: Any, store: str, row_index: int, item_id: str, origin: int,
    sales_row: np.ndarray, price_row: np.ndarray, weekday_all: np.ndarray,
    min_live_days: int, min_positive_days: int, k_trend: int,
    ci_samples: int, curve_points: int,
) -> dict[str, Any]:
    signature, weekly_rows, live_idx, y_live, dow_live = _empirical_signature(
        store=store, row_index=row_index, item_id=item_id, origin=origin,
        y_all=sales_row, price_all=price_row, weekday_all=weekday_all,
    )
    n_live = len(live_idx)
    n_positive = int(np.sum(y_live > 0))
    positive_mask = y_live > 0
    first_live = int(live_idx[0]) if n_live else None
    elapsed_live = live_idx - first_live if n_live else np.asarray([], dtype=np.int64)
    diagnostics: list[dict[str, Any]] = []
    all_curves: list[dict[str, Any]] = []
    component_features: dict[str, Any] = {
        "occurrence_fit_status": "not_attempted",
        "positive_size_fit_status": "not_attempted",
    }

    for component in ("occurrence", "positive_size"):
        base_diag = {
            "store": store.upper(), "row_index": int(row_index), "item_id": item_id,
            "forecast_origin": int(origin), "component": component,
        }
        try:
            if n_live < min_live_days:
                diagnostics.append({**base_diag, "fit_status": "insufficient_live_history",
                                    "n_live_days": n_live, "n_component_observations": 0,
                                    "error": f"Only {n_live} price-live training days; minimum is {min_live_days}"})
                component_features[f"{component}_fit_status"] = "insufficient_live_history"
                continue
            if component == "occurrence":
                x = elapsed_live.astype(np.float64)
                y = positive_mask.astype(np.float64)
                n_component = len(y)
                if np.unique(y).size < 2:
                    rate = float(np.mean(y)) if len(y) else float("nan")
                    diagnostics.append({**base_diag, "fit_status": "constant_occurrence_no_smooth",
                                        "n_live_days": n_live, "n_component_observations": n_component,
                                        "constant_occurrence_rate": rate})
                    component_features.update({"occurrence_constant_rate": rate,
                                                "occurrence_fit_status": "constant_occurrence_no_smooth"})
                    continue
                k = min(k_trend, max(6, n_live // 25))
                family = "binomial"
            else:
                x = elapsed_live[positive_mask].astype(np.float64)
                positive = y_live[positive_mask]
                y = np.log1p(positive).astype(np.float64)
                n_component = len(y)
                if n_component < min_positive_days:
                    diagnostics.append({**base_diag, "fit_status": "insufficient_positive_history",
                                        "n_live_days": n_live, "n_component_observations": n_component,
                                        "error": f"Only {n_component} positive days; minimum is {min_positive_days}"})
                    component_features["positive_size_fit_status"] = "insufficient_positive_history"
                    continue
                if np.unique(y).size < 2:
                    constant_typical = float(np.expm1(np.mean(y)))
                    diagnostics.append({**base_diag, "fit_status": "constant_positive_size_no_smooth",
                                        "n_live_days": n_live, "n_component_observations": n_component,
                                        "constant_typical_size": constant_typical})
                    component_features.update({"positive_size_constant_typical": constant_typical,
                                                "positive_size_fit_status": "constant_positive_size_no_smooth"})
                    continue
                k = min(k_trend, max(5, n_component // 8))
                family = "gaussian"

            started = time.perf_counter()
            stats, curve_rows = _fit_gam_component(
                Gam, family, x, y, k, ci_samples, curve_points
            )
            fit_seconds = time.perf_counter() - started
            stats.update({
                **base_diag, "fit_status": "fit_completed_finite_predictions",
                "n_live_days": n_live, "n_component_observations": n_component,
                "k_time_requested": int(k), "fit_seconds": float(fit_seconds),
            })
            diagnostics.append(stats)
            for row in curve_rows:
                all_curves.append({**base_diag, **row})
            prefix = "occurrence" if component == "occurrence" else "positive_size"
            component_features.update({
                f"{prefix}_fit_status": "fit_completed_finite_predictions",
                f"{prefix}_edf_time": stats.get("edf_time", float("nan")),
                f"{prefix}_lambda_time": stats.get("lambda_time", float("nan")),
                f"{prefix}_deviance_explained": stats.get("deviance_explained", float("nan")),
                f"{prefix}_aic_conditional_approx": stats.get("aic_conditional_approx", float("nan")),
                f"{prefix}_curve_start": stats.get("curve_value_start", float("nan")),
                f"{prefix}_curve_end": stats.get("curve_value_end", float("nan")),
                f"{prefix}_curve_change_per_365d": stats.get("curve_change_per_365d", float("nan")),
                f"{prefix}_curve_range_p95_p05": stats.get("curve_range_p95_p05", float("nan")),
            })
        except Exception as exc:
            diagnostics.append({**base_diag, "fit_status": "fit_exception",
                                "n_live_days": n_live,
                                "n_component_observations": n_live if component == "occurrence" else n_positive,
                                "error_type": type(exc).__name__, "error": str(exc)})
            component_features[f"{component}_fit_status"] = "fit_exception"
    signature.update(component_features)
    return {
        "signature": signature,
        "diagnostics": diagnostics,
        "curves": all_curves,
        "weekly": weekly_rows,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, allow_nan=True, separators=(",", ":")))
    tmp.replace(path)


def _load_manifest(path: Path, stores: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Sample manifest not found: {path}. Run the previous GAM prototype first, "
            "or pass --sample-manifest with the existing gam_sample_manifest.csv."
        )
    manifest = pd.read_csv(path)
    required = {"store", "row_index", "item_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    manifest["store"] = manifest["store"].astype(str).str.lower()
    manifest["row_index"] = manifest["row_index"].astype(int)
    manifest["item_id"] = manifest["item_id"].astype(str)
    manifest = manifest.loc[manifest["store"].isin(stores)].copy()
    counts = manifest.groupby("store").size().to_dict()
    expected = {s: 80 for s in stores}
    for store in stores:
        if counts.get(store, 0) != expected[store]:
            raise ValueError(f"Expected 80 manifest series for {store}, found {counts.get(store, 0)}")
    if manifest.duplicated(["store", "row_index"]).any():
        raise ValueError("Sample manifest contains duplicate store/row_index entries")
    return manifest.sort_values(["store", "row_index"]).reset_index(drop=True)


def _write_csv(rows: list[dict[str, Any]], path: Path, sort_cols: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    df = pd.DataFrame(rows)
    if sort_cols and all(c in df.columns for c in sort_cols):
        df = df.sort_values(sort_cols, kind="stable")
    df.to_csv(path, index=False)


def _make_plots(curve_df: pd.DataFrame, manifest: pd.DataFrame, output: Path, n_examples: int) -> None:
    if curve_df.empty or n_examples <= 0:
        return
    import matplotlib.pyplot as plt
    plots_dir = output / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for store in sorted(manifest["store"].unique()):
        m = manifest[manifest["store"] == store]
        if "sample_zero_share" in m.columns:
            m = m.sort_values("sample_zero_share")
        positions = np.unique(np.linspace(0, len(m) - 1, min(n_examples, len(m))).round().astype(int))
        chosen = m.iloc[positions]
        for _, item in chosen.iterrows():
            for origin in sorted(curve_df.loc[curve_df["store"] == store.upper(), "forecast_origin"].unique()):
                item_curves = curve_df[
                    (curve_df["store"] == store.upper())
                    & (curve_df["item_id"] == item["item_id"])
                    & (curve_df["forecast_origin"] == origin)
                ]
                if item_curves.empty:
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
                components = (("occurrence", "Demand occurrence probability"),
                              ("positive_size", "Typical positive-demand size"))
                for ax, (component, label) in zip(axes, components):
                    c = item_curves[item_curves["component"] == component]
                    if c.empty:
                        ax.text(0.5, 0.5, "Component not fitted", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(label)
                        continue
                    x = c["elapsed_day"].to_numpy()
                    y = c["curve_value"].to_numpy()
                    lo = c["lower_95"].to_numpy()
                    hi = c["upper_95"].to_numpy()
                    ax.plot(x, y, label="Fitted curve")
                    ax.fill_between(x, lo, hi, alpha=0.18, label="Pointwise 95% CI")
                    ax.set_title(label)
                    ax.set_xlabel("Elapsed days since first price-live day")
                    ax.legend()
                fig.suptitle(f"{store.upper()} · {item['item_id']} · cutoff d_{origin - 1}")
                fig.tight_layout()
                safe_item = re.sub(r"[^A-Za-z0-9_-]+", "_", str(item["item_id"]))
                fig.savefig(plots_dir / f"{store}_{safe_item}_d{origin}.png", dpi=140)
                plt.close(fig)


def _write_signature_drift(signatures: list[dict[str, Any]], path: Path) -> None:
    """Create paired feature changes between origins for each sampled series."""
    df = pd.DataFrame(signatures)
    if df.empty or "forecast_origin" not in df.columns:
        pd.DataFrame().to_csv(path, index=False)
        return
    id_cols = ["store", "row_index", "item_id"]
    feature_cols = [
        c for c in df.columns
        if c not in set(id_cols + ["forecast_origin", "training_end_day", "n_calendar_training_days"])
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    origins = sorted(df["forecast_origin"].dropna().unique().tolist())
    if len(origins) < 2:
        pd.DataFrame().to_csv(path, index=False)
        return
    left = df[df["forecast_origin"] == origins[0]][id_cols + feature_cols].copy()
    right = df[df["forecast_origin"] == origins[-1]][id_cols + feature_cols].copy()
    merged = left.merge(right, on=id_cols, how="outer", suffixes=(f"_d{origins[0]}", f"_d{origins[-1]}"))
    for col in feature_cols:
        a, b = f"{col}_d{origins[0]}", f"{col}_d{origins[-1]}"
        if a in merged.columns and b in merged.columns:
            merged[f"{col}_change"] = merged[b] - merged[a]
    merged.to_csv(path, index=False)


def run_smoke_test() -> None:
    Gam = load_gam_class()
    rng = np.random.default_rng(123)
    n = 500
    x = np.arange(n, dtype=np.float64)
    X = pd.DataFrame({"elapsed_day": x})
    p = 1.0 / (1.0 + np.exp(-(-0.7 + 0.35 * np.sin(x / 55.0) + 0.0008 * x)))
    occurrence = rng.binomial(1, p)
    occ_stats, occ_curve = _fit_gam_component(Gam, "binomial", x, occurrence, 25, 200, 80)
    size_log = 0.6 + 0.0005 * x + rng.normal(0, 0.18, n)
    size_stats, size_curve = _fit_gam_component(Gam, "gaussian", x, size_log, 15, 200, 80)
    assert len(occ_curve) == len(size_curve) == 80
    assert np.isfinite(occ_stats["edf_time"])
    assert np.isfinite(size_stats["lambda_time"])
    print("mgcv-rust signature smoke test passed: Binomial/Gaussian fits, EDF/lambda, deviance, residuals and pointwise CIs.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stores", nargs="+", default=list(DEFAULT_STORES))
    parser.add_argument("--origins", nargs="+", type=int, default=list(DEFAULT_ORIGINS))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--output", type=Path, default=Path("artifacts/gam_rust_signatures"))
    parser.add_argument("--min-live-days", type=int, default=60)
    parser.add_argument("--min-positive-days", type=int, default=30)
    parser.add_argument("--k-trend", type=int, default=25)
    parser.add_argument("--curve-points", type=int, default=100)
    parser.add_argument("--ci-samples", type=int, default=200)
    parser.add_argument("--plot-examples-per-store", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return
    if args.min_live_days < 30 or args.min_positive_days < 10 or args.k_trend < 5:
        parser.error("Require --min-live-days >= 30, --min-positive-days >= 10 and --k-trend >= 5")
    if len(set(args.origins)) != len(args.origins) or not args.origins:
        parser.error("--origins must be non-empty and unique")

    Gam = load_gam_class()
    stores = [s.lower() for s in args.stores]
    manifest = _load_manifest(args.sample_manifest, stores)
    args.output.mkdir(parents=True, exist_ok=True)
    cache_root = args.output / "series_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    calendar_path = args.processed / "calendar.parquet"
    if not calendar_path.is_file():
        raise FileNotFoundError(calendar_path)
    cal = pl.read_parquet(calendar_path, columns=["d_num", "date"]).sort("d_num")
    dates = pd.to_datetime(cal["date"].to_list())
    weekday_all = np.asarray([d.weekday() for d in dates], dtype=np.int8)
    if max(args.origins) - 1 > len(weekday_all):
        raise ValueError("Calendar does not cover the latest requested origin")

    config = {
        "model_version": MODEL_VERSION,
        "backend": "mgcv-rust 0.23.2; Python/Rust; no R or rpy2",
        "purpose": "descriptive demand signature extraction only; no forecasts and no clustering",
        "stores": [s.upper() for s in stores],
        "series_count": int(len(manifest)),
        "origins": list(args.origins),
        "training_history_end_days": [o - 1 for o in args.origins],
        "sample_manifest": str(args.sample_manifest),
        "reuse_previous_240_series_manifest": True,
        "occurrence_model": "Binomial(logit) GAM on price-live days with elapsed_day smooth",
        "positive_size_model": "Gaussian(identity) GAM on log1p(positive daily sales), elapsed_day smooth",
        "trend_basis": "mgcv-rust default smooth basis; k_default/term_k_mapping set per fit; no unsupported cyclic basis assumed",
        "weekly_seasonality": "empirical weekday profiles and exact-calendar-lag autocorrelations, not a cyclic GAM smooth",
        "occurrence_k": "min(k_trend, max(6, n_price_live_days // 25))",
        "positive_size_k": "min(k_trend, max(5, n_positive_days // 8))",
        "min_live_days_for_occurrence_gam": args.min_live_days,
        "min_positive_days_for_size_gam": args.min_positive_days,
        "aic_note": "aic_conditional_approx is calculated from response likelihood and effective degrees of freedom; compare only within the same component and same observations. Package AIC attribute is separately recorded when available.",
        "deviance_note": "Response-scale binomial deviance or Gaussian SSE; deviance_explained=1-model_deviance/null_deviance. For Gaussian log1p size, this is R2 on transformed response, not adjusted R2.",
        "interval_note": "pointwise coefficient-based 95% confidence intervals from Gam.predict_ci; not predictive intervals",
        "positive_size_curve_note": "expm1 of fitted log1p positive-size curve; geometric/median-like typical magnitude, not arithmetic conditional expectation",
        "no_future_information": True,
    }
    try:
        config["mgcv_rust_version"] = importlib.metadata.version("mgcv-rust")
    except importlib.metadata.PackageNotFoundError:
        config["mgcv_rust_version"] = "unknown"
    cache_parameters = {
        key: config[key] for key in (
            "model_version", "mgcv_rust_version", "origins", "sample_manifest",
            "occurrence_model", "positive_size_model", "occurrence_k",
            "positive_size_k", "min_live_days_for_occurrence_gam",
            "min_positive_days_for_size_gam", "aic_note", "interval_note",
        )
    }
    cache_parameters.update({
        "k_trend": args.k_trend,
        "curve_points": args.curve_points,
        "ci_samples": args.ci_samples,
        "sample_manifest_sha256": hashlib.sha256(args.sample_manifest.read_bytes()).hexdigest(),
    })
    cache_signature = hashlib.sha256(
        json.dumps(cache_parameters, sort_keys=True).encode("utf-8")
    ).hexdigest()
    config["cache_signature"] = cache_signature
    config_path = args.output / "gam_rust_config.json"
    if config_path.exists() and not args.force:
        old = json.loads(config_path.read_text())
        if old.get("model_version") != MODEL_VERSION or old.get("origins") != list(args.origins):
            raise ValueError("Existing output config differs. Use a new --output or --force.")
    config_path.write_text(json.dumps(config, indent=2))
    manifest.to_csv(args.output / "gam_rust_sample_manifest.csv", index=False)

    all_signatures: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    all_curves: list[dict[str, Any]] = []
    all_weekly: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for store in stores:
        subset = args.processed / "subsets" / store
        sales_path, price_path = subset / "sales.npy", subset / "price.npy"
        meta_path = subset / "series.parquet"
        for path in (sales_path, price_path, meta_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        sales = np.load(sales_path, mmap_mode="r")
        prices = np.load(price_path, mmap_mode="r")
        meta = pl.read_parquet(meta_path)
        if sales.shape != prices.shape or sales.shape[0] != meta.height:
            raise ValueError(f"Sales/price/metadata shape mismatch for {store.upper()}")
        store_manifest = manifest[manifest["store"] == store]
        if store_manifest.empty:
            continue
        meta_dict = meta.to_dict(as_series=False)
        meta_item_col = "item_id" if "item_id" in meta_dict else ("id" if "id" in meta_dict else None)

        print(f"\n{store.upper()}: {len(store_manifest)} series")
        for origin in args.origins:
            train_end = origin - 1
            if train_end > sales.shape[1]:
                raise ValueError(f"Origin d_{origin} is outside the sales arrays for {store.upper()}")
            print(f"  Signature cutoff d_{train_end} (origin d_{origin})")
            for row in store_manifest.itertuples(index=False):
                row_index = int(row.row_index)
                item_id = str(row.item_id)
                if not 0 <= row_index < sales.shape[0]:
                    raise IndexError(f"Manifest row {row_index} outside {store.upper()} array")
                if meta_item_col:
                    actual_item = str(meta_dict[meta_item_col][row_index])
                    if actual_item not in {item_id, str(getattr(row, "id", item_id))}:
                        raise ValueError(f"Manifest mismatch at {store.upper()} row {row_index}: {item_id} vs {actual_item}")
                cache_file = cache_root / store / f"origin_d{origin:04d}" / f"row_{row_index:05d}.json"
                cached = None
                if cache_file.is_file() and not args.force:
                    cached = json.loads(cache_file.read_text())
                    if (cached.get("_model_version") != MODEL_VERSION
                            or cached.get("_cache_signature") != cache_signature):
                        cached = None
                if cached is None:
                    try:
                        result = _fit_one_series_origin(
                            Gam, store, row_index, item_id, origin,
                            np.asarray(sales[row_index, :train_end]),
                            np.asarray(prices[row_index, :train_end]),
                            weekday_all[:train_end], args.min_live_days,
                            args.min_positive_days, args.k_trend,
                            args.ci_samples, args.curve_points,
                        )
                        cached = {"_model_version": MODEL_VERSION,
                                  "_cache_signature": cache_signature, **result}
                        _atomic_json(cache_file, cached)
                    except Exception as exc:
                        cached = {
                            "_model_version": MODEL_VERSION,
                            "_cache_signature": cache_signature,
                            "signature": {"store": store.upper(), "row_index": row_index,
                                          "item_id": item_id, "forecast_origin": origin,
                                          "fatal_error": str(exc)},
                            "diagnostics": [{"store": store.upper(), "row_index": row_index,
                                             "item_id": item_id, "forecast_origin": origin,
                                             "fit_status": "series_processing_exception",
                                             "error_type": type(exc).__name__, "error": str(exc)}],
                            "curves": [], "weekly": [],
                        }
                        _atomic_json(cache_file, cached)
                        print(f"    WARNING {item_id}: {type(exc).__name__}: {exc}")
                all_signatures.append(cached.get("signature", {}))
                all_diagnostics.extend(cached.get("diagnostics", []))
                all_curves.extend(cached.get("curves", []))
                all_weekly.extend(cached.get("weekly", []))
                for d in cached.get("diagnostics", []):
                    if d.get("fit_status") in {"fit_exception", "series_processing_exception"}:
                        failures.append(d)

            # Persist interim outputs after every store-origin block for safe recovery.
            _write_csv(all_signatures, args.output / "gam_rust_demand_signatures.csv", ["store", "forecast_origin", "row_index"])
            _write_csv(all_diagnostics, args.output / "gam_rust_component_diagnostics.csv", ["store", "forecast_origin", "row_index", "component"])
            _write_csv(all_curves, args.output / "gam_rust_smooth_curves.csv", ["store", "forecast_origin", "row_index", "component", "elapsed_day"])
            _write_csv(all_weekly, args.output / "gam_rust_weekly_profiles.csv", ["store", "forecast_origin", "row_index", "weekday"])
            _write_csv(failures, args.output / "gam_rust_fit_failures.csv", ["store", "forecast_origin", "row_index", "component"])
            _write_signature_drift(all_signatures, args.output / "gam_rust_signature_drift.csv")

    print("\nCompleted signature extraction.")
    print(f"Signatures: {args.output / 'gam_rust_demand_signatures.csv'}")
    print(f"Diagnostics: {args.output / 'gam_rust_component_diagnostics.csv'}")
    print(f"Curves: {args.output / 'gam_rust_smooth_curves.csv'}")
    curve_df = pd.DataFrame(all_curves)
    _make_plots(curve_df, manifest, args.output, args.plot_examples_per_store)
    print(f"Plots: {args.output / 'plots'}")


if __name__ == "__main__":
    main()
