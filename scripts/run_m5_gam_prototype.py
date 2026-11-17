#!/usr/bin/env python3
"""M5 two-part GAM demand-signature prototype using pymgcv / R mgcv.

Prototype design
----------------
* 240 bottom-level item-store series: 80 each from CA_1, WI_3 and TX_1.
* A deterministic, within-store 4x4 empirical-quantile sample over zero-share
  and mean positive demand. Quantile bins are for sample coverage only; they
  are not demand labels and are never passed to the GAMs.
* Two non-overlapping 28-day 2014 validation windows by default:
  d_1342..d_1369 and d_1387..d_1414. The 2015 windows remain untouched.
* Two-part model per series and origin:
    1) Binomial GAM for P(sales > 0).
    2) Gamma(log-link) GAM for E[sales | sales > 0].
  Expected demand is the product of the two predictions.
* Each model has a cubic-regression spline for time (k=25 for occurrence;
  adaptively smaller for sparse positive-size samples) and a cyclic cubic
  spline for weekday (k=7; boundaries at -0.5 and 6.5).
* Constant-occurrence series are handled analytically; sparse positive-size
  histories use a transparent historical-positive-mean fallback.
* Saves model diagnostics, pointwise forecast-mean intervals, partial-effect
  curves with approximate Bayesian confidence bands, residual calendar-lag
  correlations, and comparisons with cached M5 forecasts where available.

Run from the project root in a Pixi environment containing pymgcv, rpy2,
R, mgcv, pandas, polars, numpy and matplotlib.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

HORIZON = 28
DEFAULT_STORES = ("ca_1", "wi_3", "tx_1")
DEFAULT_ORIGINS = (1342, 1387)
DEFAULT_SAMPLE_PER_STORE = 80
SAMPLE_VERSION = "m5_gam_sample_v1"
MODEL_VERSION = "two_part_gam_pymgcv_v2"
WEEKDAY_KNOTS = np.asarray([-0.5, 6.5], dtype=np.float64)


def load_pymgcv():
    """Import GAM classes lazily so --help and data-only checks still work."""
    try:
        from pymgcv.basis_functions import CubicSpline
        from pymgcv.families import Binomial, Gamma
        from pymgcv.gam import GAM
        from pymgcv.terms import S
        import rpy2.robjects as ro
    except Exception as exc:  # pragma: no cover - depends on local Pixi/R setup
        raise RuntimeError(
            "Could not import pymgcv and its R/mgcv runtime. Add r-base, r-mgcv, "
            "rpy2 and pymgcv to a dedicated Pixi environment, then run with "
            "`pixi run -e gam ...`. Original error: " + repr(exc)
        ) from exc
    return GAM, S, CubicSpline, Binomial, Gamma, ro


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        return float(arr[0]) if arr.size else default
    except Exception:
        return default


def _array_from_fit_se(result: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract fit and SE from pymgcv FitAndSE or tolerate tuple-like output."""
    if hasattr(result, "fit") and hasattr(result, "se"):
        fit, se = result.fit, result.se
    elif isinstance(result, (tuple, list)) and len(result) >= 2:
        fit, se = result[0], result[1]
    else:
        fit = result
        se = np.full(np.asarray(fit).shape, np.nan, dtype=np.float64)
    return (
        np.asarray(fit, dtype=np.float64).reshape(-1),
        np.asarray(se, dtype=np.float64).reshape(-1),
    )


def _quantile_bins(values: np.ndarray, q: int = 4) -> np.ndarray:
    """Stable equal-count empirical bins; used only to spread the pilot sample."""
    values = np.asarray(values, dtype=np.float64)
    # Stable ranks handle ties without inventing value cut-offs.
    ranks = pd.Series(values).rank(method="first").to_numpy(dtype=np.float64)
    bins = np.floor((ranks - 1.0) / max(len(values), 1) * q).astype(np.int16)
    return np.clip(bins, 0, q - 1)


def select_store_sample(
    sales: np.ndarray,
    prices: np.ndarray,
    metadata: pl.DataFrame,
    train_days: int,
    sample_n: int,
    seed: int,
    store: str,
) -> pd.DataFrame:
    """Select a within-store sample spanning empirical sparsity/size quantiles."""
    if sample_n <= 0 or sample_n > sales.shape[0]:
        raise ValueError(f"Invalid sample size {sample_n} for {sales.shape[0]} series")
    y = np.asarray(sales[:, :train_days], dtype=np.float64)
    price = np.asarray(prices[:, :train_days], dtype=np.float32)
    live = np.isfinite(price)
    pos = live & (y > 0)
    n_live = live.sum(axis=1)
    n_positive = pos.sum(axis=1)
    zero_share = np.divide(
        (live & (y == 0)).sum(axis=1), n_live,
        out=np.ones(sales.shape[0], dtype=np.float64), where=n_live > 0,
    )
    positive_mean = np.divide(
        np.where(pos, y, 0.0).sum(axis=1), n_positive,
        out=np.zeros(sales.shape[0], dtype=np.float64), where=n_positive > 0,
    )
    z_bin = _quantile_bins(zero_share, 4)
    size_bin = _quantile_bins(np.log1p(positive_mean), 4)

    rng = np.random.default_rng(seed + sum(ord(c) for c in store.upper()))
    base_quota = sample_n // 16
    remainder = sample_n - base_quota * 16
    selected: list[int] = []
    selected_set: set[int] = set()
    cells = [(z, s) for z in range(4) for s in range(4)]
    # Rotate the extra quota so the sample remains deterministic and balanced.
    extra_cells = set(rng.choice(len(cells), size=remainder, replace=False).tolist()) if remainder else set()
    for cell_idx, (zb, sb) in enumerate(cells):
        quota = base_quota + int(cell_idx in extra_cells)
        candidates = np.flatnonzero((z_bin == zb) & (size_bin == sb))
        take = min(quota, len(candidates))
        if take:
            chosen = rng.choice(candidates, size=take, replace=False).astype(int).tolist()
            selected.extend(chosen)
            selected_set.update(chosen)
    if len(selected) < sample_n:
        remaining = np.asarray([i for i in range(sales.shape[0]) if i not in selected_set], dtype=int)
        fill = rng.choice(remaining, size=sample_n - len(selected), replace=False)
        selected.extend(fill.astype(int).tolist())

    selected = sorted(selected[:sample_n])
    meta = metadata.to_dict(as_series=False)
    rows: list[dict[str, Any]] = []
    for i in selected:
        row = {k: v[i] for k, v in meta.items()}
        row.update({
            "store": store.upper(),
            "row_index": int(i),
            "sample_origin": int(train_days + 1),
            "sample_zero_share": float(zero_share[i]),
            "sample_demand_frequency": float(n_positive[i] / n_live[i]) if n_live[i] else 0.0,
            "sample_mean_positive_demand": float(positive_mean[i]),
            "sample_live_days": int(n_live[i]),
            "sample_positive_days": int(n_positive[i]),
            "zero_share_quantile_bin": int(z_bin[i]),
            "positive_size_quantile_bin": int(size_bin[i]),
            "sample_method": "4x4 empirical quantile coverage; not demand classes",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _pearson_corr_by_calendar_lag(residuals: np.ndarray, day_indices: np.ndarray, lag: int) -> float:
    """Correlation of residual pairs exactly lag calendar days apart."""
    residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
    day_indices = np.asarray(day_indices, dtype=np.int64).reshape(-1)
    if len(residuals) != len(day_indices):
        return float("nan")
    finite = np.isfinite(residuals)
    residuals = residuals[finite]
    day_indices = day_indices[finite]
    if len(residuals) < 3:
        return float("nan")
    lookup = {int(day): i for i, day in enumerate(day_indices)}
    left: list[float] = []
    right: list[float] = []
    for i, day in enumerate(day_indices):
        j = lookup.get(int(day + lag))
        if j is not None:
            left.append(float(residuals[i]))
            right.append(float(residuals[j]))
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _extract_model_stats(model: Any, ro: Any, day_indices: np.ndarray) -> dict[str, Any]:
    """Extract mgcv summary, AIC, EDF, smoothing parameters and residual ACF."""
    r_model = model.fit_state.rgam
    r_summary = ro.r["summary"](r_model)
    dev_expl = _safe_float(r_summary.rx2("dev.expl"))
    deviance = _safe_float(ro.r["deviance"](r_model))
    aic = _safe_float(ro.r["AIC"](r_model))

    s_table = r_summary.rx2("s.table")
    s_arr = np.asarray(s_table, dtype=np.float64)
    if s_arr.ndim == 1:
        s_arr = s_arr.reshape(1, -1)
    edf_values = s_arr[:, 0].tolist() if s_arr.size else []
    try:
        term_names = [str(x) for x in list(ro.r["rownames"](s_table))]
    except Exception:
        term_names = []
    if len(term_names) != len(edf_values):
        term_names = [f"smooth_{i}" for i in range(len(edf_values))]

    try:
        lambdas = np.asarray(r_model.rx2("sp"), dtype=np.float64).reshape(-1).tolist()
    except Exception:
        lambdas = []
    try:
        residuals = np.asarray(ro.r["residuals"](r_model, type="pearson"), dtype=np.float64).reshape(-1)
    except Exception:
        residuals = np.full(len(day_indices), np.nan, dtype=np.float64)

    try:
        converged = bool(np.asarray(r_model.rx2("converged")).reshape(-1)[0])
    except Exception:
        converged = None

    result = {
        "fit_converged": converged,
        "deviance_explained": dev_expl,
        "deviance": deviance,
        "aic": aic,
        "edf_terms_json": json.dumps(dict(zip(term_names, edf_values))),
        "lambda_terms_json": json.dumps(dict(zip(term_names, lambdas))),
        "n_residuals": int(len(residuals)),
        "pearson_residual_sd": float(np.nanstd(residuals, ddof=1)) if len(residuals) > 1 else float("nan"),
        "pearson_residual_mean": float(np.nanmean(residuals)) if len(residuals) else float("nan"),
        "resid_corr_calendar_lag1": _pearson_corr_by_calendar_lag(residuals, day_indices, 1),
        "resid_corr_calendar_lag7": _pearson_corr_by_calendar_lag(residuals, day_indices, 7),
        "resid_corr_calendar_lag28": _pearson_corr_by_calendar_lag(residuals, day_indices, 28),
        "edf_time": float(edf_values[0]) if len(edf_values) >= 1 else float("nan"),
        "edf_weekday": float(edf_values[1]) if len(edf_values) >= 2 else float("nan"),
        "lambda_time": float(lambdas[0]) if len(lambdas) >= 1 else float("nan"),
        "lambda_weekday": float(lambdas[1]) if len(lambdas) >= 2 else float("nan"),
    }
    return result


def _get_prediction_fit_se(model: Any, frame: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    result = model.predict(frame, compute_se=True, type="response")[target]
    fit, se = _array_from_fit_se(result)
    if len(fit) != len(frame):
        raise ValueError(f"Unexpected prediction length for {target}: {len(fit)} != {len(frame)}")
    return fit, se


def _extract_partial_curve(
    model: Any,
    target: str,
    term_index: int,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    result = model.partial_effect(term_index, target=target, data=frame, compute_se=True)
    return _array_from_fit_se(result)


def _fit_component(
    train_frame: pd.DataFrame,
    target: str,
    family: Any,
    k_time: int,
    GAM: Any,
    S: Any,
    CubicSpline: Any,
) -> Any:
    terms = S("t", k=k_time, bs=CubicSpline()) + S(
        "dow", k=7, bs=CubicSpline(cyclic=True)
    )
    model = GAM({target: terms}, family=family)
    # Explicit half-day boundaries make 6 -> 0 a one-day transition on a 0..6 scale.
    model.fit(train_frame, method="REML", knots={"dow": WEEKDAY_KNOTS})
    return model


def _smooth_curve_rows(
    model: Any,
    target: str,
    store: str,
    item_id: str,
    origin: int,
    component: str,
    t_min: float,
    t_max: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    t_grid = np.linspace(t_min, t_max, 100, dtype=np.float64)
    trend_frame = pd.DataFrame({"t": t_grid, "dow": np.zeros_like(t_grid)})
    trend_fit, trend_se = _extract_partial_curve(model, target, 0, trend_frame)
    for x, fit, se in zip(t_grid, trend_fit, trend_se):
        rows.append({
            "store": store, "item_id": item_id, "forecast_start_day": origin,
            "component": component, "term": "time_trend", "x_value": float(x),
            "effect_link": float(fit), "se_link": float(se),
            "lower_95_link": float(fit - 1.96 * se), "upper_95_link": float(fit + 1.96 * se),
            "interval_type": "approximate Bayesian pointwise interval for partial smooth",
        })

    dow_grid = np.arange(7, dtype=np.float64)
    weekday_frame = pd.DataFrame({"t": np.full(7, np.median([t_min, t_max])), "dow": dow_grid})
    dow_fit, dow_se = _extract_partial_curve(model, target, 1, weekday_frame)
    for x, fit, se in zip(dow_grid, dow_fit, dow_se):
        rows.append({
            "store": store, "item_id": item_id, "forecast_start_day": origin,
            "component": component, "term": "cyclic_weekday", "x_value": float(x),
            "effect_link": float(fit), "se_link": float(se),
            "lower_95_link": float(fit - 1.96 * se), "upper_95_link": float(fit + 1.96 * se),
            "interval_type": "approximate Bayesian pointwise interval for partial smooth",
        })
    return rows


def _empty_component_stats() -> dict[str, Any]:
    return {
        "fit_converged": None,
        "deviance_explained": float("nan"), "deviance": float("nan"), "aic": float("nan"),
        "edf_terms_json": "{}", "lambda_terms_json": "{}", "n_residuals": 0,
        "pearson_residual_sd": float("nan"), "pearson_residual_mean": float("nan"),
        "resid_corr_calendar_lag1": float("nan"), "resid_corr_calendar_lag7": float("nan"),
        "resid_corr_calendar_lag28": float("nan"), "edf_time": float("nan"),
        "edf_weekday": float("nan"), "lambda_time": float("nan"),
        "lambda_weekday": float("nan"),
    }


def _fallback_rates(y_live: np.ndarray) -> tuple[float, float]:
    occurrence = float(np.mean(y_live > 0)) if len(y_live) else 0.0
    positives = y_live[y_live > 0]
    magnitude = float(np.mean(positives)) if len(positives) else 0.0
    return occurrence, magnitude


def fit_one_series_origin(
    *,
    store: str,
    row_index: int,
    item_id: str,
    train_sales: np.ndarray,
    train_price: np.ndarray,
    train_weekday: np.ndarray,
    future_weekday: np.ndarray,
    actual: np.ndarray,
    origin: int,
    min_positive_days: int,
    GAM: Any,
    S: Any,
    CubicSpline: Any,
    Binomial: Any,
    Gamma: Any,
    ro: Any,
    save_curves: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit occurrence and positive-size GAMs for one item-store/origin."""
    live_mask = np.isfinite(train_price)
    day_indices = np.flatnonzero(live_mask).astype(np.int64)
    y_live = np.asarray(train_sales[day_indices], dtype=np.float64)
    if len(day_indices) < 60:
        raise ValueError(f"Only {len(day_indices)} price-live training days")
    if (y_live < 0).any() or not np.isfinite(y_live).all():
        raise ValueError("Sales contain negative or non-finite values on live days")

    first_live = int(day_indices[0])
    t = day_indices.astype(np.float64) - first_live
    dow = np.asarray(train_weekday[day_indices], dtype=np.float64)
    occurrence = (y_live > 0).astype(np.int8)
    positive_mask = occurrence == 1
    n_positive = int(positive_mask.sum())
    occ_frame = pd.DataFrame({"t": t, "dow": dow, "occurrence": occurrence})

    # Occurrence component: all price-live days, including zeros. Constant
    # histories are handled analytically instead of being recorded as fit failures.
    occ_model = None
    occ_stats: dict[str, Any]
    occ_status = "fitted"
    occ_constant_rate = float(np.mean(occurrence))
    occ_constant_se = math.sqrt(
        occ_constant_rate * (1.0 - occ_constant_rate) / max(len(occurrence), 1)
    )
    occ_fit_seconds = 0.0
    if np.unique(occurrence).size < 2:
        occ_status = "constant_occurrence_rate_no_smooth"
        occ_stats = _empty_component_stats()
    else:
        started = time.perf_counter()
        occ_model = _fit_component(
            occ_frame, "occurrence", Binomial(link="logit"), 25, GAM, S, CubicSpline
        )
        occ_fit_seconds = time.perf_counter() - started
        occ_stats = _extract_model_stats(occ_model, ro, day_indices)

    # Positive magnitude component. Use an adaptive time basis for sparse positive
    # samples; this is a model-estimation safeguard, not a demand-pattern rule.
    size_status = "fitted"
    size_fit_seconds = 0.0
    size_model = None
    size_stats: dict[str, Any]
    if n_positive < min_positive_days or np.unique(y_live[positive_mask]).size < 2:
        size_status = "fallback_historical_positive_mean"
        _, fallback_size = _fallback_rates(y_live)
        size_stats = _empty_component_stats()
        size_k_time = 0
    else:
        size_k_time = int(min(25, max(5, n_positive // 8)))
        size_frame = pd.DataFrame({
            "t": t[positive_mask],
            "dow": dow[positive_mask],
            "positive_size": y_live[positive_mask],
        })
        started = time.perf_counter()
        try:
            size_model = _fit_component(
                size_frame, "positive_size", Gamma(link="log"), size_k_time, GAM, S, CubicSpline
            )
            size_fit_seconds = time.perf_counter() - started
            size_stats = _extract_model_stats(size_model, ro, day_indices[positive_mask])
        except Exception as exc:
            size_status = "fallback_after_fit_error: " + type(exc).__name__
            _, fallback_size = _fallback_rates(y_live)
            size_model = None
            size_fit_seconds = time.perf_counter() - started
            size_stats = _empty_component_stats()

    future_day_indices = np.arange(origin - 1, origin - 1 + HORIZON, dtype=np.int64)
    future_t = future_day_indices.astype(np.float64) - first_live
    future_frame = pd.DataFrame({"t": future_t, "dow": np.asarray(future_weekday, dtype=np.float64)})
    if occ_model is not None:
        p_fit, p_se = _get_prediction_fit_se(occ_model, future_frame, "occurrence")
        p_fit = np.clip(p_fit, 0.0, 1.0)
    else:
        p_fit = np.full(HORIZON, occ_constant_rate, dtype=np.float64)
        p_se = np.full(HORIZON, occ_constant_se, dtype=np.float64)
    if size_model is not None:
        m_fit, m_se = _get_prediction_fit_se(size_model, future_frame, "positive_size")
        m_fit = np.maximum(m_fit, 0.0)
    else:
        _, fallback_size = _fallback_rates(y_live)
        m_fit = np.full(HORIZON, fallback_size, dtype=np.float64)
        m_se = np.full(HORIZON, np.nan, dtype=np.float64)
    expected = np.maximum(p_fit * m_fit, 0.0)
    expected_se = np.sqrt(np.maximum((m_fit * p_se) ** 2 + (p_fit * m_se) ** 2, 0.0))
    expected_lower = np.maximum(expected - 1.96 * expected_se, 0.0)
    expected_upper = np.maximum(expected + 1.96 * expected_se, 0.0)

    forecast_arrays = {
        "occurrence_probability": p_fit.astype(np.float32),
        "occurrence_probability_se": p_se.astype(np.float32),
        "positive_size_mean": m_fit.astype(np.float32),
        "positive_size_mean_se": m_se.astype(np.float32),
        "expected_mean": expected.astype(np.float32),
        "expected_mean_se": expected_se.astype(np.float32),
        "expected_mean_lower95": expected_lower.astype(np.float32),
        "expected_mean_upper95": expected_upper.astype(np.float32),
        "actual": np.asarray(actual, dtype=np.float32),
    }

    diagnostics = {
        "store": store.upper(), "row_index": int(row_index), "item_id": item_id,
        "forecast_start_day": int(origin), "forecast_end_day": int(origin + HORIZON - 1),
        "n_live_training_days": int(len(day_indices)), "n_positive_training_days": n_positive,
        "training_zero_share": float(np.mean(occurrence == 0)),
        "training_positive_mean": float(np.mean(y_live[positive_mask])) if n_positive else 0.0,
        "occurrence_model_status": occ_status, "size_model_status": size_status,
        "occurrence_family": "Binomial(logit)", "positive_size_family": "Gamma(log)",
        "positive_size_interpretation": "conditional positive-demand mean; continuous/quasi-likelihood approximation for integer sales",
        "occurrence_fit_seconds": float(occ_fit_seconds), "size_fit_seconds": float(size_fit_seconds),
        "occurrence_k_time": 25, "occurrence_k_weekday": 7,
        "size_k_time": int(size_k_time), "size_k_weekday": 7 if size_model is not None else 0,
        **{f"occurrence_{k}": v for k, v in occ_stats.items()},
        **{f"size_{k}": v for k, v in size_stats.items()},
        "forecast_expected_units_28d": float(expected.sum()),
        "actual_units_28d": float(np.asarray(actual).sum()),
        "forecast_to_actual_pct": (
            100.0 * float(expected.sum()) / float(np.asarray(actual).sum())
            if float(np.asarray(actual).sum()) > 0 else float("nan")
        ),
    }

    curve_rows: list[dict[str, Any]] = []
    if save_curves:
        if occ_model is not None:
            curve_rows.extend(_smooth_curve_rows(
                occ_model, "occurrence", store.upper(), item_id, origin, "occurrence",
                float(np.min(t)), float(np.max(t)),
            ))
        if size_model is not None:
            curve_rows.extend(_smooth_curve_rows(
                size_model, "positive_size", store.upper(), item_id, origin, "positive_size",
                float(np.min(t[positive_mask])), float(np.max(t[positive_mask])),
            ))

    fit_rows = []
    for component, stats in (("occurrence", occ_stats), ("positive_size", size_stats)):
        fit_rows.append({
            "store": store.upper(), "row_index": int(row_index), "item_id": item_id,
            "forecast_start_day": int(origin), "component": component,
            "model_status": occ_status if component == "occurrence" else ("fitted" if size_model is not None else size_status),
            "k_time": 25 if component == "occurrence" else int(size_k_time),
            "k_weekday": 7 if component == "occurrence" or size_model is not None else 0,
            **stats,
        })
    return diagnostics, forecast_arrays, fit_rows, curve_rows


def _metrics(actual: np.ndarray, forecast: np.ndarray, scales: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    err = np.asarray(forecast, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    denominator = float(np.abs(actual).sum())
    abs_err = np.abs(err)
    per_series_mse = np.mean(err**2, axis=1)
    rmsse = np.full(actual.shape[0], np.nan, dtype=np.float64)
    valid_scale = np.isfinite(scales) & (scales > 0)
    rmsse[valid_scale] = np.sqrt(per_series_mse[valid_scale] / scales[valid_scale])
    valid_w = np.isfinite(rmsse) & np.isfinite(weights) & (weights > 0)
    wrmsse = float(np.sum(rmsse[valid_w] * weights[valid_w]) / np.sum(weights[valid_w])) if valid_w.any() else float("nan")
    actual_total = float(actual.sum())
    forecast_total = float(np.asarray(forecast, dtype=np.float64).sum())
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "wape": float(abs_err.sum() / denominator) if denominator > 0 else float("nan"),
        "bias_per_observation": float(err.mean()),
        "aggregate_error_units": forecast_total - actual_total,
        "aggregate_error_pct_actual": 100.0 * (forecast_total - actual_total) / actual_total if actual_total > 0 else float("nan"),
        "aggregate_forecast_to_actual_pct": 100.0 * forecast_total / actual_total if actual_total > 0 else float("nan"),
        "sample_revenue_weighted_rmsse_style": wrmsse,
        "sample_revenue_weight_coverage": float(np.sum(weights)),
        "n_series_with_valid_rmsse": int(valid_w.sum()),
    }


def _series_scales(train: np.ndarray) -> np.ndarray:
    scales = np.full(train.shape[0], np.nan, dtype=np.float64)
    for i in range(train.shape[0]):
        y = np.asarray(train[i], dtype=np.float64)
        positive = np.flatnonzero(y > 0)
        start = int(positive[0]) if positive.size else 0
        diff = np.diff(y[start:])
        if diff.size:
            value = float(np.mean(diff**2))
            if value > 0 and np.isfinite(value):
                scales[i] = value
    return scales


def _load_reference_models(
    store: str,
    origin: int,
    end_day: int,
    sample_rows: np.ndarray,
    actual_full: np.ndarray,
    reference_root: Path,
    phase_a_root: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load T0/seasonal-naive and Phase A forecasts for matched sample rows."""
    forecasts: dict[str, np.ndarray] = {}
    reference_path = reference_root / store / f"origin_d{origin:04d}_d{end_day:04d}" / "forecasts_and_actuals.npz"
    revenue_weights = np.full(actual_full.shape[0], np.nan, dtype=np.float64)
    if reference_path.is_file():
        with np.load(reference_path, allow_pickle=False) as f:
            saved_actual = np.asarray(f["actual"])
            if not np.array_equal(saved_actual, actual_full):
                raise ValueError(f"Reference actuals mismatch: {reference_path}")
            levels = np.asarray(f["quantile_levels"], dtype=np.float64)
            qi = int(np.argmin(np.abs(levels - 0.5)))
            forecasts["Timestral_T0"] = np.asarray(f["timestral_quantiles"], dtype=np.float32)[:, :, qi][sample_rows]
            forecasts["SeasonalNaive7_point"] = np.asarray(f["seasonal_naive_point"], dtype=np.float32)[sample_rows]
            revenue_weights = np.asarray(f["revenue_weights"], dtype=np.float64)
    else:
        warnings.warn(f"Reference model cache missing: {reference_path}; Timestral/naive comparison omitted")

    phase_a_path = phase_a_root / store / f"origin_d{origin:04d}_d{end_day:04d}" / "intermittent_forecasts.npz"
    if phase_a_path.is_file():
        with np.load(phase_a_path, allow_pickle=False) as f:
            if not np.array_equal(np.asarray(f["actual"]), actual_full):
                raise ValueError(f"Phase A actuals mismatch: {phase_a_path}")
            names = [str(x) for x in np.asarray(f["model_names"]).tolist()]
            for name in names:
                if name in {"TSB_ad0.10_ap0.10", "IMAPA"} and name in f.files:
                    forecasts[name] = np.asarray(f[name], dtype=np.float32)[sample_rows]
    else:
        warnings.warn(f"Phase A cache missing: {phase_a_path}; TSB/IMAPA comparison omitted")

    return forecasts, revenue_weights


def _write_frame(rows: list[dict[str, Any]], path: Path, sort_by: list[str] | None = None) -> None:
    if not rows:
        pl.DataFrame().write_csv(path)
        return
    frame = pl.DataFrame(rows, infer_schema_length=None)
    if sort_by and all(c in frame.columns for c in sort_by):
        frame = frame.sort(sort_by)
    frame.write_csv(path)


def run_smoke_test() -> None:
    GAM, S, CubicSpline, Binomial, Gamma, ro = load_pymgcv()
    rng = np.random.default_rng(7)
    n = 420
    t = np.arange(n, dtype=np.float64)
    dow = t.astype(int) % 7
    p = 1 / (1 + np.exp(-(-0.8 + 0.45 * np.sin(2 * np.pi * t / 7) + 0.0005 * t)))
    occ = rng.binomial(1, p)
    size_mean = np.exp(1.2 + 0.2 * np.cos(2 * np.pi * t / 7) + 0.0002 * t)
    y = np.where(occ > 0, rng.gamma(shape=2.5, scale=size_mean / 2.5), 0.0)
    frame = pd.DataFrame({"t": t, "dow": dow.astype(float), "occurrence": occ})
    occ_model = _fit_component(frame, "occurrence", Binomial(link="logit"), 12, GAM, S, CubicSpline)
    future = pd.DataFrame({"t": np.arange(n, n + HORIZON), "dow": np.arange(n, n + HORIZON) % 7})
    p_fit, p_se = _get_prediction_fit_se(occ_model, future, "occurrence")
    assert p_fit.shape == (HORIZON,) and np.isfinite(p_fit).all()
    assert np.isfinite(p_se).all()
    curve_rows = _smooth_curve_rows(occ_model, "occurrence", "SMOKE", "SYNTHETIC", n, "occurrence", 0, n - 1)
    assert len(curve_rows) == 107
    pos_mask = y > 0
    pos_frame = pd.DataFrame({"t": t[pos_mask], "dow": dow[pos_mask].astype(float), "positive_size": y[pos_mask]})
    size_model = _fit_component(pos_frame, "positive_size", Gamma(link="log"), 12, GAM, S, CubicSpline)
    m_fit, m_se = _get_prediction_fit_se(size_model, future, "positive_size")
    assert m_fit.shape == (HORIZON,) and np.isfinite(m_fit).all()
    stats = _extract_model_stats(occ_model, ro, np.arange(n))
    assert np.isfinite(stats["aic"]) and np.isfinite(stats["deviance_explained"])
    print("pymgcv GAM smoke test passed: cyclic smooth, predictions, CI, AIC, EDF and deviance explained.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stores", nargs="+", default=list(DEFAULT_STORES))
    parser.add_argument("--origins", nargs="+", type=int, default=list(DEFAULT_ORIGINS))
    parser.add_argument("--sample-per-store", type=int, default=DEFAULT_SAMPLE_PER_STORE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--reference", type=Path, default=Path("artifacts/independent_seasons"))
    parser.add_argument("--phase-a-reference", type=Path, default=Path("artifacts/phase_a_intermittent"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gam_prototype"))
    parser.add_argument("--min-positive-days", type=int, default=30,
                        help="Minimum positive training days for the conditional Gamma GAM; otherwise use historical positive-mean fallback")
    parser.add_argument("--plot-examples-per-store", type=int, default=2)
    parser.add_argument("--k-check-series-per-store", type=int, default=2,
                        help="Run pymgcv check_k on this many sampled series per store and origin; 0 disables")
    parser.add_argument("--force", action="store_true", help="Recompute all per-series GAM caches")
    parser.add_argument("--smoke-test", action="store_true", help="Fit synthetic two-part cyclic GAMs and check outputs")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return
    if args.sample_per_store < 1:
        parser.error("--sample-per-store must be positive")
    if len(set(args.origins)) != len(args.origins) or not args.origins:
        parser.error("--origins must be non-empty and unique")
    if args.min_positive_days < 10:
        parser.error("--min-positive-days must be at least 10")

    GAM, S, CubicSpline, Binomial, Gamma, ro = load_pymgcv()
    args.output.mkdir(parents=True, exist_ok=True)
    cache_root = args.output / "series_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    calendar_path = args.processed / "calendar.parquet"
    if not calendar_path.is_file():
        raise FileNotFoundError(calendar_path)
    calendar = pl.read_parquet(calendar_path, columns=["d_num", "date"]).sort("d_num")
    weekday_all = np.asarray([d.weekday() for d in calendar["date"].to_list()], dtype=np.int8)
    if len(weekday_all) < max(args.origins) - 1 + HORIZON:
        raise ValueError("Calendar does not cover all requested forecast windows")

    # The sample is selected once using information available at the earliest origin.
    sample_origin = min(args.origins)
    sample_train_days = sample_origin - 1
    all_manifest: list[pd.DataFrame] = []
    store_data: dict[str, dict[str, Any]] = {}
    for store in args.stores:
        store = store.lower()
        sub = args.processed / "subsets" / store
        for filename in ("sales.npy", "price.npy", "series.parquet"):
            if not (sub / filename).is_file():
                raise FileNotFoundError(sub / filename)
        sales = np.load(sub / "sales.npy", mmap_mode="r")
        prices = np.load(sub / "price.npy", mmap_mode="r")
        metadata = pl.read_parquet(sub / "series.parquet")
        if sales.shape != prices.shape or sales.shape[0] != metadata.height:
            raise ValueError(f"Array/metadata shape mismatch for {store.upper()}")
        manifest = select_store_sample(
            sales, prices, metadata, sample_train_days, args.sample_per_store, args.seed, store
        )
        all_manifest.append(manifest)
        store_data[store] = {"sales": sales, "prices": prices, "metadata": metadata, "manifest": manifest}
        print(f"{store.upper()}: selected {len(manifest)} series from {sales.shape[0]:,} using history through d_{sample_train_days}")

    manifest_all = pd.concat(all_manifest, ignore_index=True)
    sample_manifest_path = args.output / "gam_sample_manifest.csv"
    if sample_manifest_path.exists() and not args.force:
        existing = pd.read_csv(sample_manifest_path)
        cols = ["store", "row_index", "item_id"]
        if existing[cols].astype(str).values.tolist() != manifest_all[cols].astype(str).values.tolist():
            raise ValueError("Existing sample manifest differs from newly generated sample. Use --force or a new --output.")
    manifest_all.to_csv(sample_manifest_path, index=False)
    (args.output / "gam_config.json").write_text(json.dumps({
        "model_version": MODEL_VERSION,
        "sample_version": SAMPLE_VERSION,
        "stores": [s.upper() for s in args.stores],
        "sample_per_store": args.sample_per_store,
        "total_series": int(len(manifest_all)),
        "seed": args.seed,
        "sample_selection_origin": sample_origin,
        "sample_selection_history_end_day": sample_train_days,
        "sample_selection_method": "Within-store 4x4 empirical quantile grid over live-day zero share and log1p mean positive demand; no demand labels created.",
        "origins": list(args.origins),
        "horizon": HORIZON,
        "time_basis": "CubicSpline k=25 for occurrence; positive-size k=min(25,max(5,n_positive//8)).",
        "weekday_basis": "Cyclic CubicSpline k=7, weekday 0=Monday..6=Sunday, boundary knots -0.5 and 6.5.",
        "model_structure": "Binomial(logit) occurrence GAM times Gamma(log) positive-size GAM, separately fit per item-store-origin.",
        "smoothing_selection": "REML per fit",
        "positive_size_fallback": f"Use historical positive mean if fewer than {args.min_positive_days} positive observations or fewer than 2 distinct positive sizes.",
        "primary_forecast": "Unmasked expected units P(sale>0) * E(units|sale>0), no use of future price-live status.",
        "interval_note": "95% bands for expected conditional mean via delta-method propagation of occurrence and positive-size model SEs, assuming component-estimate independence; not full predictive intervals.",
        "positive_size_likelihood_note": "Gamma(log) is used as a positive continuous/quasi-likelihood approximation to the conditional mean of positive integer sales. Its AIC is interpreted only within the positive-size component and is not a discrete count-likelihood score.",
        "aic_comparison_note": "AIC is comparable only among candidate fits to the same response component and same observed rows; do not compare AIC across occurrence and positive-size components or across series with different row counts.",
        "smooth_curve_interval": "Pointwise approximate Bayesian intervals on the link-scale partial smooth, fit +/- 1.96*SE.",
        "backend": "pymgcv over R mgcv; selected for explicit cyclic cubic spline support.",
        "final_temporal_holdout": "2015 origins are not used in this GAM prototype.",
    }, indent=2))

    diagnostic_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    curve_rows_all: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    k_check_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    series_metric_rows: list[dict[str, Any]] = []

    for store in args.stores:
        store = store.lower()
        dat = store_data[store]
        sales = dat["sales"]
        prices = dat["prices"]
        manifest = dat["manifest"]
        sample_rows = manifest["row_index"].to_numpy(dtype=np.int64)
        sample_items = manifest["item_id"].astype(str).to_numpy()
        print(f"\n{'=' * 76}\nStore {store.upper()} | {len(sample_rows)} series")
        for origin in args.origins:
            end_day = origin + HORIZON - 1
            if end_day > sales.shape[1]:
                raise ValueError(f"Invalid d_{origin}..d_{end_day} for {store.upper()}")
            train_days = origin - 1
            future_weekday = weekday_all[origin - 1:end_day]
            actual_full = np.asarray(sales[:, origin - 1:end_day], dtype=np.float32)
            actual_sample = actual_full[sample_rows]
            train_sample = np.asarray(sales[sample_rows, :train_days], dtype=np.float32)
            train_price_sample = np.asarray(prices[sample_rows, :train_days], dtype=np.float32)
            sample_scales = _series_scales(train_sample)

            baseline_forecasts, full_weights = _load_reference_models(
                store, origin, end_day, sample_rows, actual_full, args.reference, args.phase_a_reference
            )
            if full_weights.size == sales.shape[0] and np.isfinite(full_weights[sample_rows]).all():
                sample_weights = full_weights[sample_rows].astype(np.float64)
                if sample_weights.sum() > 0:
                    sample_weights = sample_weights / sample_weights.sum()
            else:
                sample_weights = np.full(len(sample_rows), 1.0 / len(sample_rows), dtype=np.float64)

            origin_cache = cache_root / store / f"origin_d{origin:04d}_d{end_day:04d}"
            origin_cache.mkdir(parents=True, exist_ok=True)
            print(f"  Origin d_{origin}..d_{end_day}; fitting GAMs on d_1..d_{train_days}")
            gam_predictions = np.zeros((len(sample_rows), HORIZON), dtype=np.float32)
            origin_diagnostics: list[dict[str, Any]] = []
            origin_fit_rows: list[dict[str, Any]] = []
            origin_curve_rows: list[dict[str, Any]] = []

            for j, (row_index, item_id) in enumerate(zip(sample_rows, sample_items)):
                cache_npz = origin_cache / f"row_{int(row_index):05d}.npz"
                cache_json = origin_cache / f"row_{int(row_index):05d}.json"
                cache_is_current = False
                if cache_npz.is_file() and cache_json.is_file() and not args.force:
                    cached_json = json.loads(cache_json.read_text())
                    cache_is_current = cached_json.get("_model_version") == MODEL_VERSION
                if cache_is_current:
                    with np.load(cache_npz, allow_pickle=False) as saved:
                        arrays = {k: np.asarray(saved[k]).copy() for k in saved.files}
                    diag = json.loads(cache_json.read_text())
                    smooth_cache = diag.pop("_smooth_rows", [])
                    fit_cache = diag.pop("_fit_rows", [])
                    diag.pop("_model_version", None)
                    if diag.get("fit_error"):
                        failure_rows.append({
                            "store": store.upper(), "row_index": int(row_index), "item_id": str(item_id),
                            "forecast_start_day": int(origin),
                            "error_type": diag.get("fit_error_type", "CachedFitFailure"),
                            "error": diag.get("fit_error", "Cached fit used fallback"),
                        })
                    origin_diagnostics.append(diag)
                    origin_fit_rows.extend(fit_cache)
                    origin_curve_rows.extend(smooth_cache)
                else:
                    try:
                        y_train = np.asarray(sales[row_index, :train_days], dtype=np.float32)
                        p_train = np.asarray(prices[row_index, :train_days], dtype=np.float32)
                        diag, arrays, fit_rows, smooth_rows = fit_one_series_origin(
                            store=store, row_index=int(row_index), item_id=str(item_id),
                            train_sales=y_train, train_price=p_train,
                            train_weekday=weekday_all[:train_days], future_weekday=future_weekday,
                            actual=actual_sample[j], origin=origin,
                            min_positive_days=args.min_positive_days,
                            GAM=GAM, S=S, CubicSpline=CubicSpline, Binomial=Binomial, Gamma=Gamma, ro=ro,
                            save_curves=True,
                        )
                        np.savez_compressed(cache_npz, **arrays)
                        cache_json.write_text(json.dumps({
                            **diag, "_model_version": MODEL_VERSION,
                            "_smooth_rows": smooth_rows, "_fit_rows": fit_rows,
                        }, allow_nan=True))
                        origin_diagnostics.append(diag)
                        origin_fit_rows.extend(fit_rows)
                        origin_curve_rows.extend(smooth_rows)
                    except Exception as exc:
                        failure_rows.append({
                            "store": store.upper(), "row_index": int(row_index), "item_id": str(item_id),
                            "forecast_start_day": origin, "error_type": type(exc).__name__, "error": str(exc),
                        })
                        print(f"    WARNING {item_id}: {type(exc).__name__}: {exc}")
                        # Keep a simple-rate fallback so this sampled series remains in portfolio metrics.
                        live = np.isfinite(p_train)
                        y_live = y_train[live]
                        p0, m0 = _fallback_rates(y_live)
                        fallback = np.full(HORIZON, p0 * m0, dtype=np.float32)
                        arrays = {
                            "occurrence_probability": np.full(HORIZON, p0, dtype=np.float32),
                            "occurrence_probability_se": np.full(HORIZON, np.nan, dtype=np.float32),
                            "positive_size_mean": np.full(HORIZON, m0, dtype=np.float32),
                            "positive_size_mean_se": np.full(HORIZON, np.nan, dtype=np.float32),
                            "expected_mean": fallback,
                            "expected_mean_se": np.full(HORIZON, np.nan, dtype=np.float32),
                            "expected_mean_lower95": np.full(HORIZON, np.nan, dtype=np.float32),
                            "expected_mean_upper95": np.full(HORIZON, np.nan, dtype=np.float32),
                            "actual": np.asarray(actual_sample[j], dtype=np.float32),
                        }
                        diag = {
                            "store": store.upper(), "row_index": int(row_index), "item_id": str(item_id),
                            "forecast_start_day": int(origin), "forecast_end_day": int(end_day),
                            "n_live_training_days": int(live.sum()), "n_positive_training_days": int((y_live > 0).sum()),
                            "occurrence_model_status": "fallback_historical_rate_after_fit_error",
                            "size_model_status": "fallback_historical_positive_mean_after_fit_error",
                            "occurrence_fit_seconds": float("nan"), "size_fit_seconds": float("nan"),
                            "forecast_expected_units_28d": float(fallback.sum()),
                            "actual_units_28d": float(actual_sample[j].sum()), "forecast_to_actual_pct": float("nan"),
                            "fit_error_type": type(exc).__name__, "fit_error": str(exc),
                        }
                        np.savez_compressed(cache_npz, **arrays)
                        cache_json.write_text(json.dumps({
                            **diag, "_model_version": MODEL_VERSION,
                            "_smooth_rows": [], "_fit_rows": [],
                        }, allow_nan=True))
                        origin_diagnostics.append(diag)

                gam_predictions[j] = arrays["expected_mean"]
                for h in range(HORIZON):
                    forecast_rows.append({
                        "store": store.upper(), "row_index": int(row_index), "item_id": str(item_id),
                        "forecast_start_day": int(origin), "forecast_day": int(origin + h),
                        "horizon": h + 1, "weekday": int(future_weekday[h]),
                        "actual_units": float(arrays["actual"][h]),
                        "occurrence_probability": float(arrays["occurrence_probability"][h]),
                        "positive_size_mean": float(arrays["positive_size_mean"][h]),
                        "expected_units": float(arrays["expected_mean"][h]),
                        "expected_mean_se": float(arrays["expected_mean_se"][h]),
                        "expected_mean_lower95": float(arrays["expected_mean_lower95"][h]),
                        "expected_mean_upper95": float(arrays["expected_mean_upper95"][h]),
                    })
                if (j + 1) % 10 == 0 or j + 1 == len(sample_rows):
                    print(f"    completed/reused {j + 1:>3}/{len(sample_rows)} series")

            diagnostic_rows.extend(origin_diagnostics)
            origin_fit_rows_all = origin_fit_rows
            origin_curve_rows_all = origin_curve_rows
            _write_frame(origin_fit_rows_all, origin_cache / "gam_fit_diagnostics.csv", ["item_id", "component"])
            _write_frame(origin_curve_rows_all, origin_cache / "gam_smooth_curves.csv", ["item_id", "component", "term", "x_value"])
            curve_rows_all.extend(origin_curve_rows_all)

            # Evaluate sample GAM and cached reference forecasts on identical rows.
            model_forecasts = {"TwoPartGAM_pymgcv": gam_predictions}
            for name, full_pred in baseline_forecasts.items():
                model_forecasts[name] = full_pred
            for model_name, prediction in model_forecasts.items():
                m = _metrics(actual_sample, prediction, sample_scales, sample_weights)
                metric_rows.append({
                    "store": store.upper(), "forecast_start_day": int(origin),
                    "forecast_end_day": int(end_day), "model": model_name,
                    "n_series": int(len(sample_rows)), "horizon": HORIZON,
                    **m,
                })
                for j, (row_index, item_id) in enumerate(zip(sample_rows, sample_items)):
                    series_metric_rows.append({
                        "store": store.upper(), "forecast_start_day": int(origin),
                        "row_index": int(row_index), "item_id": str(item_id), "model": model_name,
                        "actual_28d_units": float(actual_sample[j].sum()),
                        "forecast_28d_units": float(prediction[j].sum()),
                        "error_28d_units": float(prediction[j].sum() - actual_sample[j].sum()),
                        "mae": float(np.mean(np.abs(prediction[j] - actual_sample[j]))),
                        "wape": float(np.abs(prediction[j] - actual_sample[j]).sum() / np.abs(actual_sample[j]).sum())
                        if np.abs(actual_sample[j]).sum() > 0 else float("nan"),
                    })

            # Optional k-index diagnostics on a small fixed subset; this is not run on every fit.
            if args.k_check_series_per_store > 0:
                chosen = set(sample_rows[: min(args.k_check_series_per_store, len(sample_rows))].tolist())
                for j, row_index in enumerate(sample_rows):
                    if int(row_index) not in chosen:
                        continue
                    item_id = str(sample_items[j])
                    cache_json = origin_cache / f"row_{int(row_index):05d}.json"
                    # Refit only these selected diagnostics in this process; the fitted objects are
                    # intentionally not retained for all 240 series to control memory.
                    y_train = np.asarray(sales[row_index, :train_days], dtype=np.float32)
                    p_train = np.asarray(prices[row_index, :train_days], dtype=np.float32)
                    live_idx = np.flatnonzero(np.isfinite(p_train))
                    y_live = y_train[live_idx]
                    if len(live_idx) < 60 or np.unique(y_live > 0).size < 2:
                        continue
                    t_live = live_idx.astype(np.float64) - live_idx[0]
                    dow_live = weekday_all[live_idx].astype(np.float64)
                    frame_occ = pd.DataFrame({"t": t_live, "dow": dow_live, "occurrence": (y_live > 0).astype(np.int8)})
                    try:
                        k_model = _fit_component(frame_occ, "occurrence", Binomial(link="logit"), 25, GAM, S, CubicSpline)
                        check = k_model.check_k(subsample=5000, n_rep=100)
                        if hasattr(check, "to_dict"):
                            records = check.to_dict(orient="records") if isinstance(check, pd.DataFrame) else check.to_dicts()
                        else:
                            records = []
                        for record in records:
                            k_check_rows.append({
                                "store": store.upper(), "item_id": item_id, "row_index": int(row_index),
                                "forecast_start_day": int(origin), "component": "occurrence", **record,
                            })
                    except Exception as exc:
                        failure_rows.append({
                            "store": store.upper(), "row_index": int(row_index), "item_id": item_id,
                            "forecast_start_day": int(origin), "error_type": "check_k_" + type(exc).__name__,
                            "error": str(exc),
                        })

            print(f"  Completed {store.upper()} d_{origin}..d_{end_day}: sample GAM forecast/actual="
                  f"{metric_rows[-(len(model_forecasts))]['aggregate_forecast_to_actual_pct']:.1f}%")
            gc.collect()

    # Consolidated artefacts. The per-series NPZ/JSON caches are the resume source.
    _write_frame(diagnostic_rows, args.output / "gam_model_diagnostics.csv", ["store", "forecast_start_day", "item_id"])
    _write_frame(forecast_rows, args.output / "gam_forecasts.csv", ["store", "forecast_start_day", "item_id", "horizon"])
    _write_frame(curve_rows_all, args.output / "gam_smooth_curves.csv", ["store", "forecast_start_day", "item_id", "component", "term", "x_value"])
    _write_frame(metric_rows, args.output / "gam_store_origin_metrics.csv", ["store", "forecast_start_day", "model"])
    _write_frame(series_metric_rows, args.output / "gam_per_series_metrics.csv", ["store", "forecast_start_day", "model", "item_id"])
    _write_frame(failure_rows, args.output / "gam_failures.csv", ["store", "forecast_start_day", "item_id"])
    _write_frame(k_check_rows, args.output / "gam_k_checks.csv", ["store", "forecast_start_day", "item_id"])

    # Mean/std summaries over the small set of prototype forecast windows.
    if metric_rows:
        metric_df = pl.DataFrame(metric_rows)
        summary_cols = ["mae", "rmse", "wape", "bias_per_observation", "aggregate_error_pct_actual",
                        "aggregate_forecast_to_actual_pct", "sample_revenue_weighted_rmsse_style"]
        aggs: list[pl.Expr] = [pl.len().alias("n_origins")]
        for col in summary_cols:
            aggs.extend([pl.col(col).mean().alias(f"{col}_mean"), pl.col(col).std().alias(f"{col}_std")])
        metric_df.group_by(["store", "model"]).agg(aggs).sort(["store", "model"]).write_csv(
            args.output / "gam_store_summary.csv"
        )

    # Smooth-function visual examples: two items per store, both components/terms.
    if args.plot_examples_per_store > 0 and curve_rows_all:
        import matplotlib.pyplot as plt
        curves_df = pd.DataFrame(curve_rows_all)
        plot_dir = args.output / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for store in [s.upper() for s in args.stores]:
            ids = manifest_all.loc[manifest_all["store"] == store, "item_id"].astype(str).tolist()[:args.plot_examples_per_store]
            for item_id in ids:
                for origin in args.origins:
                    subset = curves_df[(curves_df["store"] == store) & (curves_df["item_id"] == item_id) & (curves_df["forecast_start_day"] == origin)]
                    if subset.empty:
                        continue
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    for row, comp in enumerate(("occurrence", "positive_size")):
                        for col, term in enumerate(("time_trend", "cyclic_weekday")):
                            ax = axes[row, col]
                            block = subset[(subset["component"] == comp) & (subset["term"] == term)].sort_values("x_value")
                            if block.empty:
                                ax.set_visible(False)
                                continue
                            ax.plot(block["x_value"], block["effect_link"], label="partial effect")
                            ax.fill_between(block["x_value"], block["lower_95_link"], block["upper_95_link"], alpha=0.2, label="95% pointwise band")
                            ax.set_title(f"{comp}: {term}")
                            ax.set_ylabel("Link-scale effect")
                            ax.legend(loc="best")
                    fig.suptitle(f"{store} / {item_id} / origin d_{origin}")
                    fig.tight_layout()
                    safe_id = item_id.replace("/", "_")
                    fig.savefig(plot_dir / f"smooth_{store}_{safe_id}_d{origin}.png", dpi=140)
                    plt.close(fig)

    print("\nM5 GAM prototype complete.")
    print(f"  Sample manifest: {sample_manifest_path}")
    for name in (
        "gam_model_diagnostics.csv", "gam_smooth_curves.csv", "gam_forecasts.csv",
        "gam_store_origin_metrics.csv", "gam_store_summary.csv", "gam_per_series_metrics.csv",
        "gam_k_checks.csv", "gam_failures.csv",
    ):
        print(f"  {args.output / name}")


if __name__ == "__main__":
    main()
