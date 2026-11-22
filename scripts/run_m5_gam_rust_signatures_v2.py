#!/usr/bin/env python3
"""M5 descriptive demand signatures using mgcv-rust (no R dependency).

Purpose
-------
This script uses GAMs to describe the temporal structure of M5 item-store
series. It does NOT generate forecasts or select a forecasting model.

For each sampled series and historical cutoff it:
  * computes empirical demand, zero-run, inter-arrival and weekly descriptors;
  * fits a Binomial(logit) GAM to price-live-day occurrence when the minority
    class has enough observations for a minimally useful smooth;
  * fits a Gaussian(identity) GAM to log1p(positive daily sales), when there
    are enough positive observations;
  * obtains fitted curves and confidence intervals on the LINK scale, then
    transforms them to response/positive-size space using stable transforms;
  * records warnings, package convergence indicators, diagnostics, and skips.

The default 240-item manifest is reused from the first GAM prototype for
comparability. Two historical cutoffs (d_1341 and d_1386) are used, and no
observations after each cutoff enter that signature.
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

DEFAULT_STORES = ("ca_1", "wi_3", "tx_1")
DEFAULT_ORIGINS = (1342, 1387)
DEFAULT_SAMPLE_MANIFEST = Path("artifacts/gam_rust_signatures/gam_rust_sample_manifest.csv")
MODEL_VERSION = "mgcv_rust_two_part_signatures_v2_linkscale_adaptive_occurrence"
WEEKDAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def load_gam_class():
    try:
        from mgcv_rust import Gam
    except Exception as exc:  # pragma: no cover - local Pixi environment
        raise RuntimeError(
            "Could not import mgcv_rust. Install mgcv-rust==0.23.2 in the "
            "gam-rust Pixi environment. Original error: " + repr(exc)
        ) from exc
    return Gam


def stable_expit(x: Any) -> np.ndarray:
    """Numerically stable inverse-logit; avoids overflow for large |x|."""
    z = np.asarray(x, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def safe_expm1(x: Any) -> np.ndarray:
    """Non-negative, overflow-safe inverse of log1p for positive-size curves."""
    z = np.asarray(x, dtype=np.float64)
    z = np.clip(z, 0.0, 700.0)
    return np.expm1(z)


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
    finite = np.isfinite(values)
    values, day_idx = values[finite], day_idx[finite]
    lookup = {int(d): i for i, d in enumerate(day_idx)}
    left, right = [], []
    for i, day in enumerate(day_idx):
        j = lookup.get(int(day + lag))
        if j is not None:
            left.append(float(values[i]))
            right.append(float(values[j]))
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _max_consecutive_zero_run(y: np.ndarray, day_idx: np.ndarray) -> int:
    best = current = 0
    previous_day: int | None = None
    for value, day in zip(y, day_idx):
        day = int(day)
        if value == 0:
            current = current + 1 if previous_day is not None and day == previous_day + 1 else 1
            best = max(best, current)
        else:
            current = 0
        previous_day = day
    return int(best)


def _cache_hash(store: str, row_index: int, item_id: str, origin: int, config: dict[str, Any]) -> str:
    payload = {
        "model_version": MODEL_VERSION, "store": store, "row_index": int(row_index),
        "item_id": item_id, "origin": int(origin), "config": config,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _empirical_signature(
    *, store: str, row_index: int, item_id: str, origin: int,
    y_all: np.ndarray, price_all: np.ndarray, weekday_all: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """Calculate empirical descriptors using only days before `origin`."""
    train_end = origin - 1
    y_hist = np.asarray(y_all[:train_end], dtype=np.float64)
    price_hist = np.asarray(price_all[:train_end], dtype=np.float64)
    live_idx = np.flatnonzero(np.isfinite(price_hist)).astype(np.int64)
    y_live = y_hist[live_idx]
    dow_live = np.asarray(weekday_all[:train_end][live_idx], dtype=np.int8)
    positive_mask = y_live > 0
    positive_idx = live_idx[positive_mask]
    positive = y_live[positive_mask]
    n_live, n_positive = int(len(y_live)), int(len(positive))
    zero_share = float(np.mean(~positive_mask)) if n_live else float("nan")
    frequency = float(n_positive / n_live) if n_live else float("nan")
    positive_mean = float(np.mean(positive)) if n_positive else float("nan")
    positive_median = float(np.median(positive)) if n_positive else float("nan")
    positive_sd = float(np.std(positive, ddof=1)) if n_positive > 1 else float("nan")
    positive_cv = positive_sd / positive_mean if np.isfinite(positive_sd) and positive_mean > 0 else float("nan")
    gaps = np.diff(positive_idx).astype(np.float64) if n_positive >= 2 else np.asarray([], dtype=float)
    gap_mean = float(np.mean(gaps)) if len(gaps) else float("nan")
    gap_sd = float(np.std(gaps, ddof=1)) if len(gaps) > 1 else float("nan")

    signature: dict[str, Any] = {
        "store": store.upper(), "row_index": int(row_index), "item_id": item_id,
        "origin_day": int(origin), "training_end_day": int(train_end),
        "n_calendar_training_days": int(train_end), "n_price_live_days": n_live,
        "n_positive_days": n_positive, "n_zero_days": int(n_live - n_positive),
        "minority_class_events": int(min(n_positive, n_live - n_positive)),
        "zero_share": zero_share, "demand_frequency": frequency,
        "mean_positive_demand": positive_mean, "median_positive_demand": positive_median,
        "positive_demand_sd": positive_sd, "positive_demand_cv": positive_cv,
        "positive_demand_q90": float(np.quantile(positive, 0.9)) if n_positive else float("nan"),
        "positive_demand_max": float(np.max(positive)) if n_positive else float("nan"),
        "mean_interarrival_days": gap_mean, "sd_interarrival_days": gap_sd,
        "cv_interarrival_days": gap_sd / gap_mean if np.isfinite(gap_sd) and gap_mean > 0 else float("nan"),
        "median_interarrival_days": float(np.median(gaps)) if len(gaps) else float("nan"),
        "max_consecutive_zero_live_days": _max_consecutive_zero_run(y_live, live_idx),
        "sales_acf_lag1_calendar": _exact_lag_corr(y_live, live_idx, 1),
        "sales_acf_lag7_calendar": _exact_lag_corr(y_live, live_idx, 7),
        "occurrence_acf_lag7_calendar": _exact_lag_corr(positive_mask.astype(float), live_idx, 7),
        "price_live_fraction": n_live / train_end if train_end else float("nan"),
    }

    weekly_rows: list[dict[str, Any]] = []
    weekday_means = np.full(7, np.nan, dtype=np.float64)
    weekday_frequencies = np.full(7, np.nan, dtype=np.float64)
    weekday_counts = np.zeros(7, dtype=np.int64)
    for dow in range(7):
        mask = dow_live == dow
        weekday_counts[dow] = int(mask.sum())
        if mask.any():
            weekday_means[dow] = float(np.mean(y_live[mask]))
            weekday_frequencies[dow] = float(np.mean(positive_mask[mask]))
        signature[f"weekday_{WEEKDAY_NAMES[dow]}_mean_sales"] = weekday_means[dow]
        signature[f"weekday_{WEEKDAY_NAMES[dow]}_positive_frequency"] = weekday_frequencies[dow]
        weekly_rows.append({
            "store": store.upper(), "row_index": int(row_index), "item_id": item_id,
            "origin_day": int(origin), "weekday": dow, "weekday_name": WEEKDAY_NAMES[dow],
            "n_price_live_days": int(mask.sum()), "positive_sales_frequency": weekday_frequencies[dow],
            "mean_sales_per_price_live_day": weekday_means[dow],
        })

    weekday_mean = float(np.nanmean(weekday_means[:5])) if np.isfinite(weekday_means[:5]).any() else float("nan")
    weekend_mean = float(np.nanmean(weekday_means[5:])) if np.isfinite(weekday_means[5:]).any() else float("nan")
    signature["weekend_lift"] = weekend_mean / weekday_mean if np.isfinite(weekday_mean) and weekday_mean > 0 and np.isfinite(weekend_mean) else float("nan")
    signature["weekday_sales_profile_sd"] = float(np.nanstd(weekday_means)) if np.isfinite(weekday_means).any() else float("nan")
    signature["weekday_sales_profile_mean"] = float(np.nanmean(weekday_means)) if np.isfinite(weekday_means).any() else float("nan")
    signature["weekday_observation_min"] = int(weekday_counts.min())
    signature["weekday_observation_max"] = int(weekday_counts.max())
    return signature, weekly_rows, live_idx, y_live, dow_live


def _binomial_deviance(y: np.ndarray, mu: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(y, dtype=np.float64)
    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-12, 1 - 1e-12)
    if len(y) == 0 or len(y) != len(mu):
        return float("nan"), float("nan"), float("nan")
    p0 = float(np.clip(np.mean(y), 1e-12, 1 - 1e-12))
    dev = -2.0 * float(np.sum(y * np.log(mu) + (1.0 - y) * np.log1p(-mu)))
    null_dev = -2.0 * float(np.sum(y * math.log(p0) + (1.0 - y) * math.log1p(-p0)))
    explained = 1.0 - dev / null_dev if null_dev > 1e-12 else float("nan")
    return dev, null_dev, explained


def _gaussian_deviance(y: np.ndarray, fitted: np.ndarray) -> tuple[float, float, float, float]:
    y = np.asarray(y, dtype=np.float64)
    fitted = np.asarray(fitted, dtype=np.float64)
    dev = float(np.sum((y - fitted) ** 2))
    null_dev = float(np.sum((y - np.mean(y)) ** 2))
    explained = 1.0 - dev / null_dev if null_dev > 1e-12 else float("nan")
    n = max(len(y), 1)
    sigma2 = max(dev / n, 1e-12)
    loglik = -0.5 * n * (math.log(2.0 * math.pi * sigma2) + 1.0)
    return dev, null_dev, explained, loglik


def _residual_diagnostics(family: str, y: np.ndarray, fitted: np.ndarray, day_idx: np.ndarray) -> dict[str, float]:
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


def _warning_classifications(messages: list[str]) -> dict[str, Any]:
    joined = " | ".join(messages).lower()
    return {
        "warning_count": int(len(messages)),
        "warning_text_json": json.dumps(messages),
        "minority_events_warning": bool("minority" in joined or "events per parameter" in joined),
        "pirls_warning": bool("pirls" in joined or "inner" in joined and "converg" in joined),
        "outer_optimizer_warning": bool(
            "step failure" in joined or "outer" in joined and "converg" in joined
            or "failed to converge" in joined or "did not converge" in joined
        ),
        "overflow_warning": bool("overflow" in joined),
    }


def _package_convergence_flag(model: Any) -> Any:
    """Record a package flag without assuming its exact convergence semantics."""
    for name in ("converged_", "fit_converged_", "converged"):
        if hasattr(model, name):
            value = getattr(model, name)
            try:
                return bool(np.asarray(value).reshape(-1)[0])
            except Exception:
                return str(value)
    return None


def _model_stats(model: Any, family: str, y: np.ndarray, fitted_link: np.ndarray,
                 day_idx: np.ndarray, warnings_list: list[str]) -> dict[str, Any]:
    edf = _float_list(getattr(model, "edf_", []))
    lambdas = _float_list(getattr(model, "lambda_", []))
    basis_k = _float_list(getattr(model, "k_", []))
    basis_names = [str(v) for v in getattr(model, "bs_", [])] if getattr(model, "bs_", None) is not None else []
    if family == "binomial":
        fitted_response = stable_expit(fitted_link)
        dev, null_dev, dev_expl = _binomial_deviance(y, fitted_response)
        loglik = -0.5 * dev
        effective_parameters = float(sum(edf)) + 1.0
    else:
        fitted_response = fitted_link
        dev, null_dev, dev_expl, loglik = _gaussian_deviance(y, fitted_response)
        effective_parameters = float(sum(edf)) + 2.0
    aic_approx = -2.0 * loglik + 2.0 * effective_parameters
    package_flag = _package_convergence_flag(model)
    stats = {
        "fit_status": "fit_completed_finite_predictions",
        "family": family,
        "n_observations": int(len(y)),
        "edf_json": json.dumps(edf), "lambda_json": json.dumps(lambdas),
        "basis_k_json": json.dumps(basis_k), "basis_json": json.dumps(basis_names),
        "edf_time": float(edf[0]) if edf else float("nan"),
        "lambda_time": float(lambdas[0]) if lambdas else float("nan"),
        "deviance": float(dev), "null_deviance": float(null_dev),
        "deviance_explained": float(dev_expl),
        "aic_conditional_approx": float(aic_approx),
        "package_aic_attribute": _float(getattr(model, "aic_", getattr(model, "aic", np.nan))),
        "package_deviance_attribute": _float(getattr(model, "deviance_", np.nan)),
        "package_convergence_flag_uninterpreted": package_flag,
        "package_convergence_flag_is_false_uninterpreted": bool(package_flag is False),
        **_residual_diagnostics(family, y, fitted_response, day_idx),
        **_warning_classifications(warnings_list),
        "summary_text": str(model.summary()),
    }
    return stats


def _fit_component(
    Gam: Any, family: str, x: np.ndarray, y: np.ndarray, k: int,
    ci_samples: int, curve_points: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit one component; all prediction/CI calls explicitly remain on link scale."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X = pd.DataFrame({"elapsed_day": x})
    grid = np.linspace(float(np.min(x)), float(np.max(x)), curve_points, dtype=np.float64)
    grid_X = pd.DataFrame({"elapsed_day": grid})
    model = Gam(
        family=family, method="REML", k_default=int(k),
        term_k_mapping={"elapsed_day": int(k)},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
        # IMPORTANT: the Binomial inverse link is not called in the package.
        fitted_link = np.asarray(model.predict(X, scale="link"), dtype=np.float64).reshape(-1)
        mean_link, lower_link, upper_link = model.predict_ci(
            grid_X, scale="link", level=0.95, n_samples=int(ci_samples)
        )

    warning_messages = [str(w.message) for w in caught]
    if len(fitted_link) != len(y) or not np.isfinite(fitted_link).all():
        raise ValueError("GAM returned non-finite or incorrectly sized link-scale training predictions")
    mean_link = np.asarray(mean_link, dtype=np.float64).reshape(-1)
    lower_link = np.asarray(lower_link, dtype=np.float64).reshape(-1)
    upper_link = np.asarray(upper_link, dtype=np.float64).reshape(-1)
    if not (len(mean_link) == len(lower_link) == len(upper_link) == len(grid)):
        raise ValueError("predict_ci(scale='link') returned an unexpected curve shape")
    if not (np.isfinite(mean_link).all() and np.isfinite(lower_link).all() and np.isfinite(upper_link).all()):
        raise ValueError("predict_ci(scale='link') returned non-finite values")

    if family == "binomial":
        mean_response = stable_expit(mean_link)
        lower_response = stable_expit(lower_link)
        upper_response = stable_expit(upper_link)
        value_scale = "probability_response_scale_stable_expit_from_link"
    else:
        mean_response = safe_expm1(mean_link)
        lower_response = safe_expm1(lower_link)
        upper_response = safe_expm1(upper_link)
        value_scale = "typical_positive_size_expm1_from_log1p_link"

    # Monotonic transformations preserve interval ordering. Keep defensive bounds.
    lower_response = np.minimum(lower_response, mean_response)
    upper_response = np.maximum(upper_response, mean_response)
    stats = _model_stats(model, family, y, fitted_link, np.asarray(x, dtype=np.int64), warning_messages)
    stats["link_scale_prediction_api"] = "Gam.predict(..., scale='link')"
    stats["link_scale_interval_api"] = "Gam.predict_ci(..., scale='link')"
    stats["inverse_link_transform"] = "stable_expit" if family == "binomial" else "safe_expm1"
    stats["link_prediction_abs_max"] = float(np.max(np.abs(fitted_link)))
    stats["curve_probability_saturated"] = bool(family == "binomial" and np.any((mean_response < 1e-8) | (mean_response > 1 - 1e-8)))
    stats["curve_value_start"] = float(mean_response[0])
    stats["curve_value_end"] = float(mean_response[-1])
    stats["curve_change_end_minus_start"] = float(mean_response[-1] - mean_response[0])
    duration = max(float(np.max(x) - np.min(x)), 1.0)
    stats["curve_change_per_365d"] = stats["curve_change_end_minus_start"] * 365.25 / duration
    stats["curve_range_p95_p05"] = float(np.quantile(mean_response, 0.95) - np.quantile(mean_response, 0.05))
    stats["curve_mean"] = float(np.mean(mean_response))
    curve_rows = []
    for xx, ml, ll, ul, mr, lr, ur in zip(grid, mean_link, lower_link, upper_link,
                                          mean_response, lower_response, upper_response):
        curve_rows.append({
            "elapsed_day": float(xx), "link_fit": float(ml),
            "link_lower_95": float(ll), "link_upper_95": float(ul),
            "curve_value": float(mr), "lower_95": float(lr), "upper_95": float(ur),
            "value_scale": value_scale,
            "interval_type": "pointwise coefficient-based 95% confidence interval transformed from link scale; not predictive interval",
        })
    return stats, curve_rows


def _fit_one_series_origin(
    Gam: Any, store: str, row_index: int, item_id: str, origin: int,
    sales_row: np.ndarray, price_row: np.ndarray, weekday_all: np.ndarray,
    min_live_days: int, min_positive_days: int, min_minority_events: int,
    events_per_basis_parameter: int, k_trend: int,
    ci_samples: int, curve_points: int,
) -> dict[str, Any]:
    signature, weekly_rows, live_idx, y_live, dow_live = _empirical_signature(
        store=store, row_index=row_index, item_id=item_id, origin=origin,
        y_all=sales_row, price_all=price_row, weekday_all=weekday_all,
    )
    n_live = int(len(live_idx))
    n_positive = int(np.sum(y_live > 0))
    n_zero = int(n_live - n_positive)
    minority_events = min(n_positive, n_zero)
    positive_mask = y_live > 0
    first_live = int(live_idx[0]) if n_live else 0
    elapsed_live = live_idx - first_live if n_live else np.asarray([], dtype=np.int64)
    diagnostics: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    features: dict[str, Any] = {}
    components = (
        ("occurrence", "binomial", n_live),
        ("positive_size", "gaussian", n_positive),
    )

    for component, family, n_component in components:
        base = {
            "store": store.upper(), "row_index": int(row_index), "item_id": item_id,
            "origin_day": int(origin), "training_end_day": int(origin - 1),
            "component": component,
        }
        if n_live < min_live_days:
            diagnostics.append({
                **base, "fit_status": "insufficient_live_history", "n_live_days": n_live,
                "n_component_observations": int(n_component),
                "error": f"Only {n_live} price-live training days; minimum is {min_live_days}",
            })
            features[f"{component}_fit_status"] = "insufficient_live_history"
            continue

        if component == "occurrence":
            y = positive_mask.astype(np.float64)
            x = elapsed_live.astype(np.float64)
            if n_positive == 0 or n_zero == 0:
                rate = float(n_positive / n_live) if n_live else float("nan")
                diagnostics.append({
                    **base, "fit_status": "constant_occurrence_no_smooth", "n_live_days": n_live,
                    "n_component_observations": n_live, "minority_class_events": minority_events,
                    "constant_occurrence_rate": rate,
                })
                features.update({"occurrence_fit_status": "constant_occurrence_no_smooth",
                                 "occurrence_constant_rate": rate})
                continue
            # Minimum basis dimension is 5. For k basis coefficients, require
            # at least `events_per_basis_parameter * (k - 1)` minority events.
            # This directly responds to mgcv-rust's minority-events-per-parameter warning.
            max_k_from_events = minority_events // events_per_basis_parameter + 1
            max_k_from_rows = max(5, n_live // 25)
            k = min(k_trend, max_k_from_events, max_k_from_rows)
            if minority_events < min_minority_events or k < 5:
                diagnostics.append({
                    **base, "fit_status": "insufficient_minority_events_for_smooth",
                    "n_live_days": n_live, "n_component_observations": n_live,
                    "n_positive_days": n_positive, "n_zero_days": n_zero,
                    "minority_class_events": minority_events,
                    "minimum_minority_events": min_minority_events,
                    "occurrence_rate_empirical": float(n_positive / n_live),
                    "k_occurrence_selected": int(max(k, 0)),
                    "k_occurrence_minimum": 5,
                    "reason": "Retain empirical occurrence descriptors; do not fit a smooth with inadequate minority-class support.",
                })
                features.update({"occurrence_fit_status": "insufficient_minority_events_for_smooth",
                                 "occurrence_empirical_rate": float(n_positive / n_live),
                                 "occurrence_minority_events": minority_events,
                                 "occurrence_k_selected": int(max(k, 0))})
                continue
            features["occurrence_minority_events"] = minority_events
            features["occurrence_k_selected"] = int(k)
        else:
            positive = y_live[positive_mask]
            y = np.log1p(positive).astype(np.float64)
            x = elapsed_live[positive_mask].astype(np.float64)
            if n_positive < min_positive_days:
                diagnostics.append({
                    **base, "fit_status": "insufficient_positive_history", "n_live_days": n_live,
                    "n_component_observations": n_positive, "minimum_positive_days": min_positive_days,
                    "positive_size_empirical_typical": float(np.expm1(np.mean(y))) if n_positive else float("nan"),
                })
                features["positive_size_fit_status"] = "insufficient_positive_history"
                continue
            if np.unique(y).size < 2:
                typical = float(np.expm1(np.mean(y)))
                diagnostics.append({
                    **base, "fit_status": "constant_positive_size_no_smooth", "n_live_days": n_live,
                    "n_component_observations": n_positive, "constant_typical_size": typical,
                })
                features.update({"positive_size_fit_status": "constant_positive_size_no_smooth",
                                 "positive_size_constant_typical": typical})
                continue
            k = min(k_trend, max(5, n_positive // 8))
            features["positive_size_k_selected"] = int(k)

        started = time.perf_counter()
        try:
            stats, component_curves = _fit_component(
                Gam=Gam, family=family, x=x, y=y, k=int(k),
                ci_samples=ci_samples, curve_points=curve_points,
            )
            fit_seconds = time.perf_counter() - started
            warning_count = int(stats["warning_count"])
            if stats["pirls_warning"] or stats["outer_optimizer_warning"]:
                fit_status = "fit_nonconverged_warning"
            elif warning_count:
                fit_status = "fit_warning"
            else:
                fit_status = "fit_completed_finite_predictions"
            stats.update({
                **base, "fit_status": fit_status, "n_live_days": n_live,
                "n_component_observations": int(n_component), "k_selected": int(k),
                "minority_class_events": minority_events if component == "occurrence" else None,
                "fit_seconds": float(fit_seconds),
            })
            diagnostics.append(stats)
            for row in component_curves:
                curves.append({**base, **row})
            prefix = component
            features.update({
                f"{prefix}_fit_status": fit_status,
                f"{prefix}_edf_time": stats.get("edf_time", float("nan")),
                f"{prefix}_lambda_time": stats.get("lambda_time", float("nan")),
                f"{prefix}_deviance_explained": stats.get("deviance_explained", float("nan")),
                f"{prefix}_aic_conditional_approx": stats.get("aic_conditional_approx", float("nan")),
                f"{prefix}_curve_start": stats.get("curve_value_start", float("nan")),
                f"{prefix}_curve_end": stats.get("curve_value_end", float("nan")),
                f"{prefix}_curve_change_per_365d": stats.get("curve_change_per_365d", float("nan")),
                f"{prefix}_curve_range_p95_p05": stats.get("curve_range_p95_p05", float("nan")),
                f"{prefix}_warning_count": warning_count,
                f"{prefix}_outer_optimizer_warning": bool(stats["outer_optimizer_warning"]),
                f"{prefix}_pirls_warning": bool(stats["pirls_warning"]),
                f"{prefix}_overflow_warning": bool(stats["overflow_warning"]),
            })
        except Exception as exc:
            diagnostics.append({
                **base, "fit_status": "fit_exception", "n_live_days": n_live,
                "n_component_observations": int(n_component), "k_selected": int(k),
                "error_type": type(exc).__name__, "error": str(exc),
            })
            features[f"{component}_fit_status"] = "fit_exception"

    signature.update(features)
    return {"signature": signature, "diagnostics": diagnostics, "curves": curves, "weekly": weekly_rows}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, allow_nan=True, separators=(",", ":")))
    tmp.replace(path)


def _load_manifest(path: Path, stores: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Sample manifest not found: {path}")
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
    absent = [s for s in stores if counts.get(s, 0) == 0]
    if absent:
        raise ValueError(f"Sample manifest has no rows for stores: {absent}")
    if manifest.duplicated(["store", "row_index"]).any():
        raise ValueError("Sample manifest contains duplicate store/row_index entries")
    return manifest.sort_values(["store", "row_index"]).reset_index(drop=True)


def _write_csv(rows: list[dict[str, Any]], path: Path, sort_cols: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if sort_cols and not df.empty and all(c in df.columns for c in sort_cols):
        df = df.sort_values(sort_cols, kind="stable")
    df.to_csv(path, index=False)


def _make_plots(curve_df: pd.DataFrame, manifest: pd.DataFrame, output: Path, n_examples: int) -> None:
    if curve_df.empty or n_examples <= 0:
        return
    import matplotlib.pyplot as plt
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for store in sorted(manifest["store"].unique()):
        subset = manifest[manifest["store"] == store]
        if "sample_zero_share" in subset.columns:
            subset = subset.sort_values("sample_zero_share")
        else:
            subset = subset.sort_values("item_id")
        chosen_positions = np.unique(np.linspace(0, len(subset) - 1, min(n_examples, len(subset))).round().astype(int))
        for _, item in subset.iloc[chosen_positions].iterrows():
            for origin in sorted(curve_df.loc[curve_df["store"] == store.upper(), "origin_day"].unique()):
                c_all = curve_df[(curve_df["store"] == store.upper()) &
                                 (curve_df["item_id"] == item["item_id"]) &
                                 (curve_df["origin_day"] == origin)]
                if c_all.empty:
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
                for ax, component, label in zip(
                    axes, ("occurrence", "positive_size"),
                    ("Demand occurrence probability", "Typical positive-demand size"),
                ):
                    c = c_all[c_all["component"] == component].sort_values("elapsed_day")
                    if c.empty:
                        ax.text(0.5, 0.5, "Smooth unavailable / not fitted", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(label)
                        continue
                    ax.plot(c["elapsed_day"], c["curve_value"], label="Fitted curve")
                    ax.fill_between(c["elapsed_day"], c["lower_95"], c["upper_95"], alpha=0.2, label="Pointwise 95% CI")
                    ax.set_title(label)
                    ax.set_xlabel("Elapsed days since first price-live day")
                    ax.legend()
                fig.suptitle(f"{store.upper()} · {item['item_id']} · history through d_{origin - 1}")
                fig.tight_layout()
                safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(item["item_id"]))
                fig.savefig(plot_dir / f"{store}_{safe_name}_d{origin}.png", dpi=140)
                plt.close(fig)


def _write_signature_drift(rows: list[dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    id_cols = ["store", "row_index", "item_id"]
    if df.empty or len(df["origin_day"].dropna().unique()) < 2:
        pd.DataFrame().to_csv(path, index=False)
        return
    origins = sorted(df["origin_day"].dropna().unique().tolist())
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns
               if c not in set(id_cols + ["origin_day", "training_end_day", "n_calendar_training_days"])]
    left = df[df.origin_day == origins[0]][id_cols + numeric]
    right = df[df.origin_day == origins[-1]][id_cols + numeric]
    merged = left.merge(right, on=id_cols, how="outer", suffixes=(f"_d{origins[0]}", f"_d{origins[-1]}"))
    for col in numeric:
        a, b = f"{col}_d{origins[0]}", f"{col}_d{origins[-1]}"
        if a in merged.columns and b in merged.columns:
            merged[f"{col}_change"] = merged[b] - merged[a]
    merged.to_csv(path, index=False)


def _summarize_diagnostics(diag_rows: list[dict[str, Any]], path: Path) -> None:
    if not diag_rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    df = pd.DataFrame(diag_rows)
    summary = (df.groupby(["store", "origin_day", "component", "fit_status"], dropna=False)
                 .size().rename("n_fits").reset_index())
    summary.to_csv(path, index=False)


def _summarize_warnings(diag_rows: list[dict[str, Any]], path: Path) -> None:
    if not diag_rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    df = pd.DataFrame(diag_rows)
    warning_cols = [c for c in (
        "warning_count", "minority_events_warning", "pirls_warning",
        "outer_optimizer_warning", "overflow_warning",
        "package_convergence_flag_is_false_uninterpreted",
    ) if c in df.columns]
    if not warning_cols:
        pd.DataFrame().to_csv(path, index=False)
        return
    aggregations = {col: ("sum" if col == "warning_count" or col.endswith("warning") or col.endswith("uninterpreted") else "sum")
                    for col in warning_cols}
    summary = df.groupby(["store", "origin_day", "component"], dropna=False).agg(aggregations).reset_index()
    summary.to_csv(path, index=False)


def run_smoke_test() -> None:
    Gam = load_gam_class()
    rng = np.random.default_rng(123)
    n = 500
    x = np.arange(n, dtype=np.float64)
    X = pd.DataFrame({"elapsed_day": x})
    p = stable_expit(-0.7 + 0.35 * np.sin(x / 55.0) + 0.0008 * x)
    y_occ = rng.binomial(1, p)
    occ, occ_curves = _fit_component(Gam, "binomial", x, y_occ, 25, 200, 80)
    y_size = 0.6 + 0.0005 * x + rng.normal(0, 0.18, n)
    size, size_curves = _fit_component(Gam, "gaussian", x, y_size, 15, 200, 80)
    assert len(occ_curves) == len(size_curves) == 80
    assert np.isfinite(occ["edf_time"]) and np.isfinite(size["lambda_time"])
    assert all(0 <= row["lower_95"] <= row["curve_value"] <= row["upper_95"] <= 1 for row in occ_curves)
    # Extreme link values prove the inverse link is stable without overflow.
    extreme = stable_expit(np.asarray([-1000.0, -100.0, 0.0, 100.0, 1000.0]))
    assert np.isfinite(extreme).all() and np.all((extreme >= 0) & (extreme <= 1))
    print("mgcv-rust v2 smoke test passed: Binomial/Gaussian fits, link-scale predictions/CI, stable inverse-logit, EDF/lambda and curves.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stores", nargs="+", default=list(DEFAULT_STORES))
    parser.add_argument("--origins", nargs="+", type=int, default=list(DEFAULT_ORIGINS))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--output", type=Path, default=Path("artifacts/gam_rust_signatures_v2"))
    parser.add_argument("--min-live-days", type=int, default=60)
    parser.add_argument("--min-positive-days", type=int, default=30)
    parser.add_argument("--min-minority-events", type=int, default=20)
    parser.add_argument("--events-per-basis-parameter", type=int, default=5)
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
        parser.error("Require min-live-days >=30, min-positive-days >=10 and k-trend >=5")
    if args.min_minority_events < 20 or args.events_per_basis_parameter < 1:
        parser.error("Require min-minority-events >=20 and events-per-basis-parameter >=1")
    if not args.origins or len(set(args.origins)) != len(args.origins):
        parser.error("--origins must be non-empty and unique")

    Gam = load_gam_class()
    try:
        package_version = importlib.metadata.version("mgcv-rust")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    if package_version != "0.23.2":
        warnings.warn(f"Expected mgcv-rust 0.23.2; found {package_version}")

    stores = [s.lower() for s in args.stores]
    manifest = _load_manifest(args.sample_manifest, stores)
    args.output.mkdir(parents=True, exist_ok=True)
    cache_root = args.output / "series_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    try:
        import polars as pl
    except Exception as exc:
        raise RuntimeError("The project environment needs Polars to read processed parquet files") from exc

    calendar_path = args.processed / "calendar.parquet"
    if not calendar_path.is_file():
        raise FileNotFoundError(calendar_path)
    calendar = pl.read_parquet(calendar_path, columns=["d_num", "date"]).sort("d_num")
    dates = pd.to_datetime(calendar["date"].to_list())
    weekday_all = np.asarray([d.weekday() for d in dates], dtype=np.int8)
    if max(args.origins) - 1 > len(weekday_all):
        raise ValueError("Calendar does not cover the latest requested cutoff")

    fit_config = {
        "model_version": MODEL_VERSION, "package_version": package_version,
        "origins": list(args.origins), "min_live_days": args.min_live_days,
        "min_positive_days": args.min_positive_days,
        "min_minority_events": args.min_minority_events,
        "events_per_basis_parameter": args.events_per_basis_parameter,
        "k_trend": args.k_trend, "curve_points": args.curve_points,
        "ci_samples": args.ci_samples,
        "occurrence_prediction_scale": "link",
        "occurrence_inverse_link": "stable_expit; piecewise evaluation avoids exp overflow",
        "occurrence_basis_rule": "k=min(k_trend, minority_events//events_per_basis_parameter+1, max(5,n_live//25)); skip if minority_events<20 or selected k<5",
        "positive_size_model": "Gaussian identity GAM on log1p(positive sales)",
        "positive_size_inverse_transform": "safe_expm1 of link-scale fitted curve; typical (median-like) positive magnitude",
        "weekly_features": "empirical weekday profiles and exact-calendar-lag autocorrelations; no cyclic GAM basis assumed",
        "purpose": "descriptive demand signatures only; no forecasts, no model choice, no clustering",
    }
    config_path = args.output / "gam_rust_v2_config.json"
    config_path.write_text(json.dumps(fit_config, indent=2))
    manifest.to_csv(args.output / "gam_rust_v2_sample_manifest.csv", index=False)

    all_signatures: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    all_curves: list[dict[str, Any]] = []
    all_weekly: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []

    for store in stores:
        subset_dir = args.processed / "subsets" / store
        sales_path = subset_dir / "sales.npy"
        price_path = subset_dir / "price.npy"
        meta_path = subset_dir / "series.parquet"
        for path in (sales_path, price_path, meta_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        sales = np.load(sales_path, mmap_mode="r")
        prices = np.load(price_path, mmap_mode="r")
        metadata = pl.read_parquet(meta_path)
        if sales.shape != prices.shape or sales.shape[0] != metadata.height:
            raise ValueError(f"Sales/price/metadata shape mismatch for {store.upper()}")
        item_col = "item_id" if "item_id" in metadata.columns else "id" if "id" in metadata.columns else None
        metadata_ids = metadata[item_col].to_list() if item_col else None
        store_manifest = manifest[manifest["store"] == store]
        print(f"\n{store.upper()}: {len(store_manifest)} series")

        for origin in args.origins:
            train_end = origin - 1
            if train_end > sales.shape[1] or train_end > prices.shape[1]:
                raise ValueError(f"Origin d_{origin} exceeds {store.upper()} sales/price arrays")
            print(f"  Signature cutoff d_{train_end} (origin d_{origin})")
            for row in store_manifest.itertuples(index=False):
                row_index, item_id = int(row.row_index), str(row.item_id)
                if not 0 <= row_index < sales.shape[0]:
                    raise IndexError(f"Manifest row {row_index} outside {store.upper()} data")
                if metadata_ids is not None:
                    actual_id = str(metadata_ids[row_index])
                    if actual_id != item_id:
                        # Some prepared metadata can retain the official suffix.
                        normalized_actual = re.sub(r"_(evaluation|validation)$", "", actual_id)
                        normalized_expected = re.sub(r"_(evaluation|validation)$", "", item_id)
                        if normalized_actual != normalized_expected:
                            raise ValueError(f"Manifest mismatch at {store.upper()} row {row_index}: {item_id} vs {actual_id}")

                key = _cache_hash(store, row_index, item_id, origin, fit_config)
                cache_file = cache_root / store / f"origin_d{origin:04d}" / f"row_{row_index:05d}.json"
                cached = None
                if cache_file.is_file() and not args.force:
                    candidate = json.loads(cache_file.read_text())
                    if candidate.get("_cache_hash") == key:
                        cached = candidate
                if cached is None:
                    try:
                        result = _fit_one_series_origin(
                            Gam=Gam, store=store, row_index=row_index, item_id=item_id,
                            origin=origin, sales_row=np.asarray(sales[row_index, :train_end]),
                            price_row=np.asarray(prices[row_index, :train_end]),
                            weekday_all=weekday_all[:train_end], min_live_days=args.min_live_days,
                            min_positive_days=args.min_positive_days,
                            min_minority_events=args.min_minority_events,
                            events_per_basis_parameter=args.events_per_basis_parameter,
                            k_trend=args.k_trend, ci_samples=args.ci_samples,
                            curve_points=args.curve_points,
                        )
                        cached = {"_cache_hash": key, **result}
                        _atomic_json(cache_file, cached)
                    except Exception as exc:
                        failure = {
                            "store": store.upper(), "row_index": row_index, "item_id": item_id,
                            "origin_day": origin, "component": "series_processing",
                            "fit_status": "series_processing_exception",
                            "error_type": type(exc).__name__, "error": str(exc),
                        }
                        cached = {
                            "_cache_hash": key,
                            "signature": {"store": store.upper(), "row_index": row_index,
                                          "item_id": item_id, "origin_day": origin,
                                          "training_end_day": train_end,
                                          "series_processing_status": "series_processing_exception"},
                            "diagnostics": [failure], "curves": [], "weekly": [],
                        }
                        _atomic_json(cache_file, cached)
                        print(f"    WARNING {item_id}: {type(exc).__name__}: {exc}")

                all_signatures.append(cached.get("signature", {}))
                all_diagnostics.extend(cached.get("diagnostics", []))
                all_curves.extend(cached.get("curves", []))
                all_weekly.extend(cached.get("weekly", []))
                all_failures.extend([
                    d for d in cached.get("diagnostics", [])
                    if d.get("fit_status") in {"fit_exception", "series_processing_exception"}
                ])
            print(f"    processed/reused {len(store_manifest)} series")

            # Persist after each store-origin group: safe resume if interrupted.
            _write_csv(all_signatures, args.output / "gam_rust_v2_demand_signatures.csv", ["store", "origin_day", "row_index"])
            _write_csv(all_diagnostics, args.output / "gam_rust_v2_component_diagnostics.csv", ["store", "origin_day", "row_index", "component"])
            _write_csv(all_curves, args.output / "gam_rust_v2_smooth_curves.csv", ["store", "origin_day", "row_index", "component", "elapsed_day"])
            _write_csv(all_weekly, args.output / "gam_rust_v2_weekly_profiles.csv", ["store", "origin_day", "row_index", "weekday"])
            _write_csv(all_failures, args.output / "gam_rust_v2_fit_failures.csv", ["store", "origin_day", "row_index", "component"])
            _write_signature_drift(all_signatures, args.output / "gam_rust_v2_signature_drift.csv")
            _summarize_diagnostics(all_diagnostics, args.output / "gam_rust_v2_fit_status_summary.csv")
            _summarize_warnings(all_diagnostics, args.output / "gam_rust_v2_warning_summary.csv")

    curves_df = pd.DataFrame(all_curves)
    _make_plots(curves_df, manifest, args.output, args.plot_examples_per_store)
    print("\nGAM-Rust v2 signature extraction complete.")
    for name in (
        "gam_rust_v2_demand_signatures.csv", "gam_rust_v2_component_diagnostics.csv",
        "gam_rust_v2_smooth_curves.csv", "gam_rust_v2_weekly_profiles.csv",
        "gam_rust_v2_signature_drift.csv", "gam_rust_v2_fit_status_summary.csv",
        "gam_rust_v2_warning_summary.csv", "gam_rust_v2_fit_failures.csv",
    ):
        print(f"  {args.output / name}")
    print(f"  {args.output / 'plots'}")


if __name__ == "__main__":
    main()
