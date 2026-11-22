#!/usr/bin/env python3
"""Prepare the all-series, mask-aware M5 clustering input for d_660..d_1387.

Run from the M5 project root:

    pixi run python scripts/prepare_m5_clustering_data.py

Inputs (created by the existing M5 preprocessing workflow):
    data/processed/sales_wide.parquet
    data/processed/calendar.parquet
    data/processed/sell_prices.parquet

Outputs (default: data/processed/clustering/d1387_728d/):
    sales_units.npy          int16 [n_series, 728], original unit sales
    sales_log1p.npy          float32 [n_series, 728], NaN where unobserved
    observation_mask.npy     bool [n_series, 728], true on price-recorded days
    series_metadata.parquet  one row per series, including row_index and labels
    calendar_window.parquet  the 728 calendar rows, in d_660..d_1387 order
    manifest.json            shapes, dtypes, counts, provenance and semantics

Important semantics:
- The observation mask is a price-record proxy: a day is observed when a valid
  sell_prices item-store-week record exists after mapping through calendar.
  This is not a daily in-stock indicator.
- A raw zero on a masked day is NOT interpreted as an observed zero-demand day.
- sales_log1p.npy stores NaN outside the observation mask. Do not pass it directly
  to ordinary tslearn estimators; the clustering stage must use mask-aware
  distances/representations or an explicitly justified adapter.
- No series is removed. Arrays preserve the row order of sales_wide.parquet;
  series_metadata.parquet.row_index maps metadata rows to array rows.
- No per-series scaling, normalization, demand segmentation or clustering is
  performed here. Those choices remain part of the clustering experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

WINDOW_START_D = 660
WINDOW_END_D = 1387
WINDOW_DAYS = WINDOW_END_D - WINDOW_START_D + 1
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _day_columns(columns: list[str]) -> list[str]:
    return sorted(
        (name for name in columns if name.startswith("d_") and name[2:].isdigit()),
        key=lambda name: int(name[2:]),
    )


def _date_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _nullable_day(values: np.ndarray, valid: np.ndarray, start_d: int) -> list[int | None]:
    return [int(start_d + value) if ok else None for value, ok in zip(values, valid, strict=True)]


def _run_metrics(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return maximum run length, maximum internal run, and internal run count."""
    n_rows, _ = mask.shape
    max_all = np.zeros(n_rows, dtype=np.int16)
    max_internal = np.zeros(n_rows, dtype=np.int16)
    n_internal = np.zeros(n_rows, dtype=np.int16)

    for row_index, row in enumerate(mask):
        changes = np.diff(np.r_[False, row, False].astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        if starts.size == 0:
            continue
        lengths = ends - starts
        max_all[row_index] = int(lengths.max())
        internal = lengths[(starts > 0) & (ends < row.size)]
        if internal.size:
            max_internal[row_index] = int(internal.max())
            n_internal[row_index] = int(internal.size)

    return max_all, max_internal, n_internal


def _build_observation_mask(
    sales_ids: pl.DataFrame,
    calendar_window: pl.DataFrame,
    prices_path: Path,
    n_days: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Broadcast weekly price presence to daily row-aligned observation mask."""
    required_calendar = {"d_num", "wm_yr_wk"}
    missing_calendar = sorted(required_calendar - set(calendar_window.columns))
    if missing_calendar:
        raise ValueError(f"calendar.parquet is missing columns: {missing_calendar}")

    if calendar_window["wm_yr_wk"].null_count():
        raise ValueError("Null wm_yr_wk values found in the requested calendar window")

    week_to_offsets: dict[int, np.ndarray] = {}
    for week, day_values in calendar_window.group_by("wm_yr_wk").agg(
        pl.col("d_num").alias("days")
    ).iter_rows():
        offsets = np.asarray(day_values, dtype=np.int32) - WINDOW_START_D
        week_to_offsets[int(week)] = offsets

    window_weeks = list(week_to_offsets)
    prices_raw = (
        pl.scan_parquet(prices_path)
        .filter(pl.col("wm_yr_wk").cast(pl.Int32).is_in(window_weeks))
        .select(
            pl.col("store_id"),
            pl.col("item_id"),
            pl.col("wm_yr_wk").cast(pl.Int32),
            pl.col("sell_price").cast(pl.Float32),
        )
        .collect()
    )

    duplicate_prices = (
        prices_raw.group_by("store_id", "item_id", "wm_yr_wk")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
    )
    if duplicate_prices.height:
        raise ValueError(
            "Duplicate store_id/item_id/wm_yr_wk records in sell_prices.parquet. "
            f"Examples: {duplicate_prices.head(5).to_dicts()}"
        )

    invalid_price_rows = prices_raw.filter(
        pl.col("sell_price").is_null() | ~pl.col("sell_price").is_finite()
    ).height
    if invalid_price_rows:
        raise ValueError(
            f"Found {invalid_price_rows:,} null/non-finite sell_price rows in the window. "
            "Resolve these before defining the observation mask."
        )

    sales_rows = sales_ids.select("item_id", "store_id").with_row_index("row_index")
    price_keys = sales_rows.join(
        prices_raw.select("store_id", "item_id", "wm_yr_wk"),
        on=["store_id", "item_id"],
        how="inner",
    ).select("row_index", "wm_yr_wk")

    n_series = sales_ids.height
    observation_mask = np.zeros((n_series, n_days), dtype=np.bool_)
    n_price_live_weeks = np.zeros(n_series, dtype=np.int16)

    grouped = price_keys.group_by("wm_yr_wk").agg(pl.col("row_index").alias("rows"))
    for week, rows in grouped.iter_rows():
        row_indices = np.asarray(rows, dtype=np.int32)
        offsets = week_to_offsets.get(int(week))
        if offsets is None or row_indices.size == 0:
            continue
        observation_mask[row_indices[:, None], offsets[None, :]] = True
        n_price_live_weeks[row_indices] += 1

    diagnostics = {
        "price_rows_in_window_weeks": int(prices_raw.height),
        "matched_item_store_week_price_rows": int(price_keys.height),
        "invalid_price_value_rows": int(invalid_price_rows),
    }
    return observation_mask, n_price_live_weeks, diagnostics


def _build_metadata(
    sales_ids: pl.DataFrame,
    sales_units: np.ndarray,
    observation_mask: np.ndarray,
    n_price_live_weeks: np.ndarray,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    n_series, n_days = sales_units.shape
    positive = sales_units > 0
    positive_without_price = positive & ~observation_mask
    n_positive_without_price = positive_without_price.sum(axis=1).astype(np.int32)
    n_units_without_price = np.where(positive_without_price, sales_units, 0).sum(
        axis=1, dtype=np.int64
    )

    # The audit already found zero such observations. Fail rather than silently
    # losing a positive target while applying the price-derived observation mask.
    if int(n_positive_without_price.sum()) > 0:
        examples = np.argwhere(positive_without_price)[:10]
        raise ValueError(
            "Positive sales were found on days without a matching price record. "
            "The audit reported none, so this indicates an input/version mismatch. "
            f"First [row_index, day_offset] examples: {examples.tolist()}"
        )

    n_live = observation_mask.sum(axis=1).astype(np.int32)
    n_unobserved = n_days - n_live
    has_live = n_live > 0
    has_positive = positive.any(axis=1)
    n_positive = positive.sum(axis=1).astype(np.int32)
    n_positive_live = (positive & observation_mask).sum(axis=1).astype(np.int32)
    n_zero_live = ((sales_units == 0) & observation_mask).sum(axis=1).astype(np.int32)
    units_total = sales_units.sum(axis=1, dtype=np.int64)
    units_live = np.where(observation_mask, sales_units, 0).sum(axis=1, dtype=np.int64)

    first_live_offset = observation_mask.argmax(axis=1).astype(np.int32)
    last_live_offset = (n_days - 1 - observation_mask[:, ::-1].argmax(axis=1)).astype(np.int32)
    first_sale_offset = positive.argmax(axis=1).astype(np.int32)
    last_sale_offset = (n_days - 1 - positive[:, ::-1].argmax(axis=1)).astype(np.int32)

    leading_unobserved = np.where(has_live, first_live_offset, n_days).astype(np.int32)
    trailing_unobserved = np.where(has_live, n_days - 1 - last_live_offset, n_days).astype(np.int32)
    max_unobserved_run, max_internal_unobserved_gap, n_internal_unobserved_gaps = _run_metrics(
        ~observation_mask
    )
    padded_live = np.pad(
        observation_mask.astype(np.int8), ((0, 0), (1, 1)), constant_values=0
    )
    n_live_segments = (np.diff(padded_live, axis=1) == 1).sum(axis=1).astype(np.int16)

    coverage_labels: list[str] = []
    for i in range(n_series):
        if n_live[i] == 0:
            coverage_labels.append("NO_PRICE_RECORD_IN_WINDOW")
        elif n_live[i] == n_days:
            coverage_labels.append("FULL_WINDOW_PRICE_COVERAGE")
        elif (
            leading_unobserved[i] > 0
            and trailing_unobserved[i] == 0
            and n_internal_unobserved_gaps[i] == 0
            and n_live_segments[i] == 1
        ):
            coverage_labels.append("LEFT_TRUNCATED_PRICE_HISTORY")
        else:
            coverage_labels.append("PARTIAL_PRICE_COVERAGE_OTHER")

    activity_labels = [
        "NO_PRICE_RECORD_IN_WINDOW"
        if n_live[i] == 0
        else "PRICE_RECORD_NO_POSITIVE_SALES"
        if n_positive_live[i] == 0
        else "POSITIVE_SALES_OBSERVED"
        for i in range(n_series)
    ]

    positive_frequency = np.divide(
        n_positive_live,
        n_live,
        out=np.full(n_series, np.nan, dtype=np.float64),
        where=has_live,
    )
    zero_share = np.divide(
        n_zero_live,
        n_live,
        out=np.full(n_series, np.nan, dtype=np.float64),
        where=has_live,
    )
    mean_units_live_day = np.divide(
        units_live,
        n_live,
        out=np.full(n_series, np.nan, dtype=np.float64),
        where=has_live,
    )
    mean_units_positive_day = np.divide(
        units_live,
        n_positive_live,
        out=np.full(n_series, np.nan, dtype=np.float64),
        where=n_positive_live > 0,
    )

    metadata = sales_ids.with_row_index("row_index").with_columns(
        pl.lit(WINDOW_START_D, dtype=pl.Int32).alias("window_start_d"),
        pl.lit(WINDOW_END_D, dtype=pl.Int32).alias("window_end_d"),
        pl.lit(WINDOW_DAYS, dtype=pl.Int16).alias("window_days"),
        pl.Series("n_price_live_days", n_live, dtype=pl.Int32),
        pl.Series("n_days_without_price_record", n_unobserved, dtype=pl.Int32),
        pl.Series("price_live_fraction", n_live / float(n_days), dtype=pl.Float32),
        pl.Series("n_price_live_weeks", n_price_live_weeks, dtype=pl.Int16),
        pl.Series("n_live_segments", n_live_segments, dtype=pl.Int16),
        pl.Series("n_unobserved_leading_days", leading_unobserved, dtype=pl.Int32),
        pl.Series("n_unobserved_trailing_days", trailing_unobserved, dtype=pl.Int32),
        pl.Series("max_consecutive_days_without_price", max_unobserved_run, dtype=pl.Int16),
        pl.Series("max_internal_no_price_gap_days", max_internal_unobserved_gap, dtype=pl.Int16),
        pl.Series("n_internal_no_price_gaps", n_internal_unobserved_gaps, dtype=pl.Int16),
        pl.Series("n_positive_sales_days", n_positive, dtype=pl.Int32),
        pl.Series("n_positive_sales_days_with_price", n_positive_live, dtype=pl.Int32),
        pl.Series("n_positive_sales_days_without_price", n_positive_without_price, dtype=pl.Int32),
        pl.Series("n_zero_sales_days_with_price", n_zero_live, dtype=pl.Int32),
        pl.Series("positive_units_total", units_total, dtype=pl.Int64),
        pl.Series("positive_units_with_price", units_live, dtype=pl.Int64),
        pl.Series("positive_units_without_price", n_units_without_price, dtype=pl.Int64),
        pl.Series("positive_day_frequency_on_price_days", positive_frequency, dtype=pl.Float32),
        pl.Series("zero_day_share_on_price_days", zero_share, dtype=pl.Float32),
        pl.Series("mean_units_per_price_live_day", mean_units_live_day, dtype=pl.Float32),
        pl.Series("mean_units_per_positive_sales_day", mean_units_positive_day, dtype=pl.Float32),
        pl.Series("first_price_live_d", _nullable_day(first_live_offset, has_live, WINDOW_START_D), dtype=pl.Int32),
        pl.Series("last_price_live_d", _nullable_day(last_live_offset, has_live, WINDOW_START_D), dtype=pl.Int32),
        pl.Series(
            "days_since_last_price_live",
            [int(WINDOW_END_D - (WINDOW_START_D + x)) if ok else None
             for x, ok in zip(last_live_offset, has_live, strict=True)],
            dtype=pl.Int32,
        ),
        pl.Series("first_positive_sale_d", _nullable_day(first_sale_offset, has_positive, WINDOW_START_D), dtype=pl.Int32),
        pl.Series("last_positive_sale_d", _nullable_day(last_sale_offset, has_positive, WINDOW_START_D), dtype=pl.Int32),
        pl.Series(
            "days_since_last_positive_sale",
            [int(WINDOW_END_D - (WINDOW_START_D + x)) if ok else None
             for x, ok in zip(last_sale_offset, has_positive, strict=True)],
            dtype=pl.Int32,
        ),
        pl.Series("has_positive_sales_without_price", n_positive_without_price > 0, dtype=pl.Boolean),
        pl.Series("is_full_window_price_coverage", n_live == n_days, dtype=pl.Boolean),
        pl.Series("is_left_truncated_price_history", np.asarray(coverage_labels) == "LEFT_TRUNCATED_PRICE_HISTORY", dtype=pl.Boolean),
        pl.Series("has_observed_positive_sales", n_positive_live > 0, dtype=pl.Boolean),
        pl.Series("is_zero_sales_on_all_price_days", has_live & (n_positive_live == 0), dtype=pl.Boolean),
        pl.Series("coverage_label", coverage_labels, dtype=pl.String),
        pl.Series("activity_label", activity_labels, dtype=pl.String),
    )

    # Convert non-applicable ratios to nulls (rather than NaN) in the Parquet metadata.
    metadata = metadata.with_columns(
        pl.col("positive_day_frequency_on_price_days").fill_nan(None),
        pl.col("zero_day_share_on_price_days").fill_nan(None),
        pl.col("mean_units_per_price_live_day").fill_nan(None),
        pl.col("mean_units_per_positive_sales_day").fill_nan(None),
    )

    coverage_counts = {
        label: int(coverage_labels.count(label))
        for label in sorted(set(coverage_labels))
    }
    activity_counts = {
        label: int(activity_labels.count(label))
        for label in sorted(set(activity_labels))
    }
    checks = {
        "n_series": int(n_series),
        "n_window_days": int(n_days),
        "n_positive_sales_days_without_price": int(n_positive_without_price.sum()),
        "n_positive_units_without_price": int(n_units_without_price.sum()),
        "n_series_with_internal_no_price_gaps": int((n_internal_unobserved_gaps > 0).sum()),
        "n_series_with_multiple_price_live_segments": int((n_live_segments > 1).sum()),
        "n_observation_mask_true": int(observation_mask.sum()),
        "n_observation_mask_false": int(observation_mask.size - observation_mask.sum()),
        "coverage_label_counts": coverage_counts,
        "activity_label_counts": activity_counts,
    }
    return metadata, checks


def prepare_dataset(
    processed_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    sales_path = processed_dir / "sales_wide.parquet"
    calendar_path = processed_dir / "calendar.parquet"
    prices_path = processed_dir / "sell_prices.parquet"
    for path in (sales_path, calendar_path, prices_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required input not found: {path}\n"
                "Run the existing M5 preprocessing workflow first."
            )

    print(f"Reading processed M5 data from: {processed_dir}")
    sales_schema = pl.scan_parquet(sales_path).collect_schema()
    sales_columns = sales_schema.names()
    missing_ids = [name for name in ID_COLS if name not in sales_columns]
    if missing_ids:
        raise ValueError(f"sales_wide.parquet is missing identifier columns: {missing_ids}")

    available_days = _day_columns(sales_columns)
    expected_window_days = [f"d_{d}" for d in range(WINDOW_START_D, WINDOW_END_D + 1)]
    missing_window_days = [name for name in expected_window_days if name not in available_days]
    if missing_window_days:
        raise ValueError(
            "sales_wide.parquet does not contain the complete requested window. "
            f"Missing {len(missing_window_days)} columns; first missing: {missing_window_days[:5]}"
        )

    sales = pl.read_parquet(sales_path, columns=ID_COLS + expected_window_days)
    n_series = sales.height
    if n_series != 30_490:
        raise ValueError(
            f"Expected all 30,490 M5 item-store series, found {n_series:,}. "
            "Check that sales_wide.parquet was created from sales_train_evaluation.csv."
        )

    duplicate_series = (
        sales.group_by(["store_id", "item_id"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
    )
    if duplicate_series.height:
        raise ValueError(
            "Duplicate store_id/item_id series found in sales_wide.parquet: "
            f"{duplicate_series.head(5).to_dicts()}"
        )

    calendar_all = pl.read_parquet(calendar_path)
    if not {"d_num", "wm_yr_wk", "date"}.issubset(calendar_all.columns):
        raise ValueError("calendar.parquet must contain d_num, wm_yr_wk and date columns")
    calendar_window = (
        calendar_all.filter(
            (pl.col("d_num") >= WINDOW_START_D) & (pl.col("d_num") <= WINDOW_END_D)
        )
        .sort("d_num")
    )
    expected_calendar_days = list(range(WINDOW_START_D, WINDOW_END_D + 1))
    if calendar_window["d_num"].to_list() != expected_calendar_days:
        raise ValueError(
            "calendar.parquet does not contain exactly one ordered row per day "
            "from d_660 through d_1387."
        )

    sales_units = np.ascontiguousarray(sales.select(expected_window_days).to_numpy())
    if not np.issubdtype(sales_units.dtype, np.integer):
        raise TypeError(f"Expected integer unit sales, received {sales_units.dtype}")
    if np.any(sales_units < 0):
        raise ValueError("Negative unit sales found in the requested M5 window")
    if sales_units.max(initial=0) > np.iinfo(np.int16).max:
        raise ValueError("Sales values exceed int16 capacity; refusing a lossy cast")
    sales_units = np.ascontiguousarray(sales_units, dtype=np.int16)

    sales_ids = sales.select(ID_COLS)
    observation_mask, n_price_live_weeks, price_diagnostics = _build_observation_mask(
        sales_ids=sales_ids,
        calendar_window=calendar_window,
        prices_path=prices_path,
        n_days=WINDOW_DAYS,
    )
    metadata, checks = _build_metadata(
        sales_ids=sales_ids,
        sales_units=sales_units,
        observation_mask=observation_mask,
        n_price_live_weeks=n_price_live_weeks,
    )

    # Preserve original unit sales exactly. For the transformed modelling array,
    # make unobserved days explicit NaNs; the boolean mask remains the authority.
    sales_log1p = np.full(sales_units.shape, np.nan, dtype=np.float32)
    sales_log1p[observation_mask] = np.log1p(
        sales_units[observation_mask].astype(np.float32)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    units_path = output_dir / "sales_units.npy"
    log_path = output_dir / "sales_log1p.npy"
    mask_path = output_dir / "observation_mask.npy"
    metadata_path = output_dir / "series_metadata.parquet"
    calendar_path_out = output_dir / "calendar_window.parquet"
    manifest_path = output_dir / "manifest.json"

    np.save(units_path, sales_units, allow_pickle=False)
    np.save(log_path, sales_log1p, allow_pickle=False)
    np.save(mask_path, observation_mask, allow_pickle=False)
    metadata.write_parquet(metadata_path, compression="zstd")
    calendar_window.write_parquet(calendar_path_out, compression="zstd")

    window_start_date = _date_text(calendar_window["date"][0])
    window_end_date = _date_text(calendar_window["date"][-1])
    manifest: dict[str, Any] = {
        "script": Path(__file__).name,
        "window": {
            "start_d": WINDOW_START_D,
            "end_d": WINDOW_END_D,
            "n_days": WINDOW_DAYS,
            "start_date": window_start_date,
            "end_date": window_end_date,
        },
        "n_series": int(n_series),
        "series_order": "Original row order from sales_wide.parquet; row_index in series_metadata.parquet is the array row.",
        "inputs": {
            "sales_wide": str(sales_path.resolve()),
            "calendar": str((processed_dir / "calendar.parquet").resolve()),
            "sell_prices": str(prices_path.resolve()),
        },
        "outputs": {
            "sales_units": units_path.name,
            "sales_log1p": log_path.name,
            "observation_mask": mask_path.name,
            "series_metadata": metadata_path.name,
            "calendar_window": calendar_path_out.name,
        },
        "array_schema": {
            "shape": [int(n_series), WINDOW_DAYS],
            "sales_units_dtype": str(sales_units.dtype),
            "sales_log1p_dtype": str(sales_log1p.dtype),
            "observation_mask_dtype": str(observation_mask.dtype),
            "sales_log1p_missing_value": "NaN wherever observation_mask is false",
            "sales_units_missing_semantics": "Original source value retained; consumers must consult observation_mask before interpreting a value as an observed target.",
            "observation_mask_true": "A finite sell_prices record exists for this item-store-week, mapped to the day through calendar.wm_yr_wk.",
        },
        "price_observation_caveat": (
            "Price-record presence is an observation proxy, not a daily inventory or in-stock indicator."
        ),
        "preprocessing": {
            "sales_transform": "log1p(units_sold) on observed days only",
            "per_series_normalization": "not applied",
            "imputation": "not applied; unobserved transformed values are NaN",
            "filtering": "none; all 30,490 item-store series are retained",
            "clustering": "not performed",
        },
        "price_diagnostics": price_diagnostics,
        "checks": checks,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nM5 mask-aware clustering data preparation complete")
    print(f"  Window: d_{WINDOW_START_D}..d_{WINDOW_END_D} ({WINDOW_DAYS} days)")
    print(f"  Dates:  {window_start_date} .. {window_end_date}")
    print(f"  Series: {n_series:,} (all retained)")
    print(f"  Arrays: {sales_units.shape}; units={sales_units.dtype}, log1p={sales_log1p.dtype}, mask={observation_mask.dtype}")
    print(f"  Coverage labels: {checks['coverage_label_counts']}")
    print(f"  Activity labels: {checks['activity_label_counts']}")
    print(f"  Positive sales without a price record: {checks['n_positive_sales_days_without_price']:,}")
    print(f"  Output: {output_dir.resolve()}")
    print("  No series filtering, normalization or clustering was performed.")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="M5 project root (default: current working directory)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Processed input directory, relative to project root unless absolute",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/clustering/d1387_728d"),
        help="Output directory, relative to project root unless absolute",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    processed_dir = args.processed_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    try:
        prepare_dataset(processed_dir.resolve(), output_dir.resolve())
    except Exception as exc:  # surface a concise message in Pixi/terminal runs
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
