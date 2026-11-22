#!/usr/bin/env python3
"""Audit M5 item-store price coverage over a fixed historical window.

The script reads the Parquet files produced by the project's existing
``data_prep_with_subset.py`` step:

    data/processed/sales_wide.parquet
    data/processed/calendar.parquet
    data/processed/sell_prices.parquet

It does NOT classify series as suitable/unsuitable for clustering. It records
coverage and activity diagnostics so that eligibility rules can be chosen only
after inspecting the actual distributions.

Example (run from the M5 project root):

    pixi run python scripts/audit_m5_price_coverage.py

Default window: d_660 through d_1387 inclusive (728 days).

Interpretation caveat:
    A day is called "price-live" when a sell_prices row exists for the
    item-store-week mapped through calendar.csv. This is a price-record proxy,
    not proof of in-stock availability. M5 does not provide daily inventory
    availability in these files.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _day_columns(columns: list[str]) -> list[str]:
    return sorted(
        (name for name in columns if name.startswith("d_") and name[2:].isdigit()),
        key=lambda name: int(name[2:]),
    )


def _date_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _nullable_int(values: np.ndarray, valid: np.ndarray) -> list[int | None]:
    return [int(value) if ok else None for value, ok in zip(values, valid, strict=True)]


def _max_run_metrics(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return max run, max internal run, and number of internal runs by row."""
    n_rows, _ = mask.shape
    max_all = np.zeros(n_rows, dtype=np.int16)
    max_internal = np.zeros(n_rows, dtype=np.int16)
    n_internal = np.zeros(n_rows, dtype=np.int16)

    for i, row in enumerate(mask):
        # Padding with False ensures runs touching either edge are retained.
        changes = np.diff(np.r_[False, row, False].astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        if starts.size == 0:
            continue
        lengths = ends - starts
        max_all[i] = int(lengths.max())
        internal = lengths[(starts > 0) & (ends < row.size)]
        if internal.size:
            max_internal[i] = int(internal.max())
            n_internal[i] = int(internal.size)

    return max_all, max_internal, n_internal


def _summarize_groups(frame: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    return (
        frame.group_by(group_cols)
        .agg(
            pl.len().alias("n_series"),
            pl.col("price_live_fraction").mean().alias("mean_price_live_fraction"),
            pl.col("price_live_fraction").quantile(0.10).alias("p10_price_live_fraction"),
            pl.col("price_live_fraction").quantile(0.25).alias("p25_price_live_fraction"),
            pl.col("price_live_fraction").median().alias("median_price_live_fraction"),
            pl.col("price_live_fraction").quantile(0.75).alias("p75_price_live_fraction"),
            pl.col("price_live_fraction").quantile(0.90).alias("p90_price_live_fraction"),
            (pl.col("price_live_fraction") == 1.0).sum().alias("n_full_price_coverage"),
            (pl.col("price_live_fraction") >= 0.95).sum().alias("n_at_least_95pct_coverage"),
            (pl.col("price_live_fraction") >= 0.90).sum().alias("n_at_least_90pct_coverage"),
            (pl.col("price_live_fraction") >= 0.80).sum().alias("n_at_least_80pct_coverage"),
            (pl.col("n_price_live_days") == 0).sum().alias("n_no_price_record_in_window"),
            pl.col("n_positive_sales_days").sum().alias("total_positive_sales_days"),
            pl.col("n_positive_sales_days_with_price").sum().alias(
                "positive_sales_days_with_price"
            ),
            pl.col("n_positive_sales_days_without_price").sum().alias(
                "positive_sales_days_without_price"
            ),
            pl.col("positive_units_without_price").sum().alias("positive_units_without_price"),
            pl.col("has_positive_sales_without_price").sum().alias(
                "n_series_with_positive_sales_without_price"
            ),
            pl.col("n_internal_no_price_gaps").sum().alias("total_internal_no_price_gaps"),
        )
        .sort(group_cols)
    )


def run_audit(
    processed_dir: Path,
    output_dir: Path,
    cutoff_d: int = 1387,
    window_days: int = 728,
) -> dict[str, Any]:
    if cutoff_d < 1 or window_days < 1:
        raise ValueError("cutoff_d and window_days must both be positive integers")

    sales_path = processed_dir / "sales_wide.parquet"
    calendar_path = processed_dir / "calendar.parquet"
    prices_path = processed_dir / "sell_prices.parquet"
    for path in (sales_path, calendar_path, prices_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run the existing M5 preprocessing step first (the one that creates "
                "sales_wide.parquet, calendar.parquet and sell_prices.parquet)."
            )

    start_d = cutoff_d - window_days + 1
    if start_d < 1:
        raise ValueError(
            f"Window d_{start_d}..d_{cutoff_d} starts before d_1; "
            "reduce --window-days or increase --cutoff-d."
        )

    print(f"Reading M5 processed data from: {processed_dir}")
    sales_schema = pl.scan_parquet(sales_path).collect_schema()
    sales_columns = sales_schema.names()
    day_cols_all = _day_columns(sales_columns)
    if not day_cols_all:
        raise ValueError(f"No d_# columns found in {sales_path}")
    max_sales_d = max(int(name[2:]) for name in day_cols_all)
    if cutoff_d > max_sales_d:
        raise ValueError(
            f"cutoff d_{cutoff_d} exceeds the last sales day d_{max_sales_d} "
            f"available in {sales_path}"
        )
    missing_id_cols = [name for name in ID_COLS if name not in sales_columns]
    if missing_id_cols:
        raise ValueError(f"Missing identity columns in sales_wide.parquet: {missing_id_cols}")

    window_day_cols = [f"d_{d}" for d in range(start_d, cutoff_d + 1)]
    sales = pl.read_parquet(sales_path, columns=ID_COLS + window_day_cols)
    n_series = sales.height
    if n_series == 0:
        raise ValueError("sales_wide.parquet contains no item-store series")

    duplicate_series = (
        sales.group_by(["store_id", "item_id"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
    )
    if duplicate_series.height:
        raise ValueError(
            "Duplicate store_id/item_id keys found in sales_wide.parquet; "
            "cannot create an unambiguous row mapping."
        )

    calendar = pl.read_parquet(
        calendar_path,
        columns=["d_num", "date", "wm_yr_wk"],
    ).filter((pl.col("d_num") >= start_d) & (pl.col("d_num") <= cutoff_d))
    expected_days = list(range(start_d, cutoff_d + 1))
    actual_days = calendar["d_num"].to_list()
    if actual_days != expected_days:
        raise ValueError(
            "calendar.parquet does not contain exactly one ordered row for every "
            f"day in d_{start_d}..d_{cutoff_d}."
        )
    if calendar["wm_yr_wk"].null_count():
        raise ValueError("Null wm_yr_wk values found in the audit calendar window")

    calendar_dates = calendar["date"].to_list()
    window_start_date = _date_text(calendar_dates[0])
    window_end_date = _date_text(calendar_dates[-1])
    week_to_offsets: dict[int, np.ndarray] = {}
    for week, days in calendar.group_by("wm_yr_wk").agg(pl.col("d_num").alias("days")).iter_rows():
        week_to_offsets[int(week)] = np.asarray(days, dtype=np.int32) - start_d

    window_weeks = list(week_to_offsets)
    prices_window_raw = (
        pl.scan_parquet(prices_path)
        .filter(pl.col("wm_yr_wk").is_in(window_weeks))
        .select("store_id", "item_id", "wm_yr_wk", "sell_price")
        .collect()
    )
    duplicate_prices = (
        prices_window_raw.group_by(["store_id", "item_id", "wm_yr_wk"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
    )
    if duplicate_prices.height:
        examples = duplicate_prices.head(5).to_dicts()
        raise ValueError(
            "Duplicate store_id/item_id/wm_yr_wk rows found in sell_prices.parquet. "
            f"Examples: {examples}"
        )

    # Keep the same convention as the existing subset-preparation step:
    # a finite sell_price value is required for a day to count as price-live.
    invalid_price_rows = prices_window_raw.filter(
        pl.col("sell_price").is_null() | ~pl.col("sell_price").is_finite()
    ).height
    prices_window = prices_window_raw.filter(
        pl.col("sell_price").is_not_null() & pl.col("sell_price").is_finite()
    )

    # Build a compact row-indexed price-presence matrix without creating a
    # 30,490 x 728 long-form table. The matrix is ~22 MB at the full M5 scale.
    series_rows = sales.select("item_id", "store_id").with_row_index("row")
    price_keys = series_rows.join(
        prices_window.select("store_id", "item_id", "wm_yr_wk"),
        on=["store_id", "item_id"],
        how="inner",
    ).select("row", "wm_yr_wk")

    live = np.zeros((n_series, window_days), dtype=np.bool_)
    n_price_live_weeks = np.zeros(n_series, dtype=np.int16)
    grouped_price_keys = price_keys.group_by("wm_yr_wk").agg(pl.col("row").alias("rows"))
    for week, rows in grouped_price_keys.iter_rows():
        row_idx = np.asarray(rows, dtype=np.int32)
        offsets = week_to_offsets.get(int(week))
        if offsets is None or row_idx.size == 0:
            continue
        live[row_idx[:, None], offsets[None, :]] = True
        # A store/item has at most one price row per week (validated above).
        n_price_live_weeks[row_idx] += 1

    sales_arr = sales.select(window_day_cols).to_numpy()
    if not np.issubdtype(sales_arr.dtype, np.number):
        raise TypeError(f"Sales columns are not numeric: dtype={sales_arr.dtype}")
    if np.any(sales_arr < 0):
        raise ValueError("Negative sales values found in the requested window")

    positive = sales_arr > 0
    zero = sales_arr == 0
    n_live_days = live.sum(axis=1).astype(np.int32)
    n_unpriced_days = window_days - n_live_days
    n_positive_all = positive.sum(axis=1).astype(np.int32)
    n_positive_live = (positive & live).sum(axis=1).astype(np.int32)
    n_positive_unpriced = (positive & ~live).sum(axis=1).astype(np.int32)
    n_zero_live = (zero & live).sum(axis=1).astype(np.int32)
    positive_units_live = np.where(positive & live, sales_arr, 0).sum(axis=1, dtype=np.int64)
    positive_units_unpriced = np.where(positive & ~live, sales_arr, 0).sum(axis=1, dtype=np.int64)
    total_units = sales_arr.sum(axis=1, dtype=np.int64)
    live_fraction = n_live_days.astype(np.float64) / float(window_days)

    has_live = n_live_days > 0
    has_positive = n_positive_all > 0
    has_positive_live = n_positive_live > 0
    first_live_idx = live.argmax(axis=1)
    last_live_idx = window_days - 1 - live[:, ::-1].argmax(axis=1)
    first_sale_mask = positive
    first_sale_idx = first_sale_mask.argmax(axis=1)
    last_sale_idx = window_days - 1 - positive[:, ::-1].argmax(axis=1)
    live_positive = positive & live
    first_live_sale_idx = live_positive.argmax(axis=1)
    last_live_sale_idx = window_days - 1 - live_positive[:, ::-1].argmax(axis=1)

    first_live_d = start_d + first_live_idx
    last_live_d = start_d + last_live_idx
    first_sale_d = start_d + first_sale_idx
    last_sale_d = start_d + last_sale_idx
    first_live_sale_d = start_d + first_live_sale_idx
    last_live_sale_d = start_d + last_live_sale_idx

    no_price_mask = ~live
    max_no_price_run, max_internal_no_price_gap, n_internal_no_price_gaps = _max_run_metrics(
        no_price_mask
    )
    padded_live = np.pad(live.astype(np.int8), ((0, 0), (1, 0)), constant_values=0)
    n_live_segments = (np.diff(padded_live, axis=1) == 1).sum(axis=1).astype(np.int16)

    series_cov = sales.select(ID_COLS).with_columns(
        pl.lit(start_d).alias("window_start_d"),
        pl.lit(cutoff_d).alias("window_end_d"),
        pl.lit(window_days).alias("window_days"),
        pl.Series("n_price_live_days", n_live_days),
        pl.Series("n_days_without_price_record", n_unpriced_days),
        pl.Series("price_live_fraction", live_fraction),
        pl.Series("n_price_live_weeks", n_price_live_weeks),
        pl.Series("n_live_segments", n_live_segments),
        pl.Series("n_positive_sales_days", n_positive_all),
        pl.Series("n_positive_sales_days_with_price", n_positive_live),
        pl.Series("n_positive_sales_days_without_price", n_positive_unpriced),
        pl.Series("positive_units_total", total_units),
        pl.Series("positive_units_with_price", positive_units_live),
        pl.Series("positive_units_without_price", positive_units_unpriced),
        pl.Series("n_zero_sales_days_with_price", n_zero_live),
        pl.Series("positive_day_frequency_on_price_days", np.divide(
            n_positive_live, n_live_days, out=np.full(n_series, np.nan), where=has_live
        )),
        pl.Series("zero_day_share_on_price_days", np.divide(
            n_zero_live, n_live_days, out=np.full(n_series, np.nan), where=has_live
        )),
        pl.Series("has_positive_sales_without_price", n_positive_unpriced > 0),
        pl.Series("coverage_pattern", np.where(
            n_live_days == 0,
            "NO_PRICE_RECORD_IN_WINDOW",
            np.where(n_live_days == window_days, "FULL_PRICE_RECORD_COVERAGE", "PARTIAL_PRICE_RECORD_COVERAGE"),
        )),
        pl.Series("first_price_live_d", _nullable_int(first_live_d, has_live)),
        pl.Series("last_price_live_d", _nullable_int(last_live_d, has_live)),
        pl.Series("days_since_last_price_live", _nullable_int(cutoff_d - last_live_d, has_live)),
        pl.Series("first_positive_sale_d", _nullable_int(first_sale_d, has_positive)),
        pl.Series("last_positive_sale_d", _nullable_int(last_sale_d, has_positive)),
        pl.Series("days_since_last_positive_sale", _nullable_int(cutoff_d - last_sale_d, has_positive)),
        pl.Series("first_positive_sale_on_price_day_d", _nullable_int(first_live_sale_d, has_positive_live)),
        pl.Series("last_positive_sale_on_price_day_d", _nullable_int(last_live_sale_d, has_positive_live)),
        pl.Series("days_since_last_positive_sale_on_price_day", _nullable_int(cutoff_d - last_live_sale_d, has_positive_live)),
        pl.Series("max_consecutive_days_without_price", max_no_price_run),
        pl.Series("max_internal_no_price_gap_days", max_internal_no_price_gap),
        pl.Series("n_internal_no_price_gaps", n_internal_no_price_gaps),
    )

    # Convert NaN metrics for series with no price-live days to CSV nulls.
    series_cov = series_cov.with_columns(
        pl.col("positive_day_frequency_on_price_days").fill_nan(None),
        pl.col("zero_day_share_on_price_days").fill_nan(None),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    series_path = output_dir / "series_coverage.csv"
    store_path = output_dir / "coverage_by_store.csv"
    store_cat_path = output_dir / "coverage_by_store_category.csv"
    summary_path = output_dir / "coverage_audit_summary.json"
    manifest_path = output_dir / "coverage_audit_manifest.json"

    series_cov.write_csv(series_path)
    _summarize_groups(series_cov, ["store_id"]).write_csv(store_path)
    _summarize_groups(series_cov, ["store_id", "cat_id"]).write_csv(store_cat_path)

    coverage_values = live_fraction
    overall_summary: dict[str, Any] = {
        "window": {
            "start_d": start_d,
            "end_d": cutoff_d,
            "days": window_days,
            "start_date": window_start_date,
            "end_date": window_end_date,
        },
        "n_series": n_series,
        "n_stores": int(sales["store_id"].n_unique()),
        "n_categories": int(sales["cat_id"].n_unique()),
        "price_live_fraction": {
            "mean": float(np.mean(coverage_values)),
            "p10": float(np.quantile(coverage_values, 0.10)),
            "p25": float(np.quantile(coverage_values, 0.25)),
            "median": float(np.quantile(coverage_values, 0.50)),
            "p75": float(np.quantile(coverage_values, 0.75)),
            "p90": float(np.quantile(coverage_values, 0.90)),
            "p95": float(np.quantile(coverage_values, 0.95)),
            "min": float(np.min(coverage_values)),
            "max": float(np.max(coverage_values)),
        },
        "coverage_counts": {
            "no_price_record": int(np.sum(n_live_days == 0)),
            "full_price_record_coverage": int(np.sum(n_live_days == window_days)),
            "at_least_95pct": int(np.sum(coverage_values >= 0.95)),
            "at_least_90pct": int(np.sum(coverage_values >= 0.90)),
            "at_least_80pct": int(np.sum(coverage_values >= 0.80)),
            "below_50pct": int(np.sum(coverage_values < 0.50)),
        },
        "sales_activity": {
            "series_with_no_positive_sales_in_window": int(np.sum(n_positive_all == 0)),
            "total_positive_sales_days": int(np.sum(n_positive_all)),
            "positive_sales_days_with_price": int(np.sum(n_positive_live)),
            "positive_sales_days_without_price": int(np.sum(n_positive_unpriced)),
            "series_with_positive_sales_without_price": int(np.sum(n_positive_unpriced > 0)),
            "positive_units_total": int(np.sum(total_units)),
            "positive_units_with_price": int(np.sum(positive_units_live)),
            "positive_units_without_price": int(np.sum(positive_units_unpriced)),
        },
        "price_presence_gaps": {
            "series_with_internal_no_price_gap": int(np.sum(n_internal_no_price_gaps > 0)),
            "total_internal_no_price_gaps": int(np.sum(n_internal_no_price_gaps)),
            "max_internal_no_price_gap_days": int(np.max(max_internal_no_price_gap)),
            "series_with_multiple_live_segments": int(np.sum(n_live_segments > 1)),
        },
        "interpretation": (
            "Price-live is a proxy based on a matching sell_prices item-store-week row; "
            "it is not a daily inventory or in-stock indicator. No clustering eligibility "
            "thresholds are applied by this audit."
        ),
    }

    manifest = {
        "script": Path(__file__).name,
        "processed_dir": str(processed_dir.resolve()),
        "inputs": {
            "sales_wide": str(sales_path.resolve()),
            "calendar": str(calendar_path.resolve()),
            "sell_prices": str(prices_path.resolve()),
        },
        "sales_rows": n_series,
        "sales_available_through_d": max_sales_d,
        "price_rows_in_relevant_weeks": prices_window_raw.height,
        "invalid_price_value_rows_in_relevant_weeks": invalid_price_rows,
        "valid_price_rows_in_relevant_weeks": prices_window.height,
        "matched_item_store_week_price_rows": price_keys.height,
        "window": overall_summary["window"],
        "outputs": {
            "series_coverage": series_path.name,
            "coverage_by_store": store_path.name,
            "coverage_by_store_category": store_cat_path.name,
            "summary": summary_path.name,
        },
        "notes": [
            "The audit uses the sales_wide/calendar/sell_prices Parquet files from the existing M5 preprocessing step.",
            "A price-live day is determined by a sell_prices item-store-week row mapped through calendar.wm_yr_wk.",
            "Positive sales on days without a matching price row are flagged rather than silently discarded.",
            "No series is marked eligible/ineligible; thresholds are left for review after seeing the distributions.",
        ],
    }

    summary_path.write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nM5 price-coverage audit complete")
    print(f"  Window: d_{start_d}..d_{cutoff_d} ({window_days} days)")
    print(f"  Dates:  {window_start_date} .. {window_end_date}")
    print(f"  Series: {n_series:,}; stores: {overall_summary['n_stores']}")
    print(
        "  Price-live coverage: "
        f"median={overall_summary['price_live_fraction']['median']:.1%}, "
        f"p10={overall_summary['price_live_fraction']['p10']:.1%}, "
        f"full={overall_summary['coverage_counts']['full_price_record_coverage']:,}, "
        f">=95%={overall_summary['coverage_counts']['at_least_95pct']:,}"
    )
    print(
        "  Positive-sales days without a price row: "
        f"{overall_summary['sales_activity']['positive_sales_days_without_price']:,} "
        "across "
        f"{overall_summary['sales_activity']['series_with_positive_sales_without_price']:,} series"
    )
    print(f"  Outputs: {output_dir.resolve()}")
    print("\nNo clustering eligibility threshold has been applied.")
    return overall_summary


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
        help="Processed Parquet directory (relative to project root unless absolute)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/coverage_audit/d1387_728d"),
        help="Audit output directory (relative to project root unless absolute)",
    )
    parser.add_argument("--cutoff-d", type=int, default=1387, help="Inclusive cutoff day (default: 1387)")
    parser.add_argument("--window-days", type=int, default=728, help="Window length including cutoff day (default: 728)")
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
        run_audit(
            processed_dir=processed_dir.resolve(),
            output_dir=output_dir.resolve(),
            cutoff_d=args.cutoff_d,
            window_days=args.window_days,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
