"""Validate the raw M5 Forecasting Accuracy dataset without reshaping sales."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

METADATA_COLUMNS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]

HIERARCHY_LEVELS = {
    "Total": [],
    "State": ["state_id"],
    "Store": ["store_id"],
    "Category": ["cat_id"],
    "Department": ["dept_id"],
    "State × Category": ["state_id", "cat_id"],
    "State × Department": ["state_id", "dept_id"],
    "Store × Category": ["store_id", "cat_id"],
    "Store × Department": ["store_id", "dept_id"],
    "Item": ["item_id"],
    "State × Item": ["state_id", "item_id"],
    "Item × Store": ["item_id", "store_id"],
}
EXPECTED_HIERARCHY_COUNTS = {
    "Total": 1,
    "State": 3,
    "Store": 10,
    "Category": 3,
    "Department": 7,
    "State × Category": 9,
    "State × Department": 21,
    "Store × Category": 30,
    "Store × Department": 70,
    "Item": 3_049,
    "State × Item": 9_147,
    "Item × Store": 30_490,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_sorted_header(path: Path) -> list[str]:
    """Reads CSV columns and ensures day columns are sorted numerically (d_2 before d_10)."""
    raw_cols = pd.read_csv(path, nrows=0).columns.tolist()
    meta = [col for col in raw_cols if col in METADATA_COLUMNS]
    days = [col for col in raw_cols if col.startswith("d_")]
    # Sort days numerically based on the integer after 'd_'
    sorted_days = sorted(days, key=lambda x: int(x.split("_")[1]))
    return meta + sorted_days


def validate_sales_headers(
    eval_path: Path, valid_path: Path
) -> tuple[list[str], list[str]]:
    eval_columns = read_sorted_header(eval_path)
    valid_columns = read_sorted_header(valid_path)

    eval_days = [f"d_{day}" for day in range(1, 1_942)]
    valid_days = [f"d_{day}" for day in range(1, 1_914)]

    require(
        eval_columns == METADATA_COLUMNS + eval_days,
        "sales_train_evaluation.csv does not have the expected metadata/day columns.",
    )
    require(
        valid_columns == METADATA_COLUMNS + valid_days,
        "sales_train_validation.csv does not have the expected metadata/day columns.",
    )

    print("\n[1/5] Sales file schemas")
    print(
        f"  sales_train_evaluation.csv: {len(eval_columns):,} columns (6 metadata + 1,941 days)"
    )
    print(
        f"  sales_train_validation.csv: {len(valid_columns):,} columns (6 metadata + 1,913 days)"
    )
    return eval_days, valid_days


def validate_series_metadata(eval_path: Path, valid_path: Path) -> pd.DataFrame:
    string_types = {column: "string" for column in METADATA_COLUMNS}
    eval_meta = pd.read_csv(eval_path, usecols=METADATA_COLUMNS, dtype=string_types)
    valid_meta = pd.read_csv(valid_path, usecols=METADATA_COLUMNS, dtype=string_types)
    eval_meta = eval_meta[METADATA_COLUMNS]
    valid_meta = valid_meta[METADATA_COLUMNS]

    require(
        len(eval_meta) == 30_490, f"Expected 30,490 series; found {len(eval_meta):,}."
    )
    require(
        len(valid_meta) == 30_490,
        f"Expected 30,490 validation series; found {len(valid_meta):,}.",
    )
    require(
        eval_meta["id"].is_unique, "Duplicate series IDs in sales_train_evaluation.csv."
    )
    require(
        valid_meta["id"].is_unique,
        "Duplicate series IDs in sales_train_validation.csv.",
    )

    # Strip suffixes (_evaluation vs _validation) to compare base IDs and row ordering correctly
    eval_base_ids = eval_meta["id"].str.replace("_evaluation", "", regex=False)
    valid_base_ids = valid_meta["id"].str.replace("_validation", "", regex=False)

    require(
        eval_base_ids.tolist() == valid_base_ids.tolist(),
        "Series IDs or row ordering differ between the evaluation and validation files.",
    )

    # Compare metadata columns (excluding the suffix-differing id column)
    require(
        eval_meta.drop(columns=["id"]).equals(valid_meta.drop(columns=["id"])),
        "Series metadata differs between the two sales files.",
    )
    require(
        (eval_base_ids == eval_meta["item_id"] + "_" + eval_meta["store_id"]).all(),
        "Some series IDs do not match the item_id + store_id pattern.",
    )

    expected_unique = {
        "state_id": 3,
        "store_id": 10,
        "cat_id": 3,
        "dept_id": 7,
        "item_id": 3_049,
    }
    for column, expected in expected_unique.items():
        actual = eval_meta[column].nunique()
        require(
            actual == expected,
            f"Expected {expected:,} unique {column}; found {actual:,}.",
        )

    # Each item has a stable category/department, and each store belongs to one state.
    item_mapping_counts = eval_meta.groupby("item_id", observed=True)[
        ["cat_id", "dept_id"]
    ].nunique()
    require(
        (item_mapping_counts == 1).all().all(),
        "An item maps to multiple categories or departments.",
    )
    store_mapping_counts = eval_meta.groupby("store_id", observed=True)[
        "state_id"
    ].nunique()
    require((store_mapping_counts == 1).all(), "A store maps to multiple states.")

    print("\n[2/5] Bottom-level series and hierarchy")
    print(f"  Bottom-level item-store series: {len(eval_meta):,}")
    for column in ["state_id", "store_id", "cat_id", "dept_id", "item_id"]:
        print(f"  Unique {column}: {eval_meta[column].nunique():,}")

    hierarchy_total = 0
    for level, keys in HIERARCHY_LEVELS.items():
        count = 1 if not keys else len(eval_meta[keys].drop_duplicates())
        expected = EXPECTED_HIERARCHY_COUNTS[level]
        require(
            count == expected,
            f"Hierarchy level {level}: expected {expected:,}, found {count:,}.",
        )
        hierarchy_total += count
        print(f"  {level:<22} {count:>7,}")
    require(
        hierarchy_total == 42_840,
        f"Expected 42,840 hierarchy nodes; found {hierarchy_total:,}.",
    )
    print(f"  {'All 12 levels':<22} {hierarchy_total:>7,}")

    return eval_meta


def validate_calendar(calendar_path: Path, eval_days: list[str]) -> pd.DataFrame:
    calendar = pd.read_csv(calendar_path, dtype={"d": "string", "wm_yr_wk": "int32"})
    require(
        len(calendar) == 1_969,
        f"Expected 1,969 calendar rows; found {len(calendar):,}.",
    )
    require(calendar["d"].is_unique, "Calendar contains duplicate d labels.")
    require(calendar["date"].is_unique, "Calendar contains duplicate dates.")

    expected_d = [f"d_{day}" for day in range(1, 1_970)]
    require(
        calendar["d"].tolist() == expected_d,
        "Calendar d labels are missing, duplicated, or out of order.",
    )
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    day_gaps = calendar["date"].diff().dropna().dt.days
    require(
        (day_gaps == 1).all(), "Calendar dates are not consecutive daily observations."
    )
    require(
        set(eval_days).issubset(set(calendar["d"])),
        "Calendar does not cover every evaluation sales day.",
    )

    date_by_d = calendar.set_index("d")["date"]
    important_days = ["d_1", "d_1913", "d_1914", "d_1941", "d_1942", "d_1969"]

    print("\n[3/5] Calendar and competition split")
    print(f"  Calendar rows: {len(calendar):,}")
    print(
        f"  Calendar date range: {calendar['date'].min().date()} to {calendar['date'].max().date()}"
    )
    for day in important_days:
        print(f"  {day:<7} {date_by_d.loc[day].date()}")

    validation_days = calendar.iloc[1_913:1_941].set_index("d")["date"]
    require(
        len(validation_days) == 28,
        f"Expected a 28-day validation window; found {len(validation_days)} days.",
    )
    print(
        f"  Validation window: {validation_days.iloc[0].date()} to {validation_days.iloc[-1].date()} (28 days)"
    )
    print(
        "  Competition test dates d_1942..d_1969 have calendar rows but no actual sales targets in these files."
    )
    return calendar


def validate_prices(
    prices_path: Path, calendar: pd.DataFrame, meta: pd.DataFrame
) -> None:
    prices = pd.read_csv(
        prices_path,
        dtype={
            "store_id": "string",
            "item_id": "string",
            "wm_yr_wk": "int32",
            "sell_price": "float32",
        },
    )
    expected_columns = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    require(
        prices.columns.tolist() == expected_columns,
        "sell_prices.csv has unexpected columns or column order.",
    )
    require(
        not prices[expected_columns].isna().any().any(),
        "sell_prices.csv contains missing values.",
    )
    require(
        (prices["sell_price"] > 0).all(),
        "sell_prices.csv contains non-positive prices.",
    )
    require(
        not prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any(),
        "Duplicate store_id/item_id/wm_yr_wk price keys found.",
    )

    calendar_weeks = set(calendar["wm_yr_wk"].unique())
    price_weeks = set(prices["wm_yr_wk"].unique())
    unknown_weeks = price_weeks - calendar_weeks
    require(
        not unknown_weeks,
        f"Price table contains weeks absent from calendar: {sorted(unknown_weeks)[:10]}.",
    )

    series_pairs = meta[["item_id", "store_id"]].drop_duplicates()
    price_pairs = prices[["item_id", "store_id"]].drop_duplicates()
    pair_check = series_pairs.merge(
        price_pairs, on=["item_id", "store_id"], how="left", indicator=True
    )
    missing_pairs = int((pair_check["_merge"] == "left_only").sum())

    print("\n[4/5] Sell-price table and join keys")
    print(f"  Price rows: {len(prices):,}")
    print(f"  Unique item-store-week keys: {len(prices):,}")
    print(f"  Unique price weeks: {prices['wm_yr_wk'].nunique():,}")
    print(
        f"  Price item-store pairs: {len(price_pairs):,} / {len(series_pairs):,} sales series"
    )
    print(f"  Sales item-store pairs with no price row at any week: {missing_pairs:,}")
    print(
        f"  Price week range: {prices['wm_yr_wk'].min()} to {prices['wm_yr_wk'].max()}"
    )
    if missing_pairs:
        print(
            "  NOTE: Review these unmatched item-store pairs before relying on price covariates."
        )


def validate_sales_values(
    eval_path: Path, valid_path: Path, eval_days: list[str], valid_days: list[str]
) -> None:
    """Stream both files in chunks; check values and exact validation-prefix equality."""
    chunk_size = 256
    eval_reader = pd.read_csv(
        eval_path,
        usecols=eval_days,
        dtype=np.uint16,
        chunksize=chunk_size,
    )
    valid_reader = pd.read_csv(
        valid_path,
        usecols=valid_days,
        dtype=np.uint16,
        chunksize=chunk_size,
    )

    rows_checked = 0
    chunks_checked = 0
    for eval_chunk, valid_chunk in zip(eval_reader, valid_reader, strict=True):
        # Explicitly reindex or align by valid_days to maintain strict chronological parity
        eval_prefix = eval_chunk.reindex(columns=valid_days).to_numpy(copy=False)
        valid_values = valid_chunk.reindex(columns=valid_days).to_numpy(copy=False)
        require(
            np.array_equal(eval_prefix, valid_values),
            f"Validation sales differ from the first 1,913 evaluation days near row {rows_checked}.",
        )
        rows_checked += len(eval_chunk)
        chunks_checked += 1

    require(
        rows_checked == 30_490, f"Checked {rows_checked:,} sales rows; expected 30,490."
    )
    print("\n[5/5] Sales values (chunked; no long-form reshape)")
    print(f"  Rows checked: {rows_checked:,}")
    print(f"  Chunks checked: {chunks_checked:,} (chunk size {chunk_size:,})")
    print(
        "  All parsed daily sales values fit uint16 and are therefore non-negative integers."
    )
    print(
        "  sales_train_validation.csv exactly matches d_1..d_1913 in sales_train_evaluation.csv."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-sales-values",
        action="store_true",
        help="Skip the chunked daily-value and validation-prefix comparison (faster, less thorough).",
    )
    args = parser.parse_args()

    files = {
        "sales_train_evaluation.csv": RAW / "sales_train_evaluation.csv",
        "sales_train_validation.csv": RAW / "sales_train_validation.csv",
        "calendar.csv": RAW / "calendar.csv",
        "sell_prices.csv": RAW / "sell_prices.csv",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    require(not missing, f"Missing required raw data files: {', '.join(missing)}")

    print("M5 raw-data validation")
    print(f"Project root: {ROOT}")
    print(f"Raw data:     {RAW}")
    for name, path in files.items():
        print(f"  {name:<32} {path.stat().st_size / (1024 * 1024):8.2f} MiB")

    eval_days, valid_days = validate_sales_headers(
        files["sales_train_evaluation.csv"], files["sales_train_validation.csv"]
    )
    metadata = validate_series_metadata(
        files["sales_train_evaluation.csv"], files["sales_train_validation.csv"]
    )
    calendar = validate_calendar(files["calendar.csv"], eval_days)
    validate_prices(files["sell_prices.csv"], calendar, metadata)

    if args.skip_sales_values:
        print("\n[5/5] Sales-value check skipped by request.")
    else:
        validate_sales_values(
            files["sales_train_evaluation.csv"],
            files["sales_train_validation.csv"],
            eval_days,
            valid_days,
        )

    print("\nPASS: M5 raw-data validation completed.")


if __name__ == "__main__":
    main()
