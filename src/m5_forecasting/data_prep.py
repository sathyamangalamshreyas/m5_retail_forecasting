"""Phase 0: convert the raw M5 CSVs to parquet and extract a working subset.

Run from anywhere inside the project (paths are resolved from this file):

    pixi run prep-data                                # full convert + CA_1 subset
    pixi run prep-data --stores CA_1 CA_2 --subset-name ca12

Works in both the default and the eda Pixi environments (needs only polars + numpy).

Outputs, all under data/processed/:
    calendar.parquet      typed calendar with integer day index ``d_num``
    sell_prices.parquet   weekly prices, typed
    sales_wide.parquet    one row per item-store series, Int16 columns d_1..d_N
    series_meta.parquet   ids, first_sale_d, first_live_d, zero_share, total_sales
    aggregates.parquet    level / key / sales (list) for 7 aggregation levels
    subsets/<name>/
        series.parquet    ids + live/sale metadata, in array row order
        sales.npy         float32 [n_series, n_days], C-contiguous
        price.npy         float32 [n_series, n_days], NaN where no price entry
        long.parquet      long table (id, d_num, date, wm_yr_wk, sell_price, sales)

Conventions
- Only ``sales_train_evaluation.csv`` is converted: it is d_1..d_1941 and the
  validation file is exactly its first 1,913 days (checked by
  scripts/validate_m5_data.py). Hold out the last 28 days for evaluation.
- Day numbers (``d_num``, ``first_live_d``, ``first_sale_d``) are 1-based, so array
  column ``j`` is day ``j + 1``.
- "Live" means a price entry exists for that store/item/week. Zeros before the
  first price entry are "not on sale yet", not zero demand. ``first_live_d == 1``
  means live since the start of the data (left-censored).
- Subset rows keep the original CSV order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
AGG_LEVELS = [
    "total",
    "state_id",
    "store_id",
    "cat_id",
    "dept_id",
    "store_cat",
    "store_dept",
]


def _day_columns(columns: list[str]) -> list[str]:
    """Day columns sorted numerically (d_2 before d_10)."""
    return sorted((c for c in columns if c.startswith("d_")), key=lambda c: int(c[2:]))


# ---------------------------------------------------------------------- calendar
def convert_calendar(raw: Path, out: Path) -> None:
    cal = pl.read_csv(
        raw / "calendar.csv",
        infer_schema_length=None,
        schema_overrides={"date": pl.Date},
    )
    cal = cal.with_columns(
        pl.col("d").str.slice(2).cast(pl.Int32).alias("d_num"),
        pl.col("wm_yr_wk").cast(pl.Int32),
        pl.col("wday").cast(pl.Int8),
        pl.col("month").cast(pl.Int8),
        pl.col("year").cast(pl.Int16),
        pl.col("snap_CA", "snap_TX", "snap_WI").cast(pl.Int8),
    )
    assert cal["d_num"].to_list() == list(range(1, cal.height + 1)), (
        "calendar rows are not d_1..d_N in order"
    )
    cal.write_parquet(out / "calendar.parquet", compression="zstd")


# ---------------------------------------------------------------------- prices
def convert_prices(raw: Path, out: Path) -> None:
    (
        pl.scan_csv(raw / "sell_prices.csv", infer_schema_length=None)
        .with_columns(
            pl.col("wm_yr_wk").cast(pl.Int32),
            pl.col("sell_price").cast(pl.Float32),
        )
        .sink_parquet(out / "sell_prices.parquet", compression="zstd")
    )


# ---------------------------------------------------------------------- sales (wide)
def convert_sales_wide(raw: Path, out: Path, sales_file: str) -> list[str]:
    lf = pl.scan_csv(raw / sales_file)
    day_cols = _day_columns(lf.collect_schema().names())
    expected = [f"d_{i}" for i in range(1, len(day_cols) + 1)]
    assert day_cols == expected, "sales day columns are not d_1..d_N"
    lf.select(*ID_COLS, *[pl.col(c).cast(pl.Int16) for c in day_cols]).sink_parquet(
        out / "sales_wide.parquet", compression="zstd"
    )
    return day_cols


# ---------------------------------------------------------------------- series meta
def build_series_meta(out: Path, day_cols: list[str]) -> None:
    cal = pl.read_parquet(out / "calendar.parquet", columns=["d_num", "wm_yr_wk"])
    week_start = cal.group_by("wm_yr_wk").agg(pl.col("d_num").min().alias("live_d"))

    first_live = (
        pl.scan_parquet(out / "sell_prices.parquet")
        .group_by("store_id", "item_id")
        .agg(pl.col("wm_yr_wk").min())
        .collect()
        .join(week_start, on="wm_yr_wk", how="left")
        .select("store_id", "item_id", pl.col("live_d").alias("first_live_d"))
    )

    wide = pl.read_parquet(out / "sales_wide.parquet")
    arr = wide.select(day_cols).to_numpy()  # [n_series, n_days] int16
    has_sale = (arr > 0).any(axis=1)
    first_sale = np.where(has_sale, (arr > 0).argmax(axis=1) + 1, 0)

    meta = wide.select(ID_COLS).with_columns(
        pl.Series("first_sale_d", first_sale, dtype=pl.Int32),
        pl.Series("zero_share", (arr == 0).mean(axis=1), dtype=pl.Float32),
        pl.Series("total_sales", arr.sum(axis=1, dtype=np.int64), dtype=pl.Int64),
    )
    # 0 means "never sold": make it a real null.
    meta = meta.with_columns(
        pl.when(pl.col("first_sale_d") > 0)
        .then(pl.col("first_sale_d"))
        .otherwise(None)
        .alias("first_sale_d")
    )
    meta = meta.join(first_live, on=["store_id", "item_id"], how="left")
    meta.write_parquet(out / "series_meta.parquet", compression="zstd")
    print(
        f"  {meta.height:,} series; never sold: {int((~has_sale).sum())}; "
        f"no price row: {meta['first_live_d'].null_count()}"
    )


# ---------------------------------------------------------------------- aggregates
def build_aggregates(out: Path, day_cols: list[str]) -> None:
    wide = pl.read_parquet(out / "sales_wide.parquet")
    arr = wide.select(day_cols).to_numpy().astype(np.float32)
    meta = wide.select(ID_COLS).with_columns(
        (pl.col("store_id") + "|" + pl.col("cat_id")).alias("store_cat"),
        (pl.col("store_id") + "|" + pl.col("dept_id")).alias("store_dept"),
        pl.lit("total").alias("total"),
    )
    levels, keys, series = [], [], []
    for level in AGG_LEVELS:
        groups = meta.with_row_index("row").group_by(level).agg(pl.col("row"))
        for key, rows in groups.sort(level).iter_rows():
            levels.append(level)
            keys.append(key)
            series.append(arr[rows].sum(axis=0))
    pl.DataFrame(
        {"level": levels, "key": keys, "sales": [s.tolist() for s in series]},
        schema={"level": pl.String, "key": pl.String, "sales": pl.List(pl.Float32)},
    ).write_parquet(out / "aggregates.parquet", compression="zstd")
    print(f"  {len(keys)} aggregate series")


# ---------------------------------------------------------------------- subset
def build_subset(
    out: Path,
    day_cols: list[str],
    stores: list[str],
    name: str,
    write_long: bool = True,
) -> Path:
    sub_dir = out / "subsets" / name
    sub_dir.mkdir(parents=True, exist_ok=True)
    n_days = len(day_cols)

    wide = (
        pl.scan_parquet(out / "sales_wide.parquet")
        .filter(pl.col("store_id").is_in(stores))
        .collect()
    )
    if wide.height == 0:
        raise ValueError(f"No series found for stores={stores}")
    ids = wide.select(ID_COLS)
    sales = np.ascontiguousarray(wide.select(day_cols).to_numpy(), dtype=np.float32)

    meta = pl.read_parquet(out / "series_meta.parquet").select(
        "id", "first_sale_d", "first_live_d", "zero_share"
    )
    ids.join(meta, on="id", how="left").write_parquet(
        sub_dir / "series.parquet", compression="zstd"
    )
    np.save(sub_dir / "sales.npy", sales)

    # Weekly price broadcast onto days via the calendar.
    cal = pl.read_parquet(
        out / "calendar.parquet", columns=["d_num", "date", "wm_yr_wk"]
    ).filter(pl.col("d_num") <= n_days)
    prices = pl.read_parquet(out / "sell_prices.parquet").filter(
        pl.col("store_id").is_in(stores)
    )
    long = (
        ids.select("id", "item_id", "store_id")
        .with_row_index("row")
        .join(cal, how="cross")
        .join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        .sort("row", "d_num")
    )
    assert long.height == sales.size, "long table size mismatch"
    price = np.ascontiguousarray(
        long["sell_price"].to_numpy().reshape(sales.shape), dtype=np.float32
    )
    np.save(sub_dir / "price.npy", price)

    if write_long:
        long.with_columns(pl.Series("sales", sales.reshape(-1))).drop(
            "row"
        ).write_parquet(sub_dir / "long.parquet", compression="zstd")

    live = ~np.isnan(price)
    print(
        f"  {sales.shape[0]:,} series x {sales.shape[1]:,} days; "
        f"zero share {np.mean(sales == 0):.1%}"
    )
    print(
        f"  sales>0 on days with no price entry: {int(((sales > 0) & ~live).sum()):,}"
        f"; zero-sales days while live: {np.mean(sales[live] == 0):.1%}"
    )
    return sub_dir


# ---------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert M5 CSVs to parquet and extract a subset."
    )
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--sales-file", default="sales_train_evaluation.csv")
    ap.add_argument("--stores", nargs="+", default=["CA_1"])
    ap.add_argument("--subset-name", default=None, help="default: stores joined by _")
    ap.add_argument("--skip-aggregates", action="store_true")
    ap.add_argument("--skip-long", action="store_true", help="don't write long.parquet")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    name = args.subset_name or "_".join(args.stores).lower()
    print(f"polars {pl.__version__}; raw={args.raw}; out={args.out}")

    print("calendar ...")
    convert_calendar(args.raw, args.out)
    print("prices ...")
    convert_prices(args.raw, args.out)
    print("sales (wide) ...")
    day_cols = convert_sales_wide(args.raw, args.out, args.sales_file)
    print(f"  {len(day_cols)} days")
    print("series meta ...")
    build_series_meta(args.out, day_cols)
    if not args.skip_aggregates:
        print("aggregates ...")
        build_aggregates(args.out, day_cols)
    print(f"subset '{name}' ...")
    sub = build_subset(args.out, day_cols, args.stores, name, not args.skip_long)
    print(f"done -> {sub}")


if __name__ == "__main__":
    main()
