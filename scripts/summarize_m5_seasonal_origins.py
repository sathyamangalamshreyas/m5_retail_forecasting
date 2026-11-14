#!/usr/bin/env python3
"""Label and summarize the 20 non-overlapping M5 seasonal-origin results.

Run after scripts/run_m5_rolling_origin.py has generated the 20 forecast
windows. This script performs no inference; it only labels and summarizes the
combined CSV outputs. It uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# start day -> season, year, start/end dates. M5 d_1 = 2011-01-29.
WINDOWS = [
    (429, "Spring", 2012, "2012-04-01", "2012-04-28"),
    (520, "Summer", 2012, "2012-07-01", "2012-07-28"),
    (612, "Autumn", 2012, "2012-10-01", "2012-10-28"),
    (657, "Thanksgiving", 2012, "2012-11-15", "2012-12-12"),
    (687, "Christmas_NewYear", 2012, "2012-12-15", "2013-01-11"),
    (794, "Spring", 2013, "2013-04-01", "2013-04-28"),
    (885, "Summer", 2013, "2013-07-01", "2013-07-28"),
    (977, "Autumn", 2013, "2013-10-01", "2013-10-28"),
    (1022, "Thanksgiving", 2013, "2013-11-15", "2013-12-12"),
    (1052, "Christmas_NewYear", 2013, "2013-12-15", "2014-01-11"),
    (1159, "Spring", 2014, "2014-04-01", "2014-04-28"),
    (1250, "Summer", 2014, "2014-07-01", "2014-07-28"),
    (1342, "Autumn", 2014, "2014-10-01", "2014-10-28"),
    (1387, "Thanksgiving", 2014, "2014-11-15", "2014-12-12"),
    (1417, "Christmas_NewYear", 2014, "2014-12-15", "2015-01-11"),
    (1524, "Spring", 2015, "2015-04-01", "2015-04-28"),
    (1615, "Summer", 2015, "2015-07-01", "2015-07-28"),
    (1707, "Autumn", 2015, "2015-10-01", "2015-10-28"),
    (1752, "Thanksgiving", 2015, "2015-11-15", "2015-12-12"),
    (1782, "Christmas_NewYear", 2015, "2015-12-15", "2016-01-11"),
]
META = {
    start: {
        "season": season,
        "season_year": str(year),
        "forecast_start_date": start_date,
        "forecast_end_date": end_date,
    }
    for start, season, year, start_date, end_date in WINDOWS
}
METRIC_FIELDS = [
    "mae", "rmse", "wape", "bias", "smape",
    "bottom_level_subset_wrmsse_style", "weighted_pinball",
    "aggregate_error_pct_actual", "aggregate_forecast_to_actual_pct",
]
COMPARISON_FIELDS = [
    "mae_improvement_pct", "rmse_improvement_pct", "wape_improvement_pct",
    "wrmsse_style_improvement_pct", "weighted_pinball_improvement_pct",
    "seasonal_naive_aggregate_error_pct_actual", "timestral_aggregate_error_pct_actual",
    "seasonal_naive_bias", "timestral_bias",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean_std(values: list[float]) -> tuple[float | str, float | str]:
    valid = [value for value in values if math.isfinite(value)]
    if not valid:
        return "", ""
    avg = statistics.mean(valid)
    sd = statistics.stdev(valid) if len(valid) > 1 else ""
    return avg, sd


def aggregate_rows(
    rows: list[dict[str, Any]],
    keys: list[str],
    numeric_fields: list[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    result: list[dict[str, Any]] = []
    for key_values, group in sorted(groups.items()):
        item: dict[str, Any] = dict(zip(keys, key_values))
        item["n_windows"] = len(group)
        for field in numeric_fields:
            avg, sd = mean_std([number(row.get(field)) for row in group])
            item[f"{field}_mean"] = avg
            item[f"{field}_std"] = sd
        result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/independent_seasons"),
        help="directory containing rolling_origin_metrics.csv and rolling_origin_comparison.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="output directory (default: <input>/seasonal_summaries)",
    )
    args = parser.parse_args()
    output = args.output or (args.input / "seasonal_summaries")
    output.mkdir(parents=True, exist_ok=True)

    metrics_path = args.input / "rolling_origin_metrics.csv"
    comparison_path = args.input / "rolling_origin_comparison.csv"
    for path in (metrics_path, comparison_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. First run scripts/run_m5_rolling_origin.py "
                "with the 20 seasonal origin IDs and --output matching --input."
            )

    metrics_raw = read_csv(metrics_path)
    comparison_raw = read_csv(comparison_path)
    expected = set(META)

    def origin_set(rows: list[dict[str, str]]) -> set[int]:
        return {int(row["forecast_start_day"]) for row in rows}

    missing_metrics = sorted(expected - origin_set(metrics_raw))
    missing_comparisons = sorted(expected - origin_set(comparison_raw))
    extra = sorted((origin_set(metrics_raw) | origin_set(comparison_raw)) - expected)
    if missing_metrics or missing_comparisons or extra:
        raise ValueError(
            "The seasonal run is incomplete or contains unexpected origins. "
            f"Missing in metrics={missing_metrics}; missing in comparison={missing_comparisons}; "
            f"unexpected={extra}. Complete the 20-window run in a separate output directory."
        )

    def add_metadata(row: dict[str, str]) -> dict[str, Any]:
        origin = int(row["forecast_start_day"])
        return {**row, **META[origin]}

    metrics = [add_metadata(row) for row in metrics_raw]
    comparison = [add_metadata(row) for row in comparison_raw]

    # Ensure two paired model rows exist for every store and origin.
    model_counts: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in metrics:
        model_counts[(row["store"], int(row["forecast_start_day"]))].add(row["model"])
    bad_pairs = [key for key, models in model_counts.items()
                 if models != {"SeasonalNaive7_point", "Timestral_median"}]
    if bad_pairs:
        raise ValueError(f"Missing paired model results for store/origin: {bad_pairs[:10]}")

    metadata_fields = ["season", "season_year", "forecast_start_date", "forecast_end_date"]
    metric_fields = list(metrics_raw[0]) + metadata_fields
    comparison_fields = list(comparison_raw[0]) + metadata_fields
    write_csv(output / "seasonal_origin_metrics_labeled.csv", metrics, metric_fields)
    write_csv(output / "seasonal_origin_comparison_labeled.csv", comparison, comparison_fields)

    by_season_metrics = aggregate_rows(metrics, ["store", "season", "model"], METRIC_FIELDS)
    write_csv(output / "seasonal_origin_by_season_metrics.csv", by_season_metrics)
    by_year_metrics = aggregate_rows(
        metrics, ["store", "season_year", "season", "model"], METRIC_FIELDS
    )
    write_csv(output / "seasonal_origin_by_year_metrics.csv", by_year_metrics)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparison:
        grouped[(str(row["store"]), str(row["season"]))].append(row)
    bias_rows: list[dict[str, Any]] = []
    for (store, season), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "store": store,
            "season": season,
            "n_years": len(group),
            "timestral_underforecast_windows": sum(
                number(row["timestral_aggregate_error_pct_actual"]) < 0 for row in group
            ),
            "naive_underforecast_windows": sum(
                number(row["seasonal_naive_aggregate_error_pct_actual"]) < 0 for row in group
            ),
        }
        for field in COMPARISON_FIELDS:
            avg, sd = mean_std([number(row.get(field)) for row in group])
            item[f"{field}_mean"] = avg
            item[f"{field}_std"] = sd
        bias_rows.append(item)
    write_csv(output / "seasonal_origin_bias_by_season.csv", bias_rows)

    # Coverage makes it immediately obvious whether all stores have 20 windows.
    coverage_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparison:
        coverage_groups[str(row["store"])].append(row)
    coverage: list[dict[str, Any]] = []
    for store, group in sorted(coverage_groups.items()):
        coverage.append({
            "store": store,
            "n_windows": len(group),
            "n_unique_origins": len({int(row["forecast_start_day"]) for row in group}),
            "timestral_underforecast_windows": sum(
                number(row["timestral_aggregate_error_pct_actual"]) < 0 for row in group
            ),
            "naive_underforecast_windows": sum(
                number(row["seasonal_naive_aggregate_error_pct_actual"]) < 0 for row in group
            ),
        })
    write_csv(output / "seasonal_origin_coverage_check.csv", coverage)

    print(f"Wrote labeled seasonal summaries to: {output}")
    for row in coverage:
        print(row)


if __name__ == "__main__":
    main()
