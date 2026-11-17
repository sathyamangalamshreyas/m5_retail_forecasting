#!/usr/bin/env python3
"""Benchmark Timestral T2 known-future calendar covariates on one representative M5 store.

Runs the same two store/origin tasks under:
  1) CPU serial: one model process, 4 PyTorch intra-op threads;
  2) MPS serial: one model process on Apple's MPS backend;
  3) CPU parallel: two spawned persistent workers, 2 PyTorch threads each.

Default workload: CA_1, d_429 and d_1782, all 3,049 series, horizon 28,
batch size 16. The early and late origins exercise different context lengths.

This is an execution benchmark, not a model-quality evaluation. It saves forecast
quantiles so CPU-parallel and MPS outputs can be compared with CPU-serial output.
Use --max-series 64 and a separate --output for a short API smoke run before the
full benchmark. Completed modes are skipped automatically when resuming.
"""
from __future__ import annotations

# Set thread limits before NumPy/PyTorch are imported in spawned child processes.
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import platform
import resource
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

try:
    import run_ca1_timestral_baseline as base
except ImportError as exc:
    raise ImportError(
        "Place this script in scripts/ beside run_ca1_timestral_baseline.py and run it from the project root."
    ) from exc

HORIZON = 28
DEFAULT_STORE = "ca_1"
DEFAULT_ORIGINS = (429, 1782)  # Spring 2012 and Christmas/New Year 2015
WEEKDAY_FEATURES = tuple(f"wday_{i}" for i in range(1, 8))
NUMERIC_KNOWN_FEATURES = WEEKDAY_FEATURES + ("weekend", "snap", "is_event")
CATEGORICAL_KNOWN_FEATURES = (
    "event_type_1", "event_type_2", "event_name_1", "event_name_2"
)
_WORKER_STATE: dict[str, Any] = {}


def _torch_configure_threads(n_threads: int) -> Any:
    import torch

    torch.set_num_threads(n_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch allows this to be set only before inter-op work begins.
        pass
    return torch


def _peak_rss_gib() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":  # macOS reports bytes
        return peak / (1024**3)
    return peak / (1024**2)  # Linux reports KiB


def _category_codes(values: list[Any]) -> np.ndarray:
    categories = sorted({str(v) for v in values if v is not None})
    mapping = {value: i for i, value in enumerate(categories)}
    out = np.full(len(values), np.nan, dtype=np.float32)
    for i, value in enumerate(values):
        if value is not None:
            out[i] = float(mapping[str(value)])
    return out


def load_calendar_features(calendar_path: Path, store: str, max_day: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build known-future calendar features for d_1..d_max_day.

    Numeric rows: 7 weekday indicators, weekend, store-specific SNAP, event flag.
    Categorical rows: both event types and both event names, with NaN for no event.
    """
    if not calendar_path.is_file():
        raise FileNotFoundError(f"Missing M5 calendar file: {calendar_path}")
    cal = pl.read_csv(calendar_path, infer_schema_length=None)
    required = {
        "d", "wday", "weekday", "event_type_1", "event_type_2",
        "event_name_1", "event_name_2", f"snap_{store[:2].upper()}"
    }
    missing = required.difference(cal.columns)
    if missing:
        raise ValueError(f"Calendar is missing required columns: {sorted(missing)}")
    cal = cal.with_columns(pl.col("d").str.strip_prefix("d_").cast(pl.Int32).alias("d_num")).sort("d_num")
    expected = np.arange(1, cal.height + 1, dtype=np.int32)
    if not np.array_equal(cal["d_num"].to_numpy(), expected):
        raise ValueError("calendar.csv must contain contiguous d_1..d_N rows")
    if max_day > cal.height:
        raise ValueError(f"Requested d_{max_day}, but calendar contains only {cal.height} rows")
    cal = cal.head(max_day)

    wday = cal.get_column("wday").cast(pl.Int8).to_numpy()
    if not np.isin(wday, np.arange(1, 8)).all():
        raise ValueError("Unexpected wday values; expected integers 1..7")
    numeric = np.zeros((len(NUMERIC_KNOWN_FEATURES), max_day), dtype=np.float32)
    for row, day in enumerate(range(1, 8)):
        numeric[row] = (wday == day).astype(np.float32)
    weekday_names = np.asarray([str(x).strip().lower() for x in cal["weekday"].to_list()])
    numeric[7] = np.isin(weekday_names, ["saturday", "sunday"]).astype(np.float32)
    snap_col = f"snap_{store[:2].upper()}"
    numeric[8] = cal.get_column(snap_col).cast(pl.Float32).to_numpy()

    event_name_1 = cal.get_column("event_name_1").to_list()
    event_name_2 = cal.get_column("event_name_2").to_list()
    event_type_1 = cal.get_column("event_type_1").to_list()
    event_type_2 = cal.get_column("event_type_2").to_list()
    has_event = np.asarray([
        (a is not None) or (b is not None) for a, b in zip(event_name_1, event_name_2)
    ], dtype=np.float32)
    numeric[9] = has_event

    categorical = np.stack([
        _category_codes(event_type_1),
        _category_codes(event_type_2),
        _category_codes(event_name_1),
        _category_codes(event_name_2),
    ]).astype(np.float32, copy=False)
    feature_names = list(NUMERIC_KNOWN_FEATURES) + list(CATEGORICAL_KNOWN_FEATURES)
    return numeric, categorical, feature_names


def _load_store_data(processed: Path, store: str, max_series: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    subset = processed / "subsets" / store
    sales_path = subset / "sales.npy"
    prices_path = subset / "price.npy"
    if not sales_path.is_file() or not prices_path.is_file():
        raise FileNotFoundError(f"Expected {sales_path} and {prices_path}")
    sales = np.load(sales_path, mmap_mode="r")
    prices = np.load(prices_path, mmap_mode="r")
    if sales.shape != prices.shape:
        raise ValueError(f"Sales/price shape mismatch for {store}: {sales.shape} vs {prices.shape}")
    if max_series is not None:
        if max_series < 1:
            raise ValueError("max_series must be positive")
        sales = sales[:max_series]
        prices = prices[:max_series]
    return sales, prices


def _forecast_origin(
    forecaster: Any,
    sales: np.ndarray,
    prices: np.ndarray,
    numeric_calendar: np.ndarray,
    categorical_calendar: np.ndarray,
    store: str,
    origin: int,
    batch_size: int,
    device: str,
    out_dir: Path,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    train_days = origin - 1
    end_day = origin + HORIZON - 1
    if end_day > sales.shape[1] or end_day > numeric_calendar.shape[1]:
        raise ValueError(f"Origin d_{origin} exceeds available sales/calendar history")

    train = np.asarray(sales[:, :train_days], dtype=np.float32)
    train_price = np.asarray(prices[:, :train_days], dtype=np.float32)
    context = np.where(np.isfinite(train_price), train, np.nan).astype(np.float32, copy=False)
    known = numeric_calendar[:, :end_day]
    categorical = categorical_calendar[:, :end_day]
    prep_seconds = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    q_parts: list[np.ndarray] = []
    levels: np.ndarray | None = None
    forecast_start = time.perf_counter()
    n_series = context.shape[0]
    import torch
    inference_context = torch.inference_mode()
    with inference_context:
        for start in range(0, n_series, batch_size):
            stop = min(start + batch_size, n_series)
            x = context[start:stop, None, :]
            # Covariates are shared across series in this store; materialize only a batch.
            known_batch = np.broadcast_to(
                known[None, :, :], (stop - start, known.shape[0], known.shape[1])
            ).copy()
            categorical_batch = np.broadcast_to(
                categorical[None, :, :],
                (stop - start, categorical.shape[0], categorical.shape[1]),
            ).copy()
            result = forecaster.forecast(
                x,
                prediction_length=HORIZON,
                known_covariates=known_batch,
                categorical_known_covariates=categorical_batch,
            )
            if device == "mps":
                torch.mps.synchronize()
            q = base._as_numpy(result.quantiles).astype(np.float32, copy=False)[:, 0, :, :]
            current_levels = base._as_numpy(result.quantile_levels).astype(np.float32, copy=False)
            if q.shape != (stop - start, HORIZON, current_levels.size):
                raise ValueError(f"Unexpected quantile output shape: {q.shape}")
            if not np.isfinite(q).all():
                raise ValueError(f"Non-finite output for series rows {start}:{stop}")
            if levels is None:
                levels = current_levels
            elif not np.array_equal(levels, current_levels):
                raise ValueError("Quantile levels changed between batches")
            q_parts.append(q.copy())

    inference_seconds = time.perf_counter() - forecast_start
    quantiles = np.concatenate(q_parts, axis=0)
    if levels is None:
        raise RuntimeError("No forecast batches generated")
    if np.any(np.diff(quantiles, axis=-1) < -1e-5):
        raise ValueError("Forecast quantiles are crossing")

    np.save(out_dir / "quantiles.npy", quantiles)
    np.save(out_dir / "quantile_levels.npy", levels)
    total_seconds = time.perf_counter() - t0
    del train, train_price, context, q_parts, quantiles
    return {
        "store": store.upper(),
        "origin": origin,
        "forecast_end_day": end_day,
        "n_series": int(sales.shape[0]),
        "context_days": train_days,
        "horizon": HORIZON,
        "batch_size": batch_size,
        "device": device,
        "prep_seconds": prep_seconds,
        "inference_seconds": inference_seconds,
        "task_seconds": total_seconds,
        "n_batches": int((sales.shape[0] + batch_size - 1) // batch_size),
        "quantile_count": int(levels.size),
        "quantile_levels": levels.tolist(),
        "output_path": str(out_dir / "quantiles.npy"),
        "peak_rss_gib": _peak_rss_gib(),
        "mps_current_allocated_gib": (
            float(torch.mps.current_allocated_memory()) / (1024**3) if device == "mps" else None
        ),
        "mps_driver_allocated_gib": (
            float(torch.mps.driver_allocated_memory()) / (1024**3) if device == "mps" else None
        ),
    }


def _run_serial_mode(
    mode: str,
    device: str,
    torch_threads: int,
    store: str,
    origins: list[int],
    processed: Path,
    calendar_path: Path,
    model_id: str,
    batch_size: int,
    output: Path,
    max_series: int | None,
) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(torch_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(torch_threads)
    torch = _torch_configure_threads(torch_threads)
    mode_start = time.perf_counter()
    data_start = time.perf_counter()
    sales, prices = _load_store_data(processed, store, max_series)
    max_day = max(origins) + HORIZON - 1
    numeric_calendar, categorical_calendar, feature_names = load_calendar_features(
        calendar_path, store, max_day
    )
    data_load_seconds = time.perf_counter() - data_start

    model_start = time.perf_counter()
    from timestral import TimestralForecaster
    forecaster = TimestralForecaster.from_pretrained(model_id, device=device)
    model_load_seconds = time.perf_counter() - model_start

    tasks: list[dict[str, Any]] = []
    for origin in origins:
        print(f"[{mode}] {store.upper()} d_{origin}: forecasting", flush=True)
        task = _forecast_origin(
            forecaster, sales, prices, numeric_calendar, categorical_calendar,
            store, origin, batch_size, device,
            output / mode / store / f"origin_d{origin:04d}_d{origin + HORIZON - 1:04d}",
        )
        tasks.append(task)
        print(
            f"[{mode}] d_{origin}: {task['inference_seconds'] / 60:.1f} min inference",
            flush=True,
        )
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    del forecaster, sales, prices
    import gc
    gc.collect()
    return {
        "mode": mode,
        "status": "ok",
        "device": device,
        "workers": 1,
        "torch_threads_per_worker": torch_threads,
        "data_load_seconds": data_load_seconds,
        "model_load_seconds": model_load_seconds,
        "wall_seconds": time.perf_counter() - mode_start,
        "tasks": tasks,
        "calendar_feature_names": feature_names,
        "parent_peak_rss_gib": _peak_rss_gib(),
    }


def _worker_initializer(
    store: str,
    processed_path: str,
    calendar_path: str,
    model_id: str,
    torch_threads: int,
    max_day: int,
    max_series: int | None,
) -> None:
    global _WORKER_STATE
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(torch_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(torch_threads)
    torch = _torch_configure_threads(torch_threads)
    load_data_start = time.perf_counter()
    processed = Path(processed_path)
    sales, prices = _load_store_data(processed, store, max_series)
    # Calendar covariates are loaded once per persistent worker.
    numeric_calendar, categorical_calendar, feature_names = load_calendar_features(
        Path(calendar_path), store, max_day
    )
    data_load_seconds = time.perf_counter() - load_data_start

    model_start = time.perf_counter()
    from timestral import TimestralForecaster
    forecaster = TimestralForecaster.from_pretrained(model_id, device="cpu")
    model_load_seconds = time.perf_counter() - model_start
    _WORKER_STATE = {
        "torch": torch,
        "store": store,
        "sales": sales,
        "prices": prices,
        "numeric_calendar": numeric_calendar,
        "categorical_calendar": categorical_calendar,
        "feature_names": feature_names,
        "forecaster": forecaster,
        "data_load_seconds": data_load_seconds,
        "model_load_seconds": model_load_seconds,
    }


def _parallel_worker_task(payload: tuple[int, int, str, int]) -> dict[str, Any]:
    origin, batch_size, output_str, horizon = payload
    state = _WORKER_STATE
    if not state:
        raise RuntimeError("Worker initializer did not populate state")
    if horizon != HORIZON:
        raise ValueError("Worker horizon mismatch")
    task = _forecast_origin(
        state["forecaster"], state["sales"], state["prices"],
        state["numeric_calendar"], state["categorical_calendar"],
        state["store"], origin, batch_size, "cpu",
        Path(output_str) / "cpu_parallel" / state["store"]
        / f"origin_d{origin:04d}_d{origin + HORIZON - 1:04d}",
    )
    task.update({
        "pid": os.getpid(),
        "worker_data_load_seconds": state["data_load_seconds"],
        "worker_model_load_seconds": state["model_load_seconds"],
    })
    print(
        f"[cpu_parallel pid={os.getpid()}] d_{origin}: "
        f"{task['inference_seconds'] / 60:.1f} min inference",
        flush=True,
    )
    return task


def _run_parallel_cpu(
    store: str,
    origins: list[int],
    processed: Path,
    calendar_path: Path,
    model_id: str,
    batch_size: int,
    output: Path,
    workers: int,
    torch_threads: int,
    max_day: int,
    max_series: int | None,
) -> dict[str, Any]:
    mode_start = time.perf_counter()
    if workers != 2:
        raise ValueError("This benchmark is specifically configured for two persistent workers")
    # Ensure spawned workers inherit bounded native thread settings.
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(torch_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(torch_threads)
    context = mp.get_context("spawn")
    tasks: list[dict[str, Any]] = []
    errors: list[str] = []
    with cf.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_worker_initializer,
        initargs=(store, str(processed), str(calendar_path), model_id, torch_threads, max_day, max_series),
    ) as executor:
        futures = {
            executor.submit(_parallel_worker_task, (origin, batch_size, str(output), HORIZON)): origin
            for origin in origins
        }
        for future in cf.as_completed(futures):
            origin = futures[future]
            try:
                tasks.append(future.result())
            except Exception:
                errors.append(f"origin d_{origin}: {traceback.format_exc()}")
    return {
        "mode": "cpu_parallel",
        "status": "ok" if not errors and len(tasks) == len(origins) else "partial_or_failed",
        "device": "cpu",
        "workers": workers,
        "torch_threads_per_worker": torch_threads,
        "wall_seconds": time.perf_counter() - mode_start,
        "tasks": sorted(tasks, key=lambda x: x["origin"]),
        "errors": errors,
        "worker_peak_rss_gib": [t["peak_rss_gib"] for t in tasks],
        "worker_pids": sorted({t.get("pid") for t in tasks}),
    }


def _compare_outputs(output: Path, store: str, origins: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for origin in origins:
        end = origin + HORIZON - 1
        rel = Path(store) / f"origin_d{origin:04d}_d{end:04d}" / "quantiles.npy"
        ref_path = output / "cpu_serial" / rel
        if not ref_path.is_file():
            continue
        reference = np.load(ref_path, mmap_mode="r")
        for mode in ("cpu_parallel", "mps_serial"):
            candidate_path = output / mode / rel
            if not candidate_path.is_file():
                rows.append({"store": store.upper(), "origin": origin, "mode": mode, "status": "missing"})
                continue
            candidate = np.load(candidate_path, mmap_mode="r")
            if candidate.shape != reference.shape:
                rows.append({"store": store.upper(), "origin": origin, "mode": mode, "status": "shape_mismatch", "candidate_shape": str(candidate.shape), "reference_shape": str(reference.shape)})
                continue
            diff = np.abs(np.asarray(candidate) - np.asarray(reference))
            rows.append({
                "store": store.upper(), "origin": origin, "mode": mode, "status": "compared",
                "max_abs_difference": float(diff.max(initial=0.0)),
                "mean_abs_difference": float(diff.mean()),
                "p99_abs_difference": float(np.quantile(diff, 0.99)),
                "allclose_rtol_1e-3_atol_1e-4": bool(np.allclose(candidate, reference, rtol=1e-3, atol=1e-4)),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=DEFAULT_STORE, choices=("ca_1", "wi_3", "tx_1"))
    parser.add_argument("--origins", nargs="+", type=int, default=list(DEFAULT_ORIGINS))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--calendar", type=Path, default=Path("data/raw/calendar.csv"))
    parser.add_argument("--model", default=base.MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-serial-threads", type=int, default=4)
    parser.add_argument("--cpu-parallel-threads-per-worker", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("artifacts/t2_execution_benchmark"))
    parser.add_argument("--max-series", type=int, default=None, help="optional series cap for a short API smoke run")
    parser.add_argument("--force", action="store_true", help="delete the benchmark output directory before running")
    parser.add_argument("--skip", nargs="*", choices=("cpu-serial", "cpu-parallel", "mps-serial"), default=[])
    args = parser.parse_args()

    if len(args.origins) != 2 or len(set(args.origins)) != 2:
        parser.error("Use exactly two distinct origins for this benchmark (default: 429 and 1782).")
    if args.batch_size < 1 or args.cpu_serial_threads < 1 or args.cpu_parallel_threads_per_worker < 1:
        parser.error("Batch size and thread counts must be positive")
    if args.workers != 2:
        parser.error("--workers must be 2 for this benchmark")
    if args.max_series is not None and args.max_series < 1:
        parser.error("--max-series must be positive")
    max_day = max(args.origins) + HORIZON - 1
    args.output = args.output.resolve()
    existing_config_path = args.output / "benchmark_config.json"
    if args.output.exists() and args.force:
        shutil.rmtree(args.output)
    elif args.output.exists() and not existing_config_path.is_file():
        parser.error(
            f"Output directory exists without a benchmark config: {args.output}. "
            "Choose a new --output or use --force."
        )
    args.output.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "T2 calendar-covariate execution benchmark",
        "store": args.store.upper(), "origins": args.origins,
        "horizon": HORIZON, "batch_size": args.batch_size,
        "model": args.model, "workers": args.workers,
        "cpu_serial_threads": args.cpu_serial_threads,
        "cpu_parallel_threads_per_worker": args.cpu_parallel_threads_per_worker,
        "max_series": args.max_series,
        "numeric_known_features": list(NUMERIC_KNOWN_FEATURES),
        "categorical_known_features": list(CATEGORICAL_KNOWN_FEATURES),
        "tests": ["cpu_serial", "mps_serial", "cpu_parallel"],
    }
    if existing_config_path.is_file():
        previous_config = json.loads(existing_config_path.read_text())
        if previous_config != config:
            parser.error(
                "Existing benchmark configuration differs from this invocation. "
                "Use the original arguments, choose a new --output, or use --force."
            )
    (args.output / "benchmark_config.json").write_text(json.dumps(config, indent=2))
    progress_path = args.output / "benchmark_progress.json"
    summaries: list[dict[str, Any]] = json.loads(progress_path.read_text()) if progress_path.is_file() else []

    def completed(mode: str) -> bool:
        prior = next((row for row in summaries if row.get("mode") == mode), None)
        if not prior or prior.get("status") != "ok":
            return False
        return all(
            (args.output / mode / args.store
             / f"origin_d{origin:04d}_d{origin + HORIZON - 1:04d}" / "quantiles.npy").is_file()
            for origin in args.origins
        )

    def record(summary: dict[str, Any]) -> None:
        nonlocal summaries
        summaries = [row for row in summaries if row.get("mode") != summary.get("mode")]
        summaries.append(summary)
        progress_path.write_text(json.dumps(summaries, indent=2))

    if "cpu-serial" not in args.skip and not completed("cpu_serial"):
        print("\n=== Benchmark A: CPU serial (one process) ===", flush=True)
        try:
            record(_run_serial_mode(
                "cpu_serial", "cpu", args.cpu_serial_threads, args.store, args.origins,
                args.processed, args.calendar, args.model, args.batch_size, args.output,
                args.max_series,
            ))
        except Exception:
            record({"mode": "cpu_serial", "status": "failed", "error": traceback.format_exc()})
            print(traceback.format_exc(), flush=True)

    if "mps-serial" not in args.skip and not completed("mps_serial"):
        print("\n=== Benchmark B: MPS serial (one process) ===", flush=True)
        try:
            import torch
            if not torch.backends.mps.is_available():
                raise RuntimeError("torch.backends.mps.is_available() is False in this Pixi environment")
            record(_run_serial_mode(
                "mps_serial", "mps", args.cpu_serial_threads, args.store, args.origins,
                args.processed, args.calendar, args.model, args.batch_size, args.output,
                args.max_series,
            ))
        except Exception:
            record({"mode": "mps_serial", "status": "failed_or_unavailable", "error": traceback.format_exc()})
            print(traceback.format_exc(), flush=True)

    if "cpu-parallel" not in args.skip and not completed("cpu_parallel"):
        print("\n=== Benchmark C: CPU parallel (two spawned persistent workers) ===", flush=True)
        try:
            record(_run_parallel_cpu(
                args.store, args.origins, args.processed, args.calendar, args.model,
                args.batch_size, args.output, args.workers, args.cpu_parallel_threads_per_worker,
                max_day, args.max_series,
            ))
        except Exception:
            record({"mode": "cpu_parallel", "status": "failed", "error": traceback.format_exc()})
            print(traceback.format_exc(), flush=True)

    comparisons = _compare_outputs(args.output, args.store, args.origins)
    pl.DataFrame(comparisons).write_csv(args.output / "numerical_comparison.csv") if comparisons else None
    (args.output / "benchmark_results.json").write_text(json.dumps(summaries, indent=2))

    def csv_safe(value: Any) -> Any:
        """Convert nested metadata to scalar JSON strings for CSV export.

        Keep the full, structured values in benchmark_results.json. Polars CSV
        cannot serialize List/Struct columns, so flatten only the tabular copy.
        """
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    run_rows: list[dict[str, Any]] = []
    for summary in summaries:
        base_row = {k: v for k, v in summary.items() if k not in ("tasks", "errors")}
        if "tasks" in summary and summary["tasks"]:
            for task in summary["tasks"]:
                row = {**base_row, **{k: v for k, v in task.items() if k != "quantile_levels"}}
                run_rows.append({k: csv_safe(v) for k, v in row.items()})
        else:
            run_rows.append({k: csv_safe(v) for k, v in base_row.items()})
    if run_rows:
        pl.DataFrame(run_rows, infer_schema_length=None).write_csv(args.output / "benchmark_runs.csv")
    print("\nBenchmark complete. Results:", flush=True)
    for path in ("benchmark_results.json", "benchmark_runs.csv", "numerical_comparison.csv"):
        print(f"  {args.output / path}", flush=True)


if __name__ == "__main__":
    main()
