#!/usr/bin/env python3
"""Group A pilot: variable-length tslearn clustering for M5.

Methods:
  1) TimeSeriesKMeans(metric='dtw') with DBA centroids.
  2) KernelKMeans(kernel='gak') with Global Alignment Kernel.

The script deliberately starts with sampled exploratory searches. It does not
run a full-population clustering fit. All observed histories are represented as
variable-length sequences; no fixed common horizon or zero imputation is used.

Run from the M5 project root:
    pixi run python scripts/run_m5_group_a_tslearn.py --stage smoke-test
    pixi run python scripts/run_m5_group_a_tslearn.py --stage search --method both
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

WINDOW_START_D = 660
WINDOW_END_D = 1387
WINDOW_DAYS = WINDOW_END_D - WINDOW_START_D + 1
DEFAULT_K_VALUES = [2, 3, 4, 5, 6, 8, 10, 12]
DEFAULT_SEEDS = [42, 43, 44]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _parse_int_list(value: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("Values must be positive integers")
    return parsed


def _history_band(length: int) -> str:
    if length < 90:
        return "001_089"
    if length < 180:
        return "090_179"
    if length < 364:
        return "180_363"
    if length < 728:
        return "364_727"
    return "728"


def _stratum_key(meta_row: dict[str, Any], length: int) -> tuple[str, ...]:
    # Proportional stratification keeps the pilot from accidentally omitting
    # entire stores/categories or the short-history / zero-sales subgroups.
    return (
        str(meta_row.get("store_id", "NA")),
        str(meta_row.get("cat_id", "NA")),
        _history_band(length),
        str(meta_row.get("activity_label", "NA")),
    )


def _proportional_stratified_sample(
    row_indices: np.ndarray,
    meta_rows_by_index: dict[int, dict[str, Any]],
    lengths_by_index: dict[int, int],
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if sample_size > len(row_indices):
        raise ValueError(f"Requested sample {sample_size} exceeds eligible population {len(row_indices)}")
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for row_idx in row_indices.tolist():
        groups[_stratum_key(meta_rows_by_index[int(row_idx)], lengths_by_index[int(row_idx)])].append(int(row_idx))
    keys = list(groups)
    counts = np.asarray([len(groups[key]) for key in keys], dtype=np.int64)
    if sample_size < len(keys):
        # In extremely small pilot requests, fall back to random sampling rather
        # than pretending every stratum can be represented.
        return np.sort(rng.choice(row_indices, size=sample_size, replace=False))

    allocation = np.ones(len(keys), dtype=np.int64)
    remaining = sample_size - len(keys)
    capacity = counts - 1
    while remaining > 0 and int(capacity.sum()) > 0:
        ideal = remaining * capacity / capacity.sum()
        add = np.floor(ideal).astype(np.int64)
        add = np.minimum(add, capacity)
        if int(add.sum()) > 0:
            allocation += add
            capacity -= add
            remaining -= int(add.sum())
        if remaining <= 0 or int(capacity.sum()) == 0:
            break
        fractions = ideal - np.floor(ideal)
        order = np.argsort(-fractions)
        progressed = False
        for j in order:
            if capacity[j] > 0:
                allocation[j] += 1
                capacity[j] -= 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    chosen: list[int] = []
    for key, quota in zip(keys, allocation, strict=True):
        group = np.asarray(groups[key], dtype=np.int64)
        chosen.extend(rng.choice(group, size=int(quota), replace=False).tolist())
    chosen_array = np.asarray(chosen, dtype=np.int64)
    rng.shuffle(chosen_array)
    return chosen_array


def _load_prepared_data(input_dir: Path):
    required = ["sales_log1p.npy", "observation_mask.npy", "series_metadata.parquet"]
    missing = [name for name in required if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing prepared M5 inputs in {input_dir}: {missing}")

    sales_log1p = np.load(input_dir / "sales_log1p.npy", mmap_mode="r")
    observation_mask = np.load(input_dir / "observation_mask.npy", mmap_mode="r")
    metadata = pl.read_parquet(input_dir / "series_metadata.parquet")
    if sales_log1p.ndim != 2 or observation_mask.shape != sales_log1p.shape:
        raise ValueError("Prepared sales and observation mask must be matching 2D arrays")
    if sales_log1p.shape[1] != WINDOW_DAYS:
        raise ValueError(f"Expected {WINDOW_DAYS} days, got array shape {sales_log1p.shape}")
    if metadata.height != sales_log1p.shape[0]:
        raise ValueError("Metadata row count does not match array row count")
    if "row_index" not in metadata.columns or "n_price_live_days" not in metadata.columns:
        raise ValueError("Metadata must include row_index and n_price_live_days")

    row_ids = np.asarray(metadata["row_index"].to_numpy(), dtype=np.int64)
    if not np.array_equal(row_ids, np.arange(len(row_ids), dtype=np.int64)):
        raise ValueError("series_metadata.row_index must match the original array row order")
    mask_counts = np.asarray(observation_mask.sum(axis=1), dtype=np.int32)
    metadata_counts = np.asarray(metadata["n_price_live_days"].to_numpy(), dtype=np.int32)
    if not np.array_equal(mask_counts, metadata_counts):
        raise ValueError("Observation-mask counts disagree with n_price_live_days metadata")

    # The audit found continuous right-ending observed suffixes. Verify this
    # assumption before creating variable-length sequences; do not hide gaps.
    for row_idx in np.flatnonzero(mask_counts > 0):
        row_mask = np.asarray(observation_mask[row_idx], dtype=np.bool_)
        first = int(np.flatnonzero(row_mask)[0])
        if not row_mask[first:].all():
            raise ValueError(
                f"Series row_index={row_idx} contains an internal/trailing observation gap. "
                "This Group A adapter expects a continuous suffix ending at d_1387."
            )

    meta_dict = metadata.to_dict(as_series=False)
    meta_rows = [dict(zip(meta_dict.keys(), vals, strict=True)) for vals in zip(*meta_dict.values(), strict=True)]
    meta_by_index = {int(row["row_index"]): row for row in meta_rows}
    lengths_by_index = {i: int(mask_counts[i]) for i in range(len(mask_counts))}
    eligible = np.flatnonzero(mask_counts > 0).astype(np.int64)
    return sales_log1p, observation_mask, metadata, meta_rows, meta_by_index, lengths_by_index, eligible


def _make_variable_length_dataset(
    sales_log1p: np.ndarray,
    observation_mask: np.ndarray,
    row_indices: np.ndarray,
    normalize: bool = True,
):
    """Create latest-first variable-length sequences, NaN padded by tslearn.

    Original observations are a continuous suffix ending at d_1387. We reverse
    each suffix so the common endpoint is at position zero and unavailable older
    history is represented by trailing NaN padding in tslearn's representation.
    """
    from tslearn.utils import to_time_series_dataset

    sequences: list[np.ndarray] = []
    constants: list[bool] = []
    lengths: list[int] = []
    for row_idx in row_indices.tolist():
        mask = np.asarray(observation_mask[int(row_idx)], dtype=np.bool_)
        values = np.asarray(sales_log1p[int(row_idx), mask], dtype=np.float64)
        if values.size == 0:
            raise ValueError(f"No observed values for eligible row_index={row_idx}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite observed log1p values for row_index={row_idx}")
        is_constant = bool(values.size < 2 or float(np.std(values)) < 1e-8)
        if normalize:
            mean = float(np.mean(values))
            scale = float(np.std(values))
            if scale < 1e-8:
                values = np.zeros_like(values, dtype=np.float64)
            else:
                values = (values - mean) / scale
        # Latest-to-oldest: all sequences share d_1387 at their first point.
        seq = values[::-1].astype(np.float64, copy=True).reshape(-1, 1)
        sequences.append(seq)
        constants.append(is_constant)
        lengths.append(int(seq.shape[0]))
    X = to_time_series_dataset(sequences)
    if X.ndim != 3 or X.shape[0] != len(row_indices) or X.shape[2] != 1:
        raise RuntimeError(f"Unexpected tslearn variable-length dataset shape: {X.shape}")
    return X, np.asarray(constants, dtype=np.bool_), np.asarray(lengths, dtype=np.int32)


def _make_sample_split(
    eligible_rows: np.ndarray,
    meta_by_index: dict[int, dict[str, Any]],
    lengths_by_index: dict[int, int],
    train_size: int,
    validation_size: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    train_rows = _proportional_stratified_sample(
        eligible_rows, meta_by_index, lengths_by_index, train_size, rng
    )
    remaining = np.setdiff1d(eligible_rows, train_rows, assume_unique=False)
    validation_rows = _proportional_stratified_sample(
        remaining, meta_by_index, lengths_by_index, validation_size, rng
    )
    return train_rows, validation_rows


def _sample_metadata_frame(
    rows: np.ndarray,
    meta_by_index: dict[int, dict[str, Any]],
    lengths: np.ndarray,
    constants: np.ndarray,
    role: str,
) -> pl.DataFrame:
    selected = []
    for row_idx, length, constant in zip(rows.tolist(), lengths.tolist(), constants.tolist(), strict=True):
        row = dict(meta_by_index[int(row_idx)])
        row["sequence_length"] = int(length)
        row["sequence_is_constant_after_log1p"] = bool(constant)
        row["sample_role"] = role
        selected.append(row)
    return pl.DataFrame(selected)


def _silhouette_dtw(X_val, labels: np.ndarray, n_jobs: int) -> tuple[float, str | None]:
    from tslearn.clustering import silhouette_score

    distinct = np.unique(labels)
    if distinct.size < 2:
        return float("nan"), "validation assignments contain fewer than two clusters"
    if distinct.size >= len(labels):
        return float("nan"), "every validation observation has a distinct cluster label"
    try:
        score = float(silhouette_score(X_val, labels, metric="dtw", n_jobs=n_jobs))
        if not np.isfinite(score):
            return float("nan"), "tslearn returned a non-finite DTW silhouette"
        return score, None
    except Exception as exc:  # record per-replicate validation failure
        return float("nan"), f"{type(exc).__name__}: {exc}"


def _silhouette_gak(X_val, labels: np.ndarray, sigma: float, n_jobs: int) -> tuple[float, str | None]:
    from tslearn.clustering import silhouette_score
    from tslearn.metrics import cdist_gak

    distinct = np.unique(labels)
    if distinct.size < 2:
        return float("nan"), "validation assignments contain fewer than two clusters"
    if distinct.size >= len(labels):
        return float("nan"), "every validation observation has a distinct cluster label"
    try:
        kernel = cdist_gak(X_val, sigma=sigma, n_jobs=n_jobs)
        # GAK is normalized: k(x,x)=1. The induced RKHS Euclidean distance is
        # sqrt(k(x,x)+k(y,y)-2k(x,y)) = sqrt(2-2k(x,y)).
        distance = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.asarray(kernel, dtype=np.float64)))
        distance = (distance + distance.T) / 2.0
        np.fill_diagonal(distance, 0.0)
        if not np.isfinite(distance).all():
            return float("nan"), "GAK-induced distance matrix contains non-finite values"
        score = float(silhouette_score(distance, labels, metric="precomputed"))
        if not np.isfinite(score):
            return float("nan"), "tslearn returned a non-finite GAK silhouette"
        return score, None
    except Exception as exc:
        return float("nan"), f"{type(exc).__name__}: {exc}"


def _save_charts(method_dir: Path, summary: pl.DataFrame, method_key: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary.is_empty():
        return
    k = summary["k"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(k, summary["inertia_mean"].to_numpy(), yerr=summary["inertia_std"].to_numpy(), marker="o", capsize=3)
    ax.set(title=f"{method_key}: training objective by K", xlabel="Number of clusters (K)", ylabel="Training inertia / objective")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(method_dir / "k_search_objective.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(k, summary["silhouette_mean"].to_numpy(), yerr=summary["silhouette_std"].to_numpy(), marker="o", capsize=3)
    ax.axhline(0.0, linewidth=1, linestyle="--")
    ax.set(title=f"{method_key}: validation silhouette by K", xlabel="Number of clusters (K)", ylabel="Validation silhouette")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(method_dir / "k_search_silhouette.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(k, summary["ari_mean"].to_numpy(), yerr=summary["ari_std"].to_numpy(), marker="o", capsize=3)
    ax.set(title=f"{method_key}: validation assignment stability", xlabel="Number of clusters (K)", ylabel="Pairwise Adjusted Rand Index")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(method_dir / "k_search_ari_stability.png", dpi=160)
    plt.close(fig)


def _pairwise_ari(assignments_by_k: dict[int, list[tuple[int, np.ndarray]]]):
    from sklearn.metrics import adjusted_rand_score
    records = []
    for k, runs in assignments_by_k.items():
        scores = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                seed_a, labels_a = runs[i]
                seed_b, labels_b = runs[j]
                score = float(adjusted_rand_score(labels_a, labels_b))
                scores.append(score)
                records.append({"k": int(k), "seed_a": int(seed_a), "seed_b": int(seed_b), "ari": score})
        if scores:
            ari_summary = (float(np.mean(scores)), float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0, float(np.median(scores)))
        else:
            ari_summary = (float("nan"), float("nan"), float("nan"))
        yield k, ari_summary, records


def _run_dtw_kmeans(
    X_train,
    X_val,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    method_dir: Path,
    k_values: list[int],
    seeds: list[int],
    n_init: int,
    max_iter: int,
    max_iter_barycenter: int,
    n_jobs: int,
):
    from tslearn.clustering import TimeSeriesKMeans
    from sklearn.metrics import adjusted_rand_score

    method_dir.mkdir(parents=True, exist_ok=True)
    replicate_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    assignments_by_k: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    best_centroids: dict[str, np.ndarray] = {}

    for k in k_values:
        if k >= len(train_rows):
            continue
        for seed in seeds:
            started = time.perf_counter()
            record: dict[str, Any] = {"method": "tslearn_dtw_kmeans_dba", "k": k, "seed": seed, "status": "failed"}
            try:
                model = TimeSeriesKMeans(
                    n_clusters=k,
                    metric="dtw",
                    n_init=n_init,
                    max_iter=max_iter,
                    max_iter_barycenter=max_iter_barycenter,
                    random_state=seed,
                    n_jobs=n_jobs,
                    verbose=0,
                )
                model.fit(X_train)
                train_labels = np.asarray(model.labels_, dtype=np.int32)
                val_labels = np.asarray(model.predict(X_val), dtype=np.int32)
                silhouette, silhouette_error = _silhouette_dtw(X_val, val_labels, n_jobs)
                counts = np.bincount(val_labels, minlength=k)
                record.update({
                    "status": "ok",
                    "inertia": float(model.inertia_),
                    "n_iter": int(model.n_iter_),
                    "silhouette": silhouette,
                    "silhouette_error": silhouette_error,
                    "n_train_clusters": int(np.unique(train_labels).size),
                    "n_validation_clusters": int(np.unique(val_labels).size),
                    "validation_empty_clusters": int(np.sum(counts == 0)),
                    "validation_singleton_clusters": int(np.sum(counts == 1)),
                    "min_validation_cluster_size": int(counts[counts > 0].min()) if np.any(counts > 0) else 0,
                    "max_validation_cluster_size": int(counts.max()),
                    "elapsed_seconds": time.perf_counter() - started,
                })
                assignments_by_k[k].append((seed, val_labels))
                validation_rows.extend(
                    {"method": "tslearn_dtw_kmeans_dba", "k": k, "seed": seed, "row_index": int(row), "cluster": int(label)}
                    for row, label in zip(val_rows, val_labels, strict=True)
                )
                best_centroids[f"k{k}_seed{seed}"] = np.asarray(model.cluster_centers_, dtype=np.float32)
            except Exception as exc:
                record.update({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=3), "elapsed_seconds": time.perf_counter() - started})
            replicate_rows.append(record)
            print(f"  DTW-KMeans K={k:>2} seed={seed}: {record['status']} ({record.get('elapsed_seconds', 0):.1f}s)")

    pair_records: list[dict[str, Any]] = []
    pair_summaries: dict[int, tuple[float, float, float]] = {}
    for k, (mean_ari, std_ari, median_ari), records in _pairwise_ari(assignments_by_k):
        pair_records = records
        pair_summaries[k] = (mean_ari, std_ari, median_ari)
    replicate_df = pl.DataFrame(replicate_rows) if replicate_rows else pl.DataFrame()
    summaries = []
    if not replicate_df.is_empty():
        for k in k_values:
            group = replicate_df.filter((pl.col("k") == k) & (pl.col("status") == "ok"))
            if group.height == 0:
                continue
            inertias = group["inertia"].drop_nulls().to_numpy()
            inertias = inertias[np.isfinite(inertias)]
            silhouettes = group["silhouette"].drop_nulls().to_numpy()
            silhouettes = silhouettes[np.isfinite(silhouettes)]
            ari_mean, ari_std, ari_median = pair_summaries.get(k, (float("nan"),) * 3)
            summaries.append({
                "k": k,
                "valid_replicates": group.height,
                "inertia_mean": float(np.mean(inertias)) if len(inertias) else float("nan"),
                "inertia_std": float(np.std(inertias, ddof=1)) if len(inertias) > 1 else 0.0,
                "silhouette_mean": float(np.mean(silhouettes)) if len(silhouettes) else float("nan"),
                "silhouette_std": float(np.std(silhouettes, ddof=1)) if len(silhouettes) > 1 else 0.0,
                "valid_silhouette_replicates": int(len(silhouettes)),
                "ari_mean": ari_mean,
                "ari_std": ari_std,
                "ari_median": ari_median,
            })
    summary_df = pl.DataFrame(summaries) if summaries else pl.DataFrame()
    replicate_df.write_csv(method_dir / "k_search_replicates.csv")
    summary_df.write_csv(method_dir / "k_search_metrics.csv")
    pl.DataFrame(pair_records).write_csv(method_dir / "k_search_stability_pairs.csv")
    pl.DataFrame(validation_rows).write_csv(method_dir / "validation_assignments.csv")
    if best_centroids:
        np.savez_compressed(method_dir / "pilot_dba_centroids_latest_first.npz", **best_centroids)
    _save_charts(method_dir, summary_df, "DTW K-means + DBA")
    return summary_df, replicate_df


def _run_gak_kernel_kmeans(
    X_train,
    X_val,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    method_dir: Path,
    k_values: list[int],
    seeds: list[int],
    n_init: int,
    max_iter: int,
    n_jobs: int,
):
    from tslearn.clustering import KernelKMeans
    from tslearn.metrics import sigma_gak

    method_dir.mkdir(parents=True, exist_ok=True)
    # Compute the bandwidth once on the fixed training sample. Reusing it for
    # every K and seed isolates initialization/K effects from sigma variation.
    sigma = float(sigma_gak(X_train, random_state=seeds[0]))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma_gak returned an invalid bandwidth: {sigma}")

    replicate_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    assignments_by_k: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)

    for k in k_values:
        if k >= len(train_rows):
            continue
        for seed in seeds:
            started = time.perf_counter()
            record: dict[str, Any] = {"method": "tslearn_gak_kernel_kmeans", "k": k, "seed": seed, "sigma_gak": sigma, "status": "failed"}
            try:
                model = KernelKMeans(
                    n_clusters=k,
                    kernel="gak",
                    kernel_params={"sigma": sigma},
                    n_init=n_init,
                    max_iter=max_iter,
                    random_state=seed,
                    n_jobs=n_jobs,
                    verbose=0,
                )
                model.fit(X_train)
                train_labels = np.asarray(model.labels_, dtype=np.int32)
                val_labels = np.asarray(model.predict(X_val), dtype=np.int32)
                silhouette, silhouette_error = _silhouette_gak(X_val, val_labels, sigma, n_jobs)
                counts = np.bincount(val_labels, minlength=k)
                record.update({
                    "status": "ok",
                    "inertia": float(model.inertia_),
                    "n_iter": int(model.n_iter_),
                    "silhouette": silhouette,
                    "silhouette_error": silhouette_error,
                    "n_train_clusters": int(np.unique(train_labels).size),
                    "n_validation_clusters": int(np.unique(val_labels).size),
                    "validation_empty_clusters": int(np.sum(counts == 0)),
                    "validation_singleton_clusters": int(np.sum(counts == 1)),
                    "min_validation_cluster_size": int(counts[counts > 0].min()) if np.any(counts > 0) else 0,
                    "max_validation_cluster_size": int(counts.max()),
                    "elapsed_seconds": time.perf_counter() - started,
                })
                assignments_by_k[k].append((seed, val_labels))
                validation_rows.extend(
                    {"method": "tslearn_gak_kernel_kmeans", "k": k, "seed": seed, "row_index": int(row), "cluster": int(label)}
                    for row, label in zip(val_rows, val_labels, strict=True)
                )
            except Exception as exc:
                record.update({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=3), "elapsed_seconds": time.perf_counter() - started})
            replicate_rows.append(record)
            print(f"  GAK-KernelKMeans K={k:>2} seed={seed}: {record['status']} ({record.get('elapsed_seconds', 0):.1f}s)")

    pair_records: list[dict[str, Any]] = []
    pair_summaries: dict[int, tuple[float, float, float]] = {}
    for k, (mean_ari, std_ari, median_ari), records in _pairwise_ari(assignments_by_k):
        pair_records = records
        pair_summaries[k] = (mean_ari, std_ari, median_ari)
    replicate_df = pl.DataFrame(replicate_rows) if replicate_rows else pl.DataFrame()
    summaries = []
    if not replicate_df.is_empty():
        for k in k_values:
            group = replicate_df.filter((pl.col("k") == k) & (pl.col("status") == "ok"))
            if group.height == 0:
                continue
            inertias = group["inertia"].drop_nulls().to_numpy()
            inertias = inertias[np.isfinite(inertias)]
            silhouettes = group["silhouette"].drop_nulls().to_numpy()
            silhouettes = silhouettes[np.isfinite(silhouettes)]
            ari_mean, ari_std, ari_median = pair_summaries.get(k, (float("nan"),) * 3)
            summaries.append({
                "k": k,
                "valid_replicates": group.height,
                "inertia_mean": float(np.mean(inertias)) if len(inertias) else float("nan"),
                "inertia_std": float(np.std(inertias, ddof=1)) if len(inertias) > 1 else 0.0,
                "silhouette_mean": float(np.mean(silhouettes)) if len(silhouettes) else float("nan"),
                "silhouette_std": float(np.std(silhouettes, ddof=1)) if len(silhouettes) > 1 else 0.0,
                "valid_silhouette_replicates": int(len(silhouettes)),
                "ari_mean": ari_mean,
                "ari_std": ari_std,
                "ari_median": ari_median,
            })
    summary_df = pl.DataFrame(summaries) if summaries else pl.DataFrame()
    replicate_df.write_csv(method_dir / "k_search_replicates.csv")
    summary_df.write_csv(method_dir / "k_search_metrics.csv")
    pl.DataFrame(pair_records).write_csv(method_dir / "k_search_stability_pairs.csv")
    pl.DataFrame(validation_rows).write_csv(method_dir / "validation_assignments.csv")
    _save_charts(method_dir, summary_df, "GAK Kernel K-means")
    _write_json(method_dir / "gak_manifest.json", {
        "method": "tslearn.clustering.KernelKMeans(kernel='gak')",
        "tslearn_version": _package_version("tslearn"),
        "sigma_gak": sigma,
        "sigma_estimation": "Computed once on the fixed training sample and reused for all K and seeds.",
        "note": "KernelKMeans uses an in-memory training Gram matrix. This stage is deliberately pilot-sized; do not extrapolate to full-population fit without a separate memory/runtime assessment.",
    })
    return summary_df, replicate_df


def _package_version(package: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _smoke_test() -> None:
    from tslearn.clustering import KernelKMeans, TimeSeriesKMeans, silhouette_score
    from tslearn.metrics import cdist_gak, sigma_gak
    from tslearn.utils import to_time_series_dataset

    rng = np.random.default_rng(7)
    base_a = np.sin(np.linspace(0, 4 * np.pi, 64))
    base_b = np.sign(np.sin(np.linspace(0, 4 * np.pi, 64)))
    train = []
    for i in range(12):
        wave = base_a if i % 2 == 0 else base_b
        length = 64 - (i % 4) * 8
        series = wave[:length] + rng.normal(0, 0.025, size=length)
        train.append(series[::-1].reshape(-1, 1))
    val = []
    for i in range(6):
        wave = base_a if i % 2 == 0 else base_b
        length = 64 - (i % 3) * 8
        series = wave[:length] + rng.normal(0, 0.025, size=length)
        val.append(series[::-1].reshape(-1, 1))
    X_train = to_time_series_dataset(train)
    X_val = to_time_series_dataset(val)
    if not np.isnan(X_train).any() or not np.isnan(X_val).any():
        raise RuntimeError("Smoke test expected NaN padding for unequal lengths")

    dtw_model = TimeSeriesKMeans(
        n_clusters=2, metric="dtw", n_init=1, max_iter=3,
        max_iter_barycenter=2, random_state=7, n_jobs=1,
    ).fit(X_train)
    dtw_labels = dtw_model.predict(X_val)
    dtw_silhouette = float(silhouette_score(X_val, dtw_labels, metric="dtw"))

    sigma = float(sigma_gak(X_train, random_state=7))
    gak_model = KernelKMeans(
        n_clusters=2, kernel="gak", kernel_params={"sigma": sigma},
        n_init=2, max_iter=5, random_state=7, n_jobs=1,
    ).fit(X_train)
    gak_labels = gak_model.predict(X_val)
    val_kernel = cdist_gak(X_val, sigma=sigma, n_jobs=1)
    val_distance = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.asarray(val_kernel)))
    gak_silhouette = float(silhouette_score(val_distance, gak_labels, metric="precomputed"))

    if np.unique(dtw_model.labels_).size < 2 or np.unique(gak_model.labels_).size < 2:
        raise RuntimeError("Smoke test produced fewer than two training clusters")
    print("Group A tslearn smoke test passed")
    print(f"  tslearn version: {_package_version('tslearn')}")
    print(f"  variable-length training shape: {X_train.shape}")
    print(f"  variable-length validation shape: {X_val.shape}")
    print(f"  trailing NaN padding: {int(np.isnan(X_train).sum() + np.isnan(X_val).sum())} cells")
    print(f"  DTW KMeans + DBA: train clusters={np.unique(dtw_model.labels_).size}, validation silhouette={dtw_silhouette:.6f}")
    print(f"  GAK KernelKMeans: sigma={sigma:.6g}, train clusters={np.unique(gak_model.labels_).size}, validation silhouette={gak_silhouette:.6f}")
    print("  Both estimators fitted and predicted with unequal-length NaN-padded inputs.")


def _run_search(args: argparse.Namespace) -> None:
    input_dir = args.input_dir
    output_root = args.output_dir
    (
        sales_log1p, observation_mask, metadata, meta_rows,
        meta_by_index, lengths_by_index, eligible_rows,
    ) = _load_prepared_data(input_dir)

    method_config = {}
    if args.method in {"dtw-kmeans", "both"}:
        method_config["dtw-kmeans"] = (args.dtw_train_size, args.dtw_validation_size)
    if args.method in {"gak-kernel-kmeans", "both"}:
        method_config["gak-kernel-kmeans"] = (args.gak_train_size, args.gak_validation_size)

    overall_manifest = {
        "script": "run_m5_group_a_tslearn.py",
        "stage": "search",
        "window": {"start_d": WINDOW_START_D, "end_d": WINDOW_END_D, "n_days": WINDOW_DAYS},
        "n_series_total": int(sales_log1p.shape[0]),
        "n_series_with_observations": int(len(eligible_rows)),
        "n_series_without_observation_history": int(sales_log1p.shape[0] - len(eligible_rows)),
        "sequence_representation": {
            "source": "sales_log1p.npy selected where observation_mask is true",
            "normalization": "per-series z-score over the entire observed suffix; constant sequences become zero vectors",
            "orientation": "latest day d_1387 first, proceeding backwards in time",
            "padding": "tslearn.to_time_series_dataset trailing NaN padding; no imputation",
            "history_lengths": "all observed lengths retained; no minimum-history cut-off",
        },
        "methods": {},
        "candidate_k": args.k_values,
        "seeds": args.seeds,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "max_iter_barycenter": args.max_iter_barycenter,
        "n_jobs": args.n_jobs,
        "package_versions": {
            "tslearn": _package_version("tslearn"),
            "numpy": _package_version("numpy"),
            "polars": _package_version("polars"),
            "scikit_learn": _package_version("scikit-learn"),
        },
        "note": "Pilot K search only. Not a full-population clustering fit.",
    }

    for method, (train_size, val_size) in method_config.items():
        method_dir = output_root / ("dtw_kmeans_dba" if method == "dtw-kmeans" else "gak_kernel_kmeans")
        train_rows, val_rows = _make_sample_split(
            eligible_rows, meta_by_index, lengths_by_index,
            train_size, val_size, args.random_state,
        )
        X_train, train_constants, train_lengths = _make_variable_length_dataset(
            sales_log1p, observation_mask, train_rows, normalize=True
        )
        X_val, val_constants, val_lengths = _make_variable_length_dataset(
            sales_log1p, observation_mask, val_rows, normalize=True
        )
        if args.method == "both":
            # Use distinct sample sizes, but record independent stratified samples.
            # Different methods are assessed on their own merits, not direct label agreement.
            pass
        sample_dir = method_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        _sample_metadata_frame(train_rows, meta_by_index, train_lengths, train_constants, "train").write_parquet(sample_dir / "train_sample_metadata.parquet")
        _sample_metadata_frame(val_rows, meta_by_index, val_lengths, val_constants, "validation").write_parquet(sample_dir / "validation_sample_metadata.parquet")
        sample_manifest = {
            "method": method,
            "train_n": int(len(train_rows)),
            "validation_n": int(len(val_rows)),
            "train_row_indices": train_rows.tolist(),
            "validation_row_indices": val_rows.tolist(),
            "train_history_length_min_median_max": [int(train_lengths.min()), float(np.median(train_lengths)), int(train_lengths.max())],
            "validation_history_length_min_median_max": [int(val_lengths.min()), float(np.median(val_lengths)), int(val_lengths.max())],
            "train_constant_sequences": int(train_constants.sum()),
            "validation_constant_sequences": int(val_constants.sum()),
            "stratification": ["store_id", "cat_id", "history_length_band", "activity_label"],
        }
        _write_json(sample_dir / "sample_manifest.json", sample_manifest)

        print(f"\nRunning {method}")
        print(f"  Train: {len(train_rows)}; validation: {len(val_rows)}")
        print(f"  Training sequence tensor: {X_train.shape}; validation: {X_val.shape}")
        if method == "dtw-kmeans":
            summary, replicates = _run_dtw_kmeans(
                X_train, X_val, train_rows, val_rows, method_dir,
                args.k_values, args.seeds, args.n_init, args.max_iter,
                args.max_iter_barycenter, args.n_jobs,
            )
        else:
            summary, replicates = _run_gak_kernel_kmeans(
                X_train, X_val, train_rows, val_rows, method_dir,
                args.k_values, args.seeds, args.n_init, args.max_iter,
                args.n_jobs,
            )
        overall_manifest["methods"][method] = {
            "train_size": train_size,
            "validation_size": val_size,
            "summary_rows": int(summary.height),
            "replicate_rows": int(replicates.height),
            "output_directory": str(method_dir),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "group_a_search_manifest.json", overall_manifest)
    print(f"\nGroup A exploratory search complete. Results: {output_root}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Group A: variable-length tslearn clustering experiments for M5")
    parser.add_argument("--stage", choices=["smoke-test", "search"], default="smoke-test")
    parser.add_argument("--method", choices=["dtw-kmeans", "gak-kernel-kmeans", "both"], default="both")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dtw-train-size", type=int, default=600)
    parser.add_argument("--dtw-validation-size", type=int, default=300)
    parser.add_argument("--gak-train-size", type=int, default=150)
    parser.add_argument("--gak-validation-size", type=int, default=100)
    parser.add_argument("--k-values", type=_parse_int_list, default=DEFAULT_K_VALUES)
    parser.add_argument("--seeds", type=_parse_int_list, default=DEFAULT_SEEDS)
    parser.add_argument("--n-init", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--max-iter-barycenter", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=2)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.input_dir = args.input_dir or (args.project_root / "data/processed/clustering/d1387_728d")
    args.output_dir = args.output_dir or (args.input_dir / "group_a_tslearn")
    print("M5 Group A: variable-length tslearn clustering")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Method: {args.method}")
    if args.stage == "smoke-test":
        _smoke_test()
    else:
        if args.dtw_train_size < 20 or args.dtw_validation_size < 10:
            raise ValueError("DTW pilot requires at least 20 train and 10 validation series")
        if args.gak_train_size < 20 or args.gak_validation_size < 10:
            raise ValueError("GAK pilot requires at least 20 train and 10 validation series")
        _run_search(args)


if __name__ == "__main__":
    main()
