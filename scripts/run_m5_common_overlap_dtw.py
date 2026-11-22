#!/usr/bin/env python3
"""Common-overlap DTW clustering for the prepared M5 item-store series.

This experiment deliberately avoids full-sequence DTW on unequal history lengths.
For every pair, it compares only the latest calendar interval observed for BOTH
series, normalizes each series within that pairwise common interval, and computes
Sakoe-Chiba-constrained DTW. The returned dissimilarity is normalized by the
square root of the DTW path length (RMS local discrepancy), making it less
length-dependent than the raw cumulative DTW cost.

Clustering uses a CLARA-style sample-and-assign K-medoids workflow. No full
29,359 x 29,359 distance matrix is constructed.

Run from the M5 project root:
    pixi run python scripts/run_m5_common_overlap_dtw.py --stage smoke-test
    pixi run python scripts/run_m5_common_overlap_dtw.py --stage search
    pixi run python scripts/run_m5_common_overlap_dtw.py --stage stability --candidate-k-values 5,7,9
    pixi run python scripts/run_m5_common_overlap_dtw.py --stage fit --n-clusters 7
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score

WINDOW_START_D = 660
WINDOW_END_D = 1387
WINDOW_DAYS = WINDOW_END_D - WINDOW_START_D + 1
EXPECTED_N_SERIES = 30_490
DEFAULT_RADIUS = 7
DEFAULT_MIN_OVERLAP_DAYS = 56  # Eight complete weekly cycles; configurable.
DEFAULT_K_VALUES = [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20]
DEFAULT_SEEDS = [42, 43, 44]


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_int_list(raw: str, name: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} must not contain duplicates")
    return values


def _import_tslearn():
    try:
        from tslearn.clustering import silhouette_score as tslearn_silhouette_score
        from tslearn.metrics import dtw_path
    except ImportError as exc:
        raise RuntimeError(
            "tslearn is required for common-overlap DTW. Install it in the active Pixi environment:\n"
            '  pixi add --pypi "tslearn==0.10.0" --environment gam-rust\n'
            "  pixi install"
        ) from exc
    return dtw_path, tslearn_silhouette_score


def _load_inputs(input_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    required = [
        input_dir / "sales_units.npy",
        input_dir / "sales_log1p.npy",
        input_dir / "observation_mask.npy",
        input_dir / "series_metadata.parquet",
        input_dir / "calendar_window.parquet",
        input_dir / "manifest.json",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared clustering input(s):\n  " + "\n  ".join(missing)
            + "\nRun prepare_m5_clustering_data.py first."
        )

    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    window = manifest.get("window", {})
    if (window.get("start_d"), window.get("end_d"), window.get("n_days")) != (
        WINDOW_START_D, WINDOW_END_D, WINDOW_DAYS
    ):
        raise ValueError("Input manifest does not describe d_660..d_1387 (728 days)")

    units = np.load(input_dir / "sales_units.npy", mmap_mode="r")
    x = np.load(input_dir / "sales_log1p.npy", mmap_mode="r")
    mask = np.load(input_dir / "observation_mask.npy", mmap_mode="r")
    metadata = pl.read_parquet(input_dir / "series_metadata.parquet").sort("row_index")
    calendar = pl.read_parquet(input_dir / "calendar_window.parquet").sort("d_num")

    expected_shape = (EXPECTED_N_SERIES, WINDOW_DAYS)
    if units.shape != expected_shape or x.shape != expected_shape or mask.shape != expected_shape:
        raise ValueError(
            f"Expected all arrays to have shape {expected_shape}; got units={units.shape}, "
            f"log1p={x.shape}, mask={mask.shape}"
        )
    if mask.dtype != np.bool_:
        raise TypeError(f"Expected bool observation mask, received {mask.dtype}")
    if metadata.height != EXPECTED_N_SERIES or "row_index" not in metadata.columns:
        raise ValueError("Metadata must have 30,490 rows and a row_index column")
    if not np.array_equal(metadata["row_index"].to_numpy(), np.arange(EXPECTED_N_SERIES)):
        raise ValueError("series_metadata.row_index must be exactly 0..30,489")
    required_meta = {"n_price_live_days", "activity_label", "coverage_label"}
    if not required_meta.issubset(metadata.columns):
        raise ValueError(f"Metadata is missing required columns: {sorted(required_meta - set(metadata.columns))}")
    if calendar.height != WINDOW_DAYS or calendar["d_num"].to_list() != list(range(WINDOW_START_D, WINDOW_END_D + 1)):
        raise ValueError("calendar_window must contain ordered d_660..d_1387 rows")

    lengths = np.asarray(mask.sum(axis=1), dtype=np.int32)
    metadata_lengths = metadata["n_price_live_days"].to_numpy().astype(np.int32)
    if not np.array_equal(lengths, metadata_lengths):
        raise ValueError("Observation-mask counts do not match metadata.n_price_live_days")

    # Observed histories must be continuous suffixes ending at d_1387.
    if np.any(mask[:, :-1] & ~mask[:, 1:]):
        examples = np.argwhere(mask[:, :-1] & ~mask[:, 1:])[:10].tolist()
        raise ValueError(
            "Internal/trailing no-price gaps found. Common-overlap cropping currently assumes "
            f"continuous observed suffixes. First [row_index, day_offset] examples: {examples}"
        )
    if np.any((lengths > 0) & ~mask[:, -1]):
        raise ValueError("Every series with observed history must be observed on d_1387")

    # Validate log1p semantics in blocks. Unobserved entries must remain NaN.
    block_size = 1024
    for start in range(0, EXPECTED_N_SERIES, block_size):
        end = min(start + block_size, EXPECTED_N_SERIES)
        xb = np.asarray(x[start:end])
        mb = np.asarray(mask[start:end])
        if np.any(~np.isfinite(xb[mb])) or np.any(xb[mb] < 0):
            raise ValueError(f"Invalid log1p sales values in rows {start}:{end}")
        if np.any(~np.isnan(xb[~mb])):
            raise ValueError(f"Expected NaN at all unobserved positions in rows {start}:{end}")

    return units, x, mask, metadata, calendar, manifest


def _build_raw_suffixes(
    x: np.ndarray,
    lengths: np.ndarray,
) -> tuple[list[np.ndarray | None], np.ndarray, np.ndarray, np.ndarray]:
    """Build chronological raw log1p observed suffixes and full-history diagnostics.

    Pairwise normalization is deliberately deferred until the common calendar
    interval for each pair is known. This prevents unshared older observations
    from influencing either the alignment or the pairwise shape scaling.
    """
    sequences: list[np.ndarray | None] = [None] * len(lengths)
    means = np.full(len(lengths), np.nan, dtype=np.float32)
    scales = np.full(len(lengths), np.nan, dtype=np.float32)
    constant = np.zeros(len(lengths), dtype=np.bool_)
    for row_index, length_value in enumerate(lengths):
        length = int(length_value)
        if length == 0:
            continue
        start = WINDOW_DAYS - length
        values = np.asarray(x[row_index, start:], dtype=np.float32)
        if values.size != length or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid observed suffix for row_index={row_index}")
        mean = float(values.mean(dtype=np.float64))
        scale = float(values.std(dtype=np.float64))
        means[row_index] = mean
        scales[row_index] = scale
        if scale <= 1e-8:
            constant[row_index] = True
        sequences[row_index] = np.ascontiguousarray(values, dtype=np.float32)
    return sequences, means, scales, constant


def _pairwise_common_overlap_dtw(
    row_i: int,
    row_j: int,
    sequences: list[np.ndarray | None],
    radius: int,
    dtw_path,
) -> tuple[float, int]:
    """Compare only the common latest calendar suffix for a pair of M5 series.

    The pair is cropped to its latest common calendar interval first. Each
    cropped segment is then z-normalized independently. Dissimilarity is
    DTW_RMS = raw_DTW / sqrt(path_length).
    """
    a = sequences[int(row_i)]
    b = sequences[int(row_j)]
    if a is None or b is None:
        raise ValueError("Cannot calculate demand similarity without observed history")
    overlap = min(a.size, b.size)
    if overlap < 1:
        raise ValueError("Common overlap must contain at least one observation")
    aa = np.ascontiguousarray(a[-overlap:], dtype=np.float64)
    bb = np.ascontiguousarray(b[-overlap:], dtype=np.float64)
    # Pair-specific z-normalization: only the common calendar interval affects
    # the scale and shape comparison. Constant overlap segments map to zeros.
    aa_sd = float(aa.std())
    bb_sd = float(bb.std())
    if aa_sd <= 1e-8:
        aa = np.zeros_like(aa)
    else:
        aa = (aa - aa.mean()) / aa_sd
    if bb_sd <= 1e-8:
        bb = np.zeros_like(bb)
    else:
        bb = (bb - bb.mean()) / bb_sd
    path, raw_distance = dtw_path(
        aa,
        bb,
        global_constraint="sakoe_chiba",
        sakoe_chiba_radius=int(radius),
    )
    path_length = len(path)
    if path_length < 1 or not math.isfinite(float(raw_distance)):
        raise RuntimeError(f"Invalid DTW path/distance for rows {row_i}, {row_j}")
    return float(raw_distance) / math.sqrt(path_length), int(overlap)


def _pairwise_distance_matrix(
    row_ids: np.ndarray,
    sequences: list[np.ndarray | None],
    radius: int,
    n_jobs: int,
    dtw_path,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute symmetric sample dissimilarity and overlap matrices."""
    n = int(row_ids.size)
    if n < 2:
        raise ValueError("At least two series are required for pairwise distances")
    D = np.zeros((n, n), dtype=np.float32)
    O = np.zeros((n, n), dtype=np.int16)

    def compute_row(i: int):
        vals = np.empty(n - i - 1, dtype=np.float32)
        ovs = np.empty(n - i - 1, dtype=np.int16)
        row_i = int(row_ids[i])
        for offset, j in enumerate(range(i + 1, n)):
            distance, overlap = _pairwise_common_overlap_dtw(
                row_i, int(row_ids[j]), sequences, radius, dtw_path
            )
            vals[offset] = distance
            ovs[offset] = overlap
        return i, vals, ovs

    print(f"  Computing {n * (n - 1) // 2:,} pairwise common-overlap DTW distances ({label})...")
    start_time = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(compute_row)(i) for i in range(n - 1)
    )
    for i, vals, ovs in results:
        D[i, i + 1 :] = vals
        D[i + 1 :, i] = vals
        O[i, i + 1 :] = ovs
        O[i + 1 :, i] = ovs
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(O, np.asarray([len(sequences[int(r)]) for r in row_ids], dtype=np.int16))
    if not np.all(np.isfinite(D)) or not np.allclose(D, D.T, atol=1e-6):
        raise RuntimeError("Pairwise DTW dissimilarity matrix failed finite/symmetry validation")
    if np.any(D < 0) or np.any(O <= 0):
        raise RuntimeError("Pairwise DTW matrix contains negative distances or empty overlaps")
    print(f"  Completed in {(time.perf_counter() - start_time) / 60:.2f} minutes")
    return D, O


def _cross_distance_matrix(
    query_rows: np.ndarray,
    medoid_rows: np.ndarray,
    sequences: list[np.ndarray | None],
    radius: int,
    n_jobs: int,
    dtw_path,
    block_size: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute query-to-medoid common-overlap DTW in blocks."""
    n_query = int(query_rows.size)
    k = int(medoid_rows.size)
    D = np.empty((n_query, k), dtype=np.float32)
    O = np.empty((n_query, k), dtype=np.int16)
    starts = list(range(0, n_query, block_size))

    def compute_block(start: int):
        stop = min(start + block_size, n_query)
        block_d = np.empty((stop - start, k), dtype=np.float32)
        block_o = np.empty((stop - start, k), dtype=np.int16)
        for local_i, row_i in enumerate(query_rows[start:stop]):
            for j, medoid in enumerate(medoid_rows):
                dist, overlap = _pairwise_common_overlap_dtw(
                    int(row_i), int(medoid), sequences, radius, dtw_path
                )
                block_d[local_i, j] = dist
                block_o[local_i, j] = overlap
        return start, stop, block_d, block_o

    print(f"  Assigning {n_query:,} series to {k} medoids ({n_query * k:,} DTW comparisons)...")
    start_time = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(compute_block)(start) for start in starts
    )
    for start, stop, block_d, block_o in results:
        D[start:stop] = block_d
        O[start:stop] = block_o
    if not np.all(np.isfinite(D)):
        raise RuntimeError("Non-finite query-to-medoid distances encountered")
    print(f"  Assignment distances completed in {(time.perf_counter() - start_time) / 60:.2f} minutes")
    return D, O


def _length_bins(lengths: np.ndarray) -> np.ndarray:
    # All sampled rows meet the minimum-history floor. Bins describe length only.
    return np.digitize(lengths, bins=np.array([90, 180, 364, 547], dtype=np.int32), right=True)


def _stratified_sample(
    candidate_rows: np.ndarray,
    sample_size: int,
    lengths_all: np.ndarray,
    activity_all: np.ndarray,
    seed: int,
    excluded_rows: np.ndarray | None = None,
) -> np.ndarray:
    candidates = np.asarray(candidate_rows, dtype=np.int64)
    if excluded_rows is not None and excluded_rows.size:
        candidates = candidates[~np.isin(candidates, excluded_rows)]
    if sample_size >= candidates.size:
        return candidates.copy()
    if sample_size < 2:
        raise ValueError("Sample size must be at least 2")
    rng = np.random.default_rng(seed)
    bins = _length_bins(lengths_all[candidates])
    activities = activity_all[candidates].astype(str)
    strata: dict[tuple[int, str], np.ndarray] = {}
    for b in np.unique(bins):
        for activity in np.unique(activities):
            members = candidates[(bins == b) & (activities == activity)]
            if members.size:
                strata[(int(b), str(activity))] = members
    keys = list(strata)
    n_total = candidates.size
    exact = np.asarray([sample_size * strata[k].size / n_total for k in keys], dtype=np.float64)
    quotas = np.floor(exact).astype(np.int64)
    remainder = exact - quotas
    if sample_size >= len(keys):
        for idx in range(len(keys)):
            if quotas[idx] == 0:
                quotas[idx] = 1
    caps = np.asarray([strata[k].size for k in keys], dtype=np.int64)
    quotas = np.minimum(quotas, caps)
    while int(quotas.sum()) < sample_size:
        available = np.flatnonzero(quotas < caps)
        if available.size == 0:
            break
        idx = available[int(np.argmax(remainder[available]))]
        quotas[idx] += 1
        remainder[idx] = -1
    while int(quotas.sum()) > sample_size:
        available = np.flatnonzero(quotas > 1)
        if available.size == 0:
            available = np.flatnonzero(quotas > 0)
        idx = available[int(np.argmin(remainder[available]))]
        quotas[idx] -= 1
        remainder[idx] = 0
    selected: list[int] = []
    for key, quota in zip(keys, quotas, strict=True):
        if quota:
            selected.extend(rng.choice(strata[key], size=int(quota), replace=False).tolist())
    result = np.asarray(selected, dtype=np.int64)
    rng.shuffle(result)
    if result.size != min(sample_size, candidates.size):
        raise RuntimeError(f"Stratified sampler returned {result.size}, expected {min(sample_size, candidates.size)}")
    return result


def _kmedoids_plus_plus(D: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = D.shape[0]
    medoids = [int(rng.integers(n))]
    closest = D[:, medoids[0]].astype(np.float64, copy=True)
    while len(medoids) < k:
        probs = np.square(closest)
        probs[np.asarray(medoids, dtype=np.int64)] = 0.0
        total = float(probs.sum())
        if total <= 1e-12:
            remaining = np.setdiff1d(np.arange(n), np.asarray(medoids), assume_unique=False)
            chosen = int(rng.choice(remaining))
        else:
            chosen = int(rng.choice(n, p=probs / total))
            if chosen in medoids:
                remaining = np.setdiff1d(np.arange(n), np.asarray(medoids), assume_unique=False)
                chosen = int(rng.choice(remaining))
        medoids.append(chosen)
        closest = np.minimum(closest, D[:, chosen])
    return np.asarray(medoids, dtype=np.int64)


def _assign_matrix(D: np.ndarray, medoids: np.ndarray, force_medoid_ownership: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = D[:, medoids]
    labels = np.argmin(selected, axis=1).astype(np.int32)
    nearest = selected[np.arange(D.shape[0]), labels].astype(np.float32)
    if selected.shape[1] > 1:
        two = np.partition(selected, kth=1, axis=1)[:, :2]
        first_second = np.sort(two, axis=1)
        nearest = first_second[:, 0].astype(np.float32)
        second = first_second[:, 1].astype(np.float32)
    else:
        second = np.full(D.shape[0], np.nan, dtype=np.float32)
    if force_medoid_ownership:
        for cluster, medoid_position in enumerate(medoids):
            labels[int(medoid_position)] = int(cluster)
            nearest[int(medoid_position)] = 0.0
            if medoids.size > 1:
                row = selected[int(medoid_position)].copy()
                row[cluster] = np.inf
                second[int(medoid_position)] = float(np.min(row))
    return labels, nearest, second


def _fit_kmedoids(
    D: np.ndarray,
    n_clusters: int,
    random_state: int,
    n_init: int,
    max_iter: int,
) -> dict[str, Any]:
    """Alternate assignment and within-cluster medoid updates on precomputed D."""
    n = D.shape[0]
    if not 2 <= n_clusters < n:
        raise ValueError(f"n_clusters must be in [2, {n - 1}], got {n_clusters}")
    rng = np.random.default_rng(random_state)
    best: dict[str, Any] | None = None

    for init in range(n_init):
        medoids = _kmedoids_plus_plus(D, n_clusters, rng)
        n_iter = 0
        converged = False
        for iteration in range(max_iter):
            labels, nearest, _ = _assign_matrix(D, medoids, force_medoid_ownership=True)
            updated = medoids.copy()
            for cluster in range(n_clusters):
                members = np.flatnonzero(labels == cluster)
                if members.size == 0:
                    continue
                within = D[np.ix_(members, members)]
                updated[cluster] = int(members[np.argmin(within.sum(axis=1, dtype=np.float64))])
            n_iter = iteration + 1
            if np.array_equal(updated, medoids):
                converged = True
                break
            medoids = updated
        labels, nearest, second = _assign_matrix(D, medoids, force_medoid_ownership=True)
        objective = float(nearest.sum(dtype=np.float64))
        result = {
            "medoid_positions": medoids,
            "labels": labels,
            "nearest_distances": nearest,
            "second_distances": second,
            "objective_sum": objective,
            "objective_mean": objective / n,
            "n_iter": n_iter,
            "converged": converged,
            "initialization": init,
        }
        if best is None or objective < best["objective_sum"]:
            best = result
    assert best is not None
    return best


def _validation_silhouette(D_val: np.ndarray, labels: np.ndarray, silhouette_score) -> float:
    unique = np.unique(labels)
    if unique.size < 2 or unique.size >= labels.size:
        return float("nan")
    try:
        return float(silhouette_score(D_val, labels, metric="precomputed"))
    except Exception as exc:
        print(f"  Warning: silhouette unavailable: {exc}")
        return float("nan")


def _sample_manifest(metadata: pl.DataFrame, row_ids: np.ndarray, lengths: np.ndarray, sample_role: str) -> pl.DataFrame:
    return (
        pl.DataFrame(
            {
                "sample_position": np.arange(row_ids.size, dtype=np.int32),
                "row_index": row_ids.astype(np.int64),
                "sequence_length_days": lengths[row_ids].astype(np.int32),
                "sample_role": [sample_role] * row_ids.size,
            }
        )
        .join(metadata, on="row_index", how="left")
        .sort("sample_position")
    )


def _write_search_plots(metrics: pl.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    k = metrics["k"].to_numpy()
    cost = metrics["validation_mean_nearest_dtw_rms"].to_numpy()
    silhouette = metrics["validation_silhouette_mean"].to_numpy()
    silhouette_sd = metrics["validation_silhouette_sd"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k, cost, marker="o")
    ax.set_title("Common-overlap DTW medoid cost (validation sample)")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Mean nearest-medoid DTW-RMS dissimilarity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "common_overlap_dtw_elbow.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(k, silhouette, yerr=silhouette_sd, marker="o", capsize=3)
    ax.set_title("Silhouette on fixed validation sample")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Mean silhouette (common-overlap DTW-RMS)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "common_overlap_dtw_silhouette.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(k, cost, marker="o")
    axes[0].set_title("Validation medoid cost")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Mean nearest-medoid DTW-RMS")
    axes[0].grid(alpha=0.25)
    axes[1].errorbar(k, silhouette, yerr=silhouette_sd, marker="o", capsize=3)
    axes[1].set_title("Validation silhouette")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Common-overlap DTW: choose K from multiple diagnostics")
    fig.tight_layout()
    fig.savefig(output_dir / "common_overlap_dtw_k_selection.png", dpi=160)
    plt.close(fig)


def _load_series_context(input_dir: Path):
    units, x, mask, metadata, calendar, input_manifest = _load_inputs(input_dir)
    lengths = np.asarray(mask.sum(axis=1), dtype=np.int32)
    sequences, means, scales, constant = _build_raw_suffixes(x, lengths)
    return units, sequences, lengths, means, scales, constant, metadata, calendar, input_manifest


def _run_smoke_test(radius: int) -> None:
    dtw_path, silhouette_score = _import_tslearn()
    rng = np.random.default_rng(2026)
    # A long series and a short series share the exact same latest 64 points.
    shared = np.sin(np.linspace(0, 4 * np.pi, 64)).astype(np.float32)
    older = (3.0 + np.cos(np.linspace(0, 3 * np.pi, 32))).astype(np.float32)
    short = shared.copy()
    long = np.concatenate([older, shared]).astype(np.float32)
    # The distance helper performs pair-specific normalization after cropping.
    d_common, overlap = _pairwise_common_overlap_dtw(0, 1, [long, short], radius, dtw_path)
    if overlap != 64 or d_common > 1e-5:
        raise AssertionError(f"Common-overlap identity test failed: overlap={overlap}, distance={d_common}")

    # Symmetry and a non-identical sequence test.
    a = rng.normal(size=80).astype(np.float32)
    b = np.roll(a, 3).astype(np.float32)
    d_ab, ov_ab = _pairwise_common_overlap_dtw(2, 3, [long, short, a, b], radius, dtw_path)
    d_ba, ov_ba = _pairwise_common_overlap_dtw(3, 2, [long, short, a, b], radius, dtw_path)
    if not np.isclose(d_ab, d_ba, atol=1e-7) or ov_ab != ov_ba:
        raise AssertionError("Common-overlap DTW symmetry test failed")

    toy = np.asarray(
        [
            [0.0, 0.1, 0.2, 3.0, 3.1, 3.2],
            [0.1, 0.0, 0.1, 3.1, 3.0, 3.1],
            [0.2, 0.1, 0.0, 3.2, 3.1, 3.0],
            [3.0, 3.1, 3.2, 0.0, 0.1, 0.2],
            [3.1, 3.0, 3.1, 0.1, 0.0, 0.1],
            [3.2, 3.1, 3.0, 0.2, 0.1, 0.0],
        ], dtype=np.float32,
    )
    toy_labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
    toy_silhouette = float(silhouette_score(toy, toy_labels, metric="precomputed"))
    if toy_silhouette <= 0:
        raise AssertionError("Precomputed-distance silhouette smoke test failed")
    print("Common-overlap DTW smoke test passed")
    print(f"  tslearn version: {_version('tslearn')}")
    print(f"  Common overlap in nested-history test: {overlap} days")
    print(f"  Nested-history common-overlap DTW-RMS: {d_common:.8f}")
    print(f"  Symmetric distance test: {d_ab:.8f} / {d_ba:.8f} ({ov_ab} common days)")
    print(f"  Silhouette from precomputed dissimilarities: {toy_silhouette:.6f}")
    print(f"  Sakoe-Chiba radius: {radius} days")
    print("  Pairwise normalization: each series z-scored within the pairwise common interval")
    print("  No unobserved prefix is imputed or compared")


def _stage_search(args, input_dir: Path, output_root: Path) -> None:
    dtw_path, silhouette_score = _import_tslearn()
    units, sequences, lengths, means, scales, constant, metadata, calendar, input_manifest = _load_series_context(input_dir)
    fit_rows = np.flatnonzero(lengths >= args.min_overlap_days).astype(np.int64)
    no_history_rows = np.flatnonzero(lengths == 0).astype(np.int64)
    short_rows = np.flatnonzero((lengths > 0) & (lengths < args.min_overlap_days)).astype(np.int64)
    if fit_rows.size < max(args.k_values) + 2:
        raise ValueError("Not enough series meet min_overlap_days for the requested K grid")
    print("Common-overlap DTW K search")
    print(f"  Observable series: {(lengths > 0).sum():,}")
    print(f"  Fit-eligible series (>= {args.min_overlap_days} days): {fit_rows.size:,}")
    print(f"  Short-history series retained but not used to fit medoids: {short_rows.size:,}")
    print(f"  No-history series: {no_history_rows.size:,}")

    search_dir = output_root / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    activity_all = metadata["activity_label"].to_numpy().astype(str)
    val_size = min(args.validation_size, max(2, fit_rows.size // 3))
    validation_rows = _stratified_sample(
        fit_rows, val_size, lengths, activity_all, args.random_state + 9000
    )
    D_val, O_val = _pairwise_distance_matrix(
        validation_rows, sequences, args.radius, args.n_jobs, dtw_path, "fixed validation sample"
    )
    val_manifest = _sample_manifest(metadata, validation_rows, lengths, "validation")

    replicate_records: list[dict[str, Any]] = []
    for replicate in range(args.clara_replicates):
        seed = args.random_state + 101 * replicate
        train_rows = _stratified_sample(
            fit_rows, args.train_sample_size, lengths, activity_all, seed,
            excluded_rows=validation_rows,
        )
        if train_rows.size <= max(args.k_values):
            raise ValueError("Training sample is too small for the requested K grid")
        train_manifest = _sample_manifest(metadata, train_rows, lengths, f"train_replicate_{replicate + 1}")
        D_train, O_train = _pairwise_distance_matrix(
            train_rows, sequences, args.radius, args.n_jobs, dtw_path,
            f"CLARA training sample {replicate + 1}/{args.clara_replicates}",
        )
        for k in args.k_values:
            start = time.perf_counter()
            fitted = _fit_kmedoids(
                D_train, k, random_state=seed + k * 13,
                n_init=args.n_init, max_iter=args.max_iter,
            )
            medoid_rows = train_rows[fitted["medoid_positions"]]
            D_val_med, O_val_med = _cross_distance_matrix(
                validation_rows, medoid_rows, sequences, args.radius, args.n_jobs, dtw_path
            )
            val_labels = np.argmin(D_val_med, axis=1).astype(np.int32)
            val_nearest = np.min(D_val_med, axis=1)
            n_val_clusters = int(np.unique(val_labels).size)
            silhouette = (
                _validation_silhouette(D_val, val_labels, silhouette_score)
                if n_val_clusters == int(k) else float("nan")
            )
            replicate_records.append(
                {
                    "replicate": replicate + 1,
                    "seed": seed,
                    "k": int(k),
                    "train_objective_sum": float(fitted["objective_sum"]),
                    "train_objective_mean": float(fitted["objective_mean"]),
                    "validation_mean_nearest_dtw_rms": float(val_nearest.mean()),
                    "validation_total_nearest_dtw_rms": float(val_nearest.sum(dtype=np.float64)),
                    "validation_silhouette": silhouette,
                    "n_validation_clusters_observed": n_val_clusters,
                    "n_train_iterations": int(fitted["n_iter"]),
                    "converged": bool(fitted["converged"]),
                    "fit_seconds": float(time.perf_counter() - start),
                    "medoid_row_indices": ",".join(str(int(v)) for v in medoid_rows),
                }
            )
            print(
                f"  replicate={replicate + 1} K={k:2d} "
                f"train_mean={fitted['objective_mean']:.4f} "
                f"val_cost={val_nearest.mean():.4f} silhouette={silhouette:.4f}"
            )
        train_manifest.write_parquet(
            output_root / "search" / f"train_sample_replicate_{replicate + 1}.parquet",
            compression="zstd",
        )

    search_dir = output_root / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    val_manifest.write_parquet(search_dir / "validation_sample_manifest.parquet", compression="zstd")
    replicate_df = pl.DataFrame(replicate_records)
    replicate_df.write_csv(search_dir / "k_search_replicates.csv")
    aggregate = (
        replicate_df.group_by("k")
        .agg(
            pl.col("train_objective_mean").mean().alias("train_objective_mean"),
            pl.col("train_objective_mean").std().fill_null(0.0).alias("train_objective_sd"),
            pl.col("validation_mean_nearest_dtw_rms").mean().alias("validation_mean_nearest_dtw_rms"),
            pl.col("validation_mean_nearest_dtw_rms").std().fill_null(0.0).alias("validation_cost_sd"),
            pl.col("validation_silhouette").mean().alias("validation_silhouette_mean"),
            pl.col("validation_silhouette").std().fill_null(0.0).alias("validation_silhouette_sd"),
            pl.col("n_validation_clusters_observed").min().alias("min_validation_clusters_observed"),
            pl.col("converged").mean().alias("convergence_rate"),
            pl.col("fit_seconds").mean().alias("mean_fit_seconds"),
        )
        .sort("k")
    )
    aggregate.write_csv(search_dir / "k_search_metrics.csv")
    _write_search_plots(aggregate, search_dir)
    _write_json(
        search_dir / "search_manifest.json",
        {
            "algorithm": "CLARA-style alternating K-medoids on common-overlap DTW-RMS dissimilarities",
            "input_directory": str(input_dir),
            "output_directory": str(search_dir),
            "window": input_manifest["window"],
            "n_series_total": EXPECTED_N_SERIES,
            "n_observable": int((lengths > 0).sum()),
            "n_fit_eligible": int(fit_rows.size),
            "n_short_history_retained_unassigned": int(short_rows.size),
            "n_no_observation_history": int(no_history_rows.size),
            "minimum_history_days_for_medoid_fit_and_silhouette": int(args.min_overlap_days),
            "minimum_history_rationale": "Eight weeks to represent multiple repetitions of M5 weekly demand structure; configurable, not a data deletion rule.",
            "pairwise_common_overlap": "For each pair, crop both histories to the latest min(length_i, length_j) calendar days ending on d_1387.",
            "pairwise_normalization": "Z-normalize each series within that pairwise common interval; constant overlap segments are represented by zeros and flagged at series level.",
            "distance": "Sakoe-Chiba constrained DTW path cost divided by sqrt(path_length), i.e. RMS local discrepancy along the optimal path.",
            "sakoe_chiba_radius_days": int(args.radius),
            "k_values": args.k_values,
            "train_sample_size": int(args.train_sample_size),
            "validation_sample_size": int(validation_rows.size),
            "clara_replicates": int(args.clara_replicates),
            "n_init_per_k": int(args.n_init),
            "random_state": int(args.random_state),
            "silhouette": "tslearn.clustering.silhouette_score(metric='precomputed') on a fixed validation dissimilarity matrix; labels are nearest-medoid assignments from training-sample medoids.",
            "files": [
                "k_search_replicates.csv", "k_search_metrics.csv",
                "validation_sample_manifest.parquet", "common_overlap_dtw_elbow.png",
                "common_overlap_dtw_silhouette.png", "common_overlap_dtw_k_selection.png",
            ],
        },
    )
    print(f"\nK search complete. Review: {search_dir}")


def _stage_stability(args, input_dir: Path, output_root: Path) -> None:
    dtw_path, silhouette_score = _import_tslearn()
    _, sequences, lengths, _, _, _, metadata, _, _ = _load_series_context(input_dir)
    fit_rows = np.flatnonzero(lengths >= args.min_overlap_days).astype(np.int64)
    activity_all = metadata["activity_label"].to_numpy().astype(str)
    val_rows = _stratified_sample(
        fit_rows, min(args.validation_size, max(2, fit_rows.size // 3)),
        lengths, activity_all, args.random_state + 9000,
    )
    D_val, _ = _pairwise_distance_matrix(
        val_rows, sequences, args.radius, args.n_jobs, dtw_path, "fixed stability validation sample"
    )
    output_dir = output_root / "stability"
    output_dir.mkdir(parents=True, exist_ok=True)
    _sample_manifest(metadata, val_rows, lengths, "stability_validation").write_parquet(
        output_dir / "validation_sample_manifest.parquet", compression="zstd"
    )
    run_records: list[dict[str, Any]] = []
    label_sets: dict[int, list[np.ndarray]] = {int(k): [] for k in args.candidate_k_values}

    for seed in args.seeds:
        train_rows = _stratified_sample(
            fit_rows, args.train_sample_size, lengths, activity_all, int(seed), excluded_rows=val_rows
        )
        D_train, _ = _pairwise_distance_matrix(
            train_rows, sequences, args.radius, args.n_jobs, dtw_path, f"stability train seed {seed}"
        )
        for k in args.candidate_k_values:
            fitted = _fit_kmedoids(
                D_train, k, random_state=int(seed) + k * 17,
                n_init=args.n_init, max_iter=args.max_iter,
            )
            medoid_rows = train_rows[fitted["medoid_positions"]]
            D_val_med, _ = _cross_distance_matrix(
                val_rows, medoid_rows, sequences, args.radius, args.n_jobs, dtw_path
            )
            labels = np.argmin(D_val_med, axis=1).astype(np.int32)
            label_sets[int(k)].append(labels)
            n_val_clusters = int(np.unique(labels).size)
            sil = (
                _validation_silhouette(D_val, labels, silhouette_score)
                if n_val_clusters == int(k) else float("nan")
            )
            run_records.append(
                {
                    "seed": int(seed), "k": int(k),
                    "train_objective_mean": float(fitted["objective_mean"]),
                    "validation_silhouette": sil,
                    "n_validation_clusters_observed": n_val_clusters,
                    "converged": bool(fitted["converged"]),
                    "n_iter": int(fitted["n_iter"]),
                    "medoid_row_indices": ",".join(str(int(v)) for v in medoid_rows),
                }
            )
    run_df = pl.DataFrame(run_records)
    run_df.write_csv(output_dir / "stability_runs.csv")
    ari_records: list[dict[str, Any]] = []
    for k, sets in label_sets.items():
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                ari_records.append(
                    {
                        "k": k,
                        "seed_a": int(args.seeds[i]),
                        "seed_b": int(args.seeds[j]),
                        "adjusted_rand_index": float(adjusted_rand_score(sets[i], sets[j])),
                    }
                )
    ari_df = pl.DataFrame(ari_records)
    ari_df.write_csv(output_dir / "stability_pairwise_ari.csv")
    if ari_df.height:
        ari_summary = ari_df.group_by("k").agg(
            pl.col("adjusted_rand_index").mean().alias("mean_ari"),
            pl.col("adjusted_rand_index").std().fill_null(0.0).alias("sd_ari"),
            pl.col("adjusted_rand_index").min().alias("min_ari"),
        ).sort("k")
        ari_summary.write_csv(output_dir / "stability_ari_summary.csv")
    _write_json(
        output_dir / "stability_manifest.json",
        {
            "candidate_k_values": args.candidate_k_values,
            "seeds": args.seeds,
            "validation_sample_size": int(val_rows.size),
            "train_sample_size": int(args.train_sample_size),
            "minimum_history_days": int(args.min_overlap_days),
            "metric": "Adjusted Rand Index on the same held-out validation series, assigned to medoids from independent CLARA training samples.",
        },
    )
    print(f"\nStability assessment complete. Review: {output_dir}")


def _cluster_tables(
    assignments: pl.DataFrame,
    metadata: pl.DataFrame,
    output_dir: Path,
    medoid_rows: np.ndarray,
    sequences: list[np.ndarray | None],
    units: np.ndarray,
    calendar: pl.DataFrame,
    means: np.ndarray,
    scales: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assigned = assignments.filter(pl.col("dtw_cluster_id").is_not_null())
    if assigned.height:
        summary = assigned.group_by("dtw_cluster_id").agg(
            pl.len().alias("n_series"),
            pl.col("n_price_live_days").mean().alias("mean_history_days"),
            pl.col("n_price_live_days").median().alias("median_history_days"),
            pl.col("n_price_live_days").min().alias("min_history_days"),
            pl.col("n_price_live_days").max().alias("max_history_days"),
            pl.col("price_live_fraction").mean().alias("mean_price_live_fraction"),
            pl.col("positive_day_frequency_on_price_days").mean().alias("mean_positive_day_frequency"),
            pl.col("mean_units_per_price_live_day").mean().alias("mean_units_per_price_live_day"),
            (pl.col("activity_label") == "PRICE_RECORD_NO_POSITIVE_SALES").sum().alias("n_zero_sales_histories"),
            (pl.col("assignment_status") == "ASSIGNED_HIGH_OVERLAP").sum().alias("n_high_overlap_assignments"),
            (pl.col("assignment_status") == "ASSIGNED_LOW_OVERLAP").sum().alias("n_low_overlap_assignments"),
            (pl.col("is_constant_shape")).sum().alias("n_constant_shapes"),
        ).sort("dtw_cluster_id")
        summary.write_parquet(output_dir / "dtw_cluster_summary.parquet", compression="zstd")
        summary.write_csv(output_dir / "dtw_cluster_summary.csv")

        for dimension in ("store_id", "cat_id", "dept_id"):
            if dimension in assigned.columns:
                comp = (
                    assigned.group_by("dtw_cluster_id", dimension)
                    .agg(pl.len().alias("n_series"))
                    .with_columns(
                        (pl.col("n_series") / pl.col("n_series").sum().over("dtw_cluster_id")).alias("share_within_cluster")
                    )
                    .sort("dtw_cluster_id", dimension)
                )
                comp.write_parquet(output_dir / f"dtw_cluster_{dimension}_composition.parquet", compression="zstd")
                comp.write_csv(output_dir / f"dtw_cluster_{dimension}_composition.csv")

    # Every medoid is an actual M5 series. Export only its observed days.
    calendar_days = calendar.select("d_num", "date").to_dict(as_series=False)
    cal_d = calendar_days["d_num"]
    cal_dates = calendar_days["date"]
    profile_records: list[dict[str, Any]] = []
    medoid_records: list[dict[str, Any]] = []
    for cluster, row in enumerate(medoid_rows):
        row = int(row)
        seq = sequences[row]
        assert seq is not None
        length = int(seq.size)
        start_offset = WINDOW_DAYS - length
        meta_row = metadata.filter(pl.col("row_index") == row).row(0, named=True)
        medoid_records.append(
            {
                "dtw_cluster_id": cluster,
                "medoid_row_index": row,
                "item_id": meta_row.get("item_id"),
                "store_id": meta_row.get("store_id"),
                "sequence_length_days": length,
                "is_constant_shape": bool(np.std(seq) <= 1e-8),
                "mean_log1p_units": float(means[row]),
                "std_log1p_units": float(scales[row]),
            }
        )
        for local_idx in range(length):
            offset = start_offset + local_idx
            profile_records.append(
                {
                    "dtw_cluster_id": cluster,
                    "medoid_row_index": row,
                    "item_id": meta_row.get("item_id"),
                    "store_id": meta_row.get("store_id"),
                    "d_num": int(cal_d[offset]),
                    "date": str(cal_dates[offset]),
                    "units_sold": int(units[row, offset]),
                    "log1p_units": float(np.log1p(int(units[row, offset]))),
                    "full_history_zscore": float(
                        0.0 if scales[row] <= 1e-8 else (seq[local_idx] - means[row]) / scales[row]
                    ),
                }
            )
    pl.DataFrame(medoid_records).write_parquet(output_dir / "dtw_medoids.parquet", compression="zstd")
    pl.DataFrame(medoid_records).write_csv(output_dir / "dtw_medoids.csv")
    pl.DataFrame(profile_records).write_parquet(output_dir / "dtw_medoid_profiles.parquet", compression="zstd")
    pl.DataFrame(profile_records).write_csv(output_dir / "dtw_medoid_profiles.csv")


def _stage_fit(args, input_dir: Path, output_root: Path) -> None:
    dtw_path, _ = _import_tslearn()
    units, sequences, lengths, means, scales, constant, metadata, calendar, input_manifest = _load_series_context(input_dir)
    core_rows = np.flatnonzero(lengths >= args.min_overlap_days).astype(np.int64)
    short_rows = np.flatnonzero((lengths > 0) & (lengths < args.min_overlap_days)).astype(np.int64)
    no_history_rows = np.flatnonzero(lengths == 0).astype(np.int64)
    if args.n_clusters is None or args.n_clusters < 2:
        raise ValueError("Provide --n-clusters >= 2, selected from K-search/stability results")
    if core_rows.size <= args.n_clusters:
        raise ValueError("Not enough fit-eligible series for the selected K")
    activity_all = metadata["activity_label"].to_numpy().astype(str)

    best_solution: dict[str, Any] | None = None
    fit_replicate_records: list[dict[str, Any]] = []
    for replicate in range(args.fit_replicates):
        seed = args.random_state + 701 * replicate
        train_rows = _stratified_sample(
            core_rows, min(args.fit_sample_size, core_rows.size), lengths,
            activity_all, seed,
        )
        D_train, _ = _pairwise_distance_matrix(
            train_rows, sequences, args.radius, args.n_jobs, dtw_path,
            f"final CLARA sample {replicate + 1}/{args.fit_replicates}",
        )
        fitted = _fit_kmedoids(
            D_train, args.n_clusters, random_state=seed + args.n_clusters * 19,
            n_init=args.n_init, max_iter=args.max_iter,
        )
        medoid_rows = train_rows[fitted["medoid_positions"]]
        # Only histories meeting the minimum common-overlap evidence floor are
        # given behavioural cluster assignments. Shorter histories remain in
        # the complete output manifest with an explicit insufficiency status.
        clustered_rows = core_rows
        D_all_med, O_all_med = _cross_distance_matrix(
            clustered_rows, medoid_rows, sequences, args.radius, args.n_jobs, dtw_path,
        )
        labels, nearest, second = _assign_matrix(D_all_med, np.arange(args.n_clusters, dtype=np.int64))
        # Ensure medoid series own their respective cluster, including exact ties.
        observed_position = {int(row): pos for pos, row in enumerate(clustered_rows)}
        for cluster, medoid_row in enumerate(medoid_rows):
            pos = observed_position[int(medoid_row)]
            labels[pos] = cluster
            nearest[pos] = 0.0
            if args.n_clusters > 1:
                candidates = D_all_med[pos].copy()
                candidates[cluster] = np.inf
                second[pos] = float(np.min(candidates))
        overlap_to_medoid = O_all_med[np.arange(clustered_rows.size), labels]
        full_core_cost = float(nearest.sum(dtype=np.float64))
        full_core_mean = full_core_cost / int(clustered_rows.size)
        result = {
            "medoid_rows": medoid_rows,
            "observed_rows": clustered_rows,
            "labels": labels,
            "nearest": nearest,
            "second": second,
            "overlap_to_medoid": overlap_to_medoid,
            "full_core_cost": full_core_cost,
            "full_core_mean": full_core_mean,
            "sample_objective_mean": float(fitted["objective_mean"]),
            "n_iter": int(fitted["n_iter"]),
            "converged": bool(fitted["converged"]),
            "seed": seed,
            "replicate": replicate + 1,
        }
        fit_replicate_records.append({
            "replicate": replicate + 1,
            "seed": seed,
            "k": int(args.n_clusters),
            "sample_mean_dtw_rms": float(fitted["objective_mean"]),
            "full_core_mean_dtw_rms": float(full_core_mean),
            "full_core_total_dtw_rms": float(full_core_cost),
            "n_iter": int(fitted["n_iter"]),
            "converged": bool(fitted["converged"]),
            "medoid_row_indices": ",".join(str(int(v)) for v in medoid_rows),
        })
        print(
            f"  fit replicate={replicate + 1} K={args.n_clusters} "
            f"sample_mean={fitted['objective_mean']:.5f} full_core_mean={full_core_mean:.5f}"
        )
        if best_solution is None or full_core_mean < best_solution["full_core_mean"]:
            best_solution = result

    assert best_solution is not None
    out_dir = output_root / f"model_k{args.n_clusters}"
    out_dir.mkdir(parents=True, exist_ok=True)
    assignments = metadata.with_columns(
        pl.Series("dtw_cluster_id", _expand_labels(best_solution["observed_rows"], best_solution["labels"], EXPECTED_N_SERIES), dtype=pl.Int32),
        pl.Series("nearest_medoid_row_index", _expand_medoid_index(best_solution["observed_rows"], best_solution["labels"], best_solution["medoid_rows"], EXPECTED_N_SERIES), dtype=pl.Int64),
        pl.Series("nearest_medoid_dtw_rms", _expand_float(best_solution["observed_rows"], best_solution["nearest"], EXPECTED_N_SERIES), dtype=pl.Float32),
        pl.Series("second_nearest_medoid_dtw_rms", _expand_float(best_solution["observed_rows"], best_solution["second"], EXPECTED_N_SERIES), dtype=pl.Float32),
        pl.Series("medoid_distance_margin", _expand_float(best_solution["observed_rows"], best_solution["second"] - best_solution["nearest"], EXPECTED_N_SERIES), dtype=pl.Float32),
        pl.Series("common_overlap_days_to_medoid", _expand_int(best_solution["observed_rows"], best_solution["overlap_to_medoid"], EXPECTED_N_SERIES), dtype=pl.Int32),
        pl.Series("is_constant_shape", _expand_bool(np.flatnonzero(lengths > 0), constant[lengths > 0], EXPECTED_N_SERIES), dtype=pl.Boolean),
    ).with_columns(
        pl.when(pl.col("n_price_live_days") == 0).then(pl.lit("NO_OBSERVATION_HISTORY"))
        .when(pl.col("n_price_live_days") < args.min_overlap_days).then(pl.lit("INSUFFICIENT_COMMON_HISTORY"))
        .when(pl.col("n_price_live_days") < 364).then(pl.lit("ASSIGNED_LOW_OVERLAP"))
        .otherwise(pl.lit("ASSIGNED_HIGH_OVERLAP")).alias("assignment_status")
    )
    pl.DataFrame(fit_replicate_records).write_csv(out_dir / "fit_replicates.csv")
    assignments.write_parquet(out_dir / "series_dtw_cluster_assignments.parquet", compression="zstd")
    assignments.write_csv(out_dir / "series_dtw_cluster_assignments.csv")

    _cluster_tables(
        assignments, metadata, out_dir,
        best_solution["medoid_rows"],
        sequences, units, calendar, means, scales,
    )
    metrics = {
        "algorithm": "CLARA-style K-medoids on pairwise common-overlap DTW-RMS dissimilarities",
        "tslearn_version": _version("tslearn"),
        "window": input_manifest["window"],
        "n_series_total": EXPECTED_N_SERIES,
        "n_observable": int((lengths > 0).sum()),
        "n_clustered_with_sufficient_history": int(core_rows.size),
        "n_short_history_unassigned": int(short_rows.size),
        "n_no_observation_history": int(no_history_rows.size),
        "n_clusters": int(args.n_clusters),
        "minimum_history_days_for_reliable_cluster_fit": int(args.min_overlap_days),
        "sakoe_chiba_radius_days": int(args.radius),
        "fit_replicates": int(args.fit_replicates),
        "selected_fit_replicate": int(best_solution["replicate"]),
        "selected_seed": int(best_solution["seed"]),
        "selected_full_core_mean_dtw_rms": float(best_solution["full_core_mean"]),
        "selected_full_core_total_dtw_rms": float(best_solution["full_core_cost"]),
        "selected_sample_mean_dtw_rms": float(best_solution["sample_objective_mean"]),
        "n_iter": int(best_solution["n_iter"]),
        "converged": bool(best_solution["converged"]),
        "pairwise_common_overlap": "Each pair is cropped to the latest min(history_length_i, history_length_j) calendar days ending on d_1387.",
        "pairwise_normalization": "Each series is z-normalized independently within the pairwise common calendar interval; constant overlap segments become zeros. Unshared older observations do not affect pairwise normalization.",
        "distance": "Sakoe-Chiba constrained DTW path cost divided by sqrt(path length) (RMS local discrepancy).",
        "short_history_policy": "Series below the configurable minimum are retained in the complete assignment manifest but receive no behavioural cluster assignment; they do not participate in medoid fitting. The threshold is an evidence floor, not a demand-based filter.",
        "no_history_policy": "Series with no observed price-backed day are retained with null cluster assignment.",
        "outputs": [
            "fit_replicates.csv",
            "series_dtw_cluster_assignments.parquet", "series_dtw_cluster_assignments.csv",
            "dtw_cluster_summary.parquet", "dtw_medoids.parquet", "dtw_medoid_profiles.parquet",
        ],
    }
    _write_json(out_dir / "dtw_fit_metrics.json", metrics)
    _write_json(
        out_dir / "dtw_fit_manifest.json",
        {**metrics, "input_directory": str(input_dir), "output_directory": str(out_dir)},
    )
    print(f"\nCommon-overlap DTW fit complete: {out_dir}")
    print(f"  K={args.n_clusters}; assigned with sufficient history={core_rows.size:,}; insufficient history={short_rows.size:,}; no history={no_history_rows.size:,}")
    print(f"  Selected full-core mean DTW-RMS={best_solution['full_core_mean']:.6f}")


def _expand_labels(observed_rows: np.ndarray, values: np.ndarray, n_total: int) -> list[int | None]:
    result: list[int | None] = [None] * n_total
    for row, value in zip(observed_rows, values, strict=True):
        result[int(row)] = int(value)
    return result


def _expand_medoid_index(observed_rows: np.ndarray, labels: np.ndarray, medoids: np.ndarray, n_total: int) -> list[int | None]:
    result: list[int | None] = [None] * n_total
    for row, label in zip(observed_rows, labels, strict=True):
        result[int(row)] = int(medoids[int(label)])
    return result


def _expand_float(observed_rows: np.ndarray, values: np.ndarray, n_total: int) -> list[float | None]:
    result: list[float | None] = [None] * n_total
    for row, value in zip(observed_rows, values, strict=True):
        result[int(row)] = float(value) if np.isfinite(value) else None
    return result


def _expand_int(observed_rows: np.ndarray, values: np.ndarray, n_total: int) -> list[int | None]:
    result: list[int | None] = [None] * n_total
    for row, value in zip(observed_rows, values, strict=True):
        result[int(row)] = int(value)
    return result


def _expand_bool(observed_rows: np.ndarray, values: np.ndarray, n_total: int) -> list[bool | None]:
    result: list[bool | None] = [None] * n_total
    for row, value in zip(observed_rows, values, strict=True):
        result[int(row)] = bool(value)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Common-overlap DTW clustering for all M5 item-store series")
    parser.add_argument("--stage", choices=["smoke-test", "search", "stability", "fit"], default="smoke-test")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    parser.add_argument("--min-overlap-days", type=int, default=DEFAULT_MIN_OVERLAP_DAYS)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-sample-size", type=int, default=600)
    parser.add_argument("--validation-size", type=int, default=300)
    parser.add_argument("--clara-replicates", type=int, default=3)
    parser.add_argument("--n-init", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--k-values", type=lambda x: _parse_int_list(x, "k-values"), default=DEFAULT_K_VALUES)
    parser.add_argument("--candidate-k-values", type=lambda x: _parse_int_list(x, "candidate-k-values"), default=None)
    parser.add_argument("--seeds", type=lambda x: _parse_int_list(x, "seeds"), default=DEFAULT_SEEDS)
    parser.add_argument("--fit-sample-size", type=int, default=800)
    parser.add_argument("--fit-replicates", type=int, default=3)
    parser.add_argument("--n-clusters", type=int, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    project_root = args.project_root.resolve()
    input_dir = (args.input_dir or project_root / "data/processed/clustering/d1387_728d").resolve()
    output_root = (args.output_dir or project_root / "data/processed/clustering/d1387_728d/common_overlap_dtw_r7").resolve()
    if args.radius < 0:
        raise SystemExit("--radius must be >= 0")
    if args.min_overlap_days < 2:
        raise SystemExit("--min-overlap-days must be >= 2")
    if args.n_jobs < 1 or args.n_init < 1 or args.clara_replicates < 1:
        raise SystemExit("n-jobs, n-init and clara-replicates must be >= 1")
    if args.train_sample_size < 10 or args.validation_size < 10:
        raise SystemExit("train-sample-size and validation-size must be >= 10")

    print("M5 common-overlap DTW clustering")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_root}")
    print(f"  tslearn: {_version('tslearn')}")
    print(f"  NumPy: {_version('numpy')}; Polars: {_version('polars')}")
    print(f"  Radius: {args.radius}; minimum fit overlap: {args.min_overlap_days} days")

    if args.stage == "smoke-test":
        _run_smoke_test(args.radius)
    elif args.stage == "search":
        if min(args.k_values) < 2:
            raise SystemExit("All K values must be >= 2")
        _stage_search(args, input_dir, output_root)
    elif args.stage == "stability":
        if not args.candidate_k_values or min(args.candidate_k_values) < 2:
            raise SystemExit("For stability, supply --candidate-k-values with values >= 2")
        if len(args.seeds) < 2:
            raise SystemExit("Provide at least two seeds for stability assessment")
        _stage_stability(args, input_dir, output_root)
    elif args.stage == "fit":
        if args.n_clusters is None or args.n_clusters < 2:
            raise SystemExit("For fit, provide --n-clusters selected from K search/stability")
        _stage_fit(args, input_dir, output_root)


if __name__ == "__main__":
    main()
