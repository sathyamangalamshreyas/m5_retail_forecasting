#!/usr/bin/env python3
"""Mask-aware Euclidean K-means baseline for the prepared M5 dataset.

Run from the M5 project root:

    pixi run python scripts/cluster_m5_masked_euclidean.py

The script keeps all 30,490 series in its output assignment table. Every
series with at least one common observed date can receive a provisional cluster
assignment; series with no observed history remain present with a null cluster_id
and an explicit status because their demand trajectory is unobservable.

Distance definition
-------------------
1. Apply log1p sales (prepared upstream).
2. Z-normalize each series using only its observed days (shape-oriented view).
3. Compare a series to a cluster centroid only on dates observed for that
   series and represented in that centroid.
4. Distance is the root mean squared Euclidean difference over those common
   dates. A comparison is unavailable unless it has at least min_overlap days.
5. Centroids use observed member values only. Short histories receive reduced influence on the centroid, while all observed series remain eligible for assignment.

This is a mask-aware K-means-style baseline, not a drop-in sklearn KMeans
estimator. Per-series level/volume statistics remain in series_metadata.parquet
for later cluster characterization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

EXPECTED_N_SERIES = 30_490
EXPECTED_N_DAYS = 728
DEFAULT_MIN_OVERLAP = 1     # compute a distance on any actual common observed dates
DEFAULT_CONFIDENCE_OVERLAP = 364  # 52 weeks: mark a comparison as higher confidence
DEFAULT_N_CLUSTERS = 8     # exploratory starting point, not a selected final K
DEFAULT_N_INIT = 3
DEFAULT_MAX_ITER = 40
DEFAULT_TOL = 1e-4
DEFAULT_RANDOM_STATE = 42


def _load_inputs(input_dir: Path) -> tuple[np.ndarray, np.ndarray, pl.DataFrame, dict[str, Any]]:
    required = [
        input_dir / "sales_log1p.npy",
        input_dir / "observation_mask.npy",
        input_dir / "series_metadata.parquet",
        input_dir / "manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared clustering input(s):\n  " + "\n  ".join(missing)
            + "\nRun prepare_m5_clustering_data.py first."
        )

    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("window", {}).get("start_d") != 660 or manifest.get("window", {}).get("end_d") != 1387:
        raise ValueError("Prepared input manifest does not describe the expected d_660..d_1387 window")

    x = np.load(input_dir / "sales_log1p.npy", mmap_mode="r")
    mask = np.load(input_dir / "observation_mask.npy", mmap_mode="r")
    metadata = pl.read_parquet(input_dir / "series_metadata.parquet").sort("row_index")

    if x.shape != (EXPECTED_N_SERIES, EXPECTED_N_DAYS):
        raise ValueError(f"Expected sales_log1p shape {(EXPECTED_N_SERIES, EXPECTED_N_DAYS)}, got {x.shape}")
    if mask.shape != x.shape or mask.dtype != np.bool_:
        raise ValueError(f"Expected a boolean observation mask with shape {x.shape}; got {mask.shape}, {mask.dtype}")
    if metadata.height != EXPECTED_N_SERIES:
        raise ValueError(f"Expected {EXPECTED_N_SERIES:,} metadata rows, got {metadata.height:,}")
    if "row_index" not in metadata.columns:
        raise ValueError("series_metadata.parquet must contain row_index")
    expected_rows = np.arange(EXPECTED_N_SERIES, dtype=np.int64)
    if not np.array_equal(metadata["row_index"].to_numpy(), expected_rows):
        raise ValueError("Metadata row_index must be exactly 0..30,489 after sorting")
    if "n_price_live_days" not in metadata.columns:
        raise ValueError("series_metadata.parquet must contain n_price_live_days")

    mask_counts = np.asarray(mask.sum(axis=1), dtype=np.int32)
    meta_counts = metadata["n_price_live_days"].to_numpy().astype(np.int32)
    if not np.array_equal(mask_counts, meta_counts):
        raise ValueError("Observation-mask row counts do not match metadata.n_price_live_days")

    # Validate NaN semantics in bounded row blocks to avoid materializing another
    # full-size boolean array alongside the memory-mapped inputs.
    block_size = 1024
    for start in range(0, EXPECTED_N_SERIES, block_size):
        end = min(start + block_size, EXPECTED_N_SERIES)
        xb = np.asarray(x[start:end])
        mb = np.asarray(mask[start:end])
        if np.any(~np.isfinite(xb[mb])):
            raise ValueError(f"Non-finite log1p sales value found on an observed day in rows {start}:{end}")
        if np.any(~np.isnan(xb[~mb])):
            raise ValueError(f"Expected NaN for every unobserved log1p value in rows {start}:{end}")
        if np.any(xb[mb] < 0):
            raise ValueError("Negative log1p sales values found; M5 unit sales should be non-negative")

    return x, mask, metadata, manifest


def _standardize_observed(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-normalize each row over observed cells only; constants become zero."""
    n_rows = x.shape[0]
    count = mask.sum(axis=1, dtype=np.int32)
    x_filled = np.where(mask, np.asarray(x), 0.0).astype(np.float32, copy=False)
    sums = x_filled.sum(axis=1, dtype=np.float64)
    means = np.divide(sums, count, out=np.zeros(n_rows, dtype=np.float64), where=count > 0)
    centered = np.zeros_like(x_filled, dtype=np.float32)
    np.subtract(x_filled, means[:, None].astype(np.float32), out=centered, where=mask)
    centered[~mask] = 0.0
    squared = np.square(centered, dtype=np.float32)
    var_sum = squared.sum(axis=1, dtype=np.float64)
    variances = np.divide(var_sum, count, out=np.zeros(n_rows, dtype=np.float64), where=count > 0)
    scales = np.sqrt(np.maximum(variances, 0.0))
    no_observation = count == 0
    constant = (scales <= 1e-8) & ~no_observation
    safe_scales = np.where(constant | no_observation, 1.0, scales)
    z = (centered / safe_scales[:, None].astype(np.float32)).astype(np.float32, copy=False)
    z[~mask] = 0.0
    z[constant | no_observation] = 0.0
    means_out = means.astype(np.float32)
    scales_out = scales.astype(np.float32)
    means_out[no_observation] = np.nan
    scales_out[no_observation] = np.nan
    return z, means_out, scales_out, constant


def _distance_to_centroids(
    z: np.ndarray,
    mask: np.ndarray,
    centroids: np.ndarray,
    centroid_masks: np.ndarray,
    min_overlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row nearest centroid RMS distance and common-day count."""
    n_rows = z.shape[0]
    n_clusters = centroids.shape[0]
    best_sq = np.full(n_rows, np.inf, dtype=np.float64)
    best_overlap = np.zeros(n_rows, dtype=np.int32)

    for cluster_id in range(n_clusters):
        common = mask & centroid_masks[cluster_id][None, :]
        overlap = common.sum(axis=1, dtype=np.int32)
        valid = overlap >= min_overlap
        if not np.any(valid):
            continue
        diff = z - centroids[cluster_id][None, :]
        np.square(diff, out=diff)
        # Sum only over dates actually observed in both the series and centroid.
        diff *= common
        sse = diff.sum(axis=1, dtype=np.float64)
        d2 = np.full(n_rows, np.inf, dtype=np.float64)
        d2[valid] = sse[valid] / overlap[valid]
        better = d2 < best_sq
        best_sq[better] = d2[better]
        best_overlap[better] = overlap[better]

    return np.sqrt(best_sq).astype(np.float32), best_overlap


def _all_centroid_distances(
    z: np.ndarray,
    mask: np.ndarray,
    centroids: np.ndarray,
    centroid_masks: np.ndarray,
    min_overlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = z.shape[0]
    n_clusters = centroids.shape[0]
    dist_sq = np.full((n_rows, n_clusters), np.inf, dtype=np.float64)
    overlap_counts = np.zeros((n_rows, n_clusters), dtype=np.int32)
    for cluster_id in range(n_clusters):
        common = mask & centroid_masks[cluster_id][None, :]
        overlap = common.sum(axis=1, dtype=np.int32)
        valid = overlap >= min_overlap
        overlap_counts[:, cluster_id] = overlap
        if not np.any(valid):
            continue
        diff = z - centroids[cluster_id][None, :]
        np.square(diff, out=diff)
        diff *= common
        sse = diff.sum(axis=1, dtype=np.float64)
        dist_sq[valid, cluster_id] = sse[valid] / overlap[valid]
    return dist_sq, overlap_counts


def _init_kmeans_plus_plus(
    z: np.ndarray,
    mask: np.ndarray,
    eligible_rows: np.ndarray,
    full_rows: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
    min_overlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    if full_rows.size < n_clusters:
        raise ValueError(
            f"Need at least n_clusters={n_clusters} fully observed series for stable initialization; "
            f"found {full_rows.size}."
        )
    chosen: list[int] = [int(rng.choice(full_rows))]
    min_d2 = np.full(eligible_rows.size, np.inf, dtype=np.float64)

    while len(chosen) < n_clusters:
        row = chosen[-1]
        common = mask[eligible_rows] & mask[row][None, :]
        overlap = common.sum(axis=1, dtype=np.int32)
        valid = overlap >= min_overlap
        diff = z[eligible_rows] - z[row][None, :]
        np.square(diff, out=diff)
        diff *= common
        sse = diff.sum(axis=1, dtype=np.float64)
        d2 = np.full(eligible_rows.size, np.inf, dtype=np.float64)
        d2[valid] = sse[valid] / overlap[valid]
        min_d2 = np.minimum(min_d2, d2)

        remaining_full_mask = ~np.isin(full_rows, np.asarray(chosen, dtype=np.int64))
        remaining_full = full_rows[remaining_full_mask]
        candidate_positions = np.flatnonzero(np.isin(eligible_rows, remaining_full))
        if candidate_positions.size == 0:
            raise RuntimeError("Unable to find another distinct full-coverage initialization row")
        weights = min_d2[candidate_positions]
        finite_weights = np.isfinite(weights) & (weights > 1e-12)
        if finite_weights.any():
            positions = candidate_positions[finite_weights]
            probabilities = min_d2[positions]
            probabilities = probabilities / probabilities.sum()
            next_pos = int(rng.choice(positions, p=probabilities))
            chosen.append(int(eligible_rows[next_pos]))
        else:
            chosen.append(int(rng.choice(remaining_full)))

    centers = np.asarray(z[chosen], dtype=np.float32).copy()
    center_masks = np.asarray(mask[chosen], dtype=np.bool_).copy()
    return centers, center_masks


def _update_centroids(
    z: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    confidence_overlap_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_days = z.shape[1]
    centers = np.zeros((n_clusters, n_days), dtype=np.float32)
    center_masks = np.zeros((n_clusters, n_days), dtype=np.bool_)
    support = np.zeros((n_clusters, n_days), dtype=np.int32)
    weight_support = np.zeros((n_clusters, n_days), dtype=np.float64)
    for cluster_id in range(n_clusters):
        members = np.flatnonzero(labels == cluster_id)
        if members.size == 0:
            continue
        member_mask = mask[members]
        support[cluster_id] = member_mask.sum(axis=0, dtype=np.int32)
        # A series with short history still receives a cluster assignment, but
        # contributes less to the centroid. Reliability ramps linearly to 1 at
        # confidence_overlap_days. Combined with inverse-length weighting, each
        # observed value from a short series receives 1/confidence_overlap_days.
        row_counts = member_mask.sum(axis=1, dtype=np.float64)
        reliability = np.minimum(row_counts / float(confidence_overlap_days), 1.0)
        row_weights = np.divide(
            reliability, row_counts, out=np.zeros_like(row_counts), where=row_counts > 0
        )
        weighted_mask = member_mask * row_weights[:, None]
        weighted_sums = (z[members].astype(np.float64) * weighted_mask).sum(axis=0)
        weight_totals = weighted_mask.sum(axis=0)
        weight_support[cluster_id] = weight_totals
        valid_days = support[cluster_id] > 0
        centers[cluster_id, valid_days] = (
            weighted_sums[valid_days] / weight_totals[valid_days]
        ).astype(np.float32)
        center_masks[cluster_id] = valid_days
    return centers, center_masks, support, weight_support


def _run_one(
    z: np.ndarray,
    mask: np.ndarray,
    eligible_rows: np.ndarray,
    full_rows: np.ndarray,
    n_clusters: int,
    min_overlap: int,
    confidence_overlap_days: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    centers, center_masks = _init_kmeans_plus_plus(
        z, mask, eligible_rows, full_rows, n_clusters, rng, min_overlap
    )
    previous_labels: np.ndarray | None = None
    previous_objective: float | None = None
    converged = False
    iteration = 0
    row_observed_counts = mask[eligible_rows].sum(axis=1, dtype=np.float64)
    row_reliability = np.minimum(row_observed_counts / float(confidence_overlap_days), 1.0)

    for iteration in range(1, max_iter + 1):
        dist_sq, overlap_counts = _all_centroid_distances(
            z[eligible_rows], mask[eligible_rows], centers, center_masks, min_overlap
        )
        labels = np.argmin(dist_sq, axis=1).astype(np.int32)
        nearest_sq = dist_sq[np.arange(labels.size), labels]
        invalid = ~np.isfinite(nearest_sq)
        labels[invalid] = -1

        # Repair empty clusters with a far-away fully observed series. This is
        # still data-driven initialization; it does not create a manual cohort.
        used_reseeds: set[int] = set()
        for cluster_id in range(n_clusters):
            if np.any(labels == cluster_id):
                continue
            finite_positions = np.flatnonzero(np.isfinite(nearest_sq))
            if finite_positions.size == 0:
                continue
            # Prefer unused full-coverage rows, which give the new centroid a
            # complete calendar footprint at initialization.
            full_positions = np.flatnonzero(np.isin(eligible_rows, full_rows))
            options = np.intersect1d(finite_positions, full_positions, assume_unique=False)
            options = np.asarray([p for p in options if int(eligible_rows[p]) not in used_reseeds], dtype=np.int64)
            if options.size == 0:
                options = np.asarray([p for p in finite_positions if int(eligible_rows[p]) not in used_reseeds], dtype=np.int64)
            if options.size == 0:
                continue
            farthest_pos = int(options[np.argmax(nearest_sq[options])])
            chosen_row = int(eligible_rows[farthest_pos])
            used_reseeds.add(chosen_row)
            centers[cluster_id] = z[chosen_row]
            center_masks[cluster_id] = mask[chosen_row]
            labels[farthest_pos] = cluster_id
            nearest_sq[farthest_pos] = 0.0

        valid_objective = np.isfinite(nearest_sq)
        iteration_objective = float(
            np.sum(nearest_sq[valid_objective] * row_reliability[valid_objective])
        )
        new_centers, new_center_masks, support, weight_support = _update_centroids(
            z[eligible_rows], mask[eligible_rows], labels, n_clusters,
            confidence_overlap_days,
        )
        # If an empty cluster could not be repaired, retain its previous center.
        for cluster_id in range(n_clusters):
            if not new_center_masks[cluster_id].any():
                new_centers[cluster_id] = centers[cluster_id]
                new_center_masks[cluster_id] = center_masks[cluster_id]
        centers, center_masks = new_centers, new_center_masks

        labels_stable = previous_labels is not None and np.array_equal(labels, previous_labels)
        objective_stable = False
        if previous_objective is not None:
            relative_improvement = (previous_objective - iteration_objective) / max(abs(previous_objective), 1e-12)
            objective_stable = 0.0 <= relative_improvement <= tol
        if labels_stable or objective_stable:
            converged = True
            break
        previous_labels = labels.copy()
        previous_objective = iteration_objective

    # Final E-step against the final centroids.
    final_dist_sq, _ = _all_centroid_distances(
        z[eligible_rows], mask[eligible_rows], centers, center_masks, min_overlap
    )
    final_labels = np.argmin(final_dist_sq, axis=1).astype(np.int32)
    final_sq = final_dist_sq[np.arange(final_labels.size), final_labels]
    final_invalid = ~np.isfinite(final_sq)
    final_labels[final_invalid] = -1

    final_centers, final_center_masks, final_support, final_weight_support = _update_centroids(
        z[eligible_rows], mask[eligible_rows], final_labels, n_clusters,
        confidence_overlap_days,
    )
    for cluster_id in range(n_clusters):
        if not final_center_masks[cluster_id].any():
            final_centers[cluster_id] = centers[cluster_id]
            final_center_masks[cluster_id] = center_masks[cluster_id]
            final_support[cluster_id] = support[cluster_id]
            final_weight_support[cluster_id] = weight_support[cluster_id]

    # Report distances to the exact centroid profiles written to disk, rather
    # than to the centroids from the preceding E-step.
    final_distance = np.full(final_labels.size, np.nan, dtype=np.float32)
    final_common = np.zeros(final_labels.size, dtype=np.int32)
    for cluster_id in range(n_clusters):
        member_positions = np.flatnonzero(final_labels == cluster_id)
        if member_positions.size == 0:
            continue
        member_rows = eligible_rows[member_positions]
        common = mask[member_rows] & final_center_masks[cluster_id][None, :]
        overlap = common.sum(axis=1, dtype=np.int32)
        valid = overlap >= min_overlap
        if not np.all(valid):
            invalid_positions = member_positions[~valid]
            final_labels[invalid_positions] = -1
            member_positions = member_positions[valid]
            member_rows = member_rows[valid]
            common = common[valid]
            overlap = overlap[valid]
        if member_positions.size == 0:
            continue
        diff = z[member_rows] - final_centers[cluster_id][None, :]
        np.square(diff, out=diff)
        diff *= common
        sse = diff.sum(axis=1, dtype=np.float64)
        final_distance[member_positions] = np.sqrt(sse / overlap).astype(np.float32)
        final_common[member_positions] = overlap

    assigned = final_labels >= 0
    objective = float(
        np.sum(np.square(final_distance[assigned], dtype=np.float64) * row_reliability[assigned])
    )

    return {
        "labels": final_labels,
        "distances": final_distance,
        "overlap": final_common,
        "centroids": final_centers,
        "centroid_masks": final_center_masks,
        "centroid_support": final_support,
        "centroid_weight_support": final_weight_support,
        "objective": objective,
        "iterations": iteration,
        "converged": converged,
        "unassignable_eligible": int(final_invalid.sum()),
    }


def run_clustering(
    input_dir: Path,
    output_dir: Path,
    n_clusters: int,
    min_overlap: int,
    confidence_overlap_days: int,
    n_init: int,
    max_iter: int,
    tol: float,
    random_state: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    x, mask_mmap, metadata, input_manifest = _load_inputs(input_dir)
    mask = np.asarray(mask_mmap, dtype=np.bool_)
    print(f"Loaded prepared arrays: {x.shape}; observed cells={int(mask.sum()):,}")

    z, series_means, series_scales, constant_rows = _standardize_observed(x, mask)
    n_observed = mask.sum(axis=1, dtype=np.int32)
    eligible = n_observed >= min_overlap
    eligible_rows = np.flatnonzero(eligible).astype(np.int32)
    full_rows = np.flatnonzero(n_observed == EXPECTED_N_DAYS).astype(np.int32)
    if eligible_rows.size == 0:
        raise ValueError(f"No series meets min_overlap={min_overlap}")
    if full_rows.size < n_clusters:
        raise ValueError(f"Need at least {n_clusters} full-coverage series; found {full_rows.size}")

    print(
        f"Series with observed history available for assignment: {eligible_rows.size:,}; "
        f"no observed history: {EXPECTED_N_SERIES - eligible_rows.size:,}"
    )
    print(f"Running masked Euclidean K-means: k={n_clusters}, min_overlap={min_overlap}, n_init={n_init}")

    runs: list[dict[str, Any]] = []
    for init_idx in range(n_init):
        run_seed = random_state + init_idx
        print(f"  initialization {init_idx + 1}/{n_init} (seed={run_seed}) ...", flush=True)
        result = _run_one(
            z=z,
            mask=mask,
            eligible_rows=eligible_rows,
            full_rows=full_rows,
            n_clusters=n_clusters,
            min_overlap=min_overlap,
            confidence_overlap_days=confidence_overlap_days,
            max_iter=max_iter,
            tol=tol,
            seed=run_seed,
        )
        runs.append(result)
        print(
            f"    objective={result['objective']:.6f}; iterations={result['iterations']}; "
            f"converged={result['converged']}; assigned={int((result['labels'] >= 0).sum()):,}",
            flush=True,
        )
    best = min(runs, key=lambda item: item["objective"])

    all_labels = np.full(EXPECTED_N_SERIES, -1, dtype=np.int16)
    all_distances = np.full(EXPECTED_N_SERIES, np.nan, dtype=np.float32)
    all_overlap = np.zeros(EXPECTED_N_SERIES, dtype=np.int16)
    all_labels[eligible_rows] = best["labels"].astype(np.int16)
    all_distances[eligible_rows] = best["distances"]
    all_overlap[eligible_rows] = best["overlap"].astype(np.int16)

    statuses: list[str] = []
    for row_index, count in enumerate(n_observed):
        if count == 0:
            statuses.append("NO_OBSERVATION_HISTORY")
        elif all_labels[row_index] < 0:
            statuses.append("NO_VALID_CENTROID_OVERLAP")
        elif count < confidence_overlap_days:
            statuses.append("ASSIGNED_LOW_OVERLAP")
        else:
            statuses.append("ASSIGNED_HIGH_OVERLAP")

    assignments = metadata.with_columns(
        pl.Series("cluster_id", [None if value < 0 else int(value) for value in all_labels], dtype=pl.Int16),
        pl.Series("cluster_assignment_status", statuses, dtype=pl.String),
        pl.Series("distance_to_centroid_rms_z", all_distances, dtype=pl.Float32),
        pl.Series("n_overlap_days_to_centroid", all_overlap, dtype=pl.Int16),
        pl.Series("series_log1p_observed_mean", series_means, dtype=pl.Float32),
        pl.Series("series_log1p_observed_std", series_scales, dtype=pl.Float32),
        pl.Series("series_shape_is_constant", constant_rows, dtype=pl.Boolean),
    ).with_columns(
        pl.col("series_log1p_observed_mean").fill_nan(None),
        pl.col("series_log1p_observed_std").fill_nan(None),
    )

    cluster_sizes = (
        assignments.filter(pl.col("cluster_id").is_not_null())
        .group_by("cluster_id")
        .agg(
            pl.len().alias("n_series"),
            pl.col("distance_to_centroid_rms_z").mean().alias("mean_distance_rms_z"),
            pl.col("distance_to_centroid_rms_z").median().alias("median_distance_rms_z"),
            pl.col("distance_to_centroid_rms_z").quantile(0.95).alias("p95_distance_rms_z"),
            pl.col("n_overlap_days_to_centroid").min().alias("min_overlap_days"),
            pl.col("n_overlap_days_to_centroid").median().alias("median_overlap_days"),
            pl.col("price_live_fraction").mean().alias("mean_price_live_fraction"),
            pl.col("positive_day_frequency_on_price_days").mean().alias("mean_positive_day_frequency"),
            pl.col("mean_units_per_price_live_day").mean().alias("mean_units_per_price_live_day"),
        )
        .sort("cluster_id")
    )

    assignment_status_counts = assignments.group_by("cluster_assignment_status").len().sort("cluster_assignment_status")
    cluster_summary_records = cluster_sizes.to_dicts()
    profile_rows: list[dict[str, Any]] = []
    for cluster_id in range(n_clusters):
        for day_offset in range(EXPECTED_N_DAYS):
            profile_rows.append(
                {
                    "cluster_id": cluster_id,
                    "d_num": 660 + day_offset,
                    "centroid_zscore": (
                        float(best["centroids"][cluster_id, day_offset])
                        if best["centroid_masks"][cluster_id, day_offset]
                        else None
                    ),
                    "n_series_observed": int(best["centroid_support"][cluster_id, day_offset]),
                    "centroid_weight_support": float(best["centroid_weight_support"][cluster_id, day_offset]),
                    "centroid_has_observations": bool(best["centroid_masks"][cluster_id, day_offset]),
                }
            )
    profiles = pl.DataFrame(profile_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "series_cluster_assignments.parquet"
    summary_path = output_dir / "cluster_summary.parquet"
    profiles_path = output_dir / "cluster_centroid_profiles.parquet"
    centers_path = output_dir / "cluster_centroids.npz"
    metrics_path = output_dir / "baseline_metrics.json"
    manifest_path = output_dir / "baseline_manifest.json"

    assignments.write_parquet(assignments_path, compression="zstd")
    cluster_sizes.write_parquet(summary_path, compression="zstd")
    profiles.write_parquet(profiles_path, compression="zstd")
    np.savez_compressed(
        centers_path,
        centroids=best["centroids"],
        centroid_masks=best["centroid_masks"],
        centroid_support=best["centroid_support"],
        centroid_weight_support=best["centroid_weight_support"],
    )

    n_assigned = int((all_labels >= 0).sum())
    metrics: dict[str, Any] = {
        "algorithm": "mask_aware_euclidean_kmeans_baseline",
        "distance": "RMS Euclidean distance over dates observed in both series and centroid",
        "normalization": "per-series z-score of log1p sales using observed days only; constant series mapped to zero",
        "min_overlap_days_for_distance": min_overlap,
        "confidence_overlap_days": confidence_overlap_days,
        "n_clusters_requested": n_clusters,
        "n_clusters_populated": int(cluster_sizes.height),
        "n_series_total": EXPECTED_N_SERIES,
        "n_series_eligible_by_overlap": int(eligible_rows.size),
        "n_series_assigned": n_assigned,
        "n_series_assigned_high_overlap": int((np.isin(np.asarray(statuses), ["ASSIGNED_HIGH_OVERLAP"])).sum()),
        "n_series_assigned_low_overlap": int((np.isin(np.asarray(statuses), ["ASSIGNED_LOW_OVERLAP"])).sum()),
        "n_series_unassigned": EXPECTED_N_SERIES - n_assigned,
        "assignment_status_counts": {row["cluster_assignment_status"]: int(row["len"]) for row in assignment_status_counts.to_dicts()},
        "n_constant_shape_series": int(constant_rows.sum()),
        "best_initialization_objective": best["objective"],
        "best_initialization_iterations": best["iterations"],
        "best_initialization_converged": best["converged"],
        "initializations": [
            {
                "seed": random_state + idx,
                "objective": run["objective"],
                "iterations": run["iterations"],
                "converged": run["converged"],
                "n_assigned": int((run["labels"] >= 0).sum()),
            }
            for idx, run in enumerate(runs)
        ],
        "cluster_summary": cluster_summary_records,
        "caveats": [
            "The observation mask is a price-record proxy, not daily in-stock status.",
            "Per-series z-normalization emphasizes shape and removes absolute level; volume metadata is retained separately.",
            "Any positive common observation count permits a provisional assignment; assignments with fewer than 364 observed days are flagged low-overlap.",
            "Low-overlap series receive reduced influence in centroid updates and remain identifiable for cautious characterization.",
            "Series with no price-record history remain in the assignment table without a behavioural cluster_id because no demand trajectory is observed.",
            "This is an exploratory baseline; K=8 is not asserted to be the optimal number of clusters.",
        ],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    manifest = {
        "input_manifest": str((input_dir / "manifest.json").resolve()),
        "input_window": input_manifest["window"],
        "configuration": {
            "n_clusters": n_clusters,
            "min_overlap_days_for_distance": min_overlap,
            "confidence_overlap_days": confidence_overlap_days,
            "n_init": n_init,
            "max_iter": max_iter,
            "tol": tol,
            "random_state": random_state,
            "initialization": "k-means++ over full-window-coverage series",
        },
        "outputs": {
            "series_cluster_assignments": assignments_path.name,
            "cluster_summary": summary_path.name,
            "cluster_centroid_profiles": profiles_path.name,
            "cluster_centroids": centers_path.name,
            "baseline_metrics": metrics_path.name,
        },
        "n_series_total": EXPECTED_N_SERIES,
        "n_series_assigned": n_assigned,
        "n_series_assigned_high_overlap": int((np.isin(np.asarray(statuses), ["ASSIGNED_HIGH_OVERLAP"])).sum()),
        "n_series_assigned_low_overlap": int((np.isin(np.asarray(statuses), ["ASSIGNED_LOW_OVERLAP"])).sum()),
        "n_series_unassigned": EXPECTED_N_SERIES - n_assigned,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nMask-aware Euclidean baseline complete")
    print(f"  Total series retained: {EXPECTED_N_SERIES:,}")
    print(f"  Assigned:              {n_assigned:,}")
    print(f"    high-overlap (>= {confidence_overlap_days} days): "
          f"{metrics['n_series_assigned_high_overlap']:,}")
    print(f"    low-overlap  (<  {confidence_overlap_days} days): "
          f"{metrics['n_series_assigned_low_overlap']:,}")
    print(f"  Unassigned:            {EXPECTED_N_SERIES - n_assigned:,}")
    print(f"  Populated clusters:    {cluster_sizes.height}")
    print(f"  Output:                {output_dir.resolve()}")
    print(f"  Elapsed:               {manifest['elapsed_seconds']:.1f}s")
    print("  No input series were deleted or manually partitioned.")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/clustering/d1387_728d"),
        help="Prepared input directory, relative to project root unless absolute",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/clustering/d1387_728d/masked_euclidean_k8"),
        help="Output directory, relative to project root unless absolute",
    )
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--min-overlap-days", type=int, default=DEFAULT_MIN_OVERLAP,
                        help="Minimum shared observed days required to compute a provisional distance")
    parser.add_argument("--confidence-overlap-days", type=int, default=DEFAULT_CONFIDENCE_OVERLAP,
                        help="Observed-day count for high-confidence assignment and full centroid weight")
    parser.add_argument("--n-init", type=int, default=DEFAULT_N_INIT)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    if not input_dir.is_absolute():
        input_dir = root / input_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    if args.n_clusters < 2:
        print("ERROR: --n-clusters must be at least 2", file=sys.stderr)
        return 2
    if not 1 <= args.min_overlap_days <= EXPECTED_N_DAYS:
        print(f"ERROR: --min-overlap-days must be between 1 and {EXPECTED_N_DAYS}", file=sys.stderr)
        return 2
    if not 1 <= args.confidence_overlap_days <= EXPECTED_N_DAYS:
        print(f"ERROR: --confidence-overlap-days must be between 1 and {EXPECTED_N_DAYS}", file=sys.stderr)
        return 2
    if args.n_init < 1 or args.max_iter < 1 or args.tol < 0:
        print("ERROR: n-init/max-iter must be positive and tol must be non-negative", file=sys.stderr)
        return 2
    try:
        run_clustering(
            input_dir=input_dir.resolve(),
            output_dir=output_dir.resolve(),
            n_clusters=args.n_clusters,
            min_overlap=args.min_overlap_days,
            confidence_overlap_days=args.confidence_overlap_days,
            n_init=args.n_init,
            max_iter=args.max_iter,
            tol=args.tol,
            random_state=args.random_state,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
