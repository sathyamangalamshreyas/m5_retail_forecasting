#!/usr/bin/env python3
"""Group B: sparse k-NN graph clustering for the full M5 item-store panel.

The script extracts shape-oriented features from each series' observed price-live
history, creates a sparse k-nearest-neighbour graph with scikit-learn, and runs
Spectral Clustering and/or Louvain community detection on the same weighted graph.

Important design choices
------------------------
* All 30,490 source series remain in the output assignment table.
* The 1,131 series with no observed price history are retained as unassigned;
  there is no demand trajectory from which to build a behavioural neighbourhood.
* History length, price coverage, store, category and absolute demand level are
  deliberately excluded from the neighbour-distance feature matrix. They are
  retained for post-clustering diagnostics.
* Shape features are computed from the observed sequence, standardized within
  series, and include relative-history trajectory bins. These bins describe the
  evolution over each item's own observed history, not identical calendar spans.
* The exact all-pairs DTW/GAK matrix is not built. Neighbours are found in a
  compact, robustly scaled PCA representation to keep the 30k-series experiment
  practical on a memory-constrained machine.
* k-NN distances are transformed to local-scale affinities before symmetrizing.
  Louvain receives positive edge strengths, never raw distances.

Run from the M5 project root in the Group B Pixi environment:

    pixi run --environment group-b python scripts/run_m5_group_b_graph_clustering.py --stage smoke-test
    pixi run --environment group-b python scripts/run_m5_group_b_graph_clustering.py --stage run

The smoke test uses synthetic data and does not read the M5 files.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import math
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import RobustScaler

WINDOW_START_D = 660
WINDOW_END_D = 1387
WINDOW_DAYS = WINDOW_END_D - WINDOW_START_D + 1
EXPECTED_N_SERIES = 30_490
DEFAULT_GRAPH_K = [15, 30]
DEFAULT_SPECTRAL_K = [3, 6, 10]
DEFAULT_LOUVAIN_RESOLUTIONS = [0.5, 1.0, 1.5]
DEFAULT_SEEDS = [42, 43]

# This is the only feature matrix passed to the neighbour search. In particular,
# none of these columns encode store/category/history length/coverage/volume level.
FEATURE_COLUMNS = [
    "positive_day_frequency",
    "has_positive_sales",
    "positive_size_cv",
    "positive_recency_fraction",
    "max_zero_run_fraction",
    "mean_zero_run_fraction",
    "zero_run_count_per_100_days",
    "mean_interarrival_fraction",
    "interarrival_cv",
    "interarrival_q90_fraction",
    "log_shape_skew",
    "log_shape_excess_kurtosis",
    "acf_lag_1",
    "acf_lag_7",
    "acf_lag_28",
    "trend_slope_z",
    "early_late_frequency_delta",
    *[f"trajectory_bin_{i}" for i in range(1, 9)],
    *[f"weekday_mean_z_{i}" for i in range(7)],
]


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
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _parse_int_list(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("All values must be positive integers")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Values must be unique")
    return values


def _parse_float_list(value: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers") from exc
    if not values or any(not np.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError("All values must be finite and positive")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Values must be unique")
    return values


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of consecutive True runs in a one-dimensional boolean array."""
    values = np.asarray(mask, dtype=np.bool_)
    if values.size == 0 or not values.any():
        return np.empty(0, dtype=np.int32)
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return (ends - starts).astype(np.int32)


def _safe_acf(values: np.ndarray, lag: int) -> float:
    """Pearson autocorrelation at lag; NaN means too little usable information."""
    x = np.asarray(values, dtype=np.float64)
    if x.size <= lag + 2:
        return float("nan")
    left = x[:-lag]
    right = x[lag:]
    left_sd = float(left.std())
    right_sd = float(right.std())
    if left_sd < 1e-10 or right_sd < 1e-10:
        return float("nan")
    value = float(np.mean((left - left.mean()) * (right - right.mean())) / (left_sd * right_sd))
    return float(np.clip(value, -1.0, 1.0)) if np.isfinite(value) else float("nan")


def _series_feature_row(units: np.ndarray, day_of_week: np.ndarray | None = None) -> np.ndarray:
    """Extract demand-shape features from one observed sequence."""
    y = np.asarray(units, dtype=np.float64)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("Feature extraction requires a non-empty one-dimensional series")
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("Observed unit sales must be finite and non-negative")

    n = int(y.size)
    if day_of_week is None:
        weekday = (np.arange(n, dtype=np.int8) % 7)
    else:
        weekday = np.asarray(day_of_week, dtype=np.int8)
        if weekday.shape != (n,) or np.any((weekday < 0) | (weekday > 6)):
            raise ValueError("day_of_week must be a length-n array with values from 0 (Mon) to 6 (Sun)")
    positive = y > 0
    positive_positions = np.flatnonzero(positive)
    n_positive = int(positive_positions.size)
    frequency = float(n_positive / n)
    log_y = np.log1p(y)
    mean_log = float(log_y.mean())
    sd_log = float(log_y.std())
    z = (log_y - mean_log) / sd_log if sd_log >= 1e-10 else np.zeros(n, dtype=np.float64)

    zero_runs = _run_lengths(~positive)
    max_zero_fraction = float(zero_runs.max() / n) if zero_runs.size else 0.0
    mean_zero_fraction = float(zero_runs.mean() / n) if zero_runs.size else 0.0
    zero_run_count_rate = float(zero_runs.size / n * 100.0)

    if n_positive >= 2:
        gaps = np.diff(positive_positions).astype(np.float64)
        mean_gap_fraction = float(gaps.mean() / n)
        gap_mean = float(gaps.mean())
        gap_cv = float(gaps.std() / gap_mean) if gap_mean > 0 else 0.0
        gap_q90_fraction = float(np.quantile(gaps, 0.90) / n)
        positive_values = y[positive]
        positive_mean = float(positive_values.mean())
        positive_size_cv = float(positive_values.std() / positive_mean) if positive_mean > 0 else 0.0
    else:
        mean_gap_fraction = 0.0
        gap_cv = 0.0
        gap_q90_fraction = 0.0
        positive_size_cv = 0.0

    recency_fraction = (
        float((n - 1 - int(positive_positions[-1])) / n) if n_positive else 1.0
    )

    if sd_log >= 1e-10 and n >= 3:
        skew = float(np.mean(z**3))
        excess_kurtosis = float(np.mean(z**4) - 3.0)
    else:
        skew = 0.0
        excess_kurtosis = 0.0

    time_axis = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    time_centered = time_axis - time_axis.mean()
    trend_slope = float(np.dot(time_centered, z - z.mean()) / max(np.dot(time_centered, time_centered), 1e-12))

    quarter = max(1, int(math.ceil(n * 0.25)))
    early_frequency = float(positive[:quarter].mean())
    late_frequency = float(positive[-quarter:].mean())
    frequency_drift = late_frequency - early_frequency

    # Resample the standardized trajectory to a fixed number of relative-history
    # positions. This keeps the feature dimension fixed without zero-padding a
    # calendar-aligned raw series or adding history length to the distance.
    if n == 1:
        trajectory = np.zeros(8, dtype=np.float64)
    else:
        source_positions = np.linspace(0.0, 1.0, n, dtype=np.float64)
        target_positions = np.linspace(0.0, 1.0, 8, dtype=np.float64)
        trajectory = np.interp(target_positions, source_positions, z)

    # A weekday-specific z-profile captures recurring weekly shape without
    # including mean demand level. Require several observed dates per weekday;
    # otherwise leave that profile value undefined for later neutral imputation.
    weekday_profile = []
    for weekday_id in range(7):
        selected = z[weekday == weekday_id]
        weekday_profile.append(float(selected.mean()) if selected.size >= 3 else float("nan"))

    row = np.asarray(
        [
            frequency,
            float(n_positive > 0),
            positive_size_cv,
            recency_fraction,
            max_zero_fraction,
            mean_zero_fraction,
            zero_run_count_rate,
            mean_gap_fraction,
            gap_cv,
            gap_q90_fraction,
            skew,
            excess_kurtosis,
            _safe_acf(z, 1),
            _safe_acf(z, 7),
            _safe_acf(z, 28),
            trend_slope,
            frequency_drift,
            *trajectory.tolist(),
            *weekday_profile,
        ],
        dtype=np.float64,
    )
    if row.size != len(FEATURE_COLUMNS):
        raise RuntimeError(f"Feature dimension mismatch: {row.size} != {len(FEATURE_COLUMNS)}")
    return row


def extract_behavior_features(
    sales_units: np.ndarray,
    observation_mask: np.ndarray,
    calendar_weekday: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full-row feature matrix, eligible row indices, and history lengths.

    Rows with no price-live observations remain all-NaN in the feature matrix and
    are deliberately not sent to the graph algorithm.
    """
    units = np.asarray(sales_units)
    mask = np.asarray(observation_mask, dtype=np.bool_)
    if units.ndim != 2 or mask.shape != units.shape:
        raise ValueError("sales_units and observation_mask must be matching 2D arrays")
    if not np.issubdtype(units.dtype, np.number):
        raise TypeError(f"sales_units must be numeric, got {units.dtype}")
    if np.any(units < 0):
        raise ValueError("Negative sales values are not supported")
    if calendar_weekday is None:
        weekday_all = np.arange(units.shape[1], dtype=np.int8) % 7
    else:
        weekday_all = np.asarray(calendar_weekday, dtype=np.int8)
        if weekday_all.shape != (units.shape[1],) or np.any((weekday_all < 0) | (weekday_all > 6)):
            raise ValueError("calendar_weekday must match the day dimension and contain values 0..6")

    lengths = mask.sum(axis=1, dtype=np.int32)
    eligible_rows = np.flatnonzero(lengths > 0).astype(np.int64)
    full_features = np.full((units.shape[0], len(FEATURE_COLUMNS)), np.nan, dtype=np.float32)
    for row_index in eligible_rows.tolist():
        row_mask = mask[row_index]
        observed = np.asarray(units[row_index, row_mask], dtype=np.float64)
        observed_weekday = weekday_all[row_mask]
        full_features[row_index] = _series_feature_row(observed, observed_weekday).astype(np.float32)
    return full_features, eligible_rows, lengths.astype(np.int32)


def prepare_feature_embedding(
    feature_matrix: np.ndarray,
    eligible_rows: np.ndarray,
    n_components: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Median-impute undefined lags, robust-scale, and optionally PCA-reduce."""
    raw = np.asarray(feature_matrix[eligible_rows], dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 3:
        raise ValueError("Need at least three eligible series to build a neighbour graph")

    missing_fraction = np.mean(~np.isfinite(raw), axis=0)
    medians = np.zeros(raw.shape[1], dtype=np.float64)
    for column in range(raw.shape[1]):
        finite = raw[np.isfinite(raw[:, column]), column]
        medians[column] = float(np.median(finite)) if finite.size else 0.0
        missing = ~np.isfinite(raw[:, column])
        raw[missing, column] = medians[column]

    scaler = RobustScaler(quantile_range=(10.0, 90.0), unit_variance=False)
    scaled = scaler.fit_transform(raw)
    scaled = np.clip(scaled, -8.0, 8.0)

    actual_components = min(int(n_components), scaled.shape[1], scaled.shape[0] - 1)
    if actual_components > 0:
        pca = PCA(n_components=actual_components, svd_solver="randomized", random_state=random_state)
        embedding = pca.fit_transform(scaled).astype(np.float32)
        explained = pca.explained_variance_ratio_.astype(float).tolist()
        solver = "randomized_pca"
    else:
        embedding = scaled.astype(np.float32)
        explained = []
        solver = "robust_scaled_features_without_pca"

    info = {
        "feature_names": FEATURE_COLUMNS,
        "n_raw_features": int(scaled.shape[1]),
        "n_components_requested": int(n_components),
        "n_components_used": int(embedding.shape[1]),
        "pca_explained_variance_ratio_by_component": explained,
        "pca_explained_variance_ratio_cumulative": float(sum(explained)) if explained else None,
        "scaler": "RobustScaler(quantile_range=(10,90)); values clipped to [-8,8]",
        "undefined_feature_imputation": "column median; no missingness indicators are added",
        "missing_fraction_by_feature": {
            name: float(value) for name, value in zip(FEATURE_COLUMNS, missing_fraction, strict=True)
        },
        "embedding_method": solver,
        "excluded_from_distance_features": [
            "history length / number of observed price-live days",
            "price coverage and coverage label",
            "store, state, category, department and item identifiers",
            "absolute sales volume / mean demand level",
        ],
        "temporal_feature_semantics": (
            "Eight standardized trajectory values are sampled at equally spaced positions "
            "over each series' own observed history; they are not fixed calendar windows."
        ),
    }
    return embedding, info, scaled.astype(np.float32)


def _trim_directed_knn(distance_graph: sparse.csr_matrix, n_neighbors: int) -> sparse.csr_matrix:
    """Keep the closest n_neighbors in each row of a larger distance graph."""
    graph = distance_graph.tocsr(copy=False)
    n = graph.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row in range(n):
        start, end = int(graph.indptr[row]), int(graph.indptr[row + 1])
        row_cols = graph.indices[start:end]
        row_data = graph.data[start:end]
        if row_cols.size < n_neighbors:
            raise ValueError(
                f"Row {row} has only {row_cols.size} neighbours; requested {n_neighbors}."
            )
        order = np.argsort(row_data, kind="stable")[:n_neighbors]
        rows.extend([row] * len(order))
        cols.extend(row_cols[order].astype(int).tolist())
        data.extend(row_data[order].astype(float).tolist())
    result = sparse.csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))), shape=graph.shape)
    if np.any(result.diagonal() != 0):
        raise ValueError("Unexpected self-edge in the k-NN distance graph")
    if np.any(result.data < 0) or not np.isfinite(result.data).all():
        raise ValueError("k-NN distances must be finite and non-negative")
    return result


def build_affinity_graph(
    directed_distances: sparse.csr_matrix,
    symmetrize: str = "union",
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Convert directed k-NN distances to locally scaled symmetric affinities."""
    if symmetrize not in {"union", "mutual"}:
        raise ValueError("symmetrize must be 'union' or 'mutual'")
    distances = directed_distances.tocsr(copy=True)
    n = distances.shape[0]
    if distances.shape[0] != distances.shape[1]:
        raise ValueError("Distance graph must be square")

    # Estimate each node's local distance scale using its farthest retained
    # outgoing neighbour. A global positive fallback handles duplicate vectors.
    sigma = np.zeros(n, dtype=np.float64)
    positive_distances = distances.data[distances.data > 1e-12]
    global_scale = float(np.median(positive_distances)) if positive_distances.size else 1.0
    global_scale = max(global_scale, 1e-8)
    for row in range(n):
        start, end = int(distances.indptr[row]), int(distances.indptr[row + 1])
        if end > start:
            sigma[row] = float(np.max(distances.data[start:end]))
    sigma[sigma <= 1e-12] = global_scale

    coo = distances.tocoo(copy=False)
    denom = sigma[coo.row] * sigma[coo.col]
    exponent = np.square(coo.data.astype(np.float64)) / np.maximum(denom, 1e-16)
    weights = np.exp(-np.minimum(exponent, 50.0))
    directed_affinity = sparse.csr_matrix((weights, (coo.row, coo.col)), shape=(n, n))
    if symmetrize == "union":
        affinity = directed_affinity.maximum(directed_affinity.T)
    else:
        affinity = directed_affinity.minimum(directed_affinity.T)
    affinity = affinity.tocsr()
    affinity.setdiag(0.0)
    affinity.eliminate_zeros()
    affinity.sort_indices()

    if affinity.nnz and (not np.isfinite(affinity.data).all() or np.any(affinity.data <= 0)):
        raise ValueError("Affinity graph contains non-positive or non-finite weights")
    if (affinity - affinity.T).nnz:
        asymmetry = float(np.max(np.abs((affinity - affinity.T).data)))
        if asymmetry > 1e-10:
            raise ValueError(f"Affinity graph is not symmetric (max difference {asymmetry})")

    n_components, component_labels = connected_components(affinity, directed=False, return_labels=True)
    degrees = np.asarray((affinity > 0).sum(axis=1)).ravel().astype(np.int32)
    strengths = np.asarray(affinity.sum(axis=1)).ravel()
    component_sizes = np.bincount(component_labels, minlength=n_components)
    diagnostics = {
        "n_nodes": int(n),
        "n_undirected_edges": int(affinity.nnz // 2),
        "n_connected_components": int(n_components),
        "largest_component_size": int(component_sizes.max()) if component_sizes.size else 0,
        "smallest_component_size": int(component_sizes.min()) if component_sizes.size else 0,
        "singleton_component_count": int(np.sum(component_sizes == 1)),
        "largest_component_share": float(component_sizes.max() / n) if component_sizes.size and n else 0.0,
        "n_isolated_nodes": int(np.sum(degrees == 0)),
        "degree_min": int(degrees.min()) if degrees.size else 0,
        "degree_median": float(np.median(degrees)) if degrees.size else 0.0,
        "degree_max": int(degrees.max()) if degrees.size else 0,
        "weighted_degree_min": float(strengths.min()) if strengths.size else 0.0,
        "weighted_degree_median": float(np.median(strengths)) if strengths.size else 0.0,
        "weighted_degree_max": float(strengths.max()) if strengths.size else 0.0,
        "affinity_q05": float(np.quantile(affinity.data, 0.05)) if affinity.nnz else None,
        "affinity_median": float(np.median(affinity.data)) if affinity.nnz else None,
        "affinity_q95": float(np.quantile(affinity.data, 0.95)) if affinity.nnz else None,
        "symmetrization": symmetrize,
        "affinity_transform": "local scaling: exp(-distance^2 / (sigma_i * sigma_j))",
        "local_scale_fallback": global_scale,
    }
    return affinity, diagnostics


def _canonicalize_labels(labels: np.ndarray) -> np.ndarray:
    """Give labels a deterministic order by each community's smallest node index."""
    labels = np.asarray(labels, dtype=np.int32)
    communities = []
    for old_label in np.unique(labels):
        members = np.flatnonzero(labels == old_label)
        communities.append((int(members[0]), int(old_label)))
    mapping = {old: new for new, (_, old) in enumerate(sorted(communities))}
    return np.asarray([mapping[int(value)] for value in labels], dtype=np.int32)


def _graph_to_networkx(affinity: sparse.csr_matrix):
    import networkx as nx

    graph = nx.from_scipy_sparse_array(
        affinity,
        create_using=nx.Graph,
        edge_attribute="weight",
    )
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def _modularity(graph, labels: np.ndarray, resolution: float = 1.0) -> float:
    import networkx as nx

    communities = [set(np.flatnonzero(labels == value).tolist()) for value in np.unique(labels)]
    if not communities or graph.number_of_edges() == 0:
        return float("nan")
    return float(nx.community.modularity(graph, communities, weight="weight", resolution=resolution))


def fit_spectral(affinity: sparse.csr_matrix, n_clusters: int, seed: int, n_jobs: int) -> tuple[np.ndarray, list[str]]:
    from sklearn.cluster import SpectralClustering

    n = affinity.shape[0]
    if not 2 <= n_clusters < n:
        raise ValueError(f"Spectral K must be in [2, n_nodes-1], got K={n_clusters}, n={n}")
    if np.any(np.asarray((affinity > 0).sum(axis=1)).ravel() == 0):
        raise ValueError("Spectral Clustering cannot proceed with isolated nodes in this affinity graph")
    estimator = SpectralClustering(
        n_clusters=n_clusters,
        eigen_solver="arpack",
        n_components=n_clusters,
        random_state=seed,
        n_init=10,
        assign_labels="kmeans",
        affinity="precomputed",
        n_jobs=n_jobs,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        labels = estimator.fit_predict(affinity)
    return _canonicalize_labels(labels), [str(item.message) for item in captured]


def fit_louvain(graph, resolution: float, seed: int) -> np.ndarray:
    import networkx as nx

    communities = nx.community.louvain_communities(
        graph,
        weight="weight",
        resolution=float(resolution),
        seed=int(seed),
    )
    labels = np.full(graph.number_of_nodes(), -1, dtype=np.int32)
    for community_id, members in enumerate(communities):
        labels[np.fromiter(members, dtype=np.int64)] = community_id
    if np.any(labels < 0):
        raise RuntimeError("Louvain did not assign every graph node to a community")
    return _canonicalize_labels(labels)


def _cluster_size_metrics(labels: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(np.asarray(labels, dtype=np.int32))
    if counts.size == 0:
        return {
            "n_clusters": 0,
            "min_cluster_size": 0,
            "median_cluster_size": 0.0,
            "max_cluster_size": 0,
            "n_singleton_clusters": 0,
            "n_clusters_below_10": 0,
        }
    return {
        "n_clusters": int(counts.size),
        "min_cluster_size": int(counts.min()),
        "median_cluster_size": float(np.median(counts)),
        "max_cluster_size": int(counts.max()),
        "n_singleton_clusters": int(np.sum(counts == 1)),
        "n_clusters_below_10": int(np.sum(counts < 10)),
    }


def _pairwise_seed_stability(runs: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if len(runs) < 2:
        return {"n_seed_pairs": 0, "mean_pairwise_seed_ari": None, "std_pairwise_seed_ari": None, "min_pairwise_seed_ari": None}
    scores = [
        float(adjusted_rand_score(runs[i]["labels"], runs[j]["labels"]))
        for i in range(len(runs))
        for j in range(i + 1, len(runs))
    ]
    return {
        "n_seed_pairs": len(scores),
        "mean_pairwise_seed_ari": float(np.mean(scores)),
        "std_pairwise_seed_ari": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "min_pairwise_seed_ari": float(np.min(scores)),
    }


def _build_metadata_frame(input_dir: Path):
    """Load the prepared input and verify row-order/mask invariants."""
    import polars as pl

    required = [
        "sales_units.npy",
        "observation_mask.npy",
        "series_metadata.parquet",
        "calendar_window.parquet",
    ]
    missing = [name for name in required if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prepared clustering inputs in {input_dir}: {missing}")

    units = np.load(input_dir / "sales_units.npy", mmap_mode="r")
    mask = np.load(input_dir / "observation_mask.npy", mmap_mode="r")
    metadata = pl.read_parquet(input_dir / "series_metadata.parquet")
    calendar = pl.read_parquet(input_dir / "calendar_window.parquet").sort("d_num")

    if units.ndim != 2 or units.shape[1] != WINDOW_DAYS:
        raise ValueError(f"Expected sales_units shape (n_series, {WINDOW_DAYS}), got {units.shape}")
    if mask.shape != units.shape:
        raise ValueError("observation_mask shape does not match sales_units")
    if metadata.height != units.shape[0]:
        raise ValueError("series_metadata row count does not match sales_units")
    if "row_index" not in metadata.columns or "n_price_live_days" not in metadata.columns:
        raise ValueError("series_metadata must include row_index and n_price_live_days")
    row_indices = metadata["row_index"].to_numpy().astype(np.int64)
    if not np.array_equal(row_indices, np.arange(units.shape[0], dtype=np.int64)):
        raise ValueError("series_metadata.row_index does not match the original sales array order")

    mask_counts = np.asarray(mask.sum(axis=1), dtype=np.int32)
    metadata_counts = metadata["n_price_live_days"].to_numpy().astype(np.int32)
    if not np.array_equal(mask_counts, metadata_counts):
        raise ValueError("Observation-mask counts disagree with n_price_live_days metadata")

    # Group A verified this property for the prepared M5 data. Fail explicitly if
    # a future input version introduces internal gaps; they should not be silently
    # collapsed into consecutive observations by the feature extractor.
    for row_index in np.flatnonzero(mask_counts > 0):
        row_mask = np.asarray(mask[row_index], dtype=np.bool_)
        first = int(np.flatnonzero(row_mask)[0])
        if not row_mask[first:].all():
            raise ValueError(
                f"row_index={row_index} has an internal/trailing observation gap. "
                "This Group B implementation expects a continuous observed suffix."
            )

    if calendar.height != WINDOW_DAYS:
        raise ValueError(f"Expected {WINDOW_DAYS} calendar rows, got {calendar.height}")
    if "d_num" not in calendar.columns or calendar["d_num"].to_list() != list(range(WINDOW_START_D, WINDOW_END_D + 1)):
        raise ValueError("calendar_window must contain exactly ordered d_660..d_1387 rows")
    if int(np.asarray(units).max(initial=0)) < 0:
        raise ValueError("Negative sales values found")
    if np.any(np.asarray(units) < 0):
        raise ValueError("Negative sales values found in the prepared window")
    positive_without_price = np.asarray((units > 0) & ~np.asarray(mask, dtype=np.bool_)).sum()
    if int(positive_without_price) > 0:
        raise ValueError(f"Found {int(positive_without_price):,} positive sales outside the observation mask")

    return units, mask, metadata, calendar


def _write_feature_outputs(
    output_dir: Path,
    metadata,
    full_features: np.ndarray,
    history_lengths: np.ndarray,
    eligible_rows: np.ndarray,
):
    import polars as pl

    feature_columns = [pl.Series(name, full_features[:, idx], dtype=pl.Float32) for idx, name in enumerate(FEATURE_COLUMNS)]
    feature_frame = metadata.with_columns(
        pl.Series("history_days", history_lengths, dtype=pl.Int32),
        *feature_columns,
    )
    feature_path = output_dir / "series_behavior_features.parquet"
    feature_frame.write_parquet(feature_path, compression="zstd")

    np.save(output_dir / "behavior_features.npy", full_features, allow_pickle=False)
    np.save(output_dir / "eligible_row_indices.npy", eligible_rows, allow_pickle=False)
    np.save(output_dir / "history_lengths.npy", history_lengths, allow_pickle=False)
    return feature_frame


def _build_solution_frame(
    metadata,
    eligible_rows: np.ndarray,
    labels: np.ndarray,
    solution: dict[str, Any],
):
    import polars as pl

    if labels.shape != (eligible_rows.size,):
        raise ValueError("Cluster label count does not match eligible row count")
    full_labels = np.full(metadata.height, -1, dtype=np.int32)
    full_labels[eligible_rows] = labels.astype(np.int32)
    nullable = [None if value < 0 else int(value) for value in full_labels]
    return metadata.with_columns(
        pl.Series("cluster_id", nullable, dtype=pl.Int32),
        pl.Series("assignment_status", ["NO_PRICE_HISTORY" if value < 0 else "ASSIGNED" for value in full_labels], dtype=pl.String),
        pl.lit(solution["solution_id"]).alias("solution_id"),
        pl.lit(solution["algorithm"]).alias("algorithm"),
        pl.lit(int(solution["n_neighbors"])).alias("n_neighbors"),
        pl.lit(solution["symmetrize"]).alias("symmetrize"),
        pl.lit(solution["parameter_name"]).alias("parameter_name"),
        pl.lit(float(solution["parameter_value"])).alias("parameter_value"),
        pl.lit(int(solution["seed"])).alias("seed"),
    )


def _summarize_clusters(solution_frame, feature_frame, solution: dict[str, Any]):
    import polars as pl

    assigned = solution_frame.filter(pl.col("cluster_id").is_not_null())
    if assigned.is_empty():
        return pl.DataFrame()
    aggregate_exprs = [pl.len().alias("n_series")]
    if "n_price_live_days" in assigned.columns:
        aggregate_exprs.append(pl.col("n_price_live_days").median().alias("median_history_days"))
    if "price_live_fraction" in assigned.columns:
        aggregate_exprs.append(pl.col("price_live_fraction").median().alias("median_price_live_fraction"))
    if "is_full_window_price_coverage" in assigned.columns:
        aggregate_exprs.append(pl.col("is_full_window_price_coverage").cast(pl.Float64).mean().alias("full_coverage_share"))
    if "is_left_truncated_price_history" in assigned.columns:
        aggregate_exprs.append(pl.col("is_left_truncated_price_history").cast(pl.Float64).mean().alias("left_truncated_share"))
    if "is_zero_sales_on_all_price_days" in assigned.columns:
        aggregate_exprs.append(pl.col("is_zero_sales_on_all_price_days").cast(pl.Float64).mean().alias("no_positive_sales_share"))
    if "positive_day_frequency_on_price_days" in assigned.columns:
        aggregate_exprs.append(pl.col("positive_day_frequency_on_price_days").median().alias("median_positive_day_frequency"))
    if "zero_day_share_on_price_days" in assigned.columns:
        aggregate_exprs.append(pl.col("zero_day_share_on_price_days").median().alias("median_zero_day_share"))
    if "mean_units_per_price_live_day" in assigned.columns:
        aggregate_exprs.append(pl.col("mean_units_per_price_live_day").median().alias("median_units_per_price_live_day"))
    if "mean_units_per_positive_sales_day" in assigned.columns:
        aggregate_exprs.append(pl.col("mean_units_per_positive_sales_day").median().alias("median_units_per_positive_sales_day"))
    if "days_since_last_positive_sale" in assigned.columns:
        aggregate_exprs.append(pl.col("days_since_last_positive_sale").median().alias("median_days_since_last_positive_sale"))
    base = assigned.group_by("cluster_id").agg(aggregate_exprs).sort("cluster_id")
    total = assigned.height
    base = base.with_columns((pl.col("n_series") / total).alias("share_of_assigned_series"))

    cluster_features = (
        feature_frame.select(["row_index", *FEATURE_COLUMNS])
        .join(assigned.select("row_index", "cluster_id"), on="row_index", how="inner")
        .group_by("cluster_id")
        .agg([pl.col(name).median().alias(f"median_{name}") for name in FEATURE_COLUMNS])
    )
    return base.join(cluster_features, on="cluster_id", how="left").with_columns(
        pl.lit(solution["solution_id"]).alias("solution_id"),
        pl.lit(solution["algorithm"]).alias("algorithm"),
        pl.lit(int(solution["n_neighbors"])).alias("n_neighbors"),
        pl.lit(solution["symmetrize"]).alias("symmetrize"),
        pl.lit(solution["parameter_name"]).alias("parameter_name"),
        pl.lit(float(solution["parameter_value"])).alias("parameter_value"),
        pl.lit(int(solution["seed"])).alias("seed"),
    )


def _composition_rows(solution_frame, solution: dict[str, Any]):
    """Long-form store/category/coverage/activity composition and representation."""
    import polars as pl

    dimensions = [
        name for name in ["store_id", "state_id", "cat_id", "dept_id", "coverage_label", "activity_label"]
        if name in solution_frame.columns
    ]
    assigned = solution_frame.filter(pl.col("cluster_id").is_not_null())
    if assigned.is_empty():
        return pl.DataFrame()
    population = assigned.height
    rows = []
    for dimension in dimensions:
        overall = assigned.group_by(dimension).len().rename({"len": "population_n"})
        overall = overall.with_columns((pl.col("population_n") / population).alias("population_share"))
        counts = assigned.group_by("cluster_id", dimension).len().rename({"len": "n_series"})
        cluster_total = assigned.group_by("cluster_id").len().rename({"len": "cluster_n"})
        result = (
            counts.join(cluster_total, on="cluster_id", how="left")
            .join(overall, on=dimension, how="left")
            .with_columns(
                (pl.col("n_series") / pl.col("cluster_n")).alias("share_within_cluster"),
                ((pl.col("n_series") / pl.col("cluster_n")) / pl.col("population_share")).alias("representation_ratio"),
                pl.lit(dimension).alias("dimension"),
                pl.lit(solution["solution_id"]).alias("solution_id"),
                pl.lit(solution["algorithm"]).alias("algorithm"),
                pl.lit(int(solution["n_neighbors"])).alias("n_neighbors"),
                pl.lit(solution["symmetrize"]).alias("symmetrize"),
                pl.lit(solution["parameter_name"]).alias("parameter_name"),
                pl.lit(float(solution["parameter_value"])).alias("parameter_value"),
                pl.lit(int(solution["seed"])).alias("seed"),
            )
            .rename({dimension: "value"})
            .select(
                "solution_id", "algorithm", "n_neighbors", "symmetrize", "parameter_name",
                "parameter_value", "seed", "cluster_id", "dimension", "value", "n_series",
                "cluster_n", "share_within_cluster", "population_n", "population_share",
                "representation_ratio",
            )
        )
        rows.append(result)
    return pl.concat(rows, how="vertical") if rows else pl.DataFrame()


def _save_plots(output_dir: Path, run_records: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not run_records:
        return

    louvain = [r for r in run_records if r["algorithm"] == "louvain"]
    spectral = [r for r in run_records if r["algorithm"] == "spectral"]
    if louvain:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        graph_ks = sorted({int(r["n_neighbors"]) for r in louvain})
        for graph_k in graph_ks:
            subset = [r for r in louvain if int(r["n_neighbors"]) == graph_k]
            grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
            for rec in subset:
                grouped[float(rec["parameter_value"])].append(rec)
            resolutions = sorted(grouped)
            axes[0].plot(
                resolutions,
                [np.mean([x["n_clusters"] for x in grouped[value]]) for value in resolutions],
                marker="o",
                label=f"k-NN={graph_k}",
            )
            axes[1].plot(
                resolutions,
                [np.mean([x["modularity"] for x in grouped[value]]) for value in resolutions],
                marker="o",
                label=f"k-NN={graph_k}",
            )
        axes[0].set(title="Louvain community count", xlabel="Resolution", ylabel="Number of communities")
        axes[1].set(title="Louvain modularity", xlabel="Resolution", ylabel="Modularity")
        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "louvain_resolution_sensitivity.png", dpi=160)
        plt.close(fig)

    if spectral:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        graph_ks = sorted({int(r["n_neighbors"]) for r in spectral})
        for graph_k in graph_ks:
            subset = [r for r in spectral if int(r["n_neighbors"]) == graph_k]
            grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for rec in subset:
                grouped[int(rec["parameter_value"])].append(rec)
            ks = sorted(grouped)
            axes[0].plot(
                ks,
                [np.mean([x["n_clusters"] for x in grouped[value]]) for value in ks],
                marker="o",
                label=f"k-NN={graph_k}",
            )
            axes[1].plot(
                ks,
                [np.mean([x["modularity"] for x in grouped[value]]) for value in ks],
                marker="o",
                label=f"k-NN={graph_k}",
            )
        axes[0].set(title="Spectral cluster count", xlabel="Requested K", ylabel="Populated clusters")
        axes[1].set(title="Spectral partition modularity", xlabel="Requested K", ylabel="Modularity")
        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "spectral_k_sensitivity.png", dpi=160)
        plt.close(fig)


def run_smoke_test() -> dict[str, Any]:
    """Exercise feature extraction, k-NN graph, spectral and Louvain APIs."""
    rng = np.random.default_rng(2026)
    n_series, n_days = 180, 140
    units = np.zeros((n_series, n_days), dtype=np.int16)
    mask = np.zeros((n_series, n_days), dtype=np.bool_)
    for i in range(n_series):
        history = int(rng.integers(35, n_days + 1))
        mask[i, -history:] = True
        if i < 45:
            active = rng.random(history) < 0.82
            values = rng.poisson(2.5, size=history) + 1
        elif i < 90:
            active = rng.random(history) < 0.13
            values = rng.poisson(2.0, size=history) + 1
        elif i < 135:
            probs = np.linspace(0.05, 0.85, history)
            active = rng.random(history) < probs
            values = rng.poisson(2.0, size=history) + 1
        else:
            active = np.zeros(history, dtype=np.bool_)
            values = np.zeros(history, dtype=np.int16)
        units[i, -history:] = np.where(active, values, 0).astype(np.int16)

    features, eligible, lengths = extract_behavior_features(units, mask)
    if eligible.size != n_series:
        raise RuntimeError("Smoke test expected every synthetic series to have observations")
    embedding, embedding_info, scaled = prepare_feature_embedding(features, eligible, n_components=8, random_state=42)
    directed_max = kneighbors_graph(
        embedding,
        n_neighbors=12,
        mode="distance",
        metric="euclidean",
        include_self=False,
        n_jobs=1,
    ).tocsr()
    directed = _trim_directed_knn(directed_max, 8)
    affinity, graph_info = build_affinity_graph(directed, symmetrize="union")
    if affinity.shape != (n_series, n_series):
        raise RuntimeError("Smoke test produced unexpected graph dimensions")
    if (affinity - affinity.T).nnz:
        raise RuntimeError("Smoke test graph is not symmetric")

    spectral_labels, spectral_warnings = fit_spectral(affinity, n_clusters=4, seed=42, n_jobs=1)
    graph = _graph_to_networkx(affinity)
    louvain_labels = fit_louvain(graph, resolution=1.0, seed=42)
    result = {
        "status": "ok",
        "n_series": int(n_series),
        "n_eligible": int(eligible.size),
        "feature_dimensions": int(features.shape[1]),
        "embedding_dimensions": int(embedding.shape[1]),
        "embedding_explained_variance": embedding_info["pca_explained_variance_ratio_cumulative"],
        "graph": graph_info,
        "spectral": _cluster_size_metrics(spectral_labels),
        "spectral_modularity": _modularity(graph, spectral_labels),
        "spectral_warnings": spectral_warnings,
        "louvain": _cluster_size_metrics(louvain_labels),
        "louvain_modularity": _modularity(graph, louvain_labels),
        "history_length_min": int(lengths.min()),
        "history_length_max": int(lengths.max()),
    }
    print(json.dumps(result, indent=2, default=_json_default))
    return result


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    import polars as pl

    started = time.perf_counter()
    project_root = args.project_root.expanduser().resolve()
    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    if not input_dir.is_absolute():
        input_dir = project_root / input_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "graphs").mkdir(parents=True, exist_ok=True)

    print(f"Loading prepared M5 data: {input_dir}", flush=True)
    units, mask, metadata, calendar = _build_metadata_frame(input_dir)
    print(f"  sales shape={units.shape}; observed mask cells={int(np.asarray(mask).sum()):,}", flush=True)
    if units.shape[0] != EXPECTED_N_SERIES:
        raise ValueError(
            f"Expected all {EXPECTED_N_SERIES:,} M5 item-store series, got {units.shape[0]:,}. "
            "Use the complete d_660..d_1387 preparation, not a subset."
        )

    calendar_dates = calendar["date"].to_list()
    calendar_weekday = np.asarray(
        [
            value.weekday() if isinstance(value, date)
            else date.fromisoformat(str(value)[:10]).weekday()
            for value in calendar_dates
        ],
        dtype=np.int8,
    )
    features, eligible_rows, history_lengths = extract_behavior_features(units, mask, calendar_weekday)
    n_eligible = int(eligible_rows.size)
    n_no_history = int(units.shape[0] - n_eligible)
    print(f"Extracted shape features: {n_eligible:,} eligible; {n_no_history:,} no-history rows retained as unassigned", flush=True)
    if n_eligible < max(args.graph_k) + 1:
        raise ValueError("Not enough observed series for the requested maximum graph k")

    embedding, embedding_info, scaled_features = prepare_feature_embedding(
        features,
        eligible_rows,
        n_components=args.pca_components,
        random_state=args.seeds[0],
    )
    np.save(output_dir / "neighbor_embedding.npy", embedding, allow_pickle=False)
    np.save(output_dir / "robust_scaled_behavior_features.npy", scaled_features, allow_pickle=False)
    _write_json(output_dir / "feature_embedding_manifest.json", embedding_info)
    feature_frame = _write_feature_outputs(output_dir, metadata, features, history_lengths, eligible_rows)

    max_k = max(args.graph_k)
    print(f"Building maximum directed k-NN distance graph (k={max_k}) with sklearn.neighbors.kneighbors_graph ...", flush=True)
    started_neighbors = time.perf_counter()
    max_directed_distances = kneighbors_graph(
        embedding,
        n_neighbors=max_k,
        mode="distance",
        metric="euclidean",
        include_self=False,
        n_jobs=args.n_jobs,
    ).tocsr()
    print(
        f"  directed edges={max_directed_distances.nnz:,}; elapsed={time.perf_counter()-started_neighbors:.1f}s",
        flush=True,
    )
    sparse.save_npz(output_dir / "graphs" / f"directed_knn_distances_k{max_k}.npz", max_directed_distances)

    assignment_frames = []
    cluster_summary_frames = []
    composition_frames = []
    run_records: list[dict[str, Any]] = []
    per_config_runs: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    graph_manifest: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []

    for graph_k in args.graph_k:
        directed = _trim_directed_knn(max_directed_distances, graph_k)
        for symmetrize in args.symmetrize:
            affinity, graph_info = build_affinity_graph(directed, symmetrize=symmetrize)
            graph_key = f"knn_k{graph_k}_{symmetrize}"
            graph_path = output_dir / "graphs" / f"{graph_key}_affinity.npz"
            sparse.save_npz(graph_path, affinity)
            graph_info.update({
                "graph_key": graph_key,
                "n_neighbors_requested": int(graph_k),
                "affinity_path": str(graph_path.relative_to(output_dir)),
            })
            _write_json(output_dir / "graphs" / f"{graph_key}_diagnostics.json", graph_info)
            graph_manifest.append(graph_info)
            print(
                f"Graph {graph_key}: edges={graph_info['n_undirected_edges']:,}; "
                f"components={graph_info['n_connected_components']}; isolated={graph_info['n_isolated_nodes']}",
                flush=True,
            )
            graph_nx = _graph_to_networkx(affinity)

            if args.algorithms in {"both", "spectral"}:
                for requested_k in args.spectral_k:
                    if requested_k >= n_eligible:
                        continue
                    for seed in args.seeds:
                        run_started = time.perf_counter()
                        warning_texts: list[str] = []
                        try:
                            labels, warning_texts = fit_spectral(affinity, requested_k, seed, args.n_jobs)
                            run_status = "ok"
                            error_text = None
                        except Exception as exc:
                            labels = None
                            run_status = "failed"
                            error_text = f"{type(exc).__name__}: {exc}"
                        elapsed = time.perf_counter() - run_started
                        if labels is None:
                            failure_records.append({
                                "algorithm": "spectral", "n_neighbors": graph_k,
                                "symmetrize": symmetrize, "parameter_name": "n_clusters",
                                "parameter_value": requested_k, "seed": seed,
                                "status": run_status, "error": error_text, "elapsed_seconds": elapsed,
                            })
                            print(f"  Spectral k={graph_k}, K={requested_k}, seed={seed}: FAILED {error_text}", flush=True)
                            continue
                        metric = _cluster_size_metrics(labels)
                        modularity = _modularity(graph_nx or _graph_to_networkx(affinity), labels)
                        solution = {
                            "solution_id": f"spectral_knn{graph_k}_{symmetrize}_K{requested_k}_seed{seed}",
                            "algorithm": "spectral",
                            "n_neighbors": graph_k,
                            "symmetrize": symmetrize,
                            "parameter_name": "n_clusters",
                            "parameter_value": requested_k,
                            "seed": seed,
                            "labels": labels,
                        }
                        rec = {
                            "solution_id": solution["solution_id"],
                            "algorithm": "spectral",
                            "n_neighbors": graph_k,
                            "symmetrize": symmetrize,
                            "parameter_name": "n_clusters",
                            "parameter_value": requested_k,
                            "seed": seed,
                            **metric,
                            "modularity": modularity,
                            "status": run_status,
                            "elapsed_seconds": elapsed,
                            "warnings": " | ".join(warning_texts),
                        }
                        run_records.append(rec)
                        config_key = ("spectral", graph_k, symmetrize, "n_clusters", requested_k)
                        per_config_runs[config_key].append({"seed": seed, "labels": labels, **rec})
                        assignment = _build_solution_frame(metadata, eligible_rows, labels, solution)
                        assignment_frames.append(assignment)
                        cluster_summary_frames.append(_summarize_clusters(assignment, feature_frame, solution))
                        composition_frames.append(_composition_rows(assignment, solution))
                        print(
                            f"  Spectral graph-k={graph_k}, K={requested_k}, seed={seed}: "
                            f"clusters={metric['n_clusters']}, modularity={modularity:.4f}, {elapsed:.1f}s",
                            flush=True,
                        )

            if args.algorithms in {"both", "louvain"}:
                for resolution in args.louvain_resolutions:
                    for seed in args.seeds:
                        run_started = time.perf_counter()
                        try:
                            labels = fit_louvain(graph_nx, resolution, seed)
                            run_status = "ok"
                            error_text = None
                        except Exception as exc:
                            labels = None
                            run_status = "failed"
                            error_text = f"{type(exc).__name__}: {exc}"
                        elapsed = time.perf_counter() - run_started
                        if labels is None:
                            failure_records.append({
                                "algorithm": "louvain", "n_neighbors": graph_k,
                                "symmetrize": symmetrize, "parameter_name": "resolution",
                                "parameter_value": resolution, "seed": seed,
                                "status": run_status, "error": error_text, "elapsed_seconds": elapsed,
                            })
                            print(f"  Louvain k={graph_k}, resolution={resolution}, seed={seed}: FAILED {error_text}", flush=True)
                            continue
                        metric = _cluster_size_metrics(labels)
                        modularity = _modularity(graph_nx, labels, resolution=resolution)
                        solution = {
                            "solution_id": f"louvain_knn{graph_k}_{symmetrize}_r{resolution:g}_seed{seed}",
                            "algorithm": "louvain",
                            "n_neighbors": graph_k,
                            "symmetrize": symmetrize,
                            "parameter_name": "resolution",
                            "parameter_value": resolution,
                            "seed": seed,
                            "labels": labels,
                        }
                        rec = {
                            "solution_id": solution["solution_id"],
                            "algorithm": "louvain",
                            "n_neighbors": graph_k,
                            "symmetrize": symmetrize,
                            "parameter_name": "resolution",
                            "parameter_value": resolution,
                            "seed": seed,
                            **metric,
                            "modularity": modularity,
                            "status": run_status,
                            "elapsed_seconds": elapsed,
                            "warnings": "",
                        }
                        run_records.append(rec)
                        config_key = ("louvain", graph_k, symmetrize, "resolution", resolution)
                        per_config_runs[config_key].append({"seed": seed, "labels": labels, **rec})
                        assignment = _build_solution_frame(metadata, eligible_rows, labels, solution)
                        assignment_frames.append(assignment)
                        cluster_summary_frames.append(_summarize_clusters(assignment, feature_frame, solution))
                        composition_frames.append(_composition_rows(assignment, solution))
                        print(
                            f"  Louvain graph-k={graph_k}, resolution={resolution:g}, seed={seed}: "
                            f"communities={metric['n_clusters']}, modularity={modularity:.4f}, {elapsed:.1f}s",
                            flush=True,
                        )

    if not run_records:
        raise RuntimeError("No clustering solution completed successfully")

    # Add seed-to-seed stability per graph/parameter configuration.
    config_summary_rows = []
    for config_key, records in per_config_runs.items():
        stability = _pairwise_seed_stability(records)
        representative = records[0]
        config_summary_rows.append({
            "algorithm": config_key[0],
            "n_neighbors": config_key[1],
            "symmetrize": config_key[2],
            "parameter_name": config_key[3],
            "parameter_value": config_key[4],
            "n_successful_seeds": len(records),
            "mean_n_clusters": float(np.mean([r["n_clusters"] for r in records])),
            "min_n_clusters": int(min(r["n_clusters"] for r in records)),
            "max_n_clusters": int(max(r["n_clusters"] for r in records)),
            "mean_modularity": float(np.mean([r["modularity"] for r in records])),
            "min_cluster_size_across_runs": int(min(r["min_cluster_size"] for r in records)),
            "median_cluster_size_mean": float(np.mean([r["median_cluster_size"] for r in records])),
            "max_cluster_size_across_runs": int(max(r["max_cluster_size"] for r in records)),
            **stability,
        })
    config_summary = pl.DataFrame(config_summary_rows).sort(
        ["algorithm", "n_neighbors", "symmetrize", "parameter_value"]
    )
    run_summary = pl.DataFrame(run_records).sort(
        ["algorithm", "n_neighbors", "symmetrize", "parameter_value", "seed"]
    )
    assignment_table = pl.concat(assignment_frames, how="diagonal_relaxed")
    cluster_summary = pl.concat([x for x in cluster_summary_frames if not x.is_empty()], how="diagonal_relaxed")
    composition = pl.concat([x for x in composition_frames if not x.is_empty()], how="diagonal_relaxed")

    run_summary.write_csv(output_dir / "run_summary.csv")
    config_summary.write_csv(output_dir / "solution_summary.csv")
    assignment_table.write_parquet(output_dir / "series_cluster_assignments.parquet", compression="zstd")
    cluster_summary.write_csv(output_dir / "cluster_characterization_summary.csv")
    composition.write_csv(output_dir / "cluster_composition.csv")
    if failure_records:
        pl.DataFrame(failure_records).write_csv(output_dir / "failed_runs.csv")
    _save_plots(output_dir, run_records)

    manifest = {
        "stage": "full_m5_run",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "window": {"start_d": WINDOW_START_D, "end_d": WINDOW_END_D, "n_days": WINDOW_DAYS},
        "n_series_total": int(units.shape[0]),
        "n_series_with_observed_history": n_eligible,
        "n_series_without_observed_history_retained_unassigned": n_no_history,
        "feature_embedding": embedding_info,
        "graph_parameters": {
            "graph_source": "engineered demand-behaviour features -> robust scaling -> PCA -> sklearn.neighbors.kneighbors_graph",
            "n_neighbors": args.graph_k,
            "symmetrization": args.symmetrize,
            "distance_metric": "euclidean in robust-scaled PCA feature space",
            "affinity": "self-tuning local-scale RBF: exp(-distance^2/(sigma_i*sigma_j))",
            "max_directed_neighbors": max_k,
        },
        "clustering_parameters": {
            "algorithms": args.algorithms,
            "spectral_k": args.spectral_k,
            "louvain_resolutions": args.louvain_resolutions,
            "seeds": args.seeds,
            "spectral_assign_labels": "kmeans, n_init=10",
            "spectral_eigen_solver": "arpack",
            "louvain_implementation": "networkx.community.louvain_communities",
        },
        "graph_diagnostics": graph_manifest,
        "outputs": [
            "series_behavior_features.parquet",
            "behavior_features.npy",
            "eligible_row_indices.npy",
            "history_lengths.npy",
            "neighbor_embedding.npy",
            "robust_scaled_behavior_features.npy",
            "feature_embedding_manifest.json",
            "graphs/*.npz",
            "graphs/*_diagnostics.json",
            "run_summary.csv",
            "solution_summary.csv",
            "series_cluster_assignments.parquet",
            "cluster_characterization_summary.csv",
            "cluster_composition.csv",
            "failed_runs.csv (only if one or more configurations fail)",
            "louvain_resolution_sensitivity.png",
            "spectral_k_sensitivity.png",
        ],
        "caveats": [
            "The observation mask is a price-record proxy, not a daily in-stock indicator.",
            "No exact all-pairs DTW or GAK matrix is computed in Group B.",
            "The graph is built from interpretable shape features, eight relative-history trajectory samples and a weekday profile.",
            "Relative-history bins describe each item's own observed span; they are not aligned to identical calendar intervals.",
            "Undefined short-history autocorrelations are median-imputed without adding missingness indicators.",
            "History length, price coverage, store, category, identifiers and absolute demand level are excluded from distance features and audited after clustering.",
            "No-observation rows remain in the assignment table with null cluster_id; no observed series is filtered out.",
            "Louvain discovers the number of communities; its resolution parameter changes the preferred community scale.",
            "Spectral and Louvain partitions are compared on the same graph using seed stability, cluster balance and graph modularity; these diagnostics do not alone establish forecasting usefulness.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    _write_json(output_dir / "group_b_manifest.json", manifest)
    print("\nGroup B graph clustering complete", flush=True)
    print(f"  Total source series retained: {units.shape[0]:,}", flush=True)
    print(f"  With observed history:         {n_eligible:,}", flush=True)
    print(f"  No history (unassigned):       {n_no_history:,}", flush=True)
    print(f"  Successful clustering runs:    {len(run_records):,}", flush=True)
    print(f"  Output:                        {output_dir}", flush=True)
    print(f"  Elapsed:                       {manifest['elapsed_seconds']:.1f}s", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke-test", "run"], default="smoke-test")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/clustering/d1387_728d"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/clustering/d1387_728d/group_b_graph"))
    parser.add_argument("--graph-k", type=_parse_int_list, default=DEFAULT_GRAPH_K, help="Comma-separated k-NN graph sizes")
    parser.add_argument("--symmetrize", choices=["union", "mutual"], nargs="+", default=["union"], help="Graph symmetrization mode(s)")
    parser.add_argument("--spectral-k", type=_parse_int_list, default=DEFAULT_SPECTRAL_K, help="Spectral cluster counts")
    parser.add_argument("--louvain-resolutions", type=_parse_float_list, default=DEFAULT_LOUVAIN_RESOLUTIONS)
    parser.add_argument("--seeds", type=_parse_int_list, default=DEFAULT_SEEDS)
    parser.add_argument("--pca-components", type=int, default=12, help="Set 0 to use all robust-scaled features")
    parser.add_argument("--algorithms", choices=["both", "spectral", "louvain"], default="both")
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.pca_components < 0:
        print("ERROR: --pca-components must be non-negative", file=sys.stderr)
        return 2
    try:
        if args.stage == "smoke-test":
            run_smoke_test()
        else:
            run_experiment(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(limit=5)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
