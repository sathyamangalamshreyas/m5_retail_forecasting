#!/usr/bin/env python3
"""DTW clustering experiment for the prepared all-series M5 dataset.

Stages
------
1. smoke-test: verify installed tslearn APIs against unequal-length sequences.
2. search:    coarse-to-fine K search on a stratified pilot sample; inertia + DTW silhouette.
3. stability: repeat candidate K fits with different seeds on the same pilot sample; report ARI.
4. full-evaluate: fit a short list of candidate K values on every series with
                   observable history and compute a shared-sample DTW silhouette.
5. fit:       refit the selected K on every series with observable history; retain
              all 30,490 series in the assignment/metadata output.

The input is the mask-aware preparation at d_660..d_1387. The price-observation
mask is expected to describe a continuous suffix ending on d_1387 (verified
before reversing). Each observed suffix is z-normalized on observed days only,
reversed so d_1387 is sequence index zero, then right-padded with NaN. This is
the variable-length representation accepted by tslearn.

The 1,131 series with no observed price record are retained in the output with
NO_OBSERVATION_HISTORY; no demand-only cluster assignment is fabricated.

Run from the M5 project root, e.g.:
    pixi run python scripts/run_m5_dtw_clustering.py --stage smoke-test
    pixi run python scripts/run_m5_dtw_clustering.py --stage search
    pixi run python scripts/run_m5_dtw_clustering.py --stage stability --candidate-k-values 4,6,8
    pixi run python scripts/run_m5_dtw_clustering.py --stage fit --n-clusters 6
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

WINDOW_START_D = 660
WINDOW_END_D = 1387
WINDOW_DAYS = WINDOW_END_D - WINDOW_START_D + 1
EXPECTED_N_SERIES = 30_490
DEFAULT_RADIUS = 7
DEFAULT_LONG_HISTORY_DAYS = 364
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
    if isinstance(value, np.ndarray):
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
        from tslearn.clustering import TimeSeriesKMeans, silhouette_samples, silhouette_score
        from tslearn.metrics import cdist_dtw, dtw
    except ImportError as exc:
        raise RuntimeError(
            "tslearn is required for this stage. Install it in the active Pixi environment with:\n"
            '  pixi add --pypi "tslearn==0.10.0" --environment gam-rust\n'
            "Then run `pixi install` if required and retry."
        ) from exc
    return TimeSeriesKMeans, silhouette_score, silhouette_samples, cdist_dtw, dtw


def _metric_params(radius: int) -> dict[str, Any]:
    return {
        "global_constraint": "sakoe_chiba",
        "sakoe_chiba_radius": int(radius),
    }


def _load_inputs(input_dir: Path) -> tuple[np.ndarray, np.ndarray, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    required = [
        input_dir / "sales_log1p.npy",
        input_dir / "observation_mask.npy",
        input_dir / "series_metadata.parquet",
        input_dir / "calendar_window.parquet",
        input_dir / "manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared M5 clustering input(s):\n  "
            + "\n  ".join(missing)
            + "\nRun prepare_m5_clustering_data.py first."
        )

    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    window = manifest.get("window", {})
    if (
        window.get("start_d") != WINDOW_START_D
        or window.get("end_d") != WINDOW_END_D
        or window.get("n_days") != WINDOW_DAYS
    ):
        raise ValueError("Prepared manifest does not describe the expected d_660..d_1387 window")

    x = np.load(input_dir / "sales_log1p.npy", mmap_mode="r")
    mask = np.load(input_dir / "observation_mask.npy", mmap_mode="r")
    metadata = pl.read_parquet(input_dir / "series_metadata.parquet").sort("row_index")
    calendar = pl.read_parquet(input_dir / "calendar_window.parquet").sort("d_num")

    expected_shape = (EXPECTED_N_SERIES, WINDOW_DAYS)
    if x.shape != expected_shape:
        raise ValueError(f"Expected sales_log1p shape {expected_shape}, received {x.shape}")
    if mask.shape != expected_shape or mask.dtype != np.bool_:
        raise ValueError(f"Expected bool observation mask with shape {expected_shape}; received {mask.shape}/{mask.dtype}")
    if metadata.height != EXPECTED_N_SERIES:
        raise ValueError(f"Expected {EXPECTED_N_SERIES:,} metadata rows, received {metadata.height:,}")
    if "row_index" not in metadata.columns:
        raise ValueError("series_metadata.parquet must contain row_index")
    expected_rows = np.arange(EXPECTED_N_SERIES, dtype=np.int64)
    if not np.array_equal(metadata["row_index"].to_numpy(), expected_rows):
        raise ValueError("Metadata row_index must be exactly 0..30,489 after sorting")
    if not {"n_price_live_days", "activity_label", "coverage_label"}.issubset(metadata.columns):
        raise ValueError("Metadata must contain n_price_live_days, activity_label and coverage_label")
    if calendar.height != WINDOW_DAYS or calendar["d_num"].to_list() != list(range(WINDOW_START_D, WINDOW_END_D + 1)):
        raise ValueError("calendar_window.parquet must contain ordered d_660..d_1387 rows")

    mask_counts = np.asarray(mask.sum(axis=1), dtype=np.int32)
    meta_counts = metadata["n_price_live_days"].to_numpy().astype(np.int32)
    if not np.array_equal(mask_counts, meta_counts):
        raise ValueError("Observation-mask row counts do not match metadata.n_price_live_days")

    # The audit/preparation found no internal no-price gaps and all observed
    # periods end at d_1387. Reversal is valid only under this invariant.
    if np.any(mask[:, :-1] & ~mask[:, 1:]):
        examples = np.argwhere(mask[:, :-1] & ~mask[:, 1:])[:10].tolist()
        raise ValueError(
            "Internal or trailing observation-mask gaps were found. The current DTW adapter "
            "requires continuous observed suffixes ending at d_1387. "
            f"First [row_index, day_offset] examples: {examples}"
        )

    # Validate values in bounded blocks, avoiding a full boolean copy beside memmaps.
    block_size = 1024
    for start in range(0, EXPECTED_N_SERIES, block_size):
        end = min(start + block_size, EXPECTED_N_SERIES)
        xb = np.asarray(x[start:end])
        mb = np.asarray(mask[start:end])
        if np.any(~np.isfinite(xb[mb])):
            raise ValueError(f"Non-finite log1p values found on observed days in rows {start}:{end}")
        if np.any(~np.isnan(xb[~mb])):
            raise ValueError(f"Expected NaN on every unobserved day in rows {start}:{end}")
        if np.any(xb[mb] < 0):
            raise ValueError("Negative log1p sales values found; M5 unit sales should be non-negative")

    return x, mask, metadata, calendar, manifest


def _build_reversed_sequences(
    x: np.ndarray,
    mask: np.ndarray,
    metadata: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return NaN-padded, reversed, per-series z-normalized observed suffixes.

    X rows correspond to `eligible_rows` in the full 30,490-row input. Index 0
    is d_1387; increasing sequence index moves backward in calendar time.
    """
    lengths_all = np.asarray(mask.sum(axis=1), dtype=np.int32)
    eligible_rows = np.flatnonzero(lengths_all > 0).astype(np.int64)
    lengths = lengths_all[eligible_rows]
    n_eligible = int(eligible_rows.size)
    X = np.full((n_eligible, WINDOW_DAYS, 1), np.nan, dtype=np.float32)
    means = np.full(n_eligible, np.nan, dtype=np.float32)
    scales = np.full(n_eligible, np.nan, dtype=np.float32)
    constant = np.zeros(n_eligible, dtype=np.bool_)

    for position, row_index in enumerate(eligible_rows):
        length = int(lengths[position])
        # Continuous-suffix validation was performed in _load_inputs.
        vals = np.asarray(x[row_index, WINDOW_DAYS - length :], dtype=np.float32)
        if vals.size != length or not np.all(np.isfinite(vals)):
            raise ValueError(f"Invalid observed suffix for row_index={row_index}")
        mean = float(vals.mean(dtype=np.float64))
        scale = float(vals.std(dtype=np.float64))
        means[position] = mean
        scales[position] = scale
        if scale <= 1e-8:
            z = np.zeros(length, dtype=np.float32)
            constant[position] = True
        else:
            z = ((vals - np.float32(mean)) / np.float32(scale)).astype(np.float32, copy=False)
        X[position, :length, 0] = z[::-1]

    if not np.all(np.isfinite(X[:, :, 0][~np.isnan(X[:, :, 0])])):
        raise ValueError("Unexpected non-finite values in reversed sequence representation")

    # Metadata counts must be aligned with the same row sequence.
    expected = metadata["n_price_live_days"].to_numpy()[eligible_rows]
    if not np.array_equal(expected, lengths):
        raise ValueError("Eligible sequence lengths are not aligned with series metadata")

    return X, eligible_rows, lengths, means, scales, constant


def _length_bins(lengths: np.ndarray) -> np.ndarray:
    # Fixed interpretable bins, ending with the exact full-window category.
    edges = np.array([90, 180, 364, 547, 727], dtype=np.int32)
    return np.searchsorted(edges, lengths, side="left").astype(np.int8)


def _proportional_stratified_sample(
    metadata: pl.DataFrame,
    eligible_rows: np.ndarray,
    lengths: np.ndarray,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    """Sample proportionally by history-length band and activity status.

    Returns positions into `eligible_rows` / X, not original row_index values.
    """
    n = eligible_rows.size
    if sample_size >= n:
        return np.arange(n, dtype=np.int64)
    if sample_size < 20:
        raise ValueError("Pilot sample size must be at least 20")

    rng = np.random.default_rng(random_state)
    activity_all = metadata["activity_label"].to_numpy()
    activity = activity_all[eligible_rows]
    bins = _length_bins(lengths)
    strata: dict[tuple[int, str], np.ndarray] = {}
    for length_bin in np.unique(bins):
        for activity_label in np.unique(activity):
            positions = np.flatnonzero((bins == length_bin) & (activity == activity_label))
            if positions.size:
                strata[(int(length_bin), str(activity_label))] = positions

    total = float(n)
    keys = list(strata)
    exact = np.asarray([sample_size * len(strata[key]) / total for key in keys], dtype=np.float64)
    quotas = np.floor(exact).astype(np.int64)
    remainders = exact - quotas
    # Guarantee at least one from every non-empty stratum where the budget permits.
    if sample_size >= len(keys):
        for i, key in enumerate(keys):
            if quotas[i] == 0:
                quotas[i] = 1
    quotas = np.minimum(quotas, np.asarray([len(strata[k]) for k in keys], dtype=np.int64))

    while int(quotas.sum()) < sample_size:
        capacity = np.asarray([len(strata[k]) for k in keys], dtype=np.int64) - quotas
        available = np.flatnonzero(capacity > 0)
        if available.size == 0:
            break
        best = available[np.argmax(remainders[available])]
        quotas[best] += 1
        remainders[best] = -1.0
    while int(quotas.sum()) > sample_size:
        available = np.flatnonzero(quotas > 1)
        if available.size == 0:
            available = np.flatnonzero(quotas > 0)
        worst = available[np.argmin(remainders[available])]
        quotas[worst] -= 1
        remainders[worst] = 0.0

    selected: list[int] = []
    for key, quota in zip(keys, quotas, strict=True):
        if quota:
            selected.extend(rng.choice(strata[key], size=int(quota), replace=False).tolist())
    result = np.asarray(selected, dtype=np.int64)
    rng.shuffle(result)
    if result.size != sample_size:
        raise RuntimeError(f"Stratified sampler returned {result.size}, expected {sample_size}")
    return result


def _sample_manifest(
    metadata: pl.DataFrame,
    eligible_rows: np.ndarray,
    lengths: np.ndarray,
    positions: np.ndarray,
) -> pl.DataFrame:
    selected_rows = eligible_rows[positions]
    order = pl.DataFrame(
        {
            "sample_position": np.arange(positions.size, dtype=np.int32),
            "row_index": selected_rows.astype(np.int64),
            "sequence_length_days": lengths[positions].astype(np.int32),
        }
    )
    return order.join(metadata, on="row_index", how="left").sort("sample_position")


def _make_model(
    n_clusters: int,
    random_state: int,
    radius: int,
    max_iter: int,
    max_iter_barycenter: int,
    n_init: int,
    n_jobs: int,
    tol: float,
):
    TimeSeriesKMeans, *_ = _import_tslearn()
    return TimeSeriesKMeans(
        n_clusters=int(n_clusters),
        metric="dtw",
        metric_params=_metric_params(radius),
        max_iter=int(max_iter),
        max_iter_barycenter=int(max_iter_barycenter),
        n_init=int(n_init),
        random_state=int(random_state),
        n_jobs=int(n_jobs),
        tol=float(tol),
        verbose=0,
        init="k-means++",
    )


def _silhouette_distance_matrix(
    X_eval: np.ndarray,
    radius: int,
    n_jobs: int,
) -> tuple[np.ndarray, float]:
    """Compute the constrained-DTW matrix once for a bounded evaluation sample."""
    _, _, _, cdist_dtw, _ = _import_tslearn()
    started = time.perf_counter()
    distances = np.asarray(
        cdist_dtw(
            X_eval,
            global_constraint="sakoe_chiba",
            sakoe_chiba_radius=int(radius),
            n_jobs=int(n_jobs),
        ),
        dtype=np.float64,
    )
    if distances.shape != (X_eval.shape[0], X_eval.shape[0]):
        raise RuntimeError("Unexpected DTW silhouette distance-matrix shape")
    if not np.all(np.isfinite(distances)):
        raise RuntimeError("Non-finite pairwise distances found in the DTW silhouette sample")
    return distances, float(time.perf_counter() - started)


def _silhouette_from_precomputed(
    distance_matrix: np.ndarray,
    labels_eval: np.ndarray,
    expected_n_clusters: int | None = None,
) -> tuple[float | None, str | None, int, np.ndarray | None]:
    """Use tslearn silhouette APIs on a precomputed DTW distance matrix."""
    _, silhouette_score, silhouette_samples, _, _ = _import_tslearn()
    n_labels = int(np.unique(labels_eval).size)
    if n_labels < 2 or n_labels >= labels_eval.size:
        return None, "Silhouette requires between 2 and n_samples-1 represented clusters", n_labels, None
    if expected_n_clusters is not None and n_labels != expected_n_clusters:
        return (
            None,
            f"Only {n_labels} of {expected_n_clusters} fitted clusters are represented in the fixed silhouette sample",
            n_labels,
            None,
        )
    try:
        score = float(silhouette_score(distance_matrix, labels_eval, metric="precomputed"))
        values = np.asarray(
            silhouette_samples(distance_matrix, labels_eval, metric="precomputed"),
            dtype=np.float32,
        )
        if not math.isfinite(score) or not np.all(np.isfinite(values)):
            return None, "Silhouette returned non-finite values", n_labels, None
        return score, None, n_labels, values
    except Exception as exc:  # preserve the search run and record the failure
        return None, f"{type(exc).__name__}: {exc}", n_labels, None


def _plot_search(metrics: pl.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = metrics.sort("n_clusters")
    k = frame["n_clusters"].to_numpy()
    inertia = frame["inertia"].to_numpy()
    silhouette = frame["silhouette_score"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k, inertia, marker="o")
    ax.set_title("DTW K-means inertia by cluster count")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("tslearn inertia")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "dtw_inertia_elbow.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k, silhouette, marker="o")
    ax.set_title("DTW silhouette score by cluster count")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Mean silhouette score")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "dtw_silhouette_scores.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k, inertia, marker="o", label="Inertia")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("tslearn inertia")
    ax2 = ax.twinx()
    ax2.plot(k, silhouette, marker="s", label="Silhouette")
    ax2.set_ylabel("Mean DTW silhouette")
    ax.set_title("DTW K-search: inertia and silhouette")
    ax.grid(True, alpha=0.25)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "dtw_k_selection_metrics.png", dpi=160)
    plt.close(fig)


def _stage_smoke_test(radius: int, n_jobs: int) -> None:
    TimeSeriesKMeans, silhouette_score, _, cdist_dtw, dtw = _import_tslearn()
    rng = np.random.default_rng(7)
    sequences: list[np.ndarray] = []
    labels_true: list[int] = []

    # Two deliberately distinct pattern families. Shorter arrays are prefixes
    # of the reversed common-end representation and are NaN-padded on the right.
    base_a = np.sin(np.linspace(0, 4 * np.pi, 64, dtype=np.float32))
    base_b = np.cos(np.linspace(0, 4 * np.pi, 64, dtype=np.float32)) + 0.8
    for group, base in enumerate((base_a, base_b)):
        for length in (8, 16, 32, 48, 64, 64):
            noise = rng.normal(0.0, 0.025, size=length).astype(np.float32)
            # Select a chronological suffix, z-normalize it, then reverse it
            # so index zero is the shared latest date for every series.
            chronological = (base[-length:] + noise).astype(np.float32)
            chronological = (chronological - chronological.mean()) / max(
                float(chronological.std()), 1e-8
            )
            seq = chronological[::-1].copy()
            sequences.append(seq[:, None])
            labels_true.append(group)

    X = np.full((len(sequences), 64, 1), np.nan, dtype=np.float32)
    for i, seq in enumerate(sequences):
        X[i, : len(seq), :] = seq

    d = float(dtw(X[0], X[-1], **_metric_params(radius)))
    if not math.isfinite(d):
        raise RuntimeError("Unequal-length constrained DTW smoke test returned a non-finite distance")

    # Explicitly document the endpoint-alignment consequence for nested
    # observed histories: the full-sequence DTW includes the longer series'
    # additional older observations, while overlap-only DTW would not.
    shared_recent = rng.normal(0.0, 1.0, size=24).astype(np.float32)
    additional_older = rng.normal(0.0, 1.0, size=24).astype(np.float32)
    short_history = shared_recent[:, None]
    long_history = np.concatenate([shared_recent, additional_older])[:, None]
    common_overlap_distance = float(
        dtw(short_history, long_history[: len(short_history)], **_metric_params(radius))
    )
    complete_sequence_distance = float(dtw(short_history, long_history, **_metric_params(radius)))
    if common_overlap_distance > 1e-6 or not math.isfinite(complete_sequence_distance):
        raise RuntimeError("Nested-history DTW diagnostic produced an unexpected result")

    model = TimeSeriesKMeans(
        n_clusters=2,
        metric="dtw",
        metric_params=_metric_params(radius),
        n_init=1,
        max_iter=3,
        max_iter_barycenter=3,
        random_state=17,
        n_jobs=n_jobs,
        tol=1e-4,
    )
    labels = model.fit_predict(X)
    if labels.shape != (len(sequences),) or np.unique(labels).size != 2:
        raise RuntimeError("Variable-length TimeSeriesKMeans smoke test did not return two clusters")
    sil = float(
        silhouette_score(
            X,
            labels,
            metric="dtw",
            metric_params=_metric_params(radius),
            n_jobs=n_jobs,
        )
    )
    distances = cdist_dtw(X, global_constraint="sakoe_chiba", sakoe_chiba_radius=radius, n_jobs=n_jobs)
    if distances.shape != (len(sequences), len(sequences)) or not np.all(np.isfinite(distances)):
        raise RuntimeError("Variable-length cdist_dtw smoke test returned an invalid distance matrix")
    if not np.allclose(distances, distances.T, atol=1e-6) or not np.allclose(
        np.diag(distances), 0.0, atol=1e-6
    ):
        raise RuntimeError("Variable-length cdist_dtw matrix is not symmetric with a zero diagonal")

    print("tslearn DTW smoke test passed")
    print(f"  tslearn version: {_version('tslearn')}")
    print(f"  Unequal-length constrained DTW example (8..64 points): {d:.6f}")
    print(f"  TimeSeriesKMeans labels: {np.bincount(labels).tolist()}")
    print(f"  DTW silhouette on synthetic sample: {sil:.6f}")
    print(f"  Nested-history common-overlap DTW: {common_overlap_distance:.6f}")
    print(f"  Nested-history full-sequence DTW: {complete_sequence_distance:.6f}")
    print("  NaN trailing padding was accepted and removed as variable-length padding.")
    print("  Note: full-sequence DTW includes additional older observations in longer histories;")
    print("        it does not crop every pair to their common calendar overlap.")


def _stage_search(args: argparse.Namespace, input_dir: Path, output_root: Path) -> None:
    x, mask, metadata, _calendar, input_manifest = _load_inputs(input_dir)
    X_all, eligible_rows, lengths, means, scales, constant = _build_reversed_sequences(x, mask, metadata)
    if args.pilot_size > X_all.shape[0]:
        raise ValueError(f"pilot-size={args.pilot_size} exceeds observed-history population {X_all.shape[0]}")
    if max(args.k_values) >= args.pilot_size:
        raise ValueError("Every K must be smaller than pilot sample size")

    sample_positions = _proportional_stratified_sample(
        metadata, eligible_rows, lengths, args.pilot_size, args.random_state
    )
    X_pilot = X_all[sample_positions]
    pilot_manifest = _sample_manifest(metadata, eligible_rows, lengths, sample_positions)
    search_dir = output_root / "k_search"
    search_dir.mkdir(parents=True, exist_ok=True)
    pilot_manifest.write_parquet(search_dir / "pilot_sample_manifest.parquet")
    pilot_manifest.write_csv(search_dir / "pilot_sample_manifest.csv")

    # One fixed silhouette evaluation subset is reused for every K. We pass the
    # subset itself to tslearn, rather than sample_size=... on the full dataset,
    # so pairwise DTW is computed only for this bounded subset.
    rng = np.random.default_rng(args.random_state + 10_000)
    sil_size = min(args.silhouette_size, args.pilot_size)
    sil_local_positions = np.sort(rng.choice(args.pilot_size, size=sil_size, replace=False))
    X_silhouette = X_pilot[sil_local_positions]
    silhouette_rows = eligible_rows[sample_positions[sil_local_positions]]
    print(f"Computing constrained-DTW silhouette distances for {sil_size:,} fixed evaluation series...", flush=True)
    silhouette_distances, silhouette_distance_seconds = _silhouette_distance_matrix(
        X_silhouette, radius=args.radius, n_jobs=args.n_jobs
    )
    pl.DataFrame(
        {
            "silhouette_position": np.arange(sil_size, dtype=np.int32),
            "row_index": silhouette_rows.astype(np.int64),
        }
    ).join(metadata, on="row_index", how="left").write_parquet(
        search_dir / "silhouette_sample_manifest.parquet"
    )

    metric_rows: list[dict[str, Any]] = []
    assignment_columns: dict[str, np.ndarray] = {
        "sample_position": np.arange(args.pilot_size, dtype=np.int32),
        "row_index": eligible_rows[sample_positions].astype(np.int64),
    }
    all_started = time.perf_counter()

    for k in args.k_values:
        print(f"\nFitting pilot DTW K-means for K={k} ({args.pilot_size:,} series)...", flush=True)
        model = _make_model(
            n_clusters=k,
            random_state=args.random_state,
            radius=args.radius,
            max_iter=args.search_max_iter,
            max_iter_barycenter=args.search_barycenter_iter,
            n_init=1,
            n_jobs=args.n_jobs,
            tol=args.tol,
        )
        started = time.perf_counter()
        labels = model.fit_predict(X_pilot)
        fit_seconds = time.perf_counter() - started
        silhouette_started = time.perf_counter()
        silhouette, silhouette_error, n_clusters_in_silhouette, _silhouette_values = _silhouette_from_precomputed(
            silhouette_distances, labels[sil_local_positions], expected_n_clusters=k
        )
        silhouette_seconds = time.perf_counter() - silhouette_started
        cluster_sizes = np.bincount(labels, minlength=k)
        assignment_columns[f"cluster_k{k}"] = labels.astype(np.int32)
        metric_rows.append(
            {
                "n_clusters": int(k),
                "inertia": float(model.inertia_),
                "silhouette_score": silhouette,
                "silhouette_error": silhouette_error,
                "n_clusters_in_silhouette_sample": int(n_clusters_in_silhouette),
                "n_clusters_fitted": int(np.unique(labels).size),
                "min_cluster_size": int(cluster_sizes.min()),
                "median_cluster_size": float(np.median(cluster_sizes)),
                "max_cluster_size": int(cluster_sizes.max()),
                "n_iter": int(model.n_iter_),
                "fit_seconds": float(fit_seconds),
                "silhouette_seconds": float(silhouette_seconds),
                "random_state": int(args.random_state),
            }
        )
        print(
            f"  inertia={model.inertia_:.6f}; silhouette={silhouette}; "
            f"iterations={model.n_iter_}; fit={fit_seconds:.1f}s; silhouette={silhouette_seconds:.1f}s",
            flush=True,
        )

    metrics = pl.DataFrame(metric_rows).sort("n_clusters")
    metrics.write_csv(search_dir / "k_search_metrics.csv")
    pl.DataFrame(assignment_columns).write_parquet(search_dir / "pilot_cluster_assignments.parquet")
    _plot_search(metrics, search_dir)

    # Save normalized sequence information for reproducibility without duplicating
    # the full 1,200 x 728 pilot array (it can be reconstructed from row_index).
    search_payload = {
        "stage": "pilot_k_search",
        "input_manifest": str(input_dir / "manifest.json"),
        "n_series_in_population": int(metadata.height),
        "n_series_with_observed_history": int(X_all.shape[0]),
        "n_series_without_observed_history": int(metadata.height - X_all.shape[0]),
        "pilot_size": int(args.pilot_size),
        "silhouette_size": int(sil_size),
        "silhouette_distance_matrix_seconds": float(silhouette_distance_seconds),
        "pilot_sampling": "Proportional stratification by observed-history length band and activity_label",
        "sequence_representation": "Per-series z-normalized log1p observed suffix, reversed so d_1387 is index 0, NaN-padded to the right",
        "distance": "DTW with Sakoe-Chiba global constraint",
        "sakoe_chiba_radius_days": int(args.radius),
        "k_values": [int(k) for k in args.k_values],
        "search_max_iter": int(args.search_max_iter),
        "search_barycenter_max_iter": int(args.search_barycenter_iter),
        "n_init_per_k": 1,
        "random_state": int(args.random_state),
        "n_jobs": int(args.n_jobs),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "polars": pl.__version__,
            "tslearn": _version("tslearn"),
            "scikit_learn": _version("scikit-learn"),
        },
        "elapsed_seconds": float(time.perf_counter() - all_started),
        "input_window": input_manifest.get("window", {}),
        "notes": [
            "No K is selected automatically; inspect inertia, silhouette, cluster sizes and stability together.",
            "Silhouette is computed on a fixed subset of the pilot sample, using the same constrained DTW metric.",
            "tslearn silhouette_score is given only the subset, avoiding a full-population pairwise distance calculation.",
            "The 1,131 no-observation-history series remain outside demand-only fitting and are retained for later assignment output.",
        ],
    }
    _write_json(search_dir / "k_search_manifest.json", search_payload)
    print(f"\nK-search complete. Results: {search_dir}")


def _load_pilot_for_stability(
    search_dir: Path,
    X_all: np.ndarray,
    eligible_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    manifest_path = search_dir / "pilot_sample_manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pilot sample manifest not found: {manifest_path}; run --stage search first")
    sample = pl.read_parquet(manifest_path).sort("sample_position")
    row_ids = sample["row_index"].to_numpy().astype(np.int64)
    row_to_pos = {int(row): pos for pos, row in enumerate(eligible_rows)}
    try:
        positions = np.asarray([row_to_pos[int(row)] for row in row_ids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Pilot sample row_index {exc.args[0]} is no longer an observed-history series") from exc
    if positions.size == 0:
        raise ValueError("Pilot sample manifest is empty")
    if np.unique(positions).size != positions.size:
        raise ValueError("Pilot sample manifest contains duplicate row_index values")
    return X_all[positions], row_ids


def _stage_stability(args: argparse.Namespace, input_dir: Path, output_root: Path) -> None:
    x, mask, metadata, _calendar, _manifest = _load_inputs(input_dir)
    X_all, eligible_rows, lengths, _means, _scales, _constant = _build_reversed_sequences(x, mask, metadata)
    search_dir = output_root / "k_search"
    X_pilot, pilot_row_ids = _load_pilot_for_stability(search_dir, X_all, eligible_rows)
    seeds = args.seeds
    stability_dir = output_root / "stability"
    stability_dir.mkdir(parents=True, exist_ok=True)

    # Reuse seed-42 search labels when present; this keeps repeated fits to the
    # requested seeds while avoiding an unnecessary duplicate run.
    previous_assignments_path = search_dir / "pilot_cluster_assignments.parquet"
    previous = pl.read_parquet(previous_assignments_path) if previous_assignments_path.is_file() else None
    seed_labels: dict[tuple[int, int], np.ndarray] = {}
    if previous is not None and args.random_state in seeds:
        saved_rows = previous["row_index"].to_numpy().astype(np.int64)
        if np.array_equal(saved_rows, pilot_row_ids):
            for k in args.candidate_k_values:
                col = f"cluster_k{k}"
                if col in previous.columns:
                    seed_labels[(k, args.random_state)] = previous[col].to_numpy().astype(np.int32)

    assignment_table: dict[str, np.ndarray] = {"row_index": pilot_row_ids.astype(np.int64)}
    for (k, seed), labels in seed_labels.items():
        assignment_table[f"cluster_k{k}_seed{seed}"] = labels

    run_rows: list[dict[str, Any]] = []
    for k in args.candidate_k_values:
        for seed in seeds:
            cache_key = (k, seed)
            if cache_key in seed_labels:
                labels = seed_labels[cache_key]
                run_rows.append(
                    {
                        "n_clusters": k,
                        "random_state": seed,
                        "inertia": None,
                        "n_iter": None,
                        "fit_seconds": 0.0,
                        "reused_search_fit": True,
                    }
                )
                continue
            print(f"\nStability fit K={k}, seed={seed}...", flush=True)
            model = _make_model(
                n_clusters=k,
                random_state=seed,
                radius=args.radius,
                max_iter=args.stability_max_iter,
                max_iter_barycenter=args.stability_barycenter_iter,
                n_init=1,
                n_jobs=args.n_jobs,
                tol=args.tol,
            )
            started = time.perf_counter()
            labels = model.fit_predict(X_pilot)
            seconds = time.perf_counter() - started
            seed_labels[cache_key] = labels.astype(np.int32)
            assignment_table[f"cluster_k{k}_seed{seed}"] = labels.astype(np.int32)
            run_rows.append(
                {
                    "n_clusters": k,
                    "random_state": seed,
                    "inertia": float(model.inertia_),
                    "n_iter": int(model.n_iter_),
                    "fit_seconds": float(seconds),
                    "reused_search_fit": False,
                }
            )
            print(f"  inertia={model.inertia_:.6f}; iterations={model.n_iter_}; fit={seconds:.1f}s", flush=True)

    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    from sklearn.metrics import adjusted_rand_score

    for k in args.candidate_k_values:
        available = [(seed, seed_labels[(k, seed)]) for seed in seeds if (k, seed) in seed_labels]
        ari_values: list[float] = []
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                seed_a, labels_a = available[i]
                seed_b, labels_b = available[j]
                ari = float(adjusted_rand_score(labels_a, labels_b))
                ari_values.append(ari)
                pair_rows.append(
                    {
                        "n_clusters": k,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "adjusted_rand_index": ari,
                    }
                )
        summary_rows.append(
            {
                "n_clusters": k,
                "n_runs": len(available),
                "n_pairwise_comparisons": len(ari_values),
                "mean_ari": float(np.mean(ari_values)) if ari_values else None,
                "min_ari": float(np.min(ari_values)) if ari_values else None,
                "max_ari": float(np.max(ari_values)) if ari_values else None,
            }
        )

    pl.DataFrame(run_rows).write_csv(stability_dir / "stability_run_metrics.csv")
    pl.DataFrame(pair_rows).write_csv(stability_dir / "stability_pairwise_ari.csv")
    pl.DataFrame(summary_rows).write_csv(stability_dir / "stability_summary.csv")
    pl.DataFrame(assignment_table).write_parquet(stability_dir / "pilot_stability_assignments.parquet")
    _write_json(
        stability_dir / "stability_manifest.json",
        {
            "stage": "pilot_initialization_stability",
            "pilot_sample_size": int(X_pilot.shape[0]),
            "candidate_k_values": args.candidate_k_values,
            "random_seeds": seeds,
            "radius": int(args.radius),
            "metric": "DTW with Sakoe-Chiba constraint",
            "max_iter": int(args.stability_max_iter),
            "max_iter_barycenter": int(args.stability_barycenter_iter),
            "stability_measure": "Pairwise adjusted Rand index on the same pilot series",
            "notes": [
                "ARI is invariant to arbitrary cluster-label permutations.",
                "This is initialization stability on the pilot sample, not a statistical proof of population-level cluster validity.",
                "Seed-42 labels are reused from K-search where the exact pilot row order and K match.",
            ],
        },
    )
    print(f"\nStability analysis complete. Results: {stability_dir}")


def _stage_full_evaluate(args: argparse.Namespace, input_dir: Path, output_root: Path) -> None:
    """Evaluate a short list of pilot-selected K values on all observed series.

    Each K is fitted once (n_init=1) to keep this finalist stage bounded. A
    fixed silhouette sample is reused for every K so the full-population scores
    are directly comparable. Final refitting with multiple initializations is
    performed later by --stage fit after K is selected.
    """
    x, mask, metadata, _calendar, _manifest = _load_inputs(input_dir)
    X_all, eligible_rows, lengths, _means, _scales, _constant = _build_reversed_sequences(x, mask, metadata)
    if len(args.candidate_k_values) > 5:
        raise ValueError("Full-population evaluation is intended for at most five finalist K values")
    if max(args.candidate_k_values) >= X_all.shape[0]:
        raise ValueError("A candidate K must be smaller than the observed-history population")

    full_dir = output_root / "full_population_evaluation"
    full_dir.mkdir(parents=True, exist_ok=True)
    sil_positions = _proportional_stratified_sample(
        metadata,
        eligible_rows,
        lengths,
        min(args.fit_silhouette_size, X_all.shape[0]),
        args.random_state + 30_000,
    )
    X_silhouette = X_all[sil_positions]
    silhouette_rows = eligible_rows[sil_positions]
    print(
        f"Computing one shared full-population silhouette distance matrix "
        f"({sil_positions.size:,} sampled series)...",
        flush=True,
    )
    sil_distances, sil_distance_seconds = _silhouette_distance_matrix(
        X_silhouette, args.radius, args.n_jobs
    )
    pl.DataFrame(
        {
            "silhouette_position": np.arange(sil_positions.size, dtype=np.int32),
            "row_index": silhouette_rows.astype(np.int64),
        }
    ).join(metadata, on="row_index", how="left").write_parquet(
        full_dir / "full_population_silhouette_sample.parquet"
    )

    metric_rows: list[dict[str, Any]] = []
    for k in args.candidate_k_values:
        candidate_dir = full_dir / f"k{k}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = candidate_dir / "full_evaluation_metrics.json"
        if metrics_path.is_file() and not args.overwrite:
            cached = json.loads(metrics_path.read_text(encoding="utf-8"))
            metric_rows.append(cached)
            print(f"Skipping completed K={k}; found {metrics_path}", flush=True)
            continue

        print(
            f"\nFull-population evaluation K={k}: fitting {X_all.shape[0]:,} series "
            f"with n_init=1...",
            flush=True,
        )
        model = _make_model(
            n_clusters=k,
            random_state=args.random_state,
            radius=args.radius,
            max_iter=args.full_eval_max_iter,
            max_iter_barycenter=args.full_eval_barycenter_iter,
            n_init=1,
            n_jobs=args.n_jobs,
            tol=args.tol,
        )
        started = time.perf_counter()
        labels = model.fit_predict(X_all)
        fit_seconds = time.perf_counter() - started
        if np.unique(labels).size != k:
            raise RuntimeError(f"Full-population K={k} produced only {np.unique(labels).size} populated clusters")
        silhouette, silhouette_error, represented, _values = _silhouette_from_precomputed(
            sil_distances, labels[sil_positions], expected_n_clusters=k
        )
        cluster_sizes = np.bincount(labels, minlength=k)
        all_ids = np.full(EXPECTED_N_SERIES, -1, dtype=np.int32)
        all_ids[eligible_rows] = labels.astype(np.int32)
        nullable = [None if label < 0 else int(label) for label in all_ids]
        assignments = metadata.select("row_index", "item_id", "store_id", "cat_id", "dept_id", "state_id").with_columns(
            pl.Series("dtw_cluster_id", nullable, dtype=pl.Int32),
        )
        assignments.write_parquet(candidate_dir / "series_cluster_assignments.parquet")

        metric = {
            "n_clusters": int(k),
            "n_series_total": int(EXPECTED_N_SERIES),
            "n_series_assigned": int(labels.size),
            "n_series_without_history": int(EXPECTED_N_SERIES - labels.size),
            "inertia": float(model.inertia_),
            "silhouette_score": silhouette,
            "silhouette_error": silhouette_error,
            "n_clusters_represented_in_silhouette_sample": int(represented),
            "silhouette_sample_size": int(sil_positions.size),
            "min_cluster_size": int(cluster_sizes.min()),
            "median_cluster_size": float(np.median(cluster_sizes)),
            "max_cluster_size": int(cluster_sizes.max()),
            "n_iter": int(model.n_iter_),
            "fit_seconds": float(fit_seconds),
            "random_state": int(args.random_state),
            "n_init": 1,
            "max_iter": int(args.full_eval_max_iter),
            "max_iter_barycenter": int(args.full_eval_barycenter_iter),
            "radius": int(args.radius),
            "silhouette_distance_matrix_seconds_shared": float(sil_distance_seconds),
        }
        _write_json(metrics_path, metric)
        metric_rows.append(metric)
        print(
            f"  inertia={model.inertia_:.6f}; silhouette={silhouette}; "
            f"iterations={model.n_iter_}; fit={fit_seconds:.1f}s; "
            f"cluster sizes={cluster_sizes.tolist()}",
            flush=True,
        )
        pl.DataFrame(metric_rows).sort("n_clusters").write_csv(full_dir / "full_population_k_evaluation.csv")

    metrics_df = pl.DataFrame(metric_rows).sort("n_clusters")
    metrics_df.write_csv(full_dir / "full_population_k_evaluation.csv")
    _plot_search(metrics_df, full_dir)
    _write_json(
        full_dir / "full_population_evaluation_manifest.json",
        {
            "stage": "full_population_candidate_k_evaluation",
            "candidate_k_values": args.candidate_k_values,
            "n_series_total": EXPECTED_N_SERIES,
            "n_series_with_observed_history": int(X_all.shape[0]),
            "n_series_without_observed_history": int(EXPECTED_N_SERIES - X_all.shape[0]),
            "silhouette_sample_size": int(sil_positions.size),
            "silhouette_distance_matrix_seconds": float(sil_distance_seconds),
            "silhouette_sample_is_identical_for_all_candidate_k": True,
            "fit_n_init": 1,
            "random_state": int(args.random_state),
            "radius": int(args.radius),
            "notes": [
                "This stage evaluates only the finalist K values selected after pilot inertia/silhouette and pilot stability review.",
                "Each candidate is fitted on all series with observed history; the 1,131 no-history series remain unassigned and are represented in output tables.",
                "Each K uses n_init=1 for bounded finalist evaluation. The selected K is subsequently refitted with multiple initializations in the final fit stage.",
                "The same full-population silhouette sample and constrained-DTW distance matrix are reused for all candidate K values.",
            ],
        },
    )
    print(f"\nFull-population candidate K evaluation complete. Results: {full_dir}")


def _stage_fit(args: argparse.Namespace, input_dir: Path, output_root: Path) -> None:
    x, mask, metadata, calendar, input_manifest = _load_inputs(input_dir)
    X_all, eligible_rows, lengths, means, scales, constant = _build_reversed_sequences(x, mask, metadata)
    if args.n_clusters >= X_all.shape[0]:
        raise ValueError("n-clusters must be smaller than the number of series with observed history")

    fit_dir = output_root / f"model_k{args.n_clusters}"
    fit_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Fitting DTW K-means on {X_all.shape[0]:,} observed-history series; "
        f"K={args.n_clusters}, radius={args.radius}, n_init={args.n_init}...",
        flush=True,
    )
    model = _make_model(
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        radius=args.radius,
        max_iter=args.fit_max_iter,
        max_iter_barycenter=args.fit_barycenter_iter,
        n_init=args.n_init,
        n_jobs=args.n_jobs,
        tol=args.tol,
    )
    started = time.perf_counter()
    labels = model.fit_predict(X_all)
    fit_seconds = time.perf_counter() - started
    if labels.shape != (X_all.shape[0],):
        raise RuntimeError("Unexpected label vector shape returned by TimeSeriesKMeans")
    if np.unique(labels).size != args.n_clusters:
        raise RuntimeError(
            f"Expected {args.n_clusters} populated clusters; got {np.unique(labels).size}. "
            "Inspect the run before accepting the solution."
        )

    # Optional centroid distance matrix provides nearest-vs-second-nearest
    # assignment margins. It incurs one additional n_series x K DTW pass.
    centroid_distances: np.ndarray | None = None
    if args.save_centroid_distances:
        print("Calculating distances to fitted centroids for assignment margins...", flush=True)
        centroid_distances = np.asarray(model.transform(X_all), dtype=np.float32)
        if centroid_distances.shape != (X_all.shape[0], args.n_clusters):
            raise RuntimeError("Unexpected output shape from TimeSeriesKMeans.transform")

    # Build one output row per original series, preserving the preparation order.
    all_cluster_ids = np.full(EXPECTED_N_SERIES, -1, dtype=np.int32)
    all_cluster_ids[eligible_rows] = labels.astype(np.int32)
    all_n_live = metadata["n_price_live_days"].to_numpy().astype(np.int32)
    assignment_status = np.where(
        all_n_live == 0,
        "NO_OBSERVATION_HISTORY",
        np.where(all_n_live >= args.long_history_days, "ASSIGNED_LONG_HISTORY", "ASSIGNED_SHORT_HISTORY"),
    )
    cluster_id_nullable: list[int | None] = [None if value < 0 else int(value) for value in all_cluster_ids]
    assignments = metadata.with_columns(
        pl.Series("dtw_cluster_id", cluster_id_nullable, dtype=pl.Int32),
        pl.Series("dtw_assignment_status", assignment_status.tolist(), dtype=pl.String),
        pl.Series("dtw_sequence_length_days", all_n_live, dtype=pl.Int32),
        pl.Series("dtw_series_shape_is_constant", [None if n == 0 else bool(c) for n, c in zip(all_n_live, _expand_constant(constant, eligible_rows), strict=True)], dtype=pl.Boolean),
    )

    if centroid_distances is not None:
        nearest = np.full(EXPECTED_N_SERIES, np.nan, dtype=np.float32)
        second = np.full(EXPECTED_N_SERIES, np.nan, dtype=np.float32)
        margin = np.full(EXPECTED_N_SERIES, np.nan, dtype=np.float32)
        nearest[eligible_rows] = centroid_distances[np.arange(labels.size), labels]
        sorted_d = np.sort(centroid_distances, axis=1)
        second[eligible_rows] = sorted_d[:, 1] if args.n_clusters > 1 else np.nan
        margin[eligible_rows] = second[eligible_rows] - nearest[eligible_rows]
        assignments = assignments.with_columns(
            pl.Series("dtw_distance_to_assigned_centroid", nearest, dtype=pl.Float32),
            pl.Series("dtw_distance_to_second_centroid", second, dtype=pl.Float32),
            pl.Series("dtw_centroid_distance_margin", margin, dtype=pl.Float32),
        )

    assignments.write_parquet(fit_dir / "series_dtw_cluster_assignments.parquet")
    assignments.write_csv(fit_dir / "series_dtw_cluster_assignments.csv")

    assigned = assignments.filter(pl.col("dtw_cluster_id").is_not_null())
    summary = assigned.group_by("dtw_cluster_id").agg(
        pl.len().alias("n_series"),
        pl.col("n_price_live_days").mean().alias("mean_price_live_days"),
        pl.col("n_price_live_days").median().alias("median_price_live_days"),
        pl.col("price_live_fraction").mean().alias("mean_price_live_fraction"),
        pl.col("positive_day_frequency_on_price_days").mean().alias("mean_positive_day_frequency"),
        pl.col("mean_units_per_price_live_day").mean().alias("mean_units_per_price_live_day"),
        pl.col("mean_units_per_positive_sales_day").mean().alias("mean_units_per_positive_day"),
        pl.col("days_since_last_positive_sale").median().alias("median_days_since_last_sale"),
        (pl.col("coverage_label") == "FULL_WINDOW_PRICE_COVERAGE").sum().alias("n_full_coverage"),
        (pl.col("coverage_label") == "LEFT_TRUNCATED_PRICE_HISTORY").sum().alias("n_left_truncated"),
        (pl.col("activity_label") == "PRICE_RECORD_NO_POSITIVE_SALES").sum().alias("n_price_record_no_positive_sales"),
        pl.col("dtw_series_shape_is_constant").sum().alias("n_constant_shape_series"),
        (pl.col("dtw_assignment_status") == "ASSIGNED_SHORT_HISTORY").sum().alias("n_short_history_assignments"),
    ).sort("dtw_cluster_id")
    summary.write_parquet(fit_dir / "dtw_cluster_summary.parquet")
    summary.write_csv(fit_dir / "dtw_cluster_summary.csv")

    assignments.filter(pl.col("dtw_cluster_id").is_not_null()).group_by(
        ["dtw_cluster_id", "store_id"]
    ).agg(pl.len().alias("n_series")).sort(["dtw_cluster_id", "store_id"]).write_parquet(
        fit_dir / "dtw_cluster_store_composition.parquet"
    )
    assignments.filter(pl.col("dtw_cluster_id").is_not_null()).group_by(
        ["dtw_cluster_id", "cat_id"]
    ).agg(pl.len().alias("n_series")).sort(["dtw_cluster_id", "cat_id"]).write_parquet(
        fit_dir / "dtw_cluster_category_composition.parquet"
    )
    assignments.filter(pl.col("dtw_cluster_id").is_not_null()).group_by(
        ["dtw_cluster_id", "dept_id"]
    ).agg(pl.len().alias("n_series")).sort(["dtw_cluster_id", "dept_id"]).write_parquet(
        fit_dir / "dtw_cluster_department_composition.parquet"
    )

    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    if centers.ndim == 2:
        centers = centers[:, :, None]
    if centers.shape[0] != args.n_clusters or centers.shape[2] != 1:
        raise RuntimeError(f"Unexpected DTW centroid shape: {centers.shape}")
    reverse_offsets = np.arange(centers.shape[1], dtype=np.int32)
    d_nums = WINDOW_END_D - reverse_offsets
    date_map = dict(zip(calendar["d_num"].to_list(), calendar["date"].cast(pl.String).to_list(), strict=True))
    centroid_rows = {
        "dtw_cluster_id": np.repeat(np.arange(args.n_clusters, dtype=np.int32), centers.shape[1]),
        "reverse_offset_from_d1387": np.tile(reverse_offsets, args.n_clusters),
        "d_num": np.tile(d_nums, args.n_clusters),
        "date": np.tile([date_map[int(d)] for d in d_nums], args.n_clusters),
        "centroid_zscore": centers[:, :, 0].reshape(-1),
    }
    centroid_profiles = pl.DataFrame(centroid_rows)
    centroid_profiles.write_parquet(fit_dir / "dtw_centroid_profiles.parquet")
    centroid_profiles.write_csv(fit_dir / "dtw_centroid_profiles.csv")

    # Silhouette on a fixed, proportionally stratified sample from the fitted
    # full-population assignments. This assesses the final fit without building
    # a 29,359 x 29,359 distance matrix.
    sil_positions = _proportional_stratified_sample(
        metadata, eligible_rows, lengths, min(args.fit_silhouette_size, X_all.shape[0]), args.random_state + 20_000
    )
    sil_rows = eligible_rows[sil_positions]
    sil_labels = labels[sil_positions]
    sil_distances, sil_distance_seconds = _silhouette_distance_matrix(
        X_all[sil_positions], args.radius, args.n_jobs
    )
    sil_score, sil_error, represented, sil_values = _silhouette_from_precomputed(
        sil_distances, sil_labels, expected_n_clusters=args.n_clusters
    )
    silhouette_sample = pl.DataFrame(
        {
            "row_index": sil_rows.astype(np.int64),
            "dtw_cluster_id": sil_labels.astype(np.int32),
            "silhouette_value": sil_values if sil_values is not None else np.full(sil_positions.size, np.nan, dtype=np.float32),
        }
    ).join(assignments, on=["row_index", "dtw_cluster_id"], how="left")
    silhouette_sample.write_parquet(fit_dir / "dtw_silhouette_sample_assignments.parquet")

    with (fit_dir / "dtw_kmeans_model.pkl").open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

    n_long = int(np.sum(all_n_live >= args.long_history_days))
    n_short = int(np.sum((all_n_live > 0) & (all_n_live < args.long_history_days)))
    metrics = {
        "stage": "full_population_dtw_fit",
        "n_clusters": int(args.n_clusters),
        "n_series_total": EXPECTED_N_SERIES,
        "n_series_assigned": int(labels.size),
        "n_series_no_observation_history": int(EXPECTED_N_SERIES - labels.size),
        "n_assigned_long_history": n_long,
        "n_assigned_short_history": n_short,
        "metric": "DTW",
        "global_constraint": "sakoe_chiba",
        "sakoe_chiba_radius": int(args.radius),
        "sequence_representation": "Per-series z-normalized log1p observed suffix, reversed so d_1387 is sequence index 0, trailing NaN padding for variable lengths",
        "normalization": "Per-series mean/std using observed days only; constant profiles set to zero",
        "inertia": float(model.inertia_),
        "n_iter": int(model.n_iter_),
        "n_init": int(args.n_init),
        "random_state": int(args.random_state),
        "fit_seconds": float(fit_seconds),
        "silhouette_sample_size": int(sil_positions.size),
        "silhouette_score": sil_score,
        "silhouette_error": sil_error,
        "silhouette_distance_matrix_seconds": float(sil_distance_seconds),
        "n_clusters_represented_in_silhouette_sample": int(represented),
        "centroid_distances_saved": centroid_distances is not None,
        "long_history_days_threshold": int(args.long_history_days),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "polars": pl.__version__,
            "tslearn": _version("tslearn"),
            "scikit_learn": _version("scikit-learn"),
        },
        "input_window": input_manifest.get("window", {}),
        "notes": [
            "Only series with at least one observed price-record day participate in demand-only clustering.",
            "All 30,490 series remain in the assignment table; the 1,131 no-history series have null dtw_cluster_id.",
            "ASSIGNED_LONG_HISTORY/ASSIGNED_SHORT_HISTORY are based on own observed-history length, not pairwise DTW overlap.",
            "The mask is a price-record observation proxy, not a daily inventory/in-stock indicator.",
            "Z-normalization makes this a shape-oriented clustering; original level and volume metadata remain in assignments.",
            "DTW inertia is the estimator's objective and should only be compared across runs with the same representation and constraint.",
            "Silhouette is estimated on the specified sample using the same constrained DTW metric, not on the entire population.",
        ],
    }
    _write_json(fit_dir / "dtw_fit_metrics.json", metrics)
    _write_json(
        fit_dir / "dtw_fit_manifest.json",
        {
            **metrics,
            "input_directory": str(input_dir),
            "output_directory": str(fit_dir),
            "output_files": [
                "series_dtw_cluster_assignments.parquet",
                "dtw_cluster_summary.parquet",
                "dtw_cluster_store_composition.parquet",
                "dtw_cluster_category_composition.parquet",
                "dtw_cluster_department_composition.parquet",
                "dtw_centroid_profiles.parquet",
                "dtw_silhouette_sample_assignments.parquet",
                "dtw_kmeans_model.pkl",
            ],
        },
    )
    print(f"\nFull-population DTW fit complete. Results: {fit_dir}")
    print(f"  Assigned: {labels.size:,} / {EXPECTED_N_SERIES:,}")
    print(f"  K={args.n_clusters}; inertia={model.inertia_:.6f}; iterations={model.n_iter_}")
    print(f"  Sampled DTW silhouette: {sil_score}")
    print(f"  Long-history={n_long:,}; short-history={n_short:,}; no-history={EXPECTED_N_SERIES - labels.size:,}")


def _expand_constant(constant_eligible: np.ndarray, eligible_rows: np.ndarray) -> list[bool | None]:
    result: list[bool | None] = [None] * EXPECTED_N_SERIES
    for row_index, flag in zip(eligible_rows, constant_eligible, strict=True):
        result[int(row_index)] = bool(flag)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DTW clustering workflow for the prepared M5 dataset")
    parser.add_argument("--stage", choices=["smoke-test", "search", "stability", "full-evaluate", "fit"], default="smoke-test")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Prepared input directory (default: data/processed/clustering/d1387_728d)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="DTW output directory (default includes the selected Sakoe-Chiba radius)",
    )
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS, help="Sakoe-Chiba radius in daily sequence steps")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel workers for DTW distance calculations")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tol", type=float, default=1e-4)

    # K search parameters.
    parser.add_argument("--pilot-size", type=int, default=1200)
    parser.add_argument("--silhouette-size", type=int, default=400)
    parser.add_argument(
        "--k-values",
        type=lambda value: _parse_int_list(value, "k-values"),
        default=DEFAULT_K_VALUES,
        help="Pilot K grid, comma-separated; default is a coarse-to-fine 2..20 grid",
    )
    parser.add_argument("--search-max-iter", type=int, default=20)
    parser.add_argument("--search-barycenter-iter", type=int, default=8)

    # Stability parameters.
    parser.add_argument(
        "--candidate-k-values",
        type=lambda value: _parse_int_list(value, "candidate-k-values"),
        default=None,
        help="Candidate K values chosen after reviewing K-search curves; required for stability and full-evaluate stages",
    )
    parser.add_argument(
        "--seeds",
        type=lambda value: _parse_int_list(value, "seeds"),
        default=DEFAULT_SEEDS,
        help="Random seeds for repeat fits; default 42,43,44",
    )
    parser.add_argument("--stability-max-iter", type=int, default=20)
    parser.add_argument("--stability-barycenter-iter", type=int, default=8)
    parser.add_argument("--full-eval-max-iter", type=int, default=25)
    parser.add_argument("--full-eval-barycenter-iter", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="Recompute completed full-evaluation candidate K results")

    # Full-population fit parameters.
    parser.add_argument("--n-clusters", type=int, default=None, help="Final K chosen after search and stability review")
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--fit-max-iter", type=int, default=30)
    parser.add_argument("--fit-barycenter-iter", type=int, default=10)
    parser.add_argument("--fit-silhouette-size", type=int, default=500)
    parser.add_argument(
        "--long-history-days",
        type=int,
        default=DEFAULT_LONG_HISTORY_DAYS,
        help="Metadata-only threshold for long vs short observed histories; does not filter series",
    )
    parser.add_argument(
        "--save-centroid-distances",
        action="store_true",
        help="Run one additional DTW transform pass and save nearest/second-centroid margins",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.radius < 0:
        raise SystemExit("--radius must be >= 0")
    if args.long_history_days < 1:
        raise SystemExit("--long-history-days must be >= 1")
    if args.n_jobs == 0:
        raise SystemExit("--n-jobs cannot be 0; use 1 or a positive number, or -1 for all cores")
    if args.stage == "smoke-test":
        _stage_smoke_test(args.radius, args.n_jobs)
        return

    project_root = args.project_root.resolve()
    input_dir = args.input_dir.resolve() if args.input_dir else project_root / "data/processed/clustering/d1387_728d"
    output_root = args.output_dir.resolve() if args.output_dir else project_root / f"data/processed/clustering/d1387_728d/dtw_sakoe_chiba_r{args.radius}"

    if args.stage == "search":
        if not args.k_values or min(args.k_values) < 2:
            raise SystemExit("K-search values must all be >= 2")
        if args.pilot_size < max(args.k_values) + 2:
            raise SystemExit("pilot-size must be at least max(k-values) + 2")
        if args.silhouette_size < 20:
            raise SystemExit("silhouette-size must be at least 20")
        _stage_search(args, input_dir, output_root)
    elif args.stage == "stability":
        if not args.candidate_k_values:
            raise SystemExit("For --stage stability, provide --candidate-k-values after reviewing k_search_metrics.csv")
        if min(args.candidate_k_values) < 2:
            raise SystemExit("Candidate K values must all be >= 2")
        if len(args.seeds) < 2:
            raise SystemExit("Provide at least two random seeds for stability assessment")
        _stage_stability(args, input_dir, output_root)
    elif args.stage == "full-evaluate":
        if not args.candidate_k_values:
            raise SystemExit("For --stage full-evaluate, provide finalist --candidate-k-values from the pilot review")
        if min(args.candidate_k_values) < 2:
            raise SystemExit("Candidate K values must all be >= 2")
        _stage_full_evaluate(args, input_dir, output_root)
    elif args.stage == "fit":
        if args.n_clusters is None or args.n_clusters < 2:
            raise SystemExit("For --stage fit, provide --n-clusters K with K >= 2")
        if args.n_init < 1:
            raise SystemExit("n-init must be >= 1")
        _stage_fit(args, input_dir, output_root)


if __name__ == "__main__":
    main()
