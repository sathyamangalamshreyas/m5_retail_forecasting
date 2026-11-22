from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.neighbors import kneighbors_graph

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_m5_group_b_graph_clustering.py"
SPEC = importlib.util.spec_from_file_location("group_b", SCRIPT)
group_b = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(group_b)


def test_no_history_rows_are_retained_as_nan_features() -> None:
    units = np.array(
        [
            [0, 0, 2, 0, 3, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 2, 0],
        ],
        dtype=np.int16,
    )
    mask = np.array(
        [
            [False, False, True, True, True, True],
            [False, False, False, True, True, True],
            [False, False, False, False, False, False],
            [False, False, True, True, True, True],
        ],
        dtype=bool,
    )
    features, eligible, lengths = group_b.extract_behavior_features(units, mask)
    assert eligible.tolist() == [0, 1, 3]
    assert lengths.tolist() == [4, 3, 0, 4]
    undefined_columns = [
        group_b.FEATURE_COLUMNS.index(name)
        for name in (
            "acf_lag_1", "acf_lag_7", "acf_lag_28",
            *[f"weekday_mean_z_{i}" for i in range(7)],
        )
    ]
    defined_columns = [i for i in range(features.shape[1]) if i not in undefined_columns]
    assert np.isfinite(features[np.ix_(eligible, defined_columns)]).all()
    assert np.isnan(features[2]).all()
    embedded, _, _ = group_b.prepare_feature_embedding(features, eligible, n_components=8, random_state=42)
    assert np.isfinite(embedded).all()


def test_shape_features_are_invariant_to_multiplicative_volume_scale() -> None:
    base = np.array([0, 2, 0, 3, 0, 0, 4, 0, 5, 0, 0, 3, 0, 2], dtype=np.float64)
    first = group_b._series_feature_row(base)
    second = group_b._series_feature_row(base * 10)
    # Event incidence, inter-arrival, zero-run and positive-size-CV features
    # are level-independent. The z-scored log1p trajectory can still change
    # slightly after rescaling because log1p is nonlinear.
    level_independent = list(range(0, 10))
    np.testing.assert_allclose(
        first[level_independent], second[level_independent], rtol=1e-9, atol=1e-9,
        equal_nan=True,
    )


def test_zero_distance_edges_survive_affinity_conversion() -> None:
    # Duplicate points create explicit zero-distance edges in sklearn's sparse
    # distance graph. They must remain edges with affinity 1, not disappear.
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    distances = kneighbors_graph(X, n_neighbors=2, mode="distance", include_self=False).tocsr()
    directed = group_b._trim_directed_knn(distances, 2)
    affinity, diagnostics = group_b.build_affinity_graph(directed, symmetrize="union")
    assert sparse.isspmatrix_csr(affinity)
    assert (affinity - affinity.T).nnz == 0
    assert affinity[0, 1] == 1.0
    assert affinity[1, 0] == 1.0
    assert diagnostics["n_isolated_nodes"] == 0


def test_spectral_and_louvain_both_assign_every_graph_node() -> None:
    rng = np.random.default_rng(17)
    points = np.vstack(
        [
            rng.normal(loc=(-3.0, 0.0), scale=0.25, size=(35, 2)),
            rng.normal(loc=(3.0, 0.0), scale=0.25, size=(35, 2)),
            rng.normal(loc=(0.0, 3.0), scale=0.25, size=(35, 2)),
        ]
    )
    distances = kneighbors_graph(points, n_neighbors=8, mode="distance", include_self=False).tocsr()
    affinity, graph_info = group_b.build_affinity_graph(distances, symmetrize="union")
    spectral_labels, _ = group_b.fit_spectral(affinity, n_clusters=3, seed=42, n_jobs=1)
    nx_graph = group_b._graph_to_networkx(affinity)
    louvain_labels = group_b.fit_louvain(nx_graph, resolution=1.0, seed=42)
    assert spectral_labels.shape == (len(points),)
    assert louvain_labels.shape == (len(points),)
    assert np.all(spectral_labels >= 0)
    assert np.all(louvain_labels >= 0)
    assert graph_info["n_isolated_nodes"] == 0
    assert group_b._cluster_size_metrics(spectral_labels)["n_clusters"] == 3
    assert group_b._cluster_size_metrics(louvain_labels)["n_clusters"] >= 1
