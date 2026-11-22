#!/usr/bin/env python3
"""Patch the M5 Group A characterization script's validation row alignment.

Run from the M5 project root:
    pixi run python patch_m5_group_a_reproduction_diagnostic.py

By default patches scripts/characterize_m5_group_a_clusters.py, creates a .bak
backup, and syntax-compiles the patched target. Pass a different target path as
an optional positional argument.
"""
from __future__ import annotations

import ast
import re
import shutil
import sys
from pathlib import Path

DEFAULT_TARGET = Path("scripts/characterize_m5_group_a_clusters.py")

REPLACEMENT = '''def _validation_reproduction_check(method: str, k: int, seeds: list[int], validation_rows: list[int],
                                  predicted: dict[int, np.ndarray], archived: pl.DataFrame):
    """Compare reconstructed predictions to archived labels by row_index.

    The archived table is sorted by row_index for deterministic diagnostics,
    while predictions are returned in the original validation_rows order from
    sample_manifest.json. Align them by row ID before calculating exact-match
    fraction or ARI; comparing the arrays positionally creates a false mismatch.
    """
    from sklearn.metrics import adjusted_rand_score

    subset = archived.filter(pl.col("k") == k)
    expected_rows = [int(row) for row in validation_rows]
    if len(expected_rows) != len(set(expected_rows)):
        raise ValueError("Duplicate row_index values in validation_rows")

    out = []
    for seed in seeds:
        g = subset.filter(pl.col("seed") == seed).sort("row_index")
        archive_rows = [int(row) for row in g["row_index"].to_list()]
        archived_labels = np.asarray(g["cluster"].to_list(), dtype=np.int32)
        predicted_labels = np.asarray(predicted[seed], dtype=np.int32)

        if len(predicted_labels) != len(expected_rows):
            raise ValueError(
                f"Prediction length differs for {method}, K={k}, seed={seed}: "
                f"{len(predicted_labels)} predictions vs {len(expected_rows)} validation rows"
            )
        if len(archive_rows) != len(archived_labels):
            raise ValueError(
                f"Archived row/label length mismatch for {method}, K={k}, seed={seed}"
            )
        if len(archive_rows) != len(set(archive_rows)):
            raise ValueError(
                f"Duplicate archived row_index values for {method}, K={k}, seed={seed}"
            )
        if set(archive_rows) != set(expected_rows):
            raise ValueError(
                f"Validation row IDs differ for {method}, K={k}, seed={seed}: "
                f"archived={len(archive_rows)}, expected={len(expected_rows)}"
            )

        # The prediction vector is in manifest order, whereas archived labels
        # above are sorted by row_index. Re-key by row, then align to archive order.
        predicted_by_row = {
            int(row): int(label)
            for row, label in zip(expected_rows, predicted_labels, strict=True)
        }
        ordered_predictions = np.asarray(
            [predicted_by_row[row] for row in archive_rows], dtype=np.int32
        )

        out.append({
            "method": method,
            "k": k,
            "seed": seed,
            "n_validation": len(archive_rows),
            "exact_label_match_fraction": float(
                np.mean(ordered_predictions == archived_labels)
            ),
            "ari_vs_archived_assignment": float(
                adjusted_rand_score(archived_labels, ordered_predictions)
            ),
        })
    return out
'''


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET
    if not target.exists():
        raise FileNotFoundError(
            f"Could not find {target}. Run from the project root or pass the script path."
        )
    source = target.read_text()
    pattern = re.compile(
        r"(?ms)^def _validation_reproduction_check\(.*?(?=^def |\\Z)"
    )
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one top-level _validation_reproduction_check function; "
            f"found {len(matches)}. No file was changed."
        )

    # Ensure the target has the expected row-order bug before patching. If it
    # already has the corrected by-row mapping, don't rewrite or make a backup.
    old_function = matches[0].group(0)
    if "predicted_by_row" in old_function and "ordered_predictions" in old_function:
        print(f"Already patched: {target}")
        ast.parse(source)
        return

    backup = target.with_suffix(target.suffix + ".bak")
    suffix = 1
    while backup.exists():
        backup = target.with_suffix(target.suffix + f".bak.{suffix}")
        suffix += 1
    updated = source[:matches[0].start()] + REPLACEMENT + "\n\n" + source[matches[0].end():]
    ast.parse(updated)
    shutil.copy2(target, backup)
    target.write_text(updated)
    compile(updated, str(target), "exec")
    print(f"Patched validation row alignment: {target}")
    print(f"Backup: {backup}")
    print("Syntax compilation: passed")


if __name__ == "__main__":
    main()
