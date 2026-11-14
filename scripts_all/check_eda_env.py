from __future__ import annotations

import importlib.metadata as metadata
import sys

import pandas as pd
import dtale
from data_profiling import ProfileReport
from skimpy import skim


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"pandas: {pd.__version__}")
    print(f"fg-data-profiling: {metadata.version('fg-data-profiling')}")
    print(f"Skimpy: {metadata.version('skimpy')}")
    print(f"D-Tale: {metadata.version('dtale')}")
    print(f"D-Tale module: {dtale.__name__}")
    print(f"ProfileReport import: {ProfileReport.__name__}")
    print(f"Skimpy import: {skim.__name__}")

    major = int(pd.__version__.split(".")[0])
    if major >= 3:
        raise RuntimeError(
            "fg-data-profiling requires pandas <3.0. "
            "Run this check with `pixi run --environment eda check-env`."
        )

    print("EDA environment check passed.")


if __name__ == "__main__":
    main()
