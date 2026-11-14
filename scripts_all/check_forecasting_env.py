from __future__ import annotations

import importlib.metadata as metadata
import platform
import sys

import numpy as np
import pandas as pd
import torch
from timestral import TimestralForecaster
import statsforecast
import utilsforecast


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {mps_available}")
    print(f"Timestral: {package_version('timestral')}")
    print(f"StatsForecast: {statsforecast.__version__}")
    print(f"UtilsForecast: {package_version('utilsforecast')}")
    print(f"TimestralForecaster import: {TimestralForecaster.__name__}")

    major = int(pd.__version__.split(".")[0])
    if major < 3:
        raise RuntimeError(
            "The forecasting environment is expected to use pandas 3.x. "
            "Check that you are running the default Pixi environment."
        )

    if not mps_available:
        print("WARNING: MPS is unavailable; PyTorch will use CPU where supported.")

    print("Forecasting environment check passed.")


if __name__ == "__main__":
    main()
