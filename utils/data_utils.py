"""
Data I/O utilities

This module provides:
- Save/load intermediate artifacts as pickle files
- Load raw CSV datasets (pairwise, windowed, task metrics)
- Basic data quality checks (missing values, duplicates, dtypes)
"""

import pickle
from pathlib import Path

import pandas as pd

import config


def save_intermediate(name, data, directory=None):
    """
    Save an intermediate artifact to the intermediate directory.

    Parameters
    ----------
    name : str
        File stem (without extension).
    data : Any
        Python object to pickle.
    directory : Path | None
        Target directory; defaults to config.INTERMEDIATE_DIR.
    """
    if directory is None:
        directory = config.INTERMEDIATE_DIR

    # Ensure directory exists (safe even if it already exists)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    filepath = directory / f"{name}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved: {filepath}")


def load_intermediate(name, directory=None):
    """
    Load an intermediate artifact from the intermediate directory.

    Parameters
    ----------
    name : str
        File stem (without extension).
    directory : Path | None
        Source directory; defaults to config.INTERMEDIATE_DIR.

    Returns
    -------
    Any
        Unpickled Python object.
    """
    if directory is None:
        directory = config.INTERMEDIATE_DIR

    directory = Path(directory)
    filepath = directory / f"{name}.pkl"
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded: {filepath}")
    return data


def load_all_data():
    """
    Load all raw CSV datasets.

    Returns
    -------
    pairwise_df : pd.DataFrame
        Pairwise feature dataset.
    windowed_df : pd.DataFrame
        Window-level network metrics dataset.
    task_df : pd.DataFrame
        Task-level performance metrics dataset.
    """
    print("Loading raw data files...")

    pairwise_df = pd.read_csv(config.PAIRWISE_FEATURES_PATH)
    print(f"✓ Pairwise features: {len(pairwise_df)} rows, {len(pairwise_df.columns)} columns")

    windowed_df = pd.read_csv(config.WINDOWED_METRICS_PATH)
    print(f"✓ Windowed metrics: {len(windowed_df)} rows, {len(windowed_df.columns)} columns")

    task_df = pd.read_csv(config.TASK_METRICS_PATH)
    print(f"✓ Task metrics: {len(task_df)} rows, {len(task_df.columns)} columns")

    return pairwise_df, windowed_df, task_df


def check_data_quality(df, name="data"):
    """
    Basic data quality checks: shape, missing values, duplicates, and dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    name : str
        Display name used in logs.
    """
    print(f"\n=== Data quality check: {name} ===")
    print(f"Shape: {df.shape}")

    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Missing values (non-zero):")
        print(missing[missing > 0].sort_values(ascending=False))
    else:
        print("Missing values: none")

    print(f"Duplicate rows: {df.duplicated().sum()}")

    # Show dtype counts (e.g., float64/int64/object)
    print("Dtype counts:")
    print(df.dtypes.value_counts())
