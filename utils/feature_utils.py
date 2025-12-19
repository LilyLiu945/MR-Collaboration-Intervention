"""
Feature engineering utilities

This module provides:
- Aggregate pairwise (edge-level) features into window-level features
- Normalize features using training-set statistics (StandardScaler)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def aggregate_pairwise_features(pairwise_df, group_col="group", window_col="window_idx"):
    """
    Aggregate pairwise features into window-level features.

    For each (group, window), compute: mean, std, max, min for numeric columns.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise feature DataFrame (one row per pair per window).
    group_col : str
        Group ID column name.
    window_col : str
        Window index column name.

    Returns
    -------
    pd.DataFrame
        Aggregated window-level feature DataFrame.
    """
    # Columns that should not be aggregated as features
    exclude_cols = [group_col, window_col, "pair", "window_start"]
    feature_cols = [c for c in pairwise_df.columns if c not in exclude_cols]

    grouped = pairwise_df.groupby([group_col, window_col], sort=True)

    aggregated_rows = []
    for (group, window), group_df in grouped:
        row = {group_col: group, window_col: window}

        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(group_df[col]):
                values = group_df[col].dropna().to_numpy()
                if values.size > 0:
                    row[f"{col}_mean"] = float(values.mean())
                    row[f"{col}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
                    row[f"{col}_max"] = float(values.max())
                    row[f"{col}_min"] = float(values.min())
                else:
                    row[f"{col}_mean"] = np.nan
                    row[f"{col}_std"] = np.nan
                    row[f"{col}_max"] = np.nan
                    row[f"{col}_min"] = np.nan

        aggregated_rows.append(row)

    return pd.DataFrame(aggregated_rows)


def normalize_features(
    train_df,
    val_df,
    test_df,
    train_val_df=None,
    exclude_cols=None,
    scaler=None,
):
    """
    Normalize features using training-set statistics (StandardScaler).

    Parameters
    ----------
    train_df : pd.DataFrame
        Training set.
    val_df : pd.DataFrame
        Validation set.
    test_df : pd.DataFrame
        Test set.
    train_val_df : pd.DataFrame | None
        Optional train-validation set (e.g., later time split from training groups).
    exclude_cols : list[str] | None
        Columns NOT to scale (e.g., group/window identifiers).
    scaler : StandardScaler | None
        Pre-fitted scaler; if None, fit on training data.

    Returns
    -------
    If train_val_df is None:
        train_scaled, val_scaled, test_scaled, scaler
    Else:
        train_scaled, train_val_scaled, val_scaled, test_scaled, scaler
    """
    if exclude_cols is None:
        exclude_cols = ["group", "window", "window_idx"]

    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # Extract feature matrices
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    X_train_val = train_val_df[feature_cols].copy() if train_val_df is not None else None

    # Fit/transform
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Rebuild DataFrames with original indices/columns
    train_scaled = train_df.copy()
    train_scaled[feature_cols] = pd.DataFrame(X_train_scaled, columns=feature_cols, index=train_df.index)

    val_scaled = val_df.copy()
    val_scaled[feature_cols] = pd.DataFrame(X_val_scaled, columns=feature_cols, index=val_df.index)

    test_scaled = test_df.copy()
    test_scaled[feature_cols] = pd.DataFrame(X_test_scaled, columns=feature_cols, index=test_df.index)

    if train_val_df is not None:
        X_train_val_scaled = scaler.transform(X_train_val)
        train_val_scaled = train_val_df.copy()
        train_val_scaled[feature_cols] = pd.DataFrame(
            X_train_val_scaled, columns=feature_cols, index=train_val_df.index
        )
        return train_scaled, train_val_scaled, val_scaled, test_scaled, scaler

    return train_scaled, val_scaled, test_scaled, scaler
