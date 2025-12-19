"""
Model Training & Evaluation Utilities

This module provides:
1) Group-wise time-series sequence construction (per group, sorted by window index)
2) Robust model evaluation for binary / multi-class settings (handles edge cases)
3) Saving a readable performance report

Key design choices (short summary):
- We build sequences per group to avoid "time leakage" across groups.
- We enforce alignment between data rows and y_labels by resetting index first.
- Metrics are computed in a robust way (single-class / missing proba won't crash).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import config


def create_sequences_by_group(
    data: pd.DataFrame,
    y_labels: Union[pd.Series, np.ndarray, List[int]],
    group_col: str = "group",
    window_col: str = "window_idx",
    selected_features: Optional[List[str]] = None,
    sequence_length: int = 10,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create time-series sequences independently for each group.

    What it does (short summary):
    - For each group, sort rows by time (window_col).
    - Build sequences of length `sequence_length`.
    - Predict the label `horizon` steps ahead (default horizon=1 -> next window).

    Parameters
    ----------
    data : pd.DataFrame
        Window-level dataframe containing group and window index columns.
        Each row should represent one time window.
    y_labels : pd.Series | np.ndarray | List[int]
        Labels aligned with the row order of `data`.
        IMPORTANT: y_labels must correspond 1-to-1 to rows of `data` (same order).
    group_col : str
        Column name of group identifier.
    window_col : str
        Column name of time/window index (used to sort within each group).
    selected_features : list[str] | None
        Feature columns to use. If None, automatically select numeric columns
        excluding typical identifier columns.
    sequence_length : int
        Number of historical windows used as input (look-back length).
    horizon : int
        Prediction horizon:
        - horizon=1 means predicting the next window label using past windows.
        - horizon=0 means predicting the current window label (less "forecasting").

    Returns
    -------
    X_sequences : np.ndarray
        Shape: (n_sequences, sequence_length, n_features)
    y_sequences : np.ndarray
        Shape: (n_sequences,)
    group_ids : np.ndarray
        Shape: (n_sequences,) group id for each sequence

    Notes
    -----
    Why reset index:
    - In many pipelines, y_labels is stored as a plain array matching row order.
      Resetting index ensures we can safely index y_labels by positional index.
    """
    # ---- Input validation (short summary: fail fast to avoid silent misalignment) ----
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be > 0, got {sequence_length}")
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")

    # ---- Feature selection (short summary: use numeric features unless user specifies) ----
    if selected_features is None:
        # Exclude common non-feature columns / identifiers
        exclude_cols = {group_col, window_col, "window_start", "pair", "has_data"}
        feature_cols = [
            col
            for col in data.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col])
        ]
    else:
        feature_cols = list(selected_features)

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found/selected.")

    # ---- Align data and labels (short summary: guarantee row-wise 1-to-1 mapping) ----
    data_reset = data.reset_index(drop=True).copy()

    if isinstance(y_labels, pd.Series):
        y_arr = y_labels.values
    else:
        y_arr = np.asarray(y_labels)

    # Ensure strict row alignment between data and y labels
    if len(y_arr) != len(data_reset):
        raise ValueError(
            f"y_labels length ({len(y_arr)}) does not match data rows ({len(data_reset)}). "
            "Make sure y_labels is aligned with data row order."
        )

    sequences: List[np.ndarray] = []
    labels: List[int] = []
    group_ids: List[int] = []

    # ---- Build sequences per group (short summary: prevent cross-group temporal leakage) ----
    for gid in sorted(data_reset[group_col].unique()):
        group_df = data_reset[data_reset[group_col] == gid].copy()

        # Sort by time index within this group
        group_df = group_df.sort_values(window_col, kind="mergesort")  # stable sort
        group_idx = group_df.index.values  # positional indices in data_reset

        # Extract features and labels for this group in time order
        X_group = group_df[feature_cols].values
        y_group = y_arr[group_idx]

        # We use "t" as the sequence end position (exclusive):
        # - X uses [t-sequence_length : t]
        # - y uses t + horizon - 1
        #
        # Valid t range:
        # - t >= sequence_length
        # - t + horizon - 1 < len(y_group)  -> t <= len(y_group) - horizon
        max_t = len(X_group) - horizon + 1  # last t is max_t-1
        for t in range(sequence_length, max_t):
            x_seq = X_group[t - sequence_length : t]
            y_target = y_group[t + horizon - 1]

            sequences.append(x_seq)
            labels.append(int(y_target))
            group_ids.append(int(gid))

    # ---- Handle empty result (short summary: return correctly shaped empty arrays) ----
    if len(sequences) == 0:
        n_features = len(feature_cols)
        return (
            np.empty((0, sequence_length, n_features), dtype=float),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
        )

    X_sequences = np.asarray(sequences)
    y_sequences = np.asarray(labels, dtype=int)
    group_ids_arr = np.asarray(group_ids, dtype=int)

    return X_sequences, y_sequences, group_ids_arr


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    binary: bool = True,
    pos_label: int = 1,
) -> Dict:
    """
    Evaluate model performance robustly (supports binary and multi-class).

    What it does (short summary):
    - Computes accuracy / precision / recall / F1.
    - Optionally computes ROC-AUC and Average Precision if probabilities provided.
    - Always returns a confusion matrix.
    - Handles edge cases like single-class y_true (AUC is not defined -> nan).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray | None
        Predicted probabilities:
        - Binary: shape (n,) or (n,2). If (n,2), we use column `pos_label`.
        - Multi-class: shape (n, n_classes).
    binary : bool
        If True, compute precision/recall/F1 in binary mode.
        If False, use weighted averaging for multi-class.
    pos_label : int
        Positive class label for binary metrics & probability column selection.

    Returns
    -------
    metrics : dict
        Contains:
        - accuracy, precision, recall, f1_score
        - auc_roc, average_precision (may be nan)
        - confusion_matrix
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # ---- Empty input guard (short summary: avoid division by zero / sklearn errors) ----
    if y_true.shape[0] == 0:
        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
            "auc_roc": np.nan,
            "average_precision": np.nan,
            "confusion_matrix": np.empty((0, 0), dtype=int),
        }

    metrics: Dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    # ---- Core classification metrics (short summary: binary vs weighted multi-class) ----
    if binary:
        metrics["precision"] = float(
            precision_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0)
        )
        metrics["f1_score"] = float(
            f1_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0)
        )
    else:
        metrics["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["f1_score"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # ---- Probability-based metrics (short summary: only if proba exists AND y_true has >=2 classes) ----
    metrics["auc_roc"] = np.nan
    metrics["average_precision"] = np.nan

    if y_proba is not None:
        y_proba = np.asarray(y_proba)

        try:
            # ROC-AUC requires at least 2 classes in y_true
            if len(np.unique(y_true)) >= 2:
                if binary:
                    # Accept (n,) or (n,2)
                    if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                        prob_pos = y_proba[:, pos_label]
                    else:
                        prob_pos = y_proba.reshape(-1)

                    metrics["auc_roc"] = float(roc_auc_score(y_true, prob_pos))
                    metrics["average_precision"] = float(average_precision_score(y_true, prob_pos))

                else:
                    # Multi-class AUC (OVR + weighted)
                    if y_proba.ndim == 2 and y_proba.shape[0] == y_true.shape[0] and y_proba.shape[1] >= 2:
                        metrics["auc_roc"] = float(
                            roc_auc_score(y_true, y_proba, average="weighted", multi_class="ovr")
                        )
        except Exception:
            # If any error occurs (shape mismatch, invalid labels, etc.), keep nan
            pass

    # ---- Confusion matrix (short summary: always returned) ----
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    return metrics


def _fmt_float(x) -> str:
    """
    Safely format a float-like value for report writing.

    Short summary:
    - Converts to float when possible.
    - Returns 'N/A' for None or non-numeric.
    - Returns 'nan' for NaN.
    """
    try:
        if x is None:
            return "N/A"
        x = float(x)
        if np.isnan(x):
            return "nan"
        return f"{x:.4f}"
    except Exception:
        return "N/A"


def save_model_report(metrics: Dict, model_name: str, save_path: Optional[Path] = None) -> None:
    """
    Save a human-readable performance report to a text file.

    What it does (short summary):
    - Writes accuracy / precision / recall / F1.
    - Writes AUC-ROC and Average Precision if present.
    - Writes confusion matrix.
    - Never crashes due to missing fields or non-numeric values.

    Parameters
    ----------
    metrics : dict
        Output from evaluate_model().
    model_name : str
        Model identifier used in the report header and default filename.
    save_path : Path | None
        If None, uses: config.REPORTS_DIR/{model_name}_performance.txt
    """
    if save_path is None:
        save_path = config.REPORTS_DIR / f"{model_name}_performance.txt"

    cm = metrics.get("confusion_matrix", None)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"=== {model_name} Performance Report ===\n\n")
        f.write(f"Accuracy: {_fmt_float(metrics.get('accuracy'))}\n")
        f.write(f"Precision: {_fmt_float(metrics.get('precision'))}\n")
        f.write(f"Recall: {_fmt_float(metrics.get('recall'))}\n")
        f.write(f"F1 Score: {_fmt_float(metrics.get('f1_score'))}\n")
        if "auc_roc" in metrics:
            f.write(f"ROC-AUC: {_fmt_float(metrics.get('auc_roc'))}\n")
        if "average_precision" in metrics:
            f.write(f"Average Precision: {_fmt_float(metrics.get('average_precision'))}\n")

        f.write("\nConfusion Matrix:\n")
        if cm is None:
            f.write("N/A\n")
        else:
            f.write(f"{cm}\n")

    print(f"Report saved: {save_path}")
