"""
Visualization Utilities

This module provides lightweight plotting helpers for:
1) Feature importance ranking plots
2) Correlation matrix heatmaps
3) Per-group (or overall) distribution histograms

Key idea (short summary):
- Keep visualization functions simple, reusable, and configurable (title/top_n/save_path).
- Use config.PLOT_DPI for consistent output quality across the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config


def plot_feature_importance(
    importance_scores: Union[Sequence[float], np.ndarray, pd.Series],
    feature_names: Union[Sequence[str], np.ndarray, pd.Series],
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Plot a horizontal bar chart of the top-N feature importances.

    Short summary:
    - Builds a DataFrame (feature, importance), sorts descending, keeps top_n.
    - Plots a horizontal bar chart with the most important feature on top.

    Parameters
    ----------
    importance_scores : array-like
        Importance scores for each feature (same length as feature_names).
    feature_names : array-like
        Names of the features (same length as importance_scores).
    top_n : int
        Number of top features to show.
    title : str
        Plot title.
    save_path : str | Path | None
        If provided, saves the figure to this path.

    Returns
    -------
    top_df : pd.DataFrame
        Sorted top-N DataFrame used for plotting (useful for reporting/debugging).

    Notes
    -----
    - This function assumes importance_scores aligns with feature_names by index/position.
    - For large top_n, the figure height scales to keep labels readable.
    """
    # ---- Build a tidy table (short summary: makes sorting & plotting safer) ----
    top_df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": np.asarray(importance_scores, dtype=float),
        }
    )

    # ---- Sort and select top-N (short summary: focus on the most informative features) ----
    top_df = top_df.sort_values("importance", ascending=False).head(int(top_n)).reset_index(drop=True)

    # ---- Plot (short summary: horizontal bars with inverted y-axis for ranking) ----
    plt.figure(figsize=(10, max(6, len(top_df) * 0.3)))
    plt.barh(range(len(top_df)), top_df["importance"], align="center")
    plt.yticks(range(len(top_df)), top_df["feature"])
    plt.xlabel("Importance Score")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # ---- Save if requested (short summary: consistent DPI across project) ----
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    return top_df


def plot_correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
    figsize: tuple = (12, 10),
    save_path: Optional[Union[str, Path]] = None,
    numeric_only: bool = True,
) -> pd.DataFrame:
    """
    Plot a correlation matrix heatmap.

    Short summary:
    - Computes df.corr(method=...) then visualizes it as a heatmap.
    - By default, uses numeric-only columns to avoid errors.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing feature columns.
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'.
    figsize : tuple
        Figure size.
    save_path : str | Path | None
        If provided, saves the figure to this path.
    numeric_only : bool
        If True, uses only numeric columns when computing correlation.

    Returns
    -------
    corr : pd.DataFrame
        Correlation matrix.

    Notes
    -----
    - Heatmaps can become unreadable for many features; consider plotting a subset.
    """
    # ---- Prepare data (short summary: correlation is defined for numeric features) ----
    corr_df = df.select_dtypes(include=[np.number]) if numeric_only else df

    # ---- Compute correlation (short summary: returns a square matrix) ----
    corr = corr_df.corr(method=method)

    # ---- Plot heatmap (short summary: centered at 0 to highlight +/- correlation) ----
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(f"Feature Correlation Matrix ({method})")
    plt.tight_layout()

    # ---- Save if requested ----
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    return corr


def plot_data_distribution(
    df: pd.DataFrame,
    column: str,
    group_col: str = "group",
    bins: int = 30,
    alpha: float = 0.5,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot a histogram of a column, optionally split by group.

    Short summary:
    - If group_col exists, overlays per-group histograms.
    - Otherwise, plots a single overall histogram.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column name to visualize.
    group_col : str
        Group id column name. If present in df, plots per-group distributions.
    bins : int
        Number of histogram bins.
    alpha : float
        Transparency for overlayed group histograms.
    save_path : str | Path | None
        If provided, saves the figure to this path.

    Notes
    -----
    - Overlayed histograms can be hard to read if there are many groups.
      In that case, consider faceting (subplots) or plotting summary stats instead.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe.")

    plt.figure(figsize=(12, 6))

    # ---- Plot per-group if possible (short summary: compare distributions across groups) ----
    if group_col in df.columns:
        groups = df[group_col].dropna().unique()
        for g in sorted(groups):
            series = df.loc[df[group_col] == g, column].dropna()
            if len(series) == 0:
                continue
            plt.hist(series, bins=bins, alpha=alpha, label=f"Group {g}")
        plt.legend()
    else:
        # ---- Plot overall distribution (short summary: single histogram) ----
        plt.hist(df[column].dropna(), bins=bins)

    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(f"{column} Distribution")
    plt.tight_layout()

    # ---- Save if requested ----
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
