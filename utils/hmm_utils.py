"""
HMM modeling utilities

This module provides:
- Train coarse-grained and fine-grained Gaussian HMMs (with KMeans initialization)
- Predict HMM states
- Map HMM states to labels for downstream supervised learning
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.cluster import KMeans
import config


def train_coarse_hmm(
    X_train,
    n_states=None,
    n_iter=None,
    covariance_type=None,
    random_state=None,
):
    """
    Train a coarse-grained Gaussian HMM.

    Parameters
    ----------
    X_train : np.ndarray | pd.DataFrame
        Training data (n_samples, n_features).
    n_states : int | None
        Number of hidden states. Default: config.HMM_CONFIG['coarse_n_states'].
    n_iter : int | None
        EM iterations. Default: config.HMM_CONFIG['n_iter'].
    covariance_type : str | None
        Covariance type. Default: config.HMM_CONFIG['covariance_type'].
    random_state : int | None
        Random seed. Default: config.HMM_CONFIG['random_state'].

    Returns
    -------
    hmm.GaussianHMM
        Trained coarse HMM.
    """
    if n_states is None:
        n_states = config.HMM_CONFIG["coarse_n_states"]
    if n_iter is None:
        n_iter = config.HMM_CONFIG["n_iter"]
    if covariance_type is None:
        covariance_type = config.HMM_CONFIG["covariance_type"]
    if random_state is None:
        random_state = config.HMM_CONFIG["random_state"]

    # Convert DataFrame to numpy array
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values

    # Initialize means via KMeans for a more stable start
    kmeans = KMeans(n_clusters=n_states, random_state=random_state, n_init=10)
    kmeans.fit(X_train)

    # init_params excludes 'm' so we can set means_ manually (not overwritten)
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        init_params="stc",
    )
    model.means_ = kmeans.cluster_centers_

    model.fit(X_train)
    return model


def train_fine_hmm(
    X_train,
    low_comm_mask,
    n_states=None,
    n_iter=None,
    covariance_type=None,
    random_state=None,
):
    """
    Train a fine-grained Gaussian HMM on samples flagged as "low communication".

    Parameters
    ----------
    X_train : np.ndarray | pd.DataFrame
        Full training data (n_samples, n_features).
    low_comm_mask : array-like (bool) or array-like (int)
        Mask or indices selecting the low-communication samples (from coarse state).
    n_states : int | None
        Number of fine states. Default: config.HMM_CONFIG['fine_n_states'].
    n_iter : int | None
        EM iterations. Default: config.HMM_CONFIG['n_iter'].
    covariance_type : str | None
        Covariance type. Default: config.HMM_CONFIG['covariance_type'].
    random_state : int | None
        Random seed. Default: config.HMM_CONFIG['random_state'].

    Returns
    -------
    hmm.GaussianHMM | None
        Trained fine HMM, or None if not enough samples.
    """
    if n_states is None:
        n_states = config.HMM_CONFIG["fine_n_states"]
    if n_iter is None:
        n_iter = config.HMM_CONFIG["n_iter"]
    if covariance_type is None:
        covariance_type = config.HMM_CONFIG["covariance_type"]
    if random_state is None:
        random_state = config.HMM_CONFIG["random_state"]

    # Select low-communication samples
    if isinstance(X_train, pd.DataFrame):
        X_values = X_train.values
    else:
        X_values = X_train

    low_comm_mask = np.asarray(low_comm_mask)
    X_low = X_values[low_comm_mask] if low_comm_mask.dtype == bool else X_values[low_comm_mask]

    if X_low.shape[0] < n_states:
        print(f"Warning: low-communication sample count ({X_low.shape[0]}) < n_states ({n_states}); skip fine HMM.")
        return None

    # KMeans initialization on the subset
    kmeans = KMeans(n_clusters=n_states, random_state=random_state, n_init=10)
    kmeans.fit(X_low)

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        init_params="stc",
    )
    model.means_ = kmeans.cluster_centers_

    model.fit(X_low)
    return model


def predict_hmm_states(model, X, group_col="group", window_col="window_idx"):
    """
    Predict HMM hidden states for observations.

    If X is a DataFrame, it will be sorted by (group, window) before predicting.

    Parameters
    ----------
    model : hmm.GaussianHMM
        Trained HMM.
    X : pd.DataFrame | np.ndarray
        Data to predict.
    group_col : str
        Group column name (only used if X is a DataFrame).
    window_col : str
        Window column name (only used if X is a DataFrame).

    Returns
    -------
    np.ndarray
        Predicted state sequence (aligned to the sorted order if DataFrame).
    """
    if isinstance(X, pd.DataFrame):
        X_sorted = X.sort_values([group_col, window_col]).reset_index(drop=True)
        X_values = X_sorted.drop(columns=[group_col, window_col], errors="ignore").values
    else:
        X_values = X

    return model.predict(X_values)


def map_states_to_labels(
    states,
    low_comm_state_idx=None,
    binary=True,
    fine_states=None,
    fine_to_binary=None,
):
    """
    Map coarse HMM states (and optional fine states) to labels.

    Parameters
    ----------
    states : np.ndarray
        Coarse state sequence.
    low_comm_state_idx : int | None
        Which coarse state is treated as "low communication". If None, uses last state id.
    binary : bool
        If True: output binary labels (0=no intervention, 1=need intervention).
        If False: output multi-class labels equal to coarse states.
    fine_states : np.ndarray | None
        Fine state sequence for the subset where coarse state == low_comm_state_idx.
    fine_to_binary : dict[int, int] | None
        Mapping from fine state -> binary label (only used when binary=True and fine_states is provided).

    Returns
    -------
    np.ndarray
        Label array.
    """
    states = np.asarray(states)

    if not binary:
        return states.copy()

    if low_comm_state_idx is None:
        low_comm_state_idx = int(states.max())  # default: treat the max state id as low-comm

    labels = (states == low_comm_state_idx).astype(int)

    # Optional refinement: within low-comm coarse state, use fine HMM states to decide intervention
    if fine_states is not None:
        fine_states = np.asarray(fine_states)
        low_mask = states == low_comm_state_idx

        if low_mask.sum() > 0:
            if fine_to_binary is None:
                # Default: map all fine states to intervention=1 (safe fallback)
                fine_to_binary = {int(s): 1 for s in np.unique(fine_states)}

            labels_low = np.array([fine_to_binary.get(int(s), 1) for s in fine_states], dtype=int)

            # Assign refined labels back only to the low-comm positions
            labels[low_mask] = labels_low[: low_mask.sum()]

    return labels
