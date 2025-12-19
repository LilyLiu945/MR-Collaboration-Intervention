"""
Configuration file - paths and hyperparameters

This module centralizes:
- Project/data/output paths
- Dataset split settings (train/val/test groups and time split)
- Feature selection configs (unsupervised + supervised)
- HMM settings
- Time-series model settings (LSTM/GRU/Transformer)
- Evaluation settings
"""

import os
from pathlib import Path

# ==================== Path settings ====================

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT
PAIRWISE_FEATURES_PATH = DATA_DIR / "pairwise_outputs" / "pairwise_features.csv"
WINDOWED_METRICS_PATH = DATA_DIR / "windowed_output" / "data" / "windowed_metrics.csv"
TASK_METRICS_PATH = DATA_DIR / "task_metrics_output" / "task_metrics_summary.csv"

# Output paths (created if missing)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
INTERMEDIATE_DIR.mkdir(exist_ok=True)

MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
VISUALIZATIONS_DIR.mkdir(exist_ok=True)

# ==================== Data split settings ====================

# Group-based split
TRAIN_GROUPS = list(range(1, 9))  # Groups 1-8
VAL_GROUPS = [9, 10]              # Groups 9-10
TEST_GROUPS = [11, 12]            # Groups 11-12

# Time split ratio within training groups
TRAIN_TIME_SPLIT = 0.7  # First 70% for training, last 30% for train-validation

# ==================== Feature selection settings ====================

# Stage 1: Unsupervised feature selection
UNSUPERVISED_CONFIG = {
    "variance_threshold": 0.01,       # Remove features with variance below this threshold
    "correlation_threshold": 0.95,    # Remove one of a pair of highly-correlated features
    "pca_n_components": None,         # Keep all components if None, or set an integer
    "pca_variance_ratio": 0.95,       # Keep enough components to explain this variance ratio
    "n_clusters": 5,                  # Number of clusters for K-means (if used)
    "top_k": 50,                      # Keep top-K features (tune based on validation)
}

# Stage 2.5: Supervised feature selection
SUPERVISED_CONFIG = {
    "top_m": 30,                # Keep top-M features (M <= K)
    "rf_n_estimators": 100,     # Number of trees in Random Forest
    "rf_max_depth": 10,         # Max depth for Random Forest
    "lasso_alpha": 0.01,        # LASSO regularization strength (if used)
    "rfe_n_features": None,     # RFE selected feature count (None means use default logic)
}

# ==================== HMM settings ====================

HMM_CONFIG = {
    "coarse_n_states": 4,         # Coarse-grained states (typically 3-4)
    "fine_n_states": 3,           # Fine-grained states (typically 2-3)
    "n_iter": 100,                # Baum-Welch iterations
    "covariance_type": "full",    # 'full', 'tied', 'diag', or 'spherical'
    "random_state": 42,
}

# ==================== Time-series model settings ====================

# Sequence construction
SEQUENCE_CONFIG = {
    "sequence_length": 10,  # Number of history windows used as input
}

# LSTM hyperparameters
LSTM_CONFIG = {
    "lstm_units_1": 128,
    "lstm_units_2": 64,
    "dense_units": 32,
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
}

# GRU hyperparameters
GRU_CONFIG = {
    "gru_units_1": 128,
    "gru_units_2": 64,
    "dense_units": 32,
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
}

# Transformer hyperparameters
TRANSFORMER_CONFIG = {
    "d_model": 64,           # Model dimension
    "num_heads": 4,          # Number of attention heads
    "num_layers": 2,         # Number of Transformer blocks
    "dff": 128,              # Feed-forward dimension
    "dropout_rate": 0.1,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
}

# ==================== Evaluation settings ====================

EVALUATION_CONFIG = {
    "primary_metric": "f1_score",  # Primary metric for imbalanced data
    "secondary_metric": "auc_roc", # Secondary metric
}

# ==================== Misc settings ====================

RANDOM_STATE = 42

# Display options (for pandas output)
DISPLAY_MAX_COLUMNS = None  # None means show all columns
DISPLAY_MAX_ROWS = 100

# Plot settings
PLOT_STYLE = "seaborn-v0_8"
PLOT_FIGSIZE = (12, 6)
PLOT_DPI = 100
