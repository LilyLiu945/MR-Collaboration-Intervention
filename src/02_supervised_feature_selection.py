"""
02 - Supervised Feature Selection

This script:
1. Loads HMM-generated labels
2. Evaluates feature importance using multiple supervised methods:
   - Random Forest importance
   - Mutual Information
   - LASSO coefficients
   - RFE (Recursive Feature Elimination)
3. Combines scores and selects Top-M features (M <= 16 suggested; configurable)
4. Summarizes selection results
5. Saves selected features and score tables

Notes:
- Goal is to reduce redundancy and keep the most predictive features
- Evaluation of downstream model performance should be done in later training scripts
"""

# ============================================================================
# Config
# ============================================================================

import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
INTERMEDIATE_DIR.mkdir(exist_ok=True)
REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
VISUALIZATIONS_DIR.mkdir(exist_ok=True)

# Supervised feature selection config
SUPERVISED_CONFIG = {
    "top_m": 20,  # Select Top-M features from the full feature set
    # Total feature count increased (e.g., 54); keeping 20 is ~37%
    "rf_n_estimators": 100,
    "rf_max_depth": 10,
    "lasso_alpha": 0.01,
    "rfe_n_features": None,  # None means use top_m
}

# Multi-class config (must match labels produced by 01_hmm_modeling.py)
N_CLASSES = None  # Auto-detected from loaded labels

# Random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# Utility functions
# ============================================================================

def load_intermediate(name, directory=None):
    """Load an intermediate artifact from the intermediate directory."""
    if directory is None:
        directory = INTERMEDIATE_DIR
    filepath = directory / f"{name}.pkl"
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded: {filepath}")
    return data


def save_intermediate(name, data, directory=None):
    """Save an intermediate artifact to the intermediate directory."""
    if directory is None:
        directory = INTERMEDIATE_DIR
    filepath = directory / f"{name}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved: {filepath}")


def evaluate_model(y_true, y_pred, y_proba=None, n_classes=2):
    """Evaluate classification metrics (supports multi-class via weighted averaging)."""
    unique_labels = len(np.unique(y_true))
    if unique_labels < 2:
        return {
            'accuracy': 1.0 if len(y_true) == 0 else 0.0,
            'precision': np.nan,
            'recall': np.nan,
            'f1_score': np.nan,
            'auc_roc': np.nan
        }

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    # AUC-ROC (binary or multi-class ovr)
    if y_proba is not None and unique_labels > 1:
        try:
            if n_classes > 2 and y_proba.ndim > 1 and y_proba.shape[1] > 1:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            elif n_classes == 2 or (y_proba.ndim == 1 or y_proba.shape[1] == 2):
                if y_proba.ndim > 1:
                    y_proba_binary = y_proba[:, 1]
                else:
                    y_proba_binary = y_proba
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba_binary)
            else:
                metrics['auc_roc'] = np.nan
        except Exception as e:
            metrics['auc_roc'] = np.nan
    else:
        metrics['auc_roc'] = np.nan

    return metrics


# ============================================================================
# Main
# ============================================================================

print("\n" + "="*80)
print("02 - Supervised Feature Selection")
print("="*80)

# 1. Load data
print("\n" + "-"*80)
print("1. Load Data and Labels")
print("-"*80)

train_data = load_intermediate('train_data')
train_val_data = load_intermediate('train_val_data')
val_data = load_intermediate('val_data')
test_data = load_intermediate('test_data')
feature_names = load_intermediate('feature_names')

y_train = load_intermediate('y_train')
y_train_val = load_intermediate('y_train_val')
y_val = load_intermediate('y_val')
y_test = load_intermediate('y_test')

print(f"\nData shapes:")
print(f"Training set: {train_data.shape}, Labels: {y_train.shape}")
print(f"Test set: {test_data.shape}, Labels: {y_test.shape}")
print(f"Number of features: {len(feature_names)}")

# Extract feature columns
exclude_cols = ['group', 'window_idx']
feature_cols = [col for col in train_data.columns if col not in exclude_cols]

X_train = train_data[feature_cols].values

# Handle empty train_val_data (compatibility)
if len(train_val_data) > 0:
    X_train_val = train_val_data[feature_cols].values
else:
    X_train_val = np.array([]).reshape(0, len(feature_cols))

# Handle empty val_data (compatibility)
if len(val_data) > 0:
    X_val = val_data[feature_cols].values
else:
    X_val = np.array([]).reshape(0, len(feature_cols))

X_test = test_data[feature_cols].values

# Combine train + train_val for feature selection (if train_val exists)
if len(X_train_val) > 0 and len(y_train_val) > 0:
    X_train_full = np.vstack([X_train, X_train_val])
    y_train_full = np.hstack([y_train, y_train_val])
else:
    X_train_full = X_train
    y_train_full = y_train

print(f"\nData for feature selection:")
print(f"X_train_full: {X_train_full.shape}, y_train_full: {y_train_full.shape}")

# Auto-detect number of classes from labels
if N_CLASSES is None:
    N_CLASSES = len(np.unique(y_train_full))
    print(f"\nDetected number of classes: {N_CLASSES}")

# 2. Compute feature importance scores
print("\n" + "-"*80)
print("2. Evaluate Feature Importance")
print("-"*80)

feature_scores = pd.DataFrame({
    'feature': feature_cols
})

# 2.1 Random Forest importance
print("\n2.1 Random Forest Feature Importance...")
rf = RandomForestClassifier(
    n_estimators=SUPERVISED_CONFIG['rf_n_estimators'],
    max_depth=SUPERVISED_CONFIG['rf_max_depth'],
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced' if N_CLASSES > 2 else None
)
rf.fit(X_train_full, y_train_full)
feature_scores['rf_importance'] = rf.feature_importances_

# 2.2 Mutual Information
print("2.2 Mutual Information...")
mi_scores = mutual_info_classif(X_train_full, y_train_full, random_state=RANDOM_STATE)
feature_scores['mutual_info'] = mi_scores

# 2.3 LASSO coefficients (absolute value)
print("2.3 LASSO Coefficients...")
lasso = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5, random_state=RANDOM_STATE, max_iter=1000)
lasso.fit(X_train_full, y_train_full)
feature_scores['lasso_coef'] = np.abs(lasso.coef_)

# 2.4 RFE
print("2.4 RFE (Recursive Feature Elimination)...")
rfe_n_features = SUPERVISED_CONFIG['top_m'] if SUPERVISED_CONFIG['rfe_n_features'] is None else SUPERVISED_CONFIG['rfe_n_features']

rfe_estimator = RandomForestClassifier(
    n_estimators=50,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
if N_CLASSES and N_CLASSES > 2:
    rfe_estimator.set_params(class_weight='balanced')

rfe = RFE(
    estimator=rfe_estimator,
    n_features_to_select=rfe_n_features
)
rfe.fit(X_train_full, y_train_full)
feature_scores['rfe_rank'] = rfe.ranking_
feature_scores['rfe_selected'] = rfe.support_.astype(int)

# 3. Combine scores
print("\n" + "-"*80)
print("3. Combined Scoring")
print("-"*80)

feature_scores['rf_norm'] = (feature_scores['rf_importance'] - feature_scores['rf_importance'].min()) / (feature_scores['rf_importance'].max() - feature_scores['rf_importance'].min() + 1e-10)
feature_scores['mi_norm'] = (feature_scores['mutual_info'] - feature_scores['mutual_info'].min()) / (feature_scores['mutual_info'].max() - feature_scores['mutual_info'].min() + 1e-10)
feature_scores['lasso_norm'] = (feature_scores['lasso_coef'] - feature_scores['lasso_coef'].min()) / (feature_scores['lasso_coef'].max() - feature_scores['lasso_coef'].min() + 1e-10)
feature_scores['rfe_norm'] = 1.0 / feature_scores['rfe_rank']  # Lower rank -> higher score

feature_scores['combined_score'] = (
    feature_scores['rf_norm'] * 0.3 +
    feature_scores['mi_norm'] * 0.3 +
    feature_scores['lasso_norm'] * 0.2 +
    feature_scores['rfe_norm'] * 0.2
)

feature_scores = feature_scores.sort_values('combined_score', ascending=False)

print("\nFeature importance ranking (Top 10):")
print(feature_scores[['feature', 'combined_score', 'rf_norm', 'mi_norm', 'lasso_norm', 'rfe_norm']].head(10).to_string(index=False))

# 4. Select Top-M features
print("\n" + "-"*80)
print(f"4. Select Top-{SUPERVISED_CONFIG['top_m']} Features")
print("-"*80)

top_m_features = feature_scores.head(SUPERVISED_CONFIG['top_m'])['feature'].tolist()
print(f"\nSelected features ({len(top_m_features)}):")
for i, feat in enumerate(top_m_features, 1):
    print(f"  {i}. {feat}")

# 5. Summary
print("\n" + "-"*80)
print("5. Feature Selection Summary")
print("-"*80)

print(f"\nFeature selection completed:")
print(f"  - Selected {len(top_m_features)} most important features from {len(feature_cols)} features")
print(f"  - Selection ratio: {len(top_m_features)/len(feature_cols):.1%}")
print(f"\nNote: Downstream model performance should be evaluated in subsequent time series training")
print(f"      (Evaluating on the same training data used for selection would be misleading)")

# 6. Save results
print("\n" + "-"*80)
print("6. Save Results")
print("-"*80)

save_intermediate('top_m_features', top_m_features)
save_intermediate('feature_scores', feature_scores)
save_intermediate('supervised_feature_scores', feature_scores[['feature', 'combined_score']].to_dict('records'))

# 7. Write report
print("\n" + "-"*80)
print("7. Generate Feature Selection Report")
print("-"*80)

report_lines = []
report_lines.append("=" * 60)
report_lines.append("Supervised Feature Selection Report")
report_lines.append("=" * 60)
report_lines.append(f"\nConfiguration:")
report_lines.append(f"  Number of selected features: {SUPERVISED_CONFIG['top_m']}")
report_lines.append(f"  Total number of features: {len(feature_cols)}")
report_lines.append(f"\nSelected features:")
for i, feat in enumerate(top_m_features, 1):
    score = feature_scores[feature_scores['feature'] == feat]['combined_score'].values[0]
    report_lines.append(f"  {i}. {feat} (score: {score:.4f})")
report_lines.append(f"\nNotes:")
report_lines.append(f"  - Combined scores from 4 methods: RF importance, Mutual Information, LASSO, and RFE")
report_lines.append(f"  - Downstream performance evaluation should be done in later time series training scripts")
report_lines.append(f"  - Evaluating on the training set used for selection can be misleading")

report_text = "\n".join(report_lines)
print(report_text)

with open(REPORTS_DIR / "feature_selection_supervised_report.txt", 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\n✓ Report saved to {REPORTS_DIR / 'feature_selection_supervised_report.txt'}")

print("\n" + "="*80)
print("Supervised feature selection completed!")
print("="*80)
print("\nNext step: Run `03_time_series_training.py` for time series model training")
