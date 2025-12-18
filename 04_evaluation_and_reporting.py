"""
04 - Model Evaluation and Reporting

This script:
1. Loads the best saved model and test data
2. Evaluates performance on the test set (never used in training)
3. Produces confusion matrix and classification report
4. Analyzes performance by group
5. Saves metrics, predictions, plots, and a final evaluation report

Notes:
- Focuses on F1 score for imbalanced data
- Uses sequence_length consistent with training (loaded from training script settings or set explicitly)
- Handles multi-class and edge cases where a group has only one class
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

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Error: TensorFlow not installed")
    exit(1)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Output paths
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

# Number of classes (must match labels from other scripts)
N_CLASSES = 2  # 2/3/4 classes; must match 01_hmm_modeling.py and 03_time_series_training.py

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


def create_sequences_by_group(data_df, feature_cols, label_array, sequence_length=10):
    """
    Create sequences per group (time-ordered by window_idx).

    Each sample uses windows [i : i+sequence_length] to predict label at window i+sequence_length.

    Returns:
      X: (n_sequences, sequence_length, n_features)
      y: (n_sequences,)
      groups: (n_sequences,)
    """
    X_list = []
    y_list = []
    group_list = []

    data_df = data_df.reset_index(drop=True).copy()
    data_df['_label'] = label_array

    for group in data_df['group'].unique():
        group_data = data_df[data_df['group'] == group].sort_values('window_idx').reset_index(drop=True)
        group_features = group_data[feature_cols].values
        group_labels = group_data['_label'].values

        for i in range(len(group_data) - sequence_length):
            X_list.append(group_features[i:i + sequence_length])
            y_list.append(group_labels[i + sequence_length])
            group_list.append(group)

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, np.array(group_list)


def evaluate_model(y_true, y_pred, y_proba=None, n_classes=2):
    """Evaluate classification metrics (supports multi-class; robust to single-class edge cases)."""
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

    if y_proba is not None and unique_labels > 1:
        try:
            if n_classes > 2 and y_proba.ndim > 1 and y_proba.shape[1] > 1:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            elif n_classes == 2 or (y_proba.ndim == 1 or y_proba.shape[1] == 2):
                y_proba_binary = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba_binary)
            else:
                metrics['auc_roc'] = np.nan
        except Exception:
            metrics['auc_roc'] = np.nan
    else:
        metrics['auc_roc'] = np.nan

    return metrics


# ============================================================================
# Main
# ============================================================================

print("\n" + "="*80)
print("04 - Model Evaluation and Reporting")
print("="*80)

# 1. Load data and model
print("\n" + "-"*80)
print("1. Load Data and Model")
print("-"*80)

test_data = load_intermediate('test_data')
y_test = load_intermediate('y_test')
top_m_features = load_intermediate('top_m_features')
best_model_name = load_intermediate('best_model_name')

sequence_length = 3  # Must match training (03_time_series_training.py)

print(f"\nBest model: {best_model_name}")
print(f"Test set size: {len(test_data)} windows")
print(f"Number of features: {len(top_m_features)}")

model_path = MODELS_DIR / f'{best_model_name.lower()}_final.h5'
if not model_path.exists():
    model_path = MODELS_DIR / f'{best_model_name.lower()}_best.h5'

best_model = keras.models.load_model(str(model_path))
print(f"✓ Model loaded: {model_path}")

# 2. Create test sequences
print("\n" + "-"*80)
print("2. Create Test Sequences")
print("-"*80)

X_test_seq, y_test_seq, test_groups = create_sequences_by_group(
    test_data, top_m_features, y_test,
    sequence_length=sequence_length
)

print(f"Test sequence shape: {X_test_seq.shape}")
print(f"Label shape: {y_test_seq.shape}")

# 3. Predict
print("\n" + "-"*80)
print("3. Model Prediction")
print("-"*80)

y_test_pred_proba = best_model.predict(X_test_seq)
y_test_pred = np.argmax(y_test_pred_proba, axis=1)

if N_CLASSES == 2:
    y_test_proba = y_test_pred_proba[:, 1] if y_test_pred_proba.shape[1] > 1 else y_test_pred_proba.flatten()
else:
    y_test_proba = y_test_pred_proba

print("Prediction completed")

if N_CLASSES == 4:
    print(f"Predicted label distribution: 0={np.sum(y_test_pred == 0)}, 1={np.sum(y_test_pred == 1)}, 2={np.sum(y_test_pred == 2)}, 3={np.sum(y_test_pred == 3)}")
    print(f"True label distribution: 0={np.sum(y_test_seq == 0)}, 1={np.sum(y_test_seq == 1)}, 2={np.sum(y_test_seq == 2)}, 3={np.sum(y_test_seq == 3)}")
elif N_CLASSES == 3:
    print(f"Predicted label distribution: 0={np.sum(y_test_pred == 0)}, 1={np.sum(y_test_pred == 1)}, 2={np.sum(y_test_pred == 2)}")
    print(f"True label distribution: 0={np.sum(y_test_seq == 0)}, 1={np.sum(y_test_seq == 1)}, 2={np.sum(y_test_seq == 2)}")
else:
    print(f"Predicted label distribution: 0={np.sum(y_test_pred == 0)}, 1={np.sum(y_test_pred == 1)}")
    print(f"True label distribution: 0={np.sum(y_test_seq == 0)}, 1={np.sum(y_test_seq == 1)}")

# 4. Overall evaluation
print("\n" + "-"*80)
print("4. Overall Performance Evaluation")
print("-"*80)

test_metrics = evaluate_model(y_test_seq, y_test_pred, y_test_proba, n_classes=N_CLASSES)

print("\nTest set performance:")
print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall: {test_metrics['recall']:.4f}")
print(f"  F1 Score: {test_metrics['f1_score']:.4f}")
auc_str = f"{test_metrics['auc_roc']:.4f}" if not np.isnan(test_metrics['auc_roc']) else "nan"
print(f"  AUC-ROC: {auc_str}")

cm = confusion_matrix(y_test_seq, y_test_pred)
print("\nConfusion matrix:")
print(cm)

print("\nClassification report:")
unique_labels = sorted(np.unique(np.concatenate([y_test_seq, y_test_pred])))
target_names = [f'状态{i}' for i in unique_labels]
print(classification_report(
    y_test_seq, y_test_pred,
    target_names=target_names,
    labels=unique_labels,
    zero_division=0
))

# 5. Group-wise analysis
print("\n" + "-"*80)
print("5. Performance Analysis by Group")
print("-"*80)

group_metrics = {}
for group in np.unique(test_groups):
    group_mask = test_groups == group
    group_y_true = y_test_seq[group_mask]
    group_y_pred = y_test_pred[group_mask]

    if N_CLASSES == 2:
        group_y_proba = y_test_proba[group_mask]
    else:
        group_y_proba = y_test_proba[group_mask]

    if len(group_y_true) > 0:
        group_metrics[group] = evaluate_model(group_y_true, group_y_pred, group_y_proba, n_classes=N_CLASSES)
        f1_str = f"{group_metrics[group]['f1_score']:.4f}" if not np.isnan(group_metrics[group]['f1_score']) else "nan"
        print(f"\nGroup {group}:")
        print(f"  Sample count: {len(group_y_true)}")
        print(f"  F1 Score: {f1_str}")
        print(f"  Accuracy: {group_metrics[group]['accuracy']:.4f}")

# 6. Save artifacts
print("\n" + "-"*80)
print("6. Save Results")
print("-"*80)

save_intermediate('test_predictions', y_test_pred)
save_intermediate('test_metrics', test_metrics)
save_intermediate('group_metrics', group_metrics)

# 7. Visualizations
print("\n" + "-"*80)
print("7. Generate Visualizations")
print("-"*80)

# 7.1 Confusion matrix plot
plt.figure(figsize=(8, 6))
if N_CLASSES == 4:
    labels = ['State 0', 'State 1', 'State 2', 'State 3']
elif N_CLASSES == 3:
    labels = ['Class 0', 'Class 1', 'Class 2']
else:
    labels = ['No Intervention', 'Intervention Needed']

sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)
plt.title('Test Set Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(VISUALIZATIONS_DIR / 'test_confusion_matrix.png', dpi=300)
print(f"✓ Confusion matrix saved: {VISUALIZATIONS_DIR / 'test_confusion_matrix.png'}")

# 7.2 Group performance plot
if len(group_metrics) > 0:
    groups = list(group_metrics.keys())
    f1_scores = [group_metrics[g]['f1_score'] for g in groups]
    accuracies = [group_metrics[g]['accuracy'] for g in groups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(groups, f1_scores)
    ax1.set_xlabel('Group')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score by Group')
    ax1.set_ylim([0, 1])

    ax2.bar(groups, accuracies)
    ax2.set_xlabel('Group')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy by Group')
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / 'group_performance.png', dpi=300)
    print(f"✓ Group performance comparison saved: {VISUALIZATIONS_DIR / 'group_performance.png'}")

# 8. Final report
print("\n" + "-"*80)
print("8. Generate Final Evaluation Report")
print("-"*80)

report_lines = []
report_lines.append("=" * 60)
report_lines.append("Final Evaluation Report")
report_lines.append("=" * 60)
report_lines.append("\nModel Information:")
report_lines.append(f"  Best model: {best_model_name}")
report_lines.append(f"  Number of features: {len(top_m_features)}")
report_lines.append(f"  Sequence length: {sequence_length}")

report_lines.append("\nTest Set Overall Performance:")
report_lines.append(f"  Accuracy: {test_metrics['accuracy']:.4f}")
report_lines.append(f"  Precision: {test_metrics['precision']:.4f}")
report_lines.append(f"  Recall: {test_metrics['recall']:.4f}")
report_lines.append(f"  F1 Score: {test_metrics['f1_score']:.4f}")
report_lines.append(f"  AUC-ROC: {auc_str}")

report_lines.append("\nConfusion Matrix:")
if cm.shape == (2, 2):
    report_lines.append(f"  [[{cm[0,0]}, {cm[0,1]}],")
    report_lines.append(f"   [{cm[1,0]}, {cm[1,1]}]]")
else:
    report_lines.append(str(cm))

report_lines.append("\nPerformance by Group:")
for group, metrics in sorted(group_metrics.items(), key=lambda x: str(x[0])):
    f1_str = f"{metrics['f1_score']:.4f}" if not np.isnan(metrics['f1_score']) else "nan"
    prec_str = f"{metrics['precision']:.4f}" if not np.isnan(metrics['precision']) else "nan"
    rec_str = f"{metrics['recall']:.4f}" if not np.isnan(metrics['recall']) else "nan"
    report_lines.append(f"\nGroup {group}:")
    report_lines.append(f"  F1 Score: {f1_str}")
    report_lines.append(f"  Accuracy: {metrics['accuracy']:.4f}")
    report_lines.append(f"  Precision: {prec_str}")
    report_lines.append(f"  Recall: {rec_str}")

report_text = "\n".join(report_lines)
print(report_text)

with open(REPORTS_DIR / "final_evaluation_report.txt", 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\n✓ Final report saved to {REPORTS_DIR / 'final_evaluation_report.txt'}")

print("\n" + "="*80)
print("Model evaluation completed!")
print("="*80)
print("\nAll results saved to outputs/ directory")
