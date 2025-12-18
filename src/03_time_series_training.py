"""
03 - Time Series Model Training

This script:
1. Loads selected features and labels
2. Builds time series sequences (use previous windows to predict the next window)
3. Trains LSTM, GRU, and Transformer models
4. Uses a validation set (either provided, or split from training) for early stopping and model selection
5. Saves the best model and training histories

Notes:
- Sequence length is configurable (currently set to 3, i.e., windows 1-3 -> predict window 4)
- Uses Top-M selected features
- Selects the best model primarily by F1 score (handles imbalance), with safeguards for NaN metrics
- If no explicit validation set exists, splits the last 20% of training sequences as validation (time-order split)
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

# Deep learning (TensorFlow/Keras)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Error: TensorFlow not installed, please run: pip install tensorflow")
    exit(1)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

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

# Sequence config
SEQUENCE_CONFIG = {
    "sequence_length": 3,  # Number of past windows used to predict the next window (1-3 -> 4)
    # Each group has ~30-40 windows.
    # Shorter sequences create more training samples but may capture less history.
}

# Number of classes (must match labels from 01_hmm_modeling.py)
N_CLASSES = 2  # 2/3/4 classes; must match HMM_N_CLASSES used to create labels

# Model configs
LSTM_CONFIG = {
    "lstm_units_1": 64,
    "lstm_units_2": 32,
    "dense_units": 16,
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 16,
    "epochs": 50,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
}

GRU_CONFIG = {
    "gru_units_1": 64,
    "gru_units_2": 32,
    "dense_units": 16,
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 16,
    "epochs": 50,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
}

TRANSFORMER_CONFIG = {
    "d_model": 32,
    "num_heads": 2,
    "num_layers": 2,
    "dff": 64,
    "dropout_rate": 0.1,
    "learning_rate": 0.001,
    "batch_size": 16,
    "epochs": 50,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
}

# Random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

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
    """
    X_list = []
    y_list = []

    # Reset index so label_array aligns by position
    data_df = data_df.reset_index(drop=True).copy()
    data_df['_label'] = label_array  # Keep labels aligned after sorting

    for group in data_df['group'].unique():
        group_data = data_df[data_df['group'] == group].sort_values('window_idx').reset_index(drop=True)
        group_features = group_data[feature_cols].values
        group_labels = group_data['_label'].values

        for i in range(len(group_data) - sequence_length):
            X_list.append(group_features[i:i + sequence_length])
            y_list.append(group_labels[i + sequence_length])

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


def evaluate_model(y_true, y_pred, y_proba=None, n_classes=2):
    """Evaluate classification metrics (supports multi-class; robust to edge cases)."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'accuracy': 0.0,
            'precision': np.nan,
            'recall': np.nan,
            'f1_score': np.nan,
            'auc_roc': np.nan
        }

    unique_true = np.unique(y_true)
    unique_labels = len(unique_true)

    # Single-class edge case
    if unique_labels == 1:
        correct = np.sum(y_true == y_pred)
        accuracy = correct / len(y_true)
        if accuracy == 1.0:
            return {
                'accuracy': 1.0,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0,
                'auc_roc': np.nan
            }
        else:
            try:
                all_labels = np.unique(np.concatenate([y_true, y_pred]))
                return {
                    'accuracy': accuracy,
                    'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=all_labels),
                    'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0, labels=all_labels),
                    'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=all_labels),
                    'auc_roc': np.nan
                }
            except Exception:
                return {
                    'accuracy': accuracy,
                    'precision': accuracy,
                    'recall': accuracy,
                    'f1_score': accuracy,
                    'auc_roc': np.nan
                }

    # Clip predictions into valid label range if needed
    if np.any(y_pred < 0) or np.any(y_pred >= n_classes):
        y_pred = np.clip(y_pred, 0, n_classes - 1)

    try:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
    except Exception:
        return {
            'accuracy': 0.0,
            'precision': np.nan,
            'recall': np.nan,
            'f1_score': np.nan,
            'auc_roc': np.nan
        }

    # AUC-ROC (binary or multi-class ovr)
    if y_proba is not None and unique_labels > 1:
        try:
            if n_classes > 2 and y_proba.ndim > 1 and y_proba.shape[1] > 1:
                if np.any(y_true < 0) or np.any(y_true >= n_classes):
                    metrics['auc_roc'] = np.nan
                else:
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


def build_lstm_model(input_shape, config, n_classes=4):
    """Build an LSTM classifier (softmax output, supports 2+ classes)."""
    model = models.Sequential([
        layers.LSTM(config['lstm_units_1'], return_sequences=True, input_shape=input_shape),
        layers.Dropout(config['dropout_rate']),
        layers.LSTM(config['lstm_units_2'], return_sequences=False),
        layers.Dropout(config['dropout_rate']),
        layers.Dense(config['dense_units'], activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])

    # Use sparse labels with softmax output
    loss = 'sparse_categorical_crossentropy'

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=loss,
        metrics=['accuracy']
    )
    return model


def build_gru_model(input_shape, config, n_classes=4):
    """Build a GRU classifier (softmax output, supports 2+ classes)."""
    model = models.Sequential([
        layers.GRU(config['gru_units_1'], return_sequences=True, input_shape=input_shape),
        layers.Dropout(config['dropout_rate']),
        layers.GRU(config['gru_units_2'], return_sequences=False),
        layers.Dropout(config['dropout_rate']),
        layers.Dense(config['dense_units'], activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])

    loss = 'sparse_categorical_crossentropy'

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=loss,
        metrics=['accuracy']
    )
    return model


def build_transformer_model(input_shape, config, n_classes=4):
    """Build a simplified Transformer classifier (softmax output, supports 2+ classes)."""
    inputs = layers.Input(shape=input_shape)

    # Project inputs to model dimension
    x = layers.Dense(config['d_model'])(inputs)

    for _ in range(config['num_layers']):
        attn_output = layers.MultiHeadAttention(
            num_heads=config['num_heads'],
            key_dim=config['d_model'] // config['num_heads']
        )(x, x)
        x = layers.LayerNormalization()(x + attn_output)

        ffn_output = layers.Dense(config['dff'], activation='relu')(x)
        ffn_output = layers.Dense(config['d_model'])(ffn_output)
        x = layers.LayerNormalization()(x + ffn_output)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(config['dropout_rate'])(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    loss = 'sparse_categorical_crossentropy'

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=loss,
        metrics=['accuracy']
    )
    return model


# ============================================================================
# Main
# ============================================================================

print("\n" + "="*80)
print("03 - Time Series Model Training")
print("="*80)

# 1. Load data
print("\n" + "-"*80)
print("1. Load Data and Selected Features")
print("-"*80)

train_data = load_intermediate('train_data')
train_val_data = load_intermediate('train_val_data')
val_data = load_intermediate('val_data')
test_data = load_intermediate('test_data')

y_train = load_intermediate('y_train')
y_train_val = load_intermediate('y_train_val')
y_val = load_intermediate('y_val')
y_test = load_intermediate('y_test')

top_m_features = load_intermediate('top_m_features')

print(f"\nNumber of selected features: {len(top_m_features)}")
print(f"Sequence length: {SEQUENCE_CONFIG['sequence_length']}")

# 2. Create sequences
print("\n" + "-"*80)
print("2. Create Time Series Sequences")
print("-"*80)

# Merge train + train_val if available
if len(train_val_data) > 0 and len(y_train_val) > 0:
    train_full_data = pd.concat([train_data, train_val_data], ignore_index=True)
    y_train_full = np.hstack([y_train, y_train_val])
else:
    train_full_data = train_data
    y_train_full = y_train
    if len(train_val_data) == 0:
        print("Note: train_val_data is empty, using only training set data")

X_train_seq, y_train_seq = create_sequences_by_group(
    train_full_data, top_m_features, y_train_full,
    sequence_length=SEQUENCE_CONFIG['sequence_length']
)

# If no explicit validation set, split last 20% of training sequences as validation (time-order split)
if len(val_data) > 0 and len(y_val) > 0:
    X_val_seq, y_val_seq = create_sequences_by_group(
        val_data, top_m_features, y_val,
        sequence_length=SEQUENCE_CONFIG['sequence_length']
    )
else:
    print("Note: Splitting 20% from training sequences as validation set (early stopping + model selection)")
    print("Using time-order split to preserve temporal dependencies")

    split_idx = int(len(X_train_seq) * 0.8)
    X_val_seq = X_train_seq[split_idx:]
    y_val_seq = y_train_seq[split_idx:]
    X_train_seq = X_train_seq[:split_idx]
    y_train_seq = y_train_seq[:split_idx]

    unique_labels, counts = np.unique(y_val_seq, return_counts=True)
    label_dist = dict(zip(unique_labels, counts))
    print(f"Validation set label distribution: {label_dist}")
    if len(unique_labels) < N_CLASSES:
        print(f"  ⚠️  Warning: Validation set has {len(unique_labels)} classes, but expects {N_CLASSES}")
        print("  This can happen with time-order splitting; some metrics may be unreliable")

X_test_seq, y_test_seq = create_sequences_by_group(
    test_data, top_m_features, y_test,
    sequence_length=SEQUENCE_CONFIG['sequence_length']
)

print(f"\nSequence data shapes:")
print(f"X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}")
print(f"X_val_seq: {X_val_seq.shape}, y_val_seq: {y_val_seq.shape}")
print(f"X_test_seq: {X_test_seq.shape}, y_test_seq: {y_test_seq.shape}")

# Label distributions
print(f"\nTraining set label distribution:")
unique_train, counts_train = np.unique(y_train_seq, return_counts=True)
print(f"  Labels: {dict(zip(unique_train, counts_train))}")
print(f"  Label range: [{np.min(y_train_seq)}, {np.max(y_train_seq)}]")
print(f"  Expected number of classes: {N_CLASSES}")

print(f"\nValidation set label distribution:")
unique_val, counts_val = np.unique(y_val_seq, return_counts=True)
print(f"  Labels: {dict(zip(unique_val, counts_val))}")
print(f"  Label range: [{np.min(y_val_seq)}, {np.max(y_val_seq)}]")
print(f"  Expected number of classes: {N_CLASSES}")

# Check for NaN/Inf
if np.any(np.isnan(X_train_seq)) or np.any(np.isinf(X_train_seq)):
    print("  ⚠️  Warning: Training sequences contain NaN or Inf!")
if np.any(np.isnan(X_val_seq)) or np.any(np.isinf(X_val_seq)):
    print("  ⚠️  Warning: Validation sequences contain NaN or Inf!")

input_shape = (SEQUENCE_CONFIG['sequence_length'], len(top_m_features))
print(f"\nInput shape: {input_shape}")

# 3. Train models
print("\n" + "-"*80)
print("3. Train Models")
print("-"*80)

models_trained = {}
histories = {}

# 3.1 LSTM
print("\n3.1 Training LSTM model...")
lstm_model = build_lstm_model(input_shape, LSTM_CONFIG, n_classes=N_CLASSES)
print("LSTM model structure:")
lstm_model.summary()

callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=LSTM_CONFIG['early_stopping_patience'],
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        patience=LSTM_CONFIG['reduce_lr_patience'],
        factor=LSTM_CONFIG.get('reduce_lr_factor', 0.5),
        min_lr=1e-7
    ),
    callbacks.ModelCheckpoint(
        str(MODELS_DIR / 'lstm_best.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
]

history_lstm = lstm_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=LSTM_CONFIG['epochs'],
    batch_size=LSTM_CONFIG['batch_size'],
    callbacks=callbacks_list,
    verbose=1
)

models_trained['LSTM'] = lstm_model
histories['LSTM'] = history_lstm.history

# 3.2 GRU
print("\n3.2 Training GRU model...")
gru_model = build_gru_model(input_shape, GRU_CONFIG, n_classes=N_CLASSES)

callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=GRU_CONFIG['early_stopping_patience'],
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        patience=GRU_CONFIG['reduce_lr_patience'],
        factor=GRU_CONFIG.get('reduce_lr_factor', 0.5),
        min_lr=1e-7
    ),
    callbacks.ModelCheckpoint(
        str(MODELS_DIR / 'gru_best.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
]

history_gru = gru_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=GRU_CONFIG['epochs'],
    batch_size=GRU_CONFIG['batch_size'],
    callbacks=callbacks_list,
    verbose=1
)

models_trained['GRU'] = gru_model
histories['GRU'] = history_gru.history

# 3.3 Transformer
print("\n3.3 Training Transformer model...")
transformer_model = build_transformer_model(input_shape, TRANSFORMER_CONFIG, n_classes=N_CLASSES)

callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=TRANSFORMER_CONFIG['early_stopping_patience'],
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        patience=TRANSFORMER_CONFIG['reduce_lr_patience'],
        factor=TRANSFORMER_CONFIG.get('reduce_lr_factor', 0.5),
        min_lr=1e-7
    ),
    callbacks.ModelCheckpoint(
        str(MODELS_DIR / 'transformer_best.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
]

history_transformer = transformer_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=TRANSFORMER_CONFIG['epochs'],
    batch_size=TRANSFORMER_CONFIG['batch_size'],
    callbacks=callbacks_list,
    verbose=1
)

models_trained['Transformer'] = transformer_model
histories['Transformer'] = history_transformer.history

# 4. Evaluate and select the best model (by F1, with NaN handling)
print("\n" + "-"*80)
print("4. Evaluate Models and Select Best Model")
print("-"*80)

model_scores = {}

for model_name, model in models_trained.items():
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} model")
    print(f"{'='*60}")

    y_pred_proba = model.predict(X_val_seq, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    if N_CLASSES == 2:
        y_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba.flatten()
    else:
        y_proba = y_pred_proba

    print(f"\nValidation set basic information:")
    print(f"  Sample count: {len(y_val_seq)}")
    print(f"  True label range: [{np.min(y_val_seq)}, {np.max(y_val_seq)}]")
    print(f"  Predicted label range: [{np.min(y_pred)}, {np.max(y_pred)}]")
    print(f"  Probability shape: {y_pred_proba.shape}")
    print(f"  Probability range: [{np.min(y_pred_proba):.4f}, {np.max(y_pred_proba):.4f}]")

    unique_true, counts_true = np.unique(y_val_seq, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    print(f"\nLabel distribution:")
    print(f"  True: {dict(zip(unique_true, counts_true))}")
    print(f"  Pred: {dict(zip(unique_pred, counts_pred))}")

    correct = np.sum(y_val_seq == y_pred)
    print(f"\nPrediction correctness:")
    print(f"  Correct: {correct}/{len(y_val_seq)}")
    print(f"  Accuracy: {correct/len(y_val_seq):.4f}")

    if correct == 0:
        print("\n⚠️  Warning: All predictions are wrong! First 5 samples:")
        for i in range(min(5, len(y_val_seq))):
            print(f"    Sample {i}: True={y_val_seq[i]}, Pred={y_pred[i]}, Proba={y_pred_proba[i]}")

    metrics = evaluate_model(y_val_seq, y_pred, y_proba, n_classes=N_CLASSES)
    model_scores[model_name] = metrics

    f1_str = f"{metrics['f1_score']:.4f}" if not np.isnan(metrics['f1_score']) else "nan"
    acc_str = f"{metrics['accuracy']:.4f}" if not np.isnan(metrics['accuracy']) else "nan"
    prec_str = f"{metrics['precision']:.4f}" if not np.isnan(metrics['precision']) else "nan"
    rec_str = f"{metrics['recall']:.4f}" if not np.isnan(metrics['recall']) else "nan"
    auc_str = f"{metrics['auc_roc']:.4f}" if not np.isnan(metrics['auc_roc']) else "nan"

    print(f"\n{model_name} - Validation performance:")
    print(f"  F1: {f1_str}")
    print(f"  Accuracy: {acc_str}")
    print(f"  Precision: {prec_str}")
    print(f"  Recall: {rec_str}")
    print(f"  AUC-ROC: {auc_str}")

    if metrics['accuracy'] == 0.0:
        print("  ⚠️  Critical Warning: Accuracy is 0.0 (all predictions wrong)")
    elif np.isnan(metrics['f1_score']) or np.isnan(metrics['precision']) or np.isnan(metrics['recall']):
        print("  ⚠️  Warning: Some metrics are NaN (likely single-class validation or other edge case)")

def get_f1_score(metrics):
    """Return F1 score; if NaN, return -1 for ranking."""
    f1 = metrics['f1_score']
    return f1 if not np.isnan(f1) else -1

best_model_name = max(model_scores, key=lambda x: get_f1_score(model_scores[x]))
best_model = models_trained[best_model_name]

best_f1 = model_scores[best_model_name]['f1_score']
if np.isnan(best_f1):
    print("\n⚠️  Warning: All models have NaN F1; selecting the first model by insertion order")
    best_model_name = list(model_scores.keys())[0]
    best_model = models_trained[best_model_name]
    print(f"Best model: {best_model_name} (F1: nan)")
else:
    print(f"\nBest model: {best_model_name} (F1: {best_f1:.4f})")

# 5. Save models and artifacts
print("\n" + "-"*80)
print("5. Save Models")
print("-"*80)

best_model.save(str(MODELS_DIR / f'{best_model_name.lower()}_final.h5'))
print(f"✓ Best model saved: {best_model_name.lower()}_final.h5")

save_intermediate('history_lstm', histories['LSTM'])
save_intermediate('history_gru', histories['GRU'])
save_intermediate('history_transformer', histories['Transformer'])

save_intermediate('model_scores', model_scores)
save_intermediate('best_model_name', best_model_name)

# 6. Write report
print("\n" + "-"*80)
print("6. Generate Training Report")
print("-"*80)

report_lines = []
report_lines.append("=" * 60)
report_lines.append("Time Series Model Training Report")
report_lines.append("=" * 60)
report_lines.append("\nConfiguration:")
report_lines.append(f"  Sequence length: {SEQUENCE_CONFIG['sequence_length']}")
report_lines.append(f"  Number of features: {len(top_m_features)}")
report_lines.append("\nModel Performance (Validation set for model selection):")

for model_name, metrics in model_scores.items():
    f1_str = f"{metrics['f1_score']:.4f}" if not np.isnan(metrics['f1_score']) else "nan"
    acc_str = f"{metrics['accuracy']:.4f}" if not np.isnan(metrics['accuracy']) else "nan"
    prec_str = f"{metrics['precision']:.4f}" if not np.isnan(metrics['precision']) else "nan"
    rec_str = f"{metrics['recall']:.4f}" if not np.isnan(metrics['recall']) else "nan"
    auc_str = f"{metrics['auc_roc']:.4f}" if not np.isnan(metrics['auc_roc']) else "nan"

    report_lines.append(f"\n{model_name}:")
    report_lines.append(f"  F1: {f1_str}")
    report_lines.append(f"  Accuracy: {acc_str}")
    report_lines.append(f"  Precision: {prec_str}")
    report_lines.append(f"  Recall: {rec_str}")
    report_lines.append(f"  AUC-ROC: {auc_str}")

report_lines.append(f"\nBest model: {best_model_name}")

report_text = "\n".join(report_lines)
print(report_text)

with open(REPORTS_DIR / "training_report.txt", 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\n✓ Report saved to {REPORTS_DIR / 'training_report.txt'}")

print("\n" + "="*80)
print("Time series model training completed!")
print("="*80)
print("\nNext step: Run `04_evaluation_and_reporting.py` for final evaluation")
