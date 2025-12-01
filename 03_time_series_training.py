"""
03 - 时间序列模型训练

本脚本完成以下任务：
1. 加载选定的特征和标签
2. 构建时间序列序列（序列长度=5，预测未来窗口）
3. 训练LSTM、GRU、Transformer模型
4. 使用从训练集划分的验证集选择最佳模型
5. 保存最佳模型和训练历史

**专业分析**：
- 序列长度5（窗口1-5 → 窗口6，约80秒历史窗口）
- 使用选定的特征（Top-M）
- 比较LSTM、GRU、Transformer性能
- 选择F1分数最高的模型（考虑数据不平衡）
- 验证集从训练集中划分20%（用于早停和模型选择）
"""

# ============================================================================
# 配置部分
# ============================================================================

import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# 深度学习相关
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

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
INTERMEDIATE_DIR.mkdir(exist_ok=True)
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# 序列配置
SEQUENCE_CONFIG = {
    "sequence_length": 3,  # 历史窗口数（预测未来：窗口1-3 → 窗口4）
    # 注意：每组约30-40个窗口
    # 序列长度3：每组可创建27-37个序列（更多数据，可能缓解类别不平衡）
    # 序列长度5：每组可创建25-35个序列
    # 序列长度10：每组只能创建20-30个序列（数据利用率低）
}

# 多分类配置（需要与01_hmm_modeling.py中的HMM_N_CLASSES一致）
N_CLASSES = 2  # 2、3或4分类，必须与HMM_N_CLASSES一致（推荐2分类：需要干预 vs 不需要干预）

# 模型配置
LSTM_CONFIG = {
    "lstm_units_1": 64,  # 减少单元数以适应小数据集
    "lstm_units_2": 32,
    "dense_units": 16,
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 16,  # 减小batch size
    "epochs": 50,  # 减少epochs
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
    "d_model": 32,  # 减小模型维度
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

# 随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ============================================================================
# 工具函数部分
# ============================================================================

def load_intermediate(name, directory=None):
    """从intermediate目录加载中间结果"""
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
    """保存中间结果到intermediate目录"""
    if directory is None:
        directory = INTERMEDIATE_DIR
    filepath = directory / f"{name}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved: {filepath}")


def create_sequences_by_group(data_df, feature_cols, label_array, sequence_length=10):
    """
    按组创建时间序列序列
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        数据框（包含group和window_idx列）
    feature_cols : list
        特征列名列表
    label_array : np.ndarray
        标签数组（需要与data_df的行顺序对应）
    sequence_length : int
        序列长度
    
    Returns:
    --------
    X : np.ndarray
        序列数据 (n_sequences, sequence_length, n_features)
    y : np.ndarray
        标签 (n_sequences,)
    """
    X_list = []
    y_list = []
    
    # 重置索引以确保标签数组可以按位置索引访问
    data_df = data_df.reset_index(drop=True).copy()
    # 将标签添加到DataFrame中，这样排序时会自动对齐
    data_df['_label'] = label_array
    
    for group in data_df['group'].unique():
        # 获取该组的数据并按window_idx排序
        group_data = data_df[data_df['group'] == group].sort_values('window_idx').reset_index(drop=True)
        group_features = group_data[feature_cols].values
        group_labels = group_data['_label'].values
        
        # 创建序列（预测未来：窗口1-N → 窗口N+1）
        for i in range(len(group_data) - sequence_length):
            X_list.append(group_features[i:i+sequence_length])  # 窗口1-N的特征
            y_list.append(group_labels[i+sequence_length])      # 窗口N+1的标签（预测未来）
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y


def evaluate_model(y_true, y_pred, y_proba=None, n_classes=2):
    """评估模型性能（支持多分类）"""
    # 检查输入有效性
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'accuracy': 0.0,
            'precision': np.nan,
            'recall': np.nan,
            'f1_score': np.nan,
            'auc_roc': np.nan
        }
    
    # 检测实际分类数
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    unique_labels = len(unique_true)
    
    # 如果只有一个类别，需要特殊处理
    if unique_labels == 1:
        true_label = unique_true[0]
        correct = np.sum(y_true == y_pred)
        accuracy = correct / len(y_true)
        
        # 如果所有预测都正确
        if accuracy == 1.0:
            return {
                'accuracy': 1.0,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0,
                'auc_roc': np.nan  # 单类别无法计算AUC
            }
        else:
            # 单类别但预测了多个类别，使用weighted平均计算
            # 这种情况下，只有预测为真实类别的才算正确
            try:
                # 使用weighted平均，但需要确保所有类别都在labels参数中
                all_labels = np.unique(np.concatenate([y_true, y_pred]))
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=all_labels),
                    'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0, labels=all_labels),
                    'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=all_labels),
                }
            except Exception as e:
                # 如果计算失败，返回基于准确率的简单指标
                return {
                    'accuracy': accuracy,
                    'precision': accuracy,  # 单类别时，precision等于accuracy
                    'recall': accuracy,      # 单类别时，recall等于accuracy
                    'f1_score': accuracy,    # 单类别时，f1等于accuracy
                    'auc_roc': np.nan
                }
    
    # 确保预测标签在有效范围内
    if np.any(y_pred < 0) or np.any(y_pred >= n_classes):
        # 预测标签超出范围，修正为最接近的有效标签
        y_pred = np.clip(y_pred, 0, n_classes - 1)
    
    # 使用weighted平均支持多分类
    try:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
    except Exception as e:
        # 如果计算失败，返回默认值
        return {
            'accuracy': 0.0,
            'precision': np.nan,
            'recall': np.nan,
            'f1_score': np.nan,
            'auc_roc': np.nan
        }
    
    # AUC-ROC计算（支持多分类）
    if y_proba is not None and unique_labels > 1:
        try:
            if n_classes > 2 and y_proba.ndim > 1 and y_proba.shape[1] > 1:
                # 多分类：使用ovr策略
                # 确保y_true中的所有标签都在[0, n_classes-1]范围内
                if np.any(y_true < 0) or np.any(y_true >= n_classes):
                    metrics['auc_roc'] = np.nan
                else:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            elif n_classes == 2 or (y_proba.ndim == 1 or y_proba.shape[1] == 2):
                # 二分类
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


def build_lstm_model(input_shape, config, n_classes=4):
    """构建LSTM模型（多分类）"""
    model = models.Sequential([
        layers.LSTM(config['lstm_units_1'], return_sequences=True, input_shape=input_shape),
        layers.Dropout(config['dropout_rate']),
        layers.LSTM(config['lstm_units_2'], return_sequences=False),
        layers.Dropout(config['dropout_rate']),
        layers.Dense(config['dense_units'], activation='relu'),
        layers.Dense(n_classes, activation='softmax')  # 多分类输出
    ])
    
    # 对于2分类和多分类，都使用sparse_categorical_crossentropy
    # 因为输出层是softmax，输出2个值，标签是整数（0或1），所以用sparse版本
    loss = 'sparse_categorical_crossentropy'
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


def build_gru_model(input_shape, config, n_classes=4):
    """构建GRU模型（多分类）"""
    model = models.Sequential([
        layers.GRU(config['gru_units_1'], return_sequences=True, input_shape=input_shape),
        layers.Dropout(config['dropout_rate']),
        layers.GRU(config['gru_units_2'], return_sequences=False),
        layers.Dropout(config['dropout_rate']),
        layers.Dense(config['dense_units'], activation='relu'),
        layers.Dense(n_classes, activation='softmax')  # 多分类输出
    ])
    
    # 对于2分类和多分类，都使用sparse_categorical_crossentropy
    # 因为输出层是softmax，输出2个值，标签是整数（0或1），所以用sparse版本
    loss = 'sparse_categorical_crossentropy'
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


def build_transformer_model(input_shape, config, n_classes=4):
    """构建Transformer模型（简化版，多分类）"""
    inputs = layers.Input(shape=input_shape)
    
    # Positional encoding (简化版)
    x = layers.Dense(config['d_model'])(inputs)
    
    # Multi-head attention
    for _ in range(config['num_layers']):
        attn_output = layers.MultiHeadAttention(
            num_heads=config['num_heads'],
            key_dim=config['d_model'] // config['num_heads']
        )(x, x)
        x = layers.LayerNormalization()(x + attn_output)
        
        ffn_output = layers.Dense(config['dff'], activation='relu')(x)
        ffn_output = layers.Dense(config['d_model'])(ffn_output)
        x = layers.LayerNormalization()(x + ffn_output)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(config['dropout_rate'])(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)  # 多分类输出
    
    model = models.Model(inputs, outputs)
    # 对于2分类和多分类，都使用sparse_categorical_crossentropy
    # 因为输出层是softmax，输出2个值，标签是整数（0或1），所以用sparse版本
    loss = 'sparse_categorical_crossentropy'
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# 主程序部分
# ============================================================================

print("\n" + "="*80)
print("03 - Time Series Model Training")
print("="*80)

# 1. 加载数据
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

# 2. 创建时间序列序列
print("\n" + "-"*80)
print("2. Create Time Series Sequences")
print("-"*80)

# 合并训练集和训练验证集用于训练
if len(train_val_data) > 0 and len(y_train_val) > 0:
    train_full_data = pd.concat([train_data, train_val_data], ignore_index=True)
    y_train_full = np.hstack([y_train, y_train_val])
else:
    # 如果 train_val_data 为空，直接使用训练集
    train_full_data = train_data
    y_train_full = y_train
    if len(train_val_data) == 0:
        print("Note: train_val_data is empty, using only training set data")

X_train_seq, y_train_seq = create_sequences_by_group(
    train_full_data, top_m_features, y_train_full, 
    sequence_length=SEQUENCE_CONFIG['sequence_length']
)

# 检查验证集是否为空（已合并到测试集）
if len(val_data) > 0 and len(y_val) > 0:
    X_val_seq, y_val_seq = create_sequences_by_group(
        val_data, top_m_features, y_val,
        sequence_length=SEQUENCE_CONFIG['sequence_length']
    )
else:
    # 如果验证集为空，从训练集中划分20%作为验证集（用于早停和模型选择）
    print("Note: Splitting 20% from training set as validation set (for early stopping and model selection)")
    print("Using temporal order split (preserving temporal dependencies of time series)")
    
    # 时间序列数据：使用简单的前80%后20%划分，保持时间顺序
    # 注意：这可能导致验证集类别不平衡，但这是时间序列数据的标准做法
    split_idx = int(len(X_train_seq) * 0.8)
    X_val_seq = X_train_seq[split_idx:]
    y_val_seq = y_train_seq[split_idx:]
    X_train_seq = X_train_seq[:split_idx]
    y_train_seq = y_train_seq[:split_idx]
    
    # 打印验证集标签分布（用于调试）
    unique_labels, counts = np.unique(y_val_seq, return_counts=True)
    label_dist = dict(zip(unique_labels, counts))
    print(f"Validation set label distribution: {label_dist}")
    if len(unique_labels) < N_CLASSES:
        print(f"  ⚠️  Warning: Validation set only has {len(unique_labels)} classes, but expects {N_CLASSES} classes")
        print(f"  This is a common issue with time series data splitting: later time windows may only contain certain classes")
        print(f"  Evaluation metrics (e.g., F1, Precision, Recall) may be inaccurate, but accuracy is still valid")

X_test_seq, y_test_seq = create_sequences_by_group(
    test_data, top_m_features, y_test,
    sequence_length=SEQUENCE_CONFIG['sequence_length']
)

print(f"\nSequence data shapes:")
print(f"X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}")
print(f"X_val_seq: {X_val_seq.shape}, y_val_seq: {y_val_seq.shape}")
print(f"X_test_seq: {X_test_seq.shape}, y_test_seq: {y_test_seq.shape}")

# 检查标签分布
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

# 检查数据是否有NaN或Inf
if np.any(np.isnan(X_train_seq)) or np.any(np.isinf(X_train_seq)):
    print(f"  ⚠️  Warning: Training sequence data contains NaN or Inf values!")
if np.any(np.isnan(X_val_seq)) or np.any(np.isinf(X_val_seq)):
    print(f"  ⚠️  Warning: Validation sequence data contains NaN or Inf values!")

input_shape = (SEQUENCE_CONFIG['sequence_length'], len(top_m_features))
print(f"\nInput shape: {input_shape}")

# 3. 训练模型
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

# 4. 评估模型并选择最佳模型
print("\n" + "-"*80)
print("4. Evaluate Models and Select Best Model")
print("-"*80)

model_scores = {}

for model_name, model in models_trained.items():
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} model")
    print(f"{'='*60}")
    
    # 预测
    y_pred_proba = model.predict(X_val_seq, verbose=0)
    # 对于2分类和多分类，都使用argmax（因为输出层是softmax，输出n_classes个概率值）
    y_pred = np.argmax(y_pred_proba, axis=1)  # 取概率最大的类别
    if N_CLASSES == 2:
        # 2分类：y_pred_proba形状是(N, 2)，y_proba取第二列（类别1的概率）用于AUC计算
        y_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba.flatten()
    else:
        # 多分类：使用完整概率矩阵
        y_proba = y_pred_proba
    
    # 详细调试信息
    print(f"\nValidation set basic information:")
    print(f"  Validation set sample count: {len(y_val_seq)}")
    print(f"  True label range: [{np.min(y_val_seq)}, {np.max(y_val_seq)}]")
    print(f"  Predicted label range: [{np.min(y_pred)}, {np.max(y_pred)}]")
    print(f"  Prediction probability shape: {y_pred_proba.shape}")
    print(f"  Prediction probability range: [{np.min(y_pred_proba):.4f}, {np.max(y_pred_proba):.4f}]")
    
    # 标签分布
    unique_true, counts_true = np.unique(y_val_seq, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    print(f"\nLabel distribution:")
    print(f"  True labels: {dict(zip(unique_true, counts_true))}")
    print(f"  Predicted labels: {dict(zip(unique_pred, counts_pred))}")
    
    # 检查预测是否正确
    correct = np.sum(y_val_seq == y_pred)
    print(f"\nPrediction correctness:")
    print(f"  Correct predictions: {correct}/{len(y_val_seq)}")
    print(f"  Accuracy: {correct/len(y_val_seq):.4f}")
    
    # 如果准确率为0，打印一些样本的详细信息
    if correct == 0:
        print(f"\n⚠️  Warning: All predictions are wrong! Details of first 5 samples:")
        for i in range(min(5, len(y_val_seq))):
            print(f"    Sample {i}: True={y_val_seq[i]}, Predicted={y_pred[i]}, Probability={y_pred_proba[i]}")
    
    metrics = evaluate_model(y_val_seq, y_pred, y_proba, n_classes=N_CLASSES)
    model_scores[model_name] = metrics
    
    print(f"\n{model_name} - Validation set performance (split from training set, for model selection):")
    # 统一处理所有指标，避免重复打印
    f1_str = f"{metrics['f1_score']:.4f}" if not np.isnan(metrics['f1_score']) else "nan (cannot calculate)"
    acc_str = f"{metrics['accuracy']:.4f}" if not np.isnan(metrics['accuracy']) else "nan (cannot calculate)"
    prec_str = f"{metrics['precision']:.4f}" if not np.isnan(metrics['precision']) else "nan (cannot calculate)"
    rec_str = f"{metrics['recall']:.4f}" if not np.isnan(metrics['recall']) else "nan (cannot calculate)"
    auc_str = f"{metrics['auc_roc']:.4f}" if not np.isnan(metrics['auc_roc']) else "nan (cannot calculate)"
    
    print(f"  F1 Score: {f1_str}")
    print(f"  Accuracy: {acc_str}")
    print(f"  Precision: {prec_str}")
    print(f"  Recall: {rec_str}")
    print(f"  AUC-ROC: {auc_str}")
    
    if metrics['accuracy'] == 0.0:
        print(f"  ⚠️  Critical Warning: Model accuracy is 0, all predictions are wrong!")
        print(f"     Possible reasons: 1) Model training failed 2) Data preprocessing issue 3) Label mismatch")
    elif np.isnan(metrics['f1_score']) or np.isnan(metrics['precision']) or np.isnan(metrics['recall']):
        print(f"  ⚠️  Warning: Some metrics cannot be calculated (possibly due to insufficient classes in validation set or prediction failure)")

# 选择F1分数最高的模型（处理nan情况）
def get_f1_score(metrics):
    """获取F1分数，如果是nan则返回-1"""
    f1 = metrics['f1_score']
    return f1 if not np.isnan(f1) else -1

best_model_name = max(model_scores, key=lambda x: get_f1_score(model_scores[x]))
best_model = models_trained[best_model_name]

best_f1 = model_scores[best_model_name]['f1_score']
if np.isnan(best_f1):
    print(f"\n⚠️  Warning: All models have nan F1 scores, selecting first model as best model")
    best_model_name = list(model_scores.keys())[0]
    best_model = models_trained[best_model_name]
    print(f"Best model: {best_model_name} (F1: nan)")
else:
    print(f"\nBest model: {best_model_name} (F1: {best_f1:.4f})")

# 5. 保存模型
print("\n" + "-"*80)
print("5. Save Models")
print("-"*80)

# 保存最佳模型
best_model.save(str(MODELS_DIR / f'{best_model_name.lower()}_final.h5'))
print(f"✓ Best model saved: {best_model_name.lower()}_final.h5")

# 保存所有模型的历史
save_intermediate('history_lstm', histories['LSTM'])
save_intermediate('history_gru', histories['GRU'])
save_intermediate('history_transformer', histories['Transformer'])

# 保存模型评估结果
save_intermediate('model_scores', model_scores)
save_intermediate('best_model_name', best_model_name)

# 6. 生成报告
print("\n" + "-"*80)
print("6. Generate Training Report")
print("-"*80)

report_lines = []
report_lines.append("=" * 60)
report_lines.append("Time Series Model Training Report")
report_lines.append("=" * 60)
report_lines.append(f"\nConfiguration:")
report_lines.append(f"  Sequence length: {SEQUENCE_CONFIG['sequence_length']}")
report_lines.append(f"  Number of features: {len(top_m_features)}")
report_lines.append(f"\nModel Performance (Validation set: 20% split from training set, for model selection):")
for model_name, metrics in model_scores.items():
    report_lines.append(f"\n{model_name}:")
    # 统一处理所有指标，避免重复打印
    f1_str = f"{metrics['f1_score']:.4f}" if not np.isnan(metrics['f1_score']) else "nan (cannot calculate)"
    acc_str = f"{metrics['accuracy']:.4f}" if not np.isnan(metrics['accuracy']) else "nan (cannot calculate)"
    prec_str = f"{metrics['precision']:.4f}" if not np.isnan(metrics['precision']) else "nan (cannot calculate)"
    rec_str = f"{metrics['recall']:.4f}" if not np.isnan(metrics['recall']) else "nan (cannot calculate)"
    auc_str = f"{metrics['auc_roc']:.4f}" if not np.isnan(metrics['auc_roc']) else "nan (cannot calculate)"
    
    report_lines.append(f"  F1 Score: {f1_str}")
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

