"""
04 - 模型评估与报告

本脚本完成以下任务：
1. 加载最佳模型和测试数据
2. 在测试集上评估模型性能
3. 生成混淆矩阵和分类报告
4. 按组分析性能
5. 生成最终评估报告

**专业分析**：
- 使用测试集进行最终评估（从未参与训练）
- 重点关注F1分数（考虑数据不平衡）
- 提供详细的性能分析和可视化
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

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("错误: TensorFlow未安装")
    exit(1)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

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
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
VISUALIZATIONS_DIR.mkdir(exist_ok=True)

# 多分类配置（需要与01_hmm_modeling.py和03_time_series_training.py一致）
N_CLASSES = 4  # 3或4分类，必须与其他脚本一致

# 随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# 工具函数部分
# ============================================================================

def load_intermediate(name, directory=None):
    """从intermediate目录加载中间结果"""
    if directory is None:
        directory = INTERMEDIATE_DIR
    filepath = directory / f"{name}.pkl"
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"已加载: {filepath}")
    return data


def save_intermediate(name, data, directory=None):
    """保存中间结果到intermediate目录"""
    if directory is None:
        directory = INTERMEDIATE_DIR
    filepath = directory / f"{name}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"已保存: {filepath}")


def create_sequences_by_group(data_df, feature_cols, label_array, sequence_length=10):
    """按组创建时间序列序列"""
    X_list = []
    y_list = []
    group_list = []
    
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
            y_list.append(group_labels[i+sequence_length])     # 窗口N+1的标签（预测未来）
            group_list.append(group)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y, np.array(group_list)


def evaluate_model(y_true, y_pred, y_proba=None, n_classes=2):
    """评估模型性能（支持多分类）"""
    # 检测实际分类数
    unique_labels = len(np.unique(y_true))
    if unique_labels < 2:
        return {
            'accuracy': 1.0 if len(y_true) == 0 else 0.0,
            'precision': np.nan,
            'recall': np.nan,
            'f1_score': np.nan,
            'auc_roc': np.nan
        }
    
    # 使用weighted平均支持多分类
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # AUC-ROC计算（支持多分类）
    if y_proba is not None and unique_labels > 1:
        try:
            if n_classes > 2 and y_proba.ndim > 1 and y_proba.shape[1] > 1:
                # 多分类：使用ovr策略
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


# ============================================================================
# 主程序部分
# ============================================================================

print("\n" + "="*80)
print("04 - 模型评估与报告")
print("="*80)

# 1. 加载数据
print("\n" + "-"*80)
print("1. 加载数据和模型")
print("-"*80)

test_data = load_intermediate('test_data')
y_test = load_intermediate('y_test')
top_m_features = load_intermediate('top_m_features')
best_model_name = load_intermediate('best_model_name')
sequence_length = 3  # 与训练时一致（03_time_series_training.py中的SEQUENCE_CONFIG）

print(f"\n最佳模型: {best_model_name}")
print(f"测试集大小: {len(test_data)} 窗口")
print(f"特征数: {len(top_m_features)}")

# 加载最佳模型
model_path = MODELS_DIR / f'{best_model_name.lower()}_final.h5'
if not model_path.exists():
    model_path = MODELS_DIR / f'{best_model_name.lower()}_best.h5'

best_model = keras.models.load_model(str(model_path))
print(f"✓ 模型已加载: {model_path}")

# 2. 创建测试序列
print("\n" + "-"*80)
print("2. 创建测试序列")
print("-"*80)

X_test_seq, y_test_seq, test_groups = create_sequences_by_group(
    test_data, top_m_features, y_test,
    sequence_length=sequence_length
)

print(f"测试序列形状: {X_test_seq.shape}")
print(f"标签形状: {y_test_seq.shape}")

# 3. 模型预测
print("\n" + "-"*80)
print("3. 模型预测")
print("-"*80)

y_test_pred_proba = best_model.predict(X_test_seq)
if N_CLASSES > 2:
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)  # 多分类：取概率最大的类别
    y_test_proba = y_test_pred_proba  # 完整概率矩阵用于多分类AUC
else:
    y_test_pred = (y_test_pred_proba > 0.5).astype(int).flatten()
    y_test_proba = y_test_pred_proba.flatten()

print(f"预测完成")
if N_CLASSES == 4:
    print(f"预测标签分布: 0={np.sum(y_test_pred == 0)}, 1={np.sum(y_test_pred == 1)}, 2={np.sum(y_test_pred == 2)}, 3={np.sum(y_test_pred == 3)}")
    print(f"真实标签分布: 0={np.sum(y_test_seq == 0)}, 1={np.sum(y_test_seq == 1)}, 2={np.sum(y_test_seq == 2)}, 3={np.sum(y_test_seq == 3)}")
elif N_CLASSES == 3:
    print(f"预测标签分布: 0={np.sum(y_test_pred == 0)}, 1={np.sum(y_test_pred == 1)}, 2={np.sum(y_test_pred == 2)}")
    print(f"真实标签分布: 0={np.sum(y_test_seq == 0)}, 1={np.sum(y_test_seq == 1)}, 2={np.sum(y_test_seq == 2)}")
else:
    print(f"预测标签分布: 0={np.sum(y_test_pred == 0)}, 1={np.sum(y_test_pred == 1)}")
    print(f"真实标签分布: 0={np.sum(y_test_seq == 0)}, 1={np.sum(y_test_seq == 1)}")

# 4. 整体性能评估
print("\n" + "-"*80)
print("4. 整体性能评估")
print("-"*80)

test_metrics = evaluate_model(y_test_seq, y_test_pred, y_test_proba, n_classes=N_CLASSES)

print(f"\n测试集性能:")
print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall: {test_metrics['recall']:.4f}")
print(f"  F1 Score: {test_metrics['f1_score']:.4f}")
print(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test_seq, y_test_pred)
print(f"\n混淆矩阵:")
print(cm)

# 分类报告
print(f"\n分类报告:")
# 动态检测实际存在的类别
unique_labels = sorted(np.unique(np.concatenate([y_test_seq, y_test_pred])))
target_names = [f'状态{i}' for i in unique_labels]
# 使用labels参数确保只报告实际存在的类别
print(classification_report(y_test_seq, y_test_pred, target_names=target_names, labels=unique_labels, zero_division=0))

# 5. 按组分析
print("\n" + "-"*80)
print("5. 按组分析性能")
print("-"*80)

group_metrics = {}
for group in np.unique(test_groups):
    group_mask = test_groups == group
    group_y_true = y_test_seq[group_mask]
    group_y_pred = y_test_pred[group_mask]
    group_y_proba = y_test_proba[group_mask]
    
    if len(group_y_true) > 0:
        group_metrics[group] = evaluate_model(group_y_true, group_y_pred, group_y_proba, n_classes=N_CLASSES)
        print(f"\n组 {group}:")
        print(f"  样本数: {len(group_y_true)}")
        print(f"  F1 Score: {group_metrics[group]['f1_score']:.4f}")
        print(f"  Accuracy: {group_metrics[group]['accuracy']:.4f}")

# 6. 保存结果
print("\n" + "-"*80)
print("6. 保存结果")
print("-"*80)

save_intermediate('test_predictions', y_test_pred)
save_intermediate('test_metrics', test_metrics)
save_intermediate('group_metrics', group_metrics)

# 7. 可视化
print("\n" + "-"*80)
print("7. 生成可视化")
print("-"*80)

# 7.1 混淆矩阵
plt.figure(figsize=(8, 6))
if N_CLASSES == 4:
    labels = ['状态0', '状态1', '状态2', '状态3']
elif N_CLASSES == 3:
    labels = ['类别0', '类别1', '类别2']
else:
    labels = ['无需干预', '需要干预']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels,
            yticklabels=labels)
plt.title('测试集混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.tight_layout()
plt.savefig(VISUALIZATIONS_DIR / 'test_confusion_matrix.png', dpi=300)
print(f"✓ 混淆矩阵已保存: {VISUALIZATIONS_DIR / 'test_confusion_matrix.png'}")

# 7.2 组性能对比
if len(group_metrics) > 0:
    groups = list(group_metrics.keys())
    f1_scores = [group_metrics[g]['f1_score'] for g in groups]
    accuracies = [group_metrics[g]['accuracy'] for g in groups]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(groups, f1_scores)
    ax1.set_xlabel('组')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('各组F1 Score')
    ax1.set_ylim([0, 1])
    
    ax2.bar(groups, accuracies)
    ax2.set_xlabel('组')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('各组Accuracy')
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / 'group_performance.png', dpi=300)
    print(f"✓ 组性能对比已保存: {VISUALIZATIONS_DIR / 'group_performance.png'}")

# 8. 生成最终报告
print("\n" + "-"*80)
print("8. 生成最终评估报告")
print("-"*80)

report_lines = []
report_lines.append("=" * 60)
report_lines.append("最终评估报告")
report_lines.append("=" * 60)
report_lines.append(f"\n模型信息:")
report_lines.append(f"  最佳模型: {best_model_name}")
report_lines.append(f"  特征数: {len(top_m_features)}")
report_lines.append(f"  序列长度: {sequence_length}")
report_lines.append(f"\n测试集整体性能:")
report_lines.append(f"  Accuracy: {test_metrics['accuracy']:.4f}")
report_lines.append(f"  Precision: {test_metrics['precision']:.4f}")
report_lines.append(f"  Recall: {test_metrics['recall']:.4f}")
report_lines.append(f"  F1 Score: {test_metrics['f1_score']:.4f}")
report_lines.append(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
report_lines.append(f"\n混淆矩阵:")
report_lines.append(f"  [[{cm[0,0]}, {cm[0,1]}],")
report_lines.append(f"   [{cm[1,0]}, {cm[1,1]}]]")
report_lines.append(f"\n各组性能:")
for group, metrics in sorted(group_metrics.items()):
    report_lines.append(f"\n组 {group}:")
    report_lines.append(f"  F1 Score: {metrics['f1_score']:.4f}")
    report_lines.append(f"  Accuracy: {metrics['accuracy']:.4f}")
    report_lines.append(f"  Precision: {metrics['precision']:.4f}")
    report_lines.append(f"  Recall: {metrics['recall']:.4f}")

report_text = "\n".join(report_lines)
print(report_text)

with open(REPORTS_DIR / "final_evaluation_report.txt", 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\n✓ 最终报告已保存到 {REPORTS_DIR / 'final_evaluation_report.txt'}")

print("\n" + "="*80)
print("模型评估完成！")
print("="*80)
print("\n所有结果已保存到 outputs/ 目录")

