"""
模型训练和评估工具函数
"""

import numpy as np
import pandas as pd
from pathlib import Path
import config


def create_sequences_by_group(data, y_labels, group_col='group', 
                              window_col='window_idx', 
                              selected_features=None,
                              sequence_length=10):
    """
    为每个组独立创建时间序列
    
    Parameters:
    -----------
    data : pd.DataFrame
        数据框（包含group和window_idx列）
    y_labels : pd.Series or np.ndarray
        标签序列（与data的索引对应）
    group_col : str
        组ID列名
    window_col : str
        窗口ID列名
    selected_features : list, optional
        选择的特征列表，如果为None则使用所有数值特征
    sequence_length : int
        序列长度（历史窗口数）
    
    Returns:
    --------
    X_sequences : np.ndarray
        形状为 (n_sequences, sequence_length, n_features)
    Y_sequences : np.ndarray
        形状为 (n_sequences,)
    group_ids : np.ndarray
        每个序列对应的组ID
    """
    if selected_features is None:
        # 排除非特征列
        exclude_cols = [group_col, window_col, 'window_start', 'pair', 'has_data']
        feature_cols = [col for col in data.columns 
                       if col not in exclude_cols and 
                       pd.api.types.is_numeric_dtype(data[col])]
    else:
        feature_cols = selected_features
    
    sequences = []
    labels = []
    group_ids = []
    
    # 重置data的索引，确保与y_labels对齐（y_labels通常是按data的行顺序排列的）
    data_reset = data.reset_index(drop=True)
    
    # 确保y_labels是数组格式，且长度与data匹配
    if isinstance(y_labels, pd.Series):
        y_labels_array = y_labels.values
    else:
        y_labels_array = np.array(y_labels)
    
    # 检查长度是否匹配
    if len(y_labels_array) != len(data_reset):
        raise ValueError(
            f"y_labels长度({len(y_labels_array)})与data行数({len(data_reset)})不匹配。"
            f"请确保y_labels与data的行一一对应（按顺序）。"
        )
    
    # 按组处理
    for group_id in sorted(data_reset[group_col].unique()):
        group_mask = data_reset[group_col] == group_id
        group_data = data_reset[group_mask].copy()
        group_data = group_data.sort_values(window_col)
        
        # 提取特征
        X_group = group_data[feature_cols].values
        
        # 提取标签（使用重置后的位置索引）
        # group_data.index是重置后的连续索引，可以直接用于索引y_labels_array
        group_indices = group_data.index.values  # 这是重置后的连续索引（0, 1, 2, ...）
        y_group = y_labels_array[group_indices]
        
        # 创建序列
        for i in range(sequence_length, len(X_group)):
            seq = X_group[i-sequence_length:i]
            label = y_group[i]
            
            sequences.append(seq)
            labels.append(label)
            group_ids.append(group_id)
    
    X_sequences = np.array(sequences)
    Y_sequences = np.array(labels)
    group_ids = np.array(group_ids)
    
    return X_sequences, Y_sequences, group_ids


def evaluate_model(y_true, y_pred, y_proba=None, binary=True):
    """
    评估模型性能
    
    Parameters:
    -----------
    y_true : np.ndarray
        真实标签
    y_pred : np.ndarray
        预测标签
    y_proba : np.ndarray, optional
        预测概率（用于计算AUC）
    binary : bool
        是否为二分类
    
    Returns:
    --------
    metrics : dict
        评估指标字典
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix
    )
    
    metrics = {}
    
    # 基本指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary' if binary else 'weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='binary' if binary else 'weighted', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='binary' if binary else 'weighted', zero_division=0)
    
    # AUC指标（需要概率）
    if y_proba is not None:
        if binary:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_proba)
        else:
            # 多分类需要one-hot编码
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
            metrics['auc_roc'] = roc_auc_score(y_true_bin, y_proba, average='weighted', multi_class='ovr')
    
    # 混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def save_model_report(metrics, model_name, save_path=None):
    """
    保存模型评估报告
    
    Parameters:
    -----------
    metrics : dict
        评估指标字典
    model_name : str
        模型名称
    save_path : Path, optional
        保存路径，默认使用config.REPORTS_DIR
    """
    if save_path is None:
        save_path = config.REPORTS_DIR / f"{model_name}_performance.txt"
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"=== {model_name} 性能报告 ===\n\n")
        f.write(f"准确率 (Accuracy): {metrics.get('accuracy', 'N/A'):.4f}\n")
        f.write(f"精确率 (Precision): {metrics.get('precision', 'N/A'):.4f}\n")
        f.write(f"召回率 (Recall): {metrics.get('recall', 'N/A'):.4f}\n")
        f.write(f"F1分数 (F1 Score): {metrics.get('f1_score', 'N/A'):.4f}\n")
        if 'auc_roc' in metrics:
            f.write(f"AUC-ROC: {metrics['auc_roc']:.4f}\n")
        if 'average_precision' in metrics:
            f.write(f"Average Precision: {metrics['average_precision']:.4f}\n")
        
        f.write(f"\n混淆矩阵:\n")
        cm = metrics.get('confusion_matrix', None)
        if cm is not None:
            f.write(f"{cm}\n")
    
    print(f"已保存报告: {save_path}")

