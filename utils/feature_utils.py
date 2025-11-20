"""
特征工程工具函数
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def aggregate_pairwise_features(pairwise_df, group_col='group', window_col='window_idx'):
    """
    将成对特征聚合成窗口级别
    
    对每个窗口内的成对特征计算：均值、标准差、最大值、最小值
    
    Parameters:
    -----------
    pairwise_df : pd.DataFrame
        成对特征数据框
    group_col : str
        组ID列名
    window_col : str
        窗口ID列名
    
    Returns:
    --------
    aggregated_df : pd.DataFrame
        聚合后的窗口级特征
    """
    # 排除非数值列
    exclude_cols = [group_col, window_col, 'pair', 'window_start']
    feature_cols = [col for col in pairwise_df.columns if col not in exclude_cols]
    
    # 按组和窗口分组
    grouped = pairwise_df.groupby([group_col, window_col])
    
    # 聚合函数
    agg_funcs = ['mean', 'std', 'max', 'min']
    
    # 创建聚合后的数据框
    aggregated_list = []
    
    for (group, window), group_df in grouped:
        row = {group_col: group, window_col: window}
        
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(group_df[col]):
                values = group_df[col].dropna()
                if len(values) > 0:
                    row[f"{col}_mean"] = values.mean()
                    row[f"{col}_std"] = values.std() if len(values) > 1 else 0
                    row[f"{col}_max"] = values.max()
                    row[f"{col}_min"] = values.min()
                else:
                    row[f"{col}_mean"] = np.nan
                    row[f"{col}_std"] = np.nan
                    row[f"{col}_max"] = np.nan
                    row[f"{col}_min"] = np.nan
        
        aggregated_list.append(row)
    
    aggregated_df = pd.DataFrame(aggregated_list)
    return aggregated_df


def normalize_features(train_df, val_df, test_df, train_val_df=None, 
                      exclude_cols=None, scaler=None):
    """
    使用训练集的统计量标准化所有数据
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        训练集数据
    val_df : pd.DataFrame
        验证集数据
    test_df : pd.DataFrame
        测试集数据
    train_val_df : pd.DataFrame, optional
        训练验证集数据
    exclude_cols : list, optional
        不标准化的列（如group, window等）
    scaler : StandardScaler, optional
        已训练的标准化器，如果为None则从训练集训练
    
    Returns:
    --------
    train_scaled : pd.DataFrame
        标准化后的训练集
    val_scaled : pd.DataFrame
        标准化后的验证集
    test_scaled : pd.DataFrame
        标准化后的测试集
    train_val_scaled : pd.DataFrame, optional
        标准化后的训练验证集
    scaler : StandardScaler
        训练好的标准化器
    """
    if exclude_cols is None:
        exclude_cols = ['group', 'window', 'window_idx']
    
    # 确定要标准化的列
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # 提取特征
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    if train_val_df is not None:
        X_train_val = train_val_df[feature_cols].copy()
    
    # 训练标准化器（如果未提供）
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=feature_cols,
            index=X_train.index
        )
    else:
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train),
            columns=feature_cols,
            index=X_train.index
        )
    
    # 标准化其他数据集
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=feature_cols,
        index=X_val.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )
    
    # 重建完整数据框
    train_scaled = train_df.copy()
    train_scaled[feature_cols] = X_train_scaled
    
    val_scaled = val_df.copy()
    val_scaled[feature_cols] = X_val_scaled
    
    test_scaled = test_df.copy()
    test_scaled[feature_cols] = X_test_scaled
    
    if train_val_df is not None:
        X_train_val_scaled = pd.DataFrame(
            scaler.transform(X_train_val),
            columns=feature_cols,
            index=X_train_val.index
        )
        train_val_scaled = train_val_df.copy()
        train_val_scaled[feature_cols] = X_train_val_scaled
        return train_scaled, train_val_scaled, val_scaled, test_scaled, scaler
    
    return train_scaled, val_scaled, test_scaled, scaler

