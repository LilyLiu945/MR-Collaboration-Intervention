"""
HMM建模工具函数
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.cluster import KMeans
import config


def train_coarse_hmm(X_train, n_states=None, n_iter=None, covariance_type=None, random_state=None):
    """
    训练粗粒度HMM模型
    
    Parameters:
    -----------
    X_train : np.ndarray or pd.DataFrame
        训练数据（每个组的时间序列）
    n_states : int, optional
        状态数，默认使用config.HMM_CONFIG['coarse_n_states']
    n_iter : int, optional
        迭代次数，默认使用config.HMM_CONFIG['n_iter']
    covariance_type : str, optional
        协方差类型，默认使用config.HMM_CONFIG['covariance_type']
    random_state : int, optional
        随机种子，默认使用config.HMM_CONFIG['random_state']
    
    Returns:
    --------
    model : GaussianHMM
        训练好的HMM模型
    """
    if n_states is None:
        n_states = config.HMM_CONFIG['coarse_n_states']
    if n_iter is None:
        n_iter = config.HMM_CONFIG['n_iter']
    if covariance_type is None:
        covariance_type = config.HMM_CONFIG['covariance_type']
    if random_state is None:
        random_state = config.HMM_CONFIG['random_state']
    
    # 转换为numpy数组
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    
    # 使用K-means初始化
    kmeans = KMeans(n_clusters=n_states, random_state=random_state, n_init=10)
    kmeans.fit(X_train)
    
    # 初始化HMM
    # 注意：init_params='stc'表示初始化状态转移矩阵、协方差和初始概率
    # 不包含'm'（均值），这样我们可以手动设置means_而不会被覆盖
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        init_params='stc',  # 初始化状态转移矩阵、协方差和初始概率（不包含均值）
    )
    
    # 设置初始均值（从K-means聚类中心）
    model.means_ = kmeans.cluster_centers_
    
    # 训练HMM
    model.fit(X_train)
    
    return model


def train_fine_hmm(X_train, low_comm_states, n_states=None, n_iter=None, 
                   covariance_type=None, random_state=None):
    """
    训练细粒度HMM模型（针对低通信状态）
    
    Parameters:
    -----------
    X_train : np.ndarray or pd.DataFrame
        训练数据
    low_comm_states : array-like
        低通信状态的索引
    n_states : int, optional
        细粒度状态数，默认使用config.HMM_CONFIG['fine_n_states']
    n_iter : int, optional
        迭代次数
    covariance_type : str, optional
        协方差类型
    random_state : int, optional
        随机种子
    
    Returns:
    --------
    model : GaussianHMM
        训练好的细粒度HMM模型
    """
    if n_states is None:
        n_states = config.HMM_CONFIG['fine_n_states']
    if n_iter is None:
        n_iter = config.HMM_CONFIG['n_iter']
    if covariance_type is None:
        covariance_type = config.HMM_CONFIG['covariance_type']
    if random_state is None:
        random_state = config.HMM_CONFIG['random_state']
    
    # 提取低通信状态的数据
    if isinstance(X_train, pd.DataFrame):
        X_low = X_train.iloc[low_comm_states].values
    else:
        X_low = X_train[low_comm_states]
    
    if len(X_low) < n_states:
        print(f"警告：低通信状态数据量({len(X_low)})少于状态数({n_states})，跳过细粒度HMM训练")
        return None
    
    # 使用K-means初始化
    kmeans = KMeans(n_clusters=n_states, random_state=random_state, n_init=10)
    kmeans.fit(X_low)
    
    # 初始化HMM
    # 注意：init_params='stc'表示初始化状态转移矩阵、协方差和初始概率
    # 不包含'm'（均值），这样我们可以手动设置means_而不会被覆盖
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        init_params='stc',  # 初始化状态转移矩阵、协方差和初始概率（不包含均值）
    )
    
    # 设置初始均值（从K-means聚类中心）
    model.means_ = kmeans.cluster_centers_
    
    # 训练HMM
    model.fit(X_low)
    
    return model


def predict_hmm_states(model, X, group_col='group', window_col='window_idx'):
    """
    使用HMM模型预测状态
    
    Parameters:
    -----------
    model : GaussianHMM
        HMM模型
    X : pd.DataFrame or np.ndarray
        要预测的数据
    group_col : str
        组ID列名（如果X是DataFrame）
    window_col : str
        窗口ID列名（如果X是DataFrame）
    
    Returns:
    --------
    states : np.ndarray
        预测的状态序列
    """
    if isinstance(X, pd.DataFrame):
        # 按组和时间排序
        X_sorted = X.sort_values([group_col, window_col])
        X_values = X_sorted.drop(columns=[group_col, window_col], errors='ignore').values
    else:
        X_values = X
    
    # 预测状态
    states = model.predict(X_values)
    
    return states


def map_states_to_labels(states, coarse_model, fine_model=None, 
                        low_comm_state_idx=None, binary=True):
    """
    将HMM状态映射到Y标签
    
    Parameters:
    -----------
    states : np.ndarray
        粗粒度状态序列
    coarse_model : GaussianHMM
        粗粒度HMM模型
    fine_model : GaussianHMM, optional
        细粒度HMM模型
    low_comm_state_idx : int, optional
        低通信状态的索引（在粗粒度状态中）
    binary : bool
        是否使用二分类（True）或多分类（False）
    
    Returns:
    --------
    labels : np.ndarray
        Y标签
    """
    if binary:
        # 二分类：高/中等通信=0，低通信-需要干预=1
        # 假设最后一个状态是低通信状态
        if low_comm_state_idx is None:
            low_comm_state_idx = coarse_model.n_components - 1
        
        labels = (states == low_comm_state_idx).astype(int)
        
        # 如果有细粒度模型，进一步细分低通信状态
        if fine_model is not None:
            low_comm_mask = states == low_comm_state_idx
            if low_comm_mask.sum() > 0:
                # 对低通信状态使用细粒度模型
                # 这里简化处理：假设细粒度模型的第一个状态是需要干预的
                # 实际应用中需要根据细粒度模型的预测结果来设置
                pass  # TODO: 实现细粒度状态映射
    else:
        # 多分类：直接使用状态作为标签
        labels = states
    
    return labels

