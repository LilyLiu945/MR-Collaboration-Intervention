"""
数据加载和保存工具函数
"""

import pickle
import pandas as pd
from pathlib import Path
import config


def save_intermediate(name, data, directory=None):
    """
    保存中间结果到intermediate目录
    
    Parameters:
    -----------
    name : str
        文件名（不含扩展名）
    data : any
        要保存的数据
    directory : Path, optional
        保存目录，默认使用config.INTERMEDIATE_DIR
    """
    if directory is None:
        directory = config.INTERMEDIATE_DIR
    
    filepath = directory / f"{name}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"已保存: {filepath}")


def load_intermediate(name, directory=None):
    """
    从intermediate目录加载中间结果
    
    Parameters:
    -----------
    name : str
        文件名（不含扩展名）
    directory : Path, optional
        加载目录，默认使用config.INTERMEDIATE_DIR
    
    Returns:
    --------
    data : any
        加载的数据
    """
    if directory is None:
        directory = config.INTERMEDIATE_DIR
    
    filepath = directory / f"{name}.pkl"
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"已加载: {filepath}")
    return data


def load_all_data():
    """
    加载所有原始数据文件
    
    Returns:
    --------
    pairwise_df : pd.DataFrame
        成对特征数据
    windowed_df : pd.DataFrame
        窗口级网络指标数据
    task_df : pd.DataFrame
        任务性能指标数据
    """
    print("正在加载数据文件...")
    
    # 加载成对特征
    pairwise_df = pd.read_csv(config.PAIRWISE_FEATURES_PATH)
    print(f"✓ 成对特征数据: {len(pairwise_df)} 行, {len(pairwise_df.columns)} 列")
    
    # 加载窗口级网络指标
    windowed_df = pd.read_csv(config.WINDOWED_METRICS_PATH)
    print(f"✓ 窗口级网络指标: {len(windowed_df)} 行, {len(windowed_df.columns)} 列")
    
    # 加载任务性能指标
    task_df = pd.read_csv(config.TASK_METRICS_PATH)
    print(f"✓ 任务性能指标: {len(task_df)} 行, {len(task_df.columns)} 列")
    
    return pairwise_df, windowed_df, task_df


def check_data_quality(df, name="数据"):
    """
    检查数据质量
    
    Parameters:
    -----------
    df : pd.DataFrame
        要检查的数据框
    name : str
        数据名称
    """
    print(f"\n=== {name}质量检查 ===")
    print(f"形状: {df.shape}")
    print(f"缺失值:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("无缺失值")
    print(f"重复行: {df.duplicated().sum()}")
    print(f"数据类型:\n{df.dtypes.value_counts()}")

