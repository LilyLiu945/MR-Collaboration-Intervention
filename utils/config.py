"""
配置文件 - 所有路径和参数设置
"""

import os
from pathlib import Path

# ==================== 路径配置 ====================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据路径
DATA_DIR = PROJECT_ROOT
PAIRWISE_FEATURES_PATH = DATA_DIR / "pairwise_outputs" / "pairwise_features.csv"
WINDOWED_METRICS_PATH = DATA_DIR / "windowed_output" / "data" / "windowed_metrics.csv"
TASK_METRICS_PATH = DATA_DIR / "task_metrics_output" / "task_metrics_summary.csv"

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

# ==================== 数据划分配置 ====================

# 组划分
TRAIN_GROUPS = list(range(1, 9))  # 组1-8
VAL_GROUPS = [9, 10]  # 组9-10
TEST_GROUPS = [11, 12]  # 组11-12

# 训练组内时间划分比例
TRAIN_TIME_SPLIT = 0.7  # 前70%用于训练，后30%用于训练验证

# ==================== 特征选择配置 ====================

# 阶段1：无监督特征选择
UNSUPERVISED_CONFIG = {
    "variance_threshold": 0.01,  # 方差阈值（移除方差小于此值的特征）
    "correlation_threshold": 0.95,  # 相关性阈值（移除相关性大于此值的特征对）
    "pca_n_components": None,  # None表示保留所有主成分，或指定数量
    "pca_variance_ratio": 0.95,  # 保留解释95%方差的主成分
    "n_clusters": 5,  # K-means聚类数
    "top_k": 50,  # 选择Top-K特征（可根据验证集性能调整）
}

# 阶段2.5：有监督特征选择
SUPERVISED_CONFIG = {
    "top_m": 30,  # 选择Top-M特征（M ≤ K，可根据验证集性能调整）
    "rf_n_estimators": 100,  # Random Forest树的数量
    "rf_max_depth": 10,  # Random Forest最大深度
    "lasso_alpha": 0.01,  # LASSO正则化系数
    "rfe_n_features": None,  # RFE选择的特征数（None表示自动选择）
}

# ==================== HMM配置 ====================

HMM_CONFIG = {
    "coarse_n_states": 4,  # 粗粒度状态数（3-4个）
    "fine_n_states": 3,  # 细粒度状态数（2-3个）
    "n_iter": 100,  # Baum-Welch算法迭代次数
    "covariance_type": "full",  # 协方差类型：'full', 'tied', 'diag', 'spherical'
    "random_state": 42,
}

# ==================== 时间序列模型配置 ====================

# 序列构建
SEQUENCE_CONFIG = {
    "sequence_length": 10,  # 历史窗口数（10个窗口 = 320秒）
}

# LSTM配置
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

# GRU配置
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

# Transformer配置
TRANSFORMER_CONFIG = {
    "d_model": 64,  # 模型维度
    "num_heads": 4,  # 注意力头数
    "num_layers": 2,  # Transformer层数
    "dff": 128,  # 前馈网络维度
    "dropout_rate": 0.1,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
}

# ==================== 评估配置 ====================

EVALUATION_CONFIG = {
    "primary_metric": "f1_score",  # 主要评估指标（考虑数据不平衡）
    "secondary_metric": "auc_roc",  # 次要评估指标
}

# ==================== 其他配置 ====================

# 随机种子
RANDOM_STATE = 42

# 显示选项
DISPLAY_MAX_COLUMNS = None  # None表示显示所有列
DISPLAY_MAX_ROWS = 100

# 可视化配置
PLOT_STYLE = "seaborn-v0_8"
PLOT_FIGSIZE = (12, 6)
PLOT_DPI = 100

