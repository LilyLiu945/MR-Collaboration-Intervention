"""
可视化工具函数
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import config


def plot_feature_importance(importance_scores, feature_names, top_n=20, 
                           title="特征重要性", save_path=None):
    """
    绘制特征重要性图
    
    Parameters:
    -----------
    importance_scores : array-like
        特征重要性分数
    feature_names : array-like
        特征名称
    top_n : int
        显示前N个特征
    title : str
        图表标题
    save_path : Path, optional
        保存路径
    """
    # 创建数据框
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    })
    
    # 排序并选择Top-N
    df = df.sort_values('importance', ascending=False).head(top_n)
    
    # 绘图
    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    plt.barh(range(len(df)), df['importance'], align='center')
    plt.yticks(range(len(df)), df['feature'])
    plt.xlabel('重要性分数')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"已保存: {save_path}")
    
    plt.show()


def plot_correlation_matrix(df, method='pearson', figsize=(12, 10), 
                            save_path=None):
    """
    绘制特征相关性矩阵热力图
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    method : str
        相关性计算方法（'pearson', 'spearman', 'kendall'）
    figsize : tuple
        图表大小
    save_path : Path, optional
        保存路径
    """
    # 计算相关性矩阵
    corr = df.corr(method=method)
    
    # 绘图
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'特征相关性矩阵 ({method})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"已保存: {save_path}")
    
    plt.show()


def plot_data_distribution(df, column, group_col='group', save_path=None):
    """
    绘制数据分布图（按组）
    
    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    column : str
        要绘制的列名
    group_col : str
        组ID列名
    save_path : Path, optional
        保存路径
    """
    plt.figure(figsize=(12, 6))
    
    if group_col in df.columns:
        groups = df[group_col].unique()
        for group in sorted(groups):
            group_data = df[df[group_col] == group][column]
            plt.hist(group_data, alpha=0.5, label=f'组{group}', bins=30)
        plt.legend()
    else:
        plt.hist(df[column], bins=30)
    
    plt.xlabel(column)
    plt.ylabel('频数')
    plt.title(f'{column} 分布')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"已保存: {save_path}")
    
    plt.show()

