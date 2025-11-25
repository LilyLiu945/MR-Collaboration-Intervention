"""
00 - 数据准备与划分

本脚本完成以下任务：
1. 加载窗口级数据（windowed_output）- 用于训练/预测
2. 加载静态数据（task_metrics, session_output）- 用于后处理分析
3. 数据探索和质量检查
4. 按组划分数据（8组训练，4组测试）
5. 特征工程（展开模态、聚合节点指标、标准化）
6. 保存处理后的数据

**注意**：
- 只使用windowed_output作为动态特征（用于训练/预测）
- 静态特征（task_metrics, session_output）单独保存用于后处理分析
- 验证集已合并到测试集（组9-12全部作为测试集）
"""

# ============================================================================
# 配置部分（内嵌，无需外部config.py）
# ============================================================================

import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据路径
DATA_DIR = PROJECT_ROOT
WINDOWED_METRICS_PATH = DATA_DIR / "windowed_output" / "data" / "windowed_metrics.csv"
WINDOWED_NODES_PATH = DATA_DIR / "windowed_output" / "data" / "windowed_nodes.csv"
TASK_METRICS_PATH = DATA_DIR / "task_metrics_output" / "task_metrics_summary.csv"

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
INTERMEDIATE_DIR.mkdir(exist_ok=True)
REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# 数据划分配置
# 根据窗口数统计：窗口数最多的4个组作为测试集
# 组8(68), 组6(51), 组7(40), 组3(38) - 窗口数最多的4个组
TEST_GROUPS = [3, 6, 7, 8]  # 测试集（窗口数最多的4个组）
TRAIN_GROUPS = [1, 2, 4, 5, 9, 10, 11, 12]  # 训练集（其余8个组）
# 注意：不再进行训练组内的时间划分，避免数据泄露
# 注意：验证集已合并到测试集，简化数据划分

# 随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
import random
random.seed(RANDOM_STATE)

# ============================================================================
# 工具函数部分（内嵌，无需外部utils）
# ============================================================================

def save_intermediate(name, data, directory=None):
    """保存中间结果到intermediate目录"""
    if directory is None:
        directory = INTERMEDIATE_DIR
    filepath = directory / f"{name}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"已保存: {filepath}")


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


def load_windowed_data():
    """加载窗口级网络指标数据（用于训练/预测）"""
    print("正在加载窗口级网络指标数据...")
    windowed_df = pd.read_csv(WINDOWED_METRICS_PATH)
    print(f"✓ 窗口级网络指标: {len(windowed_df)} 行, {len(windowed_df.columns)} 列")
    return windowed_df


def load_windowed_nodes():
    """加载窗口级节点数据（用于聚合特征）"""
    print("正在加载窗口级节点数据...")
    nodes_df = pd.read_csv(WINDOWED_NODES_PATH)
    print(f"✓ 窗口级节点数据: {len(nodes_df)} 行, {len(nodes_df.columns)} 列")
    return nodes_df


def check_data_quality(df, name="数据"):
    """检查数据质量"""
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


def normalize_features(train_df, val_df, test_df, train_val_df=None, 
                      exclude_cols=None, scaler=None):
    """使用训练集的统计量标准化所有数据"""
    if exclude_cols is None:
        exclude_cols = ['group', 'window', 'window_idx']
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    # 检查val_df是否为空
    val_is_empty = len(val_df) == 0
    
    if not val_is_empty:
        X_val = val_df[feature_cols].copy()
    
    if train_val_df is not None and len(train_val_df) > 0:
        X_train_val = train_val_df[feature_cols].copy()
    
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
    
    # 处理验证集（可能为空）
    if val_is_empty:
        X_val_scaled = pd.DataFrame(columns=feature_cols)
        val_scaled = val_df.copy()  # 保持原始结构
    else:
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=feature_cols,
            index=X_val.index
        )
        val_scaled = val_df.copy()
        val_scaled[feature_cols] = X_val_scaled
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )
    
    train_scaled = train_df.copy()
    train_scaled[feature_cols] = X_train_scaled
    
    test_scaled = test_df.copy()
    test_scaled[feature_cols] = X_test_scaled
    
    if train_val_df is not None and len(train_val_df) > 0:
        X_train_val_scaled = pd.DataFrame(
            scaler.transform(X_train_val),
            columns=feature_cols,
            index=X_train_val.index
        )
        train_val_scaled = train_val_df.copy()
        train_val_scaled[feature_cols] = X_train_val_scaled
        return train_scaled, train_val_scaled, val_scaled, test_scaled, scaler
    
    return train_scaled, val_scaled, test_scaled, scaler

# ============================================================================
# 主程序部分
# ============================================================================

print("\n" + "="*80)
print("00 - 数据准备与划分")
print("="*80)

# 1. 数据加载
print("\n" + "-"*80)
print("1. 数据加载")
print("-"*80)
print("\n**策略**：")
print("- 加载windowed_output（窗口级动态特征）- 用于训练/预测")
print("  - windowed_metrics: 网络级别指标（密度、聚类等）")
print("  - windowed_nodes: 节点级别指标（中心性，将聚合为组级别）")
print("- 加载task_metrics和session_output（静态特征）- 用于后处理分析，不合并到训练数据")

windowed_df = load_windowed_data()
windowed_nodes_df = load_windowed_nodes()

task_df = pd.read_csv(TASK_METRICS_PATH)
print(f"✓ 任务性能指标: {len(task_df)} 行, {len(task_df.columns)} 列（仅用于后处理）")

try:
    session_edges_df = pd.read_csv(PROJECT_ROOT / "session_output" / "data" / "session_edges.csv")
    session_nodes_df = pd.read_csv(PROJECT_ROOT / "session_output" / "data" / "session_nodes.csv")
    session_metrics_df = pd.read_csv(PROJECT_ROOT / "session_output" / "data" / "session_metrics.csv")
    print(f"✓ 会话级数据已加载（仅用于后处理）")
except FileNotFoundError:
    print("⚠ 会话级数据未找到，跳过（可选）")
    session_edges_df = None
    session_nodes_df = None
    session_metrics_df = None

# 2. 数据探索
print("\n" + "-"*80)
print("2. 数据探索")
print("-"*80)

print("\n=== 窗口级网络指标数据（用于训练/预测）===")
print(f"形状: {windowed_df.shape}")
print(f"\n前10行:")
print(windowed_df.head(10))
print(f"\n列名: {list(windowed_df.columns)}")
print(f"\n各组数据量:")
print(windowed_df['group'].value_counts().sort_index())
print(f"\n模态类型:")
print(windowed_df['modality'].value_counts())

check_data_quality(windowed_df, "窗口级网络指标")

print("\n=== 静态数据（仅用于后处理分析）===")
print(f"任务性能指标: {task_df.shape}")
print("注意：这些数据不合并到训练数据中，只用于后处理分析")

# 3. 特征工程
print("\n" + "-"*80)
print("3. 特征工程")
print("-"*80)

print("\n3.1 展开窗口级网络指标的模态（宽格式）")
print("正在展开窗口级网络指标的模态...")

metric_cols = ['density', 'avg_clustering', 'eigenvector', 'reciprocity']
windowed_pivot_list = []

for metric in metric_cols:
    if metric in windowed_df.columns:
        pivot = windowed_df.pivot_table(
            index=['group', 'window'],
            columns='modality',
            values=metric,
            aggfunc='first'
        )
        pivot.columns = [f"{metric}_{col}" for col in pivot.columns]
        windowed_pivot_list.append(pivot)

if windowed_pivot_list:
    windowed_wide = pd.concat(windowed_pivot_list, axis=1)
    windowed_wide = windowed_wide.reset_index()
    # 统一窗口列名
    if 'window' in windowed_wide.columns:
        windowed_wide = windowed_wide.rename(columns={'window': 'window_idx'})
    print(f"展开后形状: {windowed_wide.shape}")
    print(f"特征数: {len(windowed_wide.columns) - 2}")
    print(f"\n列名: {list(windowed_wide.columns)}")
    print("\n前5行:")
    print(windowed_wide.head())
else:
    print("警告: 没有找到可展开的指标")
    windowed_wide = windowed_df[['group', 'window']].drop_duplicates()
if 'window' in windowed_wide.columns:
    windowed_wide = windowed_wide.rename(columns={'window': 'window_idx'})

print("\n3.2 聚合窗口级节点指标（均值和标准差）")
print("正在聚合节点级别的中心性指标...")

# 节点指标列（排除标识列）
node_metric_cols = ['betweenness_centrality', 'degree_centrality', 
                    'eigenvector_centrality', 'closeness_centrality', 'degree']

# 对每个窗口×模态的4个参与者计算均值和标准差
node_agg_list = []

for metric in node_metric_cols:
    if metric in windowed_nodes_df.columns:
        # 计算均值和标准差
        agg_df = windowed_nodes_df.groupby(['group', 'window', 'modality'])[metric].agg([
            ('mean', 'mean'),
            ('std', 'std')
        ]).reset_index()
        
        # 展开为宽格式
        pivot_mean = agg_df.pivot_table(
            index=['group', 'window'],
            columns='modality',
            values='mean',
            aggfunc='first'
        )
        pivot_mean.columns = [f"{metric}_mean_{col}" for col in pivot_mean.columns]
        
        pivot_std = agg_df.pivot_table(
            index=['group', 'window'],
            columns='modality',
            values='std',
            aggfunc='first'
        )
        pivot_std.columns = [f"{metric}_std_{col}" for col in pivot_std.columns]
        
        node_agg_list.append(pivot_mean)
        node_agg_list.append(pivot_std)

if node_agg_list:
    nodes_wide = pd.concat(node_agg_list, axis=1)
    nodes_wide = nodes_wide.reset_index()
    print(f"节点指标聚合后形状: {nodes_wide.shape}")
    print(f"节点特征数: {len(nodes_wide.columns) - 2}")
else:
    print("警告: 没有找到可聚合的节点指标")
    nodes_wide = windowed_wide[['group', 'window']].drop_duplicates()

print("\n3.3 合并网络指标和节点指标")
# 确保nodes_wide的窗口列名与windowed_wide一致
if 'window' in nodes_wide.columns:
    nodes_wide = nodes_wide.rename(columns={'window': 'window_idx'})

# 合并windowed_wide和nodes_wide
# 使用left join确保保留所有windowed_wide的数据
final_data = windowed_wide.merge(
    nodes_wide,
    on=['group', 'window_idx'],
    how='left'
)

print(f"合并后形状: {final_data.shape}")
print(f"总特征数: {len(final_data.columns) - 2}")

print("\n3.4 统一窗口列名")
if 'window' in final_data.columns:
    final_data = final_data.rename(columns={'window': 'window_idx'})
    print("✓ 窗口列名已统一为window_idx")

feature_cols = [col for col in final_data.columns if col not in ['group', 'window_idx']]
print(f"\n最终数据形状: {final_data.shape}")
print(f"总特征数: {len(feature_cols)}个")
network_features = [col for col in feature_cols if any(x in col for x in ['density', 'avg_clustering', 'eigenvector', 'reciprocity']) and '_mean_' not in col and '_std_' not in col]
node_features = [col for col in feature_cols if '_mean_' in col or '_std_' in col]
print(f"  - 网络级别特征: {len(network_features)}个（density, avg_clustering, eigenvector, reciprocity × 4种模态）")
print(f"  - 节点级别特征: {len(node_features)}个（5个指标 × 2种聚合 × 4种模态）")
print(f"特征示例（网络级别）: {network_features[:5]}")
print(f"特征示例（节点级别）: {node_features[:5]}")

print("\n3.5 处理缺失值")
missing_stats = final_data.isnull().sum()
missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)
if len(missing_stats) > 0:
    print("缺失值统计:")
    print(missing_stats)
    print(f"\n总缺失值比例: {final_data.isnull().sum().sum() / (final_data.shape[0] * final_data.shape[1]):.2%}")
    
    print("\n正在处理缺失值...")
    final_data = final_data.sort_values(['group', 'window_idx'])
    
    for col in final_data.columns:
        if final_data[col].isnull().sum() > 0 and col not in ['group', 'window_idx']:
            final_data[col] = final_data.groupby('group')[col].transform(
                lambda x: x.ffill().bfill().fillna(0)
            )
    
    print("缺失值处理完成")
    print(f"处理后缺失值: {final_data.isnull().sum().sum()}")
else:
    print("✓ 无缺失值")

# 4. 数据划分
print("\n" + "-"*80)
print("4. 数据划分")
print("-"*80)

print("\n4.1 按组划分")
train_raw = final_data[final_data['group'].isin(TRAIN_GROUPS)].copy()
# 测试集划分
test_raw = final_data[final_data['group'].isin(TEST_GROUPS)].copy()

print(f"训练组: {TRAIN_GROUPS}")
print(f"测试组: {TEST_GROUPS}")
print(f"\n训练组数据量: {len(train_raw)} 窗口")
print(f"测试组数据量: {len(test_raw)} 窗口")

print("\n4.2 使用训练组全部数据（不进行时间划分）")
print("策略：直接使用训练组（组1-8）的全部数据作为训练集")
print("原因：避免组内时间划分导致的数据泄露，充分利用训练数据")

# 直接使用训练组的全部数据
train_data = train_raw.copy().sort_values(['group', 'window_idx']).reset_index(drop=True)

# 为了保持兼容性，train_val_data 和 val_data 设为空
train_val_data = pd.DataFrame(columns=train_data.columns)
val_data = pd.DataFrame(columns=train_data.columns)

print(f"\n最终划分结果:")
print(f"训练集: {len(train_data)} 窗口（组{TRAIN_GROUPS}的全部数据）")
print(f"训练验证集: 0 窗口（已合并到训练集）")
print(f"验证集: 0 窗口（已合并到测试集）")
print(f"测试集: {len(test_raw)} 窗口（组{TEST_GROUPS}的全部数据，窗口数最多的4个组）")

print("\n4.3 特征标准化")
print("正在标准化特征...")

exclude_cols = ['group', 'window_idx']
# 创建空的val_data用于兼容性
# normalize_features需要val_df参数，创建一个与train_data结构相同的空DataFrame
empty_val = pd.DataFrame(columns=train_data.columns)
# 确保empty_val有正确的索引类型（避免后续合并问题）
empty_val = empty_val.astype(train_data.dtypes)
train_scaled, val_scaled, test_scaled, scaler = normalize_features(
    train_data, empty_val, test_raw, train_val_df=None,
    exclude_cols=exclude_cols
)
train_val_scaled = pd.DataFrame(columns=train_scaled.columns)

print("标准化完成")
print(f"训练集形状: {train_scaled.shape}（组{TRAIN_GROUPS}的全部数据）")
print(f"训练验证集: 0 窗口（已合并到训练集）")
print(f"验证集: 0 窗口（已合并到测试集）")
print(f"测试集形状: {test_scaled.shape}（组{TEST_GROUPS}的全部数据，窗口数最多的4个组）")

# 5. 保存处理后的数据
print("\n" + "-"*80)
print("5. 保存处理后的数据")
print("-"*80)

save_intermediate('train_data', train_scaled)
# 保存空的 train_val_data 以保持兼容性（其他脚本会合并 train_data 和 train_val_data）
save_intermediate('train_val_data', train_val_scaled)
save_intermediate('val_data', val_scaled)
save_intermediate('test_data', test_scaled)
save_intermediate('scaler', scaler)

feature_cols = [col for col in train_scaled.columns if col not in exclude_cols]
save_intermediate('feature_names', feature_cols)

save_intermediate('task_metrics', task_df)
if session_edges_df is not None:
    save_intermediate('session_edges', session_edges_df)
    save_intermediate('session_nodes', session_nodes_df)
    save_intermediate('session_metrics', session_metrics_df)

print(f"\n✓ 训练数据已保存到 {INTERMEDIATE_DIR}")
print(f"✓ 特征数量: {len(feature_cols)}（只包含windowed_output的动态特征）")
print(f"✓ 静态特征已单独保存（用于后处理分析）")

# 6. 数据划分报告
print("\n" + "-"*80)
print("6. 数据划分报告")
print("-"*80)

report_lines = []
report_lines.append("=" * 60)
report_lines.append("数据划分报告")
report_lines.append("=" * 60)
report_lines.append(f"\n训练组: {TRAIN_GROUPS}")
report_lines.append(f"测试组: {TEST_GROUPS}")
report_lines.append(f"\n训练集: {len(train_scaled)} 窗口（组{TRAIN_GROUPS}的全部数据）")
report_lines.append(f"训练验证集: 0 窗口（已合并到训练集，避免数据泄露）")
report_lines.append(f"验证集: 0 窗口（已合并到测试集）")
report_lines.append(f"测试集: {len(test_scaled)} 窗口（组{TEST_GROUPS}的全部数据，窗口数最多的4个组）")
report_lines.append(f"\n总特征数: {len(feature_cols)}（只包含windowed_output的动态特征）")
report_lines.append(f"\n各组窗口数统计:")
report_lines.append(f"训练组: {train_scaled['group'].value_counts().sort_index().to_dict()}")
report_lines.append(f"测试组: {test_scaled['group'].value_counts().sort_index().to_dict()}")

report_text = "\n".join(report_lines)
print(report_text)

with open(REPORTS_DIR / "data_split_report.txt", 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\n✓ 报告已保存到 {REPORTS_DIR / 'data_split_report.txt'}")

print("\n" + "="*80)
print("数据准备完成！")
print("="*80)
print("\n下一步：运行 `01_unsupervised_feature_selection.py` 进行无监督特征选择")

