"""
01 - HMM建模与标签生成

本脚本完成以下任务：
1. 加载预处理后的数据
2. 训练粗粒度HMM（4状态）识别协作状态
3. 训练细粒度HMM（针对低沟通状态）
4. 预测所有数据集的HMM状态
5. 将HMM状态映射为二分类标签（0=无需干预，1=需要干预）
6. 保存标签和HMM模型

**专业分析**：
- HMM用于发现潜在的协作状态（平衡/不平衡）
- 低沟通状态映射为需要干预（标签1）
- 高沟通状态映射为无需干预（标签0）
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

# HMM配置
HMM_CONFIG = {
    "coarse_n_states": 4,  # 粗粒度状态数
    "fine_n_states": 3,  # 细粒度状态数
    "n_iter": 100,  # Baum-Welch算法迭代次数
    "covariance_type": "full",  # 协方差类型
    "random_state": 42,
}

# HMM多分类配置
HMM_N_CLASSES = 4  # 3或4分类，4表示直接使用HMM的4个状态
HMM_STATE_MAPPING = None  # 如果n_classes=3，可以指定状态映射，例如：{0: 0, 1: 0, 2: 1, 3: 2}

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


def save_model(name, model, directory=None):
    """保存模型"""
    if directory is None:
        directory = MODELS_DIR
    filepath = directory / f"{name}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved: {filepath}")


try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Error: hmmlearn not installed, please run: pip install hmmlearn")
    exit(1)


def train_coarse_hmm(X_train, n_states=4, n_iter=100, covariance_type='full', random_state=42):
    """
    训练粗粒度HMM
    
    Parameters:
    -----------
    X_train : np.ndarray
        训练数据 (n_samples, n_features)
    n_states : int
        状态数
    n_iter : int
        迭代次数
    covariance_type : str
        协方差类型
    random_state : int
        随机种子
    
    Returns:
    --------
    model : hmm.GaussianHMM
        训练好的HMM模型
    """
    print(f"\nTraining coarse-grained HMM ({n_states} states)...")
    
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        verbose=False
    )
    
    model.fit(X_train)
    
    print(f"✓ HMM training completed, convergence iterations: {model.monitor_.iter}")
    return model


def predict_hmm_states(model, X):
    """
    使用HMM预测状态序列
    
    Parameters:
    -----------
    model : hmm.GaussianHMM
        HMM模型
    X : np.ndarray
        观测数据
    
    Returns:
    --------
    states : np.ndarray
        预测的状态序列
    """
    states = model.predict(X)
    return states


def map_states_to_labels(states, n_classes=4, state_mapping=None):
    """
    将HMM状态映射为多分类标签
    
    策略：
    - 直接使用HMM的4个状态作为4分类标签
    - 或者映射为2分类（需要干预 vs 不需要干预）
    - 或者映射为3分类（可选）
    
    Parameters:
    -----------
    states : np.ndarray
        HMM状态序列
    n_classes : int
        分类数量（2, 3或4）
    state_mapping : dict, optional
        状态映射字典，如果为None则使用默认映射
        例如：
        - 2分类：{0: 1, 1: 1, 2: 0, 3: 0} 表示状态0,1需要干预(1)，状态2,3不需要干预(0)
        - 3分类：{0: 0, 1: 0, 2: 1, 3: 2} 表示状态0,1->类别0（低沟通），状态2->类别1（中等），状态3->类别2（高沟通）
    
    Returns:
    --------
    labels : np.ndarray
        多分类标签 (0, 1, 2, ...)
    """
    if n_classes == 4:
        # 直接使用4个状态作为4分类
        labels = states.copy()
    elif n_classes == 2:
        # 映射为2分类（0=不需要干预，1=需要干预）
        if state_mapping is None:
            # 默认映射：状态0,1 -> 需要干预(1)，状态2,3 -> 不需要干预(0)
            # 这个映射会在状态分析后根据健康度分数自动生成
            state_mapping = {0: 1, 1: 1, 2: 0, 3: 0}
        labels = np.array([state_mapping[s] for s in states])
    elif n_classes == 3:
        # 映射为3分类
        if state_mapping is None:
            # 默认映射：状态0,1 -> 类别0（低沟通），状态2 -> 类别1（中等），状态3 -> 类别2（高沟通）
            state_mapping = {0: 0, 1: 0, 2: 1, 3: 2}
        labels = np.array([state_mapping[s] for s in states])
    else:
        raise ValueError(f"n_classes必须是2、3或4，当前为{n_classes}")
    
    return labels


# ============================================================================
# 主程序部分
# ============================================================================

print("\n" + "="*80)
print("01 - HMM Modeling and Label Generation")
print("="*80)

# 1. 加载数据
print("\n" + "-"*80)
print("1. Load Preprocessed Data")
print("-"*80)

train_data = load_intermediate('train_data')
train_val_data = load_intermediate('train_val_data')
val_data = load_intermediate('val_data')
test_data = load_intermediate('test_data')
feature_names = load_intermediate('feature_names')

print(f"\nData shapes:")
print(f"Training set: {train_data.shape}")
print(f"Test set: {test_data.shape}")
print(f"Number of features: {len(feature_names)}")

# 提取特征列
exclude_cols = ['group', 'window_idx']
feature_cols = [col for col in train_data.columns if col not in exclude_cols]

X_train = train_data[feature_cols].values
# 检查 train_val_data 是否为空
if len(train_val_data) > 0:
    X_train_val = train_val_data[feature_cols].values
else:
    X_train_val = np.array([]).reshape(0, len(feature_cols))
# 检查 val_data 是否为空（已合并到测试集）
if len(val_data) > 0:
    X_val = val_data[feature_cols].values
else:
    X_val = np.array([]).reshape(0, len(feature_cols))
X_test = test_data[feature_cols].values

print(f"\nFeature matrix shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# 2. 训练粗粒度HMM
print("\n" + "-"*80)
print("2. Train Coarse-grained HMM")
print("-"*80)

coarse_hmm = train_coarse_hmm(
    X_train,
    n_states=HMM_CONFIG['coarse_n_states'],
    n_iter=HMM_CONFIG['n_iter'],
    covariance_type=HMM_CONFIG['covariance_type'],
    random_state=HMM_CONFIG['random_state']
)

# 保存模型
save_model('coarse_hmm', coarse_hmm)

# 3. 预测所有数据集的状态
print("\n" + "-"*80)
print("3. Predict HMM States")
print("-"*80)

states_train = predict_hmm_states(coarse_hmm, X_train)
# 如果 X_train_val 为空，跳过预测
if len(X_train_val) > 0:
    states_train_val = predict_hmm_states(coarse_hmm, X_train_val)
else:
    states_train_val = np.array([])
# 如果 X_val 为空（已合并到测试集），跳过预测
if len(X_val) > 0:
    states_val = predict_hmm_states(coarse_hmm, X_val)
else:
    states_val = np.array([])
states_test = predict_hmm_states(coarse_hmm, X_test)

print(f"\nState distribution:")
print(f"Training set: {pd.Series(states_train).value_counts().sort_index().to_dict()}")
print(f"Test set: {pd.Series(states_test).value_counts().sort_index().to_dict()}")

# 3.1 分析HMM状态的语义含义
print("\n" + "-"*80)
print("3.1 Analyze HMM State Semantics")
print("-"*80)
print("\n⚠️  Important: HMM states have no preset semantics, need to analyze their meaning based on feature values")
print("The following analyzes the mean feature values for each state to help understand state meanings:\n")

# 创建状态分析DataFrame
state_analysis = pd.DataFrame(index=feature_cols)
state_means = {}  # 存储每个状态的特征均值

for state in range(HMM_CONFIG['coarse_n_states']):
    # 获取该状态的所有样本
    state_mask = states_train == state
    if np.sum(state_mask) > 0:
        state_features = X_train[state_mask]
        state_mean = np.mean(state_features, axis=0)
        state_means[state] = state_mean
        state_analysis[f'State_{state}_Mean'] = state_mean
        state_analysis[f'State_{state}_Count'] = np.sum(state_mask)
    else:
        state_means[state] = np.zeros(len(feature_cols))
        state_analysis[f'State_{state}_Mean'] = np.zeros(len(feature_cols))
        state_analysis[f'State_{state}_Count'] = 0

# 显示关键特征的分析（选择一些有代表性的特征）
print("="*80)
print("Mean Values of Key Features Across States (Help Understand State Meanings):")
print("="*80)

# 选择关键特征（包含density, clustering, eigenvector, reciprocity等）
key_features = [f for f in feature_cols if any(keyword in f.lower() for keyword in 
                ['density', 'clustering', 'eigenvector', 'reciprocity', 'betweenness', 'degree', 'closeness'])]

if len(key_features) > 0:
    print(f"\nKey feature analysis (total {len(key_features)} features):")
    key_analysis = state_analysis.loc[key_features]
    print(key_analysis.to_string())
else:
    # 如果没有找到关键特征，显示前10个特征
    print(f"\nAnalysis of first 10 features:")
    print(state_analysis.head(10).to_string())

# 分析每个状态的特征模式
print("\n" + "="*80)
print("State Feature Pattern Analysis (Help Determine Which State is 'Low Communication'/'High Communication'):")
print("="*80)

for state in range(HMM_CONFIG['coarse_n_states']):
    print(f"\nState {state}:")
    print(f"  Sample count: {int(state_analysis[f'State_{state}_Count'].iloc[0])}")
    
    # 分析关键指标
    state_mean = state_means[state]
    feature_dict = dict(zip(feature_cols, state_mean))
    
    # 查找density相关的特征（通常density高表示沟通频繁）
    density_features = {k: v for k, v in feature_dict.items() if 'density' in k.lower()}
    if density_features:
        avg_density = np.mean(list(density_features.values()))
        print(f"  Mean Density: {avg_density:.4f} (high values may indicate frequent communication)")
    
    # 查找clustering相关的特征
    clustering_features = {k: v for k, v in feature_dict.items() if 'clustering' in k.lower()}
    if clustering_features:
        avg_clustering = np.mean(list(clustering_features.values()))
        print(f"  Mean Clustering: {avg_clustering:.4f} (high values may indicate tight collaboration)")
    
    # 查找eigenvector相关的特征
    eigenvector_features = {k: v for k, v in feature_dict.items() if 'eigenvector' in k.lower()}
    if eigenvector_features:
        avg_eigenvector = np.mean(list(eigenvector_features.values()))
        print(f"  Mean Eigenvector: {avg_eigenvector:.4f} (high values may indicate high influence)")

print("\n" + "="*80)
print("💡 Suggestions:")
print("  1. Review the above feature values to determine which state has lower values (may be 'low communication' state)")
print("  2. Determine which state has higher values (may be 'high communication' state)")
print("  3. Based on the analysis results, set HMM_STATE_MAPPING in the configuration below")
print("  4. For example: If state 0 is low communication and state 3 is high communication, you can set:")
print("     HMM_STATE_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3}  # Keep 4 classes")
print("     Or map to 3 classes: {0: 0, 1: 0, 2: 1, 3: 2}  # Low->0, Medium->1, High->2")
print("="*80)

# 保存状态分析结果
state_analysis_path = REPORTS_DIR / "hmm_state_analysis.csv"
state_analysis.to_csv(state_analysis_path, encoding='utf-8')
print(f"\n✓ State analysis results saved to: {state_analysis_path}")
print("  You can open the CSV file to view detailed values of all features for each state")

# 3.2 结合任务性能数据判断哪些状态需要干预
print("\n" + "-"*80)
print("3.2 Analyze HMM States vs Task Performance (Determine Intervention Need)")
print("-"*80)

try:
    # 加载任务性能数据
    task_metrics = load_intermediate('task_metrics')
    
    # 将HMM状态与任务性能关联
    # 创建DataFrame：每个窗口对应一个状态和组
    state_performance_df = pd.DataFrame({
        'group': train_data['group'].values,
        'window_idx': train_data['window_idx'].values,
        'state': states_train
    })
    
    # 合并任务性能数据（按组）
    if 'group' in task_metrics.columns:
        # 如果task_metrics有group列，合并
        state_performance_df = state_performance_df.merge(
            task_metrics, on='group', how='left', suffixes=('', '_task')
        )
    else:
        # 如果没有group列，尝试用索引匹配
        print("  Note: task_metrics doesn't have 'group' column, using index matching")
        task_metrics_indexed = task_metrics.reset_index()
        if 'group' in task_metrics_indexed.columns:
            state_performance_df = state_performance_df.merge(
                task_metrics_indexed, on='group', how='left', suffixes=('', '_task')
            )
    
    # 分析每个状态对应的任务性能
    print("\n" + "="*80)
    print("Task Performance by HMM State (Lower performance may indicate need for intervention):")
    print("="*80)
    
    # 找出任务性能相关的列（如完成时间、效率等）
    performance_cols = [col for col in state_performance_df.columns 
                        if any(keyword in col.lower() for keyword in 
                              ['time', 'duration', 'efficiency', 'completion', 'score', 'performance', 'quality'])]
    
    if len(performance_cols) > 0:
        print(f"\nFound {len(performance_cols)} performance-related columns:")
        for col in performance_cols[:5]:  # 只显示前5个
            print(f"  - {col}")
        
        # 分析每个状态的平均性能
        for state in range(HMM_CONFIG['coarse_n_states']):
            state_mask = state_performance_df['state'] == state
            state_data = state_performance_df[state_mask]
            
            if len(state_data) > 0:
                print(f"\nState {state} (样本数: {len(state_data)}):")
                
                # 计算每个性能指标的平均值
                for col in performance_cols[:3]:  # 只分析前3个最重要的指标
                    if col in state_data.columns and state_data[col].notna().sum() > 0:
                        avg_value = state_data[col].mean()
                        print(f"  Average {col}: {avg_value:.4f}")
                
                # 如果有多组数据，计算组内平均
                if 'group' in state_data.columns:
                    unique_groups = state_data['group'].unique()
                    if len(unique_groups) > 1:
                        group_avgs = []
                        for g in unique_groups:
                            group_data = state_data[state_data['group'] == g]
                            if len(group_data) > 0 and len(performance_cols) > 0:
                                # 取第一个性能指标的平均值
                                perf_col = performance_cols[0]
                                if perf_col in group_data.columns and group_data[perf_col].notna().sum() > 0:
                                    group_avgs.append(group_data[perf_col].mean())
                        if len(group_avgs) > 0:
                            print(f"  Average performance across {len(unique_groups)} groups: {np.mean(group_avgs):.4f}")
    else:
        print("\n⚠️  Warning: No performance-related columns found in task_metrics")
        print("  Available columns:", list(state_performance_df.columns)[:10])
        print("\n  Alternative method: Analyze by feature values (low values = may need intervention)")
        print("  - States with low density/clustering/eigenvector may indicate low communication")
        print("  - States with low reciprocity may indicate poor collaboration")
    
    # 基于特征值和性能数据给出建议
    print("\n" + "="*80)
    print("💡 Intervention Recommendation:")
    print("="*80)
    
    # 计算每个状态的综合"健康度"分数（基于关键特征）
    state_health_scores = {}
    for state in range(HMM_CONFIG['coarse_n_states']):
        state_mean = state_means[state]
        feature_dict = dict(zip(feature_cols, state_mean))
        
        # 计算关键特征的平均值（越高越好）
        key_features_list = []
        for keyword in ['density', 'clustering', 'eigenvector', 'reciprocity']:
            features = {k: v for k, v in feature_dict.items() if keyword in k.lower()}
            if features:
                key_features_list.extend(list(features.values()))
        
        if len(key_features_list) > 0:
            health_score = np.mean(key_features_list)
            state_health_scores[state] = health_score
        else:
            state_health_scores[state] = 0.0
    
    # 按健康度分数排序
    sorted_states = sorted(state_health_scores.items(), key=lambda x: x[1])
    
    print("\nStates ranked by collaboration health (lower = may need intervention):")
    for rank, (state, score) in enumerate(sorted_states, 1):
        intervention_level = "HIGH" if rank <= 2 else "LOW"
        print(f"  Rank {rank}: State {state} (health score: {score:.4f}) - Intervention need: {intervention_level}")
    
    print("\n📋 Suggested interpretation:")
    print(f"  - States {sorted_states[0][0]} and {sorted_states[1][0]}: Likely need intervention (low collaboration)")
    print(f"  - States {sorted_states[2][0]} and {sorted_states[3][0]}: Likely no intervention needed (better collaboration)")
    print("\n  You can map states to intervention labels:")
    print(f"    - Need intervention (label 1): [{sorted_states[0][0]}, {sorted_states[1][0]}]")
    print(f"    - No intervention (label 0): [{sorted_states[2][0]}, {sorted_states[3][0]}]")
    
    # 自动生成2分类映射建议
    intervention_states = [sorted_states[0][0], sorted_states[1][0]]
    no_intervention_states = [sorted_states[2][0], sorted_states[3][0]]
    binary_mapping = {}
    for state in range(4):
        if state in intervention_states:
            binary_mapping[state] = 1  # 需要干预
        else:
            binary_mapping[state] = 0  # 不需要干预
    
    print(f"\n💡 Suggested 2-class mapping (based on health scores):")
    print(f"    HMM_STATE_MAPPING = {binary_mapping}")
    print(f"    # State {sorted_states[0][0]}, {sorted_states[1][0]} -> 1 (need intervention)")
    print(f"    # State {sorted_states[2][0]}, {sorted_states[3][0]} -> 0 (no intervention)")
    print("="*80)
    
    # 保存建议的映射供后续使用
    suggested_binary_mapping = binary_mapping
    
except Exception as e:
    print(f"\n⚠️  Warning: Could not analyze task performance: {e}")
    print("  Falling back to feature-based analysis only")
    print("\n  Recommendation: Review the feature values in hmm_state_analysis.csv")
    print("  - States with lower density/clustering/eigenvector values likely need intervention")
    print("  - States with higher values likely don't need intervention")
    # 如果没有分析结果，使用默认映射
    suggested_binary_mapping = {0: 1, 1: 1, 2: 0, 3: 0}  # 默认：状态0,1需要干预，状态2,3不需要

# 4. 映射状态到标签
print("\n" + "-"*80)
print("4. Map HMM States to Multi-class Labels")
print("-"*80)

# HMM多分类配置
# ⚠️  重要：根据上面的状态分析结果，设置状态映射
# 推荐使用2分类（需要干预 vs 不需要干预），因为：
# 1. 数据量更适合2分类
# 2. 更符合实际应用需求
# 3. 类别更平衡，性能更好

# 自动使用建议的2分类映射（如果可用）
try:
    HMM_N_CLASSES = 2  # 改为2分类
    HMM_STATE_MAPPING = suggested_binary_mapping
    print(f"\n✓ Using suggested 2-class mapping based on state health analysis")
except NameError:
    # 如果没有分析结果，使用默认2分类映射
    HMM_N_CLASSES = 2
    HMM_STATE_MAPPING = {0: 1, 1: 1, 2: 0, 3: 0}  # 默认：状态0,1需要干预，状态2,3不需要
    print(f"\n⚠️  Using default 2-class mapping (you can adjust based on state analysis)")

# 如果需要使用其他分类数，可以手动修改：
# HMM_N_CLASSES = 4  # 4分类：直接使用HMM的4个状态
# HMM_STATE_MAPPING = None
# 
# HMM_N_CLASSES = 3  # 3分类
# HMM_STATE_MAPPING = {0: 0, 1: 0, 2: 1, 3: 2}  # 状态0,1->类别0（低沟通），状态2->类别1（中等），状态3->类别2（高沟通）

print(f"\nUsing HMM states as {HMM_N_CLASSES}-class labels")
if HMM_N_CLASSES == 2:
    print(f"State mapping: {HMM_STATE_MAPPING}")
    print(f"  - Label 0: No intervention needed (States: {[s for s, l in HMM_STATE_MAPPING.items() if l == 0]})")
    print(f"  - Label 1: Intervention needed (States: {[s for s, l in HMM_STATE_MAPPING.items() if l == 1]})")
elif HMM_N_CLASSES == 3:
    print(f"State mapping: {HMM_STATE_MAPPING if HMM_STATE_MAPPING else 'Default mapping (0,1->0, 2->1, 3->2)'}")
elif HMM_N_CLASSES == 4:
    if HMM_STATE_MAPPING is None:
        print("Directly using HMM's 4 states (0,1,2,3) as 4 classes")
        print("⚠️  Note: Classes 0,1,2,3 have no preset semantics, need to understand their meanings based on the state analysis above")
    else:
        print(f"State mapping: {HMM_STATE_MAPPING}")

y_train = map_states_to_labels(states_train, n_classes=HMM_N_CLASSES, state_mapping=HMM_STATE_MAPPING)
# 如果 states_train_val 为空，创建空数组
if len(states_train_val) > 0:
    y_train_val = map_states_to_labels(states_train_val, n_classes=HMM_N_CLASSES, state_mapping=HMM_STATE_MAPPING)
else:
    y_train_val = np.array([])
# 如果 states_val 为空（已合并到测试集），创建空数组
if len(states_val) > 0:
    y_val = map_states_to_labels(states_val, n_classes=HMM_N_CLASSES, state_mapping=HMM_STATE_MAPPING)
else:
    y_val = np.array([])
y_test = map_states_to_labels(states_test, n_classes=HMM_N_CLASSES, state_mapping=HMM_STATE_MAPPING)

print(f"\nLabel distribution ({HMM_N_CLASSES} classes):")
if HMM_N_CLASSES == 2:
    print(f"Training set - Label 0 (No intervention): {np.sum(y_train == 0)}, Label 1 (Need intervention): {np.sum(y_train == 1)}")
    print(f"Test set - Label 0 (No intervention): {np.sum(y_test == 0)}, Label 1 (Need intervention): {np.sum(y_test == 1)}")
    print(f"  Training set balance: {np.sum(y_train == 0)/len(y_train):.2%} vs {np.sum(y_train == 1)/len(y_train):.2%}")
    print(f"  Test set balance: {np.sum(y_test == 0)/len(y_test):.2%} vs {np.sum(y_test == 1)/len(y_test):.2%}")
elif HMM_N_CLASSES == 3:
    print(f"Training set - Label 0: {np.sum(y_train == 0)}, Label 1: {np.sum(y_train == 1)}, Label 2: {np.sum(y_train == 2)}")
    print(f"Test set - Label 0: {np.sum(y_test == 0)}, Label 1: {np.sum(y_test == 1)}, Label 2: {np.sum(y_test == 2)}")
elif HMM_N_CLASSES == 4:
    print(f"Training set - Label 0: {np.sum(y_train == 0)}, Label 1: {np.sum(y_train == 1)}, Label 2: {np.sum(y_train == 2)}, Label 3: {np.sum(y_train == 3)}")
    print(f"Test set - Label 0: {np.sum(y_test == 0)}, Label 1: {np.sum(y_test == 1)}, Label 2: {np.sum(y_test == 2)}, Label 3: {np.sum(y_test == 3)}")

# 5. 保存结果
print("\n" + "-"*80)
print("5. Save Results")
print("-"*80)

save_intermediate('states_train', states_train)
if len(states_train_val) > 0:
    save_intermediate('states_train_val', states_train_val)
else:
    save_intermediate('states_train_val', np.array([]))
# 保存空的验证集状态（已合并到测试集）
if len(states_val) > 0:
    save_intermediate('states_val', states_val)
else:
    save_intermediate('states_val', np.array([]))
save_intermediate('states_test', states_test)

save_intermediate('y_train', y_train)
if len(y_train_val) > 0:
    save_intermediate('y_train_val', y_train_val)
else:
    save_intermediate('y_train_val', np.array([]))
# 保存空的验证集标签（已合并到测试集）
if len(y_val) > 0:
    save_intermediate('y_val', y_val)
else:
    save_intermediate('y_val', np.array([]))
save_intermediate('y_test', y_test)

# 6. 生成报告
print("\n" + "-"*80)
print("6. Generate HMM Analysis Report")
print("-"*80)

report_lines = []
report_lines.append("=" * 60)
report_lines.append("HMM Modeling Report")
report_lines.append("=" * 60)
report_lines.append(f"\nHMM Configuration:")
report_lines.append(f"  Number of states: {HMM_CONFIG['coarse_n_states']}")
report_lines.append(f"  Iterations: {HMM_CONFIG['n_iter']}")
report_lines.append(f"  Covariance type: {HMM_CONFIG['covariance_type']}")
report_lines.append(f"\nState distribution:")
report_lines.append(f"  Training set: {pd.Series(states_train).value_counts().sort_index().to_dict()}")
report_lines.append(f"  Test set: {pd.Series(states_test).value_counts().sort_index().to_dict()}")
report_lines.append(f"\nState semantic analysis:")
report_lines.append(f"  Detailed state feature analysis saved to: hmm_state_analysis.csv")
report_lines.append(f"  Please check this file to understand feature values for each state and determine state meanings")
report_lines.append(f"\nLabel distribution ({HMM_N_CLASSES} classes):")
if HMM_N_CLASSES == 2:
    report_lines.append(f"  Training set - Label 0 (No intervention): {np.sum(y_train == 0)}, Label 1 (Need intervention): {np.sum(y_train == 1)}")
    report_lines.append(f"  Test set - Label 0 (No intervention): {np.sum(y_test == 0)}, Label 1 (Need intervention): {np.sum(y_test == 1)}")
    report_lines.append(f"  State mapping: {HMM_STATE_MAPPING}")
elif HMM_N_CLASSES == 3:
    report_lines.append(f"  Training set - Label 0: {np.sum(y_train == 0)}, Label 1: {np.sum(y_train == 1)}, Label 2: {np.sum(y_train == 2)}")
    report_lines.append(f"  Test set - Label 0: {np.sum(y_test == 0)}, Label 1: {np.sum(y_test == 1)}, Label 2: {np.sum(y_test == 2)}")
    report_lines.append(f"  State mapping: {HMM_STATE_MAPPING if HMM_STATE_MAPPING else 'Default mapping (0,1->0, 2->1, 3->2)'}")
elif HMM_N_CLASSES == 4:
    report_lines.append(f"  Training set - Label 0: {np.sum(y_train == 0)}, Label 1: {np.sum(y_train == 1)}, Label 2: {np.sum(y_train == 2)}, Label 3: {np.sum(y_train == 3)}")
    report_lines.append(f"  Test set - Label 0: {np.sum(y_test == 0)}, Label 1: {np.sum(y_test == 1)}, Label 2: {np.sum(y_test == 2)}, Label 3: {np.sum(y_test == 3)}")
    if HMM_STATE_MAPPING is not None:
        report_lines.append(f"  State mapping: {HMM_STATE_MAPPING}")

report_text = "\n".join(report_lines)
print(report_text)

with open(REPORTS_DIR / "hmm_analysis_report.txt", 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\n✓ Report saved to {REPORTS_DIR / 'hmm_analysis_report.txt'}")

print("\n" + "="*80)
print("HMM modeling completed!")
print("="*80)
print("\nNext step: Run `02_supervised_feature_selection.py` for supervised feature selection")

