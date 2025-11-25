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


def save_model(name, model, directory=None):
    """保存模型"""
    if directory is None:
        directory = MODELS_DIR
    filepath = directory / f"{name}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"已保存模型: {filepath}")


try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("错误: hmmlearn未安装，请运行: pip install hmmlearn")
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
    print(f"\n训练粗粒度HMM ({n_states}状态)...")
    
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        verbose=False
    )
    
    model.fit(X_train)
    
    print(f"✓ HMM训练完成，收敛迭代次数: {model.monitor_.iter}")
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
    - 或者映射为3分类（可选）
    
    Parameters:
    -----------
    states : np.ndarray
        HMM状态序列
    n_classes : int
        分类数量（3或4）
    state_mapping : dict, optional
        状态映射字典，如果为None则使用默认映射
        例如：{0: 0, 1: 1, 2: 2, 3: 2} 表示将状态2和3合并为类别2
    
    Returns:
    --------
    labels : np.ndarray
        多分类标签 (0, 1, 2, ...)
    """
    if n_classes == 4:
        # 直接使用4个状态作为4分类
        labels = states.copy()
    elif n_classes == 3:
        # 映射为3分类
        if state_mapping is None:
            # 默认映射：状态0,1 -> 类别0（低沟通），状态2 -> 类别1（中等），状态3 -> 类别2（高沟通）
            state_mapping = {0: 0, 1: 0, 2: 1, 3: 2}
        labels = np.array([state_mapping[s] for s in states])
    else:
        raise ValueError(f"n_classes必须是3或4，当前为{n_classes}")
    
    return labels


# ============================================================================
# 主程序部分
# ============================================================================

print("\n" + "="*80)
print("01 - HMM建模与标签生成")
print("="*80)

# 1. 加载数据
print("\n" + "-"*80)
print("1. 加载预处理后的数据")
print("-"*80)

train_data = load_intermediate('train_data')
train_val_data = load_intermediate('train_val_data')
val_data = load_intermediate('val_data')
test_data = load_intermediate('test_data')
feature_names = load_intermediate('feature_names')

print(f"\n数据形状:")
print(f"训练集: {train_data.shape}")
print(f"测试集: {test_data.shape}")
print(f"特征数: {len(feature_names)}")

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

print(f"\n特征矩阵形状:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# 2. 训练粗粒度HMM
print("\n" + "-"*80)
print("2. 训练粗粒度HMM")
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
print("3. 预测HMM状态")
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

print(f"\n状态分布:")
print(f"训练集: {pd.Series(states_train).value_counts().sort_index().to_dict()}")
print(f"测试集: {pd.Series(states_test).value_counts().sort_index().to_dict()}")

# 3.1 分析HMM状态的语义含义
print("\n" + "-"*80)
print("3.1 分析HMM状态的语义含义")
print("-"*80)
print("\n⚠️  重要：HMM状态本身没有预设语义，需要根据特征值分析其含义")
print("以下分析每个状态的平均特征值，帮助理解状态含义：\n")

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
        state_analysis[f'状态{state}_均值'] = state_mean
        state_analysis[f'状态{state}_样本数'] = np.sum(state_mask)
    else:
        state_means[state] = np.zeros(len(feature_cols))
        state_analysis[f'状态{state}_均值'] = np.zeros(len(feature_cols))
        state_analysis[f'状态{state}_样本数'] = 0

# 显示关键特征的分析（选择一些有代表性的特征）
print("="*80)
print("关键特征在各状态下的平均值（帮助理解状态含义）:")
print("="*80)

# 选择关键特征（包含density, clustering, eigenvector, reciprocity等）
key_features = [f for f in feature_cols if any(keyword in f.lower() for keyword in 
                ['density', 'clustering', 'eigenvector', 'reciprocity', 'betweenness', 'degree', 'closeness'])]

if len(key_features) > 0:
    print(f"\n关键特征分析（共{len(key_features)}个）:")
    key_analysis = state_analysis.loc[key_features]
    print(key_analysis.to_string())
else:
    # 如果没有找到关键特征，显示前10个特征
    print(f"\n前10个特征的分析:")
    print(state_analysis.head(10).to_string())

# 分析每个状态的特征模式
print("\n" + "="*80)
print("状态特征模式分析（帮助判断哪个状态是'低沟通'/'高沟通'）:")
print("="*80)

for state in range(HMM_CONFIG['coarse_n_states']):
    print(f"\n状态{state}:")
    print(f"  样本数: {int(state_analysis[f'状态{state}_样本数'].iloc[0])}")
    
    # 分析关键指标
    state_mean = state_means[state]
    feature_dict = dict(zip(feature_cols, state_mean))
    
    # 查找density相关的特征（通常density高表示沟通频繁）
    density_features = {k: v for k, v in feature_dict.items() if 'density' in k.lower()}
    if density_features:
        avg_density = np.mean(list(density_features.values()))
        print(f"  平均Density: {avg_density:.4f} (高值可能表示沟通频繁)")
    
    # 查找clustering相关的特征
    clustering_features = {k: v for k, v in feature_dict.items() if 'clustering' in k.lower()}
    if clustering_features:
        avg_clustering = np.mean(list(clustering_features.values()))
        print(f"  平均Clustering: {avg_clustering:.4f} (高值可能表示协作紧密)")
    
    # 查找eigenvector相关的特征
    eigenvector_features = {k: v for k, v in feature_dict.items() if 'eigenvector' in k.lower()}
    if eigenvector_features:
        avg_eigenvector = np.mean(list(eigenvector_features.values()))
        print(f"  平均Eigenvector: {avg_eigenvector:.4f} (高值可能表示影响力大)")

print("\n" + "="*80)
print("💡 建议：")
print("  1. 查看上述特征值，判断哪个状态的特征值较低（可能是'低沟通'状态）")
print("  2. 判断哪个状态的特征值较高（可能是'高沟通'状态）")
print("  3. 根据分析结果，在下面的配置中设置 HMM_STATE_MAPPING")
print("  4. 例如：如果状态0是低沟通，状态3是高沟通，可以设置：")
print("     HMM_STATE_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3}  # 保持4分类")
print("     或者映射为3分类：{0: 0, 1: 0, 2: 1, 3: 2}  # 低沟通->0, 中等->1, 高沟通->2")
print("="*80)

# 保存状态分析结果
state_analysis_path = REPORTS_DIR / "hmm_state_analysis.csv"
state_analysis.to_csv(state_analysis_path, encoding='utf-8')
print(f"\n✓ 状态分析结果已保存到: {state_analysis_path}")
print("  可以打开CSV文件查看所有特征在每个状态下的详细值")

# 4. 映射状态到标签
print("\n" + "-"*80)
print("4. 映射HMM状态到多分类标签")
print("-"*80)

# HMM多分类配置
# ⚠️  重要：根据上面的状态分析结果，设置状态映射
# 如果保持4分类，直接使用HMM的4个状态：HMM_STATE_MAPPING = None
# 如果映射为3分类，例如：{0: 0, 1: 0, 2: 1, 3: 2} 表示状态0,1->类别0（低沟通），状态2->类别1（中等），状态3->类别2（高沟通）
HMM_N_CLASSES = 4  # 3或4分类，4表示直接使用HMM的4个状态
HMM_STATE_MAPPING = None  # 如果n_classes=3，可以指定状态映射，例如：{0: 0, 1: 0, 2: 1, 3: 2}

print(f"\n使用HMM状态作为{HMM_N_CLASSES}分类标签")
if HMM_N_CLASSES == 3:
    print(f"状态映射: {HMM_STATE_MAPPING if HMM_STATE_MAPPING else '默认映射（0,1->0, 2->1, 3->2）'}")
elif HMM_N_CLASSES == 4:
    if HMM_STATE_MAPPING is None:
        print("直接使用HMM的4个状态（0,1,2,3）作为4个类别")
        print("⚠️  注意：类别0,1,2,3没有预设语义，需要根据上面的状态分析结果理解其含义")
    else:
        print(f"状态映射: {HMM_STATE_MAPPING}")

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

print(f"\n标签分布（{HMM_N_CLASSES}分类）:")
if HMM_N_CLASSES == 4:
    print(f"训练集 - 标签0: {np.sum(y_train == 0)}, 标签1: {np.sum(y_train == 1)}, 标签2: {np.sum(y_train == 2)}, 标签3: {np.sum(y_train == 3)}")
    print(f"测试集 - 标签0: {np.sum(y_test == 0)}, 标签1: {np.sum(y_test == 1)}, 标签2: {np.sum(y_test == 2)}, 标签3: {np.sum(y_test == 3)}")
else:  # 3分类
    print(f"训练集 - 标签0: {np.sum(y_train == 0)}, 标签1: {np.sum(y_train == 1)}, 标签2: {np.sum(y_train == 2)}")
    print(f"测试集 - 标签0: {np.sum(y_test == 0)}, 标签1: {np.sum(y_test == 1)}, 标签2: {np.sum(y_test == 2)}")

# 5. 保存结果
print("\n" + "-"*80)
print("5. 保存结果")
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
print("6. 生成HMM分析报告")
print("-"*80)

report_lines = []
report_lines.append("=" * 60)
report_lines.append("HMM建模报告")
report_lines.append("=" * 60)
report_lines.append(f"\nHMM配置:")
report_lines.append(f"  状态数: {HMM_CONFIG['coarse_n_states']}")
report_lines.append(f"  迭代次数: {HMM_CONFIG['n_iter']}")
report_lines.append(f"  协方差类型: {HMM_CONFIG['covariance_type']}")
report_lines.append(f"\n状态分布:")
report_lines.append(f"  训练集: {pd.Series(states_train).value_counts().sort_index().to_dict()}")
report_lines.append(f"  测试集: {pd.Series(states_test).value_counts().sort_index().to_dict()}")
report_lines.append(f"\n状态语义分析:")
report_lines.append(f"  详细的状态特征分析已保存到: hmm_state_analysis.csv")
report_lines.append(f"  请查看该文件了解每个状态的特征值，判断状态含义")
report_lines.append(f"\n标签分布（{HMM_N_CLASSES}分类）:")
if HMM_N_CLASSES == 4:
    report_lines.append(f"  训练集 - 标签0: {np.sum(y_train == 0)}, 标签1: {np.sum(y_train == 1)}, 标签2: {np.sum(y_train == 2)}, 标签3: {np.sum(y_train == 3)}")
    report_lines.append(f"  测试集 - 标签0: {np.sum(y_test == 0)}, 标签1: {np.sum(y_test == 1)}, 标签2: {np.sum(y_test == 2)}, 标签3: {np.sum(y_test == 3)}")
else:  # 3分类
    report_lines.append(f"  训练集 - 标签0: {np.sum(y_train == 0)}, 标签1: {np.sum(y_train == 1)}, 标签2: {np.sum(y_train == 2)}")
    report_lines.append(f"  测试集 - 标签0: {np.sum(y_test == 0)}, 标签1: {np.sum(y_test == 1)}, 标签2: {np.sum(y_test == 2)}")
if HMM_N_CLASSES == 3:
    report_lines.append(f"\n状态映射: {HMM_STATE_MAPPING if HMM_STATE_MAPPING else '默认映射（0,1->0, 2->1, 3->2）'}")

report_text = "\n".join(report_lines)
print(report_text)

with open(REPORTS_DIR / "hmm_analysis_report.txt", 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\n✓ 报告已保存到 {REPORTS_DIR / 'hmm_analysis_report.txt'}")

print("\n" + "="*80)
print("HMM建模完成！")
print("="*80)
print("\n下一步：运行 `02_supervised_feature_selection.py` 进行有监督特征选择")

