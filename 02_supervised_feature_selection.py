"""
02 - 有监督特征选择

本脚本完成以下任务：
1. 加载HMM生成的标签
2. 使用多种有监督方法评估特征重要性：
   - Random Forest特征重要性
   - Mutual Information
   - LASSO系数
   - RFE (Recursive Feature Elimination)
3. 综合评分选择Top-M特征（M ≤ 16）
4. 评估特征选择效果
5. 保存选定的特征

**专业分析**：
- 16个特征数量较少，但仍需去除冗余特征
- 有监督方法可以识别对预测最重要的特征
- 提高模型可解释性和性能
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
INTERMEDIATE_DIR.mkdir(exist_ok=True)
REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
VISUALIZATIONS_DIR.mkdir(exist_ok=True)

# 有监督特征选择配置
SUPERVISED_CONFIG = {
    "top_m": 20,  # 选择Top-M特征（从54个特征中选择，建议保留约37%的特征）
    # 注意：总特征数从14增加到54，因此增加保留特征数
    # 如果特征数=54，保留20个 ≈ 37%，保留30个 ≈ 56%
    "rf_n_estimators": 100,
    "rf_max_depth": 10,
    "lasso_alpha": 0.01,
    "rfe_n_features": None,  # None表示自动选择
}

# 多分类配置（需要与01_hmm_modeling.py一致）
# 注意：这里需要知道标签是几分类，可以从加载的标签中推断
N_CLASSES = None  # 将在加载标签后自动检测

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
print("02 - 有监督特征选择")
print("="*80)

# 1. 加载数据
print("\n" + "-"*80)
print("1. 加载数据和标签")
print("-"*80)

train_data = load_intermediate('train_data')
train_val_data = load_intermediate('train_val_data')
val_data = load_intermediate('val_data')
test_data = load_intermediate('test_data')
feature_names = load_intermediate('feature_names')

y_train = load_intermediate('y_train')
y_train_val = load_intermediate('y_train_val')
y_val = load_intermediate('y_val')
y_test = load_intermediate('y_test')

print(f"\n数据形状:")
print(f"训练集: {train_data.shape}, 标签: {y_train.shape}")
print(f"测试集: {test_data.shape}, 标签: {y_test.shape}")
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

# 合并训练集和训练验证集用于特征选择
if len(X_train_val) > 0 and len(y_train_val) > 0:
    X_train_full = np.vstack([X_train, X_train_val])
    y_train_full = np.hstack([y_train, y_train_val])
else:
    # 如果 train_val_data 为空，直接使用训练集
    X_train_full = X_train
    y_train_full = y_train

print(f"\n用于特征选择的数据:")
print(f"X_train_full: {X_train_full.shape}, y_train_full: {y_train_full.shape}")

# 检测分类数量（从标签中推断）
if N_CLASSES is None:
    N_CLASSES = len(np.unique(y_train_full))
    print(f"\n检测到分类数量: {N_CLASSES}")

# 2. 多种方法评估特征重要性
print("\n" + "-"*80)
print("2. 评估特征重要性")
print("-"*80)

feature_scores = pd.DataFrame({
    'feature': feature_cols
})

# 2.1 Random Forest特征重要性
print("\n2.1 Random Forest特征重要性...")
rf = RandomForestClassifier(
    n_estimators=SUPERVISED_CONFIG['rf_n_estimators'],
    max_depth=SUPERVISED_CONFIG['rf_max_depth'],
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced' if N_CLASSES > 2 else None  # 多分类时使用平衡权重
)
rf.fit(X_train_full, y_train_full)
feature_scores['rf_importance'] = rf.feature_importances_

# 2.2 Mutual Information
print("2.2 Mutual Information...")
mi_scores = mutual_info_classif(X_train_full, y_train_full, random_state=RANDOM_STATE)
feature_scores['mutual_info'] = mi_scores

# 2.3 LASSO系数
print("2.3 LASSO系数...")
lasso = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5, random_state=RANDOM_STATE, max_iter=1000)
lasso.fit(X_train_full, y_train_full)
feature_scores['lasso_coef'] = np.abs(lasso.coef_)

# 2.4 RFE (Recursive Feature Elimination)
print("2.4 RFE (Recursive Feature Elimination)...")
rfe_n_features = SUPERVISED_CONFIG['top_m'] if SUPERVISED_CONFIG['rfe_n_features'] is None else SUPERVISED_CONFIG['rfe_n_features']
# RFE中的estimator需要支持多分类
rfe_estimator = RandomForestClassifier(
    n_estimators=50, 
    random_state=RANDOM_STATE, 
    n_jobs=-1
)
if N_CLASSES and N_CLASSES > 2:
    rfe_estimator.set_params(class_weight='balanced')
rfe = RFE(
    estimator=rfe_estimator,
    n_features_to_select=rfe_n_features
)
rfe.fit(X_train_full, y_train_full)
feature_scores['rfe_rank'] = rfe.ranking_
feature_scores['rfe_selected'] = rfe.support_.astype(int)

# 3. 综合评分
print("\n" + "-"*80)
print("3. 综合评分")
print("-"*80)

# 归一化各项得分到0-1范围
feature_scores['rf_norm'] = (feature_scores['rf_importance'] - feature_scores['rf_importance'].min()) / (feature_scores['rf_importance'].max() - feature_scores['rf_importance'].min() + 1e-10)
feature_scores['mi_norm'] = (feature_scores['mutual_info'] - feature_scores['mutual_info'].min()) / (feature_scores['mutual_info'].max() - feature_scores['mutual_info'].min() + 1e-10)
feature_scores['lasso_norm'] = (feature_scores['lasso_coef'] - feature_scores['lasso_coef'].min()) / (feature_scores['lasso_coef'].max() - feature_scores['lasso_coef'].min() + 1e-10)
feature_scores['rfe_norm'] = 1.0 / feature_scores['rfe_rank']  # RFE排名越小越好，转换为得分

# 综合得分（加权平均）
feature_scores['combined_score'] = (
    feature_scores['rf_norm'] * 0.3 +
    feature_scores['mi_norm'] * 0.3 +
    feature_scores['lasso_norm'] * 0.2 +
    feature_scores['rfe_norm'] * 0.2
)

# 按综合得分排序
feature_scores = feature_scores.sort_values('combined_score', ascending=False)

print("\n特征重要性排序（Top 10）:")
print(feature_scores[['feature', 'combined_score', 'rf_norm', 'mi_norm', 'lasso_norm', 'rfe_norm']].head(10).to_string(index=False))

# 4. 选择Top-M特征
print("\n" + "-"*80)
print(f"4. 选择Top-{SUPERVISED_CONFIG['top_m']}特征")
print("-"*80)

top_m_features = feature_scores.head(SUPERVISED_CONFIG['top_m'])['feature'].tolist()
print(f"\n选定的特征 ({len(top_m_features)}个):")
for i, feat in enumerate(top_m_features, 1):
    print(f"  {i}. {feat}")

# 5. 特征选择总结
print("\n" + "-"*80)
print("5. 特征选择总结")
print("-"*80)

print(f"\n特征选择完成:")
print(f"  - 从 {len(feature_cols)} 个特征中选择了 {len(top_m_features)} 个最重要的特征")
print(f"  - 选择比例: {len(top_m_features)/len(feature_cols):.1%}")
print(f"\n注意: 特征选择的性能评估将在后续的时间序列模型训练中进行")
print(f"      （在训练集上评估会导致过拟合，结果无意义）")

# 6. 保存结果
print("\n" + "-"*80)
print("6. 保存结果")
print("-"*80)

save_intermediate('top_m_features', top_m_features)
save_intermediate('feature_scores', feature_scores)
save_intermediate('supervised_feature_scores', feature_scores[['feature', 'combined_score']].to_dict('records'))

# 7. 生成报告
print("\n" + "-"*80)
print("7. 生成特征选择报告")
print("-"*80)

report_lines = []
report_lines.append("=" * 60)
report_lines.append("有监督特征选择报告")
report_lines.append("=" * 60)
report_lines.append(f"\n配置:")
report_lines.append(f"  选择特征数: {SUPERVISED_CONFIG['top_m']}")
report_lines.append(f"  总特征数: {len(feature_cols)}")
report_lines.append(f"\n选定的特征:")
for i, feat in enumerate(top_m_features, 1):
    score = feature_scores[feature_scores['feature'] == feat]['combined_score'].values[0]
    report_lines.append(f"  {i}. {feat} (得分: {score:.4f})")
report_lines.append(f"\n说明:")
report_lines.append(f"  - 特征选择基于4种方法的综合评分（RF重要性、互信息、LASSO、RFE）")
report_lines.append(f"  - 性能评估将在后续时间序列模型训练中进行")
report_lines.append(f"  - 在训练集上评估会导致过拟合，结果无意义")

report_text = "\n".join(report_lines)
print(report_text)

with open(REPORTS_DIR / "feature_selection_supervised_report.txt", 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\n✓ 报告已保存到 {REPORTS_DIR / 'feature_selection_supervised_report.txt'}")

print("\n" + "="*80)
print("有监督特征选择完成！")
print("="*80)
print("\n下一步：运行 `03_time_series_training.py` 进行时间序列模型训练")

