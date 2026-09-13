# -*- coding: utf-8 -*-
"""
问题 2（1）：挖掘高钾、铅钡玻璃的分类规律
================================================
方法依据（文献）：
  - 类型-成分相关性：基于 CatBoost 的玻璃成分与风化关系实证探究（2023）发现
    玻璃类型与 SiO2、K2O、PbO、BaO 的相关系数可达 0.7，本文先对类型与各成分
    做 Spearman 秩相关排序，刻画"哪些成分决定类别"；
  - 随机森林特征重要性：刘振文等（2025）以随机森林特征重要性筛出 7 个分类
    核心成分（PbO>BaO>SiO2>SrO>K2O>Al2O3>P2O5），本文复用该降维流程，
    得到数据驱动的分类核心成分；
  - 决策树可解释规则：熊强等（2025）用决策树特征重要性刻画判别主导成分，
    本文以深度受限决策树输出可读的分类规则（阈值路径），作为可解释判别依据；
  - Logistic 判别函数：钱鸿羽等（2024）构建玻璃体系判别函数，本文以标准化
    成分拟合逻辑回归，输出判别系数方向（正→倾向铅钡、负→倾向高钾）；
  - 模型评估：按第 3 讲课件（泛化与模型选择）的 K 折交叉验证原则，采用
    分层 5 折交叉验证报告模型准确率，标准化参数在训练折内估计，
    避免数据泄漏（课件明确指出"预处理必须在训练数据内部拟合"）。

数据口径：文物级全量表（58 件）中的 56 件有效样本（成分总和 85%~105%），
          类型编码：高钾=0、铅钡=1；未检出成分以 0 填充（填0版）。
输入：cleaned/表单2_填0版_全量58.csv
输出：results/q2_1_相关系数表.csv、results/q2_1_特征重要性表.csv、
      results/q2_1_判别系数表.csv、results/q2_1_决策树规则.txt、
      results/q2_1_结果.txt
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

# ----------------------------------------------------------------------
# 0. 路径与常量
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # 2-解答/2-第二问
REPO_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))           # 仓库根
CLEAN_DIR = os.path.join(REPO_ROOT, "2-解答", "0-数据预处理", "cleaned")
OUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

COMPONENTS = [
    "SiO2", "Na2O", "K2O", "CaO", "MgO", "Al2O3", "Fe2O3",
    "CuO", "PbO", "BaO", "P2O5", "SrO", "SnO2", "SO2",
]
RANDOM_STATE = 42
N_SPLITS = 5

# ----------------------------------------------------------------------
# 1. 读取数据：填0版全量表 → 有效样本 + 类型编码
# ----------------------------------------------------------------------
def load_data():
    df = pd.read_csv(os.path.join(CLEAN_DIR, "表单2_填0版_全量58.csv"), encoding="utf-8")
    df = df[df["是否有效"]].copy()                       # 56 件有效样本
    df["类型编码"] = df["类型"].map({"高钾": 0, "铅钡": 1})
    return df


# ----------------------------------------------------------------------
# 2. 类型-成分 Spearman 相关（CatBoost 文献思路）
# ----------------------------------------------------------------------
def spearman_correlation(df):
    rows = []
    for col in COMPONENTS:
        rho, p = stats.spearmanr(df["类型编码"], df[col])
        rows.append({"成分": col, "Spearman相关系数": round(rho, 4),
                     "P值": round(p, 4)})
    corr = pd.DataFrame(rows).sort_values("Spearman相关系数", key=abs, ascending=False)
    return corr.reset_index(drop=True)


# ----------------------------------------------------------------------
# 3. 随机森林特征重要性（刘振文等，2025）+ 分层 K 折交叉验证
# ----------------------------------------------------------------------
def rf_importance(df):
    X = df[COMPONENTS].values
    y = df["类型编码"].values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    imp_sum = np.zeros(len(COMPONENTS))
    accs = []
    for train_idx, val_idx in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE)
        clf.fit(X[train_idx], y[train_idx])
        imp_sum += clf.feature_importances_
        accs.append(clf.score(X[val_idx], y[val_idx]))

    imp = imp_sum / N_SPLITS
    imp_df = pd.DataFrame({
        "成分": COMPONENTS,
        "特征重要性": np.round(imp, 4),
    }).sort_values("特征重要性", ascending=False).reset_index(drop=True)
    imp_df["累计重要性"] = np.round(imp_df["特征重要性"].cumsum(), 4)
    return imp_df, np.mean(accs), np.std(accs)


# ----------------------------------------------------------------------
# 4. 决策树可解释规则（熊强等，2025）
# ----------------------------------------------------------------------
def decision_tree_rules(df):
    X = df[COMPONENTS].values
    y = df["类型编码"].values
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2,
                                  random_state=RANDOM_STATE)
    tree.fit(X, y)
    rules = export_text(tree, feature_names=COMPONENTS, decimals=2)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accs = []
    for train_idx, val_idx in skf.split(X, y):
        t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2,
                                   random_state=RANDOM_STATE)
        t.fit(X[train_idx], y[train_idx])
        accs.append(t.score(X[val_idx], y[val_idx]))
    return rules, np.mean(accs), np.std(accs)


# ----------------------------------------------------------------------
# 5. Logistic 判别函数（钱鸿羽等，2024）：标准化 + 判别系数
# ----------------------------------------------------------------------
def logistic_discriminant(df):
    X = df[COMPONENTS].values
    y = df["类型编码"].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accs = []
    for train_idx, val_idx in skf.split(X, y):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        accs.append(pipe.score(X[val_idx], y[val_idx]))

    # 全量拟合取判别系数（系数作用于标准化成分）
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    pipe.fit(X, y)
    coef = pipe.named_steps["lr"].coef_[0]
    coef_df = pd.DataFrame({
        "成分": COMPONENTS,
        "判别系数": np.round(coef, 4),
    }).sort_values("判别系数", key=abs, ascending=False).reset_index(drop=True)
    return coef_df, np.mean(accs), np.std(accs)


# ----------------------------------------------------------------------
# 6. 汇总
# ----------------------------------------------------------------------
def main():
    df = load_data()
    n_high_k = int((df["类型"] == "高钾").sum())
    n_pb = int((df["类型"] == "铅钡").sum())

    # 2. Spearman 相关
    corr = spearman_correlation(df)
    corr.to_csv(os.path.join(OUT_DIR, "q2_1_相关系数表.csv"),
                index=False, encoding="utf-8-sig")

    # 3. 随机森林
    imp_df, rf_acc, rf_std = rf_importance(df)
    imp_df.to_csv(os.path.join(OUT_DIR, "q2_1_特征重要性表.csv"),
                  index=False, encoding="utf-8-sig")

    # 4. 决策树规则
    rules_text, dt_acc, dt_std = decision_tree_rules(df)
    with open(os.path.join(OUT_DIR, "q2_1_决策树规则.txt"),
              "w", encoding="utf-8") as f:
        f.write(rules_text)

    # 5. Logistic 判别
    coef_df, lr_acc, lr_std = logistic_discriminant(df)
    coef_df.to_csv(os.path.join(OUT_DIR, "q2_1_判别系数表.csv"),
                   index=False, encoding="utf-8-sig")

    # 6. 汇总文本
    lines = []
    lines.append("=" * 60)
    lines.append("问题 2（1）分类规律挖掘结果")
    lines.append("=" * 60)
    lines.append(f"有效样本：{len(df)} 件（高钾 {n_high_k}、铅钡 {n_pb}），"
                 f"类型编码 高钾=0 / 铅钡=1")
    lines.append("")
    lines.append("【1】类型-成分 Spearman 相关（CatBoost 文献思路，|ρ| 降序）")
    for _, r in corr.head(8).iterrows():
        lines.append(f"  {r['成分']:>7s}  ρ={r['Spearman相关系数']:+.4f}  "
                     f"P={r['P值']:.4f}")
    lines.append("")
    lines.append("【2】随机森林特征重要性（刘振文等，2025 思路）")
    lines.append(f"  分层 {N_SPLITS} 折交叉验证准确率：{rf_acc:.4f} ± {rf_std:.4f}")
    for i, r in imp_df.iterrows():
        mark = "  <-- 分类核心成分" if i < 7 else ""
        lines.append(f"  {i+1:2d}. {r['成分']:>7s}  "
                     f"重要性={r['特征重要性']:.4f}  累计={r['累计重要性']:.4f}{mark}")
    lines.append("")
    lines.append("【3】决策树可解释规则（熊强等，2025 思路，深度 3）")
    lines.append(f"  分层 {N_SPLITS} 折交叉验证准确率：{dt_acc:.4f} ± {dt_std:.4f}")
    lines.append("  （完整规则见 q2_1_决策树规则.txt）")
    lines.append("")
    lines.append("【4】Logistic 判别函数（钱鸿羽等，2024 思路，标准化成分）")
    lines.append(f"  分层 {N_SPLITS} 折交叉验证准确率：{lr_acc:.4f} ± {lr_std:.4f}")
    lines.append("  系数 > 0 促进类型编码=1（铅钡），< 0 促进类型编码=0（高钾）")
    for _, r in coef_df.head(8).iterrows():
        lines.append(f"  {r['成分']:>7s}  系数={r['判别系数']:+.4f}")
    lines.append("")
    lines.append(f"分类核心成分（RF 前 7）：{', '.join(imp_df['成分'].head(7).tolist())}")
    lines.append(f"判别主导方向（Logistic 前 3）："
                 f"{', '.join(coef_df['成分'].head(3).tolist())}")
    lines.append("=" * 60)

    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "q2_1_结果.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
