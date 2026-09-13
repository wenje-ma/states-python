# -*- coding: utf-8 -*-
"""
多变量 SVR 插补 —— 对中等缺失成分（缺失率 20%~50%）
=====================================================
方法依据：侯兴汉等《基于SVR插补高缺失率的地下水水质监测数据插补研究》
  - 其核心主张：小样本、高缺失率场景下，SVR 在模型复杂度与学习能力之间
    取得平衡，插补精度优于线性插值；
  - 但其机制是"时间序列滑动窗口"（用前后观测预测当前值），
    本题为截面数据（文物间无先后关系），故本脚本将其改造为
    **多变量插补**：以低缺失成分作特征，预测中缺失成分的缺失值。

处理范围（默认，可在运行时覆盖）：
  - 插补对象：缺失率 20%~50% 的中缺失成分；
  - 特征：缺失率 < 20% 的低缺失成分；
  - 高缺失成分（>=50%）不做数值插补（大概率低于检出限，插补即编造）。

评估：每个成分 5 折交叉验证（MAE / RMSE / R2），并对比插补前后分布。

运行：python svr_impute.py
输出：cleaned/表单2_文物级_SVR插补.csv、cleaned/SVR插补报告.txt
"""

import os
import pandas as pd
import numpy as np

# 仅在运行时导入 sklearn，便于明确报错信息
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------
# 0. 路径与常量
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # 脚本所在目录
OUT_DIR = os.path.join(BASE_DIR, "cleaned")                       # 输出目录

COMPONENTS = [
    "SiO2", "Na2O", "K2O", "CaO", "MgO", "Al2O3", "Fe2O3",
    "CuO", "PbO", "BaO", "P2O5", "SrO", "SnO2", "SO2",
]

HIGH_RATE, MID_RATE = 50.0, 20.0       # 与 preprocess2.py 一致的分级阈值

# SVR 超参数网格（分阶段网格搜索的粗网格，参照侯兴汉的做法）
PARAM_GRID = {
    "C": [0.1, 1, 10, 50],
    "gamma": [0.01, 0.05, 0.1, 0.5],
    "epsilon": [0.01, 0.05, 0.1],
}


# ----------------------------------------------------------------------
# 1. 数据准备
# ----------------------------------------------------------------------
def prepare():
    """读取文物级主表，只保留有效文物；返回 (有效表, 各成分缺失率)。"""
    art = pd.read_csv(os.path.join(OUT_DIR, "表单2_文物级_clean.csv"),
                      encoding="utf-8-sig")
    valid = art[art["是否有效"]].copy()               # 85%~105% 有效样本
    miss_rate = valid[COMPONENTS].isna().mean() * 100
    return valid, miss_rate


def choose_features_and_targets(miss_rate):
    """
    按缺失率选择特征与插补目标：
      - 特征（X）：低缺失成分（缺失率 < MID_RATE），数据质量高；
      - 目标（Y）：中缺失成分（MID_RATE <= 缺失率 < HIGH_RATE）；
      - 高缺失成分（>= HIGH_RATE）不插补，仅保留原值。
    """
    features = [c for c in COMPONENTS if miss_rate[c] < MID_RATE]
    targets = [c for c in COMPONENTS if MID_RATE <= miss_rate[c] < HIGH_RATE]
    return features, targets


# ----------------------------------------------------------------------
# 2. 单成分 SVR 插补
# ----------------------------------------------------------------------
def impute_one(col, valid, features):
    """
    对单个目标成分执行 SVR 插补：
      1) 训练集 = 该成分已检出的有效文物行；
      2) 特征用低缺失成分（缺失处用训练集均值填充）；
      3) 网格搜索 + 5 折交叉验证选最优超参数；
      4) 用最优模型预测该成分缺失的行；
    返回 (插补后的列, 评估字典)。
    """
    # 特征矩阵：训练与预测共用的行内特征
    X_all = valid[features].copy()
    # 特征缺失用该特征在全量有效文物上的均值填充（特征本身缺失率低，影响小）
    for f in features:
        X_all[f] = X_all[f].fillna(X_all[f].mean())

    # 训练集：目标成分已检出
    known = valid[col].notna()
    X_tr = X_all[known].values
    y_tr = valid.loc[known, col].values
    X_te = X_all[~known].values                       # 待插补行

    # 标准化（侯兴汉：Z-score），保证 SVR 对量纲不敏感
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    # 分阶段网格搜索 + 5 折交叉验证（小样本，网格较小，耗时可控）
    base = SVR(kernel="rbf")
    gs = GridSearchCV(base, PARAM_GRID, cv=5,
                      scoring="neg_mean_absolute_error")
    gs.fit(X_tr_s, y_tr)
    best = gs.best_estimator_

    # 交叉验证误差（用最优参数重估，5 折）
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []
    for tr_idx, va_idx in kf.split(X_tr_s):
        m = SVR(kernel="rbf", **gs.best_params_)
        m.fit(X_tr_s[tr_idx], y_tr[tr_idx])
        y_pred_all.append(m.predict(X_tr_s[va_idx]))
        y_true_all.append(y_tr[va_idx])
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    eval_res = {
        "MAE": mean_absolute_error(y_true_all, y_pred_all),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_all, y_pred_all))),
        "R2": r2_score(y_true_all, y_pred_all),
        "best_params": gs.best_params_,
        "n_known": int(known.sum()),
        "n_missing": int((~known).sum()),
    }

    # 预测缺失值
    imputed = valid[col].copy()
    if (~known).sum() > 0:
        X_te_s = scaler.transform(X_te)
        imputed.loc[~known] = best.predict(X_te_s)

    return imputed, eval_res


# ----------------------------------------------------------------------
# 3. 主流程
# ----------------------------------------------------------------------
def main():
    valid, miss_rate = prepare()
    features, targets = choose_features_and_targets(miss_rate)

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 64)
    log("SVR 多变量插补（侯兴汉方法 · 截面化改造）")
    log("=" * 64)
    log("有效文物样本: {} 件".format(len(valid)))
    log("特征(低缺失): {}".format(features))
    log("插补目标(中缺失): {}".format(targets))

    # 逐成分插补
    out = valid.copy()
    report_rows = []
    for col in targets:
        imputed_col, ev = impute_one(col, valid, features)
        out[col + "_SVR插补"] = imputed_col
        report_rows.append({
            "成分": col, "缺失率%": round(miss_rate[col], 1),
            "已检出数": ev["n_known"], "插补数": ev["n_missing"],
            "CV_MAE": round(ev["MAE"], 3), "CV_RMSE": round(ev["RMSE"], 3),
            "CV_R2": round(ev["R2"], 3),
            "最优参数": str(ev["best_params"]),
        })
        log("\n[{}] 缺失率 {:.1f}% | 已检出 {} / 插补 {} | 5折CV MAE={:.3f} RMSE={:.3f} R2={:.3f}".format(
            col, miss_rate[col], ev["n_known"], ev["n_missing"],
            ev["MAE"], ev["RMSE"], ev["R2"]))
        log("    最优超参数: {}".format(ev["best_params"]))

    # 分布对比（插补前后均值/标准差，敏感性检查）
    log("\n[分布对比] 原检出值 vs 插补后全量（均值/标准差）:")
    for col in targets:
        orig = valid[col]
        full = out[col + "_SVR插补"]
        log("    {:<7} 原: {:.2f}±{:.2f}  →  插补后: {:.2f}±{:.2f}".format(
            col, orig.mean(), orig.std(), full.mean(), full.std()))

    # 输出：插补后的文物级表 + 评估报告
    out.to_csv(os.path.join(OUT_DIR, "表单2_文物级_SVR插补.csv"),
               index=False, encoding="utf-8-sig")
    pd.DataFrame(report_rows).to_csv(
        os.path.join(OUT_DIR, "SVR插补评估表.csv"),
        index=False, encoding="utf-8-sig")

    with open(os.path.join(OUT_DIR, "SVR插补报告.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n输出: 表单2_文物级_SVR插补.csv / SVR插补评估表.csv / SVR插补报告.txt")


if __name__ == "__main__":
    main()
