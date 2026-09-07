# -*- coding: utf-8 -*-
"""
问题 1（3）：根据风化点检测数据，预测其风化前的化学成分含量
================================================================
方法（双模型对照）：
  主线 A（钱鸿羽等，2024）——风化程度函数 + 反应比例反推：
    1) 按玻璃类型分别建立"风化程度 Y ~ 关键成分"的多元线性回归
       （高钾：SiO2/K2O/CaO/Al2O3，Y=未风化0、一般0.8；
        铅钡：SiO2/CaO/Al2O3/PbO/P2O5，Y=未风化0、一般0.75、严重1）；
    2) 由类型内"风化组均值 − 无风化组均值"估计风化反应比例
       r_j = ΔC_j / ΔC_SiO2（各成分相对 SiO2 的变化比例）；
    3) 对风化样本，由回归系数与反应比例解出风化引起的 SiO2 变化量
       ΔC_SiO2 = ΔY / (b_SiO2 + Σ b_j·r_j)，
       其余成分 C_j* = C_j − r_j·ΔC_SiO2，即得风化前含量。
  对照 B（熊强等，2025）——多元线性回归 + 风化虚拟变量：
    风化物质含量 = α + Σβ·一般物质 + λ·WG（WG：无风化0/风化1），
    代入 WG=0 得风化前预测值。样本量小，自变量控制在 4 个以内。

数据口径：采样点级（表单 2，67 个有效采样点），与题目"风化点检测数据"一致；
          未风化点视为风化前近似（Y=0）。

输入：cleaned/表单2_成分_clean.csv
输出：results/q1_3_风化前预测_主线A.csv、results/q1_3_风化前预测_对照B.csv、
      results/q1_3_结果.txt
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ----------------------------------------------------------------------
# 0. 路径与常量
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # 2-解答/1-第一问
REPO_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))           # 仓库根
CLEAN_DIR = os.path.join(REPO_ROOT, "2-解答", "0-数据预处理", "cleaned")
OUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

COMPONENTS = [
    "SiO2", "Na2O", "K2O", "CaO", "MgO", "Al2O3", "Fe2O3",
    "CuO", "PbO", "BaO", "P2O5", "SrO", "SnO2", "SO2",
]

# 每类玻璃的风化程度函数成分（钱鸿羽等，2024）
MODEL_COMPS = {
    "高钾": ["SiO2", "K2O", "CaO", "Al2O3"],
    "铅钡": ["SiO2", "CaO", "Al2O3", "PbO", "P2O5"],
}
# 风化程度赋值：无风化=0；一般风化（高钾0.8、铅钡0.75）；严重风化=1
WEATHER_Y = {"高钾": {"一般": 0.8, "严重": 1.0},
             "铅钡": {"一般": 0.75, "严重": 1.0}}
# 对照 B 的风化物质（熊强等，2025 筛选出的风化显著相关成分）
CONTROL_TARGETS = ["K2O", "SrO", "P2O5"]
# 对照 B 的自变量（低缺失、代表性成分 + 风化虚拟变量；小样本下控制维数）
CONTROL_FEATURES = {
    "高钾": ["SiO2", "CaO"],                    # 高钾风化样本仅 6 条，特征必须最少
    "铅钡": ["SiO2", "CaO", "Al2O3", "PbO"],
}


# ----------------------------------------------------------------------
# 1. 数据准备：采样点级 + 风化程度赋值
# ----------------------------------------------------------------------
def build_samples():
    """读表单2清洗表，为每个采样点赋风化程度 Y 与组别。返回 (表, 组别列)。"""
    df = pd.read_csv(os.path.join(CLEAN_DIR, "表单2_成分_clean.csv"),
                     encoding="utf-8-sig")
    df = df.copy()
    df = df[df["是否有效"]].copy()          # 仅保留有效采样点（剔除15/17）

    def assign_y(row):
        t = row["类型"]
        if row["采样点类型"] == "未风化点":
            return 0.0
        if row["采样点类型"] == "严重风化点":
            return WEATHER_Y[t]["严重"]
        # 普通点/部位点：按文物表面风化属性
        if row["表面风化"] == "无风化":
            return 0.0
        return WEATHER_Y[t]["一般"]

    df["Y"] = df.apply(assign_y, axis=1)
    df["组别"] = df["Y"].map({0.0: "未风化", 0.75: "一般风化",
                              0.8: "一般风化", 1.0: "严重风化"})
    return df


# ----------------------------------------------------------------------
# 2. 主线 A：风化程度函数 + 反应比例反推
# ----------------------------------------------------------------------
def method_A(df, comps, y_weather):
    """
    对单个玻璃类型执行主线 A：
      1) OLS：Y ~ 关键成分；
      2) 反应比例 r_j = ΔC_j / ΔC_SiO2（风化组均值 − 无风化组均值）；
      3) 对一般/严重风化样本反推风化前成分。
    返回 (预测表, 回归汇总)。
    """
    sub = df.dropna(subset=comps).copy()          # 关键成分完整的样本
    X = sm.add_constant(sub[comps])
    model = sm.OLS(sub["Y"], X).fit()
    b = model.params

    # 反应比例：均值差之比（SiO2 为基准，自身 r=1）
    w = sub[sub["Y"] > 0][COMPONENTS]
    u = sub[sub["Y"] == 0][COMPONENTS]
    d_sio2 = w["SiO2"].mean() - u["SiO2"].mean()
    ratio = (w.mean() - u.mean()) / d_sio2        # 14 成分的反应比例向量

    # 反推公式分母：b_SiO2 + Σ_{j≠SiO2} b_j·r_j
    denom = b["SiO2"]
    for c in comps:
        if c != "SiO2":
            denom += b[c] * ratio[c]

    # 对风化样本反推
    pred_rows = []
    for _, row in sub[sub["Y"] > 0].iterrows():
        dY = row["Y"]                             # 该样本的风化程度
        d_sio2_est = dY / denom                   # 风化引起的 SiO2 变化量
        pre = {}
        for c in COMPONENTS:
            pre[c] = row[c] - ratio[c] * d_sio2_est
        pred_rows.append({
            "文物采样点": row["文物采样点"], "文物编号": row["文物编号"],
            "类型": row["类型"], "组别": row["组别"], "Y": row["Y"],
            "观测SiO2": row["SiO2"], "预测风化前SiO2": pre["SiO2"],
            "反推分母": denom, **pre,
        })
    pred = pd.DataFrame(pred_rows)

    summary = {
        "R2": model.rsquared, "adjR2": model.rsquared_adj,
        "系数": {c: round(b[c], 4) for c in ["const"] + comps},
        "反应比例": {c: round(ratio[c], 4) for c in COMPONENTS},
        "反推分母": denom,
    }
    return pred, summary


# ----------------------------------------------------------------------
# 3. 对照 B：多元线性回归 + 风化虚拟变量
# ----------------------------------------------------------------------
def method_B(df, targets, features):
    """
    对照 B（熊强等，2025）：对每个风化物质分别回归
    target ~ features + WG（风化虚拟变量），代入 WG=0 预测风化前。
    返回 (预测表, 回归汇总)。
    """
    sub = df.dropna(subset=features + ["SiO2"]).copy()   # 特征完整样本
    sub["WG"] = (sub["Y"] > 0).astype(float)             # 风化虚拟变量 0/1
    X_all = sm.add_constant(sub[features + ["WG"]])

    pred_rows = []
    summaries = {}
    for tgt in targets:
        valid = sub.dropna(subset=[tgt])                 # 因变量非缺失
        if len(valid) < len(features) + 3:               # 样本不足
            summaries[tgt] = "样本不足，跳过"
            continue
        X = sm.add_constant(valid[features + ["WG"]])
        m = sm.OLS(valid[tgt], X).fit()
        summaries[tgt] = {"R2": round(m.rsquared, 4),
                          "系数": {k: round(v, 4) for k, v in m.params.items()}}
        # 对风化样本：WG 置 0 得风化前预测（显式构造设计矩阵 + 矩阵乘，
        # 避免 DataFrame 列名对齐在极小样本下的不确定性）
        w_rows = valid[valid["Y"] > 0]
        Xw = w_rows[features + ["WG"]].copy()
        Xw.insert(0, "const", 1.0)
        Xw["WG"] = 0.0
        yhat = Xw.values @ m.params.values
        for (_, row), y in zip(w_rows.iterrows(), yhat):
            pred_rows.append({
                "文物采样点": row["文物采样点"], "文物编号": row["文物编号"],
                "类型": row["类型"], "组别": row["组别"],
                "风化物质": tgt,
                "观测含量": row[tgt], "预测风化前含量": float(y),
            })
    pred = pd.DataFrame(pred_rows)
    return pred, summaries


# ----------------------------------------------------------------------
# 4. 主流程
# ----------------------------------------------------------------------
def main():
    df = build_samples()
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("问题1(3)：由风化点数据预测风化前化学成分")
    log("样本（采样点级）：共 {} 个有效采样点".format(len(df)))
    log("  未风化(Y=0)：{} | 一般风化：{} | 严重风化(Y=1)：{}".format(
        (df["Y"] == 0).sum(),
        df["Y"].isin([0.75, 0.8]).sum(),
        (df["Y"] == 1).sum()))
    log("=" * 70)

    predA_parts, predB_parts = [], []
    for t in ["高钾", "铅钡"]:
        sub = df[df["类型"] == t]
        log("\n[{}玻璃] 未风化 {} / 一般风化 {} / 严重风化 {}".format(
            t, (sub["Y"] == 0).sum(), sub["Y"].isin([0.75, 0.8]).sum(),
            (sub["Y"] == 1).sum()))

        # ---- 主线 A ----
        predA, sumA = method_A(sub, MODEL_COMPS[t], WEATHER_Y[t])
        predA_parts.append(predA)
        log("\n主线A（风化程度函数+反应比例反推）：R²={:.4f}，adjR²={:.4f}".format(
            sumA["R2"], sumA["adjR2"]))
        log("  风化程度函数系数：{}".format(sumA["系数"]))
        log("  反应比例 r_j（相对SiO2）：{}".format(sumA["反应比例"]))
        log("  预测 {} 个风化样本风化前含量（节选，完整见csv）：".format(len(predA)))
        if len(predA):
            log(predA[["文物采样点", "组别", "观测SiO2", "预测风化前SiO2",
                       "K2O", "PbO"]].head(6).to_string(index=False))

            # ---- 预测质量验证：预测风化前值应落在无风化组分布附近 ----
            u_mean = sub[sub["Y"] == 0][COMPONENTS].mean()
            u_std = sub[sub["Y"] == 0][COMPONENTS].std()
            p_mean = predA[COMPONENTS].mean()
            check = pd.DataFrame({
                "成分": COMPONENTS,
                "预测风化前均值": p_mean.round(3).values,
                "无风化组均值": u_mean.round(3).values,
                "无风化组标准差": u_std.round(3).values,
            })
            check = check[check["预测风化前均值"].notna() &
                          check["无风化组均值"].notna()]
            check["|预测−无风化|/无风化标准差"] = (
                (check["预测风化前均值"] - check["无风化组均值"]).abs()
                / check["无风化组标准差"]).round(3)
            log("  预测质量验证（预测均值 vs 无风化组分布，标准化偏差<1 即合理）：")
            log(check.to_string(index=False))

        # ---- 对照 B ----
        predB, sumB = method_B(sub, CONTROL_TARGETS, CONTROL_FEATURES[t])
        predB_parts.append(predB)
        log("\n对照B（多元回归+风化虚拟变量）：")
        for k, v in sumB.items():
            log("  {}: {}".format(k, v))
        log("  预测 {} 条（风化物质×样本，完整见csv）".format(len(predB)))

    pd.concat(predA_parts, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "q1_3_风化前预测_主线A.csv"),
        index=False, encoding="utf-8-sig")
    pd.concat(predB_parts, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "q1_3_风化前预测_对照B.csv"),
        index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT_DIR, "q1_3_结果.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n输出目录: {}".format(OUT_DIR))


if __name__ == "__main__":
    main()
