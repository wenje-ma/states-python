# -*- coding: utf-8 -*-
"""
问题 3：未知类别玻璃文物的鉴别与敏感性分析
============================================
方法（文献依据）：
  - 判别底座：复用问题 2（q2_1）三模型——Logistic 判别函数（钱鸿羽等，2024）、
    随机森林特征重要性筛选的分类核心成分（刘振文等，2025）、决策树可解释规则
    （熊强等，2025）。模型在已知样本（表单2 填0版全量58，56 件有效）上全量拟合，
    并以分层 5 折交叉验证报告准确率；未知样本成分同样填 0，标准化参数仅由
    已知样本估计，避免数据泄漏（第 3 讲课件：预处理必须在训练数据内部拟合）。
  - 未风化样本：风化前成分≈观测成分，直接代入判别模型。
  - 风化样本：采用钱鸿羽等（2024）的"假设-验证"判别流程——分别假设其为
    高钾 / 铅钡，用对应玻璃类型的风化程度函数与反应比例（q1_3 主线 A 参数，
    基于采样点级数据重算）还原风化前成分，再将还原成分代入判别模型；
    若还原后判别结果与假设一致，则该假设成立。关键成分缺失时对应假设不可行。
  - 敏感性分析：熊强等（2025）对化合物含量正向调整 5% 及 10% 检验判别稳健性，
    本文对判别输入施加 ±5%、±10% 乘性扰动重复 100 次统计类别翻转率；
    同时报告三模型投票一致性与风化程度 Y 扫描下的类别稳定性。

数据口径：
  判别训练：2-解答/0-数据预处理/cleaned/表单2_填0版_全量58.csv（文物级，56 件）
  风化还原：2-解答/0-数据预处理/cleaned/表单2_成分_clean.csv（采样点级，67 个有效点）
  未知样本：2-解答/0-数据预处理/cleaned/表单3_未知_clean.csv（8 件，A1~A8）

输出：results/q3_未知样品判别结果.csv、q3_风化还原成分.csv、
      q3_敏感性.csv、q3_结果.txt
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

# ----------------------------------------------------------------------
# 0. 路径与常量
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # 2-解答/3-第三问
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

# 每类玻璃风化程度函数的成分（与 q1_3 主线 A 一致）
MODEL_COMPS = {
    "高钾": ["SiO2", "K2O", "CaO", "Al2O3"],
    "铅钡": ["SiO2", "CaO", "Al2O3", "PbO", "P2O5"],
}
# 一般风化赋值（未知样本仅"风化/无风化"二值，风化按一般风化处理）
WEATHER_Y = {"高钾": 0.8, "铅钡": 0.75}

TYPE_MAP = {"高钾": 0, "铅钡": 1}
TYPE_INV = {0: "高钾", 1: "铅钡"}


# ----------------------------------------------------------------------
# 1. 数据读取
# ----------------------------------------------------------------------
def load_known():
    """判别训练集：文物级填0全量表 → 56 件有效样本 + 类型编码。"""
    df = pd.read_csv(os.path.join(CLEAN_DIR, "表单2_填0版_全量58.csv"),
                     encoding="utf-8-sig")
    df = df[df["是否有效"]].copy()
    df["类型编码"] = df["类型"].map(TYPE_MAP)
    return df


def load_unknown():
    """未知样本：表单3 → 14 成分列填 0（未检出视作低于检出限）。"""
    df = pd.read_csv(os.path.join(CLEAN_DIR, "表单3_未知_clean.csv"),
                     encoding="utf-8-sig")
    df = df.copy()
    for c in COMPONENTS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def load_sample_level():
    """风化还原训练数据：采样点级（67 个有效采样点），与 q1_3 同口径。"""
    df = pd.read_csv(os.path.join(CLEAN_DIR, "表单2_成分_clean.csv"),
                     encoding="utf-8-sig")
    df = df[df["是否有效"]].copy()

    def assign_y(row):
        t = row["类型"]
        if row["采样点类型"] == "未风化点":
            return 0.0
        if row["采样点类型"] == "严重风化点":
            return 1.0
        if row["表面风化"] == "无风化":
            return 0.0
        return WEATHER_Y[t]

    df["Y"] = df.apply(assign_y, axis=1)
    return df


# ----------------------------------------------------------------------
# 2. 风化还原模型（q1_3 主线 A 参数重算，按玻璃类型）
# ----------------------------------------------------------------------
def fit_preweather_models(df):
    """
    对每类玻璃拟合风化程度函数（Y ~ 关键成分，OLS）并估计反应比例
    r_j = ΔC_j / ΔC_SiO2（风化组均值 − 无风化组均值）。
    返回 {"高钾": {...}, "铅钡": {...}}。
    """
    import statsmodels.api as sm
    models = {}
    for t in ["高钾", "铅钡"]:
        sub = df[df["类型"] == t].dropna(subset=MODEL_COMPS[t]).copy()
        X = sm.add_constant(sub[MODEL_COMPS[t]])
        m = sm.OLS(sub["Y"], X).fit()
        b = m.params

        w = sub[sub["Y"] > 0][COMPONENTS]
        u = sub[sub["Y"] == 0][COMPONENTS]
        d_sio2 = w["SiO2"].mean() - u["SiO2"].mean()
        ratio = (w.mean() - u.mean()) / d_sio2

        denom = b["SiO2"]
        for c in MODEL_COMPS[t]:
            if c != "SiO2":
                denom += b[c] * ratio[c]

        models[t] = {"coef": b, "ratio": ratio, "denom": denom,
                     "n": len(sub), "R2": m.rsquared}
    return models


def reconstruct(obs_row, hyp_type, models):
    """
    在假设类型 hyp_type 下还原风化前成分。
    obs_row：含 14 成分的观测行；Y = WEATHER_Y[hyp_type]（一般风化）。
    返回 14 成分还原向量（numpy）；若该假设关键成分在观测中缺失，返回 None。
    """
    comps = MODEL_COMPS[hyp_type]
    if any(pd.isna(obs_row[c]) or obs_row[c] == 0 for c in comps):
        return None                      # 关键成分缺失（含未检出），假设不可行
    m = models[hyp_type]
    dY = WEATHER_Y[hyp_type]
    d_sio2 = dY / m["denom"]
    pre = np.empty(len(COMPONENTS))
    for i, c in enumerate(COMPONENTS):
        v = obs_row[c]
        r_j = m["ratio"][c]
        if pd.isna(r_j):
            pre[i] = 0.0 if pd.isna(v) else v   # 该成分无反应比例可依，保持观测
        elif pd.isna(v):
            pre[i] = 0.0                        # 未检出成分按检出限处理（填0口径）
        else:
            pre[i] = v - r_j * d_sio2
    # 物理有效性校验：还原成分须落在可行区间且总和不超过成分数据上限，
    # 否则还原失真（如反推 SiO2 超过 100%），该假设视为不可行
    if np.any(pre < -0.5) or np.any(pre > 100.5) or pre.sum() > 105.0:
        return None
    return pre


# ----------------------------------------------------------------------
# 3. 判别模型：全量拟合 + 分层 5 折 CV
# ----------------------------------------------------------------------
def fit_discriminators(known):
    X = known[COMPONENTS].values
    y = known["类型编码"].values

    # 分层 5 折交叉验证准确率
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    lr_acc, rf_acc, dt_acc = [], [], []
    for tr, va in skf.split(X, y):
        pipe = Pipeline([("scale", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=2000,
                                                   random_state=RANDOM_STATE))])
        pipe.fit(X[tr], y[tr])
        lr_acc.append(pipe.score(X[va], y[va]))

        rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE)
        rf.fit(X[tr], y[tr])
        rf_acc.append(rf.score(X[va], y[va]))

        dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2,
                                    random_state=RANDOM_STATE)
        dt.fit(X[tr], y[tr])
        dt_acc.append(dt.score(X[va], y[va]))

    # 全量拟合（用于未知样本判别）
    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE)
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2,
                                random_state=RANDOM_STATE)
    lr.fit(scaler.fit_transform(X), y)
    rf.fit(X, y)
    dt.fit(X, y)

    return ({"scaler": scaler, "lr": lr, "rf": rf, "dt": dt},
            {"lr": np.mean(lr_acc), "rf": np.mean(rf_acc), "dt": np.mean(dt_acc)})


# ----------------------------------------------------------------------
# 4. 判别器：单样本三模型判别（概率口径：P(铅钡)）
# ----------------------------------------------------------------------
def predict_prob(vec, disc):
    """vec：14 成分向量。返回 dict(lr/rf/dt 的 P(铅钡)) 与三模型投票类别。"""
    X = np.array(vec, dtype=float).reshape(1, -1)
    p_lr = disc["lr"].predict_proba(disc["scaler"].transform(X))[0, 1]
    p_rf = disc["rf"].predict_proba(X)[0, 1]
    p_dt = disc["dt"].predict_proba(X)[0, 1]
    votes = [TYPE_INV[int(round(p))] for p in (p_lr, p_rf, p_dt)]
    return {"p_lr": float(p_lr), "p_rf": float(p_rf), "p_dt": float(p_dt),
            "votes": votes}


def majority(votes):
    return "铅钡" if votes.count("铅钡") >= 2 else "高钾"


# ----------------------------------------------------------------------
# 5. 敏感性：±5% / ±10% 扰动下类别翻转率
# ----------------------------------------------------------------------
def vec_pb_prob(vec, disc):
    X = np.array(vec, dtype=float).reshape(1, -1)
    return disc["lr"].predict_proba(disc["scaler"].transform(X))[0, 1]


def perturbation_sensitivity(vec, disc, n_rep=100, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    vec = np.asarray(vec, dtype=float)
    base = vec_pb_prob(vec, disc)
    out = {}
    for amp in (0.05, 0.10):
        flips = 0
        for _ in range(n_rep):
            vec_p = vec * rng.normal(1.0, amp, size=len(vec))
            p = vec_pb_prob(vec_p, disc)
            if (p > 0.5) != (base > 0.5):
                flips += 1
        out[amp] = flips / n_rep
    return out


# ----------------------------------------------------------------------
# 6. 主流程
# ----------------------------------------------------------------------
def main():
    known = load_known()
    unknown = load_unknown()
    sample_df = load_sample_level()

    disc, cv = fit_discriminators(known)
    pw_models = fit_preweather_models(sample_df)

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 72)
    log("问题 3：未知类别玻璃文物鉴别（A1~A8）")
    log("判别底座 CV 准确率（分层 5 折）：Logistic {:.4f} / RF {:.4f} / 决策树 {:.4f}"
        .format(cv["lr"], cv["rf"], cv["dt"]))
    log("风化还原模型（采样点级重算）：")
    for t in ["高钾", "铅钡"]:
        m = pw_models[t]
        log("  {}：R²={:.4f}，反推分母={:.4f}（n={}）".format(t, m["R2"], m["denom"], m["n"]))
    log("=" * 72)

    # 逐件判别
    rows, recon_rows, sens_rows = [], [], []
    for _, r in unknown.iterrows():
        wid = r["文物编号"]
        weathered = (r["表面风化"] == "风化")
        obs = np.array([r[c] for c in COMPONENTS], dtype=float)

        # (a) 直接判别（观测成分）
        p_direct = predict_prob(obs, disc)
        cls_direct = majority(p_direct["votes"])

        # (b) 风化样本：双假设还原判别
        cand = []                     # (类别, 概率P(铅钡), 依据标签)
        cand.append((cls_direct, p_direct["p_lr"], "直接判别"))
        recons = {}
        if weathered:
            for hyp in ["高钾", "铅钡"]:
                pre = reconstruct(r, hyp, pw_models)
                if pre is None:
                    recons[hyp] = None
                    continue
                recons[hyp] = pre
                pp = predict_prob(pre, disc)
                hyp_cls = majority(pp["votes"])
                tag = "{0}假设还原(判为{1})".format(hyp, hyp_cls)
                cand.append((hyp_cls, pp["p_lr"], tag))
                recon_rows.append({
                    "文物编号": wid, "假设类型": hyp,
                    "表面风化": r["表面风化"],
                    "观测SiO2": round(obs[0], 3),
                    "还原前SiO2": round(pre[0], 3),
                    "还原前K2O": round(pre[2], 3),
                    "还原前PbO": round(pre[8], 3),
                    "还原前BaO": round(pre[9], 3),
                    "还原判别P(铅钡)": round(pp["p_lr"], 4),
                })

        # 自洽筛选：候选类别与概率方向一致的保留；无一致时全保留
        consist = [c for c in cand if (c[0] == "铅钡") == (c[1] > 0.5)]
        pool = consist if consist else cand
        final_cls, final_p, basis = max(
            pool, key=lambda c: abs(c[1] - 0.5))

        rows.append({
            "文物编号": wid, "表面风化": r["表面风化"],
            "检出成分数": int(r["检出成分数"]),
            "直接判别": cls_direct,
            "直接判别P(铅钡)": round(p_direct["p_lr"], 4),
            "高钾假设还原P(铅钡)":
                (round(predict_prob(recons["高钾"], disc)["p_lr"], 4)
                 if weathered and recons.get("高钾") is not None else ""),
            "铅钡假设还原P(铅钡)":
                (round(predict_prob(recons["铅钡"], disc)["p_lr"], 4)
                 if weathered and recons.get("铅钡") is not None else ""),
            "最终类型": final_cls,
            "最终置信度": round(abs(final_p - 0.5) * 2, 4),
            "判别依据": basis,
        })

        # (c) 敏感性：直接判别输入的 ±5% / ±10% 扰动
        flip = perturbation_sensitivity(obs, disc)
        sens_rows.append({
            "文物编号": wid, "表面风化": r["表面风化"],
            "直接判别": cls_direct,
            "±5%翻转率": round(flip[0.05], 4),
            "±10%翻转率": round(flip[0.10], 4),
        })

        log("\n[{}] 表面{}，检出 {} 成分".format(wid, r["表面风化"], int(r["检出成分数"])))
        log("  直接判别：{}（P(铅钡)={:.4f}，三模型投票 {}）".format(
            cls_direct, p_direct["p_lr"], "/".join(p_direct["votes"])))
        if weathered:
            for hyp in ["高钾", "铅钡"]:
                if recons.get(hyp) is None:
                    log("  {}假设还原：关键成分缺失或还原结果超出物理范围，"
                        "假设不可行".format(hyp))
                else:
                    pp = predict_prob(recons[hyp], disc)
                    log("  {}假设还原 → 判为{}（P(铅钡)={:.4f}，投票 {}）".format(
                        hyp, majority(pp["votes"]), pp["p_lr"],
                        "/".join(pp["votes"])))
        log("  → 最终类型：{}（依据：{}，置信度 {:.2f}）".format(
            final_cls, basis, abs(final_p - 0.5) * 2))

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUT_DIR, "q3_未知样品判别结果.csv"),
                  index=False, encoding="utf-8-sig")
    pd.DataFrame(recon_rows).to_csv(
        os.path.join(OUT_DIR, "q3_风化还原成分.csv"),
        index=False, encoding="utf-8-sig")
    pd.DataFrame(sens_rows).to_csv(os.path.join(OUT_DIR, "q3_敏感性.csv"),
                                   index=False, encoding="utf-8-sig")

    log("\n" + "=" * 72)
    log("汇总（最终判别结果）：")
    log(res_df[["文物编号", "表面风化", "最终类型", "最终置信度",
                "判别依据"]].to_string(index=False))
    log("\n敏感性（±5% / ±10% 扰动翻转率，乘性高斯噪声 100 次）：")
    log(pd.DataFrame(sens_rows).to_string(index=False))

    with open(os.path.join(OUT_DIR, "q3_结果.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n输出目录: {}".format(OUT_DIR))


if __name__ == "__main__":
    main()
