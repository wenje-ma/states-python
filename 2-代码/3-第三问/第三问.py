# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
CLEAN_DIR = os.path.join(REPO_ROOT, "2-解答", "0-数据预处理", "cleaned")
FIG_DIR = os.path.join(REPO_ROOT, "3-论文", "figures")
OUT_XLSX = os.path.join(BASE_DIR, "第三问.xlsx")
os.makedirs(FIG_DIR, exist_ok=True)

COMPONENTS = [
    "SiO2", "Na2O", "K2O", "CaO", "MgO", "Al2O3", "Fe2O3",
    "CuO", "PbO", "BaO", "P2O5", "SrO", "SnO2", "SO2",
]
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEAT = 100

MODEL_COMPS = {
    "高钾": ["SiO2", "K2O", "CaO", "Al2O3"],
    "铅钡": ["SiO2", "CaO", "Al2O3", "PbO", "P2O5"],
}
WEATHER_Y = {"高钾": 0.8, "铅钡": 0.75}
TYPE_MAP = {"高钾": 0, "铅钡": 1}
TYPE_INV = {0: "高钾", 1: "铅钡"}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

_t0 = time.time()
def stage(no, total, msg):
    print("[{}/{}] {}  （累计耗时 {:.1f}s）".format(no, total, msg, time.time() - _t0))


def load_known():
    df = pd.read_csv(os.path.join(CLEAN_DIR, "表单2_填0版_全量58.csv"),
                     encoding="utf-8-sig")
    df = df[df["是否有效"]].copy()
    df["类型编码"] = df["类型"].map(TYPE_MAP)
    return df


def load_unknown():
    df = pd.read_csv(os.path.join(CLEAN_DIR, "表单3_未知_clean.csv"),
                     encoding="utf-8-sig").copy()
    for c in COMPONENTS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def load_sample_level():
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


def fit_discriminators(known):
    X, y = known[COMPONENTS].values, known["类型编码"].values
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    lr_acc, rf_acc, dt_acc = [], [], []
    for tr, va in skf.split(X, y):
        lr_acc.append(Pipeline([("scale", StandardScaler()),
                                ("lr", LogisticRegression(max_iter=2000,
                                                          random_state=RANDOM_STATE))])
                      .fit(X[tr], y[tr]).score(X[va], y[va]))
        rf_acc.append(RandomForestClassifier(n_estimators=500,
                                             random_state=RANDOM_STATE)
                      .fit(X[tr], y[tr]).score(X[va], y[va]))
        dt_acc.append(DecisionTreeClassifier(max_depth=3, min_samples_leaf=2,
                                             random_state=RANDOM_STATE)
                      .fit(X[tr], y[tr]).score(X[va], y[va]))
    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE)
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2,
                                random_state=RANDOM_STATE)
    lr.fit(scaler.fit_transform(X), y)
    rf.fit(X, y)
    dt.fit(X, y)
    disc = {"scaler": scaler, "lr": lr, "rf": rf, "dt": dt}
    cv = {"lr": np.mean(lr_acc), "rf": np.mean(rf_acc), "dt": np.mean(dt_acc)}
    print("  判别底座 分层5折CV: Logistic {:.4f} / RF {:.4f} / 决策树 {:.4f}".format(
        cv["lr"], cv["rf"], cv["dt"]))
    return disc, cv


def fit_preweather_models(df):
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
        print("  风化还原[{}] R²={:.4f} 反推分母={:.4f}（n={}）".format(
            t, m.rsquared, denom, len(sub)))
    return models


def reconstruct(obs_row, hyp_type, models):
    comps = MODEL_COMPS[hyp_type]
    if any(pd.isna(obs_row[c]) or obs_row[c] == 0 for c in comps):
        return None
    m = models[hyp_type]
    dY = WEATHER_Y[hyp_type]
    d_sio2 = dY / m["denom"]
    pre = np.empty(len(COMPONENTS))
    for i, c in enumerate(COMPONENTS):
        v = obs_row[c]
        r_j = m["ratio"][c]
        if pd.isna(r_j):
            pre[i] = 0.0 if pd.isna(v) else v
        elif pd.isna(v):
            pre[i] = 0.0
        else:
            pre[i] = v - r_j * d_sio2
    if np.any(pre < -0.5) or np.any(pre > 100.5) or pre.sum() > 105.0:
        return None
    return pre


def predict_prob(vec, disc):
    X = np.array(vec, dtype=float).reshape(1, -1)
    p_lr = disc["lr"].predict_proba(disc["scaler"].transform(X))[0, 1]
    p_rf = disc["rf"].predict_proba(X)[0, 1]
    p_dt = disc["dt"].predict_proba(X)[0, 1]
    votes = [TYPE_INV[int(round(p))] for p in (p_lr, p_rf, p_dt)]
    return {"p_lr": float(p_lr), "p_rf": float(p_rf), "p_dt": float(p_dt),
            "votes": votes}


def majority(votes):
    return "铅钡" if votes.count("铅钡") >= 2 else "高钾"


def perturbation_sensitivity(vec, disc):
    rng = np.random.default_rng(RANDOM_STATE)
    vec = np.asarray(vec, dtype=float)
    base = disc["lr"].predict_proba(
        disc["scaler"].transform(vec.reshape(1, -1)))[0, 1]
    out = {}
    for amp in (0.05, 0.10):
        flips = 0
        for _ in range(N_REPEAT):
            vec_p = vec * rng.normal(1.0, amp, size=len(vec))
            p = disc["lr"].predict_proba(
                disc["scaler"].transform(vec_p.reshape(1, -1)))[0, 1]
            if (p > 0.5) != (base > 0.5):
                flips += 1
        out[amp] = flips / N_REPEAT
    return out


def main():
    t_all = time.time()

    stage(1, 5, "构建判别底座（三模型 + 分层 5 折）")
    known = load_known()
    disc, cv = fit_discriminators(known)

    stage(2, 5, "重算风化还原模型（采样点级）")
    sample_df = load_sample_level()
    pw_models = fit_preweather_models(sample_df)

    stage(3, 5, "8 件未知样品 假设-验证 判别")
    unknown = load_unknown()
    rows, sens_rows = [], []
    for _, r in unknown.iterrows():
        wid = r["文物编号"]
        weathered = (r["表面风化"] == "风化")
        obs = np.array([r[c] for c in COMPONENTS], dtype=float)

        p_direct = predict_prob(obs, disc)
        cls_direct = majority(p_direct["votes"])

        cand = [(cls_direct, p_direct["p_lr"], "直接判别")]
        recons = {}
        if weathered:
            for hyp in ["高钾", "铅钡"]:
                pre = reconstruct(r, hyp, pw_models)
                recons[hyp] = pre
                if pre is None:
                    print("  [{}] {}假设还原：关键成分缺失或超出物理范围 → 不可行".format(
                        wid, hyp))
                    continue
                pp = predict_prob(pre, disc)
                hyp_cls = majority(pp["votes"])
                cand.append((hyp_cls, pp["p_lr"], "{0}假设还原(判为{1})".format(hyp, hyp_cls)))
                print("  [{}] {}假设还原 → 判为{}（P(铅钡)={:.4f}，投票 {}）".format(
                    wid, hyp, hyp_cls, pp["p_lr"], "/".join(pp["votes"])))

        consist = [c for c in cand if (c[0] == "铅钡") == (c[1] > 0.5)]
        pool = consist if consist else cand
        final_cls, final_p, basis = max(pool, key=lambda c: abs(c[1] - 0.5))

        flip = perturbation_sensitivity(obs, disc)
        rows.append({
            "样品": wid, "表面风化": r["表面风化"],
            "直接判别P(铅钡)": round(p_direct["p_lr"], 4),
            "高钾假设还原P(铅钡)":
                round(predict_prob(recons["高钾"], disc)["p_lr"], 4)
                if weathered and recons.get("高钾") is not None else "",
            "铅钡假设还原P(铅钡)":
                round(predict_prob(recons["铅钡"], disc)["p_lr"], 4)
                if weathered and recons.get("铅钡") is not None else "",
            "最终类型": final_cls,
            "置信度": round(abs(final_p - 0.5) * 2, 4),
            "判别依据": basis,
        })
        sens_rows.append({
            "样品": wid, "表面风化": r["表面风化"], "最终类型": final_cls,
            "±5%扰动翻转率": round(flip[0.05], 4),
            "±10%扰动翻转率": round(flip[0.10], 4),
        })
        print("  [{}] 表面{} 直接判别={}（P(铅钡)={:.4f}，投票 {}）→ 最终类型={}（依据：{}，置信度 {:.4f}）".format(
            wid, r["表面风化"], cls_direct, p_direct["p_lr"],
            "/".join(p_direct["votes"]), final_cls, basis, abs(final_p - 0.5) * 2))

    tab4 = pd.DataFrame(rows)
    tab5 = pd.DataFrame(sens_rows)[["样品", "±5%扰动翻转率", "±10%扰动翻转率"]]

    stage(4, 5, "敏感性汇总与核对")
    print("  扰动翻转率：全部为 {}（论文表5）".format(
        "0" if (tab5["±5%扰动翻转率"].astype(str) == "0.0").all()
        and (tab5["±10%扰动翻转率"].astype(str) == "0.0").all() else "非零，见下表"))
    print(tab5.to_string(index=False))

    stage(5, 5, "生成论文插图（PDF）并写出 第三问.xlsx")
    order = pd.concat([
        tab4[tab4["最终类型"] == "高钾"].sort_values("直接判别P(铅钡)"),
        tab4[tab4["最终类型"] == "铅钡"].sort_values("直接判别P(铅钡)"),
    ]).reset_index(drop=True)
    ids, probs = order["样品"].tolist(), order["直接判别P(铅钡)"].astype(float).tolist()
    colors = ["#E8A13D" if t == "高钾" else "#5B8FF9" for t in order["最终类型"]]

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    y = np.arange(len(ids))
    ax.barh(y, probs, height=0.62, color=colors, edgecolor="none")
    ax.axvline(0.5, color="#D9534F", linestyle="--", linewidth=1.2, alpha=0.9)
    ax.text(0.5, len(ids) - 0.30, "0.5 判界", color="#D9534F", fontsize=9,
            ha="center", va="bottom")
    for yi, p, pid in zip(y, probs, ids):
        ax.text(p + 0.012, yi, "{:.2f}".format(p), va="center", fontsize=9, color="#333")
        ax.text(-0.012, yi, pid, va="center", ha="right", fontsize=9,
                color="#333", fontweight="bold")
    ax.set_yticks([]); ax.set_xlim(0, 1.06); ax.set_ylim(-0.6, len(ids) - 0.4)
    ax.set_xlabel("判别概率 P(铅钡)", fontsize=10)
    ax.set_title("未知样品判别概率（高钾橙色 / 铅钡蓝色）", fontsize=11)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.5)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(handles=[Patch(color="#E8A13D", label="高钾"),
                       Patch(color="#5B8FF9", label="铅钡")],
              loc="lower right", fontsize=9, frameon=False, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_q3_prob.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  图4 判别概率 → fig_q3_prob.pdf")

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        tab4.to_excel(w, sheet_name="表4-判别结果", index=False)
        tab5.to_excel(w, sheet_name="表5-翻转率", index=False)

    from openpyxl import load_workbook
    wb = load_workbook(OUT_XLSX)
    for ws in wb.worksheets:
        for col in ws.columns:
            width = max(len(str(c.value)) * 2.2 if c.value is not None else 0
                        for c in col) + 2
            ws.column_dimensions[col[0].column_letter].width = min(width, 40)
    wb.save(OUT_XLSX)

    print("\n问题 3 完成：输出 {}；图 → {} （总耗时 {:.1f}s）".format(
        OUT_XLSX, FIG_DIR, time.time() - t_all))


if __name__ == "__main__":
    main()
