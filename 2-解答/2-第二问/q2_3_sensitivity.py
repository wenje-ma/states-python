# -*- coding: utf-8 -*-
"""
问题 2（3）：亚类划分的合理性与敏感性分析
================================================
方法依据（文献）：
  - 参数扰动敏感性：熊强等（2025）对玻璃成分施加 ±5%、±10% 扰动后重新
    聚类，检验亚类划分是否稳定；本文对每件样本的核心成分施加乘性高斯扰动
    （σ=5%、10%），重复 100 次重聚类，以调整兰德指数（ARI）与标签变化率
    度量亚类结构稳定性；
  - 特征方案扰动：刘振文等（2025）指出特征选择影响聚类结果，本文对比
    RF 核心 7 成分 / 全部 14 成分 / CLR 对数比 14 变量三套特征方案下
    的聚类结构与轮廓系数，检验亚类划分对特征口径的依赖；
  - 类数敏感性：课件第 3 讲（泛化与模型选择）"验证集用于选择"，亚类数
    K 属于模型设定，本文对推荐 K±1 重聚类，用 ARI 衡量亚类结构对类数
    设定的敏感程度（与熊强等（2025）高钾 3 类、彭丽红等（2024）高钾 3 类
    的文献设定对照）。

数据口径：56 件有效样本（高钾 16、铅钡 40），与 q2_2 相同的核心特征；
          扰动施加于原始成分含量后重新标准化，避免标准化参数泄漏。
输入：cleaned/表单2_填0版_全量58.csv、cleaned/表单2_对数比变换_全量58.csv、
      results/q2_1_特征重要性表.csv、results/q2_2_聚类评估表.csv
输出：results/q2_3_扰动敏感性.csv、results/q2_3_特征方案敏感性.csv、
      results/q2_3_类数敏感性.csv、results/q2_3_结果.txt
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

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
N_REPEAT = 100          # 扰动重复次数
TOP_K_FEAT = 7
PERTURB_SCALES = [0.05, 0.10]   # 熊强等（2025）±5%、±10%


# ----------------------------------------------------------------------
# 1. 读取数据与推荐 K
# ----------------------------------------------------------------------
def load_data():
    df = pd.read_csv(os.path.join(CLEAN_DIR, "表单2_填0版_全量58.csv"),
                     encoding="utf-8")
    df = df[df["是否有效"]].copy()
    imp = pd.read_csv(os.path.join(OUT_DIR, "q2_1_特征重要性表.csv"),
                      encoding="utf-8")
    feats = imp["成分"].head(TOP_K_FEAT).tolist()

    # CLR 对数比数据（56 行有效样本）
    clr = pd.read_csv(os.path.join(CLEAN_DIR, "表单2_对数比变换_全量58.csv"),
                      encoding="utf-8")
    clr = clr[["文物编号"] + [f"CLR_{c}" for c in COMPONENTS]]
    df = df.merge(clr, on="文物编号", how="left")

    # 推荐 K（读取 q2_2 评估表中"推荐"标记行）
    ev = pd.read_csv(os.path.join(OUT_DIR, "q2_2_聚类评估表.csv"),
                     encoding="utf-8")
    rec = ev[ev["推荐"].astype(str).str.contains("√")]
    rec_k = dict(zip(rec["类型"], rec["K"].astype(int)))
    return df, feats, rec_k


# ----------------------------------------------------------------------
# 2. 扰动敏感性（熊强等，2025 ±5%/±10%）
# ----------------------------------------------------------------------
def perturbation_sensitivity(df, feats, rec_k):
    """对每类型：核心成分加乘性高斯扰动后重聚类，统计 ARI 与标签变化率。"""
    rows = []
    rng = np.random.default_rng(RANDOM_STATE)
    for type_name in ["高钾", "铅钡"]:
        sub = df[df["类型"] == type_name]
        X_raw = sub[feats].values
        k = rec_k[type_name]
        scaler = StandardScaler()
        X0 = scaler.fit_transform(X_raw)
        base = KMeans(n_clusters=k, init="k-means++", n_init=10,
                      random_state=RANDOM_STATE).fit_predict(X0)

        for scale in PERTURB_SCALES:
            aris, change_rates = [], []
            for rep in range(N_REPEAT):
                noise = rng.normal(0, scale, size=X_raw.shape)
                Xn = X_raw * (1 + noise)          # 成分含量乘性扰动
                Xn = np.maximum(Xn, 0)            # 含量非负
                Xn_s = scaler.fit_transform(Xn)   # 扰动后重新标准化
                labels = KMeans(n_clusters=k, init="k-means++", n_init=10,
                                random_state=RANDOM_STATE).fit_predict(Xn_s)
                aris.append(adjusted_rand_score(base, labels))
                change_rates.append(np.mean(labels != base))
            rows.append({
                "类型": type_name, "扰动幅度": f"±{int(scale*100)}%",
                "ARI均值": round(np.mean(aris), 4),
                "ARI标准差": round(np.std(aris), 4),
                "标签变化率": round(np.mean(change_rates), 4),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 3. 特征方案敏感性（核心7 / 全部14 / CLR14）
# ----------------------------------------------------------------------
def feature_scheme_sensitivity(df, feats, rec_k):
    """三套特征方案分别聚类，以核心 7 成分为基准比较 ARI 与轮廓系数。"""
    rows = []
    schemes = [
        ("核心7成分", feats),
        ("全部14成分", COMPONENTS),
        ("CLR14变量", [f"CLR_{c}" for c in COMPONENTS]),
    ]
    for type_name in ["高钾", "铅钡"]:
        sub = df[df["类型"] == type_name]
        k = rec_k[type_name]
        scaler = StandardScaler()
        X_base = scaler.fit_transform(sub[feats].values)
        base = KMeans(n_clusters=k, init="k-means++", n_init=10,
                      random_state=RANDOM_STATE).fit_predict(X_base)
        base_sil = silhouette_score(X_base, base)

        for name, cols in schemes:
            Xs = scaler.fit_transform(sub[cols].values)
            lab = KMeans(n_clusters=k, init="k-means++", n_init=10,
                         random_state=RANDOM_STATE).fit_predict(Xs)
            rows.append({
                "类型": type_name, "特征方案": name,
                "轮廓系数": round(silhouette_score(Xs, lab), 4),
                "与核心7方案ARI": round(adjusted_rand_score(base, lab), 4),
                "基准轮廓系数": round(base_sil, 4),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 4. 类数敏感性（推荐 K±1）
# ----------------------------------------------------------------------
def k_sensitivity(df, feats, rec_k):
    """类数取推荐 K±1 时与推荐 K 亚类结构的 ARI。"""
    rows = []
    scaler = StandardScaler()
    for type_name in ["高钾", "铅钡"]:
        sub = df[df["类型"] == type_name]
        Xs = scaler.fit_transform(sub[feats].values)
        k = rec_k[type_name]
        base = KMeans(n_clusters=k, init="k-means++", n_init=10,
                      random_state=RANDOM_STATE).fit_predict(Xs)
        for k2 in [k - 1, k + 1]:
            if k2 < 2:
                continue
            lab = KMeans(n_clusters=k2, init="k-means++", n_init=10,
                         random_state=RANDOM_STATE).fit_predict(Xs)
            rows.append({
                "类型": type_name, "推荐K": k, "对比K": k2,
                "ARI": round(adjusted_rand_score(base, lab), 4),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 5. 主流程
# ----------------------------------------------------------------------
def main():
    df, feats, rec_k = load_data()

    pert = perturbation_sensitivity(df, feats, rec_k)
    pert.to_csv(os.path.join(OUT_DIR, "q2_3_扰动敏感性.csv"),
                index=False, encoding="utf-8-sig")

    feat = feature_scheme_sensitivity(df, feats, rec_k)
    feat.to_csv(os.path.join(OUT_DIR, "q2_3_特征方案敏感性.csv"),
                index=False, encoding="utf-8-sig")

    ks = k_sensitivity(df, feats, rec_k)
    ks.to_csv(os.path.join(OUT_DIR, "q2_3_类数敏感性.csv"),
              index=False, encoding="utf-8-sig")

    lines = ["=" * 60, "问题 2（3）亚类划分合理性与敏感性结果", "=" * 60]
    lines.append(f"推荐 K：高钾 {rec_k['高钾']}、铅钡 {rec_k['铅钡']}；"
                 f"核心特征：{', '.join(feats)}")
    lines.append("")
    lines.append("【1】成分扰动敏感性（熊强等 2025，±5% / ±10%，"
                 f"重复 {N_REPEAT} 次重聚类）")
    lines.append("  幅度越高、ARI 越接近 1 且标签变化率越小 → 亚类越稳定")
    for _, r in pert.iterrows():
        lines.append(f"  {r['类型']} {r['扰动幅度']}: "
                     f"ARI={r['ARI均值']:.4f}±{r['ARI标准差']:.4f}，"
                     f"标签变化率={r['标签变化率']*100:.1f}%")
    lines.append("")
    lines.append("【2】特征方案敏感性（核心7 / 全部14 / CLR14）")
    for _, r in feat.iterrows():
        lines.append(f"  {r['类型']} {r['特征方案']}: "
                     f"轮廓系数={r['轮廓系数']:.4f}，"
                     f"与核心7方案ARI={r['与核心7方案ARI']:.4f}")
    lines.append("")
    lines.append("【3】类数敏感性（推荐 K±1）")
    for _, r in ks.iterrows():
        lines.append(f"  {r['类型']} K={r['推荐K']} vs K={r['对比K']}: "
                     f"ARI={r['ARI']:.4f}")
    lines.append("=" * 60)

    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "q2_3_结果.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
