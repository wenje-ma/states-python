# -*- coding: utf-8 -*-
"""
问题 2（2）：高钾、铅钡玻璃内部的亚类划分
================================================
方法依据（文献）：
  - K-means++ 聚类：刘振文等（2025）以随机森林筛出的核心成分对玻璃分类别
    做 K-means++ 聚类（铅钡聚为 5 类，肘部图定类数），本文复用该流程，
    分类核心成分动态取自 q2_1 随机森林特征重要性前 7 位；
  - 层次聚类对照：彭丽红等（2024）采用系统聚类划分亚类（高钾 3 类、铅钡
    2 类，并据主导成分命名），本文以 Ward 层次聚类作对照，用调整兰德指数
    （ARI）衡量两种聚类结构的一致性；
  - 类数选择：以轮廓系数最大为定量准则，辅以肘部法（组内平方和 SSE 随 K
    的下降率突变点）；类数需同时满足 2 ≤ K ≤ n-1；
  - 亚类命名：参照彭丽红等（2024）"高硅/高钾/高钙"式命名，按各亚类相对
    同类型全体样本的主导成分偏移（标准化均值差）自动生成亚类标签。

数据口径：与 q2_1 一致，56 件有效样本（高钾 16、铅钡 40），特征为填 0 版
          原始成分中 RF 重要性前 7 位，聚类前逐类型做 Z-score 标准化。
输入：cleaned/表单2_填0版_全量58.csv、results/q2_1_特征重要性表.csv
输出：results/q2_2_聚类评估表.csv、results/q2_2_亚类划分.csv、
      results/q2_2_亚类画像.csv、results/q2_2_结果.txt
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
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
MAX_K = 8          # 类数扫描上限
TOP_K_FEAT = 7     # 分类核心成分个数（刘振文等，2025 取 7 个）


# ----------------------------------------------------------------------
# 1. 读取数据与核心特征
# ----------------------------------------------------------------------
def load_data_and_features():
    df = pd.read_csv(os.path.join(CLEAN_DIR, "表单2_填0版_全量58.csv"),
                     encoding="utf-8")
    df = df[df["是否有效"]].copy()

    imp = pd.read_csv(os.path.join(OUT_DIR, "q2_1_特征重要性表.csv"),
                      encoding="utf-8")
    feats = imp["成分"].head(TOP_K_FEAT).tolist()
    return df, feats


# ----------------------------------------------------------------------
# 2. K-means++ 聚类评估（肘部法 + 轮廓系数）
# ----------------------------------------------------------------------
def evaluate_kmeans(X, max_k):
    """对标准化矩阵扫描 K，返回评估表与推荐 K。"""
    rows = []
    for k in range(2, min(max_k, X.shape[0]) + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                    random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k < X.shape[0] else np.nan
        rows.append({"K": k, "SSE": round(km.inertia_, 4),
                     "轮廓系数": round(sil, 4)})
    ev = pd.DataFrame(rows)

    # 轮廓系数最大优先；若轮廓系数随 K 递减，退化为肘部法（SSE 下降率最大点）
    best_sil = ev.loc[ev["轮廓系数"].idxmax()]
    if best_sil["轮廓系数"] < 0.05 or ev["轮廓系数"].is_monotonic_decreasing:
        sse = ev["SSE"].values
        drop = np.diff(sse) / np.abs(sse[:-1])
        k_elbow = int(ev["K"].iloc[int(np.argmax(drop)) + 1])
        rec = k_elbow
        rec_mode = "肘部法"
    else:
        rec = int(best_sil["K"])
        rec_mode = "轮廓系数"
    ev["推荐"] = np.where(ev["K"] == rec, f"√({rec_mode})", "")
    return ev, rec


# ----------------------------------------------------------------------
# 3. 层次聚类对照（彭丽红等，2024 系统聚类）
# ----------------------------------------------------------------------
def hierarchical_ari(X, k, km_labels):
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
    hc_labels = hc.fit_predict(X)
    return hc_labels, adjusted_rand_score(km_labels, hc_labels)


# ----------------------------------------------------------------------
# 4. 亚类画像与命名（彭丽红等，2024 主导成分命名）
# ----------------------------------------------------------------------
def subclass_profile(df, type_name, sub_labels, feats):
    """按亚类统计 14 原始成分均值，据相对全体样本的标准化偏移命名。"""
    sub = df[df["类型"] == type_name].copy()
    sub["亚类"] = sub_labels
    all_mean = sub[COMPONENTS].mean()
    all_std = sub[COMPONENTS].std().replace(0, np.nan)

    rows = []
    for c in sorted(sub["亚类"].unique()):
        g = sub[sub["亚类"] == c]
        means = g[COMPONENTS].mean()
        diff = (means - all_mean) / all_std
        lead = diff.abs().idxmax()
        direction = "高" if means[lead] > all_mean[lead] else "低"
        name = f"{direction}{lead}型"
        rows.append({"类型": type_name, "亚类": int(c), "命名": name, "n": int(len(g)),
                     **{col: round(means[col], 2) for col in COMPONENTS}})
    profile = pd.DataFrame(rows)
    return profile


# ----------------------------------------------------------------------
# 5. 主流程
# ----------------------------------------------------------------------
def main():
    df, feats = load_data_and_features()
    scaler = StandardScaler()

    all_ev = []
    all_sub = []
    all_profile = []
    summary = []
    for type_name in ["高钾", "铅钡"]:
        sub = df[df["类型"] == type_name].copy()
        X = scaler.fit_transform(sub[feats].values)

        ev, rec_k = evaluate_kmeans(X, MAX_K)
        ev.insert(0, "类型", type_name)
        all_ev.append(ev)

        km = KMeans(n_clusters=rec_k, init="k-means++", n_init=10,
                    random_state=RANDOM_STATE)
        km_labels = km.fit_predict(X)
        sub["亚类"] = km_labels

        # 层次聚类对照
        hc_labels, ari = hierarchical_ari(X, rec_k, km_labels)
        sub["层次亚类"] = hc_labels

        # 画像命名
        profile = subclass_profile(df, type_name, km_labels, feats)
        all_profile.append(profile)

        # 合并亚类命名到划分表
        name_map = dict(zip(profile["亚类"], profile["命名"]))
        sub["亚类命名"] = sub["亚类"].map(name_map)
        all_sub.append(sub)

        summary.append({
            "类型": type_name, "样本数": len(sub), "核心特征数": len(feats),
            "推荐K": rec_k,
            "轮廓系数": float(ev.loc[ev["K"] == rec_k, "轮廓系数"].iloc[0]),
            "SSE": float(ev.loc[ev["K"] == rec_k, "SSE"].iloc[0]),
            "层次聚类ARI": round(ari, 4),
        })
        ev.loc[ev["K"] == rec_k, "层次聚类ARI"] = round(ari, 4)

    ev_all = pd.concat(all_ev, ignore_index=True)
    ev_all.to_csv(os.path.join(OUT_DIR, "q2_2_聚类评估表.csv"),
                  index=False, encoding="utf-8-sig")

    sub_all = pd.concat(all_sub, ignore_index=True)
    sub_all[["文物编号", "类型", "亚类", "亚类命名", "层次亚类"]].to_csv(
        os.path.join(OUT_DIR, "q2_2_亚类划分.csv"),
        index=False, encoding="utf-8-sig")

    prof_all = pd.concat(all_profile, ignore_index=True)
    prof_all.to_csv(os.path.join(OUT_DIR, "q2_2_亚类画像.csv"),
                    index=False, encoding="utf-8-sig")

    # 汇总文本
    lines = ["=" * 60, "问题 2（2）亚类划分结果", "=" * 60]
    lines.append(f"分类核心成分（RF 前 {TOP_K_FEAT}）：{', '.join(feats)}")
    for s in summary:
        lines.append("")
        lines.append(f"【{s['类型']}】样本 {s['样本数']} 件，推荐 K = {s['推荐K']}"
                     f"（轮廓系数 {s['轮廓系数']:.4f}，SSE {s['SSE']:.4f}，"
                     f"层次聚类 ARI = {s['层次聚类ARI']}）")
    lines.append("")
    lines.append("【亚类画像（主导成分命名，彭丽红等 2024 思路）】")
    for _, r in prof_all.iterrows():
        lines.append(f"  {r['类型']} 亚类{r['亚类']}（{r['命名']}，n={r['n']}）")
    lines.append("=" * 60)

    # 亚类画像的差异成分说明
    lines.append("")
    lines.append("【各亚类 vs 类型全体的主导差异成分】")
    for _, r in prof_all.iterrows():
        g = sub_all[(sub_all["类型"] == r["类型"]) & (sub_all["亚类"] == r["亚类"])]
        sub_mean = r
        type_all = df[df["类型"] == r["类型"]]
        diffs = {}
        for c in COMPONENTS:
            m_all = type_all[c].mean()
            sd_all = type_all[c].std()
            if sd_all > 0:
                diffs[c] = (sub_mean[c] - m_all) / sd_all
        top3 = sorted(diffs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        desc = "；".join([f"{c}{'+' if v>0 else '-'}({abs(v):.2f}σ)" for c, v in top3])
        lines.append(f"  {r['类型']} 亚类{r['亚类']} {r['命名']}：{desc}")

    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "q2_2_结果.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
