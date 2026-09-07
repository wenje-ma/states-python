# -*- coding: utf-8 -*-
"""
问题 2 论文插图生成（图 6-1 / 6-2 / 6-3）
================================================
图 6-1：随机森林特征重要性条形图（6.1 分类规律，前 7 位核心成分高亮）
图 6-2：高钾、铅钡玻璃亚类划分散点图（6.2，核心成分标准化后 PCA 投影，
        按 K-means++ 亚类着色并标注质心）
图 6-3：K-means++ 聚类评估图（6.2，SSE 柱状 + 轮廓系数折线随 K 变化，
        标注推荐类数）

输入：results/q2_1_特征重要性表.csv、results/q2_2_聚类评估表.csv、
      results/q2_2_亚类划分.csv、cleaned/表单2_填0版_全量58.csv
输出：3-论文/figures/fig_q2_importance.png、fig_q2_cluster.png、fig_q2_elbow.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------
# 0. 路径与中文字体
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # 2-解答/2-第二问
REPO_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))           # 仓库根
CLEAN_DIR = os.path.join(REPO_ROOT, "2-解答", "0-数据预处理", "cleaned")
RES_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(REPO_ROOT, "3-论文", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200

COMPONENTS = [
    "SiO2", "Na2O", "K2O", "CaO", "MgO", "Al2O3", "Fe2O3",
    "CuO", "PbO", "BaO", "P2O5", "SrO", "SnO2", "SO2",
]
TOP_K_FEAT = 7
RANDOM_STATE = 42
CORE_COLOR = "#1f77b4"      # 核心成分高亮色
GRAY = "#9e9e9e"


# ----------------------------------------------------------------------
# 图 6-1：RF 特征重要性条形图
# ----------------------------------------------------------------------
def fig_importance():
    imp = pd.read_csv(os.path.join(RES_DIR, "q2_1_特征重要性表.csv"),
                      encoding="utf-8")
    imp = imp.sort_values("特征重要性", ascending=True)   # 条形图自下而上
    names = imp["成分"].tolist()
    vals = imp["特征重要性"].values
    colors = [CORE_COLOR if i >= len(vals) - TOP_K_FEAT else GRAY
              for i in range(len(vals))]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.barh(names, vals, color=colors, edgecolor="white", linewidth=0.4)
    for b, v in zip(bars, vals):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("特征重要性")
    ax.set_ylabel("化学成分")
    ax.set_title("图6-1  随机森林特征重要性（前7位为分类核心成分）")
    ax.set_xlim(0, 0.32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # 图例
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=CORE_COLOR, label="分类核心成分（前7位）"),
                       Patch(color=GRAY, label="其余成分")],
              loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_q2_importance.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("已生成 图6-1（特征重要性）")


# ----------------------------------------------------------------------
# 图 6-2：亚类划分散点图（PCA 投影）
# ----------------------------------------------------------------------
def fig_cluster():
    df = pd.read_csv(os.path.join(CLEAN_DIR, "表单2_填0版_全量58.csv"),
                     encoding="utf-8")
    df = df[df["是否有效"]].copy()
    sub = pd.read_csv(os.path.join(RES_DIR, "q2_2_亚类划分.csv"),
                      encoding="utf-8")
    imp = pd.read_csv(os.path.join(RES_DIR, "q2_1_特征重要性表.csv"),
                      encoding="utf-8")
    feats = imp["成分"].head(TOP_K_FEAT).tolist()
    df = df.merge(sub[["文物编号", "亚类", "亚类命名"]], on="文物编号")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    cmap = plt.get_cmap("tab10")
    for ax, type_name in zip(axes, ["高钾", "铅钡"]):
        g = df[df["类型"] == type_name]
        X = StandardScaler().fit_transform(g[feats].values)
        xy = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
        labels = g["亚类"].values
        names = g["亚类命名"].values
        n_cls = labels.max() + 1

        for c in range(n_cls):
            mask = labels == c
            ax.scatter(xy[mask, 0], xy[mask, 1], s=38, alpha=0.85,
                       color=cmap(c), edgecolor="white", linewidth=0.5,
                       label=f"{names[mask][0]}（{mask.sum()}件）")
            cx, cy = xy[mask, 0].mean(), xy[mask, 1].mean()
            ax.scatter(cx, cy, marker="X", s=110, color=cmap(c),
                       edgecolor="black", linewidth=0.6, zorder=5)
        ax.set_title(f"{type_name}玻璃（K={n_cls}）", fontsize=11)
        ax.set_xlabel("第一主成分")
        ax.set_ylabel("第二主成分")
        ax.legend(fontsize=7, frameon=False, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("图6-2  高钾、铅钡玻璃亚类划分（核心成分PCA投影，×为亚类质心）",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_q2_cluster.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("已生成 图6-2（亚类散点）")


# ----------------------------------------------------------------------
# 图 6-3：聚类评估（SSE + 轮廓系数）
# ----------------------------------------------------------------------
def fig_elbow():
    ev = pd.read_csv(os.path.join(RES_DIR, "q2_2_聚类评估表.csv"),
                     encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, type_name in zip(axes, ["高钾", "铅钡"]):
        g = ev[ev["类型"] == type_name]
        k = g["K"].values
        sse = g["SSE"].values
        sil = g["轮廓系数"].values
        rec = g.loc[g["推荐"].astype(str).str.contains("√"), "K"].values
        rec_k = int(rec[0]) if len(rec) else int(k[np.argmax(sil)])

        ax.bar(k, sse, width=0.55, color="#d5e8f7",
               edgecolor=CORE_COLOR, linewidth=0.8, label="SSE（左轴）")
        ax.set_xlabel("类数 K")
        ax.set_ylabel("SSE", color=CORE_COLOR)
        ax.tick_params(axis="y", labelcolor=CORE_COLOR)

        ax2 = ax.twinx()
        ax2.plot(k, sil, "-o", color="#e07b39", linewidth=1.6, markersize=4,
                 label="轮廓系数（右轴）")
        ax2.set_ylabel("轮廓系数", color="#e07b39")
        ax2.tick_params(axis="y", labelcolor="#e07b39")
        ax2.set_ylim(0, 0.7)

        ax.axvline(rec_k, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(rec_k, ax.get_ylim()[1] * 0.98, f"K={rec_k}",
                ha="center", va="top", color="red", fontsize=9)
        ax.set_title(f"{type_name}玻璃", fontsize=11)
        ax.set_xticks(k)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

    fig.suptitle("图6-3  K-means++聚类评估（SSE与轮廓系数随K的变化）",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_q2_elbow.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("已生成 图6-3（聚类评估）")


if __name__ == "__main__":
    fig_importance()
    fig_cluster()
    fig_elbow()
    print("全部图片生成完毕 ->", FIG_DIR)
