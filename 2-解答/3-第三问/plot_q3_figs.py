# -*- coding: utf-8 -*-
"""
问题 3 论文图：8 件未知样品判别概率（P(铅钡)）
=================================================
横向条形图，0.5 处判界虚线；高钾橙色、铅钡蓝色。
数据来源：2-解答/3-第三问/results/q3_未知样品判别结果.csv
输出：3-论文/figures/fig_q3_prob.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体
for f in ["Microsoft YaHei", "SimHei", "STHeiti"]:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
RES = os.path.join(BASE_DIR, "results", "q3_未知样品判别结果.csv")
FIG = os.path.join(REPO_ROOT, "3-论文", "figures", "fig_q3_prob.png")

df = pd.read_csv(RES, encoding="utf-8-sig")

# 分组排序：高钾在前（按概率升序），铅钡在后（按概率升序）
hk = df[df["最终类型"] == "高钾"].sort_values("直接判别P(铅钡)")
pb = df[df["最终类型"] == "铅钡"].sort_values("直接判别P(铅钡)")
order = pd.concat([hk, pb]).reset_index(drop=True)

ids = order["文物编号"].tolist()
probs = order["直接判别P(铅钡)"].astype(float).tolist()
colors = ["#E8A13D" if t == "高钾" else "#5B8FF9"
          for t in order["最终类型"]]

fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
y = np.arange(len(ids))
bars = ax.barh(y, probs, height=0.62, color=colors, edgecolor="none")
ax.axvline(0.5, color="#D9534F", linestyle="--", linewidth=1.2, alpha=0.9)
ax.text(0.5, len(ids) - 0.30, "0.5 判界", color="#D9534F", fontsize=9,
        ha="center", va="bottom")
for yi, p, t in zip(y, probs, order["最终类型"]):
    ax.text(p + 0.012, yi, "{:.2f}".format(p), va="center", fontsize=9,
            color="#333")
    ax.text(-0.012, yi, ids[yi], va="center", ha="right", fontsize=9,
            color="#333", fontweight="bold")
ax.set_yticks([])
ax.set_xlim(0, 1.06)
ax.set_ylim(-0.6, len(ids) - 0.4)
ax.set_xlabel("判别概率 P(铅钡)", fontsize=10)
ax.set_title("未知样品判别概率（高钾橙色 / 铅钡蓝色）", fontsize=11)
ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.5)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.tick_params(axis="x", labelsize=9)

# 图例（手动）
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#E8A13D", label="高钾"),
                   Patch(color="#5B8FF9", label="铅钡")],
          loc="lower right", fontsize=9, frameon=False, ncol=2)

plt.tight_layout()
plt.savefig(FIG, bbox_inches="tight")
print("saved:", FIG)
