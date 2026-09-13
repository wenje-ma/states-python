# -*- coding: utf-8 -*-
"""
问题 1（2）：结合玻璃类型，分析表面有无风化的化学成分统计规律
================================================================
方法依据（文献）：
  - 分组含量统计（均值/中位数/标准差/含量范围）：熊强等（2025）给出
    未风化/风化/严重风化各组分的含量范围表；
  - 差异检验：Mann-Whitney U 非参数检验 + Cohen's d 效应量
    （钱鸿羽等，2024 用 Mann-Whitney U + Cohen's d 排序筛选风化相关成分；
     成分数据不满足正态性假设，故采用非参数检验）。

数据口径：文物级主表（51 件，多采样点已聚合），按 类型 × 风化 分组。
          （严重风化点仅 3 条且位于特殊点表，属采样点级，留待问题 1（3）使用。）

输入：cleaned/表单2_文物级_clean.csv
输出：results/q1_2_含量统计表.csv、results/q1_2_检验结果.csv、results/q1_2_结果.txt
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

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
ALPHA = 0.05


# ----------------------------------------------------------------------
# 1. 分组含量统计（含量范围表）
# ----------------------------------------------------------------------
def group_stats(group):
    """
    对一组样本计算各成分统计量：n、均值、中位数、标准差、最小、最大。
    返回长表 DataFrame（成分 × 统计量）。
    """
    rows = []
    for col in COMPONENTS:
        s = group[col].dropna()
        rows.append({
            "成分": col,
            "n": int(s.size),
            "均值": round(s.mean(), 3),
            "中位数": round(s.median(), 3),
            "标准差": round(s.std(), 3) if s.size > 1 else np.nan,
            "最小": round(s.min(), 3),
            "最大": round(s.max(), 3),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 2. Mann-Whitney U + Cohen's d
# ----------------------------------------------------------------------
def cohen_d(x, y):
    """Cohen's d 效应量：两组均值差 / 合并标准差。"""
    n1, n2 = len(x), len(y)
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return (x.mean() - y.mean()) / sp


def compare_weathering(weathered, unweathered):
    """
    对单个类型内 无风化 vs 风化 逐成分检验。
    返回逐成分 DataFrame：U 统计量、P 值、效应量 d、方向、显著性。
    """
    rows = []
    for col in COMPONENTS:
        g1 = weathered[col].dropna()      # 风化组
        g0 = unweathered[col].dropna()    # 无风化组
        if len(g1) < 2 or len(g0) < 2:    # 样本不足无法检验
            rows.append({"成分": col, "n风化": len(g1), "n无风化": len(g0),
                         "U": np.nan, "P值": np.nan, "Cohen_d": np.nan,
                         "方向": "—", "显著性": "样本不足"})
            continue
        u, p = stats.mannwhitneyu(g1, g0, alternative="two-sided")
        d = cohen_d(g1, g0)
        direction = "风化高" if g1.mean() > g0.mean() else "风化低"
        rows.append({
            "成分": col, "n风化": len(g1), "n无风化": len(g0),
            "U": round(u, 2), "P值": round(p, 4),
            "Cohen_d": round(d, 3), "方向": direction,
            "显著性": "显著" if p < ALPHA else "不显著",
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 3. 主流程
# ----------------------------------------------------------------------
def main():
    art = pd.read_csv(os.path.join(CLEAN_DIR, "表单2_文物级_clean.csv"),
                      encoding="utf-8-sig")
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("问题1(2)：风化前后化学成分含量的统计规律（文物级，n={}）".format(len(art)))
    log("=" * 70)

    stat_parts = []      # 含量统计表（各类型×风化组）
    test_parts = []      # 检验表（各类型）
    for t in ["高钾", "铅钡"]:
        sub = art[art["类型"] == t]
        w = sub[sub["表面风化"] == "风化"]
        u = sub[sub["表面风化"] == "无风化"]

        log("\n[{}玻璃] 无风化 {} 件 / 风化 {} 件".format(t, len(u), len(w)))

        # 含量统计
        st = group_stats(w)
        st.insert(0, "分组", "风化")
        st0 = group_stats(u)
        st0.insert(0, "分组", "无风化")
        st_all = pd.concat([st0, st], ignore_index=True)
        st_all.insert(0, "类型", t)
        stat_parts.append(st_all)

        # 差异检验
        test = compare_weathering(w, u)
        test.insert(0, "类型", t)
        test_parts.append(test)

        log("含量统计（均值/范围）与检验结果：")
        show = test.copy()
        show["风化均值"] = [round(w.loc[w[c].notna(), c].mean(), 2)
                          if w[c].notna().sum() else np.nan for c in COMPONENTS]
        show["无风化均值"] = [round(u.loc[u[c].notna(), c].mean(), 2)
                            if u[c].notna().sum() else np.nan for c in COMPONENTS]
        log(show[["成分", "风化均值", "无风化均值", "P值", "Cohen_d",
                  "方向", "显著性"]].to_string(index=False))

        sig_cols = test.loc[test["显著性"] == "显著", "成分"].tolist()
        log("显著成分（P<0.05）：{}".format(sig_cols if sig_cols else "无"))

    # 汇总输出
    pd.concat(stat_parts, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "q1_2_含量统计表.csv"),
        index=False, encoding="utf-8-sig")
    pd.concat(test_parts, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "q1_2_检验结果.csv"),
        index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT_DIR, "q1_2_结果.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n输出目录: {}".format(OUT_DIR))


if __name__ == "__main__":
    main()
