# -*- coding: utf-8 -*-
"""
古代玻璃成分数据 —— 预处理第二阶段（结合文献方法）
=====================================================
方法依据（三篇文献）：
  [1] 钱鸿羽等：成分数据需做对数比变换（ALR）以消除定和约束；未检出按 0 处理；
      用 Mann-Whitney U + Cohen's d 识别风化相关成分（本脚本只做数据准备）。
  [2] 彭丽红等：85%~105% 有效区间；有效数据空值按 0 处理。
  [3] 侯兴汉等：高缺失率下不可粗暴插补（尤其低于检出限的缺失），
      小样本场景可用 SVR 插补（另见 svr_impute.py）。

本阶段产出（写入 cleaned/）：
  1. 文物级主表（51 件）：普通点/部位点多采样点聚合为一件文物的代表成分；
  2. 文物级全量表（58 件）：另将"只有未风化点、无普通采样点"的 7 件文物
     （23/25/28/29/42/44/53）用未风化点成分回填，保证问题2/3建模覆盖全部文物；
  3. 特殊点表：未风化点、严重风化点单独保留（问题1 风化前/后对照用）；
  4. 缺失分级：14 种成分按缺失率分高/中/低三档，给出处理策略；
  5. 两套数值口径：保留 NaN 版（统计用）+ 未检出填 0 版（闭合性计算用）；
  6. 高缺失成分的"检出/未检出"二值特征列；
  7. ALR / CLR 对数比变换（在有效样本、填 0 后 0 值替换为小正值的基础上）。

运行：python preprocess2.py
"""

import os
import re
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# 0. 路径与常量（相对路径，基于脚本位置推导，仓库可整体搬迁）
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # 脚本所在目录（2-解答/0-数据预处理）
REPO_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))           # 仓库根目录（states-python）
DATA_DIR = os.path.join(REPO_ROOT, "1-题目")                      # 原始数据目录
CLEAN_DIR = os.path.join(BASE_DIR, "cleaned")                     # 第一阶段清洗结果目录
OUT_DIR = os.path.join(BASE_DIR, "cleaned")                       # 输出目录（与第一阶段一致）
os.makedirs(OUT_DIR, exist_ok=True)

# 14 种化学成分的标准列名
COMPONENTS = [
    "SiO2", "Na2O", "K2O", "CaO", "MgO", "Al2O3", "Fe2O3",
    "CuO", "PbO", "BaO", "P2O5", "SrO", "SnO2", "SO2",
]

# 有效数据区间（题目规定）
SUM_MIN, SUM_MAX = 85.0, 105.0

# 缺失率分级阈值：缺失率 >= HIGH_RATE 视为高缺失，>= MID_RATE 视为中缺失，其余低缺失
HIGH_RATE, MID_RATE = 50.0, 20.0

# 特殊采样点类型（问题1 风化前后对照专用，不并入文物主成分）
SPECIAL_TYPES = ["未风化点", "严重风化点"]

# ALR 参考成分（缺失率最低、含量最高的成分作分母，结果最稳定）
ALR_REF = "SiO2"


# ----------------------------------------------------------------------
# 1. 读取第一阶段清洗结果
# ----------------------------------------------------------------------
def load_cleaned():
    """读取第一阶段输出的三张清洗表。"""
    f1 = pd.read_csv(os.path.join(CLEAN_DIR, "表单1_基本信息_clean.csv"),
                     encoding="utf-8-sig")
    f2 = pd.read_csv(os.path.join(CLEAN_DIR, "表单2_成分_clean.csv"),
                     encoding="utf-8-sig")
    f3 = pd.read_csv(os.path.join(CLEAN_DIR, "表单3_未知_clean.csv"),
                     encoding="utf-8-sig")
    return f1, f2, f3


# ----------------------------------------------------------------------
# 2. 多采样点聚合：文物主成分 + 特殊点
# ----------------------------------------------------------------------
def aggregate_artifacts(f2):
    """
    将同一文物的多个普通/部位采样点聚合为文物级代表成分（均值）；
    未风化点、严重风化点单独保留为特殊点表。
    返回 (文物级主表, 特殊点表)。
    """
    # 主采样点：普通点 + 部位（同一文物的多个检测值取平均，代表该文物）
    main = f2[~f2["采样点类型"].isin(SPECIAL_TYPES)].copy()
    # 特殊点：未风化点 / 严重风化点（保留原始记录，供风化前后对照）
    special = f2[f2["采样点类型"].isin(SPECIAL_TYPES)].copy()

    # ---- 文物级主表 ----
    g = main.groupby("文物编号")
    art = g[COMPONENTS].mean()                      # 各成分均值（该文物所有主采样点）
    art = art.reset_index()

    # 元信息（类型/纹饰/颜色/风化）取该文物第一条记录（同一文物应一致）
    meta = main.groupby("文物编号")[["纹饰", "类型", "颜色", "表面风化"]].first()
    art = art.merge(meta, on="文物编号", how="left")

    # 聚合信息：该文物参与聚合的主采样点个数
    art["主采样点数"] = g.size().values

    # 复用第一阶段派生列逻辑：累加和（NaN 跳过，等价按 0 计）、检出数、有效性
    art["成分累加和"] = art[COMPONENTS].sum(axis=1)
    art["检出成分数"] = art[COMPONENTS].notna().sum(axis=1)
    art["是否有效"] = art["成分累加和"].between(SUM_MIN, SUM_MAX)

    # ---- 特殊点表 ----
    special = special.copy()
    special = special.reset_index(drop=True)

    return art, special


# ----------------------------------------------------------------------
# 3. 全量 58 件文物表：未风化点回填
# ----------------------------------------------------------------------
def build_full_table(f2, art):
    """
    构建覆盖全部 58 件文物的全量表：
      - 有主采样点的文物（51 件）用主点均值；
      - 只有未风化点、无主采样点的文物（23/25/28/29/42/44/53）用其未风化点
        成分均值作为该文物代表（未风化点近似文物未风化状态，问题2/3建模可用）。
    返回全量表（含"成分来源"列区分两种口径）。
    """
    main_ids = set(art["文物编号"])

    # 未风化点中、且不在主表里的文物（即"只有未风化点"的 7 件）
    uw = f2[f2["采样点类型"] == "未风化点"]
    uw = uw[~uw["文物编号"].isin(main_ids)].copy()

    g = uw.groupby("文物编号")[COMPONENTS].mean()          # 多未风化点取均值（如42有两个）
    g = g.reset_index()
    meta = uw.groupby("文物编号")[["纹饰", "类型", "颜色", "表面风化"]].first()
    g = g.merge(meta, on="文物编号", how="left")
    g["主采样点数"] = 0                                   # 无普通/部位采样点
    g["成分来源"] = "未风化点回填"

    art2 = art.copy()
    art2["成分来源"] = "主采样点均值"
    full = pd.concat([art2, g], ignore_index=True)
    full = full.sort_values("文物编号").reset_index(drop=True)

    # 派生列（与主表同口径）
    full["成分累加和"] = full[COMPONENTS].sum(axis=1)
    full["检出成分数"] = full[COMPONENTS].notna().sum(axis=1)
    full["是否有效"] = full["成分累加和"].between(SUM_MIN, SUM_MAX)
    return full


# ----------------------------------------------------------------------
# 4. 缺失分级与两套数值口径
# ----------------------------------------------------------------------
def missing_classify(table):
    """
    按缺失率把 14 种成分分为 高缺失 / 中缺失 / 低缺失，
    返回 (分级字典, 缺失率 Series)。
    """
    miss_rate = table[COMPONENTS].isna().mean() * 100       # 每成分缺失率（%）
    level = {}
    for col in COMPONENTS:
        r = miss_rate[col]
        if r >= HIGH_RATE:
            level[col] = "高缺失"                          # 大概率低于检出限，数值不可信
        elif r >= MID_RATE:
            level[col] = "中缺失"                          # 可考虑插补或谨慎使用
        else:
            level[col] = "低缺失"                          # 数据较完整，正常使用
    return level, miss_rate


def build_detected_flags(table):
    """
    为 14 种成分生成"检出/未检出"二值特征（检出=1，未检出=0）。
    未检出信息本身可能携带分类价值（如某成分只在某类玻璃中检出）。
    """
    flags = table[["文物编号"]].copy()
    for col in COMPONENTS:
        flags["检出_" + col] = table[col].notna().astype(int)
    return flags


def zero_filled(table):
    """
    未检出填 0 版（钱鸿羽/彭丽红口径）：用于累加和、闭合性、对数比变换等
    需要完整矩阵的计算。注意：仅作计算口径，不用于"未检出=含量为0"的解读。
    """
    zf = table.copy()
    zf[COMPONENTS] = zf[COMPONENTS].fillna(0.0)
    return zf


# ----------------------------------------------------------------------
# 5. 对数比变换（钱鸿羽：成分数据变换，消除定和约束）
# ----------------------------------------------------------------------
def replace_zero_small(df, comps):
    """
    0 值替换为极小正值：0 无法取对数，按成分数据惯例（multiplicative
    replacement 的简化版）将 0 替换为该成分最小正检测值的一半。
    在填 0 版数据上执行，返回替换后的副本。
    """
    out = df.copy()
    for col in comps:
        vals = out[col]
        pos = vals[vals > 0]
        if len(pos) > 0:
            small = pos.min() * 0.5                    # 最小正检测值的一半
        else:
            small = 0.001                              # 整列全为 0 的兜底值
        out[col] = vals.replace(0.0, small)
    return out


def alr_transform(df, comps, ref=ALR_REF):
    """
    加性对数比变换（Additive Log-Ratio）：以参考成分为分母，
    y_i = ln(x_i / x_ref)，i ≠ ref。
    将成分数据从单形空间映射到欧氏空间，可用经典统计方法。
    """
    out = df[["文物编号"]].copy()
    for col in comps:
        if col == ref:
            continue                                   # 参考成分自身不参与（信息冗余）
        out["ALR_" + col] = np.log(df[col] / df[ref])
    return out


def clr_transform(df, comps):
    """
    中心对数比变换（Centered Log-Ratio）：以几何均值为基准，
    y_i = ln(x_i / g(x))，g(x) 为 14 成分的几何均值。
    """
    out = df[["文物编号"]].copy()
    # 几何均值 = exp(各成分 ln 的均值)
    gmean = np.exp(df[comps].apply(np.log).mean(axis=1))
    for col in comps:
        out["CLR_" + col] = np.log(df[col] / gmean)
    return out


def save_derived(table, tag):
    """
    对一张文物级表生成配套文件（填0版 / 检出标志 / 对数比变换），并统一命名。
    tag: 文件名后缀标识，如 "_51" / "_全量58"
    """
    flags = build_detected_flags(table)
    flags.to_csv(os.path.join(OUT_DIR, "表单2_检出标志{}.csv".format(tag)),
                 index=False, encoding="utf-8-sig")
    zf = zero_filled(table)
    zf.to_csv(os.path.join(OUT_DIR, "表单2_填0版{}.csv".format(tag)),
              index=False, encoding="utf-8-sig")
    # 对数比变换：只在有效文物上做（保证闭合性可信），0 先替换为小正值
    valid = zf[zf["是否有效"]].copy()
    valid = replace_zero_small(valid, COMPONENTS)
    alr = alr_transform(valid, COMPONENTS)
    clr = clr_transform(valid, COMPONENTS)
    logratio = alr.merge(clr, on="文物编号")
    logratio.to_csv(os.path.join(OUT_DIR, "表单2_对数比变换{}.csv".format(tag)),
                    index=False, encoding="utf-8-sig")
    return flags, zf, logratio


# ----------------------------------------------------------------------
# 6. 概览报告
# ----------------------------------------------------------------------
def report(art, full, special, level, miss_rate, flags, logratio):
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 64)
    log("第二阶段预处理概览（结合文献方法）")
    log("=" * 64)

    log("\n[1] 文物级主表: {} 件（多采样点已聚合）".format(len(art)))
    log("  类型×风化（文物级）: ")
    ct = pd.crosstab(art["类型"], art["表面风化"])
    for idx, row in ct.iterrows():
        log("    {}: 无风化={}, 风化={}".format(idx, row.get("无风化", 0), row.get("风化", 0)))
    log("  有效(85%~105%): {} / {}".format(art["是否有效"].sum(), len(art)))

    log("\n[2] 文物级全量表: {} 件（回填 7 件未风化点文物）".format(len(full)))
    log("  类型×风化（全量）: ")
    ct2 = pd.crosstab(full["类型"], full["表面风化"])
    for idx, row in ct2.iterrows():
        log("    {}: 无风化={}, 风化={}".format(idx, row.get("无风化", 0), row.get("风化", 0)))
    log("  有效(85%~105%): {} / {}".format(full["是否有效"].sum(), len(full)))
    backfill = full[full["成分来源"] == "未风化点回填"]
    log("  未风化点回填文物: {}".format(backfill["文物编号"].tolist()))

    log("\n[3] 特殊点表: {} 条（未风化/严重风化，问题1对照用）".format(len(special)))
    log("  未风化点 {} 条 | 严重风化点 {} 条".format(
        (special["采样点类型"] == "未风化点").sum(),
        (special["采样点类型"] == "严重风化点").sum()))
    log("  严重风化点所在文物: {}".format(
        special.loc[special["采样点类型"] == "严重风化点", "文物编号"].tolist()))

    log("\n[4] 缺失分级（文物级主表，n={}）:".format(len(art)))
    for col in COMPONENTS:
        log("    {:<7} 缺失率 {:>5.1f}%  →  {}".format(
            col, miss_rate[col], level[col]))

    log("\n[5] 两套口径: 保留NaN版（统计用）/ 填0版（闭合计算用）已生成")
    log("    高缺失成分({}个)不参与数值插补，仅保留检出标志".format(
        sum(1 for v in level.values() if v == "高缺失")))

    log("\n[6] 对数比变换: ALR(参考={}) + CLR，主表 {} 列 / 全量表 {} 列".format(
        ALR_REF, logratio.shape[1] - 1, logratio.shape[1] - 1))

    with open(os.path.join(OUT_DIR, "数据概览2.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------------
# 7. 主流程
# ----------------------------------------------------------------------
def main():
    f1, f2, f3 = load_cleaned()

    # 1) 聚合：文物主成分 + 特殊点；全量 58 件表
    art, special = aggregate_artifacts(f2)
    full = build_full_table(f2, art)

    # 2) 缺失分级（以文物级主表口径）
    level, miss_rate = missing_classify(art)

    # 3) 主表配套文件（51 件）
    flags51, zf51, lr51 = save_derived(art, "_51")

    # 4) 全量表配套文件（58 件）
    flags58, zf58, lr58 = save_derived(full, "_全量58")

    # 5) 写表
    art.to_csv(os.path.join(OUT_DIR, "表单2_文物级_clean.csv"),
               index=False, encoding="utf-8-sig")       # 保留 NaN 版（51件，原名保持兼容）
    full.to_csv(os.path.join(OUT_DIR, "表单2_文物级_全量58.csv"),
                index=False, encoding="utf-8-sig")
    special.to_csv(os.path.join(OUT_DIR, "表单2_特殊点_clean.csv"),
                   index=False, encoding="utf-8-sig")

    # 6) 概览
    report(art, full, special, level, miss_rate, flags51, lr51)
    print("\n输出目录: {}".format(OUT_DIR))


if __name__ == "__main__":
    main()
