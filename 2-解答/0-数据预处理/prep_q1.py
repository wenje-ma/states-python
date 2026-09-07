# -*- coding: utf-8 -*-
"""
问题1 数据准备 —— 全量文物表 + 三张求解输入表
==============================================
生成内容（写入 cleaned/Q1/）：
  [0] 表单2_文物级_全量.csv
      58 件文物全量表：主采样点均值聚合；仅含特殊点的 7 件文物
      （23/25/28/29/42/44/53，只有未风化点）以未风化点成分并入，
      并标注"成分数据来源"，避免口径混淆。

  [1] Q1_类别属性表.csv   —— 问题1第(1)问：风化 × (类型/纹饰/颜色)
      58 件文物的类别属性 + 各采样点类型计数 + 颜色缺失标记，
      直接用于列联表 / 卡方独立性检验（含 Fisher 精确检验）。

  [2] Q1_成分对照表.csv   —— 问题1第(2)问：风化前后化学成分统计规律
      采样点级（69 条），含"风化状态"维度（未风化点/风化文物点/严重风化点），
      按类型分组做均值±标准差统计、显著性检验的输入。

  [3] Q1_风化反推表.csv   —— 问题1第(3)问：由风化点反推风化前成分
      文物级有效样本（85%~105%），含风化程度等级（无风化0/风化1/严重风化2），
      填0版成分 + ALR 对数比变换列，供风化程度函数拟合与反推建模。

运行：python prep_q1.py
"""

import os
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# 0. 路径与常量
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # 脚本所在目录
OUT_DIR = os.path.join(BASE_DIR, "cleaned")                       # 第一阶段/第二阶段输出目录
Q1_DIR = os.path.join(OUT_DIR, "Q1")                              # 问题1数据目录
os.makedirs(Q1_DIR, exist_ok=True)

COMPONENTS = [
    "SiO2", "Na2O", "K2O", "CaO", "MgO", "Al2O3", "Fe2O3",
    "CuO", "PbO", "BaO", "P2O5", "SrO", "SnO2", "SO2",
]
SUM_MIN, SUM_MAX = 85.0, 105.0

# 采样点类型分类
MAIN_TYPES = ["普通采样点", "部位采样点"]          # 主采样点：并入文物均值
SPECIAL_TYPES = ["未风化点", "严重风化点"]          # 特殊点：风化对照


# ----------------------------------------------------------------------
# 1. 全量 58 件文物表（未风化点并入）
# ----------------------------------------------------------------------
def full_artifact_table(f1, f2):
    """
    全量文物表：
      - 有主采样点的文物：主点成分均值；
      - 只有特殊点的文物：用未风化点（多个则取均值）作为成分代表，
        并标注"成分数据来源=未风化点"。
    返回 58 行文物表。
    """
    rows = []
    for iid in f1["文物编号"]:
        sub = f2[f2["文物编号"] == iid]
        main = sub[sub["采样点类型"].isin(MAIN_TYPES)]
        special = sub[sub["采样点类型"].isin(SPECIAL_TYPES)]

        if len(main) > 0:
            comp = main[COMPONENTS].mean()                       # 主点均值
            source = "主采样点均值"
            n_main = len(main)
        else:
            # 只有特殊点：未风化点优先（代表该文物未风化区域成分）
            unw = special[special["采样点类型"] == "未风化点"]
            comp = unw[COMPONENTS].mean() if len(unw) > 0 else special[COMPONENTS].mean()
            source = "未风化点" if len(unw) > 0 else "严重风化点"
            n_main = 0

        rec = {"文物编号": iid, "成分数据来源": source, "主采样点数": n_main}
        for c in COMPONENTS:
            rec[c] = comp[c]
        rows.append(rec)

    art = pd.DataFrame(rows)
    # 合并类别信息（表单1）
    art = art.merge(
        f1[["文物编号", "类型", "纹饰", "颜色", "表面风化"]],
        on="文物编号", how="left")

    # 派生列
    art["成分累加和"] = art[COMPONENTS].sum(axis=1)
    art["检出成分数"] = art[COMPONENTS].notna().sum(axis=1)
    art["是否有效"] = art["成分累加和"].between(SUM_MIN, SUM_MAX)
    return art


def sample_type_counts(f2):
    """
    每件文物各类采样点数量（普通/部位/未风化/严重风化），
    供类别属性表使用（反映该文物数据来自哪些采样点）。
    """
    cnt = f2.pivot_table(index="文物编号", columns="采样点类型",
                         values="采样点编号", aggfunc="count").fillna(0)
    # 统一列名
    for t in MAIN_TYPES + SPECIAL_TYPES:
        if t not in cnt.columns:
            cnt[t] = 0
    cnt = cnt[MAIN_TYPES + SPECIAL_TYPES].astype(int).reset_index()
    cnt.columns = ["文物编号", "普通点数", "部位点数", "未风化点数", "严重风化点数"]
    return cnt


# ----------------------------------------------------------------------
# 2. Q1 三张输入表
# ----------------------------------------------------------------------
def q1_category_table(art, f2):
    """问题1(1)：类别 × 风化 关联分析表（列联表输入）。"""
    cnt = sample_type_counts(f2)
    t = art[["文物编号", "类型", "纹饰", "颜色", "表面风化"]].copy()
    t = t.merge(cnt, on="文物编号", how="left")
    # 颜色缺失标记（4 件风化铅钡文物缺颜色，卡方检验需单独处理）
    t["颜色是否缺失"] = t["颜色"].isna()
    return t


def q1_component_table(f2):
    """
    问题1(2)：风化前后化学成分对照（采样点级）。
    风化状态 = 未风化点 / 风化文物点 / 严重风化点：
      - 采样点类型=未风化点        → "未风化"
      - 采样点类型=严重风化点      → "严重风化"
      - 其余（普通/部位点）        → 按所属文物标签："无风化文物" / "风化文物"
    """
    f2 = f2.copy()
    f2["风化状态"] = np.select(
        [f2["采样点类型"] == "未风化点",
         f2["采样点类型"] == "严重风化点",
         f2["表面风化"] == "无风化"],
        ["未风化", "严重风化", "无风化文物"],
        default="风化文物")
    return f2


def q1_restore_table(art):
    """
    问题1(3)：风化前反推建模输入（文物级有效样本）。
      - 仅保留 85%~105% 有效文物（闭合性可信才可做成分反推）；
      - 风化程度等级：无风化=0 / 风化=1 / 严重风化=2（赋值留给建模层）；
      - 成分提供"填0版"（反推计算用）与"保留NaN版"（统计参考）；
      - 附带 ALR 对数比变换列（对数比变换预处理的结果）。
    """
    valid = art[art["是否有效"]].copy().reset_index(drop=True)

    # 风化程度等级：严重风化的采样点存在于 8/26/54（文物标签仍为"风化"）
    # 用文物级严重风化标记：该文物存在严重风化采样点
    f2 = pd.read_csv(os.path.join(OUT_DIR, "表单2_特殊点_clean.csv"),
                     encoding="utf-8-sig")
    severe_ids = set(f2.loc[f2["采样点类型"] == "严重风化点", "文物编号"])
    valid["风化程度"] = np.select(
        [valid["表面风化"] == "无风化", valid["文物编号"].isin(severe_ids)],
        [0, 2], default=1)

    # 填 0 版成分
    zf = valid.copy()
    zf[COMPONENTS] = zf[COMPONENTS].fillna(0.0)

    # ALR 变换（与 preprocess2.py 同口径：参考 SiO2，0 替换为最小正值的一半）
    ref = "SiO2"
    zf2 = zf.copy()
    for col in COMPONENTS:
        pos = zf2[col][zf2[col] > 0]
        small = pos.min() * 0.5 if len(pos) > 0 else 0.001
        zf2[col] = zf2[col].replace(0.0, small)
    out = valid[["文物编号", "类型", "表面风化", "风化程度",
                 "成分数据来源", "成分累加和", "是否有效"]].copy()
    for col in COMPONENTS:
        out["填0_" + col] = zf[col].values
    for col in COMPONENTS:
        if col != ref:
            out["ALR_" + col] = np.log(zf2[col] / zf2[ref])
    return out


# ----------------------------------------------------------------------
# 3. 主流程
# ----------------------------------------------------------------------
def main():
    f1 = pd.read_csv(os.path.join(OUT_DIR, "表单1_基本信息_clean.csv"),
                     encoding="utf-8-sig")
    f2 = pd.read_csv(os.path.join(OUT_DIR, "表单2_成分_clean.csv"),
                     encoding="utf-8-sig")

    # [0] 全量 58 件文物表
    art = full_artifact_table(f1, f2)
    art.to_csv(os.path.join(OUT_DIR, "表单2_文物级_全量.csv"),
               index=False, encoding="utf-8-sig")

    # [1] 类别属性表
    cat = q1_category_table(art, f2)
    cat.to_csv(os.path.join(Q1_DIR, "Q1_类别属性表.csv"),
               index=False, encoding="utf-8-sig")

    # [2] 成分对照表（采样点级）
    comp = q1_component_table(f2)
    comp.to_csv(os.path.join(Q1_DIR, "Q1_成分对照表.csv"),
                index=False, encoding="utf-8-sig")

    # [3] 风化反推表（文物级有效）
    rest = q1_restore_table(art)
    rest.to_csv(os.path.join(Q1_DIR, "Q1_风化反推表.csv"),
                index=False, encoding="utf-8-sig")

    # ---- 概览 ----
    print("=" * 64)
    print("问题1 数据准备完成")
    print("=" * 64)
    print("[0] 全量文物表: {} 件 | 有效 {} | 来源: 主点均值 {}, 未风化点 {}".format(
        len(art), art["是否有效"].sum(),
        (art["成分数据来源"] == "主采样点均值").sum(),
        (art["成分数据来源"] == "未风化点").sum()))
    print("\n  类型×风化（全量58件）:")
    for idx, row in pd.crosstab(art["类型"], art["表面风化"]).iterrows():
        print("    {}: 无风化={}, 风化={}".format(idx, row.get("无风化", 0), row.get("风化", 0)))
    print("  颜色缺失: {} 件 ({})".format(
        art["颜色"].isna().sum(), art.loc[art["颜色"].isna(), "文物编号"].tolist()))

    print("\n[1] Q1_类别属性表: {} 件 × {} 列（列联表/卡方输入）".format(*cat.shape))
    print("[2] Q1_成分对照表: {} 条采样点 × {} 列".format(*comp.shape))
    print("    风化状态分布: {}".format(
        dict(comp["风化状态"].value_counts())))
    print("[3] Q1_风化反推表: {} 件有效 × {} 列".format(*rest.shape))
    print("    风化程度分布: {}".format(dict(rest["风化程度"].value_counts())))
    print("\n输出目录: {}".format(Q1_DIR))


if __name__ == "__main__":
    main()
