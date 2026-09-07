# -*- coding: utf-8 -*-
"""
古代玻璃制品成分数据 —— 预处理脚本
=====================================
目标：为问题 1/2/3 提供清洗后的基础数据集
  1. 读取三张表单（基本信息 / 已分类成分 / 未知成分）
  2. 统一成分列名，缺失值保留为 NaN（空白 = 未检测到，而非 0）
  3. 解析表单 2 采样点信息（文物编号 + 采样点类型：普通/部位/未风化/严重风化）
  4. 计算成分累加和，按 85%~105% 筛选有效数据
  5. 将表单 1 的分类信息关联到表单 2 成分数据
  6. 输出清洗后 csv + 数据概览报告

输出目录：0-数据预处理/cleaned/
运行方式：python preprocess.py
"""

import os
import re
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# 0. 路径与常量
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # 脚本所在目录（2-解答/0-数据预处理）
REPO_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))           # 仓库根目录（states-python）
DATA_DIR = os.path.join(REPO_ROOT, "1-题目")                      # 原始数据目录（相对仓库根）
OUT_DIR = os.path.join(BASE_DIR, "cleaned")                       # 清洗结果目录（脚本同级）
os.makedirs(OUT_DIR, exist_ok=True)                               # 输出目录不存在则创建

# 14 种化学成分的标准列名（对应 CSV 表头括号中的化学式）
COMPONENTS = [
    "SiO2", "Na2O", "K2O", "CaO", "MgO", "Al2O3", "Fe2O3",
    "CuO", "PbO", "BaO", "P2O5", "SrO", "SnO2", "SO2",
]

# 题目规定的有效数据区间：成分累加和介于 85%~105% 视为有效
SUM_MIN, SUM_MAX = 85.0, 105.0


# ----------------------------------------------------------------------
# 1. 通用工具函数
# ----------------------------------------------------------------------
def rename_components(df):
    """
    把 "二氧化硅(SiO2)" 这类带中文的列名统一改为括号内的化学式。
    例如 "二氧化硅(SiO2)" -> "SiO2"，以便后续按 COMPONENTS 统一引用。
    """
    mapping = {}
    for col in df.columns:
        # 提取括号中的化学式
        m = re.search(r"\(([^()]+)\)", str(col))
        if m:
            mapping[col] = m.group(1).strip()
    return df.rename(columns=mapping)


def to_float_components(df):
    """把成分列统一转为 float（空字符串 / 缺失自动变为 NaN）。"""
    for col in COMPONENTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_summary_cols(df):
    """
    为成分数据追加三个派生列：
      - 成分累加和：各成分相加，未检出(NaN)按 0 计（成分性数据的常用口径）
      - 检出成分数：实际检测到的成分个数（非 NaN 计数）
      - 是否有效：累加和是否落在 85%~105%
    """
    df["成分累加和"] = df[COMPONENTS].sum(axis=1)          # NaN 自动跳过，等价于按 0 求和
    df["检出成分数"] = df[COMPONENTS].notna().sum(axis=1)  # 每行非空成分个数
    df["是否有效"] = df["成分累加和"].between(SUM_MIN, SUM_MAX)  # 85% <= sum <= 105%
    return df


# ----------------------------------------------------------------------
# 2. 表单 1：文物基本信息
# ----------------------------------------------------------------------
def load_form1():
    """读取文物基本信息，规范列名并整理编号。"""
    df = pd.read_csv(
        os.path.join(DATA_DIR, "附件_表单1.csv"), encoding="utf-8"
    )
    df.columns = ["文物编号", "纹饰", "类型", "颜色", "表面风化"]
    df["文物编号"] = pd.to_numeric(df["文物编号"], errors="coerce").astype("Int64")
    # 颜色存在缺失（如编号 19/40/48/58），保留 NaN 并在概览中统计
    return df


# ----------------------------------------------------------------------
# 3. 表单 2：已分类文物成分（含特殊采样点）
# ----------------------------------------------------------------------
def parse_sample_point(col):
    """
    解析"文物采样点"列，返回 (文物编号, 采样点类型, 序号)。
    规则：编号为行首数字，剩余文字用于判断采样点类型。
      "01"           -> (1,  普通点,  无)
      "03部位1"      -> (3,  部位,    1)
      "08严重风化点" -> (8,  严重风化点, 无)
      "23未风化点"   -> (23, 未风化点, 无)
      "42未风化点1"  -> (42, 未风化点, 1)
    """
    m = re.match(r"^(\d+)(.*)$", str(col).strip())
    if not m:                      # 格式异常（理论上不会出现）
        return (np.nan, "未知", np.nan)
    num = int(m.group(1))          # 文物编号
    rest = m.group(2)              # 剩余文字
    if "严重风化点" in rest:
        ptype = "严重风化点"        # 风化层采样：代表风化后的状态
    elif "未风化点" in rest:
        ptype = "未风化点"          # 风化文物表面未风化区域：近似风化前状态
    elif "部位" in rest:
        ptype = "部位"              # 文物造型上的不同部位
    else:
        ptype = "普通点"            # 表面随机采样
    # 提取序号（部位1、未风化点2 等；无序号时为 None）
    nums = re.findall(r"\d+", rest)
    seq = int(nums[0]) if nums else np.nan
    return (num, ptype, seq)


def load_form2():
    """读取已分类成分数据，解析采样点并与表单 1 关联。"""
    df = pd.read_csv(
        os.path.join(DATA_DIR, "附件_表单2.csv"), encoding="utf-8"
    )
    df = rename_components(df)          # 统一成分列名
    df = to_float_components(df)        # 成分列转数值（空 -> NaN）

    # 解析采样点：拆出 文物编号 / 采样点类型 / 序号
    parsed = df["文物采样点"].apply(parse_sample_point)
    df["文物编号"] = parsed.apply(lambda x: x[0]).astype("Int64")
    df["采样点类型"] = parsed.apply(lambda x: x[1])
    df["采样点序号"] = parsed.apply(lambda x: x[2])

    df = add_summary_cols(df)           # 累加和 / 检出数 / 有效性

    # 关联表单 1 的分类信息（类型 / 纹饰 / 颜色 / 风化）
    f1 = load_form1()
    df = df.merge(f1, on="文物编号", how="left", suffixes=("", "_表1"))
    return df


# ----------------------------------------------------------------------
# 4. 表单 3：未知类别文物成分
# ----------------------------------------------------------------------
def load_form3():
    """读取未知类别文物成分数据。"""
    df = pd.read_csv(
        os.path.join(DATA_DIR, "附件_表单3.csv"), encoding="utf-8"
    )
    df = rename_components(df)          # 统一成分列名
    df = to_float_components(df)        # 成分列转数值
    df = add_summary_cols(df)           # 累加和 / 检出数 / 有效性
    return df


# ----------------------------------------------------------------------
# 5. 数据概览报告
# ----------------------------------------------------------------------
def report(f1, f2, f3):
    """打印并保存一份数据概览，便于核对预处理结果。"""
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 60)
    log("数据预处理概览")
    log("=" * 60)

    # ---- 表单 1 ----
    log("\n[表单1] 文物基本信息: {} 行".format(len(f1)))
    log("  类型分布: " + ", ".join(
        "{}={}".format(k, v) for k, v in f1["类型"].value_counts().items()))
    log("  风化分布: " + ", ".join(
        "{}={}".format(k, v) for k, v in f1["表面风化"].value_counts().items()))
    log("  颜色缺失: {} 件 (编号 {})".format(
        f1["颜色"].isna().sum(),
        ",".join(f1.loc[f1["颜色"].isna(), "文物编号"].astype(str).tolist())))

    # ---- 表单 2 ----
    log("\n[表单2] 已分类成分数据: {} 个采样点".format(len(f2)))
    log("  采样点类型: " + ", ".join(
        "{}={}".format(k, v) for k, v in f2["采样点类型"].value_counts().items()))
    log("  有效数据(85%~105%): {} / {}".format(f2["是否有效"].sum(), len(f2)))
    log("  无效数据编号: {}".format(
        f2.loc[~f2["是否有效"], "文物采样点"].tolist()))
    log("  类型×风化 分布:")
    ct = pd.crosstab(f2["类型"], f2["表面风化"])
    for idx, row in ct.iterrows():
        log("    {}: 无风化={}, 风化={}".format(idx, row.get("无风化", 0), row.get("风化", 0)))
    log("  各成分检出率(未检出比例):")
    for col in COMPONENTS:
        miss = f2[col].isna().mean() * 100
        log("    {:<7} 缺失 {:.1f}%".format(col, miss))

    # ---- 表单 3 ----
    log("\n[表单3] 未知类别成分数据: {} 件".format(len(f3)))
    log("  有效数据(85%~105%): {} / {}".format(f3["是否有效"].sum(), len(f3)))
    log("  风化分布: " + ", ".join(
        "{}={}".format(k, v) for k, v in f3["表面风化"].value_counts().items()))

    # 保存概览
    with open(os.path.join(OUT_DIR, "数据概览.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------------
# 6. 主流程：读取 -> 清洗 -> 输出
# ----------------------------------------------------------------------
def main():
    f1 = load_form1()                    # 表单 1
    f2 = load_form2()                    # 表单 2（含关联的分类信息）
    f3 = load_form3()                    # 表单 3

    # 输出清洗后的数据集（utf-8-sig 便于 Excel 直接打开）
    f1.to_csv(os.path.join(OUT_DIR, "表单1_基本信息_clean.csv"),
              index=False, encoding="utf-8-sig")
    f2.to_csv(os.path.join(OUT_DIR, "表单2_成分_clean.csv"),
              index=False, encoding="utf-8-sig")
    f3.to_csv(os.path.join(OUT_DIR, "表单3_未知_clean.csv"),
              index=False, encoding="utf-8-sig")

    report(f1, f2, f3)                   # 打印并保存概览
    print("\n输出目录: {}".format(OUT_DIR))


if __name__ == "__main__":
    main()
