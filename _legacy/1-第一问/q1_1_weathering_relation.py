# -*- coding: utf-8 -*-
"""
问题 1（1）：表面风化与玻璃类型、纹饰、颜色的关系
=====================================================
方法依据（文献）：
  - 主方法：卡方独立性检验；期望频数不足时采用蒙特卡洛置换检验
    （费希尔精确检验在 2×2 之外的扩展，彭丽红等，2024 对零频数采用 Fisher）；
  - 互证 1：Spearman 秩相关（熊强等，2025）；
  - 互证 2：Kendall tau 相关（刘振文等，2025）。

数据口径：使用文物级分类信息（表单 1，58 件）。
颜色处理：原始颜色 8 类频数过稀，归并为 4 大色系；
          4 件文物（19/40/48/58）颜色缺失（均为风化铅钡），颜色分析剔除并说明。

输入：cleaned/表单1_基本信息_clean.csv
输出：results/q1_1_列联表.csv、results/q1_1_检验结果.csv、results/q1_1_结果.txt
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

# 颜色归并规则：相近颜色并为色系（频数过稀的类别合并后列联表才可检验）
COLOR_MAP = {
    "浅蓝": "蓝绿系", "深蓝": "蓝绿系", "蓝绿": "蓝绿系",
    "浅绿": "绿系", "绿": "绿系", "深绿": "绿系",
    "黑": "黑", "紫": "紫",
}

# 秩相关所需的类别编码（列联表检验的补充，刻画单调关联方向）
TYPE_CODE = {"铅钡": 0, "高钾": 1}
WEATHER_CODE = {"无风化": 0, "风化": 1}
ORNA_CODE = {"A": 1, "B": 2, "C": 3}
COLOR_CODE = {"蓝绿系": 1, "绿系": 2, "黑": 3, "紫": 4}

N_PERM = 5000          # 蒙特卡洛置换次数
SEED = 42


# ----------------------------------------------------------------------
# 1. 读取数据与预处理
# ----------------------------------------------------------------------
def load_and_prepare():
    """读取表单 1 清洗表；颜色归并色系；返回副本。"""
    df = pd.read_csv(os.path.join(CLEAN_DIR, "表单1_基本信息_clean.csv"),
                     encoding="utf-8-sig")
    df = df.copy()
    df["颜色系"] = df["颜色"].map(COLOR_MAP)
    return df


# ----------------------------------------------------------------------
# 2. 卡方独立性检验（含蒙特卡洛置换兜底）
# ----------------------------------------------------------------------
def chi2_contingency_robust(x, y, n_perm=N_PERM, seed=SEED):
    """
    卡方独立性检验；若期望频数过小（存在 <5 的格）则退化为蒙特卡洛
    置换检验：随机重排 y 标签 n_perm 次，统计卡方统计量超过观测值的
    比例作为 P 值（费希尔精确检验在多行多列表中的一般化）。
    返回 (卡方统计量, P 值, 是否使用置换, 期望频数矩阵)。
    """
    ct = pd.crosstab(x, y)
    chi2, p, dof, expected = stats.chi2_contingency(ct)

    use_perm = False
    if (expected < 5).any():                    # 期望频数不足，卡方近似不可靠
        use_perm = True
        rng = np.random.default_rng(seed)
        y_arr = y.values
        idx = y.index
        cnt = 0
        for _ in range(n_perm):
            y_shuf = pd.Series(rng.permutation(y_arr), index=idx)
            ct_p = pd.crosstab(x, y_shuf)
            chi2_p = stats.chi2_contingency(ct_p)[0]
            if chi2_p >= chi2:
                cnt += 1
        p = (cnt + 1) / (n_perm + 1)            # +1 避免 P=0 的伪精确
    return chi2, p, use_perm, expected


# ----------------------------------------------------------------------
# 3. 秩相关（Spearman 与 Kendall tau 互证）
# ----------------------------------------------------------------------
def rank_correlations(df, col, code_map):
    """
    对给定属性列做秩相关分析（编码后），返回与表面风化的
    Spearman 与 Kendall tau 相关系数与 P 值。
    """
    sub = df.dropna(subset=[col])               # 剔除该属性缺失的样本
    x = sub[col].map(code_map).astype(float)
    y = sub["表面风化"].map(WEATHER_CODE).astype(float)
    rho, p_s = stats.spearmanr(x, y)
    tau, p_t = stats.kendalltau(x, y)
    return rho, p_s, tau, p_t, len(sub)


# ----------------------------------------------------------------------
# 4. 主流程
# ----------------------------------------------------------------------
def main():
    df = load_and_prepare()
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("问题1(1)：表面风化与类型/纹饰/颜色的关系")
    log("样本：{} 件文物（表单1）".format(len(df)))
    log("=" * 70)

    rows = []                                    # 汇总表行
    tables = {}

    # ---- 逐属性做列联表 + 卡方/置换 ----
    for col in ["类型", "纹饰", "颜色系"]:
        sub = df.dropna(subset=[col])            # 颜色缺失 4 件在此剔除
        ct = pd.crosstab(sub[col], sub["表面风化"])
        chi2, p, use_perm, expected = chi2_contingency_robust(sub[col], sub["表面风化"])

        tables[col] = ct
        log("\n[{}] × 表面风化  列联表（n={}，剔除缺失 {} 件）".format(
            col, len(sub), len(df) - len(sub)))
        log(ct.to_string())
        log("卡方统计量 = {:.3f}，P 值 = {:.4f}{}".format(
            chi2, p, "（蒙特卡洛置换检验）" if use_perm else "（卡方近似）"))
        sig = "显著" if p < 0.05 else "不显著"
        log("结论：风化与{}的关系{}（P={:.4f}）".format(col, sig, p))

        # 秩相关互证
        code_map = {"类型": TYPE_CODE, "纹饰": ORNA_CODE,
                    "颜色系": COLOR_CODE}[col]
        rho, p_s, tau, p_t, n = rank_correlations(df, col, code_map)
        log("互证：Spearman ρ={:+.3f} (P={:.4f})；Kendall τ={:+.3f} (P={:.4f})".format(
            rho, p_s, tau, p_t))

        rows.append({
            "属性": col, "样本数": n, "卡方统计量": round(chi2, 3),
            "P值": round(p, 4), "检验方式": "置换" if use_perm else "卡方",
            "显著性(0.05)": sig,
            "Spearman_rho": round(rho, 3), "Spearman_P": round(p_s, 4),
            "Kendall_tau": round(tau, 3), "Kendall_P": round(p_t, 4),
        })

    # ---- 缺失情况说明 ----
    miss_color = df[df["颜色"].isna()]
    log("\n[颜色缺失说明] 共 {} 件缺失：编号 {}，均为{}玻璃且{}。".format(
        len(miss_color), miss_color["文物编号"].tolist(),
        "、".join(miss_color["类型"].unique()), "、".join(miss_color["表面风化"].unique())))

    # ---- 输出 ----
    for name, ct in tables.items():
        ct.to_csv(os.path.join(OUT_DIR, "q1_1_列联表_{}.csv".format(name)),
                  encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, "q1_1_检验结果.csv"),
        index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT_DIR, "q1_1_结果.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n输出目录: {}".format(OUT_DIR))


if __name__ == "__main__":
    main()
