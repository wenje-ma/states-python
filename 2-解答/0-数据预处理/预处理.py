# -*- coding: utf-8 -*-
import os
import re
import time
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(REPO_ROOT, "1-题目")
OUT_DIR = os.path.join(BASE_DIR, "cleaned")
Q1_DIR = os.path.join(OUT_DIR, "Q1")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(Q1_DIR, exist_ok=True)

COMPONENTS = [
    "SiO2", "Na2O", "K2O", "CaO", "MgO", "Al2O3", "Fe2O3",
    "CuO", "PbO", "BaO", "P2O5", "SrO", "SnO2", "SO2",
]
SUM_MIN, SUM_MAX = 85.0, 105.0
HIGH_RATE, MID_RATE = 50.0, 20.0
SPECIAL_TYPES = ["未风化点", "严重风化点"]
ALR_REF = "SiO2"
SVR_PARAM_GRID = {"C": [0.1, 1, 10, 50], "gamma": [0.01, 0.05, 0.1, 0.5],
                  "epsilon": [0.01, 0.05, 0.1]}

_t0 = time.time()
def stage(no, total, msg):
    print("[{}/{}] {}  （累计耗时 {:.1f}s）".format(no, total, msg, time.time() - _t0))


def rename_components(df):
    mapping = {}
    for col in df.columns:
        m = re.search(r"\(([^()]+)\)", str(col))
        if m:
            mapping[col] = m.group(1).strip()
    return df.rename(columns=mapping)


def to_float_components(df):
    for col in COMPONENTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_summary_cols(df):
    df["成分累加和"] = df[COMPONENTS].sum(axis=1)
    df["检出成分数"] = df[COMPONENTS].notna().sum(axis=1)
    df["是否有效"] = df["成分累加和"].between(SUM_MIN, SUM_MAX)
    return df


def load_form1():
    df = pd.read_csv(os.path.join(DATA_DIR, "附件_表单1.csv"), encoding="utf-8")
    df.columns = ["文物编号", "纹饰", "类型", "颜色", "表面风化"]
    df["文物编号"] = pd.to_numeric(df["文物编号"], errors="coerce").astype("Int64")
    return df


def parse_sample_point(col):
    m = re.match(r"^(\d+)(.*)$", str(col).strip())
    if not m:
        return (np.nan, "未知", np.nan)
    num = int(m.group(1))
    rest = m.group(2)
    if "严重风化点" in rest:
        ptype = "严重风化点"
    elif "未风化点" in rest:
        ptype = "未风化点"
    elif "部位" in rest:
        ptype = "部位"
    else:
        ptype = "普通点"
    nums = re.findall(r"\d+", rest)
    seq = int(nums[0]) if nums else np.nan
    return (num, ptype, seq)


def load_form2():
    df = pd.read_csv(os.path.join(DATA_DIR, "附件_表单2.csv"), encoding="utf-8")
    df = rename_components(df)
    df = to_float_components(df)
    parsed = df["文物采样点"].apply(parse_sample_point)
    df["文物编号"] = parsed.apply(lambda x: x[0]).astype("Int64")
    df["采样点类型"] = parsed.apply(lambda x: x[1])
    df["采样点序号"] = parsed.apply(lambda x: x[2])
    df = add_summary_cols(df)
    f1 = load_form1()
    df = df.merge(f1, on="文物编号", how="left", suffixes=("", "_表1"))
    return df


def load_form3():
    df = pd.read_csv(os.path.join(DATA_DIR, "附件_表单3.csv"), encoding="utf-8")
    df = rename_components(df)
    df = to_float_components(df)
    df = add_summary_cols(df)
    return df


def aggregate_artifacts(f2):
    main = f2[~f2["采样点类型"].isin(SPECIAL_TYPES)].copy()
    special = f2[f2["采样点类型"].isin(SPECIAL_TYPES)].copy()

    g = main.groupby("文物编号")
    art = g[COMPONENTS].mean().reset_index()
    meta = main.groupby("文物编号")[["纹饰", "类型", "颜色", "表面风化"]].first()
    art = art.merge(meta, on="文物编号", how="left")
    art["主采样点数"] = g.size().values
    art["成分累加和"] = art[COMPONENTS].sum(axis=1)
    art["检出成分数"] = art[COMPONENTS].notna().sum(axis=1)
    art["是否有效"] = art["成分累加和"].between(SUM_MIN, SUM_MAX)
    special = special.reset_index(drop=True)
    return art, special


def build_full_table(f2, art):
    main_ids = set(art["文物编号"])
    uw = f2[f2["采样点类型"] == "未风化点"]
    uw = uw[~uw["文物编号"].isin(main_ids)].copy()

    g = uw.groupby("文物编号")[COMPONENTS].mean().reset_index()
    meta = uw.groupby("文物编号")[["纹饰", "类型", "颜色", "表面风化"]].first()
    g = g.merge(meta, on="文物编号", how="left")
    g["主采样点数"] = 0
    g["成分来源"] = "未风化点回填"

    art2 = art.copy()
    art2["成分来源"] = "主采样点均值"
    full = pd.concat([art2, g], ignore_index=True)
    full = full.sort_values("文物编号").reset_index(drop=True)
    full["成分累加和"] = full[COMPONENTS].sum(axis=1)
    full["检出成分数"] = full[COMPONENTS].notna().sum(axis=1)
    full["是否有效"] = full["成分累加和"].between(SUM_MIN, SUM_MAX)
    return full


def missing_classify(table):
    miss_rate = table[COMPONENTS].isna().mean() * 100
    level = {}
    for col in COMPONENTS:
        r = miss_rate[col]
        level[col] = "高缺失" if r >= HIGH_RATE else ("中缺失" if r >= MID_RATE else "低缺失")
    return level, miss_rate


def build_detected_flags(table):
    flags = table[["文物编号"]].copy()
    for col in COMPONENTS:
        flags["检出_" + col] = table[col].notna().astype(int)
    return flags


def zero_filled(table):
    zf = table.copy()
    zf[COMPONENTS] = zf[COMPONENTS].fillna(0.0)
    return zf


def replace_zero_small(df, comps):
    out = df.copy()
    for col in comps:
        vals = out[col]
        pos = vals[vals > 0]
        small = pos.min() * 0.5 if len(pos) > 0 else 0.001
        out[col] = vals.replace(0.0, small)
    return out


def alr_transform(df, comps, ref=ALR_REF):
    out = df[["文物编号"]].copy()
    for col in comps:
        if col == ref:
            continue
        out["ALR_" + col] = np.log(df[col] / df[ref])
    return out


def clr_transform(df, comps):
    out = df[["文物编号"]].copy()
    gmean = np.exp(df[comps].apply(np.log).mean(axis=1))
    for col in comps:
        out["CLR_" + col] = np.log(df[col] / gmean)
    return out


def save_derived(table, tag):
    flags = build_detected_flags(table)
    flags.to_csv(os.path.join(OUT_DIR, "表单2_检出标志{}.csv".format(tag)),
                 index=False, encoding="utf-8-sig")
    zf = zero_filled(table)
    zf.to_csv(os.path.join(OUT_DIR, "表单2_填0版{}.csv".format(tag)),
              index=False, encoding="utf-8-sig")
    valid = zf[zf["是否有效"]].copy()
    valid = replace_zero_small(valid, COMPONENTS)
    alr = alr_transform(valid, COMPONENTS)
    clr = clr_transform(valid, COMPONENTS)
    logratio = alr.merge(clr, on="文物编号")
    logratio.to_csv(os.path.join(OUT_DIR, "表单2_对数比变换{}.csv".format(tag)),
                    index=False, encoding="utf-8-sig")
    return flags, zf, logratio


def svr_impute_all(valid):
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    miss_rate = valid[COMPONENTS].isna().mean() * 100
    features = [c for c in COMPONENTS if miss_rate[c] < MID_RATE]
    targets = [c for c in COMPONENTS if MID_RATE <= miss_rate[c] < HIGH_RATE]

    print("  有效文物样本: {} 件 | 特征(低缺失): {} | 插补目标(中缺失): {}".format(
        len(valid), features, targets))

    X_all = valid[features].copy()
    for f in features:
        X_all[f] = X_all[f].fillna(X_all[f].mean())

    out = valid.copy()
    report_rows = []
    for col in targets:
        known = valid[col].notna()
        X_tr = X_all[known].values
        y_tr = valid.loc[known, col].values
        X_te = X_all[~known].values
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        gs = GridSearchCV(SVR(kernel="rbf"), SVR_PARAM_GRID, cv=5,
                          scoring="neg_mean_absolute_error")
        gs.fit(X_tr_s, y_tr)
        best = gs.best_estimator_

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        y_true_all, y_pred_all = [], []
        for tr_idx, va_idx in kf.split(X_tr_s):
            m = SVR(kernel="rbf", **gs.best_params_)
            m.fit(X_tr_s[tr_idx], y_tr[tr_idx])
            y_pred_all.append(m.predict(X_tr_s[va_idx]))
            y_true_all.append(y_tr[va_idx])
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)

        ev = {"MAE": mean_absolute_error(y_true_all, y_pred_all),
              "RMSE": float(np.sqrt(mean_squared_error(y_true_all, y_pred_all))),
              "R2": r2_score(y_true_all, y_pred_all),
              "best_params": gs.best_params_,
              "n_known": int(known.sum()),
              "n_missing": int((~known).sum())}
        imputed = valid[col].copy()
        if (~known).sum() > 0:
            X_te_s = scaler.transform(X_te)
            imputed.loc[~known] = best.predict(X_te_s)
        out[col + "_SVR插补"] = imputed
        report_rows.append({
            "成分": col, "缺失率%": round(miss_rate[col], 1),
            "已检出数": ev["n_known"], "插补数": ev["n_missing"],
            "CV_MAE": round(ev["MAE"], 3), "CV_RMSE": round(ev["RMSE"], 3),
            "CV_R2": round(ev["R2"], 3), "最优参数": str(ev["best_params"])})
        print("    [{}] 缺失率 {:.1f}% | 已检出 {} / 插补 {} | 5折CV R2={:.3f} | 参数 {}".format(
            col, miss_rate[col], ev["n_known"], ev["n_missing"], ev["R2"], ev["best_params"]))
    return out, pd.DataFrame(report_rows)


def full_artifact_table(f1, f2):
    rows = []
    for iid in f1["文物编号"]:
        sub = f2[f2["文物编号"] == iid]
        main = sub[sub["采样点类型"].isin(["普通点", "部位"])]
        special = sub[sub["采样点类型"].isin(SPECIAL_TYPES)]
        if len(main) > 0:
            comp = main[COMPONENTS].mean()
            source, n_main = "主采样点均值", len(main)
        else:
            unw = special[special["采样点类型"] == "未风化点"]
            comp = unw[COMPONENTS].mean() if len(unw) > 0 else special[COMPONENTS].mean()
            source = "未风化点" if len(unw) > 0 else "严重风化点"
            n_main = 0
        rec = {"文物编号": iid, "成分数据来源": source, "主采样点数": n_main}
        for c in COMPONENTS:
            rec[c] = comp[c]
        rows.append(rec)
    art = pd.DataFrame(rows)
    art = art.merge(f1[["文物编号", "类型", "纹饰", "颜色", "表面风化"]],
                    on="文物编号", how="left")
    art["成分累加和"] = art[COMPONENTS].sum(axis=1)
    art["检出成分数"] = art[COMPONENTS].notna().sum(axis=1)
    art["是否有效"] = art["成分累加和"].between(SUM_MIN, SUM_MAX)
    return art


def sample_type_counts(f2):
    cnt = pd.crosstab(f2["文物编号"], f2["采样点类型"])
    for t in ["普通点", "部位", "未风化点", "严重风化点"]:
        if t not in cnt.columns:
            cnt[t] = 0
    cnt = cnt[["普通点", "部位", "未风化点", "严重风化点"]].astype(int).reset_index()
    cnt.columns = ["文物编号", "普通点数", "部位点数", "未风化点数", "严重风化点数"]
    return cnt


def q1_category_table(art, f2):
    cnt = sample_type_counts(f2)
    t = art[["文物编号", "类型", "纹饰", "颜色", "表面风化"]].copy()
    t = t.merge(cnt, on="文物编号", how="left")
    t["颜色是否缺失"] = t["颜色"].isna()
    return t


def q1_component_table(f2):
    f2 = f2.copy()
    f2["风化状态"] = np.select(
        [f2["采样点类型"] == "未风化点",
         f2["采样点类型"] == "严重风化点",
         f2["表面风化"] == "无风化"],
        ["未风化", "严重风化", "无风化文物"],
        default="风化文物")
    return f2


def q1_restore_table(art, special):
    valid = art[art["是否有效"]].copy().reset_index(drop=True)
    severe_ids = set(special.loc[special["采样点类型"] == "严重风化点", "文物编号"])
    valid["风化程度"] = np.select(
        [valid["表面风化"] == "无风化", valid["文物编号"].isin(severe_ids)],
        [0, 2], default=1)
    zf = valid.copy()
    zf[COMPONENTS] = zf[COMPONENTS].fillna(0.0)
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


def main():
    t_all = time.time()

    stage(1, 4, "读取三张表单并清洗（列名/采样点/85%~105%筛选/关联分类信息）")
    f1 = load_form1()
    f2 = load_form2()
    f3 = load_form3()
    print("  表单1 {} 件 | 表单2 {} 个采样点（有效 {}）| 表单3 {} 件（有效 {}）".format(
        len(f1), len(f2), f2["是否有效"].sum(), len(f3), f3["是否有效"].sum()))
    f1.to_csv(os.path.join(OUT_DIR, "表单1_基本信息_clean.csv"),
              index=False, encoding="utf-8-sig")
    f2.to_csv(os.path.join(OUT_DIR, "表单2_成分_clean.csv"),
              index=False, encoding="utf-8-sig")
    f3.to_csv(os.path.join(OUT_DIR, "表单3_未知_clean.csv"),
              index=False, encoding="utf-8-sig")

    stage(2, 4, "文物级聚合与多口径文件")
    art, special = aggregate_artifacts(f2)
    full = build_full_table(f2, art)
    level, miss_rate = missing_classify(art)
    save_derived(art, "_51")
    flags58, zf58, lr58 = save_derived(full, "_全量58")
    art.to_csv(os.path.join(OUT_DIR, "表单2_文物级_clean.csv"),
               index=False, encoding="utf-8-sig")
    full.to_csv(os.path.join(OUT_DIR, "表单2_文物级_全量58.csv"),
                index=False, encoding="utf-8-sig")
    special.to_csv(os.path.join(OUT_DIR, "表单2_特殊点_clean.csv"),
                   index=False, encoding="utf-8-sig")
    print("  文物级主表 {} 件（有效 {}）| 全量 58 件（有效 {}）| 特殊点 {} 条".format(
        len(art), art["是否有效"].sum(), len(full), full["是否有效"].sum(), len(special)))

    stage(3, 4, "SVR 插补（K2O / PbO / BaO）")
    valid = art[art["是否有效"]].copy()
    out_svr, ev_df = svr_impute_all(valid)
    out_svr.to_csv(os.path.join(OUT_DIR, "表单2_文物级_SVR插补.csv"),
                   index=False, encoding="utf-8-sig")
    ev_df.to_csv(os.path.join(OUT_DIR, "SVR插补评估表.csv"),
                 index=False, encoding="utf-8-sig")

    stage(4, 4, "问题 1 准备表")
    art_q1 = full_artifact_table(f1, f2)
    art_q1.to_csv(os.path.join(OUT_DIR, "表单2_文物级_全量.csv"),
                  index=False, encoding="utf-8-sig")
    q1_category_table(art_q1, f2).to_csv(
        os.path.join(Q1_DIR, "Q1_类别属性表.csv"), index=False, encoding="utf-8-sig")
    q1_component_table(f2).to_csv(
        os.path.join(Q1_DIR, "Q1_成分对照表.csv"), index=False, encoding="utf-8-sig")
    q1_restore_table(art_q1, special).to_csv(
        os.path.join(Q1_DIR, "Q1_风化反推表.csv"), index=False, encoding="utf-8-sig")

    print("\n预处理全部完成，输出目录: {}（总耗时 {:.1f}s）".format(OUT_DIR, time.time() - t_all))


if __name__ == "__main__":
    main()
