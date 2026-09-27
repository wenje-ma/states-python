import os
import time
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
REPO_ROOT=os.path.dirname(os.path.dirname(BASE_DIR))
CLEAN_DIR=os.path.join(REPO_ROOT,"2-解答","0-数据预处理","cleaned")
OUT_XLSX=os.path.join(BASE_DIR,"第一问.xlsx")
COMPONENTS=["SiO2","Na2O","K2O","CaO","MgO","Al2O3","Fe2O3","CuO","PbO","BaO","P2O5","SrO","SnO2","SO2"]
ALPHA=0.05
TYPE_CODE={"铅钡":0,"高钾":1}
WEATHER_CODE={"无风化":0,"风化":1}
ORNA_CODE={"A":1,"B":2,"C":3}
COLOR_CODE={"蓝绿系":1,"绿系":2,"黑":3,"紫":4}
COLOR_MAP={"浅蓝":"蓝绿系","深蓝":"蓝绿系","蓝绿":"蓝绿系","浅绿":"绿系","绿":"绿系","深绿":"绿系","黑":"黑","紫":"紫"}
N_PERM=5000
SEED=42
MODEL_COMPS={"高钾":["SiO2","K2O","CaO","Al2O3"],"铅钡":["SiO2","CaO","Al2O3","PbO","P2O5"]}
WEATHER_Y={"高钾":{"一般":0.8,"严重":1.0},"铅钡":{"一般":0.75,"严重":1.0}}
CONTROL_TARGETS=["K2O","SrO","P2O5"]
CONTROL_FEATURES={"高钾":["SiO2","CaO"],"铅钡":["SiO2","CaO","Al2O3","PbO"]}
_t0=time.time()
def stage(no,total,msg):
    print("[{}/{}] {}  （累计耗时 {:.1f}s）".format(no,total,msg,time.time()-_t0))
def chi2_contingency_robust(x,y,n_perm=N_PERM,seed=SEED):
    ct=pd.crosstab(x,y)
    chi2,p,dof,expected=stats.chi2_contingency(ct)
    use_perm=False
    if(expected<5).any():
        use_perm=True
        rng=np.random.default_rng(seed)
        y_arr=y.values
        idx=y.index
        cnt=0
        for _ in range(n_perm):
            y_shuf=pd.Series(rng.permutation(y_arr),index=idx)
            chi2_p=stats.chi2_contingency(pd.crosstab(x,y_shuf))[0]
            if chi2_p>=chi2:
                cnt+=1
        p=(cnt+1)/(n_perm+1)
    return chi2,p,use_perm
def rank_correlations(df,col,code_map):
    sub=df.dropna(subset=[col])
    x=sub[col].map(code_map).astype(float)
    y=sub["表面风化"].map(WEATHER_CODE).astype(float)
    rho,_=stats.spearmanr(x,y)
    tau,_=stats.kendalltau(x,y)
    return rho,tau,len(sub)
def analysis_51(df_form1):
    df=df_form1.copy()
    df["颜色系"]=df["颜色"].map(COLOR_MAP)
    rows=[]
    for col in["类型","纹饰","颜色系"]:
        sub=df.dropna(subset=[col])
        chi2,p,use_perm=chi2_contingency_robust(sub[col],sub["表面风化"])
        rho,tau,n=rank_correlations(df,col,{"类型":TYPE_CODE,"纹饰":ORNA_CODE,"颜色系":COLOR_CODE}[col])
        sig="显著相关" if p<0.05 else "不显著"
        rows.append({"属性":col,"样本数":n,"卡方统计量":round(chi2,3),"P值":round(p,4),"检验方式":"蒙特卡洛置换" if use_perm else "卡方近似","Spearman ρ":round(rho,3),"Kendall τ":round(tau,3),"结论":sig})
        print("  [{}] 卡方={:.3f} P={:.4f}（{}）| ρ={:+.3f} τ={:+.3f} → {}".format(col,chi2,p,"置换" if use_perm else "卡方",rho,tau,sig))
    tab1=pd.DataFrame(rows)
    ct=pd.crosstab(df["类型"],df["表面风化"])
    share={}
    for t in["高钾","铅钡"]:
        share[t]=ct.loc[t,"风化"]/ct.loc[t].sum()*100 if t in ct.index else np.nan
    share_df=pd.DataFrame({"类型":list(share.keys()),"风化占比%":[round(share[t],1) for t in share]})
    return tab1,share_df
def cohen_d(x,y):
    n1,n2=len(x),len(y)
    s1,s2=x.std(ddof=1),y.std(ddof=1)
    sp=np.sqrt(((n1-1)*s1**2+(n2-1)*s2**2)/(n1+n2-2))
    return (x.mean()-y.mean())/sp
def analysis_52(art):
    print("  文物级主表 {} 件（多采样点已聚合）".format(len(art)))
    all_tests=[]
    for t in["高钾","铅钡"]:
        sub=art[art["类型"]==t]
        w=sub[sub["表面风化"]=="风化"]
        u=sub[sub["表面风化"]=="无风化"]
        print("  [{}玻璃] 无风化 {} 件 / 风化 {} 件".format(t,len(u),len(w)))
        for col in COMPONENTS:
            g1=w[col].dropna()
            g0=u[col].dropna()
            if len(g1)<2 or len(g0)<2:
                continue
            u_stat,p=stats.mannwhitneyu(g1,g0,alternative="two-sided")
            d=cohen_d(g1,g0)
            all_tests.append({"类型":t,"成分":col,"无风化均值%":round(g0.mean(),2),"风化均值%":round(g1.mean(),2),"P值":round(p,4),"Cohen_d":round(d,3),"变化方向":"上升" if g1.mean()>g0.mean() else "下降","显著性":"显著" if p<ALPHA else "不显著","n无风化":len(g0),"n风化":len(g1)})
    tests=pd.DataFrame(all_tests)
    sig=tests[tests["显著性"]=="显著"]
    print("  显著成分（P<0.05）: "+", ".join("{}:{}".format(t,c) for t,c in zip(sig["类型"],sig["成分"])))
    PAPER_COMPS={"高钾":["SiO2","K2O","CaO","Al2O3","Fe2O3"],"铅钡":["SiO2","PbO","CaO","P2O5"]}
    paper_rows=[]
    for t,comps in PAPER_COMPS.items():
        for c in comps:
            m=tests[(tests["类型"]==t)&(tests["成分"]==c)]
            if len(m):
                paper_rows.append(m.iloc[0])
    tab52=pd.DataFrame(paper_rows).reset_index(drop=True)
    tab52["P值显示"]=tab52["P值"].apply(lambda p:"<0.001" if p<0.001 else "{:.4f}".format(p))
    return tab52
def build_samples():
    df=pd.read_csv(os.path.join(CLEAN_DIR,"表单2_成分_clean.csv"),encoding="utf-8-sig")
    df=df[df["是否有效"]].copy()
    def assign_y(row):
        t=row["类型"]
        if row["采样点类型"]=="未风化点":
            return 0.0
        if row["采样点类型"]=="严重风化点":
            return WEATHER_Y[t]["严重"]
        if row["表面风化"]=="无风化":
            return 0.0
        return WEATHER_Y[t]["一般"]
    df["Y"]=df.apply(assign_y,axis=1)
    df["组别"]=df["Y"].map({0.0:"未风化",0.75:"一般风化",0.8:"一般风化",1.0:"严重风化"})
    return df
def method_A(df,comps):
    sub=df.dropna(subset=comps).copy()
    X=sm.add_constant(sub[comps])
    model=sm.OLS(sub["Y"],X).fit()
    b=model.params
    w=sub[sub["Y"]>0][COMPONENTS]
    u=sub[sub["Y"]==0][COMPONENTS]
    d_sio2=w["SiO2"].mean()-u["SiO2"].mean()
    ratio=(w.mean()-u.mean())/d_sio2
    denom=b["SiO2"]
    for c in comps:
        if c!="SiO2":
            denom+=b[c]*ratio[c]
    pred_rows=[]
    for _,row in sub[sub["Y"]>0].iterrows():
        d_sio2_est=row["Y"]/denom
        pre={c:row[c]-ratio[c]*d_sio2_est for c in COMPONENTS}
        pred_rows.append({"文物采样点":row["文物采样点"],"文物编号":row["文物编号"],"类型":row["类型"],"组别":row["组别"],"Y":row["Y"],"观测SiO2":row["SiO2"],"预测风化前SiO2":pre["SiO2"],"反推分母":denom,**pre})
    pred=pd.DataFrame(pred_rows)
    summary={"R2":model.rsquared,"系数":{c:round(b[c],4) for c in["const"]+comps},"反应比例":{c:round(ratio[c],4) for c in COMPONENTS},"反推分母":denom}
    return pred,summary
def method_B(df,targets,features):
    sub=df.dropna(subset=features+["SiO2"]).copy()
    sub["WG"]=(sub["Y"]>0).astype(float)
    pred_rows=[]
    summaries={}
    for tgt in targets:
        valid=sub.dropna(subset=[tgt])
        if len(valid)<len(features)+3:
            summaries[tgt]="样本不足，跳过"
            continue
        X=sm.add_constant(valid[features+["WG"]])
        m=sm.OLS(valid[tgt],X).fit()
        summaries[tgt]={"R2":round(m.rsquared,4),"系数":{k:round(v,4) for k,v in m.params.items()}}
        w_rows=valid[valid["Y"]>0]
        Xw=w_rows[features+["WG"]].copy()
        Xw.insert(0,"const",1.0)
        Xw["WG"]=0.0
        yhat=Xw.values@m.params.values
        for(_,row),y in zip(w_rows.iterrows(),yhat):
            pred_rows.append({"文物采样点":row["文物采样点"],"文物编号":row["文物编号"],"类型":row["类型"],"组别":row["组别"],"风化物质":tgt,"观测含量":row[tgt],"预测风化前含量":float(y)})
    return pd.DataFrame(pred_rows),summaries
def analysis_53(df):
    print("  有效采样点 {} 个（未风化 {} / 一般风化 {} / 严重风化 {}）".format(len(df),(df["Y"]==0).sum(),df["Y"].isin([0.75,0.8]).sum(),(df["Y"]==1).sum()))
    func_rows,ratio_rows,pred_notes=[],[],[]
    for t in["高钾","铅钡"]:
        sub=df[df["类型"]==t]
        predA,sumA=method_A(sub,MODEL_COMPS[t])
        print("\n  [{}玻璃] 主线A：R²={:.4f} | 系数 {} | 反推 {} 个风化样本".format(t,sumA["R2"],sumA["系数"],len(predA)))
        row={"类型":t,"截距":sumA["系数"]["const"],"R²":round(sumA["R2"],3)}
        for c in COMPONENTS:
            row[c]=sumA["系数"].get(c,"")
        func_rows.append(row)
        for c in MODEL_COMPS[t]:
            ratio_rows.append({"类型":t,"成分":c,"反应比例":sumA["反应比例"][c]})
        print("    反应比例: "+", ".join("{}={}".format(c,sumA["反应比例"][c]) for c in MODEL_COMPS[t]))
        if len(predA):
            u_mean=sub[sub["Y"]==0][COMPONENTS].mean()
            u_std=sub[sub["Y"]==0][COMPONENTS].std()
            p_mean=predA[COMPONENTS].mean()
            check=pd.DataFrame({"成分":COMPONENTS,"预测均值":p_mean.values,"无风化均值":u_mean.values,"无风化标准差":u_std.values})
            check=check.dropna()
            check["标准化偏差"]=((check["预测均值"]-check["无风化均值"]).abs()/check["无风化标准差"]).round(3)
            miss=sub[COMPONENTS].isna().mean()*100
            high_miss=set(miss[miss>=50].index)
            ok_all=(check["标准化偏差"]<1).all()
            ok_low=check[~check["成分"].isin(high_miss)]["标准化偏差"].lt(1).all()
            print("    标准化偏差：全部 {}；除高缺失组分外 {} （论文口径）".format("通过" if ok_all else "未通过","通过" if ok_low else "未通过"))
            if t=="高钾":
                sio2_lo,sio2_hi=predA["预测风化前SiO2"].min(),predA["预测风化前SiO2"].max()
                k2o_lo,k2o_hi=predA["K2O"].min(),predA["K2O"].max()
                pred_notes.append({"类型":t,"指标":"预测风化前SiO2区间%","值":"{:.1f}~{:.1f}".format(sio2_lo,sio2_hi),"说明":"论文 5.3：{} 个{}风化采样点".format(len(predA),t)})
                pred_notes.append({"类型":t,"指标":"预测风化前K2O区间%","值":"{:.1f}~{:.1f}".format(k2o_lo,k2o_hi),"说明":"论文 5.3 报告值"})
                print("    预测区间：SiO2 {:.1f}~{:.1f} | K2O {:.1f}~{:.1f}".format(sio2_lo,sio2_hi,k2o_lo,k2o_hi))
            p08=predA[(predA["文物编号"]==8)&(predA["Y"]==1.0)]
            if len(p08):
                v=p08["预测风化前SiO2"].iloc[0]
                pred_notes.append({"类型":t,"指标":"严重风化样本08原始SiO2预测%","值":"{:.2f}".format(v),"说明":"论文 5.3：采样点 08严重风化点"})
                print("    严重风化样本08 原始SiO2预测 = {:.2f}%".format(v))
    print("\n  [对照B] 风化虚拟变量回归（WG=0 得风化前预测）：")
    PAPER_B={("高钾","K2O"),("铅钡","P2O5")}
    for t in["高钾","铅钡"]:
        _,sumB=method_B(df[df["类型"]==t],CONTROL_TARGETS,CONTROL_FEATURES[t])
        for k,v in sumB.items():
            if isinstance(v,dict) and"R2" in v:
                print("    {} {}: R²={} WG系数={}".format(t,k,v["R2"],v["系数"].get("WG")))
                if (t,k) in PAPER_B:
                    pred_notes.append({"类型":t,"指标":"对照B {}模型".format(k),"值":"R²={}，WG系数={}".format(v["R2"],v["系数"].get("WG")),"说明":"论文 5.3：风化虚拟变量回归对照"})
    return (pd.DataFrame(func_rows),pd.DataFrame(ratio_rows),pd.DataFrame(pred_notes))
def main():
    t_all=time.time()
    stage(1,3,"5.1 表面风化与类型/纹饰/颜色的关系")
    f1=pd.read_csv(os.path.join(CLEAN_DIR,"表单1_基本信息_clean.csv"),encoding="utf-8-sig")
    tab1,share_df=analysis_51(f1)
    stage(2,3,"5.2 风化前后化学成分统计规律")
    art=pd.read_csv(os.path.join(CLEAN_DIR,"表单2_文物级_clean.csv"),encoding="utf-8-sig")
    tab52=analysis_52(art)
    stage(3,3,"5.3 风化前化学成分预测（主线A + 对照B）")
    df53=build_samples()
    func_df,ratio_df,pred_df=analysis_53(df53)
    with pd.ExcelWriter(OUT_XLSX,engine="openpyxl") as w:
        tab1.to_excel(w,sheet_name="表1-关联检验",index=False)
        share_df.to_excel(w,sheet_name="5.1-风化占比",index=False)
        tab52.to_excel(w,sheet_name="5.2-风化前后显著成分",index=False)
        func_df.to_excel(w,sheet_name="5.3-风化程度函数",index=False)
        ratio_df.to_excel(w,sheet_name="5.3-反应比例",index=False)
        pred_df.to_excel(w,sheet_name="5.3-预测与对照",index=False)
    from openpyxl import load_workbook
    wb=load_workbook(OUT_XLSX)
    for ws in wb.worksheets:
        for col in ws.columns:
            width=max(len(str(c.value))*2.2 if c.value is not None else 0 for c in col)+2
            ws.column_dimensions[col[0].column_letter].width=min(width,40)
    wb.save(OUT_XLSX)
    print("\n问题 1 完成：输出 {} （总耗时 {:.1f}s）".format(OUT_XLSX,time.time()-t_all))
if __name__=="__main__":
    main()
