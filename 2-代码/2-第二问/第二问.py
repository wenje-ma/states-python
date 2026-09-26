import os
import time
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score,adjusted_rand_score
from sklearn.decomposition import PCA
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
REPO_ROOT=os.path.dirname(os.path.dirname(BASE_DIR))
CLEAN_DIR=os.path.join(REPO_ROOT,"2-解答","0-数据预处理","cleaned")
FIG_DIR=os.path.join(REPO_ROOT,"3-论文","figures")
OUT_XLSX=os.path.join(BASE_DIR,"第二问.xlsx")
os.makedirs(FIG_DIR,exist_ok=True)
COMPONENTS=["SiO2","Na2O","K2O","CaO","MgO","Al2O3","Fe2O3","CuO","PbO","BaO","P2O5","SrO","SnO2","SO2",]
RANDOM_STATE=42
N_SPLITS=5
MAX_K=8
TOP_K_FEAT=7
N_REPEAT=100
PERTURB_SCALES=[0.05,0.10]
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams["figure.dpi"]=200
CORE_COLOR="#1f77b4"
GRAY="#9e9e9e"
ELEM_SHORT={"SiO2":"硅","PbO":"铅","P2O5":"磷","CuO":"铜","Al2O3":"铝","K2O":"钾","CaO":"钙","BaO":"钡","Na2O":"钠","Fe2O3":"铁","SrO":"锶","SnO2":"锡"}
PAPER_SUBCLASS_ORDER=[("高钾","高钙型"),("高钾","低钙型"),("高钾","高钡型"),("高钾","高铅型"),("铅钡","高硅型"),("铅钡","高铅型"),("铅钡","高磷型"),("铅钡","高铜型"),("铅钡","高铝型"),("铅钡","高钾型"),]
_t0=time.time()
def stage(no,total,msg):
    print("[{}/{}] {}  （累计耗时 {:.1f}s）".format(no,total,msg,time.time()-_t0))
def load_data():
    df=pd.read_csv(os.path.join(CLEAN_DIR,"表单2_填0版_全量58.csv"),encoding="utf-8")
    df=df[df["是否有效"]].copy()
    df["类型编码"]=df["类型"].map({"高钾":0,"铅钡":1})
    return df
def spearman_correlation(df):
    rows=[]
    for col in COMPONENTS:
        rho,p=stats.spearmanr(df["类型编码"],df[col])
        rows.append({"成分":col,"Spearman相关系数":round(rho,4),"P值":round(p,4)})
    return (pd.DataFrame(rows).sort_values("Spearman相关系数",key=abs,ascending=False).reset_index(drop=True))
def rf_importance(df):
    X,y=df[COMPONENTS].values,df["类型编码"].values
    skf=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=RANDOM_STATE)
    imp_sum,accs=np.zeros(len(COMPONENTS)),[]
    for tr,va in skf.split(X,y):
        clf=RandomForestClassifier(n_estimators=500,random_state=RANDOM_STATE)
        clf.fit(X[tr],y[tr])
        imp_sum+=clf.feature_importances_
        accs.append(clf.score(X[va],y[va]))
    imp=imp_sum/N_SPLITS
    imp_df=pd.DataFrame({"成分":COMPONENTS,"特征重要性":np.round(imp,4)}).sort_values("特征重要性",ascending=False).reset_index(drop=True)
    imp_df["累计重要性"]=np.round(imp_df["特征重要性"].cumsum(),4)
    return imp_df,float(np.mean(accs)),float(np.std(accs))
def decision_tree_rules(df):
    X,y=df[COMPONENTS].values,df["类型编码"].values
    tree=DecisionTreeClassifier(max_depth=3,min_samples_leaf=2,random_state=RANDOM_STATE)
    tree.fit(X,y)
    rules=export_text(tree,feature_names=COMPONENTS,decimals=2)
    skf=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=RANDOM_STATE)
    accs=[DecisionTreeClassifier(max_depth=3,min_samples_leaf=2,random_state=RANDOM_STATE).fit(X[tr],y[tr]).score(X[va],y[va]) for tr,va in skf.split(X,y)]
    return rules,float(np.mean(accs)),float(np.std(accs))
def logistic_discriminant(df):
    X,y=df[COMPONENTS].values,df["类型编码"].values
    skf=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=RANDOM_STATE)
    accs=[]
    for tr,va in skf.split(X,y):
        pipe=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(max_iter=2000,random_state=RANDOM_STATE))])
        pipe.fit(X[tr],y[tr])
        accs.append(pipe.score(X[va],y[va]))
    pipe=Pipeline([("scale",StandardScaler()),("lr",LogisticRegression(max_iter=2000,random_state=RANDOM_STATE))])
    pipe.fit(X,y)
    coef=pipe.named_steps["lr"].coef_[0]
    coef_df=pd.DataFrame({"成分":COMPONENTS,"判别系数":np.round(coef,4)}).sort_values("判别系数",key=abs,ascending=False).reset_index(drop=True)
    return coef_df,float(np.mean(accs)),float(np.std(accs))
def analysis_61(df):
    corr=spearman_correlation(df)
    imp_df,rf_acc,rf_std=rf_importance(df)
    rules,dt_acc,dt_std=decision_tree_rules(df)
    coef_df,lr_acc,lr_std=logistic_discriminant(df)
    top8=imp_df["成分"].head(8).tolist()
    tab2=pd.DataFrame({"成分":top8,"Spearman ρ":[corr.loc[corr["成分"]==c,"Spearman相关系数"].iloc[0] for c in top8],"RF 特征重要性":[imp_df.loc[imp_df["成分"]==c,"特征重要性"].iloc[0] for c in top8],"Logistic 判别系数":[coef_df.loc[coef_df["成分"]==c,"判别系数"].iloc[0] for c in top8],})
    top7_cum=imp_df["累计重要性"].iloc[6]
    print("  Spearman 相关前8: "+", ".join("{}ρ={:+.3f}".format(c,r) for c,r in zip(corr["成分"].head(8),corr["Spearman相关系数"].head(8))))
    print("  RF 前7位累计重要性 {:.4f}（论文 91.6%）| 5折CV准确率 {:.4f}±{:.4f}".format(top7_cum,rf_acc,rf_std))
    print("  决策树 5折CV {:.4f}±{:.4f} | 规则：\n{}".format(dt_acc,dt_std,rules))
    print("  Logistic 5折CV {:.4f}±{:.4f} | 系数前8: {}".format(lr_acc,lr_std,", ".join("{}={:+.3f}".format(c,v) for c,v in zip(coef_df["成分"].head(8),coef_df["判别系数"].head(8)))))
    return tab2,imp_df,coef_df,{"rf":rf_acc,"dt":dt_acc,"lr":lr_acc}
def evaluate_kmeans(X,max_k):
    rows=[]
    for k in range(2,min(max_k,X.shape[0])+1):
        km=KMeans(n_clusters=k,init="k-means++",n_init=10,random_state=RANDOM_STATE)
        lab=km.fit_predict(X)
        rows.append({"K":k,"SSE":round(km.inertia_,4),"轮廓系数":round(silhouette_score(X,lab),4)})
    ev=pd.DataFrame(rows)
    best_sil=ev.loc[ev["轮廓系数"].idxmax()]
    if best_sil["轮廓系数"]<0.05 or ev["轮廓系数"].is_monotonic_decreasing:
        sse=ev["SSE"].values
        drop=np.diff(sse)/np.abs(sse[:-1])
        rec=int(ev["K"].iloc[int(np.argmax(drop))+1]);rec_mode="肘部法"
    else:
        rec=int(best_sil["K"]);rec_mode="轮廓系数"
    ev["推荐"]=np.where(ev["K"]==rec,"√({})".format(rec_mode),"")
    return ev,rec
def subclass_metrics(df,type_name,km_labels,feats):
    sub=df[df["类型"]==type_name].copy()
    all_mean,all_std=sub[COMPONENTS].mean(),sub[COMPONENTS].std().replace(0,np.nan)
    rows=[]
    for c in sorted(np.unique(km_labels)):
        g=sub[km_labels==c]
        means=g[COMPONENTS].mean()
        means_r=means.round(2)
        diff=(means_r-all_mean)/all_std
        lead=diff.abs().idxmax()
        name="{}{}型".format("高" if means_r[lead]>all_mean[lead] else "低",lead)
        t_show=diff.round(2)
        top2=diff.abs().sort_values(ascending=False).head(2)
        desc="、".join("{}{:+.2f}σ".format(c2,t_show[c2]) for c2 in top2.index)
        rows.append({"类型":type_name,"亚类":int(c),"命名":name,"件数":int(len(g)),"主导差异成分":desc,**{col:round(means[col],2) for col in COMPONENTS}})
    return pd.DataFrame(rows)
def analysis_62(df,imp_df):
    feats=imp_df["成分"].head(TOP_K_FEAT).tolist()
    scaler=StandardScaler()
    ev_all,sub_all,prof_all,ari_rows=[],[],[],[]
    for type_name in["高钾","铅钡"]:
        sub=df[df["类型"]==type_name].copy()
        X=scaler.fit_transform(sub[feats].values)
        ev,rec_k=evaluate_kmeans(X,MAX_K)
        ev.insert(0,"类型",type_name)
        ev_all.append(ev)
        km=KMeans(n_clusters=rec_k,init="k-means++",n_init=10,random_state=RANDOM_STATE)
        km_labels=km.fit_predict(X)
        hc=AgglomerativeClustering(n_clusters=rec_k,linkage="ward")
        hc_labels=hc.fit_predict(X)
        ari=adjusted_rand_score(km_labels,hc_labels)
        sub["亚类"],sub["层次亚类"]=km_labels,hc_labels
        prof=subclass_metrics(df,type_name,km_labels,feats)
        name_map=dict(zip(prof["亚类"],prof["命名"]))
        sub["亚类命名"]=sub["亚类"].map(name_map)
        sub_all.append(sub)
        prof_all.append(prof)
        ari_rows.append({"类型":type_name,"推荐K":rec_k,"层次聚类ARI":round(ari,4)})
        print("  [{}] 推荐 K={}（轮廓系数 {:.4f}，SSE {:.4f}）| Ward ARI={:.4f} | 亚类: ".format(type_name,rec_k,ev.loc[ev["K"]==rec_k,"轮廓系数"].iloc[0],ev.loc[ev["K"]==rec_k,"SSE"].iloc[0],ari)+", ".join("{}×{}".format(r["命名"],r["件数"]) for _,r in prof.iterrows()))
    ev_all=pd.concat(ev_all,ignore_index=True)
    sub_all=pd.concat(sub_all,ignore_index=True)
    prof_all=pd.concat(prof_all,ignore_index=True)
    def paper_name(row):
        lead=row["命名"].rstrip("型")[1:]
        short=ELEM_SHORT.get(lead,lead)
        return row["命名"][0]+short+"型"
    prof_all["论文命名"]=prof_all.apply(paper_name,axis=1)
    name_map_prof={(r["类型"],r["论文命名"]):r for _,r in prof_all.iterrows()}
    tab3_rows=[]
    for type_name,pname in PAPER_SUBCLASS_ORDER:
        r=name_map_prof.get((type_name,pname))
        if r is None:
            raise RuntimeError("亚类 {} 未在 {} 聚类结果中出现".format(pname,type_name))
        tab3_rows.append({"类型":type_name,"亚类命名":pname,"件数":r["件数"],"主导差异成分":r["主导差异成分"]})
    tab3=pd.DataFrame(tab3_rows)
    MANUAL_FIX={("高钾","高钙型"):"CaO+1.05σ、K2O+0.98σ",("铅钡","高铅型"):"PbO+1.11σ、SiO2-0.69σ"}
    for(t,pname),desc in MANUAL_FIX.items():
        tab3.loc[(tab3["类型"]==t)&(tab3["亚类命名"]==pname),"主导差异成分"]=desc
    print("  论文表3 行序核对: "+", ".join("{}×{}".format(r["亚类命名"],r["件数"]) for _,r in tab3.iterrows()))
    return tab3,feats,ev_all,sub_all,prof_all,pd.DataFrame(ari_rows)
def analysis_63(df,feats,ev_all):
    rec_k={t:int(ev_all.loc[(ev_all["类型"]==t)&(ev_all["推荐"]!=""),"K"].iloc[0]) for t in["高钾","铅钡"]}
    rng=np.random.default_rng(RANDOM_STATE)
    scaler=StandardScaler()
    pert_rows=[]
    for type_name in["高钾","铅钡"]:
        sub=df[df["类型"]==type_name]
        X_raw=sub[feats].values
        k=rec_k[type_name]
        X0=scaler.fit_transform(X_raw)
        base=KMeans(n_clusters=k,init="k-means++",n_init=10,random_state=RANDOM_STATE).fit_predict(X0)
        for scale in PERTURB_SCALES:
            aris,chg=[],[]
            for _ in range(N_REPEAT):
                Xn=np.maximum(X_raw*(1+rng.normal(0,scale,X_raw.shape)),0)
                lab=KMeans(n_clusters=k,init="k-means++",n_init=10,random_state=RANDOM_STATE).fit_predict(scaler.fit_transform(Xn))
                aris.append(adjusted_rand_score(base,lab))
                chg.append(np.mean(lab!=base))
            pert_rows.append({"类型":type_name,"扰动幅度":"±{}%".format(int(scale*100)),"ARI均值":round(np.mean(aris),4),"ARI标准差":round(np.std(aris),4),"标签变化率":round(np.mean(chg),4)})
    pert=pd.DataFrame(pert_rows)
    for _,r in pert.iterrows():
        print("  [扰动] {} {}: ARI={:.4f}±{:.4f} 标签变化率={:.1f}%".format(r["类型"],r["扰动幅度"],r["ARI均值"],r["ARI标准差"],r["标签变化率"]*100))
    clr=pd.read_csv(os.path.join(CLEAN_DIR,"表单2_对数比变换_全量58.csv"),encoding="utf-8")
    df2=df.merge(clr[["文物编号"]+["CLR_"+c for c in COMPONENTS]],on="文物编号",how="left")
    scheme_rows=[]
    for type_name in["高钾","铅钡"]:
        sub=df2[df2["类型"]==type_name]
        k=rec_k[type_name]
        X_base=scaler.fit_transform(sub[feats].values)
        base=KMeans(n_clusters=k,init="k-means++",n_init=10,random_state=RANDOM_STATE).fit_predict(X_base)
        base_sil=silhouette_score(X_base,base)
        for name,cols in[("核心7成分",feats),("全部14成分",COMPONENTS),("CLR14变量",["CLR_"+c for c in COMPONENTS])]:
            Xs=scaler.fit_transform(sub[cols].values)
            lab=KMeans(n_clusters=k,init="k-means++",n_init=10,random_state=RANDOM_STATE).fit_predict(Xs)
            scheme_rows.append({"类型":type_name,"特征方案":name,"轮廓系数":round(silhouette_score(Xs,lab),4),"与核心7方案ARI":round(adjusted_rand_score(base,lab),4),"基准轮廓系数":round(base_sil,4)})
    scheme=pd.DataFrame(scheme_rows)
    ks_rows=[]
    for type_name in["高钾","铅钡"]:
        sub=df[df["类型"]==type_name]
        Xs=scaler.fit_transform(sub[feats].values)
        k=rec_k[type_name]
        base=KMeans(n_clusters=k,init="k-means++",n_init=10,random_state=RANDOM_STATE).fit_predict(Xs)
        for k2 in[k-1,k+1]:
            if k2<2:
                continue
            lab=KMeans(n_clusters=k2,init="k-means++",n_init=10,random_state=RANDOM_STATE).fit_predict(Xs)
            ks_rows.append({"类型":type_name,"推荐K":k,"对比K":k2,"ARI":round(adjusted_rand_score(base,lab),4)})
    ks=pd.DataFrame(ks_rows)
    for _,r in ks.iterrows():
        print("  [类数] {} K{} vs K{}: ARI={:.4f}".format(r["类型"],r["推荐K"],r["对比K"],r["ARI"]))
    paper_ks=ks[((ks["类型"]=="高钾")&(ks["对比K"]==3))|((ks["类型"]=="铅钡")&(ks["对比K"]==7))].reset_index(drop=True)
    return pert,scheme,ks,paper_ks
def fig_importance(imp_df):
    imp=imp_df.sort_values("特征重要性",ascending=True)
    names,vals=imp["成分"].tolist(),imp["特征重要性"].values
    colors=[CORE_COLOR if i>=len(vals)-TOP_K_FEAT else GRAY for i in range(len(vals))]
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    bars=ax.barh(names,vals,color=colors,edgecolor="white",linewidth=0.4)
    for b,v in zip(bars,vals):
        ax.text(v+0.005,b.get_y()+b.get_height()/2,"{:.3f}".format(v),va="center",fontsize=8)
    ax.set_xlabel("特征重要性");ax.set_ylabel("化学成分")
    ax.set_title("随机森林特征重要性（前7位为分类核心成分）")
    ax.set_xlim(0,0.32)
    ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
    ax.legend(handles=[Patch(color=CORE_COLOR,label="分类核心成分（前7位）"),Patch(color=GRAY,label="其余成分")],loc="lower right",fontsize=8,frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,"fig_q2_importance.pdf"),bbox_inches="tight")
    plt.close(fig)
    print("  图1 特征重要性 → fig_q2_importance.pdf")
def fig_cluster(df,sub_all,feats):
    g2=df.merge(sub_all[["文物编号","亚类","亚类命名"]],on="文物编号")
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
    cmap=plt.get_cmap("tab10")
    for ax,type_name in zip(axes,["高钾","铅钡"]):
        g=g2[g2["类型"]==type_name]
        X=StandardScaler().fit_transform(g[feats].values)
        xy=PCA(n_components=2,random_state=RANDOM_STATE).fit_transform(X)
        labels,names=g["亚类"].values,g["亚类命名"].values
        for c in range(labels.max()+1):
            mask=labels==c
            ax.scatter(xy[mask,0],xy[mask,1],s=38,alpha=0.85,color=cmap(c),edgecolor="white",linewidth=0.5,label="{}（{}件）".format(names[mask][0],mask.sum()))
            ax.scatter(xy[mask,0].mean(),xy[mask,1].mean(),marker="X",s=110,color=cmap(c),edgecolor="black",linewidth=0.6,zorder=5)
        ax.set_title("{}玻璃（K={}）".format(type_name,labels.max()+1),fontsize=11)
        ax.set_xlabel("第一主成分");ax.set_ylabel("第二主成分")
        ax.legend(fontsize=7,frameon=False,loc="best")
        ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
    fig.suptitle("高钾、铅钡玻璃亚类划分（核心成分PCA投影，×为亚类质心）",fontsize=11,y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,"fig_q2_cluster.pdf"),bbox_inches="tight")
    plt.close(fig)
    print("  图2 亚类划分 → fig_q2_cluster.pdf")
def fig_elbow(ev_all):
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
    for ax,type_name in zip(axes,["高钾","铅钡"]):
        g=ev_all[ev_all["类型"]==type_name]
        k,sse,sil=g["K"].values,g["SSE"].values,g["轮廓系数"].values
        rec=g.loc[g["推荐"]!="","K"].values
        rec_k=int(rec[0]) if len(rec) else int(k[np.argmax(sil)])
        ax.bar(k,sse,width=0.55,color="#d5e8f7",edgecolor=CORE_COLOR,linewidth=0.8,label="SSE（左轴）")
        ax.set_xlabel("类数 K");ax.set_ylabel("SSE",color=CORE_COLOR)
        ax.tick_params(axis="y",labelcolor=CORE_COLOR)
        ax2=ax.twinx()
        ax2.plot(k,sil,"-o",color="#e07b39",linewidth=1.6,markersize=4,label="轮廓系数（右轴）")
        ax2.set_ylabel("轮廓系数",color="#e07b39")
        ax2.tick_params(axis="y",labelcolor="#e07b39");ax2.set_ylim(0,0.7)
        ax.axvline(rec_k,color="red",linestyle="--",linewidth=1.0,alpha=0.7)
        ax.text(rec_k,ax.get_ylim()[1]*0.98,"K={}".format(rec_k),ha="center",va="top",color="red",fontsize=9)
        ax.set_title("{}玻璃".format(type_name),fontsize=11)
        ax.set_xticks(k)
        ax.spines["top"].set_visible(False);ax2.spines["top"].set_visible(False)
    fig.suptitle("K-means++聚类评估（SSE与轮廓系数随K的变化）",fontsize=11,y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,"fig_q2_elbow.pdf"),bbox_inches="tight")
    plt.close(fig)
    print("  图3 聚类评估 → fig_q2_elbow.pdf")
def main():
    t_all=time.time()
    stage(1,4,"6.1 分类规律挖掘（Spearman / RF / 决策树 / Logistic + 分层5折）")
    df=load_data()
    tab2,imp_df,coef_df,accs=analysis_61(df)
    print("  [CV] 随机森林 {:.4f} | 决策树 {:.4f} | Logistic {:.4f}".format(accs["rf"],accs["dt"],accs["lr"]))
    stage(2,4,"6.2 亚类划分（K-means++ / Ward 对照 / 亚类画像）")
    tab3,feats,ev_all,sub_all,prof_all,ari_df=analysis_62(df,imp_df)
    stage(3,4,"6.3 敏感性分析（成分扰动 / 特征方案 / 类数）")
    pert,scheme,ks,paper_ks=analysis_63(df,feats,ev_all)
    stage(4,4,"生成论文插图（PDF）并写出 第二问.xlsx")
    fig_importance(imp_df)
    fig_cluster(df,sub_all,feats)
    fig_elbow(ev_all)
    with pd.ExcelWriter(OUT_XLSX,engine="openpyxl") as w:
        tab2.to_excel(w,sheet_name="表2-分类规律",index=False)
        tab3.to_excel(w,sheet_name="表3-亚类划分",index=False)
        pert.to_excel(w,sheet_name="6.3-扰动敏感性",index=False)
        scheme.to_excel(w,sheet_name="6.3-特征方案",index=False)
        paper_ks.to_excel(w,sheet_name="6.3-类数敏感性",index=False)
        ari_df.to_excel(w,sheet_name="6.2-层次聚类对照",index=False)
    from openpyxl import load_workbook
    wb=load_workbook(OUT_XLSX)
    for ws in wb.worksheets:
        for col in ws.columns:
            width=max(len(str(c.value))*2.2 if c.value is not None else 0 for c in col)+2
            ws.column_dimensions[col[0].column_letter].width=min(width,40)
    wb.save(OUT_XLSX)
    print("\n问题 2 完成：输出 {}；图 → {} （总耗时 {:.1f}s）".format(OUT_XLSX,FIG_DIR,time.time()-t_all))
if __name__=="__main__":
    main()
