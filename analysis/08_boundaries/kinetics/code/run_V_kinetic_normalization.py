from __future__ import annotations
import json, itertools
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, rankdata

HERE=Path(__file__).resolve().parent
ROUND=HERE.parents[1]; ROOT=ROUND.parent
Q=REPO/"analysis/03_awinda/frequency"; SRC=REPO/"data/derived/awinda_source"
UOUT=REPO/"analysis/03_awinda/continuum/results"
OUT=HERE.parent/"results"; FIG=HERE.parent/"figures"
OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
TARGETS=[0.05,0.10,0.25,0.50,1.00,2.50]
SEED=20261002; N_PERM=50000

def read_kinetics():
    kin=pd.read_csv(SRC/"Noise_analysis_ABC_GWN_WithFits_DQ1_ReFits_2020_08_04_Ver_2024.csv")
    force=pd.read_csv(SRC/"Transgenic_mice_Fiber_Tensions_with_Fits_DQ1_2019_09_06.csv")
    link=force[["AnimalID","Fiber","Mutation","Treatment"]].drop_duplicates()
    if link["Fiber"].duplicated().any(): raise ValueError("Fibre map is not one-to-one")
    kin=kin.loc[kin["pCa"].eq(4.8)&pd.to_numeric(kin["Quality"],errors="coerce").eq(1)].copy()
    kin["ATP (mM)"]=pd.to_numeric(kin["ATP (mM)"],errors="coerce")
    for col in ["b (Hz)","c (Hz)"]: kin[col]=pd.to_numeric(kin[col],errors="coerce")
    kin=kin[kin["ATP (mM)"].isin(TARGETS)].merge(link[["AnimalID","Fiber","Mutation","Treatment"]],
         left_on="Filename",right_on="Fiber",how="left",suffixes=("_kin","_link"),validate="many_to_one")
    if kin["AnimalID"].isna().any(): raise ValueError("Kinetic rows have unresolved animal linkage")
    if not kin["Mutation_kin"].eq(kin["Mutation_link"]).all(): raise ValueError("Mutation mismatch")
    kin["drug_state"] = np.where(kin["Treatment_kin"].eq("Control"), "Control", "0.3_uM_mavacamten")
    link_state = np.where(kin["Treatment_link"].eq("Control"), "Control", "0.3_uM_mavacamten")
    if not np.array_equal(kin["drug_state"].to_numpy(), link_state): raise ValueError("Drug-state mismatch after label normalization")
    if kin[["b (Hz)","c (Hz)"]].isna().any().any() or (kin[["b (Hz)","c (Hz)"]]<=0).any().any():
        raise ValueError("Missing or nonpositive b/c characteristic frequency")
    if kin.groupby(["Filename","ATP (mM)"]).size().max()!=1:
        raise ValueError("More than one active fit per fibre-target; resolve before aggregation")
    fibre=kin.groupby(["AnimalID","Filename","Mutation_kin","drug_state","ATP (mM)"],as_index=False).agg(
        b_Hz=("b (Hz)","mean"),c_Hz=("c (Hz)","mean"))
    cells=fibre.groupby(["AnimalID","Mutation_kin","drug_state","ATP (mM)"],as_index=False).agg(
        n_fibres=("Filename","nunique"),b_Hz=("b_Hz","mean"),c_Hz=("c_Hz","mean"),
        sd_b_Hz=("b_Hz","std"),sd_c_Hz=("c_Hz","std"))
    cells=cells.rename(columns={"Mutation_kin":"genotype","ATP (mM)":"target_ATP_mM"})
    counts=kin.groupby("ATP (mM)").agg(rows=("Filename","size"),fibres=("Filename","nunique"),animals=("AnimalID","nunique")).reset_index()
    if len(counts)!=6 or not counts["fibres"].eq(62).all() or not counts["animals"].eq(10).all():
        raise ValueError("V c/b target coverage differs from frozen complete hierarchy")
    return kin,cells

def cv_models(data):
    # Leave one mouse out. Outcome utility rows are all 72 frequencies x six targets.
    mice=sorted(data.heldout_mouse.astype(str).unique()); targets=sorted(data.target_ATP_mM.unique())
    f=np.log(data.frequency_Hz.to_numpy(float))
    x=np.log(data.frequency_Hz.to_numpy(float)/data.c_Hz.to_numpy(float))
    xb=np.log(data.frequency_Hz.to_numpy(float)/data.b_Hz.to_numpy(float))
    t=data.target_ATP_mM.to_numpy(float)
    dummies=np.column_stack([(t==tt).astype(float) for tt in targets[1:]])
    base=np.column_stack([np.ones(len(data)),dummies])
    models={"H1_target_only":base,
            "H0_absolute_frequency":np.column_stack([base,f,f*f]),
            "H3_detachment_normalized":np.column_stack([base,x,x*x]),
            "H3_recruitment_normalized":np.column_stack([base,xb,xb*xb])}
    y=data.gain_pp.to_numpy(float); rec=[]
    for name,X in models.items():
        pred=np.full(len(y),np.nan)
        for mouse in mice:
            te=data.heldout_mouse.astype(str).to_numpy()==mouse; tr=~te
            coef=np.linalg.lstsq(X[tr],y[tr],rcond=None)[0]
            pred[te]=X[te]@coef
        err=y-pred
        rec.append({"model":name,"LOMO_RMSE_pp":float(np.sqrt(np.mean(err**2))),
                    "LOMO_MAE_pp":float(np.mean(np.abs(err))),
                    "LOMO_R2":float(1-np.sum(err**2)/np.sum((y-y.mean())**2)),
                    "predicted_rows":int(np.isfinite(pred).sum())})
        data[f"cv_pred_{name}"]=pred
    # Full pooled coefficient/vertex only for the primary c-normalized explanatory model.
    X=models["H3_detachment_normalized"]; coef=np.linalg.lstsq(X,y,rcond=None)[0]
    b1,b2=coef[-2],coef[-1]
    vertex=float(-b1/(2*b2)) if b2<0 else np.nan
    xlo=float(x.min()); xhi=float(x.max())
    info={"quadratic_curvature":float(b2),"linear_term":float(b1),"log_frequency_ratio_vertex":vertex,
          "observed_log_ratio_range":[xlo,xhi],"vertex_within_observed_range":bool(np.isfinite(vertex) and xlo<=vertex<=xhi),
          "concave_down":bool(b2<0),"interpretation":"Explanatory only: target c is fitted from that same held-out future spectrum."}
    return pd.DataFrame(rec),info,data

def main():
    kin,cells=read_kinetics()
    cells.to_csv(OUT/"V_TARGET_KINETIC_TIMESCALES.csv",index=False,encoding="utf-8-sig")
    mice=cells.groupby(["AnimalID","target_ATP_mM"],as_index=False).agg(
       n_state_cells=("genotype","size"),c_Hz=("c_Hz","mean"),b_Hz=("b_Hz","mean"),
       sd_cell_c_Hz=("c_Hz","std"),sd_cell_b_Hz=("b_Hz","std"))
    mice=mice.rename(columns={"AnimalID":"heldout_mouse"})
    u=pd.read_csv(UOUT/"U_MGATP_CONTINUUM_SELECTIONS.csv")
    select=u.merge(mice,on=["heldout_mouse","target_ATP_mM"],how="inner",validate="one_to_one")
    if len(select)!=60: raise ValueError(f"Expected 60 mouse-target selections, got {len(select)}")
    select["xi_c"]=select["selected_frequency_Hz"]/select["c_Hz"]
    select["d_c"]=np.abs(np.log(select["xi_c"]))
    select["xi_b"]=select["selected_frequency_Hz"]/select["b_Hz"]
    select["d_b"]=np.abs(np.log(select["xi_b"]))
    select.to_csv(OUT/"V_SELECTED_NORMALIZED_FREQUENCY.csv",index=False,encoding="utf-8-sig")
    trends=[]; bymouse={}
    for m,s in select.groupby("heldout_mouse"):
        rho_c=float(spearmanr(np.log(s["selected_frequency_Hz"]),np.log(s["c_Hz"])).statistic)
        rho_b=float(spearmanr(np.log(s["selected_frequency_Hz"]),np.log(s["b_Hz"])).statistic)
        tau_c=float(kendalltau(np.log(s["selected_frequency_Hz"]),np.log(s["c_Hz"])).statistic)
        bymouse[m]=(rho_c,rho_b)
        trends.append({"heldout_mouse":m,"spearman_log_selected_f_vs_log_c":rho_c,
                       "spearman_log_selected_f_vs_log_b":rho_b,"kendall_tau_selected_f_vs_c":tau_c,
                       "mean_selected_over_c":float(s["xi_c"].mean()),"median_selected_over_c":float(s["xi_c"].median()),
                       "sd_log_selected_over_c":float(np.std(np.log(s["xi_c"]),ddof=1))})
    tr=pd.DataFrame(trends); tr.to_csv(OUT/"V_MOUSE_KINETIC_ASSOCIATION.csv",index=False,encoding="utf-8-sig")
    diff=tr["spearman_log_selected_f_vs_log_c"]-tr["spearman_log_selected_f_vs_log_b"]
    # Mouse-cluster bootstrap paired difference in within-mouse association.
    rng=np.random.default_rng(SEED)
    boot=rng.choice(diff.to_numpy(),size=(50000,len(diff)),replace=True).mean(axis=1)
    # Full selected-frequency normalization by target.
    norm=select.groupby("target_ATP_mM",as_index=False).agg(
        n_mice=("heldout_mouse","nunique"),median_xi_c=("xi_c","median"),q25_xi_c=("xi_c",lambda z:z.quantile(.25)),
        q75_xi_c=("xi_c",lambda z:z.quantile(.75)),median_d_c=("d_c","median"),median_xi_b=("xi_b","median"),
        median_d_b=("d_b","median"),median_selected_Hz=("selected_frequency_Hz","median"),median_c_Hz=("c_Hz","median"))
    norm.to_csv(OUT/"V_NORMALIZATION_BY_TARGET.csv",index=False,encoding="utf-8-sig")
    # All candidate utility curves; average condition cells within mouse before scoring.
    uall=pd.read_csv(UOUT/"U_ALL_FREQUENCY_HELDOUT.csv")
    per_mouse=uall.groupby(["target_ATP_mM","heldout_mouse","frequency_Hz"],as_index=False).agg(
        gain_pp=("gain_pp","mean"),nrmse_pct=("augmented_nrmse_pct","mean"))
    per_mouse=per_mouse.merge(mice,on=["heldout_mouse","target_ATP_mM"],how="inner",validate="many_to_one")
    per_mouse["xi_c"]=per_mouse["frequency_Hz"]/per_mouse["c_Hz"]
    per_mouse["d_c"]=np.abs(np.log(per_mouse["xi_c"]))
    per_mouse["xi_b"]=per_mouse["frequency_Hz"]/per_mouse["b_Hz"]
    per_mouse["d_b"]=np.abs(np.log(per_mouse["xi_b"]))
    per_mouse.to_csv(OUT/"V_UTILITY_DISTANCE_ALL_CANDIDATES.csv",index=False,encoding="utf-8-sig")
    target_summ=[]; corrmat={}
    for (target,mouse),s in per_mouse.groupby(["target_ATP_mM","heldout_mouse"]):
        rc=float(spearmanr(s["gain_pp"],-s["d_c"]).statistic)
        rb=float(spearmanr(s["gain_pp"],-s["d_b"]).statistic)
        target_summ.append({"target_ATP_mM":target,"heldout_mouse":mouse,"rho_utility_vs_negative_detachment_distance":rc,
                            "rho_utility_vs_negative_recruitment_distance":rb,
                            "best_frequency_by_gain_Hz":float(s.loc[s["gain_pp"].idxmax(),"frequency_Hz"]),
                            "c_Hz":float(s["c_Hz"].iloc[0]),"b_Hz":float(s["b_Hz"].iloc[0])})
        corrmat.setdefault(float(target),{})[str(mouse)]=(rc,rb)
    corr=pd.DataFrame(target_summ)
    corr.to_csv(OUT/"V_CLUSTER_UTILITY_DISTANCE_CORRELATIONS.csv",index=False,encoding="utf-8-sig")
    # Mouse-label permutation of kinetic clocks within target, retaining each utility curve.
    targets=sorted(corrmat); mouseids=sorted(per_mouse.heldout_mouse.astype(str).unique())
    rho_c_mats=[]; rho_b_mats=[]
    for target in targets:
        d=corr[corr["target_ATP_mM"].eq(target)].set_index("heldout_mouse").reindex(mouseids)
        rho_c_mats.append(d["rho_utility_vs_negative_detachment_distance"].to_numpy()[None,:].repeat(len(mouseids),axis=0))
        rho_b_mats.append(d["rho_utility_vs_negative_recruitment_distance"].to_numpy()[None,:].repeat(len(mouseids),axis=0))
    # To preserve full frequency curves under clock reassignment, use raw gain arrays and recompute each curve vs every clock.
    R_c=[];R_b=[]
    for target in targets:
        sub=per_mouse[per_mouse["target_ATP_mM"].eq(target)]
        pivot=sub.pivot(index="heldout_mouse",columns="frequency_Hz",values="gain_pp").reindex(index=mouseids)
        fc=sub.drop_duplicates("heldout_mouse").set_index("heldout_mouse").reindex(mouseids)["c_Hz"].to_numpy()
        fb=sub.drop_duplicates("heldout_mouse").set_index("heldout_mouse").reindex(mouseids)["b_Hz"].to_numpy()
        fs=np.array(sorted(pivot.columns),float); curves=pivot.to_numpy(float)
        matc=np.zeros((len(mouseids),len(mouseids))); matb=np.zeros_like(matc)
        for i in range(len(mouseids)):
            for j in range(len(mouseids)):
                matc[i,j]=spearmanr(curves[i],-np.abs(np.log(fs/fc[j]))).statistic
                matb[i,j]=spearmanr(curves[i],-np.abs(np.log(fs/fb[j]))).statistic
        R_c.append(matc); R_b.append(matb)
    obs_c=float(np.mean([np.mean(np.diag(a)) for a in R_c]))
    obs_b=float(np.mean([np.mean(np.diag(a)) for a in R_b]))
    rng=np.random.default_rng(SEED)
    ix=rng.integers(0,len(mouseids),size=(N_PERM,len(targets),len(mouseids)))
    nullc=np.zeros(N_PERM); nullb=np.zeros(N_PERM)
    row=np.arange(len(mouseids))[None,:]
    for k in range(len(targets)):
        nullc+=R_c[k][row,ix[:,k,:]].mean(axis=1)/len(targets)
        nullb+=R_b[k][row,ix[:,k,:]].mean(axis=1)/len(targets)
    perm_p_c=(1+int((nullc>=obs_c).sum()))/(N_PERM+1)
    perm_p_b=(1+int((nullb>=obs_b).sum()))/(N_PERM+1)
    # Explanatory leave-one-mouse-out models of the full utility surface.
    cv,quad,per_mouse=cv_models(per_mouse)
    cv.to_csv(OUT/"V_UTILITY_MODEL_CV.csv",index=False,encoding="utf-8-sig")
    per_mouse.to_csv(OUT/"V_UTILITY_DISTANCE_ALL_CANDIDATES.csv",index=False,encoding="utf-8-sig")
    summary={"status":"COMPLETED","kinetic_units":"b and c are Hz; 2pi*b and 2pi*c are angular rates in s^-1",
       "n_fibres_per_target":62,"n_mice":10,"n_mouse_genotype_drug_cells":19,"n_selected_frequency_rows":60,
       "V1_concentration_to_kinetics":{"per_mouse_spearman_c_median":float(mice.groupby("heldout_mouse").apply(lambda z:spearmanr(np.log(z["target_ATP_mM"]),np.log(z["c_Hz"])).statistic,include_groups=False).median()),
           "per_mouse_spearman_b_median":float(mice.groupby("heldout_mouse").apply(lambda z:spearmanr(np.log(z["target_ATP_mM"]),np.log(z["b_Hz"])).statistic,include_groups=False).median()),
           "descriptive":"mouse-aware correlations across six concentrations; c is measured from the same frequency-domain fits"},
       "V2_selected_frequency_kinetic_tracking":{"median_mouse_rho_c":float(tr["spearman_log_selected_f_vs_log_c"].median()),
           "median_mouse_rho_b":float(tr["spearman_log_selected_f_vs_log_b"].median()),
           "mean_paired_rho_difference_c_minus_b":float(diff.mean()),
           "mouse_bootstrap_95pct_ci_difference":[float(v) for v in np.quantile(boot,[.025,.975])]},
       "V3_selected_normalized_frequency":{"median_xi_c":float(select["xi_c"].median()),
           "q25_xi_c":float(select["xi_c"].quantile(.25)),"q75_xi_c":float(select["xi_c"].quantile(.75)),
           "iqr_factor_q75_over_q25":float(select["xi_c"].quantile(.75)/select["xi_c"].quantile(.25)),
           "by_target":norm.to_dict(orient="records")},
       "V4_all_candidate_utility_distance":{"mean_cluster_rho_c":float(corr["rho_utility_vs_negative_detachment_distance"].mean()),
           "median_cluster_rho_c":float(corr["rho_utility_vs_negative_detachment_distance"].median()),
           "mean_cluster_rho_b":float(corr["rho_utility_vs_negative_recruitment_distance"].mean()),
           "median_cluster_rho_b":float(corr["rho_utility_vs_negative_recruitment_distance"].median()),
           "clock_mouse_permutation_replicates":N_PERM,"seed":SEED,"permutation_p_c":perm_p_c,"permutation_p_b":perm_p_b,
           "cv_models":cv.to_dict(orient="records"),"quadratic_shape":quad},
       "qualification":"All c/b values are fitted from each future target spectrum; V explains outcomes retrospectively and is not a deployable design test."}
    (OUT/"V_RESULTS_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    # Figures
    fig,ax=plt.subplots(figsize=(7.5,5))
    ax.scatter(select["c_Hz"],select["selected_frequency_Hz"],c=select["target_ATP_mM"],cmap="viridis",alpha=.8)
    lim=[min(select["c_Hz"].min(),select["selected_frequency_Hz"].min()),max(select["c_Hz"].max(),select["selected_frequency_Hz"].max())]
    ax.plot(lim,lim,"k--",lw=1); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Future detachment characteristic frequency c (Hz)"); ax.set_ylabel("Nested-selected baseline frequency (Hz)")
    ax.set_title("Selected frequency versus fitted future detachment clock")
    fig.colorbar(ax.collections[0],ax=ax,label="Future MgATP (mM)"); fig.tight_layout()
    fig.savefig(FIG/"V_selected_frequency_vs_c.png",dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,5))
    ax.scatter(select["xi_c"],select["target_ATP_mM"],c=select["heldout_mouse"].astype("category").cat.codes,cmap="tab10",alpha=.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.axvline(1,color="k",ls="--",lw=1)
    ax.set_xlabel("Selected frequency / future c (dimensionless)"); ax.set_ylabel("Future MgATP target (mM)")
    ax.set_title("Normalization does not preassume a common ratio")
    fig.tight_layout(); fig.savefig(FIG/"V_selected_normalized_ratio.png",dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,5))
    ax.scatter(per_mouse["d_c"],per_mouse["gain_pp"],s=8,alpha=.35,label="detachment distance")
    ax.set_xlabel("Absolute log frequency distance |log(f/c)|"); ax.set_ylabel("Held-out prediction gain (pp)")
    ax.set_title("All candidate frequencies: held-out gain versus kinetic distance")
    ax.grid(True,alpha=.2); fig.tight_layout(); fig.savefig(FIG/"V_utility_vs_detachment_distance.png",dpi=220); plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
