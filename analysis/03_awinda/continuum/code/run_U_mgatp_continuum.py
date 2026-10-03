from __future__ import annotations
import itertools, json, hashlib
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, rankdata, binomtest
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROUND=HERE.parents[1]
ROOT=ROUND.parent
Q=REPO/"analysis/03_awinda/frequency"
SRC=REPO/"data/derived/awinda_source"
OUT=HERE.parent/"results"
FIG=HERE.parent/"figures"
OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
ALPHA=1.0
BASE=5.0
PCA=4.8
TARGETS=[0.05,0.10,0.25,0.50,1.00,2.50]
SEED=20261002
N_PERM=100000

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def read_cells():
    xld=pd.read_csv(SRC/"WT_RLC_N47K_Control_Mavacamten_XLD_GWN_DQ1_2019_09_09.csv")
    force=pd.read_csv(SRC/"Transgenic_mice_Fiber_Tensions_with_Fits_DQ1_2019_09_06.csv")
    link=force[["AnimalID","Fiber","Mutation","Treatment"]].drop_duplicates()
    if link["Fiber"].duplicated().any(): raise ValueError("Fibre-to-animal map is not one-to-one")
    files=xld[["Filename","Mutation","Cond"]].dropna(subset=["Filename"]).drop_duplicates()
    fmap=files.merge(link,left_on="Filename",right_on="Fiber",how="left",validate="one_to_one")
    if fmap["AnimalID"].isna().any() or not fmap["Mutation_x"].eq(fmap["Mutation_y"]).all():
        raise ValueError("Unmapped fibre or genotype mismatch")
    fmap["drug_state"]=np.where(fmap["Cond"].eq("Control"),"Control","0.3_uM_mavacamten")
    fmap["group"]=fmap["Mutation_x"].astype(str)+"|"+fmap["drug_state"]
    d=xld.loc[xld["pCa"].eq(PCA)&pd.to_numeric(xld["Quality"],errors="coerce").eq(1)].copy()
    d=d.merge(fmap[["Filename","AnimalID","Mutation_x","drug_state","group"]],on="Filename",how="left",validate="many_to_one")
    for c in ["ATP (mM)","Freq (Hz)","Em (kPa)","Vm (kPa)"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    freq=np.sort(d.loc[d["ATP (mM)"].eq(BASE),"Freq (Hz)"].unique())
    if len(freq)!=72: raise ValueError(f"Frequency count {len(freq)} != 72")
    cellcols=["AnimalID","Mutation_x","drug_state","group"]
    means=d.groupby(cellcols+["ATP (mM)","Freq (Hz)"],as_index=False).agg(
        Re=("Em (kPa)","mean"),Im=("Vm (kPa)","mean"),n_fibres=("Filename","nunique"))
    rows=[]
    for key,sub in means.groupby(cellcols,sort=True):
        row=dict(zip(cellcols,key))
        for atp in [BASE]+TARGETS:
            z=sub[sub["ATP (mM)"].eq(atp)].set_index("Freq (Hz)").reindex(freq)
            if z[["Re","Im"]].isna().any().any(): raise ValueError(f"Incomplete spectrum: {key}, ATP {atp}")
            if atp==BASE:
                row["n_fibres_baseline"]=int(z["n_fibres"].max())
                for f,re,im in zip(freq,z["Re"],z["Im"]):
                    row[f"x_re_{f:.4f}"]=float(re); row[f"x_im_{f:.4f}"]=float(im)
            else:
                row[f"n_fibres_ATP_{atp:g}"]=int(z["n_fibres"].max())
                row[f"y{atp:g}"]=np.r_[z["Re"].to_numpy(float),z["Im"].to_numpy(float)]
        rows.append(row)
    cells=pd.DataFrame(rows).sort_values(cellcols).reset_index(drop=True)
    if cells["AnimalID"].nunique()!=10 or len(cells)!=19: raise ValueError("Hierarchy differs from frozen U target set")
    return cells,freq

def predict(groups,feats,y,train,test,ordered_groups):
    G=np.column_stack([(groups==g).astype(float) for g in ordered_groups])
    if feats.shape[1]:
        mu=feats[train].mean(axis=0); sd=feats[train].std(axis=0,ddof=0)
        sd=np.where(sd>1e-12,sd,1.0)
        X=np.column_stack([G,(feats-mu)/sd])
        penalty=np.r_[np.zeros(G.shape[1]),np.full(feats.shape[1],ALPHA)]
    else:
        X=G; penalty=np.zeros(G.shape[1])
    beta=np.linalg.solve(X[train].T@X[train]+np.diag(penalty),X[train].T@y[train])
    return X[test]@beta

def nrmse(y,p):
    return float(100*np.sqrt(np.mean((p-y)**2)/np.mean(y**2)))

def fit_target(cells,freq,target,outer_animals,group_filter=None,include_all_freq=True):
    frame=cells if group_filter is None else cells[cells["group"].eq(group_filter)].reset_index(drop=True)
    allcells=cells if group_filter is None else frame
    cellcols=["AnimalID","Mutation_x","drug_state","group"]
    y= np.vstack(frame[f"y{target:g}"].to_numpy())
    animals=sorted(frame["AnimalID"].astype(str).unique())
    groups=frame["group"].to_numpy()
    ordered_groups=sorted(frame["group"].unique())
    feature_names=[f"x_re_{f:.4f}" for f in freq]+[f"x_im_{f:.4f}" for f in freq]
    X=frame[feature_names].to_numpy(float)
    by_animal={a:np.flatnonzero(frame["AnimalID"].astype(str).to_numpy()==a) for a in animals}
    detail=[]; nested=[]; summaries=[]
    for outer in animals:
        test=by_animal[outer]
        train=np.flatnonzero(frame["AnimalID"].astype(str).to_numpy()!=outer)
        basepred=predict(groups,np.empty((len(frame),0)),y,train,test,ordered_groups)
        base_metrics=[nrmse(y[ix],pr) for ix,pr in zip(test,basepred)]
        # Inner animal-held-out selection, matching Q's 2-scalar complex pair and score.
        inner_animals=sorted(set(frame.iloc[train]["AnimalID"].astype(str)))
        scores=[]
        for j,f in enumerate(freq):
            feat=X[:,[j,j+len(freq)]]
            errors=[]
            for inner in inner_animals:
                inner_test=train[frame.iloc[train]["AnimalID"].astype(str).to_numpy()==inner]
                inner_train=train[frame.iloc[train]["AnimalID"].astype(str).to_numpy()!=inner]
                pi=predict(groups,feat,y,inner_train,inner_test,ordered_groups)
                scale=np.sqrt(np.mean(y[inner_test]**2))
                errors.append(float(np.mean(((pi-y[inner_test])/scale)**2)))
            cid=f"CM_ATP5_{f:.4f}Hz_pair"
            scores.append((float(np.mean(errors)),cid,float(f),j))
        best=min(scores,key=lambda z:(z[0],z[1]))
        _,cid,selected_f,j=best
        selected_feat=X[:,[j,j+len(freq)]]
        pred=predict(groups,selected_feat,y,train,test,ordered_groups)
        selected_err=[nrmse(y[ix],pr) for ix,pr in zip(test,pred)]
        for row_ix,(ix,baseerr,augerr) in enumerate(zip(test,base_metrics,selected_err)):
            r=frame.iloc[ix]
            nested.append({"target_ATP_mM":target,"heldout_mouse":outer,"AnimalID":r["AnimalID"],
                "genotype":r["Mutation_x"],"drug_state":r["drug_state"],"group":r["group"],
                "selected_panel_id":cid,"selected_frequency_Hz":selected_f,"measurement_budget_scalars":2,
                "inner_cv_mean_scaled_mse":best[0],"baseline_only_nrmse_pct":baseerr,
                "augmented_nrmse_pct":augerr,"gain_pp":baseerr-augerr,
                "inner_selection_grouped_by_mouse":True,"outer_mouse_excluded_from_selection":True})
        summaries.append({"target_ATP_mM":target,"heldout_mouse":outer,"selected_frequency_Hz":selected_f,
                          "selected_panel_id":cid,"inner_cv_mean_scaled_mse":best[0],
                          "n_mouse_condition_cells":len(test),"baseline_only_nrmse_pct":float(np.mean(base_metrics)),
                          "augmented_nrmse_pct":float(np.mean(selected_err)),
                          "gain_pp":float(np.mean(np.array(base_metrics)-np.array(selected_err))),
                          "genotype_drug_cells":";".join(frame.iloc[test]["group"].astype(str).tolist()),
                          "n_outer_training_mice":len(inner_animals)})
        if include_all_freq:
            for k,f in enumerate(freq):
                feat=X[:,[k,k+len(freq)]]
                pr=predict(groups,feat,y,train,test,ordered_groups)
                for ix,baseerr,phat in zip(test,base_metrics,pr):
                    r=frame.iloc[ix]; err=nrmse(y[ix],phat)
                    detail.append({"target_ATP_mM":target,"heldout_mouse":outer,"AnimalID":r["AnimalID"],
                      "genotype":r["Mutation_x"],"drug_state":r["drug_state"],"group":r["group"],
                      "frequency_Hz":float(f),"baseline_only_nrmse_pct":baseerr,
                      "augmented_nrmse_pct":err,"gain_pp":baseerr-err,"candidate_budget_scalars":2})
    return pd.DataFrame(nested),pd.DataFrame(summaries),pd.DataFrame(detail)

def main():
    frozen=json.loads((OUT/"U_TARGET_FREEZE.json").read_text(encoding="utf-8"))
    if frozen["status"]!="FROZEN_BEFORE_OUTCOME_ANALYSIS" or frozen["frozen_targets"]!=TARGETS:
        raise ValueError("U target freeze differs from runner")
    cells,freq=read_cells()
    animals=sorted(cells["AnimalID"].astype(str).unique())
    all_nested=[]; all_summary=[]; all_detail=[]
    for target in TARGETS:
        print("Running U target",target,flush=True)
        n,s,d=fit_target(cells,freq,target,animals,None,True)
        all_nested.append(n); all_summary.append(s); all_detail.append(d)
    held=pd.concat(all_nested,ignore_index=True)
    sels=pd.concat(all_summary,ignore_index=True)
    detail=pd.concat(all_detail,ignore_index=True)
    held.to_csv(OUT/"U_MGATP_CONTINUUM_HELDOUT.csv",index=False,encoding="utf-8-sig")
    sels.to_csv(OUT/"U_MGATP_CONTINUUM_SELECTIONS.csv",index=False,encoding="utf-8-sig")
    detail.to_csv(OUT/"U_ALL_FREQUENCY_HELDOUT.csv",index=False,encoding="utf-8-sig")
    # Mouse-level selected-frequency trend and the predeclared within-mouse label-permutation test.
    wide=sels.pivot(index="heldout_mouse",columns="target_ATP_mM",values="selected_frequency_Hz").reindex(columns=TARGETS)
    trend=[]
    for mouse,row in wide.iterrows():
        v=row.to_numpy(float)
        trend.append({"heldout_mouse":mouse,"spearman_rho_log_ATP_log_frequency":float(spearmanr(np.log(TARGETS),np.log(v)).statistic),
                      "kendall_tau_b":float(kendalltau(np.log(TARGETS),np.log(v)).statistic),
                      "monotonic_downward_transitions":int(np.sum(np.diff(v)<0)),
                      "nondecreasing_transitions":int(np.sum(np.diff(v)>=0)),
                      **{f"selected_Hz_ATP_{t:g}":float(row[t]) for t in TARGETS}})
    trend=pd.DataFrame(trend)
    trend.to_csv(OUT/"U_MOUSE_TREND_SUMMARY.csv",index=False,encoding="utf-8-sig")
    observed=float(trend["spearman_rho_log_ATP_log_frequency"].mean())
    # Under the null, the six selected-frequency labels are exchangeable within each mouse.
    perms=np.array(list(itertools.permutations(range(len(TARGETS)))),dtype=int)
    rho_null_by_mouse=[]
    xrank=rankdata(np.log(TARGETS))
    for mouse,row in wide.iterrows():
        vrank=rankdata(np.log(row.to_numpy(float)))
        denom=np.sqrt(np.sum((xrank-xrank.mean())**2)*np.sum((vrank-vrank.mean())**2))
        vals=[]
        for ix in perms:
            vals.append(float(np.sum((xrank-xrank.mean())*(vrank[ix]-vrank[ix].mean()))/denom) if denom else 0.0)
        rho_null_by_mouse.append(np.asarray(vals))
    rng=np.random.default_rng(SEED)
    idx=rng.integers(0,len(perms),size=(N_PERM,len(rho_null_by_mouse)))
    null=np.mean(np.column_stack([rho_null_by_mouse[j][idx[:,j]] for j in range(len(rho_null_by_mouse))]),axis=1)
    pperm=(1+int(np.sum(null>=observed)))/(N_PERM+1)
    npos=int((trend["spearman_rho_log_ATP_log_frequency"]>0).sum())
    nneg=int((trend["spearman_rho_log_ATP_log_frequency"]<0).sum())
    signp=float(binomtest(npos,npos+nneg,0.5,alternative="greater").pvalue) if npos+nneg else 1.0
    # Low vs high selected-panel prediction gain; average cells within mouse first.
    sels["band"]=np.select([sels["target_ATP_mM"].isin([0.05,0.1,0.25]),sels["target_ATP_mM"].eq(0.5),sels["target_ATP_mM"].isin([1.0,2.5])],["low","middle","high"],default="NA")
    mouse_band=sels.groupby(["heldout_mouse","band"],as_index=False)["gain_pp"].mean()
    piv=mouse_band.pivot(index="heldout_mouse",columns="band",values="gain_pp")
    paired=(piv["low"]-piv["high"]).dropna().to_numpy(float)
    rng=np.random.default_rng(SEED)
    boot=np.mean(rng.choice(paired,size=(50000,len(paired)),replace=True),axis=1)
    low_high={"n_mice":int(len(paired)),"mean_low_gain_pp":float(piv["low"].mean()),
              "mean_high_gain_pp":float(piv["high"].mean()),"paired_low_minus_high_gain_pp":float(paired.mean()),
              "mouse_bootstrap_95pct_ci_pp":[float(x) for x in np.quantile(boot,[0.025,0.975])]}
    sels.to_csv(OUT/"U_MGATP_CONTINUUM_SELECTIONS.csv",index=False,encoding="utf-8-sig")
    # Whole candidate utility curve, first average within mouse then equally across mice.
    bymouse=detail.groupby(["target_ATP_mM","heldout_mouse","frequency_Hz"],as_index=False).agg(
        gain_pp=("gain_pp","mean"),nrmse=("augmented_nrmse_pct","mean"),baseline=("baseline_only_nrmse_pct","mean"))
    utility=bymouse.groupby(["target_ATP_mM","frequency_Hz"],as_index=False).agg(
        mean_gain_pp=("gain_pp","mean"),median_gain_pp=("gain_pp","median"),mean_nrmse_pct=("nrmse","mean"),
        improved_mice=("gain_pp",lambda x:int((x>0).sum())),n_mice=("heldout_mouse","nunique"))
    utility.to_csv(OUT/"U_FREQUENCY_UTILITY_SUMMARY.csv",index=False,encoding="utf-8-sig")
    # Global nested-selected outcomes by genotype and drug, with mice as summary units.
    stratum=held.groupby(["target_ATP_mM","group"],as_index=False).agg(
        n_cells=("AnimalID","size"),n_mice=("AnimalID","nunique"),mean_gain_pp=("gain_pp","mean"),
        median_gain_pp=("gain_pp","median"),improved_cells=("gain_pp",lambda x:int((x>0).sum())))
    stratum.to_csv(OUT/"U_GLOBAL_SELECTION_STRATA.csv",index=False,encoding="utf-8-sig")
    # Explicitly low-n sensitivity: train and select only within each genotype x drug stratum.
    sec=[]
    for group in sorted(cells["group"].unique()):
        print("Running descriptive stratum sensitivity",group,flush=True)
        for target in TARGETS:
            _,s,_=fit_target(cells,freq,target,animals,group,False)
            sec.append(s)
    stratsel=pd.concat(sec,ignore_index=True)
    stratsel.to_csv(OUT/"U_STRATUM_SELECTIONS.csv",index=False,encoding="utf-8-sig")
    # Figures.
    fig,ax=plt.subplots(figsize=(8,5))
    for _,row in wide.iterrows(): ax.plot(TARGETS,row.values,marker="o",alpha=.65,lw=1,label=str(row.name))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Future MgATP target (mM; log scale)"); ax.set_ylabel("Nested-selected baseline frequency (Hz; log scale)")
    ax.set_title("Mouse-held-out selected measurement frequency across MgATP continuum")
    ax.grid(True,which="both",alpha=.2); fig.tight_layout()
    fig.savefig(FIG/"U_selected_frequency_by_target.png",dpi=220); plt.close(fig)
    matrix=utility.pivot(index="frequency_Hz",columns="target_ATP_mM",values="mean_gain_pp")
    fig,ax=plt.subplots(figsize=(9,6))
    im=ax.imshow(matrix.to_numpy(),aspect="auto",origin="lower",cmap="coolwarm",vmin=-np.nanmax(np.abs(matrix.to_numpy())),vmax=np.nanmax(np.abs(matrix.to_numpy())))
    ax.set_xticks(np.arange(len(matrix.columns)),[f"{x:g}" for x in matrix.columns]); ax.set_yticks(np.arange(0,len(matrix.index),8),[f"{matrix.index[i]:g}" for i in range(0,len(matrix.index),8)])
    ax.set_xlabel("Future MgATP target (mM)"); ax.set_ylabel("Baseline measurement frequency (Hz)")
    ax.set_title("Outer held-out NRMSE gain by candidate frequency and target")
    fig.colorbar(im,ax=ax,label="Gain vs group-only baseline (pp)"); fig.tight_layout()
    fig.savefig(FIG/"U_frequency_target_utility_heatmap.png",dpi=220); plt.close(fig)
    summary={"status":"COMPLETED","n_mice":len(animals),"n_animal_condition_cells":len(cells),"n_frequencies":len(freq),
      "targets_mM":TARGETS,"nested_rows":len(held),"all_frequency_rows":len(detail),
      "trend":{"mean_mouse_spearman_rho":observed,"median_mouse_spearman_rho":float(trend["spearman_rho_log_ATP_log_frequency"].median()),
               "mean_mouse_kendall_tau":float(trend["kendall_tau_b"].mean()),"total_downward_adjacent_transitions":int(trend["monotonic_downward_transitions"].sum()),
               "mouse_trend_rhos_positive":npos,"mouse_trend_rhos_negative":nneg,"exact_sign_test_one_sided_p":signp,
               "within_mouse_label_permutation_monte_carlo_replicates":N_PERM,"seed":SEED,"permutation_p_upper_tail":pperm,
               "interpretation":"Selection uses overlapping nested folds; p-values are sensitivity analyses, not independent confirmation."},
      "gain_low_vs_high":low_high,
      "stratum_selection_sensitivity":"available in U_STRATUM_SELECTIONS.csv; exploratory because each stratum has 4-5 mice",
      "source_sha256":{p.name:sha(p) for p in sorted(SRC.glob("*.xlsx"))}}
    (OUT/"U_RESULTS_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
