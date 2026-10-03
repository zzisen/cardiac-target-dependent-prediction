#!/usr/bin/env python
"""Simple leave-one-target-out comparison of absolute and kinetic coordinates."""
from __future__ import annotations
import json, sys
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve(); UX=HERE.parents[2]; PROJECT=HERE.parents[3]
OLD=PROJECT/"P2_QT_UPLIFT_2026-10-02"; OUT=HERE.parents[1]/"results"; FIG=HERE.parents[1]/"figures"
OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)

def design(model, d):
    if model=="H0_absolute_frequency":
        x=np.log(d.frequency_Hz.to_numpy(float));return np.c_[np.ones(len(x)),x,x*x]
    if model=="H1_target_state_only":
        x=np.log(d.target_ATP_mM.to_numpy(float));return np.c_[np.ones(len(x)),x,x*x]
    if model=="H3_kinetic_distance":
        x=d.d_c.to_numpy(float);return np.c_[np.ones(len(x)),x,x*x]
    raise ValueError(model)

def main():
    u= pd.read_csv(REPO/"analysis/03_awinda/continuum/results"/"U_ALL_FREQUENCY_HELDOUT.csv")
    v= pd.read_csv(REPO/"analysis/08_boundaries/kinetics/results"/"V_TARGET_KINETIC_TIMESCALES.csv")
    c=v.groupby(["AnimalID","target_ATP_mM"],as_index=False).c_Hz.median()
    d=(u.groupby(["AnimalID","target_ATP_mM","frequency_Hz"],as_index=False)
         .agg(gain_pp=("gain_pp","mean"),baseline_nrmse_pct=("baseline_only_nrmse_pct","mean"),
              augmented_nrmse_pct=("augmented_nrmse_pct","mean")))
    d=d.merge(c,on=["AnimalID","target_ATP_mM"],how="left",validate="many_to_one")
    d["d_c"]=np.abs(np.log(d.frequency_Hz/d.c_Hz))
    d["xi_c"]=d.frequency_Hz/d.c_Hz
    d.to_csv(OUT/"X_U_MOUSE_TARGET_FREQUENCY_UTILITY_WITH_CLOCK.csv",index=False,encoding="utf-8-sig")
    models=["H0_absolute_frequency","H1_target_state_only","H3_kinetic_distance"]
    test=[]
    for model in models:
        for target in sorted(d.target_ATP_mM.unique()):
            tr=d[d.target_ATP_mM.ne(target)];te=d[d.target_ATP_mM.eq(target)]
            xtr=design(model,tr); xte=design(model,te); ytr=tr.gain_pp.to_numpy(float); yte=te.gain_pp.to_numpy(float)
            beta=np.linalg.lstsq(xtr,ytr,rcond=None)[0]
            pred=xte@beta
            test.append({"model":model,"heldout_target_ATP_mM":target,"n_test_mouse_frequency_points":len(te),
                         "rmse_gain_pp":float(np.sqrt(np.mean((pred-yte)**2))),
                         "mae_gain_pp":float(np.mean(np.abs(pred-yte))),
                         "bias_pp":float(np.mean(pred-yte)),"beta_intercept":float(beta[0]),
                         "beta_linear":float(beta[1]),"beta_quadratic":float(beta[2]),
                         "test_uses_future_kinetics":model=="H3_kinetic_distance"})
    testdf=pd.DataFrame(test)
    testdf.to_csv(OUT/"X_LOTO_TARGET_TEST.csv",index=False,encoding="utf-8-sig")
    summary=(testdf.groupby("model",as_index=False).agg(equal_target_mean_rmse_pp=("rmse_gain_pp","mean"),
             median_target_rmse_pp=("rmse_gain_pp","median"),equal_target_mean_mae_pp=("mae_gain_pp","mean"),
             n_heldout_targets=("heldout_target_ATP_mM","nunique")))
    summary["relative_to_H0_rmse"] = summary.equal_target_mean_rmse_pp/float(summary.loc[summary.model.eq("H0_absolute_frequency"),"equal_target_mean_rmse_pp"].iloc[0])
    summary.to_csv(OUT/"X_MODEL_COMPARISON.csv",index=False,encoding="utf-8-sig")

    # Evidence rows retain observational/model status and preserve per-mouse U outcomes.
    selected=pd.read_csv(REPO/"analysis/03_awinda/continuum/results"/"U_MGATP_CONTINUUM_SELECTIONS.csv")
    # Explicit join because the frozen U table calls the mouse key heldout_mouse.
    selected=selected.merge(c,left_on=["heldout_mouse","target_ATP_mM"],right_on=["AnimalID","target_ATP_mM"],how="left")
    selected["normalized_selected_f_over_c"]=selected.selected_frequency_Hz/selected.c_Hz
    selected["evidence_branch"]="U";selected["system"]="Awinda";selected["model_domain"]="empirical; mouse-held-out nested frequency selection"
    selected["empirical_or_model"]="empirical retrospective validation"
    selected["kinetic_timescale_hz"]=selected.c_Hz
    selected["absolute_selected_frequency_hz"]=selected.selected_frequency_Hz
    selected["predictive_utility_gain_pp"]=selected.gain_pp
    selected["target_identity"]=selected.target_ATP_mM.astype(str)+" mM MgATP"
    evid=selected[["evidence_branch","target_identity","system","heldout_mouse","genotype_drug_cells",
                   "absolute_selected_frequency_hz","kinetic_timescale_hz","normalized_selected_f_over_c",
                   "predictive_utility_gain_pp","model_domain","empirical_or_model","n_outer_training_mice"]].copy()
    evid.to_csv(OUT/"X_U_SELECTION_EVIDENCE.csv",index=False,encoding="utf-8-sig")

    # Concise test figure: equal-target LOTO RMSE for the three comparable explanatory models.
    fig,ax=plt.subplots(figsize=(7.4,4.3))
    order=models;vals=[summary.set_index("model").loc[m,"equal_target_mean_rmse_pp"] for m in order]
    ax.bar(["Absolute Hz\nH0","ATP target state\nH1","Distance to c\nH3"],vals,color=["#426a8c","#91a7b2","#cc7958"])
    ax.set_ylabel("Leave-one-target-out RMSE of utility gain (pp)")
    ax.set_title("Kinetic normalization does not improve held-out-target utility prediction")
    ax.grid(axis="y",alpha=.25)
    fig.tight_layout();fig.savefig(FIG/"X_loto_utility_model_comparison.png",dpi=180);plt.close(fig)

    print(summary.to_string(index=False))
    print("rows",len(evid),"utility surface",len(d),"LOTO targets",sorted(d.target_ATP_mM.unique().tolist()))

if __name__=="__main__":main()
