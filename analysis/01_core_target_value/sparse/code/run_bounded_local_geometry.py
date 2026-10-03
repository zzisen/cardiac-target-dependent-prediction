"""Bounded local quadratic profile checks for selected K panels and M."""
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))

import json,itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from p2.constants import LR95_DF1

HERE=Path(__file__).resolve().parent; K=HERE.parent; ROOT=K.parent.parent
INP=K/"input"; OUT=K/"results"
sys_path=str(HERE/"code")
import sys;sys.path.insert(0,sys_path)

def z_of_p(p):
    p=np.asarray(p,float);z=np.log(np.maximum(p,1e-300));z[9]=p[9]/45.0;return z

def qwidth(H,g,u0):
    H=(H+H.T)/2; g=np.asarray(g,float);u0=np.asarray(u0,float)
    V=np.linalg.pinv(H,rcond=1e-11);var=max(0,float(g@V@g))
    uncon=np.zeros(len(g)) if var<=0 else np.sqrt(LR95_DF1/var)*(V@g)
    starts=[np.zeros(len(g))]
    c=np.clip(uncon,-u0,1-u0);radius=float(c@H@c)
    if radius>LR95_DF1 and radius>0:c*=0.95*np.sqrt(LR95_DF1/radius)
    starts.append(c)
    bounds=list(zip(-u0,1-u0)); endpoints=[]; records=[]
    for side,sign in (("low",1.0),("high",-1.0)):
        opts=[]
        for si,x0 in enumerate(starts):
            opt=minimize(lambda x:sign*float(g@x),x0=x0,jac=lambda x:sign*g,method="SLSQP",bounds=bounds,
             constraints=[{"type":"ineq","fun":lambda x:LR95_DF1-float(x@H@x),"jac":lambda x:-2*H@x}],
             options={"maxiter":1500,"ftol":1e-10,"disp":False})
            ds=float(opt.x@H@opt.x);feas=ds<=LR95_DF1+1e-6 and np.all(opt.x>=-u0-1e-8) and np.all(opt.x<=1-u0+1e-8)
            records.append({"side":side,"start_index":si,"success":bool(opt.success),"message":str(opt.message),"delta_objective":ds,"feasible":bool(feas),"target_delta":float(g@opt.x)})
            if feas:opts.append((float(g@opt.x),opt,ds))
        if not opts: endpoints.append((0.0,False,False,0.0,[]))
        else:
            selected=min(opts,key=lambda x:x[0]) if side=="low" else max(opts,key=lambda x:x[0])
            q=selected[0];x=selected[1].x
            active=np.flatnonzero((x+u0<1e-5)|(1-u0-x<1e-5)).tolist()
            endpoints.append((q,True,bool(selected[1].success),selected[2],active))
    return endpoints[1][0]-endpoints[0][0],endpoints,records

def mat(rows):
    if not rows:return np.empty((0,14))
    return np.asarray(rows,float)

def main():
    kdf=pd.read_csv(OUT/"K_SPARSE_PANEL_RESULTS.csv")
    gdf=pd.read_csv(OUT/"K_TARGET_GRADIENTS.csv")
    bdf=pd.read_csv(OUT/"K_BASELINE_SENSITIVITIES.csv")
    sdf=pd.read_csv(OUT/"K_LOCAL_SENSITIVITIES.csv")
    amap=json.loads((INP/"A_measurement_target_map.json").read_text(encoding="utf-8"))
    base_lookup={};grad_lookup={};sens_lookup={};u_lookup={}
    for (dn,t),sub in bdf.groupby(["domain","target"]):
        rows=[]
        for _,r in sub.groupby("row_id",sort=False): rows.append(r.sort_values("parameter_index").sensitivity.to_numpy(float))
        base_lookup[(dn,t)]=mat(rows)
    for (dn,t),sub in gdf.groupby(["domain","target"]): grad_lookup[(dn,t)]=sub.sort_values("parameter_index").target_gradient.to_numpy(float)
    for (dn,t),sub in sdf.groupby(["domain","target"]):
        sens_lookup[(dn,t)]={sid:x.sort_values("parameter_index").whitened_sensitivity.to_numpy(float) for sid,x in sub.groupby("scalar_id")}
    for dn in ["P1_pair_plus_0.5","P1_pair_plus_1.0"]:
        dom=amap["domains"][dn]["domain"];fit=amap["domains"][dn]["baseline_fit"]["params"]
        zlo=z_of_p(dom["lo"]);zhi=z_of_p(dom["hi"]);u=np.clip((z_of_p(fit)-zlo)/(zhi-zlo),0,1)
        for t in ["ATP0.1","ATP1","Pi0","Pi5"]:u_lookup[(dn,t)]=u
    out=[]; optimizer_rows=[]
    for row in kdf.itertuples():
        key=(row.domain,row.target);J=base_lookup[key];g=grad_lookup[key];u0=u_lookup[key]
        H=J.T@J; ids=[x for x in str(row.selected_scalar_ids).split(";") if x]
        R=mat([sens_lookup[key][sid] for sid in ids])
        Hnew=H+R.T@R
        bw0,e0,attempts0=qwidth(H,g,u0);bw1,e1,attempts1=qwidth(Hnew,g,u0)
        for state,records in (("baseline",attempts0),("after_panel",attempts1)):
            for rec in records:optimizer_rows.append({"target":row.target,"domain":row.domain,"accounting_view":row.accounting_view,"budget":row.budget,"selected_scalar_ids":row.selected_scalar_ids,"state":state,**rec})
        out.append({"target":row.target,"domain":row.domain,"accounting_view":row.accounting_view,"budget":row.budget,
          "selected_scalar_ids":row.selected_scalar_ids,"n_scalar_observations":len(ids),
          "baseline_bounded_local_width_pp":bw0,"after_panel_bounded_local_width_pp":bw1,
          "bounded_width_reduction_pp":bw0-bw1,"bounded_width_reduction_pct":100*(1-bw1/bw0) if bw0 else np.nan,
          "best_full_single_CM_spectrum_bounded_utility_ratio":np.nan,
          "all_selected_endpoints_feasible":all(x[1] for x in e0+e1),
          "all_selected_endpoint_optimizers_converged":all(x[2] for x in e0+e1),
          "optimizer_all_attempts_recorded":True,"maximum_endpoint_constraint_slack":max([LR95_DF1-r["delta_objective"] for r in attempts0+attempts1 if r["feasible"]],default=np.nan)})
    # Normalize bounded width gains by the best eligible full spectrum for the same stress target/domain.
    outdf=pd.DataFrame(out)
    full=[]
    for dn in ["P1_pair_plus_0.5","P1_pair_plus_1.0"]:
      for t in ["ATP0.1","ATP1","Pi0","Pi5"]:
        key=(dn,t);J=base_lookup[key];g=grad_lookup[key];u0=u_lookup[key];H=J.T@J
        eligible=[c for c in ["ATP0.1","ATP1","Pi0","Pi5"] if c!=t]
        for c in eligible:
          ss=[s for s in sens_lookup[key] if s.startswith(f"CM_{c}_")]
          R=mat([sens_lookup[key][s] for s in ss]);bw0,_,_=qwidth(H,g,u0);bw1,_,_=qwidth(H+R.T@R,g,u0)
          full.append({"domain":dn,"target":t,"condition":c,"baseline_bounded_width_pp":bw0,"after_full_spectrum_bounded_width_pp":bw1,"bounded_width_reduction_pp":bw0-bw1,"bounded_width_reduction_pct":100*(1-bw1/bw0) if bw0 else np.nan})
    fulldf=pd.DataFrame(full)
    bestfull=fulldf.groupby(["domain","target"]).bounded_width_reduction_pp.max().to_dict()
    outdf["best_full_single_CM_spectrum_bounded_utility_ratio"]=[r.bounded_width_reduction_pp/max(bestfull.get((r.domain,r.target),0),1e-15) for r in outdf.itertuples()]
    outdf.to_csv(OUT/"K_BOUNDED_LOCAL_SUPPORT.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(optimizer_rows).to_csv(OUT/"K_BOUNDED_OPTIMIZER_DIAGNOSTICS.csv",index=False,encoding="utf-8-sig")
    fulldf.to_csv(OUT/"K_FULL_SPECTRUM_BOUNDED_LOCAL_SUPPORT.csv",index=False,encoding="utf-8-sig")
    # M's bounded local alternative: compare a bounded local quadratic profile with exact frozen A profile widths.
    amap=json.loads((INP/"A_measurement_target_map.json").read_text(encoding="utf-8"));mrows=[]
    for dn in ["P1_pair_plus_0.5","P1_pair_plus_1.0"]:
      for t in ["ATP0.1","ATP1","Pi0","Pi5"]:
        key=(dn,t);J=base_lookup[key];g=grad_lookup[key];u0=u_lookup[key];H=J.T@J
        for cond in [c for c in ["ATP0.1","ATP1","Pi0","Pi5"] if c!=t]:
          for kind,ids in [("stress",[f"stress_{cond}"]),("CM_spectrum",[s for s in sens_lookup[key] if s.startswith(f"CM_{cond}_")])]:
            R=mat([sens_lookup[key][sid] for sid in ids]); bw0,_,_=qwidth(H,g,u0);bw1,_,_=qwidth(H+R.T@R,g,u0)
            m_id=ids[0] if kind=="stress" else f"CM_spectrum_{cond}"
            a_sup=amap["domains"][dn]["measurements"][m_id]["support"].get(t,{})
            mrows.append({"domain":dn,"target":t,"measurement_kind":kind,"measurement_id":m_id,
              "bounded_local_width_reduction_pct":100*(1-bw1/bw0) if bw0 else np.nan,
              "A_exact_profile_width_reduction_pct":a_sup.get("relative_width_reduction_pct",np.nan),
              "bounded_local_width_reduction_pp":bw0-bw1,"baseline_bounded_local_width_pp":bw0,"after_bounded_local_width_pp":bw1})
    mdf=pd.DataFrame(mrows);mdf.to_csv(OUT/"M_BOUNDED_GEOMETRY_VS_A.csv",index=False,encoding="utf-8-sig")
    rho=[]
    for (dn,k),s in mdf.groupby(["domain","measurement_kind"]):
        r,p=spearmanr(s.bounded_local_width_reduction_pct,s.A_exact_profile_width_reduction_pct,nan_policy="omit")
        rho.append({"domain":dn,"measurement_kind":k,"n":len(s),"spearman_rho":float(r),"descriptive_p_value":float(p)})
    pd.DataFrame(rho).to_csv(OUT/"M_BOUNDED_GEOMETRY_ASSOCIATION.csv",index=False,encoding="utf-8-sig")
    # Panel overlap across domain and across future stress targets.
    ov=[]
    for (view,budget),s in kdf[kdf.accounting_view.isin(["scalar_count","frequency_assay_unit"])].groupby(["accounting_view","budget"]):
      for target in ["ATP0.1","ATP1","Pi0","Pi5"]:
        a=s[(s.target==target)&(s.domain=="P1_pair_plus_0.5")]
        b=s[(s.target==target)&(s.domain=="P1_pair_plus_1.0")]
        if len(a) and len(b):
          A=set(a.iloc[0].selected_scalar_ids.split(";"));B=set(b.iloc[0].selected_scalar_ids.split(";"));ov.append({"accounting_view":view,"budget":budget,"comparison":"same_target_across_domains","target":target,"jaccard":len(A&B)/len(A|B) if A|B else np.nan,"shared_scalar_count":len(A&B)})
      for dn in ["P1_pair_plus_0.5","P1_pair_plus_1.0"]:
        panels={t:set(s[(s.target==t)&(s.domain==dn)].iloc[0].selected_scalar_ids.split(";")) for t in ["ATP0.1","ATP1","Pi0","Pi5"] if len(s[(s.target==t)&(s.domain==dn)])}
        for t1,t2 in itertools.combinations(panels,2):
          A=panels[t1];B=panels[t2];ov.append({"accounting_view":view,"budget":budget,"comparison":"cross_target_within_domain","domain":dn,"target":f"{t1} vs {t2}","jaccard":len(A&B)/len(A|B) if A|B else np.nan,"shared_scalar_count":len(A&B)})
    pd.DataFrame(ov).to_csv(OUT/"K_PANEL_OVERLAP.csv",index=False,encoding="utf-8-sig")
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for (target,dn),sub in outdf[outdf.accounting_view.eq("scalar_count")].groupby(["target","domain"]):
        axs[0].plot(sub.budget,sub.bounded_width_reduction_pct,marker="o",label=f"{target} / {dn.rsplit('_',1)[-1]}")
    axs[0].set(xlabel="Additional scalar observations",ylabel="Bounded local quadratic width reduction (%)",title="Scalar-counted panels",ylim=(0,80),xticks=[1,2,4,6,8])
    for (target,dn),sub in outdf[outdf.accounting_view.eq("frequency_assay_unit")].groupby(["target","domain"]):
        axs[1].plot(sub.budget,sub.bounded_width_reduction_pct,marker="o",label=f"{target} / {dn.rsplit('_',1)[-1]}")
    axs[1].set(xlabel="Additional assay/frequency units",ylabel="Bounded local quadratic width reduction (%)",title="One stress scalar = one CM frequency pair = one unit",ylim=(0,80),xticks=[1,2,3])
    for ax in axs:ax.legend(fontsize=7,ncol=2)
    fig.suptitle("Bounded local support inside inherited parameter boxes; not calibrated intervals")
    fig.tight_layout();fig.savefig(OUT/"K_BOUNDED_BUDGET_PERFORMANCE.png",dpi=180);plt.close(fig)
    print(json.dumps({"bounded_rows":len(outdf),"full_spectrum_rows":len(fulldf),"M_bounded_comparisons":len(mdf),"M_bounded_rho":rho,"overlap_rows":len(ov)},indent=2))

if __name__=="__main__":main()
