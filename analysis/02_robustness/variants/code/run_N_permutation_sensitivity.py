"""Preselected published-fit-ranked structural sensitivity for P2 A/K."""
from __future__ import annotations
import itertools,json,sys,hashlib
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import lsq_linear,minimize
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent; N=HERE.parent; INP=N/"input_public"; OUT=N/"results"; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE));sys.path.insert(0,str(N.parents[0]/"K"/"code"))
import permutation_model as pm
from p2.constants import LOWER,UPPER,CONDITIONS,LR95_DF1,PARAM_NAMES

TARGETS=["ATP0.1","ATP1","Pi0","Pi5"]
CONDS=["ATP0.1","ATP1","Pi0","Pi5"]
STEP=2e-5
MODEL_VARIANTS=[
 {"variant":"published_Model16D","s":16,"md":4,"selection_reason":"reported final_fit.mat final_params; source row 16, md4"},
 {"variant":"md4_best_alternative","s":14,"md":4,"selection_reason":"lowest source OBJs among md4 structures other than the reported Model16D"},
 {"variant":"md4_second_best_alternative","s":9,"md":4,"selection_reason":"second-lowest source OBJs among md4 structures other than reported Model16D and md4 best"},
 {"variant":"md4_best_single_strain","s":5,"md":4,"selection_reason":"lowest source OBJs among one-strain md4 structures; complexity-class diversity"},
 {"variant":"md4_best_without_kminus2_strain","s":4,"md":4,"selection_reason":"lowest source OBJs among md4 structures without k_minus_2 strain; prespecified mechanistic diversity"},
]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def sd_from_s(s):
    sd=np.zeros(5,int)
    if s<=1: return sd
    if s<=6: idx=[s-2]
    elif s<=10: idx=[0,s-6]
    elif s<=13: idx=[1,s-9]
    elif s<=15: idx=[2,s-11]
    else: idx=[3,4]
    sd[idx]=1
    return sd

def read_rat():
    m=loadmat(INP/"rat_data.mat",squeeze_me=True,struct_as_record=False);a=np.asarray(m["data"],object)
    labels=[str(x) for x in a[0,1:]];col={x:j+1 for j,x in enumerate(labels)};rows={str(a[i,0]):i for i in range(1,a.shape[0])}
    d={"freqs":np.asarray(m["freqs"],float).ravel(),"cm":{},"se_re":{},"se_im":{},"stress":{},"stress_se":{}}
    aliases={"baseline":"Baseline","ATP0.1":"Super-Low ATP","ATP1":"Low ATP","Pi0":"Low Pi","Pi5":"High Pi"}
    for k,label in aliases.items():
        j=col[label];d["cm"][k]=np.asarray(a[rows["mean CM"],j],complex).ravel();d["se_re"][k]=np.asarray(a[rows["EM SE"],j],float).ravel();d["se_im"][k]=np.asarray(a[rows["VM SE"],j],float).ravel();d["stress"][k]=float(a[rows["mean F0"],j]);d["stress_se"][k]=float(a[rows["F0 SE"],j])
    return d

def nonlinear_indices(sd):
    idx=list(range(8));n=int(np.sum(sd))
    if n>=1:idx.append(10)
    if n>=2:idx.append(11)
    idx.extend([12,13]);return np.asarray(idx,int)

def zvec(p,idx):return np.log(np.maximum(np.asarray(p,float)[idx],1e-300))
def p_from_u(u,idx):
    lo=np.log(np.maximum(LOWER[idx],1e-300));hi=np.log(UPPER[idx]);p=np.empty(14,float);p[:]=np.nan
    p[idx]=np.exp(lo+np.asarray(u,float)*(hi-lo));p[9]=0.0;p[10:12]=0.0
    return p

def scales_and_residual(nl,idx,sd,d):
    tmpl=np.zeros(14,float);tmpl[idx]=nl;freq=d["freqs"]
    yk,yks,fk=pm.bases(nl,tmpl,*CONDITIONS["baseline"],freq,sd,idx)
    sr=d["se_re"]["baseline"];si=d["se_im"]["baseline"];sf=d["stress_se"]["baseline"]
    A=np.vstack([np.column_stack([yk.real/sr,yks.real/sr]),np.column_stack([yk.imag/si,np.zeros(len(freq))]),np.asarray([[fk/sf,.3/sf]])])
    y=np.concatenate([d["cm"]["baseline"].real/sr,d["cm"]["baseline"].imag/si,np.asarray([d["stress"]["baseline"]/sf])])
    fit=lsq_linear(A,y,bounds=([LOWER[8],LOWER[9]],[UPPER[8],UPPER[9]]),tol=1e-10,max_iter=200)
    p=tmpl.copy();p[8:10]=fit.x
    pred,fs=pm.predict(p,*CONDITIONS["baseline"],freq,sd)
    r=np.concatenate([(pred.real-d["cm"]["baseline"].real)/sr,(pred.imag-d["cm"]["baseline"].imag)/si,np.asarray([(fs-d["stress"]["baseline"])/sf])])
    return p,r,float(r@r)

def fit_variant(sd,variant,starts,d):
    idx=nonlinear_indices(sd);lo=np.log(LOWER[idx]);hi=np.log(UPPER[idx]);records=[];cands=[]
    for record in starts:
        p0=np.asarray(record["params"],float).copy();n=int(np.sum(sd))
        if n==0:p0[10:12]=0
        elif n==1:
            p0[10]=max(p0[10],1.0);p0[11]=0
        else:
            p0[10]=max(p0[10],1.0);p0[11]=max(p0[11],1.0)
        p0[idx]=np.clip(p0[idx],LOWER[idx],UPPER[idx]);u0=np.clip((np.log(p0[idx])-lo)/(hi-lo),0,1)
        def obj(u):
            nl=np.exp(lo+np.clip(u,0,1)*(hi-lo))
            try:return scales_and_residual(nl,idx,sd,d)[2]
            except (ValueError,FloatingPointError,OverflowError,ZeroDivisionError):return 1e100
        init=obj(u0)
        opt=minimize(obj,u0,method="L-BFGS-B",bounds=[(0,1)]*len(idx),options={"maxiter":1200,"ftol":1e-11,"gtol":1e-7,"maxls":40})
        nl=np.exp(lo+np.clip(opt.x,0,1)*(hi-lo));p,r,exact=scales_and_residual(nl,idx,sd,d)
        records.append({"start":record["name"],"initial_objective":init,"final_objective":exact,"optimizer_success":bool(opt.success),"message":str(opt.message),"iterations":int(getattr(opt,"nit",0))})
        cands.append((exact,p,nl,str(record["name"]),bool(opt.success)))
    best=min(cands,key=lambda x:x[0]);return {"variant":variant,"sd":sd.tolist(),"free_nonlinear_indices":idx.tolist(),"params":best[1],"nonlinear":best[2],"objective":best[0],"winner":best[3],"success":best[4],"starts":records}

def all_values(u,fit,d,idx,sd,base_layout,cand_layout):
    lo=np.log(LOWER[idx]);hi=np.log(UPPER[idx]);nl=np.exp(lo+np.asarray(u,float)*(hi-lo));p,_,_=scales_and_residual(nl,idx,sd,d)
    values=[]
    for _,cond,i,comp,se in base_layout:
        y,f=pm.predict(p,*CONDITIONS[cond],d["freqs"],sd)
        v=f if comp=="stress" else (y[i].real if comp=="Re" else y[i].imag)
        values.append(float(v)/float(se))
    for _,kind,cond,i,comp,se in cand_layout:
        y,f=pm.predict(p,*CONDITIONS[cond],d["freqs"],sd)
        v=f if comp=="stress" else (y[i].real if comp=="Re" else y[i].imag)
        values.append(float(v)/float(se))
    values.extend([100*(pm.predict(p,*CONDITIONS[t],np.array([1.0]),sd)[1]/pm.predict(p,*CONDITIONS["baseline"],np.array([1.0]),sd)[1]-1) for t in TARGETS])
    return np.asarray(values,float),p

def panel_var(V,g,rows):
    if len(rows)==0:return max(0,float(g@V@g))
    R=np.asarray(rows,float);A=np.eye(len(R))+R@V@R.T;c=R@V@g
    return max(0.0,float(g@V@g)-float(c@np.linalg.solve(A,c)))

def action_list(target,cand_layout,Jc,freqs):
    pos={x[0]:i for i,x in enumerate(cand_layout)};actions=[];scalars=[]
    for cond in CONDS:
        if cond==target:continue
        sid=f"stress_{cond}";actions.append({"action_id":sid,"type":"stress","condition":cond,"freq_hz":None,"scalar_ids":[sid],"rows":[Jc[pos[sid]] ]});scalars.append({"scalar_id":sid,"type":"stress","unit_id":sid,"row":Jc[pos[sid]],"freq_hz":None})
        for i,f in enumerate(freqs):
            rid=f"CM_{cond}_{f:g}Hz_Re";iid=f"CM_{cond}_{f:g}Hz_Im";unit=f"CM_{cond}_{f:g}Hz"
            rows=[Jc[pos[rid]],Jc[pos[iid]]]
            actions.append({"action_id":unit,"type":"CM_frequency_pair","condition":cond,"freq_hz":float(f),"scalar_ids":[rid,iid],"rows":rows})
            scalars.extend([{"scalar_id":rid,"type":"CM_scalar","unit_id":unit,"row":rows[0],"freq_hz":float(f)},{"scalar_id":iid,"type":"CM_scalar","unit_id":unit,"row":rows[1],"freq_hz":float(f)}])
    return actions,scalars

def main():
    d=read_rat();mat=loadmat(INP/"final_fit.mat",squeeze_me=True,struct_as_record=False)
    OBJs=np.asarray(mat["OBJs"],float);xs=np.asarray(mat["xs"],dtype=object);final=np.asarray(mat["final_params"],float).ravel()
    starts=json.loads((INP/"A_E_F_J_start_vectors.json").read_text(encoding="utf-8"))["records"]
    # Preselection is generated from source fit objectives only; no P2 K/A outcomes enter this list.
    rank=[]
    for s in range(1,17):
      sd=sd_from_s(s)
      for md in range(1,5):rank.append({"s_index":s,"md":md,"sd":"".join(map(str,sd.tolist())),"n_active_strain_rates":int(sd.sum()),"source_objective":float(OBJs[s-1,md-1]),"source_rank_within_md4":None,"delta_from_best_md4":None})
    md4=sorted([x for x in rank if x["md"]==4],key=lambda x:x["source_objective"])
    for rnk,x in enumerate(md4,1):x["source_rank_within_md4"]=rnk;x["delta_from_best_md4"]=x["source_objective"]-md4[0]["source_objective"]
    rankmap={(x["s"],x["md"]):x for x in MODEL_VARIANTS}
    for x in rank:x["selected_for_P2_N"]=int((x["s_index"],x["md"]) in rankmap)
    pd.DataFrame(rank).to_csv(OUT/"N_PRESELECTION_RANKING.csv",index=False,encoding="utf-8-sig")
    base_layout=[];cand_layout=[]
    for i,f in enumerate(d["freqs"]):
        base_layout.extend([(f"base_re_{f:g}","baseline",i,"Re",d["se_re"]["baseline"][i]),(f"base_im_{f:g}","baseline",i,"Im",d["se_im"]["baseline"][i])])
    base_layout.append(("base_stress","baseline",0,"stress",d["stress_se"]["baseline"]))
    for cond in CONDS:
        cand_layout.append((f"stress_{cond}","stress",cond,0,"stress",d["stress_se"][cond]))
        for i,f in enumerate(d["freqs"]):cand_layout.extend([(f"CM_{cond}_{f:g}Hz_Re","cm",cond,i,"Re",d["se_re"][cond][i]),(f"CM_{cond}_{f:g}Hz_Im","cm",cond,i,"Im",d["se_im"][cond][i])])
    fits=[];sens=[];grads=[];bases_out=[];actionrows=[];panelrows=[]
    fitmap={}
    for v in MODEL_VARIANTS:
        s=v["s"];sd=sd_from_s(s);fit=fit_variant(sd,v["variant"],starts,d);fit["s_index"]=s;fit["md"]=4;fit["source_objective_all_conditions"]=float(OBJs[s-1,3]);fit["source_md4_rank"]=next(x["source_rank_within_md4"] for x in md4 if x["s_index"]==s)
        fit["published_final_params_match"]=(s==16 and np.allclose(final,np.asarray(xs[15,3],float).ravel(),rtol=0,atol=1e-7))
        fitmap[v["variant"]]=fit
        fits.append({"variant":v["variant"],"s_index":s,"md":4,"sd":"".join(map(str,sd.tolist())),"n_active_strain_rates":int(sd.sum()),"source_rank_within_md4":fit["source_md4_rank"],"published_all_condition_objective":fit["source_objective_all_conditions"],"baseline_only_P2_objective":fit["objective"],"best_start":fit["winner"],"fit_success":fit["success"],"n_starts":len(fit["starts"]),"successful_starts":sum(x["optimizer_success"] for x in fit["starts"]),"best_start_range":max(x["final_objective"] for x in fit["starts"])-min(x["final_objective"] for x in fit["starts"]),"parameter_vector":";".join(f"{x:.12g}" for x in fit["params"]),"P2_target_data_used_for_fit":False})
        idx=np.asarray(fit["free_nonlinear_indices"],int);lo=np.log(LOWER[idx]);hi=np.log(UPPER[idx]);u0=np.clip((np.log(fit["params"][idx])-lo)/(hi-lo),0,1)
        # Finite differences in the published global fitting box, only free coordinates included.
        f0,p0=all_values(u0,fit,d,idx,sd,base_layout,cand_layout);J=np.empty((len(f0),len(u0)))
        for j in range(len(u0)):
            if u0[j]<=STEP:
                up=u0.copy();up[j]+=STEP;J[:,j]=(all_values(up,fit,d,idx,sd,base_layout,cand_layout)[0]-f0)/STEP
            elif u0[j]>=1-STEP:
                dn=u0.copy();dn[j]-=STEP;J[:,j]=(f0-all_values(dn,fit,d,idx,sd,base_layout,cand_layout)[0])/STEP
            else:
                up=u0.copy();dn=u0.copy();up[j]+=STEP;dn[j]-=STEP;J[:,j]=(all_values(up,fit,d,idx,sd,base_layout,cand_layout)[0]-all_values(dn,fit,d,idx,sd,base_layout,cand_layout)[0])/(2*STEP)
        nb=len(base_layout);nc=len(cand_layout);Jb=J[:nb];Jc=J[nb:nb+nc]
        for i,(lab,*_) in enumerate(base_layout):
            for j,x in enumerate(Jb[i]):bases_out.append({"variant":v["variant"],"row_id":lab,"parameter_index":int(idx[j]),"local_coordinate":j,"sensitivity":x})
        for tpos,t in enumerate(TARGETS):
            g=J[nb+nc+tpos];
            for j,x in enumerate(g):grads.append({"variant":v["variant"],"target":t,"parameter_index":int(idx[j]),"local_coordinate":j,"target_gradient":x})
            for i,entry in enumerate(cand_layout):
                sid=entry[0]
                for j,x in enumerate(Jc[i]):sens.append({"variant":v["variant"],"target":t,"scalar_id":sid,"condition":entry[2],"type":entry[1],"component":entry[4],"local_coordinate":j,"parameter_index":int(idx[j]),"whitened_sensitivity":x})
            H=Jb.T@Jb;V=np.linalg.pinv(H,rcond=1e-11);v0=max(0,float(g@V@g));actions,scalars=action_list(t,cand_layout,Jc,d["freqs"])
            for a in actions:
                var=panel_var(V,g,a["rows"]);actionrows.append({"variant":v["variant"],"target":t,"action_id":a["action_id"],"type":a["type"],"condition":a["condition"],"frequency_hz":a["freq_hz"],"variance_reduction":v0-var,"relative_variance_reduction_pct":100*(1-var/v0) if v0 else np.nan})
            for budget in [1,2]:
                candidates=[]
                for chosen in itertools.combinations(actions,budget):
                    R=[r for a in chosen for r in a["rows"]];vv=panel_var(V,g,R);candidates.append((vv,chosen))
                vv,chosen=min(candidates,key=lambda x:x[0]);panelrows.append({"variant":v["variant"],"target":t,"accounting_view":"assay_unit","budget":budget,"panel_ids":";".join(a["action_id"] for a in chosen),"scalar_ids":";".join(s for a in chosen for s in a["scalar_ids"]),"selected_frequencies_hz":";".join(f"{a['freq_hz']:g}" for a in chosen if a["freq_hz"] is not None),"stress_selected":any(a["type"]=="stress" for a in chosen),"baseline_variance":v0,"after_panel_variance":vv,"local_width_reduction_pct":100*(1-np.sqrt(vv/v0)) if v0 else np.nan,"fraction_best_single_full_CM_spectrum_utility":np.nan,"exact_exhaustive":True})
            # Reduced A-like comparison: all eligible stress blocks vs each full 17-frequency complex spectrum.
            full=[]
            for cond in [c for c in CONDS if c!=t]:
                a=next(a for a in actions if a["action_id"]==f"stress_{cond}");v_st=panel_var(V,g,a["rows"])
                ids=[f"CM_{cond}_{f:g}Hz_{comp}" for f in d["freqs"] for comp in ("Re","Im")];rr=[next(x["row"] for x in scalars if x["scalar_id"]==sid) for sid in ids]
                v_cm=panel_var(V,g,rr);full.append(v0-v_cm)
                panelrows.append({"variant":v["variant"],"target":t,"accounting_view":"A_full_assay_reference","budget":1,"panel_ids":f"stress_{cond}","scalar_ids":f"stress_{cond}","selected_frequencies_hz":"","stress_selected":True,"baseline_variance":v0,"after_panel_variance":v_st,"local_width_reduction_pct":100*(1-np.sqrt(v_st/v0)) if v0 else np.nan,"exact_exhaustive":True})
                panelrows.append({"variant":v["variant"],"target":t,"accounting_view":"A_full_assay_reference","budget":34,"panel_ids":f"CM_spectrum_{cond}","scalar_ids":";".join(ids),"selected_frequencies_hz":"all17","stress_selected":False,"baseline_variance":v0,"after_panel_variance":v_cm,"local_width_reduction_pct":100*(1-np.sqrt(v_cm/v0)) if v0 else np.nan,"exact_exhaustive":True})
    fdf=pd.DataFrame(fits);adf=pd.DataFrame(actionrows);pdf=pd.DataFrame(panelrows);gdf=pd.DataFrame(grads);sdf=pd.DataFrame(sens);bdf=pd.DataFrame(bases_out)
    # Fill full-spectrum utility fractions and rank candidate actions against the published Model16D baseline.
    single=pdf[pdf.accounting_view.eq("A_full_assay_reference")&pdf.panel_ids.str.startswith("CM_spectrum_")].groupby(["variant","target"]).after_panel_variance.min()
    for idx0,row in pdf.iterrows():
        if row.accounting_view=="assay_unit":
            bestvar=single[(row.variant,row.target)] if (row.variant,row.target) in single else np.nan
            bestutil=row.baseline_variance-bestvar if np.isfinite(bestvar) else np.nan
            pdf.loc[idx0,"fraction_best_single_full_CM_spectrum_utility"]=(row.baseline_variance-row.after_panel_variance)/bestutil if bestutil and bestutil>0 else np.nan
    pd.DataFrame(fits).drop(columns=["params"],errors="ignore").to_csv(OUT/"N_VARIANT_FITS.csv",index=False,encoding="utf-8-sig")
    adf.to_csv(OUT/"N_A_STYLE_ACTION_UTILITIES.csv",index=False,encoding="utf-8-sig")
    pdf.to_csv(OUT/"N_REDUCED_AK_PANEL_RESULTS.csv",index=False,encoding="utf-8-sig")
    gdf.to_csv(OUT/"N_TARGET_GRADIENTS.csv",index=False,encoding="utf-8-sig")
    sdf.to_csv(OUT/"N_LOCAL_SENSITIVITIES.csv",index=False,encoding="utf-8-sig")
    bdf.to_csv(OUT/"N_BASELINE_SENSITIVITIES.csv",index=False,encoding="utf-8-sig")
    # Candidate rank stability and best 1-unit/2-unit panel overlap against the published Model16D.
    stability=[]
    for t in TARGETS:
      ref=adf[(adf.variant=="published_Model16D")&(adf.target==t)].set_index("action_id").relative_variance_reduction_pct
      for variant in fdf.variant:
        sub=adf[(adf.variant==variant)&(adf.target==t)].set_index("action_id").relative_variance_reduction_pct
        a=ref.align(sub,join="inner");rho,p=spearmanr(a[0],a[1],nan_policy="omit")
        p1=pdf[(pdf.variant==variant)&(pdf.target==t)&(pdf.accounting_view=="assay_unit")&(pdf.budget==1)].iloc[0]
        r1=pdf[(pdf.variant=="published_Model16D")&(pdf.target==t)&(pdf.accounting_view=="assay_unit")&(pdf.budget==1)].iloc[0]
        p2=pdf[(pdf.variant==variant)&(pdf.target==t)&(pdf.accounting_view=="assay_unit")&(pdf.budget==2)].iloc[0]
        r2=pdf[(pdf.variant=="published_Model16D")&(pdf.target==t)&(pdf.accounting_view=="assay_unit")&(pdf.budget==2)].iloc[0]
        set1=set(p1.scalar_ids.split(";"));setr1=set(r1.scalar_ids.split(";"));set2=set(p2.scalar_ids.split(";"));setr2=set(r2.scalar_ids.split(";"))
        stability.append({"target":t,"variant":variant,"spearman_rank_correlation_of_single_action_utilities_vs_published_Model16D":float(rho),"descriptive_p":float(p),"best_unit_panel_overlap_K1_scalar_jaccard":len(set1&setr1)/len(set1|setr1) if set1|setr1 else np.nan,"best_unit_panel_overlap_K2_scalar_jaccard":len(set2&setr2)/len(set2|setr2) if set2|setr2 else np.nan,"same_exact_K1_unit_panel":p1.panel_ids==r1.panel_ids,"same_exact_K2_unit_panel":p2.panel_ids==r2.panel_ids})
    pd.DataFrame(stability).to_csv(OUT/"N_MODEL_RANK_STABILITY.csv",index=False,encoding="utf-8-sig")
    meta={"status":"COMPLETED","selection_was_predeclared_from_P2_outputs":False,"source_model_rank_file_sha256":sha(INP/"final_fit.mat"),"rat_data_sha256":sha(INP/"rat_data.mat"),"number_variants":len(MODEL_VARIANTS),"fit_data":"baseline CM and baseline steady stress only; published group SEM diagonal likelihood","parameter_domain":"published full optimizer box; source-disabled strain factors fixed to zero; P1 pair box not imposed on distinct structures","metabolite_mode":"fixed md=4 for every selected structure","utility":"expected local Gauss-Newton/Fisher variance reduction; not exact profile likelihood and no structural adequacy verdict","variant_rule":"reported final Model16D plus source-objective-ranked alternatives and predeclared rate-strain-diverse structure"}
    (OUT/"N_RUN_METADATA.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps({"variants":len(fits),"A_action_rows":len(adf),"panel_rows":len(pdf),"sensitivity_rows":len(sdf),"out":str(OUT)},indent=2))

if __name__=="__main__":main()
