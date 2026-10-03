"""K sparse panels + L sequential designs + M local explanation + P CM targets.

All utilities are local expected-information calculations conditional on the
frozen public rat Model16D fit. See branch reports before interpreting them as
anything other than a local design forecast.
"""
from __future__ import annotations
import itertools, json, sys, hashlib
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE / "code"))
from p2.constants import CONDITIONS, PARAM_NAMES, LR95_DF1
from p2.model16d import predict, q_delta_stress

INP = HERE.parent / "input"
OUT = HERE.parent / "results"
OUT.mkdir(exist_ok=True)
TARGETS = ["ATP0.1", "ATP1", "Pi0", "Pi5"]
CONDS = ["ATP0.1", "ATP1", "Pi0", "Pi5"]
DOMAIN_NAMES = ["P1_pair_plus_0.5", "P1_pair_plus_1.0"]
STEP = 2e-5

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def load_data():
    mat = loadmat(INP / "rat_data.mat", squeeze_me=True, struct_as_record=False)
    cell = np.asarray(mat["data"], dtype=object)
    labels = [str(x) for x in cell[0, 1:]]
    col = {name: j + 1 for j, name in enumerate(labels)}
    rows = {str(cell[i, 0]): i for i in range(1, cell.shape[0])}
    freqs = np.asarray(mat["freqs"], float).ravel()
    d = {"freqs": freqs, "cm": {}, "se_re": {}, "se_im": {}, "stress": {}, "stress_se": {}}
    m = {"baseline":"Baseline", "ATP0.1":"Super-Low ATP", "ATP1":"Low ATP", "Pi0":"Low Pi", "Pi5":"High Pi"}
    for k, label in m.items():
        j = col[label]
        d["cm"][k] = np.asarray(cell[rows["mean CM"], j], complex).ravel()
        d["se_re"][k] = np.asarray(cell[rows["EM SE"], j], float).ravel()
        d["se_im"][k] = np.asarray(cell[rows["VM SE"], j], float).ravel()
        d["stress"][k] = float(cell[rows["mean F0"], j])
        d["stress_se"][k] = float(cell[rows["F0 SE"], j])
    return d

def z_of_p(p):
    p = np.asarray(p, float); z = np.log(np.maximum(p,1e-300)); z[9] = p[9] / 45.0
    return z

def p_of_u(u, zlo, zhi):
    z = zlo + np.asarray(u, float) * (zhi - zlo)
    p = np.exp(z); p[9] = z[9] * 45.0
    return p

def all_model_outputs(u, domain, d):
    p = p_of_u(u, domain["zlo"], domain["zhi"])
    out = {}
    for cond in ["baseline", *CONDS]:
        y, f = predict(p, *CONDITIONS[cond], d["freqs"])
        out[("cm", cond)] = y
        out[("stress", cond)] = f
    out[("q", "baseline")] = np.array([q_delta_stress(p, CONDITIONS[t], CONDITIONS["baseline"]) for t in TARGETS])
    return out

def scalar_layout(d):
    layout=[]
    # Existing baseline likelihood coordinates.
    for i,f in enumerate(d["freqs"]):
        layout.append(("base_re", f"baseline_{f:g}Hz_Re", "baseline", i, "Re", d["se_re"]["baseline"][i]))
        layout.append(("base_im", f"baseline_{f:g}Hz_Im", "baseline", i, "Im", d["se_im"]["baseline"][i]))
    layout.append(("base_stress", "baseline_stress", "baseline", 0, "stress", d["stress_se"]["baseline"]))
    cand=[]
    for cond in CONDS:
        cand.append((f"stress_{cond}", "stress", cond, 0, "stress", d["stress_se"][cond]))
        for i,f in enumerate(d["freqs"]):
            cand.append((f"CM_{cond}_{f:g}Hz_Re", "cm", cond, i, "Re", d["se_re"][cond][i]))
            cand.append((f"CM_{cond}_{f:g}Hz_Im", "cm", cond, i, "Im", d["se_im"][cond][i]))
    return layout, cand

def prediction_vector(u, domain, d, base_layout, cand_layout):
    o = all_model_outputs(u, domain, d); vals=[]
    for _,_,cond,i,component,se in base_layout:
        x = o[("stress",cond)] if component=="stress" else (o[("cm",cond)][i].real if component=="Re" else o[("cm",cond)][i].imag)
        vals.append(float(x)/float(se))
    for _,typ,cond,i,component,se in cand_layout:
        x = o[(typ,cond)] if typ=="stress" else (o[(typ,cond)][i].real if component=="Re" else o[(typ,cond)][i].imag)
        vals.append(float(x)/float(se))
    vals.extend(o[("q","baseline")].tolist())
    return np.asarray(vals, float)

def jacobian_at(u0, domain, d, base_layout, cand_layout):
    x0=np.asarray(u0,float); f0=prediction_vector(x0,domain,d,base_layout,cand_layout)
    J=np.empty((len(f0),len(x0)),float)
    for j in range(len(x0)):
        h=STEP
        if x0[j] <= h:
            x=x0.copy(); x[j]+=h
            J[:,j]=(prediction_vector(x,domain,d,base_layout,cand_layout)-f0)/h
        elif x0[j] >= 1-h:
            x=x0.copy(); x[j]-=h
            J[:,j]=(f0-prediction_vector(x,domain,d,base_layout,cand_layout))/h
        else:
            xp=x0.copy(); xm=x0.copy(); xp[j]+=h; xm[j]-=h
            J[:,j]=(prediction_vector(xp,domain,d,base_layout,cand_layout)-prediction_vector(xm,domain,d,base_layout,cand_layout))/(2*h)
    return J

def inv_info(H):
    H=(H+H.T)/2
    return np.linalg.pinv(H,rcond=1e-11)

def qvar(V,g): return max(0.0,float(g@V@g))

def panel_variance(V,g,rows):
    if len(rows)==0: return qvar(V,g)
    R=np.asarray(rows,float); RV=R@V; A=np.eye(len(R))+RV@R.T; c=R@V@g
    try: red=float(c@np.linalg.solve(A,c))
    except np.linalg.LinAlgError: red=float(c@np.linalg.pinv(A)@c)
    return max(0.0,qvar(V,g)-max(0.0,red))

def width(v): return 2*np.sqrt(LR95_DF1*max(0.0,v))

def action_map(cand_layout, Jcand, target):
    byid={id_:np.asarray(Jcand[i],float) for i,(id_,*_) in enumerate(cand_layout)}
    valid=[x for x in cand_layout if target is None or x[2]!=target]
    actions=[]; scalar=[]
    for row in valid:
        id_,typ,cond,i,comp,se=row
        if typ=="stress":
            actions.append({"action_id":id_,"type":"stress","condition":cond,"freq":None,"scalar_ids":[id_],"rows":[byid[id_]]})
            scalar.append({"scalar_id":id_,"condition":cond,"unit_id":id_,"type":"stress","component":"stress","freq":None,"row":byid[id_]})
    freqs=sorted(set(float(x[3] and 0) for x in []))
    for cond in CONDS:
        if cond==target: continue
        for i,row in enumerate([x for x in valid if x[1]=="cm" and x[2]==cond and x[4]=="Re"]):
            f=row[3] # frequency index; display frequency is added by caller
            re_id=row[0]; im_id=f"CM_{cond}_{float(FREQS[i]):g}Hz_Im"
            # IDs use the exact same source frequency order.
            re=row; im=next(x for x in valid if x[0]==im_id)
            unit=f"CM_{cond}_{float(FREQS[i]):g}Hz"
            action={"action_id":unit,"type":"CM_frequency_pair","condition":cond,"freq_hz":float(FREQS[i]),"scalar_ids":[re_id,im_id],"rows":[byid[re_id],byid[im_id]]}
            actions.append(action)
            for comp,entry in (("Re",re),("Im",im)):
                scalar.append({"scalar_id":entry[0],"condition":cond,"unit_id":unit,"type":"CM_scalar","component":comp,"freq_hz":float(FREQS[i]),"row":byid[entry[0]]})
    return actions, scalar

def exact_best(V,g,items,k,rows_of=lambda x:x["rows"],exclude_overlap=True):
    best=None; base=qvar(V,g)
    for chosen in itertools.combinations(items,k):
        rows=[r for x in chosen for r in rows_of(x)]
        v=panel_variance(V,g,rows); gain=base-v
        if best is None or gain>best[0]: best=(gain,chosen,v)
    return best

def greedy(V,g,items,k,rows_of=lambda x:x["rows"],cost_of=lambda x:1):
    chosen=[]; avail=list(items); panels=[]; curV=V.copy()
    for step in range(k):
        best=None
        for x in avail:
            rows=[r for y in [*chosen,x] for r in rows_of(y)]
            v=panel_variance(V,g,rows)
            score=(qvar(V,g)-v)/sum(cost_of(y) for y in [*chosen,x])
            # Rank by total utility at the resulting cardinality, not a per-step ratio.
            xid=x.get("action_id",x.get("scalar_id",""))
            bid=(best[1].get("action_id",best[1].get("scalar_id","")) if best is not None else "")
            if best is None or (v < best[0]-1e-14) or (abs(v-best[0])<1e-14 and xid<bid): best=(v,x)
        if best is None: break
        x=best[1]; chosen.append(x); avail.remove(x)
        rows=[r for y in chosen for r in rows_of(y)]
        panels.append((list(chosen),panel_variance(V,g,rows)))
    return panels

def rows_from_scalar(s): return [s["row"]]

def add_record(records, target, domain, view, budget, selected, v0, v_after, full_u, exact=None, note=""):
    acts=[a["action_id"] for a in selected]
    scalar_ids=[s for a in selected for s in a["scalar_ids"]] if selected and "action_id" in selected[0] else [x["scalar_id"] for x in selected]
    nscalar=sum(len(a["scalar_ids"]) for a in selected) if selected and "scalar_ids" in selected[0] else len(selected)
    records.append({"target":target,"domain":domain,"accounting_view":view,"budget":budget,
       "selected_action_ids":";".join(acts),"selected_scalar_ids":";".join(scalar_ids),
       "n_assay_units":len(selected),"n_scalar_observations":nscalar,
       "baseline_local_wald_width_pp":width(v0),"after_panel_local_wald_width_pp":width(v_after),
       "width_reduction_pct":100*(1-np.sqrt(v_after/v0)) if v0>0 else np.nan,
       "fraction_best_single_full_CM_spectrum_utility_recovered":(v0-v_after)/full_u if full_u>0 else np.nan,
       "stress_selected":any(a["type"]=="stress" for a in selected) if selected and "type" in selected[0] else False,
       "frequencies_hz":";".join(f"{a['freq_hz']:g}" for a in selected if a.get("freq_hz") is not None) if selected and "freq_hz" in selected[0] else "",
       "exact_search_or_gap_pct":exact,"note":note})

def main():
    global FREQS
    d=load_data(); FREQS=d["freqs"]
    am=json.loads((INP/"A_measurement_target_map.json").read_text(encoding="utf-8"))
    domains={}
    for dn in DOMAIN_NAMES:
        dm=am["domains"][dn]["domain"]
        lo=np.asarray(dm["lo"],float); hi=np.asarray(dm["hi"],float)
        # Same transformed coordinates as the frozen A analysis.
        zlo=z_of_p(lo); zhi=z_of_p(hi)
        fit=np.asarray(am["domains"][dn]["baseline_fit"]["params"],float)
        uf=np.clip((z_of_p(fit)-zlo)/(zhi-zlo),0,1)
        domains[dn]={"zlo":zlo,"zhi":zhi,"p":fit,"u":uf}
    base_layout,cand_layout=scalar_layout(d)
    # Write finite-difference sensitivities for independent downstream recomputation.
    sens_rows=[]; grad_rows=[]; base_rows=[]; records=[]; heat=[]; full_rows=[]; seq_rows=[]; seq_meta=[]; P_all=[]; pgrad_rows=[]
    local_cache={}
    for dn,dom in domains.items():
      for target in TARGETS:
        J=jacobian_at(dom["u"],dom,d,base_layout,cand_layout)
        nb=len(base_layout); nc=len(cand_layout)
        Jbase=J[:nb]; Jcand=J[nb:nb+nc]; qidx=nb+nc+TARGETS.index(target); g=J[qidx]
        H=Jbase.T@Jbase; V=inv_info(H); v0=qvar(V,g)
        residual_span=np.linalg.norm(g - (g@V@H)) / max(np.linalg.norm(g),1e-15)
        for i,(_,row_id,*_) in enumerate(base_layout):
          for j,val in enumerate(Jbase[i]): base_rows.append({"domain":dn,"target":target,"row_id":row_id,"parameter_index":j,"sensitivity":val})
        for j,val in enumerate(g): grad_rows.append({"domain":dn,"target":target,"parameter_index":j,"target_gradient":val})
        for i,(sid,typ,cond,fi,comp,se) in enumerate(cand_layout):
          for j,val in enumerate(Jcand[i]): sens_rows.append({"domain":dn,"target":target,"scalar_id":sid,"measurement_type":typ,"condition":cond,"frequency_index":fi if typ=="cm" else "","component":comp,"parameter_index":j,"whitened_sensitivity":val})
        # Unit/individual scalar candidate sets exclude same-condition target.
        actions, scalars=action_map(cand_layout,Jcand,target)
        for a in actions: a["type"]=a["type"]
        V0=v0
        # Full single-assay references across all eligible conditions.
        for cond in [c for c in CONDS if c!=target]:
          st=next(a for a in actions if a["action_id"]==f"stress_{cond}")
          cm=[a for a in actions if a["condition"]==cond and a["type"]=="CM_frequency_pair"]
          v_st=panel_variance(V,g,st["rows"])
          v_cm=panel_variance(V,g,[r for a in cm for r in a["rows"]])
          full_rows.append({"domain":dn,"target":target,"condition":cond,"stress_utility_variance_reduction":V0-v_st,
                            "CM_spectrum_utility_variance_reduction":V0-v_cm,"CM_spectrum_width_reduction_pct":100*(1-np.sqrt(v_cm/V0)) if V0 else np.nan})
        full_u=max(x["CM_spectrum_utility_variance_reduction"] for x in full_rows if x["domain"]==dn and x["target"]==target)
        local_cache[(dn,target)]={"V":V,"g":g,"v0":v0,"actions":actions,"scalars":scalars,"full_u":full_u,"J":J,"residual_span":residual_span}

        # Scalar budget exact K=1 and K=2, followed by deterministic greedy K=4,6,8.
        for budget in [1,2,4,6,8]:
          gs=greedy(V,g,scalars,budget,rows_of=rows_from_scalar)
          chosen=gs[-1][0]; v=gs[-1][1]
          gap=None; note="deterministic greedy marginal target-variance reduction"
          if budget==1:
            exact=exact_best(V,g,scalars,1,rows_of=rows_from_scalar)
            gap=100*(v-exact[2])/(v0-exact[2]) if v0>exact[2] else 0
            note="one-scalar exhaustive optimum"
          elif budget==2:
            exact=exact_best(V,g,scalars,2,rows_of=rows_from_scalar)
            gap=100*(v-exact[2])/(v0-exact[2]) if v0>exact[2] else 0
            note="greedy K=2 compared with exhaustive scalar-pair optimum"
          records.append({"target":target,"domain":dn,"accounting_view":"scalar_count","budget":budget,
             "selected_action_ids":";".join(x["unit_id"]+":"+x["component"] for x in chosen),
             "selected_scalar_ids":";".join(x["scalar_id"] for x in chosen),"n_assay_units":len(set(x["unit_id"] for x in chosen)),
             "n_scalar_observations":len(chosen),"baseline_local_wald_width_pp":width(v0),"after_panel_local_wald_width_pp":width(v),
             "width_reduction_pct":100*(1-np.sqrt(v/v0)) if v0>0 else np.nan,
             "fraction_best_single_full_CM_spectrum_utility_recovered":(v0-v)/full_u if full_u>0 else np.nan,
             "stress_selected":any(x["type"]=="stress" for x in chosen),
             "frequencies_hz":";".join(f"{x['freq_hz']:g}" for x in chosen if "freq_hz" in x),
             "exact_search_or_gap_pct":gap,"note":note})

        # Assay/frequency-unit budget: a stress scalar and a Re/Im frequency pair each cost one unit.
        exact_oracles={k:exact_best(V,g,actions,k) for k in [1,2,3]}
        for budget in [1,2,3]:
          gp=greedy(V,g,actions,budget)
          chosen, v=gp[-1]
          gap=None; note="unit-cost greedy; stress=1 unit and CM Re/Im pair=1 unit"
          if budget==1:
            exact=exact_oracles[1]
            gap=100*(v-exact[2])/(v0-exact[2]) if v0>exact[2] else 0
            note="one-assay-unit exhaustive optimum"
          elif budget==2:
            exact=exact_oracles[2]
            gap=100*(v-exact[2])/(v0-exact[2]) if v0>exact[2] else 0
            note="greedy K=2 compared with exhaustive assay-unit-pair optimum"
          add_record(records,target,dn,"frequency_assay_unit",budget,chosen,v0,v,full_u,gap,note)
          for a in chosen:
            if a.get("freq_hz") is not None: heat.append({"target":target,"domain":dn,"budget":budget,"condition":a["condition"],"frequency_hz":a["freq_hz"],"selected":1})
        # Exact constrained panels for requested assay-unit comparisons, up to 3 units.
        stresses=[a for a in actions if a["type"]=="stress"]
        cm_actions=[a for a in actions if a["type"]=="CM_frequency_pair"]
        patt=[("one_stress",[1,0]),("one_CM_frequency",[0,1]),("stress_plus_one_CM",[1,1]),
              ("two_CM_frequencies",[0,2]),("stress_plus_two_CM",[1,2]),("three_CM_frequencies",[0,3])]
        for label,(ns,nc2) in patt:
          items=[];basev=v0
          for ss in itertools.combinations(stresses,ns):
            for cc in itertools.combinations(cm_actions,nc2):
              chosen=list(ss)+list(cc); vv=panel_variance(V,g,[r for a in chosen for r in a["rows"]]); items.append((vv,chosen))
          if items:
            vv,chosen=min(items,key=lambda x:x[0])
            add_record(records,target,dn,"unit_pattern:"+label,ns+nc2,chosen,v0,vv,full_u,0,"exhaustive within the stated stress/frequency composition")
        # K heatmap from best 8-scalar greedy panel, split Re/Im as scalar coordinates.
        chosen8=greedy(V,g,scalars,8,rows_of=rows_from_scalar)[-1][0]
        for x in chosen8:
          if x["type"]=="CM_scalar": heat.append({"target":target,"domain":dn,"budget":8,"condition":x["condition"],"frequency_hz":x["freq_hz"],"selected":1})
        # L: target-wise and robust sequential acquisition for three whole assay units.
        target_seq=greedy(V,g,actions,3)
        oracle3=exact_oracles[3]
        oracle_var=oracle3[2]
        for step,(panel,vseq) in enumerate(target_seq,1):
          oracle_step=exact_oracles[step][2]
          seq_rows.append({"target":target,"domain":dn,"strategy":"target_greedy","step":step,"action_id":panel[-1]["action_id"],"panel_ids":";".join(a["action_id"] for a in panel),"support_width_pp":width(vseq),"variance":vseq,"oracle_variance_at_budget":oracle_step,"regret_pct_of_oracle_gain":100*(vseq-oracle_step)/(v0-oracle_step) if v0>oracle_step else 0})
        # Parameter A-opt sequence: rank-one unit actions, minimize trace posterior covariance.
        chosen=[]; avail=list(actions); Vcur=V.copy()
        for step in range(1,4):
          best=None
          for a in avail:
            R=np.asarray(a["rows"],float); A=np.eye(len(R))+R@Vcur@R.T
            Vn=Vcur - Vcur@R.T@np.linalg.solve(A,R@Vcur)
            score=float(np.trace(Vn))
            if best is None or score<best[0]: best=(score,a,Vn)
          score,a,Vcur=best; chosen.append(a);avail.remove(a)
          pv=panel_variance(V,g,[r for x in chosen for r in x["rows"]]); oracle_step=exact_oracles[step][2]
          seq_rows.append({"target":target,"domain":dn,"strategy":"parameter_A_opt","step":step,"action_id":a["action_id"],"panel_ids":";".join(x["action_id"] for x in chosen),"support_width_pp":width(pv),"variance":pv,"oracle_variance_at_budget":oracle_step,"regret_pct_of_oracle_gain":100*(pv-oracle_step)/(v0-oracle_step) if v0>oracle_step else 0})
        # Static non-adaptive panel: preselect the exact 3-unit oracle panel at the baseline geometry.
        stat=sorted(oracle3[1],key=lambda a:a["action_id"])
        for step in range(1,4):
          p=stat[:step]; vv=panel_variance(V,g,[r for a in p for r in a["rows"]])
          oracle_step=exact_oracles[step][2]
          seq_rows.append({"target":target,"domain":dn,"strategy":"fixed_panel_oracle_ordered_by_id","step":step,"action_id":p[-1]["action_id"],"panel_ids":";".join(x["action_id"] for x in p),"support_width_pp":width(vv),"variance":vv,"oracle_variance_at_budget":oracle_step,"regret_pct_of_oracle_gain":100*(vv-oracle_step)/(v0-oracle_step) if v0>oracle_step else 0})
        # Seeded random 3-unit panel baseline (expected design; no noisy values are drawn).
        rng=np.random.default_rng(20261002 + TARGETS.index(target) + 100*(DOMAIN_NAMES.index(dn)))
        rand_paths=[]
        for _ in range(500):
          pick=rng.choice(len(actions),size=3,replace=False)
          rand_paths.append([actions[int(i)] for i in pick])
        for step in range(1,4):
          vals=[panel_variance(V,g,[r for a in path[:step] for r in a["rows"]]) for path in rand_paths]
          seq_rows.append({"target":target,"domain":dn,"strategy":"random_500_mean","step":step,"action_id":"","panel_ids":"","support_width_pp":width(float(np.mean(vals))),"variance":float(np.mean(vals)),"variance_median":float(np.median(vals)),"oracle_variance_at_budget":exact_oracles[step][2],"regret_pct_of_oracle_gain":100*(float(np.mean(vals))-exact_oracles[step][2])/(v0-exact_oracles[step][2]) if v0>exact_oracles[step][2] else 0,"seed":20261002 + TARGETS.index(target) + 100*(DOMAIN_NAMES.index(dn))})
        # Reuse local sensitivity vectors to evaluate P dynamic complex-modulus target.
        P_rows=[]
        for tcond in TARGETS:
          p_actions,_=action_map(cand_layout,Jcand,None)
          p_actions=[a for a in p_actions if a["condition"]!=tcond]
          y0=all_model_outputs(dom["u"],dom,d)[("cm",tcond)]
          # Whiten each component by the target condition's published SEM; summarize average component-wise target variance.
          gt=[]
          for i in range(len(FREQS)):
            for comp,se in (("Re",d["se_re"][tcond][i]),("Im",d["se_im"][tcond][i])):
              # finite-difference standardized CM target component
              f=lambda u: (predict(p_of_u(u,dom["zlo"],dom["zhi"]),*CONDITIONS[tcond],FREQS)[0][i].real if comp=="Re" else predict(p_of_u(u,dom["zlo"],dom["zhi"]),*CONDITIONS[tcond],FREQS)[0][i].imag)/se
              h=STEP; gg=np.empty(14)
              for jj in range(14):
                up=dom["u"].copy();dn2=dom["u"].copy()
                if dom["u"][jj]<=h: up[jj]+=h; gg[jj]=(f(up)-f(dom["u"])) / h
                elif dom["u"][jj]>=1-h: dn2[jj]-=h; gg[jj]=(f(dom["u"])-f(dn2))/h
                else: up[jj]+=h;dn2[jj]-=h;gg[jj]=(f(up)-f(dn2))/(2*h)
              gt.append(gg)
              for jj,val in enumerate(gg): pgrad_rows.append({"domain":dn,"panel_design_for_stress_target":target,"dynamic_target":tcond,"component_id":f"{tcond}_{FREQS[i]:g}Hz_{comp}","sem":se,"parameter_index":jj,"standardized_sensitivity":val})
          GT=np.asarray(gt); target_cov=GT@V@GT.T; Pbase=float(np.trace(target_cov)/len(GT)); Pbase_max=float(np.max(np.diag(target_cov)))
          bestfull_mean=0; bestfull_max=0
          for cond in [c for c in CONDS if c!=tcond]:
            aa=[a for a in p_actions if a["condition"]==cond and a["type"]=="CM_frequency_pair"]
            Rfull=np.asarray([r for a in aa for r in a["rows"]],float)
            Vfull=V-V@Rfull.T@np.linalg.solve(np.eye(len(Rfull))+Rfull@V@Rfull.T,Rfull@V)
            Cfull=GT@Vfull@GT.T
            bestfull_mean=max(bestfull_mean,Pbase-float(np.trace(Cfull)/len(GT)))
            bestfull_max=max(bestfull_max,Pbase_max-float(np.max(np.diag(Cfull))))
          for k,panel in enumerate(greedy(V,g,p_actions,3)):
            chosenp,vstress=panel
            R=np.asarray([r for a in chosenp for r in a["rows"]],float)
            Vp=V - V@R.T@np.linalg.solve(np.eye(len(R))+R@V@R.T,R@V)
            Cp=GT@Vp@GT.T; Pv=float(np.trace(Cp)/len(GT)); Pmax=float(np.max(np.diag(Cp)))
            P_rows.append({"domain":dn,"panel_design_for_stress_target":target,"future_dynamic_target":tcond,"panel_step":k+1,"panel_ids":";".join(a["action_id"] for a in chosenp),"metric":"mean_component_variance","baseline_standardized_variance":Pbase,"after_panel_standardized_variance":Pv,"relative_uncertainty_reduction_pct":100*(1-Pv/Pbase) if Pbase else np.nan,"stress_target_variance_reduction":V0-vstress,"fraction_best_single_full_CM_spectrum_utility_recovered":(Pbase-Pv)/bestfull_mean if bestfull_mean>0 else np.nan})
            P_rows.append({"domain":dn,"panel_design_for_stress_target":target,"future_dynamic_target":tcond,"panel_step":k+1,"panel_ids":";".join(a["action_id"] for a in chosenp),"metric":"maximum_component_variance","baseline_standardized_variance":Pbase_max,"after_panel_standardized_variance":Pmax,"relative_uncertainty_reduction_pct":100*(1-Pmax/Pbase_max) if Pbase_max else np.nan,"stress_target_variance_reduction":V0-vstress,"fraction_best_single_full_CM_spectrum_utility_recovered":(Pbase_max-Pmax)/bestfull_max if bestfull_max>0 else np.nan})
        P_all.extend(P_rows)

    # Robust/minimax step selection is shared across the two frozen parameter domains for each target.
    for target in TARGETS:
      states={dn:local_cache[(dn,target)] for dn in DOMAIN_NAMES}
      common_ids=set.intersection(*[{a["action_id"] for a in x["actions"]} for x in states.values()])
      byid={dn:{a["action_id"]:a for a in x["actions"]} for dn,x in states.items()}
      chosen_ids=[]
      for step in range(1,4):
        best=None
        for aid in sorted(common_ids-set(chosen_ids)):
          worst=1.0
          for dn,obj in states.items():
            rows=[r for x in chosen_ids+[aid] for r in byid[dn][x]["rows"]]
            v=panel_variance(obj["V"],obj["g"],rows)
            gain=(obj["v0"]-v)/obj["v0"] if obj["v0"] else 0.0
            worst=min(worst,gain)
          if best is None or worst>best[0]: best=(worst,aid)
        chosen_ids.append(best[1])
        for dn,obj in states.items():
          rows=[r for aid in chosen_ids for r in byid[dn][aid]["rows"]]
          vv=panel_variance(obj["V"],obj["g"],rows)
          seq_rows.append({"target":target,"domain":dn,"strategy":"minimax_across_domains","step":step,"action_id":best[1],"panel_ids":";".join(chosen_ids),"support_width_pp":width(vv),"variance":vv,"worst_domain_relative_reduction_through_step":best[0]})

    # Persist K/L/M/P tables and sufficient statistics for independent recomputation.
    pd.DataFrame(records).to_csv(OUT/"K_SPARSE_PANEL_RESULTS.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(sens_rows).to_csv(OUT/"K_LOCAL_SENSITIVITIES.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(grad_rows).to_csv(OUT/"K_TARGET_GRADIENTS.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(base_rows).to_csv(OUT/"K_BASELINE_SENSITIVITIES.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(full_rows).to_csv(OUT/"K_FULL_ASSAY_REFERENCES.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(seq_rows).to_csv(OUT/"L_SEQUENTIAL_DESIGN_RESULTS.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(P_all).to_csv(OUT/"P_DYNAMIC_TARGET_RESULTS.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(pgrad_rows).to_csv(OUT/"P_DYNAMIC_TARGET_GRADIENTS.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(heat).to_csv(OUT/"K_FREQUENCY_SELECTION_MAP.csv",index=False,encoding="utf-8-sig")
    # M alignment-vs-A: single candidate stress/spectrum utilities, matched to the frozen A profile audit.
    a_map=json.loads((INP/"A_measurement_target_map.json").read_text(encoding="utf-8")); mrows=[]
    for (dn,target),obj in local_cache.items():
      V=obj["V"];g=obj["g"];v0=obj["v0"];actions=obj["actions"]
      m_actions=[]
      for cond in [c for c in CONDS if c!=target]:
        m_actions.append(next(a for a in actions if a["action_id"]==f"stress_{cond}"))
        m_actions.append({"action_id":f"CM_spectrum_{cond}","type":"CM_spectrum","condition":cond,
          "rows":[r for a in actions if a["condition"]==cond and a["type"]=="CM_frequency_pair" for r in a["rows"]]})
      for act in m_actions:
        selected_rows=act["rows"]
        newv=panel_variance(V,g,selected_rows); util=(v0-newv)/v0 if v0 else 0
        R=np.asarray(selected_rows,float); align=(float(g@V@R.T@R@V@g)/(max(v0,1e-15)*max(float(np.trace(R@V@R.T)),1e-15)))
        # A's exact profile support utility uses the restarted full audit map.
        akey=act["action_id"]
        a_rec=a_map["domains"][dn]["measurements"][akey]["support"].get(target,{})
        aval=a_rec.get("relative_width_reduction_pct",np.nan)
        mrows.append({"domain":dn,"target":target,"measurement_id":akey,"measurement_kind":act["type"],"condition":act["condition"],"whitened_alignment_score":align,"local_Fisher_relative_variance_reduction_pct":100*util,"A_profile_relative_width_reduction_pct":aval})
    mdf=pd.DataFrame(mrows); mdf.to_csv(OUT/"M_ALIGNMENT_A_UTILITY.csv",index=False,encoding="utf-8-sig")
    rho_rows=[]
    perm_rows=[]; rng_m=np.random.default_rng(20261002)
    for (dn,kind),sub in mdf.groupby(["domain","measurement_kind"]):
      r,p=spearmanr(sub.whitened_alignment_score,sub.A_profile_relative_width_reduction_pct,nan_policy="omit")
      rho_rows.append({"domain":dn,"measurement_kind":kind,"n_target_action_rows":len(sub),"spearman_rho":float(r),"descriptive_p_value_not_inferential":float(p)})
      x=sub.whitened_alignment_score.to_numpy(float); y=sub.A_profile_relative_width_reduction_pct.to_numpy(float)
      rp=[]
      for _ in range(5000): rp.append(float(spearmanr(rng_m.permutation(x),y).statistic))
      perm_p=float((1+np.sum(np.abs(rp)>=abs(r)))/(1+len(rp)))
      perm_rows.append({"domain":dn,"measurement_kind":kind,"n":len(sub),"observed_spearman_rho":float(r),"permutation_count":len(rp),"two_sided_shuffle_p":perm_p,"seed":20261002,"interpretation":"descriptive sanity check; action rows are model-derived and not independent biological replicates"})
    pd.DataFrame(rho_rows).to_csv(OUT/"M_ALIGNMENT_ASSOCIATION.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(perm_rows).to_csv(OUT/"M_ALIGNMENT_PERMUTATION_CHECK.csv",index=False,encoding="utf-8-sig")
    # M parameter groups are source-model process labels, not independent biological measurements.
    groups={"ATP_gating_detachment":[4,3,12],"Pi_gating_transitions":[1,2,13],"strain_velocity_length":[5,6,7,10,11],"common_cycle_and_force_scale":[0,8,9]}
    gr=[]
    for (dn,target),obj in local_cache.items():
      # Mean squared target gradient and high-value measurements by model-defined groups.
      for name,ix in groups.items():
        gr.append({"domain":dn,"target":target,"model_process_group":name,"parameter_names":";".join(PARAM_NAMES[i] for i in ix),"target_gradient_norm":float(np.linalg.norm(obj["g"][ix]))})
    pd.DataFrame(gr).to_csv(OUT/"M_MODEL_PROCESS_GROUPS.csv",index=False,encoding="utf-8-sig")
    # P metric sensitivity uses mean standardized variance; this is deliberately not a clinical loss function.
    # K plots: best scalar-counted greedy and unit-counted target-greedy curves.
    kdf=pd.DataFrame(records)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for (target,dn),sub in kdf[kdf.accounting_view.eq("scalar_count")].groupby(["target","domain"]):
      axs[0].plot(sub.budget,sub.width_reduction_pct,marker="o",label=f"{target} / {dn[-3:]}")
    axs[0].set(xlabel="Additional scalar observations",ylabel="Unbounded local Wald width reduction (%)",title="Scalar-counted greedy design")
    for (target,dn),sub in kdf[kdf.accounting_view.eq("frequency_assay_unit")].groupby(["target","domain"]):
      axs[1].plot(sub.budget,sub.width_reduction_pct,marker="o",label=f"{target} / {dn[-3:]}")
    axs[1].set(xlabel="Additional assay/frequency units",ylabel="Unbounded local Wald width reduction (%)",title="One stress scalar = one CM frequency pair = one unit")
    for ax in axs: ax.legend(fontsize=6,ncol=2)
    fig.tight_layout();fig.savefig(OUT/"K_BUDGET_PERFORMANCE.png",dpi=180);plt.close(fig)
    # Four-panel heatmap: target/frequency selection by candidate CM condition.
    heatdf=pd.DataFrame(heat); heat8=heatdf[(heatdf.budget==8)].groupby(["target","condition","frequency_hz"],as_index=False).selected.sum()
    fig,axs=plt.subplots(2,2,figsize=(13,7),sharex=True,sharey=True,constrained_layout=True);im=None
    for ax,cond in zip(axs.ravel(),CONDS):
      z=heat8[heat8.condition.eq(cond)].pivot_table(index="target",columns="frequency_hz",values="selected",aggfunc="sum",fill_value=0)
      z=z.reindex(index=TARGETS,columns=FREQS,fill_value=0)
      im=ax.imshow(z.to_numpy(),aspect="auto",interpolation="nearest",cmap="Blues",vmin=0,vmax=2)
      ax.set_title(f"Added CM condition: {cond}");ax.set_yticks(range(len(TARGETS)),TARGETS);ax.set_xticks(range(len(FREQS)),[f"{f:g}" for f in FREQS],rotation=50,ha="right");ax.set_xlabel("Frequency (Hz)")
    fig.suptitle("Selected CM scalars in best 8-scalar panels (counts across the two parameter domains)")
    fig.colorbar(im,ax=axs.ravel().tolist(),label="selected count (0–2)",shrink=.88);fig.savefig(OUT/"K_TARGET_FREQUENCY_SELECTION.png",dpi=180);plt.close(fig)
    # Compact run metadata.
    meta={"status":"COMPLETED","finite_difference_step_u":STEP,"baseline_and_candidates_whitened_by_published_SEM":True,
      "n_frequency_points":len(FREQS),"freqs_hz":FREQS.tolist(),"target_definitions":"A frozen stress response ratio vs baseline",
      "same_target_condition_excluded":True,"baseline_includes_baseline_CM_and_stress":True,
      "K_metric":"expected local Gauss-Newton information with pseudoinverse; width is 95% local Wald equivalent, not exact profile likelihood or calibrated interval",
      "scalar_exact_search_budgets":[1,2],"unit_exact_search_budgets":[1,2],"larger_budgets":"deterministic greedy; exhaustive validation at K<=2",
      "no_observation_noise_draws":"expected design only; L is design-adaptive, not observation-adaptive",
      "rat_source_sha256":sha(INP/"rat_data.mat"),"a_map_sha256":sha(INP/"A_measurement_target_map.json"),
      "model":"frozen Model16D; md=4, sd=[0,0,0,1,1]"}
    (OUT/"KLP_RUN_METADATA.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps({"records":len(records),"sensitivities":len(sens_rows),"L_sequences":len(seq_rows),"M_pairs":len(mrows),"P_rows":len(P_all),"out":str(OUT)},indent=2))

if __name__=="__main__": main()
