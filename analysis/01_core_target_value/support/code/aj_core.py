"""Public-rat A/E/F/J likelihood, domain, and optimizer routines."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))

import csv, hashlib, json
import numpy as np
from scipy.io import loadmat
from scipy.optimize import lsq_linear, minimize

from p2.constants import CONDITIONS, LOWER, UPPER, NONLINEAR_IDX, LR95_DF1
from p2.model16d import predict, q_delta_stress, nonlinear_basis

HERE=Path(__file__).resolve().parent
BRANCH=HERE.parents[1]
PUBLIC=REPO/"data/third_party/rat/rat_data.mat"
CONFIG=HERE.parent/"configs"/"A_E_F_J_freeze_2026-10-01.json"
SEEDFILE=HERE.parent/"configs"/"A_E_F_J_start_vectors.json"
OUT=HERE.parent/"results"

CONDITION_LABELS={"baseline":"Baseline","ATP0.1":"Super-Low ATP","ATP1":"Low ATP","Pi0":"Low Pi","Pi5":"High Pi"}
CONDS={k:CONDITIONS[k] for k in CONDITION_LABELS}

@dataclass
class PublicRat:
    freqs: np.ndarray
    stress_mean: dict[str,float]
    stress_sem: dict[str,float]
    cm_mean: dict[str,np.ndarray]
    cm_sem_re: dict[str,np.ndarray]
    cm_sem_im: dict[str,np.ndarray]
    raw_sha256: str

def load_public_rat()->PublicRat:
    mat=loadmat(PUBLIC,squeeze_me=True,struct_as_record=False)
    data=np.asarray(mat["data"],dtype=object)
    labels=[str(x) for x in data[0,1:]]
    col={name:j+1 for j,name in enumerate(labels)}
    rows={str(data[i,0]):i for i in range(1,data.shape[0])}
    freqs=np.asarray(mat["freqs"],dtype=float).ravel()
    out=PublicRat(freqs,{}, {}, {}, {}, {},hashlib.sha256(PUBLIC.read_bytes()).hexdigest())
    for key,label in CONDITION_LABELS.items():
        j=col[label]
        out.cm_mean[key]=np.asarray(data[rows["mean CM"],j],dtype=complex).ravel()
        out.cm_sem_re[key]=np.asarray(data[rows["EM SE"],j],dtype=float).ravel()
        out.cm_sem_im[key]=np.asarray(data[rows["VM SE"],j],dtype=float).ravel()
        out.stress_mean[key]=float(data[rows["mean F0"],j])
        out.stress_sem[key]=float(data[rows["F0 SE"],j])
    if out.freqs.size!=17 or any(x.size!=17 for x in out.cm_mean.values()):
        raise ValueError("Unexpected public MAT schema; expected 17 frequencies/CM values")
    if not all(np.all(np.isfinite(x)) for x in out.cm_mean.values()):
        raise ValueError("Nonfinite complex modulus input")
    return out

def load_seeds()->list[dict]:
    o=json.loads(SEEDFILE.read_text(encoding="utf-8"))
    if o.get("runtime_rng")!="none": raise ValueError("Frozen run allows no runtime RNG")
    return o["records"]

def to_z(p):
    p=np.asarray(p,dtype=float); z=np.empty_like(p)
    ix=np.arange(p.size)!=9; z[ix]=np.log(p[ix]); z[9]=p[9]/45.0
    return z

def from_z(z):
    z=np.asarray(z,dtype=float); p=np.exp(z)
    p=p.copy(); p[9]=z[9]*45.0
    return p

def domain_for(margin:float, seeds:list[dict]):
    by={x["name"]:np.asarray(x["params"],float) for x in seeds}
    zr,za=to_z(by["P1_Reference"]),to_z(by["P1_Alternative"])
    zglo,zghi=to_z(LOWER),to_z(UPPER)
    zlo=np.maximum(zglo,np.minimum(zr,za)-margin)
    zhi=np.minimum(zghi,np.maximum(zr,za)+margin)
    if np.any(zhi<=zlo): raise ValueError("Empty pair domain")
    return {"margin":float(margin),"zlo":zlo,"zhi":zhi,"lo":from_z(zlo),"hi":from_z(zhi)}

def p_to_u(p,dom):
    z=to_z(p); span=dom["zhi"]-dom["zlo"]
    return np.clip((z-dom["zlo"])/span,0.0,1.0)

def u_to_p(u,dom):
    return from_z(dom["zlo"]+np.asarray(u,float)*(dom["zhi"]-dom["zlo"]))

def clip_p(p,dom):
    return from_z(np.minimum(np.maximum(to_z(p),dom["zlo"]),dom["zhi"]))

class PublicLikelihood:
    """Diagonal group-SEM likelihood for baseline and one added assay block."""
    def __init__(self,data:PublicRat,candidate:dict|None=None,*,candidate_sem_factor:float=1.0,condition_map:dict|None=None):
        self.data=data; self.candidate=candidate; self.candidate_sem_factor=float(candidate_sem_factor)
        self.condition_map=CONDS if condition_map is None else condition_map
        if candidate_sem_factor<=0: raise ValueError("SEM multiplier must be positive")
        self.blocks=[{"kind":"cm","condition":"baseline","mean":data.cm_mean["baseline"],"se_re":data.cm_sem_re["baseline"],"se_im":data.cm_sem_im["baseline"],"candidate":False},
                     {"kind":"stress","condition":"baseline","mean":data.stress_mean["baseline"],"se":data.stress_sem["baseline"],"candidate":False}]
        if candidate:
            typ=candidate["type"]; cond=candidate["condition"]
            if typ=="steady_stress":
                mean=float(candidate.get("mean",data.stress_mean[cond]))
                sem=float(candidate.get("sem",data.stress_sem[cond]))
                self.blocks.append({"kind":"stress","condition":cond,"mean":mean,"se":sem,"candidate":True})
            elif typ=="complex_modulus_spectrum":
                mean=np.asarray(candidate.get("mean",data.cm_mean[cond]),dtype=complex)
                self.blocks.append({"kind":"cm","condition":cond,"mean":mean,"se_re":data.cm_sem_re[cond],"se_im":data.cm_sem_im[cond],"candidate":True})
            else: raise ValueError(f"Unknown candidate type: {typ}")

    def residual_vector(self,p):
        vals=[]
        for b in self.blocks:
            factor=self.candidate_sem_factor if b["candidate"] else 1.0
            if b["kind"]=="cm":
                y,_=predict(p,*self.condition_map[b["condition"]],self.data.freqs)
                vals.append((y.real-np.asarray(b["mean"]).real)/(np.asarray(b["se_re"])*factor))
                vals.append((y.imag-np.asarray(b["mean"]).imag)/(np.asarray(b["se_im"])*factor))
            else:
                _,stress=predict(p,*self.condition_map[b["condition"]],self.data.freqs[:1])
                vals.append(np.asarray([(stress-float(b["mean"]))/(float(b["se"])*factor)]))
        return np.concatenate(vals)

    def objective(self,p):
        try:
            r=self.residual_vector(p)
            return float(r@r) if np.all(np.isfinite(r)) else 1e100
        except (FloatingPointError,ValueError,OverflowError,ZeroDivisionError):
            return 1e100

    def _linear_system(self,nl,domain):
        aa=[]; yy=[]
        for b in self.blocks:
            f=self.candidate_sem_factor if b["candidate"] else 1.0
            at,pi=self.condition_map[b["condition"]]
            if b["kind"]=="cm":
                yk,yks,_=nonlinear_basis(nl,at,pi,self.data.freqs,NONLINEAR_IDX)
                m=np.asarray(b["mean"],complex); sr=np.asarray(b["se_re"],float)*f; si=np.asarray(b["se_im"],float)*f
                aa.append(np.column_stack([yk.real/sr,yks.real/sr])); yy.append(m.real/sr)
                aa.append(np.column_stack([yk.imag/si,yks.imag/si])); yy.append(m.imag/si)
            else:
                _,_,ak=nonlinear_basis(nl,at,pi,self.data.freqs[:1],NONLINEAR_IDX)
                se=float(b["se"])*f
                aa.append(np.asarray([[ak/se,0.3/se]],float)); yy.append(np.asarray([float(b["mean"])/se],float))
        return np.vstack(aa),np.concatenate(yy)

    def solve_scales(self,nl,domain):
        A,y=self._linear_system(nl,domain)
        fit=lsq_linear(A,y,bounds=([domain["lo"][8],domain["lo"][9]],
                                   [domain["hi"][8],domain["hi"][9]]),
                       method="trf",tol=1e-11,lsmr_tol="auto",max_iter=300)
        p=np.empty(14,float); p[NONLINEAR_IDX]=nl; p[8:10]=fit.x
        return p,self.objective(p)

def fit_mle(like:PublicLikelihood,domain,seeds:list[dict],*,extra_seeds:list[dict]|None=None):
    starts=[]
    for s in (seeds+(extra_seeds or [])):
        p=clip_p(s["params"],domain)
        if not any(np.max(np.abs(p-x["params"]))<1e-9 for x in starts):
            starts.append({"name":s["name"],"source_params":s["params"],"params":p})
    if not starts: raise ValueError("No deterministic fit starts")
    log=[]; candidates=[]
    loglo=np.log(domain["lo"][NONLINEAR_IDX]); loghi=np.log(domain["hi"][NONLINEAR_IDX])
    def u_to_nl(u): return np.exp(loglo+np.asarray(u)*(loghi-loglo))
    def nl_to_u(p): return np.clip((np.log(np.asarray(p)[NONLINEAR_IDX])-loglo)/(loghi-loglo),0,1)
    def f(u):
        try: return like.solve_scales(u_to_nl(np.clip(u,0,1)),domain)[1]
        except Exception: return 1e100
    for s in starts:
        p0=s["params"]
        p_start,obj_start=like.solve_scales(p0[NONLINEAR_IDX],domain)
        candidates.append((obj_start,p_start,f"start:{s['name']}",False,"scored serialized start",0))
        rr=minimize(f,nl_to_u(p0),method="L-BFGS-B",bounds=[(0.0,1.0)]*12,
                    options={"maxiter":1500,"ftol":1e-11,"gtol":1e-8,"maxls":50})
        pfit,obj_ind=like.solve_scales(u_to_nl(np.clip(rr.x,0,1)),domain)
        obj_re=like.objective(pfit)
        log.append({"start":s["name"],"initial_objective":float(obj_start),
                    "optimizer_objective":float(rr.fun),"independent_objective":float(obj_re),
                    "objective_difference":float(abs(obj_re-rr.fun)),"success":bool(rr.success),
                    "message":str(rr.message),"nit":int(getattr(rr,"nit",0)),
                    "selected_candidate_feasible":bool(np.all(pfit>=domain["lo"]-1e-7) and np.all(pfit<=domain["hi"]+1e-7))})
        candidates.append((obj_re,pfit,f"local:{s['name']}",bool(rr.success),str(rr.message),int(getattr(rr,"nit",0))))
    win=min(candidates,key=lambda x:x[0]); best_start=min(x[0] for x in candidates if x[2].startswith("start:"))
    if win[0]>best_start+1e-8: raise AssertionError("Fit result is worse than serialized feasible incumbent")
    p=win[1]; obj=like.objective(p)
    if abs(obj-win[0])>1e-7: raise AssertionError("Final fit objective recomputation mismatch")
    return {"params":p.tolist(),"objective":float(obj),"winner":win[2],"success":bool(win[3]),
            "message":win[4],"n_residuals":int(len(like.residual_vector(p))),"starts":log,
            "all_start_outcomes_recorded":True,"nonworsening_vs_best_serialized_start":bool(obj<=best_start+1e-8)}

def target_q(p,target,condition_map=None):
    cc=CONDS if condition_map is None else condition_map
    return q_delta_stress(np.asarray(p,float),cc[target],cc["baseline"])

def support_endpoint(like:PublicLikelihood,domain,mle:dict,target:str,side:str,seeds:list[dict],*,warm:list[np.ndarray]|None=None):
    p_mle=np.asarray(mle["params"],float); obj_mle=float(mle["objective"])
    threshold=obj_mle+LR95_DF1
    sign=1.0 if side=="low" else -1.0
    allp=[p_mle]+[clip_p(s["params"],domain) for s in seeds]
    allp += [clip_p(x,domain) for x in (warm or [])]
    uniq=[]
    for p in allp:
        if not any(np.max(np.abs(p-q))<1e-8 for q in uniq): uniq.append(p)
    uniq=uniq[:6]
    records=[]; candidates=[]
    for k,p0 in enumerate(uniq):
        u0=p_to_u(p0,domain)
        def pof(u): return u_to_p(np.clip(u,0,1),domain)
        def f(u): return sign*target_q(pof(u),target,like.condition_map)
        def feasible(u): return threshold-like.objective(pof(u))
        try:
            rr=minimize(f,u0,method="SLSQP",bounds=[(0,1)]*14,
                        constraints=[{"type":"ineq","fun":feasible}],
                        options={"maxiter":1300,"ftol":1e-9,"disp":False})
            p=pof(rr.x); obj=like.objective(p); q=target_q(p,target,like.condition_map)
            ok=bool(np.isfinite(obj) and obj<=threshold+1e-4 and np.all(p>=domain["lo"]-1e-7) and np.all(p<=domain["hi"]+1e-7))
            records.append({"start_index":k,"start_name":("mle" if k==0 else f"seed_{k}"),"success":bool(rr.success),"message":str(rr.message),"nit":int(getattr(rr,"nit",0)),"objective":float(obj),"q":float(q),"feasible":ok,"constraint_slack":float(threshold-obj)})
            if ok:candidates.append((q,obj,p,bool(rr.success),str(rr.message),int(getattr(rr,"nit",0)),f"start_{k}"))
        except Exception as e:
            records.append({"start_index":k,"success":False,"message":f"exception:{type(e).__name__}:{e}","feasible":False})
    if not candidates:
        # Always preserve a feasible witness, but mark endpoint search failure.
        q=target_q(p_mle,target,like.condition_map)
        candidates=[(q,obj_mle,p_mle,False,"fallback to feasible MLE; endpoint search failed",0,"mle_fallback")]
    best=(min(candidates,key=lambda x:x[0]) if side=="low" else max(candidates,key=lambda x:x[0]))
    q,obj,p,success,msg,nit,src=best
    slack=threshold-obj
    z=to_z(p); active=np.flatnonzero((z-domain["zlo"]<1e-5)|(domain["zhi"]-z<1e-5)).tolist()
    return {"side":side,"q":float(q),"objective":float(obj),"delta_lr":float(obj-obj_mle),
            "threshold":float(threshold),"constraint_slack":float(slack),"params":p.tolist(),
            "feasible":bool(obj<=threshold+1e-4 and np.all(p>=domain["lo"]-1e-7) and np.all(p<=domain["hi"]+1e-7)),
            "optimizer_success":bool(success),"optimizer_message":msg,"optimizer_iterations":nit,
            "selected_start":src,"domain_bound_active_indices":active,"all_attempts":records,
            "reached_lr_boundary":bool(abs(slack)<1e-3)}

def read_freeze(): return json.loads(CONFIG.read_text(encoding="utf-8"))

def candidate_definitions():
    out=[]
    for cond in ["ATP0.1","ATP1","Pi0","Pi5"]:
        out.append({"id":f"stress_{cond}","type":"steady_stress","condition":cond,"n_scalar_observations":1})
    for cond in ["ATP0.1","ATP1","Pi0","Pi5"]:
        out.append({"id":f"CM_spectrum_{cond}","type":"complex_modulus_spectrum","condition":cond,"n_scalar_observations":34})
    return out

def save_json(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")

def sha256(path:Path): return hashlib.sha256(path.read_bytes()).hexdigest()
