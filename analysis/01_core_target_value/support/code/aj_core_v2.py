"""Restart-safe endpoint search wrapper for the audited A rank check.

Preserves the MLE and every supplied warm endpoint incumbent, then fills the
remaining deterministic-start budget from the serialized seed bank.  Feasible
incumbents are evaluated before optimization and remain eligible if a solver
call fails or returns an infeasible point.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from p2.constants import LR95_DF1
from aj_core import (clip_p, p_to_u, u_to_p, to_z, target_q)

def support_endpoint_v2(like,domain,mle,target,side,seeds,*,warm=None,max_starts=6,maxiter=1300):
    p_mle=np.asarray(mle["params"],float); obj_mle=float(like.objective(p_mle))
    if not np.isfinite(obj_mle) or abs(obj_mle-float(mle["objective"]))>2e-6:
        raise ValueError("MLE objective failed independent scorer consistency check")
    threshold=obj_mle+LR95_DF1
    # Explicit ordering: MLE, every provided incumbent, then deterministic
    # serialized alternatives.  Warm starts are never truncated; the cap
    # applies only to added deterministic seeds.
    starts=[{"name":"augmented_mle" if like.candidate else "baseline_mle","params":p_mle}]
    for i,p in enumerate(warm or []):
        p=clip_p(p,domain)
        if not any(np.max(np.abs(p-x["params"]))<1e-8 for x in starts):
            starts.append({"name":f"warm_endpoint_{i:02d}","params":p})
    for s in seeds:
        p=clip_p(s["params"],domain)
        if not any(np.max(np.abs(p-x["params"]))<1e-8 for x in starts):
            if len(starts)>=max_starts: break
            starts.append({"name":str(s["name"]),"params":p})

    sign=1.0 if side=="low" else -1.0
    attempts=[]; candidates=[]
    def feasible_params(p):
        z=to_z(p)
        return bool(np.all(p>=domain["lo"]-1e-7) and np.all(p<=domain["hi"]+1e-7) and
                    np.all(z>=domain["zlo"]-1e-7) and np.all(z<=domain["zhi"]+1e-7))
    # Preserve every feasible pre-run incumbent as an actual endpoint witness.
    for s in starts:
        p=s["params"]; obj=float(like.objective(p))
        if np.isfinite(obj) and obj<=threshold+2e-4 and feasible_params(p):
            q=float(target_q(p,target,like.condition_map))
            candidates.append((q,obj,p,False,"stored feasible incumbent",0,s["name"]))
    for k,s in enumerate(starts):
        p0=s["params"]; u0=p_to_u(p0,domain)
        def pof(u): return u_to_p(np.clip(u,0,1),domain)
        def objective(u): return sign*target_q(pof(u),target,like.condition_map)
        def lr_constraint(u): return threshold-like.objective(pof(u))
        try:
            rr=minimize(objective,u0,method="SLSQP",bounds=[(0,1)]*14,
                constraints=[{"type":"ineq","fun":lr_constraint}],
                options={"maxiter":maxiter,"ftol":1e-9,"disp":False})
            p=pof(rr.x); obj=float(like.objective(p)); q=float(target_q(p,target,like.condition_map))
            ok=bool(np.isfinite(obj) and obj<=threshold+2e-4 and feasible_params(p))
            attempts.append({"start_index":k,"start_name":s["name"],"start_params":p0.tolist(),
                "success":bool(rr.success),"message":str(rr.message),"iterations":int(getattr(rr,"nit",0)),
                "objective":obj,"q":q,"feasible":ok,"constraint_slack":float(threshold-obj)})
            if ok: candidates.append((q,obj,p,bool(rr.success),str(rr.message),int(getattr(rr,"nit",0)),s["name"]))
        except Exception as e:
            attempts.append({"start_index":k,"start_name":s["name"],"start_params":p0.tolist(),
                "success":False,"message":f"exception:{type(e).__name__}:{e}","feasible":False})
    if not candidates:
        raise RuntimeError("No feasible endpoint incumbent or optimizer result; support may not shrink silently")
    best=min(candidates,key=lambda x:x[0]) if side=="low" else max(candidates,key=lambda x:x[0])
    q,obj,p,success,msg,nit,src=best; slack=threshold-obj; z=to_z(p)
    active=np.flatnonzero((z-domain["zlo"]<1e-5)|(domain["zhi"]-z<1e-5)).tolist()
    return {"side":side,"q":float(q),"objective":float(obj),"delta_lr":float(obj-obj_mle),
        "threshold":float(threshold),"constraint_slack":float(slack),"params":p.tolist(),
        "feasible":bool(obj<=threshold+2e-4 and feasible_params(p)),"optimizer_success":bool(success),
        "optimizer_message":msg,"optimizer_iterations":nit,"selected_start":src,
        "domain_bound_active_indices":active,"all_attempts":attempts,
        "n_feasible_pre_run_incumbents":sum(1 for x in starts if like.objective(x["params"])<=threshold+2e-4 and feasible_params(x["params"])),
        "reached_lr_boundary":bool(abs(slack)<1e-3)}
