#!/usr/bin/env python3
"""
P2 human atrial Stage-1 local-information pilot.

Reads frozen PUBLIC human model/data resources from the P1 project read-only.
Writes only to this P2 pilot's results directory.

This is a local information screen, not calibrated human inference.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

CHI2 = 3.841458820694124
FREE_IDX = np.array([0,1,2,3,4,6,7,8,10,11,12], dtype=int)
FREE_NAMES = ["k1","k_1","k2","k_2","k3","phi_v","phi_l","K","phi_s1","phi_s3","kd_ATP"]
LB = np.array([0.5,0.5,0.5,0.1,0.5,0.005,1,2000,1,1,0.02], float)
UB = np.array([200,200,200,500,200,0.5,5,10000,1000,1000,5], float)
CONDS = ["Baseline","ATP0.1","ATP1","Pi0","Pi10"]

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

def unwrap(x):
    y=x
    for _ in range(10):
        if isinstance(y,np.ndarray) and y.dtype==object and y.size==1:
            y=y.item()
        else:
            break
    return y

def vec(x, dtype=None):
    a=np.asarray(unwrap(x))
    if a.dtype==object:
        vals=[]
        for v in a.ravel():
            z=np.asarray(unwrap(v)).squeeze()
            if z.size==1: vals.append(z.item())
            else: vals.extend(np.asarray(z).ravel().tolist())
        a=np.asarray(vals)
    a=np.asarray(a).squeeze().ravel()
    return a.astype(dtype) if dtype is not None else a

def scal(x):
    a=np.asarray(unwrap(x)).squeeze()
    if a.size!=1: raise ValueError(f"Expected scalar, got {a.shape}")
    return float(np.real(a).item())

def cname(met):
    atp,pi=map(float,met[:2])
    if abs(atp-5)<1e-5 and abs(pi-1)<1e-5: return "Baseline"
    if abs(atp-.1)<1e-5 and abs(pi-1)<1e-5: return "ATP0.1"
    if abs(atp-1)<1e-5 and abs(pi-1)<1e-5: return "ATP1"
    if abs(atp-5)<1e-5 and pi<1e-3: return "Pi0"
    if abs(atp-5)<1e-5 and abs(pi-10)<1e-5: return "Pi10"
    return f"ATP{atp:g}_Pi{pi:g}"

def find_sources(root):
    d = REPO / "data/third_party/human_model"
    if (d / "ND_xb_fit.mat").is_file():
        return d / "ND_xb_fit.mat", d / "ave_human_fitting_data.mat", d / "XBmodel_2024_linear_perms.m"
    canonical=[
        root/"02_Data/01_Public/Human_2026_Musgrave/Public_Source/Model_Source_Snapshot/AtrialModel_2025_Human/MATLAB",
        root/"02_Data/Public/Human_2025_2026_Musgrave/Source_Code_Snapshot/AtrialModel_2025_Human/MATLAB",
    ]
    for d in canonical:
        a,b,c=d/"ND_xb_fit.mat",d/"ave_human_fitting_data.mat",d/"XBmodel_2024_linear_perms.m"
        if a.exists() and b.exists() and c.exists(): return a,b,c
    fits=list(root.rglob("ND_xb_fit.mat"))
    datas=list(root.rglob("ave_human_fitting_data.mat"))
    mods=[p for p in root.rglob("XBmodel_2024_linear_perms.m") if "human" in str(p).lower()]
    if not fits or not datas or not mods:
        raise FileNotFoundError("Could not find required public human files under P1 root.")
    return fits[0],datas[0],mods[0]

def load_ref(path):
    m=loadmat(path,squeeze_me=False,struct_as_record=False)
    if "x_p" not in m: raise KeyError("ND_xb_fit.mat does not contain x_p.")
    x=vec(m["x_p"],float)
    if x.size!=14: raise ValueError(f"x_p expected 14 slots; got {x.size}")
    return x

def load_data(path):
    m=loadmat(path,squeeze_me=False,struct_as_record=False)
    d=np.asarray(m["ND_xb_data"],dtype=object)
    freqs=vec(m["freqs"],float)
    out={}
    for j in range(1,6):
        met=vec(d[1,j],float)
        out[cname(met)]={
            "met":met[:2],
            "Y":vec(d[2,j]).astype(complex),
            "se_re":vec(d[3,j],float),
            "se_im":vec(d[4,j],float),
            "F0":scal(d[5,j]),
            "se_F0":scal(d[6,j]),
        }
    missing=[x for x in CONDS if x not in out]
    if missing: raise ValueError(f"Missing expected conditions: {missing}")
    return freqs,out

def model(p,met,freqs,T=295.0):
    p=np.asarray(p,float)
    k1,k_1,k2,k_2,k3=p[:5]
    phi_x,phi_v,phi_l,K,Ks=p[5:10]
    kd_ATP,kd_Pi=p[12],p[13]
    ATP,Pi=map(float,met[:2])
    sd=np.array([p[10],0,0,0,p[11]],float)
    Lmax,xC0,L0=2.3,.01,2.2
    Z0=1-(Lmax-L0)*phi_l/Lmax

    # human md=3
    k_1=k_1*Pi
    k3=k3*ATP/(ATP+kd_ATP)
    k_2=k_2*kd_ATP/(ATP+kd_ATP)

    G0=-30.0; R=8.314/1000; ADP=36e-6
    Pi_M,ATP_M=Pi/1000,ATP/1000
    if Pi_M<=0 or ATP_M<=0: raise ValueError("ATP and Pi must be positive for thermo expression.")
    G=G0+R*T*np.log((ADP*Pi_M)/ATP_M)
    k_3=k1*k2*k3/(k_1*k_2*np.exp(-G/(R*T)))

    denom=k3*(k1+k2+k_1)+k1*k2+k_2*(k1+k_1)
    B0=k1*(k_2+k3)/denom*Z0
    C0=k1*k2/denom*Z0
    A0=Z0-B0-C0

    d11=-(k1+k_1+k2); d21=k_2-k1
    d31=sd[0]*(-k1*A0)+sd[1]*(-k_1*B0)+sd[2]*(k2*B0)
    d41=sd[3]*(k_2*C0); du1=k1*phi_l/Lmax
    d12=k2-k_3; d22=-(k3+k_2+k_3)
    d32=sd[2]*(-k2*B0)
    d42=sd[3]*(-k_2*C0)+sd[4]*(-k3*C0)
    du2=k_3*phi_l/Lmax
    d33=-phi_x/B0*(A0*k1+C0*k_2)
    d44=-phi_x/C0*(B0*k2+A0*k_3)

    omi=1j*(2*np.pi*np.asarray(freqs,float))
    HxB=phi_v*omi/(omi-d33)
    HxC=phi_v*omi/(omi-d44)
    den=(omi-d22)*(omi-d11)-d12*d21
    HC_s=(d12*(d31*HxB+d41*HxC)+(d32*HxB+d42*HxC)*(omi-d11))/den
    HC_l=(du1*d12+du2*(omi-d11))/den
    scale=L0/1000
    Y=scale*K*(B0*HxB+C0*HxC+(HC_s+HC_l)*xC0)+scale*Ks
    F=K*C0*xC0+Ks*.3
    return Y,float(F)

def qresp(p,data,target,T=295.0):
    _,f0=model(p,data["Baseline"]["met"],[1],T)
    _,ft=model(p,data[target]["met"],[1],T)
    return 100*(ft/f0-1)

def obs(p,data,freqs11,aug,T=295.0):
    b=data["Baseline"]
    Y,F=model(p,b["met"],freqs11,T)
    vals=[Y.real/b["se_re"][:11],Y.imag/b["se_im"][:11],np.array([F/b["se_F0"]])]
    if aug:
        a=data["ATP0.1"]
        _,Fa=model(p,a["met"],[1],T)
        vals.append(np.array([Fa/a["se_F0"]]))
    return np.concatenate(vals)

def jac_log(fun,p,h=1e-5):
    y0=np.asarray(fun(p),float); J=np.zeros((y0.size,len(FREE_IDX)))
    for c,i in enumerate(FREE_IDX):
        up,dn=p[i]*np.exp(h),p[i]*np.exp(-h)
        if dn>=LB[c] and up<=UB[c]:
            a=p.copy(); b=p.copy(); a[i]=up; b[i]=dn
            J[:,c]=(np.asarray(fun(a))-np.asarray(fun(b)))/(2*h)
        elif up<=UB[c]:
            a=p.copy(); a[i]=up; J[:,c]=(np.asarray(fun(a))-y0)/h
        else:
            b=p.copy(); b[i]=dn; J[:,c]=(y0-np.asarray(fun(b)))/h
    return J

def grad_q(p,data,target,T=295.0,h=1e-5):
    q0=qresp(p,data,target,T); g=np.zeros(len(FREE_IDX))
    for c,i in enumerate(FREE_IDX):
        up,dn=p[i]*np.exp(h),p[i]*np.exp(-h)
        if dn>=LB[c] and up<=UB[c]:
            a=p.copy(); b=p.copy(); a[i]=up; b[i]=dn
            g[c]=(qresp(a,data,target,T)-qresp(b,data,target,T))/(2*h)
        elif up<=UB[c]:
            a=p.copy(); a[i]=up; g[c]=(qresp(a,data,target,T)-q0)/h
        else:
            b=p.copy(); b[i]=dn; g[c]=(q0-qresp(b,data,target,T))/h
    return q0,g

def support(J,q0,g,p):
    ref=p[FREE_IDX]; lo=np.log(LB/ref); hi=np.log(UB/ref); bounds=list(zip(lo,hi))
    def ds(x): return float(np.dot(J@x,J@x))
    cons={"type":"ineq","fun":lambda x:CHI2-ds(x)}
    seeds=[np.zeros(len(ref))]
    gn=np.linalg.norm(g)
    if gn:
        u=g/gn
        for s in (.05,.2,.5,1,2):
            for sign in (-1,1):
                x=sign*s*u
                x=np.clip(x,lo,hi)
                v=ds(x)
                if v>CHI2 and v>0: x*=.95*math.sqrt(CHI2/v)
                seeds.append(np.clip(x,lo,hi))
    _,_,vh=np.linalg.svd(J,full_matrices=False)
    for v in vh[-min(3,len(vh)):]:
        for sign in (-1,1):
            x=np.clip(sign*v,lo,hi); val=ds(x)
            if val>CHI2 and val>0: x*=.95*math.sqrt(CHI2/val)
            seeds.append(np.clip(x,lo,hi))
    out={}
    for name,sgn in (("min",1),("max",-1)):
        best=None
        for x0 in seeds:
            r=minimize(lambda x:sgn*float(g@x),x0,method="SLSQP",
                       bounds=bounds,constraints=[cons],
                       options={"ftol":1e-12,"maxiter":3000})
            d=ds(r.x)
            if d<=CHI2+1e-6:
                q=q0+float(g@r.x)
                cand={"q":q,"deltaS":d,"success":bool(r.success),
                      "active_bounds":int(np.sum(np.isclose(r.x,lo,atol=1e-6)|
                                                np.isclose(r.x,hi,atol=1e-6))),
                      "delta_z":r.x.tolist()}
                if best is None or (q<best["q"] if name=="min" else q>best["q"]):
                    best=cand
        if best is None: raise RuntimeError(f"No feasible {name} endpoint.")
        out[name]=best
    out["width"]=out["max"]["q"]-out["min"]["q"]
    return out

def write_csv(path,rows):
    with open(path,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--p1-root",default=str(REPO / "data" / "third_party" / "human_model"))
    args=ap.parse_args()
    here=Path(__file__).resolve().parent.parent
    out=here/"results"; out.mkdir(parents=True,exist_ok=True)

    fit,datafile,modelsrc=find_sources(Path(args.p1_root))
    p=load_ref(fit); freqs,data=load_data(datafile); f11=freqs[:11]
    if np.any(p[FREE_IDX]<LB-1e-10) or np.any(p[FREE_IDX]>UB+1e-10):
        raise ValueError("ND x_p is outside declared human bounds.")

    manifest={
        "ND_xb_fit.mat":{"path":str(fit),"sha256":sha256(fit)},
        "ave_human_fitting_data.mat":{"path":str(datafile),"sha256":sha256(datafile)},
        "XBmodel_2024_linear_perms.m":{"path":str(modelsrc),"sha256":sha256(modelsrc)},
        "fixed_slots":{"phi_x":float(p[5]),"Ks":float(p[9]),"kd_Pi":float(p[13])},
        "n_freq_stored":int(len(freqs)),"n_freq_primary":11
    }
    (out/"input_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    fitrows=[]; temprows=[]
    for name in CONDS:
        d=data[name]
        Y295,F295=model(p,d["met"],f11,295)
        Y310,F310=model(p,d["met"],f11,310)
        yobs=d["Y"][:11]; sr=d["se_re"][:11]; si=d["se_im"][:11]
        rm295=float(np.sqrt(.5/11*np.sum(np.abs(yobs-Y295)**2)))
        rm310=float(np.sqrt(.5/11*np.sum(np.abs(yobs-Y310)**2)))
        pen295=max(abs(d["F0"]-F295)-d["se_F0"],0)/1000
        pen310=max(abs(d["F0"]-F310)-d["se_F0"],0)/1000
        fitrows.append({
            "condition":name,"ATP_mM":d["met"][0],"Pi_mM":d["met"][1],
            "observed_stress_kPa":d["F0"],"stress_SE_kPa":d["se_F0"],
            "model_stress_295K_kPa":F295,"model_stress_310K_kPa":F310,
            "CM_RMSE_295K_MPa":rm295,"CM_RMSE_310K_MPa":rm310,
            "source_objective_295K_MPa":rm295+pen295,
            "source_objective_310K_MPa":rm310+pen310
        })
        re=np.abs((Y310-Y295).real)/sr; im=np.abs((Y310-Y295).imag)/si
        temprows.append({
            "condition":name,
            "stress_shift_kPa":F310-F295,
            "stress_shift_SE_units":(F310-F295)/d["se_F0"],
            "max_CM_shift_SE_units":float(max(re.max(),im.max())),
            "median_CM_shift_SE_units":float(np.median(np.r_[re,im]))
        })
    write_csv(out/"reference_fit_diagnostics.csv",fitrows)
    write_csv(out/"temperature_sensitivity.csv",temprows)

    Jb=jac_log(lambda x:obs(x,data,f11,False,295),p)
    Ja=jac_log(lambda x:obs(x,data,f11,True,295),p)
    svrows=[]
    svdict={}
    for label,J in (("baseline",Jb),("augmented_ATP0.1_stress",Ja)):
        sv=np.linalg.svd(J,compute_uv=False); svdict[label]=sv.tolist()
        for i,s in enumerate(sv,1):
            svrows.append({"design":label,"rank_index":i,"singular_value":float(s),
                           "relative_to_largest":float(s/sv[0])})
    write_csv(out/"svd_summary.csv",svrows)

    summaries=[]; endpoint={}
    for target in ("ATP1","Pi10"):
        q0,g=grad_q(p,data,target,295)
        rb=support(Jb,q0,g,p); ra=support(Ja,q0,g,p)
        red=100*(1-ra["width"]/rb["width"]) if rb["width"]>0 else float("nan")
        summaries.append({
            "target":target,"q_reference_pp":q0,
            "baseline_low_pp":rb["min"]["q"],"baseline_high_pp":rb["max"]["q"],
            "baseline_width_pp":rb["width"],
            "augmented_low_pp":ra["min"]["q"],"augmented_high_pp":ra["max"]["q"],
            "augmented_width_pp":ra["width"],"width_reduction_percent":red,
            "baseline_min_active_bounds":rb["min"]["active_bounds"],
            "baseline_max_active_bounds":rb["max"]["active_bounds"],
            "augmented_min_active_bounds":ra["min"]["active_bounds"],
            "augmented_max_active_bounds":ra["max"]["active_bounds"]
        })
        endpoint[target]={"baseline":rb,"augmented":ra,"gradient_q_log":g.tolist()}
    write_csv(out/"local_information_summary.csv",summaries)
    (out/"local_information_endpoints.json").write_text(json.dumps(endpoint,indent=2),encoding="utf-8")

    f0=data["Baseline"]["F0"]
    observed={t:100*(data[t]["F0"]/f0-1) for t in ("ATP0.1","ATP1","Pi10")}
    q295={t:qresp(p,data,t,295) for t in ("ATP0.1","ATP1","Pi10")}
    q310={t:qresp(p,data,t,310) for t in ("ATP0.1","ATP1","Pi10")}
    result={
        "classification":"HUMAN_PILOT_STAGE1_LOCAL_INFORMATION",
        "primary_temperature_K":295.0,
        "sensitivity_temperature_K":310.0,
        "observed_group_responses_pp":observed,
        "reference_model_responses_295K_pp":q295,
        "reference_model_responses_310K_pp":q310,
        "local_information":{r["target"]:r for r in summaries},
        "svd":svdict,
        "temperature_summary":{
            "max_abs_stress_shift_SE_units":max(abs(r["stress_shift_SE_units"]) for r in temprows),
            "max_CM_shift_SE_units":max(r["max_CM_shift_SE_units"] for r in temprows),
            "ATP1_q_shift_pp":q310["ATP1"]-q295["ATP1"],
            "Pi10_q_shift_pp":q310["Pi10"]-q295["Pi10"]
        },
        "cautions":[
            "Local nominal support proxy, not calibrated patient-level inference.",
            "Human reference/model historically used the same five-condition dataset.",
            "No preparation-to-patient map is used.",
            "310 K is a no-refit sensitivity, not a corrected fit."
        ],
        "next_decision":"Review Stage-1 target contrast, bound activity and temperature sensitivity before any Stage-2 profile pilot."
    }
    (out/"human_pilot_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (out/"RUN_STATUS.md").write_text(
        "# Human Pilot Stage 1 — Run Status\n\n**Execution: PASS**\n\n"
        "Review `local_information_summary.csv`, `reference_fit_diagnostics.csv`, "
        "`svd_summary.csv`, and `temperature_sensitivity.csv` before any Stage 2 decision.\n",
        encoding="utf-8"
    )
    print(json.dumps({
        "status":"PASS",
        "ATP1_width_reduction_percent":summaries[0]["width_reduction_percent"],
        "Pi10_width_reduction_percent":summaries[1]["width_reduction_percent"],
        "temperature_summary":result["temperature_summary"],
        "results_dir":str(out)
    },indent=2))

if __name__=="__main__":
    main()
