"""Run frozen A measurement-by-target map on the public rat MAT only."""
from __future__ import annotations
import json, sys, time
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parents[1]/"01_Baseline_Reproduction"/"code"))
from aj_core import (BRANCH,OUT,CONFIG,PublicLikelihood,load_public_rat,load_seeds,
                     domain_for,fit_mle,support_endpoint,candidate_definitions,
                     read_freeze,save_json,sha256)

def main():
    freeze=read_freeze(); data=load_public_rat(); seeds=load_seeds()
    if data.raw_sha256!=freeze["primary_input"]["sha256"]: raise RuntimeError("Public input hash changed after freeze")
    targets=freeze["future_targets"]; margins=freeze["domains"]["margins"]; candidates=candidate_definitions()
    state={"freeze_id":freeze["freeze_id"],"input_sha256":data.raw_sha256,"fit_log":{},"support_rows":[],"exclusions":[],"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    progress=OUT/"A_partial_progress.json"; results={"freeze_id":freeze["freeze_id"],"input_sha256":data.raw_sha256,"domains":{}}
    start_clock=time.time()
    for margin in margins:
        dom=domain_for(float(margin),seeds); dname=f"P1_pair_plus_{float(margin):.1f}"
        print(f"DOMAIN {dname}: baseline fit",flush=True)
        base_like=PublicLikelihood(data)
        base_fit=fit_mle(base_like,dom,seeds)
        state["fit_log"][f"{dname}:baseline_only"]=base_fit
        # Baseline support is target-specific even though the likelihood is shared.
        base_support={}
        for target in targets:
            print(f"  {dname} baseline endpoints {target}",flush=True)
            low=support_endpoint(base_like,dom,base_fit,target,"low",seeds)
            high=support_endpoint(base_like,dom,base_fit,target,"high",seeds,warm=[np.asarray(low["params"])])
            rec={"domain":dname,"margin":float(margin),"measurement_id":"baseline_only","measurement_type":"baseline","measurement_condition":"baseline","target":target,"status":"computed","baseline_mle_objective":base_fit["objective"],"augmented_mle_objective":base_fit["objective"],"low":low,"high":high}
            rec["width_q_pct_points"]=high["q"]-low["q"]
            rec["endpoint_pair_feasible"]=bool(low["feasible"] and high["feasible"])
            base_support[target]=rec
            state["support_rows"].append(rec)
            save_json(progress,state)
        results["domains"][dname]={"domain":{"margin":float(margin),"lo":dom["lo"].tolist(),"hi":dom["hi"].tolist()},
                                    "baseline_fit":base_fit,"baseline_support":base_support,"measurements":{}}
        warm_by_target={t:{"low":[np.asarray(base_support[t]["low"]["params"])],"high":[np.asarray(base_support[t]["high"]["params"])]} for t in targets}
        for cand in candidates:
            cid=cand["id"]; print(f"  {dname} fit {cid}",flush=True)
            like=PublicLikelihood(data,cand)
            fit=fit_mle(like,dom,seeds)
            state["fit_log"][f"{dname}:{cid}"]=fit
            measurement={"definition":cand,"fit":fit,"support":{}}
            for target in targets:
                if cand["condition"]==target:
                    exclusion={"domain":dname,"measurement_id":cid,"target":target,"reason":"excluded_same_condition_future_target","status":"predeclared_exclusion"}
                    measurement["support"][target]=exclusion
                    state["exclusions"].append(exclusion)
                    continue
                print(f"    {dname} endpoints {cid} -> {target}",flush=True)
                low=support_endpoint(like,dom,fit,target,"low",seeds,warm=warm_by_target[target]["low"])
                high=support_endpoint(like,dom,fit,target,"high",seeds,warm=warm_by_target[target]["high"]+[np.asarray(low["params"])])
                baseline=base_support[target]
                bw=float(baseline["width_q_pct_points"]); aw=float(high["q"]-low["q"])
                rec={"domain":dname,"margin":float(margin),"measurement_id":cid,"measurement_type":cand["type"],"measurement_condition":cand["condition"],"n_scalar_observations":cand["n_scalar_observations"],"target":target,"status":"computed","baseline_mle_objective":base_fit["objective"],"augmented_mle_objective":fit["objective"],"baseline_width_q_pct_points":bw,"augmented_width_q_pct_points":aw,"absolute_width_reduction_q_pct_points":bw-aw,"relative_width_reduction_pct":((bw-aw)/bw*100.0 if bw!=0 else None),"baseline_low":baseline["low"],"baseline_high":baseline["high"],"low":low,"high":high,"endpoint_pair_feasible":bool(low["feasible"] and high["feasible"]),"both_endpoint_optimizers_report_success":bool(low["optimizer_success"] and high["optimizer_success"]),"both_endpoints_within_nominal_lr_cutoff":bool(low["delta_lr"]<=3.841458820694124+1e-4 and high["delta_lr"]<=3.841458820694124+1e-4)}
                measurement["support"][target]=rec; state["support_rows"].append(rec)
                warm_by_target[target]["low"].append(np.asarray(low["params"]))
                warm_by_target[target]["high"].append(np.asarray(high["params"]))
                save_json(progress,state)
            results["domains"][dname]["measurements"][cid]=measurement
            save_json(progress,state)
    results["completed_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    results["elapsed_seconds"]=time.time()-start_clock
    results["solver_summary"]={"total_valid_cells":sum(1 for r in state["support_rows"] if r.get("measurement_id")!="baseline_only"),
                               "predeclared_exclusions":len(state["exclusions"]),
                               "baseline_intervals":sum(1 for r in state["support_rows"] if r.get("measurement_id")=="baseline_only"),
                               "endpoint_optimizer_failures":sum(1 for r in state["support_rows"] if not (r["low"]["optimizer_success"] and r["high"]["optimizer_success"])),
                               "infeasible_endpoint_pairs":sum(1 for r in state["support_rows"] if not r["endpoint_pair_feasible"])}
    outjson=OUT/"A_measurement_target_map.json"; save_json(outjson,results)
    rows=[]
    for dname,dr in results["domains"].items():
        for target,s in dr["baseline_support"].items():
            rows.append({"domain":dname,"measurement_id":"baseline_only","measurement_type":"baseline","measurement_condition":"baseline","target":target,"status":"baseline_reference","n_scalar_observations":0,"baseline_width_q_pct_points":s["width_q_pct_points"],"augmented_width_q_pct_points":s["width_q_pct_points"],"absolute_width_reduction_q_pct_points":0.0,"relative_width_reduction_pct":0.0,"low_q":s["low"]["q"],"high_q":s["high"]["q"],"low_delta_lr":s["low"]["delta_lr"],"high_delta_lr":s["high"]["delta_lr"],"endpoint_pair_feasible":s["endpoint_pair_feasible"],"endpoint_optimizer_success":bool(s["low"]["optimizer_success"] and s["high"]["optimizer_success"])})
        for cid,m in dr["measurements"].items():
            for target,s in m["support"].items():
                if s["status"]!="computed":
                    rows.append({"domain":dname,"measurement_id":cid,"measurement_type":m["definition"]["type"],"measurement_condition":m["definition"]["condition"],"target":target,"status":s["status"],"n_scalar_observations":m["definition"]["n_scalar_observations"]}); continue
                rows.append({"domain":dname,"measurement_id":cid,"measurement_type":s["measurement_type"],"measurement_condition":s["measurement_condition"],"target":target,"status":s["status"],"n_scalar_observations":s["n_scalar_observations"],"baseline_width_q_pct_points":s["baseline_width_q_pct_points"],"augmented_width_q_pct_points":s["augmented_width_q_pct_points"],"absolute_width_reduction_q_pct_points":s["absolute_width_reduction_q_pct_points"],"relative_width_reduction_pct":s["relative_width_reduction_pct"],"low_q":s["low"]["q"],"high_q":s["high"]["q"],"low_delta_lr":s["low"]["delta_lr"],"high_delta_lr":s["high"]["delta_lr"],"endpoint_pair_feasible":s["endpoint_pair_feasible"],"endpoint_optimizer_success":s["both_endpoint_optimizers_report_success"]})
    frame=pd.DataFrame(rows)
    for dname in frame["domain"].dropna().unique():
        mask=(frame.domain==dname)&(frame.status=="computed")
        frame.loc[mask,"rank_within_target"]=frame.loc[mask].groupby("target")["relative_width_reduction_pct"].rank(method="min",ascending=False)
        frame.loc[mask,"rank_within_measurement"]=frame.loc[mask].groupby("measurement_id")["relative_width_reduction_pct"].rank(method="min",ascending=False)
    frame.to_csv(OUT/"A_support_matrix.csv",index=False)
    print(json.dumps({"out":str(outjson),"sha256":sha256(outjson),"summary":results["solver_summary"],"elapsed_seconds":results["elapsed_seconds"]},indent=2),flush=True)

if __name__=="__main__": main()
