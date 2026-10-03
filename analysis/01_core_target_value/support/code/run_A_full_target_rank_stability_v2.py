"""Bounded multi-start endpoint/rank cross-check for F's common action set.

This is a sensitivity analysis, not a replacement for the complete A search.
It checks baseline intervals and the four common F actions for ATP1/Pi5 under
both frozen pair-domain margins, with current endpoint, baseline endpoint,
MLE, and deterministic serialized alternatives retained as starts.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

HERE=Path(__file__).resolve().parent
BRANCH=HERE.parents[1]
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(BRANCH/"01_Baseline_Reproduction"/"code"))
from aj_core import (OUT,PublicLikelihood,load_public_rat,load_seeds,domain_for,
                     candidate_definitions,save_json,sha256)
from aj_core_v2 import support_endpoint_v2

SOURCE=OUT/"A_measurement_target_map.json"
RESULT=OUT/"A_full_target_rank_stability_v2.json"
PROGRESS=OUT/"A_full_target_rank_stability_v2_progress.json"
ACTION_IDS_BY_TARGET={
    "ATP1":["stress_ATP0.1","stress_Pi0","stress_Pi5","CM_spectrum_ATP0.1","CM_spectrum_Pi0","CM_spectrum_Pi5"],
    "Pi5":["stress_ATP0.1","stress_ATP1","stress_Pi0","CM_spectrum_ATP0.1","CM_spectrum_ATP1","CM_spectrum_Pi0"]}
TARGETS=["ATP1","Pi5"]
DOMAINS=[0.5,1.0]

def main():
    original=json.loads(SOURCE.read_text(encoding="utf-8")); data=load_public_rat(); seeds=load_seeds()
    if data.raw_sha256!=original["input_sha256"]: raise RuntimeError("A input hash mismatch")
    candidates={x["id"]:x for x in candidate_definitions()}
    out={"source_result_sha256":sha256(SOURCE),"input_sha256":data.raw_sha256,
         "scope":"All six candidate actions eligible after same-condition exclusion for targets ATP1/Pi5, both pair-domain margins; restarts from frozen MLE, stored endpoint, corresponding baseline endpoint, other-side incumbent, plus deterministic seeds.",
         "max_starts":6,"maxiter":1300,"domains":{},"endpoint_searches":[],"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    start=time.time()
    for margin in DOMAINS:
        dname=f"P1_pair_plus_{margin:.1f}"; src=original["domains"][dname]
        dom=domain_for(margin,seeds); like0=PublicLikelihood(data); basefit=src["baseline_fit"]
        recdom={"margin":margin,"baseline":{},"actions":{}}
        for target in TARGETS:
            oldbase=src["baseline_support"][target]
            ends={}
            for side in ("low","high"):
                other="high" if side=="low" else "low"
                new=support_endpoint_v2(like0,dom,basefit,target,side,seeds,
                     warm=[np.asarray(oldbase[side]["params"]),np.asarray(oldbase[other]["params"])],max_starts=6)
                ends[side]=new
                out["endpoint_searches"].append({"domain":dname,"measurement":"baseline_only","target":target,
                    "side":side,"old_q":oldbase[side]["q"],"new_q":new["q"],
                    "abs_q_shift":abs(new["q"]-oldbase[side]["q"]),"old_objective":oldbase[side]["objective"],
                    "new_objective":new["objective"],"selected_start":new["selected_start"],
                    "optimizer_success":new["optimizer_success"],"n_starts":len(new["all_attempts"]),
                    "n_feasible_starts":sum(bool(x.get("feasible")) for x in new["all_attempts"]),"attempts":new["all_attempts"]})
                out["_partial"]={"n_searches":len(out["endpoint_searches"]),"elapsed_seconds":time.time()-start}
                save_json(PROGRESS,out)
            recdom["baseline"][target]={"old_width":oldbase["high"]["q"]-oldbase["low"]["q"],
                 "new_width":ends["high"]["q"]-ends["low"]["q"],"low":ends["low"],"high":ends["high"]}
        for target in TARGETS:
            for cid in ACTION_IDS_BY_TARGET[target]:
                cand=candidates[cid]; like=PublicLikelihood(data,cand); oldm=src["measurements"][cid]
                recdom["actions"].setdefault(cid,{})
                old=oldm["support"][target]
                newends={}
                for side in ("low","high"):
                    other="high" if side=="low" else "low"
                    baseend=src["baseline_support"][target][side]
                    # The actual archived augmented endpoint and baseline
                    # endpoint are forced ahead of the deterministic seed bank.
                    warm=[np.asarray(old[side]["params"]),np.asarray(baseend["params"]),np.asarray(old[other]["params"])]
                    new=support_endpoint_v2(like,dom,oldm["fit"],target,side,seeds,warm=warm,max_starts=6)
                    newends[side]=new
                    out["endpoint_searches"].append({"domain":dname,"measurement":cid,"target":target,"side":side,
                        "old_q":old[side]["q"],"new_q":new["q"],"abs_q_shift":abs(new["q"]-old[side]["q"]),
                        "old_objective":old[side]["objective"],"new_objective":new["objective"],
                        "selected_start":new["selected_start"],"optimizer_success":new["optimizer_success"],
                        "n_starts":len(new["all_attempts"]),"n_feasible_starts":sum(bool(x.get("feasible")) for x in new["all_attempts"]),
                        "attempts":new["all_attempts"]})
                    out["_partial"]={"n_searches":len(out["endpoint_searches"]),"elapsed_seconds":time.time()-start}
                    save_json(PROGRESS,out)
                basewidth=recdom["baseline"][target]["new_width"]
                width=newends["high"]["q"]-newends["low"]["q"]
                oldwidth=old["high"]["q"]-old["low"]["q"]
                recdom["actions"][cid][target]={"old_relative_reduction_pct":100*(old["baseline_width_q_pct_points"]-oldwidth)/old["baseline_width_q_pct_points"],
                    "new_relative_reduction_pct":100*(basewidth-width)/basewidth,
                    "old_width":oldwidth,"new_width":width,"baseline_old_width":old["baseline_width_q_pct_points"],
                    "baseline_new_width":basewidth,"low":newends["low"],"high":newends["high"]}
        out["domains"][dname]=recdom
        # Keep a durable domain checkpoint if the second margin is interrupted.
        out["_partial"]={"n_searches":len(out["endpoint_searches"]),"elapsed_seconds":time.time()-start}
        save_json(PROGRESS,out)
    # Summarize head-to-head ranks only within this equal-budget action subset.
    ranks=[]
    for dname,dr in out["domains"].items():
        for target in TARGETS:
            rows=[{"measurement":cid,"relative_width_reduction_pct":dr["actions"][cid][target]["new_relative_reduction_pct"]} for cid in ACTION_IDS_BY_TARGET[target]]
            rows.sort(key=lambda r:r["relative_width_reduction_pct"],reverse=True)
            for i,r in enumerate(rows,1): r["rank"]=i;r["domain"]=dname;r["target"]=target
            ranks.extend(rows)
    out["equal_budget_rankings"]=ranks
    out["optimizer_summary"]={"endpoint_searches":len(out["endpoint_searches"]),
      "total_optimizer_attempts":sum(len(x.get("attempts",[])) for x in out["endpoint_searches"]),
      "non_successful_attempts":sum(1 for x in out["endpoint_searches"] for a in x.get("attempts",[]) if not a.get("success",False)),
      "max_abs_endpoint_q_shift":max(x["abs_q_shift"] for x in out["endpoint_searches"])}
    # Compute exact cutoff excess with serialized source MLE values.
    max_excess=-float("inf")
    for x in out["endpoint_searches"]:
        dr=original["domains"][x["domain"]]
        obj=(dr["baseline_fit"]["objective"] if x["measurement"]=="baseline_only" else dr["measurements"][x["measurement"]]["fit"]["objective"])
        max_excess=max(max_excess,x["new_objective"]-(obj+3.841458820694124))
    out["optimizer_summary"]["max_lr_excess_vs_source_mle"]=max_excess
    out["completed_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()); out["elapsed_seconds"]=time.time()-start
    out.pop("_partial",None); save_json(RESULT,out)
    print(json.dumps({k:v for k,v in out.items() if k not in ("endpoint_searches","domains","equal_budget_rankings")},indent=2))
    print("EQUAL_BUDGET_RANKINGS")
    print(json.dumps(ranks,indent=2))

if __name__=="__main__": main()
