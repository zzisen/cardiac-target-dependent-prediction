#!/usr/bin/env python3
"""Train and freeze B2 policies from the frozen B1 44-patient summary only."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def readcsv(p: Path):
    with p.open("r",encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))

def writecsv(p: Path,rows):
    with p.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def dumpjson(p: Path,obj):
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def fit_state(x: np.ndarray, y: np.ndarray, impl):
    xmean=float(np.mean(x,dtype=np.float64))
    xsd=float(np.std(x,ddof=1,dtype=np.float64))
    if xsd==0.0: xsd=1.0
    xz=(x-xmean)/xsd
    ymean,ysd=impl.train_center_scale(y)
    yz=(y-ymean)/ysd
    xc=xz-float(np.mean(xz,dtype=np.float64))
    yc=yz-np.mean(yz,axis=0,dtype=np.float64)
    den=float(np.dot(xc,xc))
    slope=np.zeros(y.shape[1],dtype=np.float64) if den==0.0 else (xc[:,None]*yc).sum(axis=0)/den
    intercept=np.mean(yz,axis=0,dtype=np.float64)-slope*float(np.mean(xz,dtype=np.float64))
    raw_slope=ysd*slope/xsd
    raw_intercept=ymean+ysd*intercept-raw_slope*xmean
    # Confirm coefficient form against the frozen Luna14 predictor implementation.
    for i in sorted(set((0,len(x)//2,len(x)-1))):
        ref=impl.predict_standardized_ols(x,y,float(x[i]))[0]
        got=intercept+slope*((float(x[i])-xmean)/xsd)
        if not np.allclose(ref,got,rtol=0.0,atol=1e-12):
            raise RuntimeError("B2 fit differs from the frozen B1 OLS implementation")
    return {"predictor_mean":xmean,"predictor_sample_sd":xsd,
        "response_mean":ymean.tolist(),"response_sample_sd":ysd.tolist(),
        "standardized_intercept":intercept.tolist(),"standardized_slope":slope.tolist(),
        "raw_intercept":raw_intercept.tolist(),"raw_slope":raw_slope.tolist()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[2])
    a=ap.parse_args(); root=a.project_root.resolve(); out=Path(__file__).resolve().parent
    b1=root/"P2_V3_ABT"/"B1_OUTCOME_EXECUTION_LUNA14_2026-10-06"
    b13=root/"P2_V3_ABT"/"B13R2_LOCAL_RECONCILIATION_2026-10-06"
    summary_path=b1/"B1_PATIENT_CONDITION_CHANNEL_SUMMARY.csv"
    manifest=json.loads((b1/"B1_MANIFEST.json").read_text(encoding="utf-8-sig"))
    mentry=next(x for x in manifest["payload_and_code_files"] if x["path"]=="B1_PATIENT_CONDITION_CHANNEL_SUMMARY.csv")
    if sha(summary_path)!=mentry["sha256"]: raise RuntimeError("B1 summary manifest hash mismatch")
    import zipfile
    with zipfile.ZipFile(root/"P2_V3_ABT"/"P2_V3_B1_OUTCOME_EXECUTION_LUNA14_2026-10-06.zip") as z:
        summary_bytes=z.read("B1_PATIENT_CONDITION_CHANNEL_SUMMARY.csv")
    if hashlib.sha256(summary_bytes).hexdigest()!=sha(summary_path): raise RuntimeError("B1 ZIP summary mismatch")

    amend=json.loads((b13/"B1_FINAL_PREOUTCOME_AMENDMENT.json").read_text(encoding="utf-8-sig"))
    patients=amend["cohort"]["primary_patient_ids"]
    targets=amend["conditions"]["targets"]
    channels=amend["channels"]["ordered"]
    conditions=["Basal"]+targets
    if len(patients)!=44 or len(set(patients))!=44 or len(channels)!=8 or len(targets)!=7:
        raise RuntimeError("Frozen B1 cohort/target/channel contract mismatch")
    if any(p in {f"UPN{i}" for i in range(90,99)} for p in patients):
        raise RuntimeError("Stanford candidate entered the training cohort")
    rows=readcsv(summary_path)
    expected={(p,c,ch) for p in patients for c in conditions for ch in channels}
    vals={}
    for r in rows:
        key=(r["patient_id"],r["condition"],r["channel"])
        if key in vals: raise RuntimeError("Duplicate B1 training summary key")
        vals[key]=float(r["positive_fraction"])
    if len(rows)!=44*8*8 or set(vals)!=expected or not all(math.isfinite(v) for v in vals.values()):
        raise RuntimeError("B1 summary is incomplete, non-finite, or differs from frozen training cohort")

    sys.path.insert(0,str(b1))
    import main_execution as b1_impl
    if b1_impl.CHANNELS!=channels or b1_impl.TARGETS!=targets:
        raise RuntimeError("B1 execution code channel/target definitions differ from frozen amendment")
    pidx={p:i for i,p in enumerate(patients)}
    tidx={t:i for i,t in enumerate(targets)}
    cidx={c:i for i,c in enumerate(channels)}
    x=np.asarray([[vals[(p,"Basal",c)] for c in channels] for p in patients],dtype=np.float64)
    y=np.asarray([[[vals[(p,t,c)] for c in channels] for t in targets] for p in patients],dtype=np.float64)
    n=len(patients); nt=len(targets); nc=len(channels)

    # Full-cohort inner LOPO for the transported SHARED action and each EXACT target.
    losses=np.empty((n,nc,nt),dtype=np.float64)
    for held in range(n):
        train=np.asarray([i for i in range(n) if i!=held],dtype=int)
        for ti in range(nt):
            yt=y[train,ti,:]
            ym,ys=b1_impl.train_center_scale(yt)
            testz=(y[held,ti,:]-ym)/ys
            for ci in range(nc):
                predz=b1_impl.predict_standardized_ols(x[train,ci],yt,x[held,ci])[0]
                losses[held,ci,ti]=float(np.mean((testz-predz)**2))
    if not np.isfinite(losses).all(): raise RuntimeError("Training-only inner LOPO scores are non-finite")
    by_target=np.mean(losses,axis=0,dtype=np.float64)
    shared_scores=np.mean(by_target,axis=1,dtype=np.float64)
    shared_ci=int(np.argmin(shared_scores))
    exact_ci=np.argmin(by_target,axis=0)
    shared_channel=channels[shared_ci]
    b2_targets=["Dasatinib","IL-7"]
    shared_rows=[]
    for ci,ch in enumerate(channels):
        row={"candidate_channel":ch,"candidate_order":ci+1,
             "all_seven_target_equal_mean_inner_LOPO_loss":float(shared_scores[ci]),
             "selected_for_B2_SHARED":ci==shared_ci}
        for ti,t in enumerate(targets): row["mean_inner_LOPO_loss__"+t]=float(by_target[ci,ti])
        shared_rows.append(row)
    exact_rows=[]
    for t in b2_targets:
        ti=tidx[t]; sel=int(exact_ci[ti])
        for ci,ch in enumerate(channels):
            exact_rows.append({"target":t,"candidate_channel":ch,"candidate_order":ci+1,
                "mean_inner_LOPO_loss":float(by_target[ci,ti]),"selected_for_B2_EXACT":ci==sel})
    writecsv(out/"B2_TRAINING_ONLY_SHARED_SELECTION.csv",shared_rows)
    writecsv(out/"B2_TRAINING_ONLY_EXACT_SELECTION.csv",exact_rows)

    # Freeze response scaling for all seven training targets; fit the three transported
    # policy classes for the two B2 targets on all 44 training patients.
    scaling=[]
    target_stats={}
    for ti,t in enumerate(targets):
        ym,ys=b1_impl.train_center_scale(y[:,ti,:])
        target_stats[t]={"mean":ym.tolist(),"sample_sd":ys.tolist()}
        for ci,ch in enumerate(channels):
            scaling.append({"target":t,"channel":ch,"training_mean":float(ym[ci]),
                "training_sample_sd_ddof1_zero_replaced_by_1":float(ys[ci])})
    writecsv(out/"B2_TRAINING_ONLY_SCALING.csv",scaling)

    selected={"SHARED":{t:shared_channel for t in b2_targets},
              "EXACT":{t:channels[int(exact_ci[tidx[t]])] for t in b2_targets}}
    models={"SHARED":{},"EXACT":{}}
    coef_rows=[]
    for policy in ("SHARED","EXACT"):
        for t in b2_targets:
            ch=selected[policy][t]; ci=cidx[ch]; ti=tidx[t]
            state=fit_state(x[:,ci],y[:,ti,:],b1_impl)
            models[policy][t]={"selected_baseline_channel":ch,**state}
            for oi,response_channel in enumerate(channels):
                coef_rows.append({"policy":policy,"target":t,"predictor_channel":ch,
                    "response_channel":response_channel,"predictor_training_mean":state["predictor_mean"],
                    "predictor_training_sample_sd":state["predictor_sample_sd"],
                    "response_training_mean":state["response_mean"][oi],
                    "response_training_sample_sd":state["response_sample_sd"][oi],
                    "intercept_standardized":state["standardized_intercept"][oi],
                    "slope_standardized":state["standardized_slope"][oi],
                    "intercept_raw":state["raw_intercept"][oi],"slope_raw":state["raw_slope"][oi]})
    writecsv(out/"B2_TRAINING_ONLY_MODEL_COEFFICIENTS.csv",coef_rows)

    amendment_path=out/"B2_STANFORD_SITE_PROVENANCE_AMENDMENT.md"
    preflight_add=out/"B2_STANFORD_PREFLIGHT_AMENDMENT.json"
    pdf=root/"01_Data"/"Good2018_source_docs"/"41591_2018_BFnm4505_MOESM1_ESM.pdf"
    amendment_sha=sha(amendment_path)
    pre= json.loads(preflight_add.read_text(encoding="utf-8-sig"))
    if amendment_sha!=pre["site_provenance_amendment_sha256"]: raise RuntimeError("Site amendment hash changed")
    if pre["stanford_event_values_decoded"] or sha(pdf)!=pre["source_pdf"]["sha256"]:
        raise RuntimeError("Site provenance freeze or PDF identity changed before training freeze")
    if (out/"B2_STANFORD_SITE_PROVENANCE_AMENDMENT.sha256").read_text(encoding="ascii").split()[0]!=amendment_sha:
        raise RuntimeError("Site amendment sidecar mismatch")
    root_b1_zip=root/"P2_V3_ABT"/"P2_V3_B1_OUTCOME_EXECUTION_LUNA14_2026-10-06.zip"
    root_pre_zip=root/"P2_V3_ABT"/"P2_V3_B1_FINAL_PREOUTCOME_AMENDMENT_LOCAL_B13R2_2026-10-06.zip"
    hash_inputs={
        "B1_summary_sha256":sha(summary_path),"B1_execution_code_sha256":sha(b1/"main_execution.py"),
        "B1_primary_registry_sha256":sha(b1/"B1_PRIMARY_FILE_REGISTRY.csv"),
        "B1_contract_lock_sha256":sha(b1/"B1_OUTCOME_EXECUTION_CONTRACT_LOCK.md"),
        "B1_outcome_zip_sha256":sha(root_b1_zip),"B1_preoutcome_zip_sha256":sha(root_pre_zip),
        "B13R2_channel_registry_sha256":sha(b13/"B13R2_FINAL_CHANNEL_REGISTRY.csv"),
        "B2_site_amendment_sha256":amendment_sha,"B2_preflight_amendment_sha256":sha(preflight_add),
        "site_provenance_pdf_sha256":sha(pdf),"training_freeze_code_sha256":sha(Path(__file__).resolve())}
    registry={
        "package_id":"P2_V3_B2_STANFORD_TRANSPORT_LUNA15_2026-10-06",
        "state":"FROZEN_TRAINING_ONLY_BEFORE_STANFORD_EVENT_DECODE",
        "training_patients":patients,"training_patient_count":44,"B2_test_patient_overlap":[],
        "baseline":"Basal","B1_training_targets_all_seven":targets,"B2_evaluation_targets":b2_targets,
        "candidate_channels_ordered":channels,
        "summary_definition":"positive_fraction = count(event value >= 10) / N(events); reused from frozen B1 summary",
        "predictor":"one scalar Basal channel; separate OLS with intercept for each response channel and target",
        "scaling":"training-only means and sample SD (ddof=1); zero SD replaced with 1; test scaling prohibited",
        "loss":"mean squared standardized error over 8 response channels",
        "selection":{
            "SHARED":{"targets_used_for_selection":targets,"equal_target_weights":True,
                "inner_validation":"leave-one-patient-out over all 44 B1 training patients",
                "scores_by_candidate":{channels[i]:float(shared_scores[i]) for i in range(nc)},
                "selected_channel":shared_channel},
            "EXACT":{t:{"inner_validation":"leave-one-patient-out over all 44 B1 training patients",
                "scores_by_candidate":{channels[i]:float(by_target[i,tidx[t]]) for i in range(nc)},
                "selected_channel":selected["EXACT"][t]} for t in b2_targets}},
        "C0_target_response_means_and_scales":target_stats,
        "refit_models_on_all_44_training_patients":models,
        "source_hashes":hash_inputs,
        "stanford_values_accessed_before_freeze":False}
    dumpjson(out/"B2_TRAINING_ONLY_MODEL_REGISTRY.json",registry)
    freeze_files=["B2_TRAINING_ONLY_SHARED_SELECTION.csv","B2_TRAINING_ONLY_EXACT_SELECTION.csv",
        "B2_TRAINING_ONLY_SCALING.csv","B2_TRAINING_ONLY_MODEL_COEFFICIENTS.csv","B2_TRAINING_ONLY_MODEL_REGISTRY.json",
        "B2_TRAINING_ONLY_FREEZE.py","B2_TRAINING_ONLY_MODEL_REGISTRY.sha256",
        "B2_TRAINING_SOURCE_HASHES.csv","B2_STANFORD_SITE_PROVENANCE_AMENDMENT.md",
        "B2_STANFORD_SITE_PROVENANCE_AMENDMENT.sha256","B2_STANFORD_PREFLIGHT_AMENDMENT.json"]
    # Extend source ledger with the now-frozen amendment and training inputs.
    src_rows=[]
    paths=[
        ("B2_site_provenance_amendment",amendment_path,"pre-outcome amendment SHA verified before B2 event decoding"),
        ("B2_site_provenance_pdf",pdf,"local official supplementary PDF; provenance footnote only"),
        ("B2_additive_preflight_amendment",preflight_add,"confirms provenance fix and no event decode at freeze"),
        ("B1_training_summary",summary_path,"authoritative summary matches package manifest and ZIP member"),
        ("B1_training_primary_registry",b1/"B1_PRIMARY_FILE_REGISTRY.csv","frozen 44-patient cohort membership"),
        ("B1_training_contract",b1/"B1_OUTCOME_EXECUTION_CONTRACT_LOCK.md","source-defined B1 regression and loss contract"),
        ("B1_execution_code",b1/"main_execution.py","reused standardization/OLS implementation"),
        ("B13R2_channel_registry",b13/"B13R2_FINAL_CHANNEL_REGISTRY.csv","frozen ordered channels"),
        ("B1_outcome_zip",root_b1_zip,"authoritative B1 package"),
        ("B1_preoutcome_zip",root_pre_zip,"authoritative B1 pre-outcome package")]
    for name,p,note in paths:
        src_rows.append({"source_id":name,"path_or_member":str(p),"byte_size":p.stat().st_size,"sha256":sha(p),"verification_note":note})
    writecsv(out/"B2_TRAINING_SOURCE_HASHES.csv",src_rows)
    # Refresh registry source ledger with the generated ledger's digest, then reserialize.
    registry["source_hashes"]["B2_TRAINING_SOURCE_HASHES.csv"]=sha(out/"B2_TRAINING_SOURCE_HASHES.csv")
    dumpjson(out/"B2_TRAINING_ONLY_MODEL_REGISTRY.json",registry)
    model_sha=sha(out/"B2_TRAINING_ONLY_MODEL_REGISTRY.json")
    (out/"B2_TRAINING_ONLY_MODEL_REGISTRY.sha256").write_text(model_sha+"  B2_TRAINING_ONLY_MODEL_REGISTRY.json\n",encoding="ascii")
    lock_obj={"state":"TRAINING_ONLY_STATE_FROZEN_BEFORE_STANFORD_EVENT_DECODE",
        "freeze_timestamp_utc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "training_patient_count":44,"test_event_values_decoded":False,
        "site_amendment_sha256":amendment_sha,"model_registry_sha256":model_sha,
        "frozen_file_hashes":{name:sha(out/name) for name in freeze_files},
        "assertions":{"only_frozen_B1_44_patients_used":True,"shared_selection_uses_all_seven_B1_targets":True,
            "exact_selection_uses_training_patients_only":True,"C0_shared_exact_refit_on_all_44":True,
            "no_Stanford_values_used_for_selection_scaling_or_fit":True}}
    dumpjson(out/"B2_TRAINING_FREEZE_LOCK.json",lock_obj)
    (out/"B2_TRAINING_FREEZE_LOCK.json.sha256").write_text(sha(out/"B2_TRAINING_FREEZE_LOCK.json")+"  B2_TRAINING_FREEZE_LOCK.json\n",encoding="ascii")
    print(json.dumps({"training_state":"FROZEN_BEFORE_STANFORD_EVENT_DECODE","training_patients":44,
        "shared_channel":shared_channel,"exact_channels":selected["EXACT"],
        "model_registry_sha256":model_sha,"freeze_lock_sha256":sha(out/"B2_TRAINING_FREEZE_LOCK.json"),
        "stanford_values_accessed":False},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
