#!/usr/bin/env python3
"""Decode only the frozen 27 Stanford FCS files and score frozen B2 policies."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

PATIENTS=[f"UPN{i}" for i in range(90,99)]
CONDITIONS=["Basal","Dasatinib","IL-7"]
TARGETS=["Dasatinib","IL-7"]
CHANNELS=["pS6","p4EBP1","pSTAT5","pSYK","pPLCγ2","pAKT","pERK1/2","pIKAROS"]

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def readcsv(p):
    with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def writecsv(p,rows,fields=None):
    if fields is None:fields=list(rows[0])
    with p.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def readjson(p):return json.loads(p.read_text(encoding="utf-8-sig"))

def verify_frozen_state(out,root):
    lock=readjson(out/"B2_TRAINING_FREEZE_LOCK.json")
    for name,h in lock["frozen_file_hashes"].items():
        if sha(out/name)!=h:raise RuntimeError("Training freeze artifact changed: "+name)
    registry_path=out/"B2_TRAINING_ONLY_MODEL_REGISTRY.json"
    registry=readjson(registry_path); registry_hash=sha(registry_path)
    if registry_hash!=lock["model_registry_sha256"]:raise RuntimeError("Training model registry hash mismatch")
    if (out/"B2_TRAINING_ONLY_MODEL_REGISTRY.sha256").read_text(encoding="ascii").split()[0]!=registry_hash:
        raise RuntimeError("Training model registry sidecar mismatch")
    amendment=out/"B2_STANFORD_SITE_PROVENANCE_AMENDMENT.md"
    pre=readjson(out/"B2_STANFORD_PREFLIGHT_AMENDMENT.json")
    pdf=root/"01_Data"/"Good2018_source_docs"/"41591_2018_BFnm4505_MOESM1_ESM.pdf"
    ah=sha(amendment)
    if ah!=lock["site_amendment_sha256"] or ah!=pre["site_provenance_amendment_sha256"]:
        raise RuntimeError("Site-provenance amendment changed after pre-outcome freeze")
    if sha(pdf)!=pre["source_pdf"]["sha256"]:raise RuntimeError("Local supplementary PDF hash changed")
    return lock,registry,registry_hash,pre

def norm(s):
    import unicodedata
    s=unicodedata.normalize("NFKC",s).casefold().replace("gamma","g").replace("γ","g")
    import re
    return re.sub(r"[^a-z0-9]","",s)
ALIASES={"ps6":"pS6","p4ebp1":"p4EBP1","pstat5":"pSTAT5","psyk":"pSYK",
    "pplcg12":"pPLCγ2","pplcg2":"pPLCγ2","pakt":"pAKT","perk":"pERK1/2",
    "perk12":"pERK1/2","pikaros":"pIKAROS"}

def build_registry_rows(inv):
    out=[]
    for r in sorted(inv,key=lambda q:(PATIENTS.index(q["sample_id"]),CONDITIONS.index(q["condition"]))):
        pars=json.loads(r["parameter_mapping_json"]); hits={c:[] for c in CHANNELS}
        for p in pars:
            c=ALIASES.get(norm(p.get("PnS","")))
            if c in hits:hits[c].append(p)
        if any(len(hits[c])!=1 for c in CHANNELS):raise RuntimeError("Test channel mapping not unique: "+r["filename"])
        pcreb=[p for p in pars if p.get("PnS","").strip().casefold()=="pcreb"]
        if len(pcreb)!=1:raise RuntimeError("Test pCREB header identity not unique: "+r["filename"])
        iso=pcreb[0].get("PnN","").lower()
        panel="Lu176_panel" if "lu176" in iso else "Yb176_group" if "yb176" in iso else ""
        if not panel:raise RuntimeError("Unsupported panel header group: "+r["filename"])
        idx={c:int(hits[c][0]["parameter_number"]) for c in CHANNELS}
        pnn={c:hits[c][0]["PnN"] for c in CHANNELS}
        pns={c:hits[c][0]["PnS"] for c in CHANNELS}
        out.append({"patient_id":r["sample_id"],"condition":r["condition"],
            "target_or_baseline":"baseline" if r["condition"]=="Basal" else "target",
            "archive_member_path":r["archive_member_path"],"source_fcs_bytes":int(r["byte_size"]),
            "source_member_sha256":r["sha256"],"expected_event_count":int(r["TOT"]),
            "parameter_count":int(r["PAR"]),"panel_registry_group":panel,
            "channel_parameter_indices_json":json.dumps(idx,separators=(",",":")),
            "channel_parameter_pnn_json":json.dumps(pnn,ensure_ascii=False,separators=(",",":")),
            "channel_parameter_pns_json":json.dumps(pns,ensure_ascii=False,separators=(",",":"))})
    expected={(p,c) for p in PATIENTS for c in CONDITIONS}
    if len(out)!=27 or {(r["patient_id"],r["condition"]) for r in out}!=expected:
        raise RuntimeError("Frozen test registry differs from exact 9 x 3 set")
    return out

def model_predict(model,x):
    xz=(float(x)-model["predictor_mean"])/model["predictor_sample_sd"]
    predz=np.asarray(model["standardized_intercept"],dtype=np.float64)+np.asarray(model["standardized_slope"],dtype=np.float64)*xz
    ym=np.asarray(model["response_mean"],dtype=np.float64)
    ys=np.asarray(model["response_sample_sd"],dtype=np.float64)
    return predz,ym+ys*predz

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[2])
    a=ap.parse_args();root=a.project_root.resolve();out=Path(__file__).resolve().parent
    lock,model,model_hash,pre=verify_frozen_state(out,root)
    b1dir=root/"P2_V3_ABT"/"B1_OUTCOME_EXECUTION_LUNA14_2026-10-06"
    channelpath=root/"P2_V3_ABT"/"B13R2_LOCAL_RECONCILIATION_2026-10-06"/"B13R2_FINAL_CHANNEL_REGISTRY.csv"
    wrapper=root/"01_Data"/"doi_10_5061_dryad_pvmcvdnxc__v20250711.zip"
    srcledger=readcsv(b1dir/"B1_SOURCE_AND_FREEZE_HASHES.csv")
    frozen_wrapper=next(r["sha256"] for r in srcledger if r["source_id"]=="outer_archive_wrapper")
    if not wrapper.exists() or sha(wrapper)!=frozen_wrapper:raise RuntimeError("DDPR source wrapper differs from frozen B1 source hash")
    inv=readcsv(out/"support"/"B2_SOURCE_FCS_HEADER_ROWS.csv")
    sample_rows=readcsv(out/"support"/"B2_SOURCE_SAMPLE_REGISTRY_ROWS.csv")
    dup_rows=readcsv(out/"support"/"B2_SOURCE_DUPLICATE_SPLIT_ROWS.csv")
    if len(sample_rows)!=9 or {r["sample_id"] for r in sample_rows}!=set(PATIENTS):raise RuntimeError("Patient set changed")
    if len(dup_rows)!=27 or any(r["classification"]!="SINGLE_FILE" or r["file_count"]!="1" for r in dup_rows):
        raise RuntimeError("Duplicate/split structural freeze changed")
    registry=build_registry_rows(inv)
    # Add an amended registry without altering the prior-hold registry or original preflight.
    prior=readcsv(out/"B2_STANFORD_STRUCTURAL_REGISTRY.csv")
    amended=[]
    site_info=pre["source_pdf"]
    for r in prior:
        q=dict(r)
        q["patient_level_stanford_provenance"]="VERIFIED_FROM_LOCAL_SUPPLEMENTARY_TABLE1"
        q["site_provenance_pdf"]=site_info["filename"]
        q["site_provenance_pdf_sha256"]=site_info["sha256"]
        q["site_provenance_locator"]=site_info["location"]
        amended.append(q)
    writecsv(out/"B2_STANFORD_STRUCTURAL_REGISTRY_AMENDED.csv",amended)

    sys.path.insert(0,str(b1dir))
    import main_execution as b1
    _,channel_map=b1.read_channel_registry(channelpath)
    outer,inner,raw=b1.make_inner_zip(wrapper)
    summaries=[];audit=[]
    try:
        members=inner.namelist()
        expected=[r["archive_member_path"] for r in registry]
        if any(members.count(x)!=1 for x in expected):raise RuntimeError("A frozen FCS member is absent or duplicated")
        for i,row in enumerate(registry,1):
            ar,sr=b1.decode_one_fcs(inner,row,channel_map)
            audit.append(ar);summaries.extend(sr)
            print(f"Stanford FCS files decoded: {i}/27",flush=True)
    finally:
        inner.close();raw.close();outer.close()
    if len(audit)!=27 or len(summaries)!=27*8:raise RuntimeError("Stanford decode coverage is incomplete")
    if sha(wrapper)!=frozen_wrapper:raise RuntimeError("Source wrapper changed during test decoding")
    fields=["patient_id","condition","target_or_baseline","channel","fcs_member_path",
        "channel_parameter_index","event_count","positive_count_ge10","positive_fraction"]
    writecsv(out/"B2_STANFORD_PATIENT_CONDITION_CHANNEL_SUMMARY.csv",summaries,fields)
    writecsv(out/"B2_STANFORD_EVENT_AGGREGATION_AUDIT.csv",audit)

    observed={(r["patient_id"],r["condition"],r["channel"]):float(r["positive_fraction"]) for r in summaries}
    train_stats=model["C0_target_response_means_and_scales"]
    frozen_models=model["refit_models_on_all_44_training_patients"]
    shared_channel=model["selection"]["SHARED"]["selected_channel"]
    exact_channels={t:model["selection"]["EXACT"][t]["selected_channel"] for t in TARGETS}
    prediction_rows=[];loss_rows=[]
    for patient in PATIENTS:
        x_test={ch:observed[(patient,"Basal",ch)] for ch in CHANNELS}
        for target in TARGETS:
            y=np.asarray([observed[(patient,target,ch)] for ch in CHANNELS],dtype=np.float64)
            stats=train_stats[target]
            ym=np.asarray(stats["mean"],dtype=np.float64);ys=np.asarray(stats["sample_sd"],dtype=np.float64)
            if not np.isfinite(y).all() or not np.isfinite(ym).all() or not np.isfinite(ys).all():
                raise RuntimeError("Non-finite test or frozen training values")
            yz=(y-ym)/ys
            preds={}
            for policy in ("C0","SHARED","EXACT"):
                if policy=="C0":predz=np.zeros(8,dtype=np.float64);predraw=ym.copy()
                else:
                    mm=frozen_models[policy][target]
                    if mm["selected_baseline_channel"]!=(shared_channel if policy=="SHARED" else exact_channels[target]):
                        raise RuntimeError("Model registry selection does not match frozen channel decision")
                    predz,predraw=model_predict(mm,x_test[mm["selected_baseline_channel"]])
                preds[policy]=(predz,predraw)
            losses={policy:float(np.mean((yz-preds[policy][0])**2)) for policy in ("C0","SHARED","EXACT")}
            loss_rows.append({"patient_id":patient,"target":target,"loss_C0":losses["C0"],
                "loss_SHARED":losses["SHARED"],"loss_EXACT":losses["EXACT"],
                "selected_shared_channel":shared_channel,"selected_exact_channel":exact_channels[target]})
            for ci,ch in enumerate(CHANNELS):
                prediction_rows.append({"patient_id":patient,"target":target,"response_channel":ch,
                    "observed_positive_fraction":y[ci],"observed_standardized_response":yz[ci],
                    "C0_predicted_positive_fraction":preds["C0"][1][ci],"C0_predicted_standardized":preds["C0"][0][ci],
                    "SHARED_predicted_positive_fraction":preds["SHARED"][1][ci],"SHARED_predicted_standardized":preds["SHARED"][0][ci],
                    "EXACT_predicted_positive_fraction":preds["EXACT"][1][ci],"EXACT_predicted_standardized":preds["EXACT"][0][ci],
                    "selected_shared_channel":shared_channel,"selected_exact_channel":exact_channels[target]})
    if len(loss_rows)!=18 or len(prediction_rows)!=18*8:raise RuntimeError("Test loss/prediction row count mismatch")
    writecsv(out/"B2_STANFORD_TEST_PREDICTIONS.csv",prediction_rows)
    writecsv(out/"B2_STANFORD_PATIENT_TARGET_LOSSES.csv",loss_rows)

    policy_names=["C0","SHARED","EXACT"]
    risks={k:float(np.mean([r["loss_"+k] for r in loss_rows],dtype=np.float64)) for k in policy_names}
    risk_rows=[{"policy":k,"risk":risks[k],"patients":9,"targets":2,"patient_target_cells":18,
        "weighting":"equal patient, equal target"} for k in policy_names]
    writecsv(out/"B2_STANFORD_PRIMARY_RISK_SUMMARY.csv",risk_rows)
    contrast_names=["C0_minus_SHARED","C0_minus_EXACT","SHARED_minus_EXACT"]
    patient_contrast={}
    for p in PATIENTS:
        prs=[r for r in loss_rows if r["patient_id"]==p]
        patient_contrast[p]={
            "C0_minus_SHARED":float(np.mean([r["loss_C0"]-r["loss_SHARED"] for r in prs])),
            "C0_minus_EXACT":float(np.mean([r["loss_C0"]-r["loss_EXACT"] for r in prs])),
            "SHARED_minus_EXACT":float(np.mean([r["loss_SHARED"]-r["loss_EXACT"] for r in prs]))}
    contrasts={name:risks[{"C0_minus_SHARED": "C0","C0_minus_EXACT":"C0","SHARED_minus_EXACT":"SHARED"}[name]]-
        risks[{"C0_minus_SHARED":"SHARED","C0_minus_EXACT":"EXACT","SHARED_minus_EXACT":"EXACT"}[name]] for name in contrast_names}
    contrast_rows=[{"contrast":n,"risk_difference":contrasts[n],"sign_convention":"first policy risk minus second policy risk"} for n in contrast_names]
    writecsv(out/"B2_STANFORD_PRIMARY_CONTRASTS.csv",contrast_rows)
    patient_rows=[{"patient_id":p,**patient_contrast[p]} for p in PATIENTS]
    writecsv(out/"B2_STANFORD_PATIENT_LEVEL_CONTRASTS.csv",patient_rows)

    target_rows=[]
    for t in TARGETS:
        rts=[r for r in loss_rows if r["target"]==t]
        rr={k:float(np.mean([r["loss_"+k] for r in rts])) for k in policy_names}
        target_rows.append({"target":t,"risk_C0":rr["C0"],"risk_SHARED":rr["SHARED"],"risk_EXACT":rr["EXACT"],
            "C0_minus_SHARED":rr["C0"]-rr["SHARED"],"C0_minus_EXACT":rr["C0"]-rr["EXACT"],
            "SHARED_minus_EXACT":rr["SHARED"]-rr["EXACT"],"patients":9})
    writecsv(out/"B2_STANFORD_TARGET_LEVEL_SUMMARY.csv",target_rows)

    rng=np.random.default_rng(20261006)
    patient_values=np.asarray([[patient_contrast[p][n] for n in contrast_names] for p in PATIENTS],dtype=np.float64)
    draws=rng.integers(0,9,size=(50000,9),endpoint=False)
    bootstrap=[]
    for ci,name in enumerate(contrast_names):
        means=np.mean(patient_values[draws,ci],axis=1,dtype=np.float64)
        lo,hi=np.quantile(means,[.025,.975],method="linear")
        bootstrap.append({"contrast":name,"estimate":contrasts[name],"percentile_95_low":float(lo),
            "percentile_95_high":float(hi),"replicates":50000,"seed":20261006,
            "label":"DESCRIPTIVE_TEST_COHORT_BOOTSTRAP","resampling_unit":"whole patient; both targets retained"})
    writecsv(out/"B2_STANFORD_BOOTSTRAP.csv",bootstrap)

    loo=[]
    for omitted in PATIENTS:
        keep=[r for r in loss_rows if r["patient_id"]!=omitted]
        rr={k:float(np.mean([r["loss_"+k] for r in keep])) for k in policy_names}
        cc={"C0_minus_SHARED":rr["C0"]-rr["SHARED"],"C0_minus_EXACT":rr["C0"]-rr["EXACT"],
            "SHARED_minus_EXACT":rr["SHARED"]-rr["EXACT"]}
        row={"omitted_patient":omitted,"risk_C0":rr["C0"],"risk_SHARED":rr["SHARED"],"risk_EXACT":rr["EXACT"]}
        for n in contrast_names:
            row[n]=cc[n];row[n+"_preserves_full_direction"]=(cc[n]>0 and contrasts[n]>0) or (cc[n]<0 and contrasts[n]<0) or (cc[n]==0 and contrasts[n]==0)
        loo.append(row)
    writecsv(out/"B2_STANFORD_LEAVE_ONE_PATIENT.csv",loo)

    if contrasts["C0_minus_SHARED"]>0 and contrasts["SHARED_minus_EXACT"]<0:case="A"
    elif contrasts["C0_minus_SHARED"]>0 and contrasts["SHARED_minus_EXACT"]>=0:case="B"
    elif contrasts["C0_minus_SHARED"]<=0 and contrasts["C0_minus_EXACT"]<=0:case="C"
    else:case="D"
    direction={n:sum(bool(r[n+"_preserves_full_direction"]) for r in loo) for n in contrast_names}
    status={"gate":"B2_STANFORD_TRANSPORT_EXECUTION_RESULTS_READY_FOR_INDEPENDENT_REPLAY",
        "site_provenance_verified":True,"site_amendment_sha256":lock["site_amendment_sha256"],
        "training_model_registry_sha256":model_hash,"test_event_decode_status":"PASS",
        "patients":9,"conditions":CONDITIONS,"FCS_files_decoded":len(audit),
        "test_patient_target_cells":len(loss_rows),"predictions":len(prediction_rows),
        "source_wrapper_sha256_before_decode":frozen_wrapper,"source_wrapper_sha256_after_decode":sha(wrapper),
        "event_summary_definition":"count(value >= 10) / total events; final eight channels; no arcsinh before threshold",
        "training_only_channels":{"SHARED":shared_channel,"EXACT":exact_channels},
        "risks":risks,"contrasts":contrasts,"bootstrap_replicates":50000,"leave_one_patient_direction_preservation":direction,
        "interpretation_case":case,"no_test_fit_or_reselection":True}
    if status["source_wrapper_sha256_before_decode"]!=status["source_wrapper_sha256_after_decode"]:
        raise RuntimeError("Source wrapper hash changed over Stanford decode")
    (out/"B2_STANFORD_EXECUTION_STATUS.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"B2_STANFORD_GATE_REPORT.md").write_text(f"""# B2 Stanford transport gate report

## Execution status

Frozen 44-patient B1 models were verified before event decoding. Site provenance was verified from the local Supplementary Table 1 footnote and hashed before the test decode.

- Stanford patients: 9/9 (UPN90–UPN98).
- Test conditions: Basal, Dasatinib, IL-7.
- FCS files decoded: 27/27.
- Patient-by-target test cells: 18/18.
- Summary values: 216/216 finite (27 files × 8 final channels).
- Test scaling: frozen 44-patient B1 target means and sample SDs only.
- No test-set training, feature selection, scaling, calibration, or refitting.

## Training-only selections

- SHARED: {shared_channel}, selected across all seven B1 training targets.
- EXACT: Dasatinib {exact_channels["Dasatinib"]}; IL-7 {exact_channels["IL-7"]}.

## Test results

| Policy | Equal-patient/equal-target risk |
|---|---:|
| C0 | {risks["C0"]:.12f} |
| SHARED | {risks["SHARED"]:.12f} |
| EXACT | {risks["EXACT"]:.12f} |

Paired contrasts (first risk minus second risk): C0 minus SHARED {contrasts["C0_minus_SHARED"]:.12f}; C0 minus EXACT {contrasts["C0_minus_EXACT"]:.12f}; SHARED minus EXACT {contrasts["SHARED_minus_EXACT"]:.12f}.

The whole-patient bootstrap is descriptive for this n=9 test cohort, uses seed 20261006 and 50,000 replicates, and retains both target results within each resampled patient. See B2_STANFORD_BOOTSTRAP.csv for intervals. Leave-one-patient direction stability is reported separately for each contrast in B2_STANFORD_LEAVE_ONE_PATIENT.csv.

Prespecified interpretation case: {case}. The interpretation is a transport stress test for a biologically shifted cohort, not prospective validation or independent study replication.

## Data-access boundary

Only the frozen 27 Stanford FCS files were decoded for this test. Their event values were not used in training, selection, target scaling, threshold choice, or model fitting. No cohort expansion or rescue analysis was performed.

Independent replay is required before the final pass gate.
""",encoding="utf-8")
    print(json.dumps(status,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
