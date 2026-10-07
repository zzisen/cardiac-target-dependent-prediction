#!/usr/bin/env python3
"""Independent B2 replay from B1 summary, frozen model registry, and test summary."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np

CHANNELS=["pS6","p4EBP1","pSTAT5","pSYK","pPLCγ2","pAKT","pERK1/2","pIKAROS"]
ALL_TARGETS=["BCR-Crosslink","BEZ-235","Dasatinib","IL-7","Pervanadate","TSLP","Tofacitinib"]
TEST_TARGETS=["Dasatinib","IL-7"]
TEST_PATIENTS=[f"UPN{i}" for i in range(90,99)]

def readcsv(p):
    with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def writecsv(p,rows):
    with p.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def center_scale(y):
    mean=np.mean(y,axis=0,dtype=np.float64)
    sd=np.std(y,axis=0,ddof=1,dtype=np.float64)
    sd=np.where(sd==0.0,1.0,sd)
    return mean,sd
def fit(x,y):
    xm=float(np.mean(x,dtype=np.float64)); xs=float(np.std(x,ddof=1,dtype=np.float64))
    if xs==0.0:xs=1.0
    xz=(x-xm)/xs
    ym,ys=center_scale(y); yz=(y-ym)/ys
    xc=xz-float(np.mean(xz,dtype=np.float64)); yc=yz-np.mean(yz,axis=0,dtype=np.float64)
    den=float(np.dot(xc,xc))
    b=np.zeros(y.shape[1],dtype=np.float64) if den==0.0 else (xc[:,None]*yc).sum(axis=0)/den
    a=np.mean(yz,axis=0,dtype=np.float64)-b*float(np.mean(xz,dtype=np.float64))
    rb=ys*b/xs
    ra=ym+ys*a-rb*xm
    return xm,xs,ym,ys,a,b,ra,rb

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[2])
    a=ap.parse_args();root=a.project_root.resolve();out=Path(__file__).resolve().parent
    b1=root/"P2_V3_ABT"/"B1_OUTCOME_EXECUTION_LUNA14_2026-10-06"
    train_rows=readcsv(b1/"B1_PATIENT_CONDITION_CHANNEL_SUMMARY.csv")
    registry=json.loads((out/"B2_TRAINING_ONLY_MODEL_REGISTRY.json").read_text(encoding="utf-8-sig"))
    test_rows=readcsv(out/"B2_STANFORD_PATIENT_CONDITION_CHANNEL_SUMMARY.csv")
    # Core replay uses only these three frozen inputs. Execution products are read
    # only after the independent losses and aggregates have been recomputed.
    patients=registry["training_patients"]
    assert len(patients)==44 and len(set(patients))==44 and not set(patients)&set(TEST_PATIENTS)
    assert registry["B1_training_targets_all_seven"]==ALL_TARGETS
    assert registry["candidate_channels_ordered"]==CHANNELS
    b1vals={}
    for r in train_rows:
        k=(r["patient_id"],r["condition"],r["channel"])
        assert k not in b1vals
        b1vals[k]=float(r["positive_fraction"])
    assert len(train_rows)==44*8*8 and all(math.isfinite(v) for v in b1vals.values())
    b1expected={(p,c,ch) for p in patients for c in ["Basal"]+ALL_TARGETS for ch in CHANNELS}
    assert set(b1vals)==b1expected
    x=np.asarray([[b1vals[(p,"Basal",ch)] for ch in CHANNELS] for p in patients],dtype=np.float64)
    y=np.asarray([[[b1vals[(p,t,ch)] for ch in CHANNELS] for t in ALL_TARGETS] for p in patients],dtype=np.float64)
    nt=len(ALL_TARGETS);nc=len(CHANNELS);n=len(patients);ti={t:i for i,t in enumerate(ALL_TARGETS)}
    inner=np.empty((n,nc,nt),dtype=np.float64)
    for held in range(n):
        tr=np.asarray([i for i in range(n) if i!=held],dtype=int)
        for j,t in enumerate(ALL_TARGETS):
            ym,ys=center_scale(y[tr,j,:]); ytest=(y[held,j,:]-ym)/ys
            for c in range(nc):
                xm,xs,_,_,inter,b,_,_=fit(x[tr,c],y[tr,j,:])
                pred=inter+b*((float(x[held,c])-xm)/xs)
                inner[held,c,j]=float(np.mean((ytest-pred)**2))
    bytarget=np.mean(inner,axis=0,dtype=np.float64)
    shared_scores=np.mean(bytarget,axis=1,dtype=np.float64)
    shared_channel=CHANNELS[int(np.argmin(shared_scores))]
    exact={t:CHANNELS[int(np.argmin(bytarget[:,ti[t]]))] for t in TEST_TARGETS}
    assert shared_channel==registry["selection"]["SHARED"]["selected_channel"]
    assert exact=={t:registry["selection"]["EXACT"][t]["selected_channel"] for t in TEST_TARGETS}

    # Independently refit the selected policies on all 44 training patients and
    # compare the result with the immutable hashed training registry.
    for policy in ("SHARED","EXACT"):
        for t in TEST_TARGETS:
            ch=shared_channel if policy=="SHARED" else exact[t]
            st=fit(x[:,CHANNELS.index(ch)],y[:,ti[t],:])
            ref=registry["refit_models_on_all_44_training_patients"][policy][t]
            assert ch==ref["selected_baseline_channel"]
            for key,idx in [("response_mean",2),("response_sample_sd",3),
                            ("standardized_intercept",4),("standardized_slope",5),
                            ("raw_intercept",6),("raw_slope",7)]:
                assert np.allclose(np.asarray(st[idx]),np.asarray(ref[key]),rtol=0.0,atol=1e-12),(policy,t,key)
            assert np.isclose(st[0],ref["predictor_mean"],rtol=0.0,atol=1e-12)
            assert np.isclose(st[1],ref["predictor_sample_sd"],rtol=0.0,atol=1e-12)

    tv={}
    for r in test_rows:
        k=(r["patient_id"],r["condition"],r["channel"])
        assert k not in tv
        tv[k]=float(r["positive_fraction"])
    assert len(test_rows)==27*8 and set(tv)=={(p,c,ch) for p in TEST_PATIENTS for c in ["Basal"]+TEST_TARGETS for ch in CHANNELS}
    assert all(math.isfinite(v) for v in tv.values())
    stats=registry["C0_target_response_means_and_scales"]
    fitted=registry["refit_models_on_all_44_training_patients"]
    loss_rows=[]
    for p in TEST_PATIENTS:
        for t in TEST_TARGETS:
            obs=np.asarray([tv[(p,t,ch)] for ch in CHANNELS],dtype=np.float64)
            st=stats[t]; ym=np.asarray(st["mean"],dtype=np.float64);ys=np.asarray(st["sample_sd"],dtype=np.float64)
            oz=(obs-ym)/ys
            l={"C0":float(np.mean(oz**2))}
            for policy in ("SHARED","EXACT"):
                m=fitted[policy][t];base=tv[(p,"Basal",m["selected_baseline_channel"])]
                xz=(base-m["predictor_mean"])/m["predictor_sample_sd"]
                predz=np.asarray(m["standardized_intercept"])+np.asarray(m["standardized_slope"])*xz
                l[policy]=float(np.mean((oz-predz)**2))
            for policy in ("C0","SHARED","EXACT"):
                loss_rows.append({"patient_id":p,"target":t,"policy":policy,"loss":l[policy]})
    assert len(loss_rows)==54
    writecsv(out/"B2_STANFORD_REPLAY_PATIENT_TARGET_LOSSES.csv",loss_rows)
    values={(r["patient_id"],r["target"],r["policy"]):r["loss"] for r in loss_rows}
    risks={policy:float(np.mean([v for (p,t,k),v in values.items() if k==policy],dtype=np.float64)) for policy in ("C0","SHARED","EXACT")}
    risk_rows=[{"policy":k,"risk":v} for k,v in risks.items()]
    writecsv(out/"B2_STANFORD_REPLAY_PRIMARY_RISKS.csv",risk_rows)
    contrasts={"C0_minus_SHARED":risks["C0"]-risks["SHARED"],
        "C0_minus_EXACT":risks["C0"]-risks["EXACT"],
        "SHARED_minus_EXACT":risks["SHARED"]-risks["EXACT"]}
    contrast_rows=[{"contrast":k,"risk_difference":v} for k,v in contrasts.items()]
    writecsv(out/"B2_STANFORD_REPLAY_PRIMARY_CONTRASTS.csv",contrast_rows)

    # Compare only after independent computation.
    produced=readcsv(out/"B2_STANFORD_PATIENT_TARGET_LOSSES.csv")
    prod={(r["patient_id"],r["target"],policy):float(r["loss_"+policy]) for r in produced for policy in ("C0","SHARED","EXACT")}
    assert set(prod)==set(values)
    assert all(np.isclose(prod[k],values[k],rtol=0.0,atol=1e-12) for k in values)
    prod_risk={r["policy"]:float(r["risk"]) for r in readcsv(out/"B2_STANFORD_PRIMARY_RISK_SUMMARY.csv")}
    assert all(np.isclose(prod_risk[k],risks[k],rtol=0.0,atol=1e-12) for k in risks)
    prod_contrast={r["contrast"]:float(r["risk_difference"]) for r in readcsv(out/"B2_STANFORD_PRIMARY_CONTRASTS.csv")}
    assert all(np.isclose(prod_contrast[k],contrasts[k],rtol=0.0,atol=1e-12) for k in contrasts)
    result={"replay":"PASS","inputs":["frozen B1 compact summary","frozen B2 training-only model registry","B2 Stanford compact summary"],
        "training_selection_recomputed":True,"training_refit_coefficients_recomputed":True,
        "patient_target_policy_losses_reproduced":54,"risks_reproduced":3,"primary_contrasts_reproduced":3,
        "max_absolute_loss_difference":float(max(abs(prod[k]-values[k]) for k in values)),
        "max_absolute_risk_difference":float(max(abs(prod_risk[k]-risks[k]) for k in risks)),
        "max_absolute_contrast_difference":float(max(abs(prod_contrast[k]-contrasts[k]) for k in contrasts)),
        "independent_risks":risks,"independent_contrasts":contrasts,
        "training_channels":{"SHARED":shared_channel,"EXACT":exact},
        "event_data_reread_by_replay":False}
    (out/"B2_STANFORD_INDEPENDENT_REPLAY_RESULT.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"B2_STANFORD_INDEPENDENT_REPLAY_REPORT.md").write_text(f"""# B2 independent outcome replay

Replay status: PASS.

The replay started from only the frozen B1 compact summary, the frozen B2 training-only model registry, and the compact Stanford summary. It independently repeated the 44-patient inner LOPO channel selection and full-cohort refits, then regenerated all 54 patient-target-policy losses (18 cells × 3 policies), all 3 aggregate risks, and all 3 primary contrasts.

The independent calculations agree with the execution outputs within an absolute tolerance of 1e-12. Maximum absolute differences: loss {result["max_absolute_loss_difference"]:.3g}; risk {result["max_absolute_risk_difference"]:.3g}; contrast {result["max_absolute_contrast_difference"]:.3g}.

The replay did not reopen or decode any FCS file.
""",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
