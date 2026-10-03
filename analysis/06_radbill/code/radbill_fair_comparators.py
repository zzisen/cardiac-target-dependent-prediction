#!/usr/bin/env python3
"""Add fair Radbill comparators using the frozen primary analysis functions.
Requires RADBILL_primary_analysis.py and original source files in the supplied Radbill review directory.
"""
import argparse, importlib.util
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--radbill-dir',required=True); ap.add_argument('--out-dir',required=True)
    a=ap.parse_args(); root=Path(a.radbill_dir); dest=Path(a.out_dir); dest.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('rad',root/'code/RADBILL_primary_analysis.py'); rad=importlib.util.module_from_spec(spec); spec.loader.exec_module(rad)
    ds=rad.load_data(root/'RADBILL_SOURCE_DATA/HCM_pacing_study_data_Dryad.xlsx',root/'RADBILL_SOURCE_DATA/HCMpacingstudydatadryadreadme.txt')
    ps=ds['ps']; pids=sorted(ps); targs=[t[0] for t in rad.TARGS]; family={t[0]:t[4] for t in rad.TARGS}
    family_cands={'QT':['DT5_QT','DTend_QT'],'PP':['DT5_PP','DTend_PP']}; fixed={'QT':'DTend_QT','PP':'DTend_PP'}
    rows=[]; selection=[]
    for hold in pids:
        train=[i for i in pids if i!=hold]; inn=rad.inner(ds,train); fam_choice={}
        for fam,cands in family_cands.items():
            fam_t=[t for t in targs if family[t]==fam]; scores={c:float(np.mean([inn['means'][t][c] for t in fam_t])) for c in cands}
            fam_choice[fam]=min(cands,key=lambda c:scores[c]); selection.append({'heldout_participant':hold,'family':fam,'family_global_candidate':fam_choice[fam],**{f'inner_mean_{c}':scores[c] for c in cands}})
        for targ in targs:
            y=ps[hold]['target'][targ]
            if y is None: continue
            fam=family[targ]; choices={'C0':'NO_ADDED_MEASUREMENT','C1_original_global':inn['c1'],'C1_family_global':fam_choice[fam],'C1_fixed_family':fixed[fam],'C2_target_specific':inn['c2'][targ]}
            yvals=[ps[i]['target'][targ] for i in train if ps[i]['target'][targ] is not None]; ym,yd=rad.mean_sd(yvals)
            if ym is None or yd is None: continue
            for comp,cand in choices.items():
                if comp=='C0': pred=ym
                else:
                    xv=ps[hold]['cand'][cand]; m=rad.fit(ds,train,cand,targ)
                    if xv is None or m is None: continue
                    pred=rad.predict(m,xv)
                rows.append({'heldout_participant':hold,'cohort':ps[hold]['cohort'],'target':targ,'target_family':fam,'comparator':comp,'candidate':cand,'observed_native':y,'predicted_native':pred,'standardized_AE':abs(pred-y)/yd})
    out=pd.DataFrame(rows); counts=out.groupby(['heldout_participant','target'])['comparator'].nunique(); shared=set(counts[counts==5].index)
    out['shared_all5']=out.apply(lambda r:(r.heldout_participant,r.target) in shared,axis=1); sh=out[out.shared_all5].copy()
    summary=sh.groupby('comparator').standardized_AE.agg(['count','mean','median']).reset_index(); c0=float(summary.loc[summary.comparator=='C0','mean'].iloc[0])
    summary['improvement_vs_C0_percent']=[(c0-m)/c0*100 for m in summary['mean']]
    c2=float(summary.loc[summary.comparator=='C2_target_specific','mean'].iloc[0]); cf=float(summary.loc[summary.comparator=='C1_family_global','mean'].iloc[0]); cfix=float(summary.loc[summary.comparator=='C1_fixed_family','mean'].iloc[0])
    summary['C2_vs_family_global_percent']=np.nan; summary['C2_vs_fixed_family_percent']=np.nan
    summary.loc[summary.comparator=='C2_target_specific','C2_vs_family_global_percent']=(cf-c2)/cf*100
    summary.loc[summary.comparator=='C2_target_specific','C2_vs_fixed_family_percent']=(cfix-c2)/cfix*100
    wide=sh.pivot_table(index=['heldout_participant','target'],columns='comparator',values='standardized_AE',aggfunc='first').reset_index(); infl=[]
    for pid in pids:
        d=wide[wide.heldout_participant!=pid]; infl.append({'deleted_participant':pid,'n_cells':len(d),'family_global_minus_C2':float((d['C1_family_global']-d['C2_target_specific']).mean()),'fixed_family_minus_C2':float((d['C1_fixed_family']-d['C2_target_specific']).mean())})
    sub=sh.groupby(['cohort','comparator']).standardized_AE.agg(['count','mean']).reset_index()
    out.to_csv(dest/'RADBILL_FAIR_COMPARATOR_PREDICTIONS.csv',index=False); summary.to_csv(dest/'RADBILL_FAIR_COMPARATOR_SUMMARY.csv',index=False); pd.DataFrame(selection).to_csv(dest/'RADBILL_FAMILY_GLOBAL_SELECTION.csv',index=False); pd.DataFrame(infl).to_csv(dest/'RADBILL_FAIR_COMPARATOR_PARTICIPANT_INFLUENCE.csv',index=False); sub.to_csv(dest/'RADBILL_FAIR_COMPARATOR_SUBGROUP.csv',index=False)
    print(summary.to_string(index=False))
if __name__=='__main__': main()
