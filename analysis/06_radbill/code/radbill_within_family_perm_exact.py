#!/usr/bin/env python3
"""Exact within-family target-label permutation using the original Radbill common-support semantics."""
import argparse, importlib.util, itertools
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--radbill-dir',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); root=Path(a.radbill_dir)
    spec=importlib.util.spec_from_file_location('rad',root/'code/RADBILL_primary_analysis.py'); rad=importlib.util.module_from_spec(spec); spec.loader.exec_module(rad)
    ds=rad.load_data(root/'RADBILL_SOURCE_DATA/HCM_pacing_study_data_Dryad.xlsx',root/'RADBILL_SOURCE_DATA/HCMpacingstudydatadryadreadme.txt')
    ps=ds['ps']; ids=sorted(ps); ckeys=[c[0] for c in rad.CANDS]; tkeys=[t[0] for t in rad.TARGS]
    sel_list=[]; obs=np.full((len(ids),len(tkeys),len(ckeys)),np.nan); all4_mask=np.zeros((len(ids),len(tkeys)),dtype=bool)
    for oi,hold in enumerate(ids):
        train=[i for i in ids if i!=hold]; sel=rad.inner(ds,train); sel_list.append(sel)
        for ti,(targ,*_) in enumerate(rad.TARGS):
            models={c:rad.fit(ds,train,c,targ) for c in ckeys}; hp=ps[hold]; y=hp['target'][targ]
            cpred={c:rad.predict(models[c],hp['cand'][c]) if y is not None else None for c in ckeys}; ytrain=[ps[i]['target'][targ] for i in train if ps[i]['target'][targ] is not None]; _,yd=rad.mean_sd(ytrain)
            for ci,c in enumerate(ckeys):
                if y is not None and cpred[c] is not None: obs[oi,ti,ci]=abs(cpred[c]-y)/yd
            all4_mask[oi,ti]=bool(y is not None and all(hp['cand'][c] is not None and cpred[c] is not None for c in ckeys))
    smat=np.array([[ckeys.index(s['c2'][t]) for t in tkeys] for s in sel_list]); identity=float(np.mean(np.take_along_axis(obs,smat[:,:,None],axis=2)[:,:,0][all4_mask]))
    qt=[i for i,t in enumerate(rad.TARGS) if t[4]=='QT']; pp=[i for i,t in enumerate(rad.TARGS) if t[4]=='PP']; vals=[]; nle=0
    for pqt in itertools.permutations(qt):
        for ppp in itertools.permutations(pp):
            perm=list(range(len(tkeys)))
            for j,k in zip(qt,pqt): perm[j]=k
            for j,k in zip(pp,ppp): perm[j]=k
            mapped=smat[:,perm]; scores=np.take_along_axis(obs,mapped[:,:,None],axis=2)[:,:,0]; val=float(np.mean(scores[all4_mask])); vals.append(val); nle += int(val <= identity+1e-12)
    row={'identity_MAE':identity,'n_permutations':len(vals),'n_no_worse':nle,'p_fraction':nle/len(vals),'best_MAE':min(vals)}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame([row]).to_csv(a.out,index=False); print(row)
if __name__=='__main__': main()
