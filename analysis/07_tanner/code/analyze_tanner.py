from __future__ import annotations
import argparse, csv, hashlib, itertools, json, math, platform, sys
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
CAN=ROOT/'input/TANNER_CANONICAL_SUBJECT_LEVEL_DATA.csv'
FREEZE=ROOT/'01_TANNER_SCIENTIFIC_FREEZE.json'
FREEZE_SHA=ROOT/'01_TANNER_SCIENTIFIC_FREEZE.sha256'
OUT=ROOT/'results'
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
CAND_RATES=[100,250]
TARGET_RATES=[25,100,250,1000]
CAND_TIME=10.0; TARGET_TIME=100.0
BASE_SEED=20261003
OUTCOME_COLUMNS={
 'peak_stress_response':'peak_stress_response_mN_per_mm2',
 'minimum_stress_response':'minimum_stress_response_mN_per_mm2',
 't12_s':'t12_s','t2_s':'t2_s','t23_s':'t23_s'
}

def write_df(path, rows):
    pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig',float_format='%.15g')

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def read_data():
    actual=sha256(FREEZE)
    expected=FREEZE_SHA.read_text(encoding='ascii').split()[0]
    if actual!=expected: raise RuntimeError(f'Frozen protocol hash mismatch: {actual} != {expected}')
    freeze=json.loads(FREEZE.read_text(encoding='utf-8'))
    d=pd.read_csv(CAN)
    d['trabecula_id_as_listed']=d['trabecula_id_as_listed'].astype(int)
    d['time_to_stretch_ms']=d['time_to_stretch_ms'].astype(float)
    d['strain_rate_s_inv']=d['strain_rate_s_inv'].astype(int)
    if d.duplicated(['trabecula_id_as_listed','time_to_stretch_ms','strain_rate_s_inv']).any():
        raise RuntimeError('Duplicate subject-condition keys in canonical data')
    return d,freeze

def prep(d, endpoint, requested_ids=None):
    col=OUTCOME_COLUMNS[endpoint]
    ids=sorted(set(d['trabecula_id_as_listed'])) if requested_ids is None else sorted(requested_ids)
    table={(int(r.trabecula_id_as_listed),float(r.time_to_stretch_ms),int(r.strain_rate_s_inv)):
           (None if pd.isna(getattr(r,col)) else float(getattr(r,col))) for r in d.itertuples(index=False)}
    xx={c:{} for c in CAND_RATES}; yy={t:{} for t in TARGET_RATES}; missing=[]
    for sid in ids:
        for c in CAND_RATES:
            v=table.get((sid,CAND_TIME,c))
            if v is None: missing.append((sid,CAND_TIME,c,endpoint))
            else: xx[c][sid]=v
        for t in TARGET_RATES:
            v=table.get((sid,TARGET_TIME,t))
            if v is None: missing.append((sid,TARGET_TIME,t,endpoint))
            else: yy[t][sid]=v
    eligible=[sid for sid in ids if all(sid in xx[c] for c in CAND_RATES) and all(sid in yy[t] for t in TARGET_RATES)]
    if missing: eligible=[sid for sid in eligible]
    return eligible,xx,yy,missing

def affine_predict(x,y,xstar):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    xm=float(np.mean(x)); ym=float(np.mean(y)); den=float(np.sum((x-xm)**2))
    slope=0.0 if den<=1e-14 else float(np.sum((x-xm)*(y-ym))/den)
    intercept=ym-slope*xm
    return slope*float(xstar)+intercept,slope,intercept

def inner_score(train_ids,xx,yy):
    scores=np.zeros((len(CAND_RATES),len(TARGET_RATES)),dtype=float)
    for ci,c in enumerate(CAND_RATES):
        for ti,t in enumerate(TARGET_RATES):
            errs=[]
            for valid in train_ids:
                inner_train=[s for s in train_ids if s!=valid]
                assert valid not in inner_train
                pred,_,_=affine_predict([xx[c][s] for s in inner_train],[yy[t][s] for s in inner_train],xx[c][valid])
                errs.append(abs(pred-yy[t][valid]))
            scores[ci,ti]=float(np.mean(errs))
    return scores

def fold_prediction(train_ids,test_id,candidate,target,xx,yy):
    assert test_id not in train_ids
    return affine_predict([xx[candidate][s] for s in train_ids],[yy[target][s] for s in train_ids],xx[candidate][test_id])

def pred_row(endpoint,comp,sid,fold,target,pred,obs,candidate,train_ids,slope,intercept,inner,selection_status='TRAINING_ONLY'):
    err=float(pred-obs)
    return {'outcome_family':endpoint,'comparator':comp,'outer_heldout_subject':int(sid),'outer_fold':int(fold),
      'target_time_to_stretch_ms':TARGET_TIME,'target_strain_rate_s_inv':int(target),'observed_target':float(obs),
      'predicted_target':float(pred),'signed_error_prediction_minus_observed':err,'absolute_error':abs(err),
      'squared_error':err*err,'selected_candidate_time_ms':CAND_TIME if candidate is not None else '',
      'selected_candidate_strain_rate_s_inv':candidate if candidate is not None else '',
      'candidate_condition':(f'{CAND_TIME:g}ms/{candidate}s^-1' if candidate is not None else 'none'),
      'target_condition':f'{TARGET_TIME:g}ms/{target}s^-1','training_subject_ids_json':json.dumps(list(map(int,train_ids))),
      'n_training_subjects':len(train_ids),'inner_candidate_target_mae_json':json.dumps(inner.tolist(),separators=(',',':')) if inner is not None else '',
      'fit_slope':slope if slope is not None else '', 'fit_intercept':intercept if intercept is not None else '',
      'model_status':'OK','selection_status':selection_status}

def metric_rows(rows):
    es=np.array([float(r['absolute_error']) for r in rows],dtype=float)
    return {'n_predictions':len(es),'mae':float(np.mean(es)),'rmse':float(np.sqrt(np.mean(es**2))),'median_absolute_error':float(np.median(es))}

def run_outer(endpoint,ids,xx,yy,include_f1=False,include_sensitivity=False):
    ids=sorted(ids); standard=[]; candidate_preds=[]; selections=[]; inner_by_fold={}; choices_by_fold={}; c3=[]
    policies=list(itertools.product(range(len(CAND_RATES)),repeat=len(TARGET_RATES)))
    for fold,test in enumerate(ids,1):
        train=[s for s in ids if s!=test]
        assert test not in train and len(train)==len(ids)-1
        inner=inner_score(train,xx,yy)
        target_choices={t:CAND_RATES[int(np.argmin(inner[:,ti]))] for ti,t in enumerate(TARGET_RATES)}
        global_choice=CAND_RATES[int(np.argmin(np.mean(inner,axis=1)))]
        inner_by_fold[test]=inner; choices_by_fold[test]=target_choices
        for ti,target in enumerate(TARGET_RATES):
            chosen=target_choices[target]
            nearest=min(CAND_RATES,key=lambda c:(abs(math.log10(target)-math.log10(c)),c))
            obs=yy[target][test]
            p,b,a=fold_prediction(train,test,chosen,target,xx,yy)
            standard.append(pred_row(endpoint,'C2_TARGET_SPECIFIC',test,fold,target,p,obs,chosen,train,b,a,inner))
            gp,gb,ga=fold_prediction(train,test,global_choice,target,xx,yy)
            standard.append(pred_row(endpoint,'C1_GLOBAL_FIXED',test,fold,target,gp,obs,global_choice,train,gb,ga,inner))
            yp=float(np.mean([yy[target][s] for s in train]))
            standard.append(pred_row(endpoint,'C0_TARGET_MEAN',test,fold,target,yp,obs,None,train,None,None,inner))
            npred,nb,na=fold_prediction(train,test,nearest,target,xx,yy)
            standard.append(pred_row(endpoint,'C4_NEAREST_LOG_RATE',test,fold,target,npred,obs,nearest,train,nb,na,inner,'MECHANICAL_HEURISTIC'))
            selections.append({'outcome_family':endpoint,'outer_heldout_subject':test,'outer_fold':fold,'target_strain_rate_s_inv':target,
              'candidate_100_inner_mae':inner[0,ti],'candidate_250_inner_mae':inner[1,ti],'c2_selected_candidate_rate_s_inv':chosen,
              'c1_global_candidate_rate_s_inv':global_choice,'heldout_c2_prediction':p,'heldout_observed_target':obs,
              'heldout_c2_absolute_error':abs(p-obs),'outer_subject_excluded_from_selection':True,
              'inner_training_subject_ids_json':json.dumps(train)})
            for ci,c in enumerate(CAND_RATES):
                cp,cb,ca=fold_prediction(train,test,c,target,xx,yy)
                candidate_preds.append(pred_row(endpoint,'CANDIDATE_FIXED_FOR_MATRIX',test,fold,target,cp,obs,c,train,cb,ca,inner,'NO_SELECTION_FIXED_CANDIDATE'))
        # C3: exact full enumeration of one candidate assignment per target.
        for pi,policy in enumerate(policies):
            for ti,target in enumerate(TARGET_RATES):
                c=CAND_RATES[policy[ti]]
                rp,rb,ra=fold_prediction(train,test,c,target,xx,yy)
                row=pred_row(endpoint,'C3_RANDOM_POLICY',test,fold,target,rp,yy[target][test],c,train,rb,ra,inner,'EXACT_POLICY_ENUMERATION')
                row['random_policy_id']=f'P{pi:02d}'
                row['random_policy_assignment_json']=json.dumps({str(t):CAND_RATES[policy[j]] for j,t in enumerate(TARGET_RATES)},sort_keys=True)
                c3.append(row)
    mat=[]
    for target in TARGET_RATES:
        for c in CAND_RATES:
            rr=[r for r in candidate_preds if r['target_strain_rate_s_inv']==target and r['selected_candidate_strain_rate_s_inv']==c]
            mm=metric_rows(rr)
            mat.append({'outcome_family':endpoint,'candidate_time_ms':CAND_TIME,'candidate_strain_rate_s_inv':c,'target_time_ms':TARGET_TIME,
              'target_strain_rate_s_inv':target,**mm})
    # The all-data oracle selects using all outer errors, then reuses those errors: explicitly optimistic.
    winners={}
    for target in TARGET_RATES:
        vals={c:next(r['mae'] for r in mat if r['target_strain_rate_s_inv']==target and r['candidate_strain_rate_s_inv']==c) for c in CAND_RATES}
        winners[target]=min(CAND_RATES,key=lambda c:(vals[c],c))
        for r in candidate_preds:
            if r['target_strain_rate_s_inv']==target and r['selected_candidate_strain_rate_s_inv']==winners[target]:
                q=dict(r); q['comparator']='C5_ORACLE_UPPER_BOUND'; q['selection_status']='LEAKAGE_BY_ALL_SUBJECT_SELECTION_UPPER_BOUND'; standard.append(q)
    f1=[]
    if include_f1:
        for perm_idx,perm in enumerate(itertools.permutations(range(len(TARGET_RATES)))):
            permrows=[]
            for fold,test in enumerate(ids,1):
                train=[s for s in ids if s!=test]
                for ti,target in enumerate(TARGET_RATES):
                    selector_target=TARGET_RATES[perm[ti]]
                    chosen=choices_by_fold[test][selector_target]
                    pp,bb,aa=fold_prediction(train,test,chosen,target,xx,yy)
                    permrows.append(pred_row(endpoint,'F1_TARGET_LABEL_PERMUTATION',test,fold,target,pp,yy[target][test],chosen,train,bb,aa,inner_by_fold[test],
                                             'IDENTITY' if perm==tuple(range(4)) else 'PERMUTED_SELECTOR_LABEL'))
            overall=metric_rows(permrows)
            f1.append({'outcome_family':endpoint,'permutation_id':f'F1P{perm_idx:02d}','selector_target_mapping_json':json.dumps({str(TARGET_RATES[i]):TARGET_RATES[perm[i]] for i in range(4)},sort_keys=True),
                       'is_identity_mapping':perm==tuple(range(4)),'target_strain_rate_s_inv':'ALL',**overall})
            for t in TARGET_RATES:
                sub=[r for r in permrows if r['target_strain_rate_s_inv']==t]
                f1.append({'outcome_family':endpoint,'permutation_id':f'F1P{perm_idx:02d}','selector_target_mapping_json':json.dumps({str(TARGET_RATES[i]):TARGET_RATES[perm[i]] for i in range(4)},sort_keys=True),
                           'is_identity_mapping':perm==tuple(range(4)),'target_strain_rate_s_inv':t,**metric_rows(sub)})
    sens=[]
    if include_sensitivity:
        for omitted in ids:
            subids=[s for s in ids if s!=omitted]
            e2,x2,y2,miss=prep_endpoint_dict(xx,yy,subids)
            # Outer LOTO is rerun over the six remaining trabeculae.
            mini,_,_,_,_,_,_,_=run_outer(endpoint,subids,x2,y2,include_f1=False,include_sensitivity=False)
            for comp in ['C0_TARGET_MEAN','C1_GLOBAL_FIXED','C2_TARGET_SPECIFIC']:
                rr=[r for r in mini if r['comparator']==comp]
                sens.append({'outcome_family':endpoint,'omitted_subject':omitted,'n_subjects':len(subids),'comparator':comp,**metric_rows(rr),
                             'mean_inner_selection_training_n':len(subids)-2})
            e2rows=[r for r in mini if r['comparator']=='C2_TARGET_SPECIFIC']
            e1rows=[r for r in mini if r['comparator']=='C1_GLOBAL_FIXED']
            e0rows=[r for r in mini if r['comparator']=='C0_TARGET_MEAN']
            sens.append({'outcome_family':endpoint,'omitted_subject':omitted,'n_subjects':len(subids),'comparator':'DELTA_C2_MINUS_C1',
                         'n_predictions':len(e2rows),'mae':metric_rows(e2rows)['mae']-metric_rows(e1rows)['mae'],'rmse':'','median_absolute_error':'','mean_inner_selection_training_n':len(subids)-2})
            sens.append({'outcome_family':endpoint,'omitted_subject':omitted,'n_subjects':len(subids),'comparator':'DELTA_C2_MINUS_C0',
                         'n_predictions':len(e2rows),'mae':metric_rows(e2rows)['mae']-metric_rows(e0rows)['mae'],'rmse':'','median_absolute_error':'','mean_inner_selection_training_n':len(subids)-2})
    return standard,candidate_preds,mat,selections,c3,f1,sens,winners

def prep_endpoint_dict(xx,yy,ids):
    eids=[s for s in ids if all(s in xx[c] for c in CAND_RATES) and all(s in yy[t] for t in TARGET_RATES)]
    return eids,xx,yy,[]

def summary_table(endpoint,standard,c3,mat,winners):
    rows=[]
    basecomps=['C0_TARGET_MEAN','C1_GLOBAL_FIXED','C2_TARGET_SPECIFIC','C4_NEAREST_LOG_RATE','C5_ORACLE_UPPER_BOUND']
    for comp in basecomps:
        rr=[r for r in standard if r['comparator']==comp]
        if not rr: continue
        for t in TARGET_RATES:
            rt=[r for r in rr if r['target_strain_rate_s_inv']==t]
            rows.append({'outcome_family':endpoint,'comparator':comp,'target_strain_rate_s_inv':t,**metric_rows(rt),
                         'oracle_warning':'selection uses all outer outcomes; optimistic only' if comp.startswith('C5') else ''})
        rows.append({'outcome_family':endpoint,'comparator':comp,'target_strain_rate_s_inv':'ALL',**metric_rows(rr),
                     'oracle_warning':'selection uses all outer outcomes; optimistic only' if comp.startswith('C5') else ''})
    # C3: summarize 16 policy-level means, equally weighted; intervals show policy variability, not sampling CIs.
    policy_rows=[]
    for policy in sorted({r['random_policy_id'] for r in c3}):
        pr=[r for r in c3 if r['random_policy_id']==policy]
        policy_rows.append((policy,pr))
    n_subjects=len({int(r['outer_heldout_subject']) for r in standard if r['comparator']=='C0_TARGET_MEAN'})
    for t in TARGET_RATES+['ALL']:
        vals=[]
        for policy,pr in policy_rows:
            q=pr if t=='ALL' else [r for r in pr if r['target_strain_rate_s_inv']==t]
            vals.append(metric_rows(q)['mae'])
        rows.append({'outcome_family':endpoint,'comparator':'C3_RANDOM_EXPECTATION_16_POLICIES','target_strain_rate_s_inv':t,
                     'n_predictions':len(policy_rows)*n_subjects*(len(TARGET_RATES) if t=='ALL' else 1),
                     'mae':float(np.mean(vals)),'rmse':'','median_absolute_error':'','policy_mae_min':float(np.min(vals)),
                     'policy_mae_median':float(np.median(vals)),'policy_mae_max':float(np.max(vals)),
                     'policy_mae_q025':float(np.quantile(vals,.025)),'policy_mae_q975':float(np.quantile(vals,.975)),
                     'oracle_warning':'Policy spread is exact finite enumeration; policies are not independent biological units.'})
    return rows,policy_rows

def subject_summaries(endpoint,standard,c3):
    rows=[]
    comps=['C0_TARGET_MEAN','C1_GLOBAL_FIXED','C2_TARGET_SPECIFIC','C4_NEAREST_LOG_RATE','C5_ORACLE_UPPER_BOUND']
    for sid in sorted({int(r['outer_heldout_subject']) for r in standard}):
        for comp in comps:
            rr=[r for r in standard if int(r['outer_heldout_subject'])==sid and r['comparator']==comp]
            if not rr: continue
            rows.append({'outcome_family':endpoint,'subject_id':sid,'comparator':comp,**metric_rows(rr),
                         'mean_signed_error':float(np.mean([r['signed_error_prediction_minus_observed'] for r in rr]))})
        rr=[r for r in c3 if int(r['outer_heldout_subject'])==sid]
        by_target=[]
        for t in TARGET_RATES:
            errs=[r['absolute_error'] for r in rr if r['target_strain_rate_s_inv']==t]
            by_target.extend(errs)
        # Mean absolute error over exact policy set for the subject; not a new independent observation.
        rows.append({'outcome_family':endpoint,'subject_id':sid,'comparator':'C3_RANDOM_EXPECTATION_16_POLICIES',
                     'n_predictions':len(by_target),'mae':float(np.mean(by_target)),'rmse':'','median_absolute_error':float(np.median(by_target)),
                     'mean_signed_error':''})
    return rows

def bootstrap_deltas(endpoint,standard):
    ids=sorted({int(r['outer_heldout_subject']) for r in standard if r['comparator']=='C2_TARGET_SPECIFIC'})
    comps={c:{(int(r['outer_heldout_subject']),int(r['target_strain_rate_s_inv'])):float(r['absolute_error']) for r in standard if r['comparator']==c}
           for c in ['C0_TARGET_MEAN','C1_GLOBAL_FIXED','C2_TARGET_SPECIFIC']}
    out=[]; rng=np.random.Generator(np.random.PCG64(BASE_SEED))
    for base in ['C0_TARGET_MEAN','C1_GLOBAL_FIXED']:
        per_subject=[np.mean([comps['C2_TARGET_SPECIFIC'][(sid,t)]-comps[base][(sid,t)] for t in TARGET_RATES]) for sid in ids]
        obs=float(np.mean(per_subject)); sims=[]
        a=np.asarray(per_subject,dtype=float)
        for _ in range(10000): sims.append(float(np.mean(a[rng.integers(0,len(a),size=len(a))])))
        out.append({'outcome_family':endpoint,'contrast':'C2_MINUS_'+base,'n_subject_clusters':len(ids),'observed_mean_paired_mae_difference':obs,
                    'bootstrap_2_5_percentile':float(np.quantile(sims,.025)),'bootstrap_97_5_percentile':float(np.quantile(sims,.975)),
                    'resamples':10000,'seed':BASE_SEED,'interpretation':'Descriptive subject-cluster bootstrap interval; small n.'})
    return out

def rank_summaries(endpoint,mat):
    bytarget={}
    rows=[]
    for t in TARGET_RATES:
        vals={int(r['candidate_strain_rate_s_inv']):float(r['mae']) for r in mat if int(r['target_strain_rate_s_inv'])==t}
        winner=min(CAND_RATES,key=lambda c:(vals[c],c)); bytarget[t]=winner
        rows.append({'outcome_family':endpoint,'row_type':'TARGET_WINNER','target_a_s_inv':t,'target_b_s_inv':'',
                     'candidate_100_mae':vals[100],'candidate_250_mae':vals[250],'candidate_100_rank':1+int(vals[250]<vals[100]),
                     'candidate_250_rank':1+int(vals[100]<vals[250]),'best_candidate_rate_s_inv':winner,'spearman_rank_correlation':'',
                     'top1_same_candidate':'','interpretation':'Ranks from outer-LOTO predictions; descriptive across only seven subjects.'})
    for a,b in itertools.combinations(TARGET_RATES,2):
        va={int(r['candidate_strain_rate_s_inv']):float(r['mae']) for r in mat if int(r['target_strain_rate_s_inv'])==a}
        vb={int(r['candidate_strain_rate_s_inv']):float(r['mae']) for r in mat if int(r['target_strain_rate_s_inv'])==b}
        ra=np.array([1+int(va[other]<va[c]) for c,other in [(100,250),(250,100)]],dtype=float)
        rb=np.array([1+int(vb[other]<vb[c]) for c,other in [(100,250),(250,100)]],dtype=float)
        rho=float(np.corrcoef(ra,rb)[0,1]) if np.std(ra)>0 and np.std(rb)>0 else float('nan')
        rows.append({'outcome_family':endpoint,'row_type':'TARGET_PAIR_RANK_ASSOCIATION','target_a_s_inv':a,'target_b_s_inv':b,
                     'candidate_100_mae':'','candidate_250_mae':'','candidate_100_rank':'','candidate_250_rank':'','best_candidate_rate_s_inv':'',
                     'spearman_rank_correlation':rho,'top1_same_candidate':bytarget[a]==bytarget[b],
                     'interpretation':'With two candidates, rank correlation is descriptive and can only show concordance/reversal.'})
    counts={c:sum(v==c for v in bytarget.values()) for c in CAND_RATES}
    rows.append({'outcome_family':endpoint,'row_type':'RANK_REVERSAL_SUMMARY','target_a_s_inv':'ALL','target_b_s_inv':'',
                 'candidate_100_mae':'','candidate_250_mae':'','candidate_100_rank':'','candidate_250_rank':'','best_candidate_rate_s_inv':'',
                 'spearman_rank_correlation':'','top1_same_candidate':'','candidate_100_target_wins':counts[100],
                 'candidate_250_target_wins':counts[250],'interpretation':f"Candidate 100 wins {counts[100]}/4 targets; candidate 250 wins {counts[250]}/4."})
    return rows

def source_grid_figure(ids):
    fig,ax=plt.subplots(figsize=(8.2,3.6))
    ax.set_xlim(20,1200); ax.set_ylim(-.8,len(ids)-.2)
    for yi,sid in enumerate(ids):
        ax.scatter([100,250],[yi+0.12,yi+0.12],marker='o',s=55,color='#1f77b4',zorder=3)
        ax.scatter([25,100,250,1000],[yi-0.12,yi-0.12,yi-0.12,yi-0.12],marker='s',s=45,facecolors='none',edgecolors='#c44e52',linewidths=1.5,zorder=2)
    ax.scatter([],[],marker='o',s=55,color='#1f77b4',label='Candidate at 10 ms')
    ax.scatter([],[],marker='s',s=45,facecolors='none',edgecolors='#c44e52',label='Target at 100 ms')
    ax.set_xscale('log'); ax.set_xticks([25,100,250,1000],labels=['25','100','250','1000'])
    ax.set_yticks(range(len(ids)),[str(x) for x in ids]); ax.invert_yaxis()
    ax.set_xlabel('Strain rate (s$^{-1}$)'); ax.set_ylabel('Trabecula ID')
    ax.set_title('Frozen complete repeated-measures grid (n = 7)')
    ax.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.23))
    fig.tight_layout(); fig.savefig(FIG/'TANNER_FIG1_SOURCE_GRID.png',dpi=220); fig.savefig(FIG/'TANNER_FIG1_SOURCE_GRID.svg'); plt.close(fig)

def make_figures(mat,standard,f1rows,ids):
    source_grid_figure(ids)
    rates=['25','100','250','1000']; cs=[100,250]
    z=np.array([[next(r['mae'] for r in mat if int(r['candidate_strain_rate_s_inv'])==c and int(r['target_strain_rate_s_inv'])==int(t)) for t in rates] for c in cs])
    fig,ax=plt.subplots(figsize=(7,3.7)); im=ax.imshow(z,aspect='auto',cmap='viridis')
    ax.set_xticks(range(4),rates); ax.set_yticks(range(2),['100 s$^{-1}$','250 s$^{-1}$'])
    ax.set_xlabel('Future target strain rate at 100 ms (s$^{-1}$)'); ax.set_ylabel('Candidate strain rate at 10 ms')
    ax.set_title('Outer LOTO MAE by fixed candidate and future target')
    for i in range(2):
        for j in range(4): ax.text(j,i,f'{z[i,j]:.3g}',ha='center',va='center',color='white' if z[i,j]<(z.max()+z.min())/2 else 'black')
    fig.colorbar(im,ax=ax,label='MAE (outcome units)'); fig.tight_layout(); fig.savefig(FIG/'TANNER_FIG2_CANDIDATE_TARGET_MAE.png',dpi=220); fig.savefig(FIG/'TANNER_FIG2_CANDIDATE_TARGET_MAE.svg'); plt.close(fig)
    comps=['C0_TARGET_MEAN','C1_GLOBAL_FIXED','C2_TARGET_SPECIFIC','C4_NEAREST_LOG_RATE']
    means=[metric_rows([r for r in standard if r['comparator']==c])['mae'] for c in comps]
    fig,ax=plt.subplots(figsize=(7,4)); labels=['Target mean','Global fixed','Target-specific','Nearest rate']
    bars=ax.bar(labels,means,color=['#999999','#4c72b0','#55a868','#c44e52'])
    ax.set_ylabel('Outer LOTO MAE (outcome units)'); ax.set_title('Held-out error by comparator (n = 7 subjects)'); ax.tick_params(axis='x',rotation=15)
    for bar,v in zip(bars,means): ax.text(bar.get_x()+bar.get_width()/2,v,f'{v:.3g}',ha='center',va='bottom')
    fig.tight_layout(); fig.savefig(FIG/'TANNER_FIG3_COMPARATOR_MAE.png',dpi=220); fig.savefig(FIG/'TANNER_FIG3_COMPARATOR_MAE.svg'); plt.close(fig)
    d={c:{} for c in ['C0_TARGET_MEAN','C1_GLOBAL_FIXED','C2_TARGET_SPECIFIC']}
    for c in d:
        for sid in ids:
            d[c][sid]=metric_rows([r for r in standard if r['comparator']==c and int(r['outer_heldout_subject'])==sid])['mae']
    fig,ax=plt.subplots(figsize=(7,4))
    for sid in ids:
        ax.plot([0,1,2],[d['C0_TARGET_MEAN'][sid],d['C1_GLOBAL_FIXED'][sid],d['C2_TARGET_SPECIFIC'][sid]],marker='o',alpha=.75,label=str(sid))
    ax.set_xticks([0,1,2],['Target mean','Global fixed','Target-specific']); ax.set_ylabel('Subject-level mean absolute error')
    ax.set_title('Per-trabecula held-out error across four targets'); ax.legend(title='Trabecula',fontsize=7,ncol=2,frameon=False)
    fig.tight_layout(); fig.savefig(FIG/'TANNER_FIG4_SUBJECT_ERROR.png',dpi=220); fig.savefig(FIG/'TANNER_FIG4_SUBJECT_ERROR.svg'); plt.close(fig)
    perms=[r for r in f1rows if r['target_strain_rate_s_inv']=='ALL']
    if perms:
        vals=[float(r['mae']) for r in perms]; ident=next(float(r['mae']) for r in perms if r['is_identity_mapping'])
        fig,ax=plt.subplots(figsize=(7,4)); ax.hist(vals,bins=12,color='#8da0cb',edgecolor='white')
        ax.axvline(ident,color='#c44e52',lw=2,label=f'Identity mapping = {ident:.3g}')
        ax.set_xlabel('Outer LOTO MAE under target-selector mapping permutation'); ax.set_ylabel('Number of exact permutations')
        ax.set_title('F1 target-label permutation falsification (24 permutations)'); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(FIG/'TANNER_FIG5_TARGET_LABEL_PERMUTATION.png',dpi=220); fig.savefig(FIG/'TANNER_FIG5_TARGET_LABEL_PERMUTATION.svg'); plt.close(fig)

def run_primary(d,freeze):
    endpoint='peak_stress_response'
    ids,xx,yy,missing=prep(d,endpoint)
    if ids!=freeze['eligible_subjects']: raise RuntimeError(f'Eligible subject mismatch: {ids}')
    if len(ids)<6: raise RuntimeError('Fewer than six complete subjects')
    standard,cand,mat,sel,c3,f1,sens,winners=run_outer(endpoint,ids,xx,yy,True,True)
    rank=rank_summaries(endpoint,mat)
    summary,policy_rows=summary_table(endpoint,standard,c3,mat,winners)
    subjects=subject_summaries(endpoint,standard,c3)
    bootstrap=bootstrap_deltas(endpoint,standard)
    leverage=[]
    for fold,test in enumerate(ids,1):
        train=[s for s in ids if s!=test]
        for rate in CAND_RATES:
            xv=np.asarray([xx[rate][s] for s in train],dtype=float)
            mean=float(np.mean(xv)); den=float(np.sum((xv-mean)**2))
            hs=[(1/len(train) if den<=1e-14 else 1/len(train)+(float(x)-mean)**2/den) for x in xv]
            for sid,xval,h in zip(train,xv,hs):
                leverage.append({'outcome_family':endpoint,'outer_heldout_subject':test,'candidate_strain_rate_s_inv':rate,
                                 'training_subject_id':sid,'candidate_value':float(xval),'training_leverage':h,
                                 'fold_max_leverage':max(hs),'fold_max_leverage_subject':train[int(np.argmax(hs))],
                                 'reference_threshold_2p_over_n':2*2/len(train)})
    write_df(OUT/'TANNER_OUTER_LOTO_PREDICTIONS.csv',standard+c3)
    write_df(OUT/'TANNER_CANDIDATE_TARGET_MATRIX.csv',mat)
    write_df(OUT/'TANNER_TARGET_SPECIFIC_SELECTION.csv',sel)
    write_df(OUT/'TANNER_GLOBAL_COMPARATOR_RESULTS.csv',summary)
    random_table=[]
    for p,pr in policy_rows:
        assignment=pr[0].get('random_policy_assignment_json','')
        random_table.append({'outcome_family':endpoint,'random_policy_id':p,'assignment_json':assignment,'target_strain_rate_s_inv':'ALL',**metric_rows(pr)})
        for t in TARGET_RATES:
            q=[r for r in pr if r['target_strain_rate_s_inv']==t]
            random_table.append({'outcome_family':endpoint,'random_policy_id':p,'assignment_json':assignment,'target_strain_rate_s_inv':t,**metric_rows(q)})
    write_df(OUT/'TANNER_RANDOM_COMPARATOR_RESULTS.csv',random_table)
    write_df(OUT/'TANNER_TARGET_RANKING_SUMMARY.csv',rank)
    write_df(OUT/'TANNER_SUBJECT_LEVEL_SUMMARY.csv',subjects)
    # Falsification summaries: exact selector-label shuffles, comparator baselines, leakage audit and condition dominance.
    frows=list(f1)
    ident=next(float(r['mae']) for r in f1 if r['target_strain_rate_s_inv']=='ALL' and r['is_identity_mapping'])
    frows.append({'outcome_family':endpoint,'falsification':'F1_IDENTITY_VS_NONIDENTITY','target_strain_rate_s_inv':'ALL','result':ident,
                  'detail':f"Identity MAE={ident:.12g}; nonidentity permutation MAE range {min(float(r['mae']) for r in f1 if r['target_strain_rate_s_inv']=='ALL' and not r['is_identity_mapping']):.12g} to {max(float(r['mae']) for r in f1 if r['target_strain_rate_s_inv']=='ALL' and not r['is_identity_mapping']):.12g}; exact 24 mappings."})
    frows.append({'outcome_family':endpoint,'falsification':'F2_RANDOM_CANDIDATE','target_strain_rate_s_inv':'ALL','result':metric_rows(c3)['mae'],
                  'detail':'Full exact enumeration of 16 policies; policy-level results in TANNER_RANDOM_COMPARATOR_RESULTS.csv.'})
    frows.append({'outcome_family':endpoint,'falsification':'F3_GLOBAL_FIXED','target_strain_rate_s_inv':'ALL','result':metric_rows([r for r in standard if r['comparator']=='C1_GLOBAL_FIXED'])['mae'],
                  'detail':'Global candidate is nested-selected from outer-training subjects only.'})
    leaks=[r for r in standard+c3 if int(r['outer_heldout_subject']) in json.loads(r['training_subject_ids_json'])]
    frows.append({'outcome_family':endpoint,'falsification':'F4_SUBJECT_LEAKAGE_AUDIT','target_strain_rate_s_inv':'ALL','result':'PASS' if not leaks else 'FAIL',
                  'detail':f'{len(leaks)} prediction rows include held-out subject among fit IDs; all fold memberships asserted in code.'})
    counts={c:sum(next(r['candidate_strain_rate_s_inv'] for r in mat if int(r['target_strain_rate_s_inv'])==t)==c for t in TARGET_RATES) for c in CAND_RATES}
    winners_all=[min(CAND_RATES,key=lambda c:(next(r['mae'] for r in mat if int(r['target_strain_rate_s_inv'])==t and int(r['candidate_strain_rate_s_inv'])==c),c)) for t in TARGET_RATES]
    frows.append({'outcome_family':endpoint,'falsification':'F6_ONE_CONDITION_DOMINANCE','target_strain_rate_s_inv':'ALL','result':f'{counts[100]} targets favor 100; {counts[250]} favor 250',
                  'detail':'Outer-LOTO candidate winners by target: '+json.dumps(dict(zip(map(str,TARGET_RATES),winners_all)))})
    write_df(OUT/'TANNER_FALSIFICATION_RESULTS.csv',frows)
    write_df(OUT/'TANNER_LOO_SENSITIVITY.csv',sens)
    write_df(OUT/'TANNER_BOOTSTRAP_SUMMARY.csv',bootstrap)
    write_df(OUT/'TANNER_LEVERAGE_AUDIT.csv',leverage)
    make_figures(mat,standard,f1,ids)
    env={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':plt.matplotlib.__version__,
         'seed':BASE_SEED,'freeze_sha256':sha256(FREEZE),'independent_n':len(ids),'analysis_entrypoint':Path(__file__).name}
    (OUT/'TANNER_ANALYSIS_ENVIRONMENT.json').write_text(json.dumps(env,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'endpoint':endpoint,'n':len(ids),'c0_mae':metric_rows([r for r in standard if r['comparator']=='C0_TARGET_MEAN'])['mae'],
      'c1_mae':metric_rows([r for r in standard if r['comparator']=='C1_GLOBAL_FIXED'])['mae'],
      'c2_mae':metric_rows([r for r in standard if r['comparator']=='C2_TARGET_SPECIFIC'])['mae'],
      'candidate_winners':dict(zip(map(str,TARGET_RATES),winners_all)),'files_written':True},indent=2))

def run_secondaries(d):
    outpred=[]; outmat=[]; outsumm=[]
    for endpoint in ['minimum_stress_response','t12_s','t2_s','t23_s']:
        ids,xx,yy,missing=prep(d,endpoint)
        if len(ids)<6: 
            outsumm.append({'outcome_family':endpoint,'status':'NOT_RUN_INADEQUATE_COVERAGE','n_subjects':len(ids),'missing_conditions_json':json.dumps(missing)})
            continue
        standard,cand,mat,sel,c3,f1,sens,winners=run_outer(endpoint,ids,xx,yy,False,False)
        summary,_=summary_table(endpoint,standard,c3,mat,winners)
        outpred.extend(standard+c3); outmat.extend(mat); outsumm.extend(summary)
    write_df(OUT/'TANNER_SECONDARY_OUTER_LOTO_PREDICTIONS.csv',outpred)
    write_df(OUT/'TANNER_SECONDARY_CANDIDATE_TARGET_MATRIX.csv',outmat)
    write_df(OUT/'TANNER_SECONDARY_SUMMARY.csv',outsumm)
    print(json.dumps({'secondary_families':4,'predictions':len(outpred),'matrix_rows':len(outmat),'summary_rows':len(outsumm)},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--outcomes',choices=['primary','secondary'],required=True)
    args=ap.parse_args()
    d,freeze=read_data()
    if args.outcomes=='primary': run_primary(d,freeze)
    else: run_secondaries(d)


