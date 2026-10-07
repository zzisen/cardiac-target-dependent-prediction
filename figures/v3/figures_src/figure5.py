"""Rebuild Figure 5 from frozen numerical snapshots, from any working directory.

No historical artwork or renderer is used. Contrast intervals are never placed
on absolute risks. Deletion summaries are plotted without policy refitting.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from style import INPUT, ROOT, COLORS, INK, edge, clean, save

POLICIES = ['C0', 'SHARED', 'EXACT']
PRETTY = {'C0': 'C0', 'SHARED': 'Shared', 'EXACT': 'Exact'}
CONTRASTS = ['C0_minus_SHARED', 'C0_minus_EXACT', 'SHARED_minus_EXACT']
CONTRAST_LABELS = ['C0 − Shared', 'C0 − Exact', 'Shared − Exact']
CONTRAST_COLORS = [COLORS['Shared'], '#81708C', COLORS['Exact']]
SOURCES = ['canonical_training.csv', 'canonical_transport.csv', 'training_risks.csv',
           'training_bootstrap.csv', 'training_contrasts.csv',
           'training_exact_actions.csv', 'training_shared_actions.csv',
           'training_deletion_contrasts.csv', 'transport_risks.csv',
           'transport_target_summary.csv']

def read(name):
    with (INPUT/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def load_and_validate():
    training = read('canonical_training.csv')
    transport = read('canonical_transport.csv')
    assert {int(r['biological_n']) for r in training} == {44}
    assert {int(r['primary_FCS']) for r in training} == {352}
    assert {int(r['channel_count']) for r in training} == {8}
    channels = training[0]['channels'].split('|')
    targets = training[0]['targets'].split('|')
    assert len(channels) == 8 and len(targets) == 7 and 'pCREB' not in channels
    assert {int(r['patients']) for r in transport} == {9}
    assert {int(r['FCS_files']) for r in transport} == {27}
    assert {len(r['targets'].split('|')) for r in transport} == {2}
    tr = {r['policy']: float(r['overall_equal_patient_equal_target_risk'])
          for r in read('training_risks.csv')}
    te = {r['policy']: float(r['risk']) for r in read('transport_risks.csv')}
    for rows, values in [(training, tr), (transport, te)]:
        for r in rows:
            if r['record_type'] == 'POLICY_RISK':
                assert values[r['policy_or_contrast']] == float(r['estimate'])
    assert sorted(tr, key=tr.get) == ['SHARED', 'C0', 'EXACT']
    assert sorted(te, key=te.get) == ['EXACT', 'SHARED', 'C0']
    boot = {r['contrast']: r for r in read('training_bootstrap.csv')}
    for c in CONTRASTS:
        assert int(boot[c]['replicates']) == 50000
        a, b = c.split('_minus_')
        assert np.isclose(tr[a]-tr[b], float(boot[c]['estimate']), atol=1e-12)
        canonical = next(r for r in training if r['policy_or_contrast'] == c)
        for a, b in [('estimate', 'estimate'), ('percentile_95_low', 'bootstrap_95_low'),
                     ('percentile_95_high', 'bootstrap_95_high')]:
            assert float(boot[c][a]) == float(canonical[b])
    actions = read('training_exact_actions.csv')
    assert len(actions) == 308
    matrix = []
    for t in targets:
        canonical = next(r for r in training
                         if r['policy_or_contrast'] == 'EXACT; '+t)
        counts = json.loads(canonical['selected_channel_frequencies'])
        observed = Counter(r['selected_exact_channel'] for r in actions if r['target'] == t)
        assert sum(counts.values()) == 44
        assert {k: observed[k] for k in channels} == counts
        assert len({r['held_out_patient'] for r in actions if r['target'] == t}) == 44
        matrix.append([counts[k] for k in channels])
    shared = read('training_shared_actions.csv')
    key = next(k for k in shared[0] if 'selected_shared_channel' in k)
    assert Counter(r[key] for r in shared) == {'pSYK':43, 'pERK1/2':1}
    deletion = read('training_deletion_contrasts.csv')
    assert len(deletion) == 44 and len({r['omitted_patient'] for r in deletion}) == 44
    for r in deletion:
        for c in CONTRASTS:
            a,b=c.split('_minus_')
            assert np.isclose(float(r[c]), float(r['risk_'+a])-float(r['risk_'+b]), atol=1e-12)
    directions = [int(sum(np.sign(float(r[c])) == np.sign(float(boot[c]['estimate']))
                         for r in deletion)) for c in CONTRASTS]
    assert directions == [44,43,44]
    target_risks = read('transport_target_summary.csv')
    assert [r['target'] for r in target_risks] == ['Dasatinib','IL-7']
    for p in POLICIES:
        assert np.isclose(np.mean([float(r['risk_'+p]) for r in target_risks]),te[p],atol=1e-15)
    assert target_risks[0]['risk_SHARED'] == target_risks[0]['risk_EXACT']
    mapping = transport[0]['training_only_exact_selection']
    assert mapping == 'Dasatinib=pSYK; IL-7=pSTAT5'
    return training,transport,tr,te,boot,targets,channels,np.array(matrix),deletion,target_risks,directions

def export_processed(data):
    training,transport,tr,te,boot,targets,channels,matrix,deletion,target_risks,directions=data
    rows=[]
    def add(panel,obj,category,value,source,row,**kwargs):
        rows.append(dict(panel=panel,object=obj,category=category,value=value,
                         source_input=source,source_row=row,**kwargs))
    for p in POLICIES:
        r=next(r for r in read('training_risks.csv') if r['policy']==p)
        add('B','absolute_policy_risk',p,r['overall_equal_patient_equal_target_risk'],
            'training_risks.csv',p,low='',high='')
    for c in CONTRASTS:
        r=boot[c]
        add('B','paired_contrast_50000_patient_bootstrap',c,r['estimate'],
            'training_bootstrap.csv',c,low=r['percentile_95_low'],high=r['percentile_95_high'])
    for i,t in enumerate(targets):
        for j,k in enumerate(channels):
            add('C','selected_outer_folds_out_of_44',t+'|'+k,int(matrix[i,j]),
                'canonical_training.csv','EXACT; '+t,low='',high='')
    for r in deletion:
        for c in CONTRASTS:
            add('D','leave_one_patient_contrast',r['omitted_patient']+'|'+c,r[c],
                'training_deletion_contrasts.csv',r['omitted_patient'],low='',high='')
    for p in POLICIES:
        r=next(r for r in read('transport_risks.csv') if r['policy']==p)
        add('E','training_frozen_standardized_MSE','Overall|'+p,r['risk'],'transport_risks.csv',p,low='',high='')
    for r in target_risks:
        for p in POLICIES:
            add('E','training_frozen_standardized_MSE',r['target']+'|'+p,r['risk_'+p],
                'transport_target_summary.csv',r['target'],low='',high='')
    path=INPUT/'figure5_processed_values.csv'
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'figure':5,'dimensions_inches':[7.2,7.4], 'png_dpi':600,
        'source_sha256':{f:hashlib.sha256((INPUT/f).read_bytes()).hexdigest() for f in SOURCES},
        'checks':{'training_support':[44,352,7,8], 'transport_support':[9,27,2],
                  'heatmap_shape':list(matrix.shape),'heatmap_rows_sum_to_44':bool(np.all(matrix.sum(axis=1)==44)),
                  'exact_map_matches_canonical_json':True,'shared_action_counts':{'pSYK':43,'pERK1/2':1},
                  'deletion_rows':44,'direction_preserving_deletions':dict(zip(CONTRASTS,directions)),
                  'contrast_intervals_not_on_policy_risks':True,'no_new_resampling_or_refit':True,
                  'Stanford_target_means_match_overall':True,'historical_artwork_read':False},
        'transforms':'Exact count aggregation validated against 308 source selections; frozen point values and intervals retained. No scientific model rerun.',
        'minimum_text_pt':7.5,
        'transport_metric_authority':{'source':'P2_V3_ABT/B2_STANFORD_TRANSPORT_LUNA15_2026-10-06/B2_STANFORD_EXECUTION.py',
                                     'lines':[154,157,167],
                                     'definition':'yz=(y-training_mean)/training_sample_sd; loss=mean((yz-pred_z)**2)',
                                     'display':'Mean squared standardized error; training-frozen scaling. Not raw fraction MSE.'}}
    (ROOT/'figures_src'/'figure5_qa.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

def label(fig,letter,title,x,y):
    fig.text(x,y,letter,fontsize=11,fontweight='bold',va='bottom')
    fig.text(x+.028,y+.001,title,fontsize=9,fontweight='bold',va='bottom')

def main():
    data=load_and_validate(); export_processed(data)
    training,transport,tr,te,boot,targets,channels,matrix,deletion,target_risks,directions=data
    fig=plt.figure(figsize=(7.2,7.4))
    label(fig,'A','Datasets and evaluation',.045,.954)
    label(fig,'B','Training cohort',.363,.954)
    label(fig,'C','Exact channel selections',.699,.954)
    label(fig,'D','Patient-deletion sensitivity',.045,.465)
    label(fig,'E','Stanford transport cohort',.363,.465)
    label(fig,'F','Observed ordering shift',.699,.465)
    # A: counts and the prespecified evaluation boundaries, no anatomical artwork.
    ax=fig.add_axes([.045,.537,.265,.38]); ax.axis('off')
    ax.text(0,.96,'Training cohort',weight='bold',size=9,va='top')
    ax.text(0,.84,'44 patients · 352 FCS\n7 targets · 8 response channels',size=8,va='top',linespacing=1.65)
    ax.text(0,.65,'Nested patient holdout\nTraining-only action selection\nOnce-only outer evaluation',size=8,va='top',linespacing=1.65)
    ax.annotate('',xy=(.48,.40),xytext=(.48,.48),arrowprops={'arrowstyle':'->','color':INK,'lw':1})
    ax.text(0,.35,'Stanford transport cohort',weight='bold',size=8.5,va='top')
    ax.text(0,.23,'9 patients · 27 FCS\n2 scored targets',size=8,va='top',linespacing=1.65)
    ax.text(0,.065,'Training-frozen policies\nNo transport refit or recalibration',size=7.5,va='top',linespacing=1.55)
    # B: physically separate axes for absolute risks and paired intervals.
    ax=fig.add_axes([.408,.756,.230,.148]); clean(ax,grid=True)
    xs=np.arange(3); vals=[tr[p] for p in POLICIES]
    bars=ax.bar(xs,vals,width=.60,color=[COLORS[PRETTY[p]] for p in POLICIES],
                edgecolor=[edge(COLORS[PRETTY[p]]) for p in POLICIES],lw=.8)
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+1.1,f'{v:.2f}',ha='center',size=8)
    ax.set_ylim(0,50);ax.set_yticks([0,25,50]);ax.set_xticks(xs,[PRETTY[p] for p in POLICIES])
    ax.set_ylabel('Risk',labelpad=3)
    fig.text(.408,.716,'Mean squared standardized error',size=7.5)
    fig.text(.363,.676,'Paired contrasts',size=8,weight='bold')
    cx=fig.add_axes([.468,.600,.17,.076]);clean(cx)
    cx.axvline(0,color='#C1C7CD',lw=.8,zorder=0)
    for i,c in enumerate(CONTRASTS):
        r=boot[c];v=float(r['estimate']);lo=float(r['percentile_95_low']);hi=float(r['percentile_95_high'])
        cx.plot([lo,hi],[2-i]*2,color=CONTRAST_COLORS[i],lw=1.4)
        cx.plot(v,2-i,'o',color=CONTRAST_COLORS[i],ms=3.7)
    cx.set_ylim(-.5,2.5);cx.set_xlim(-33,14);cx.set_xticks([-30,-15,0,10]);cx.set_yticks([2,1,0],CONTRAST_LABELS)
    cx.tick_params(axis='y',length=0,pad=3)
    fig.text(.363,.526,'95% descriptive percentile intervals\n50,000 whole-patient resamples',size=7.5,linespacing=1.5)
    # C: discrete bins, count text in every cell, candidate vocabulary preserved.
    hx=fig.add_axes([.808,.646,.172,.257])
    cmap=ListedColormap(['#F4F6F7','#D9E8ED','#AACBD8','#689EB5','#326C87'])
    bounds=[-.5,.5,5.5,15.5,30.5,44.5]
    hx.imshow(matrix,aspect='auto',cmap=cmap,norm=BoundaryNorm(bounds,cmap.N),interpolation='none')
    for i in range(7):
        for j in range(8):hx.text(j,i,str(matrix[i,j]),ha='center',va='center',size=7.5,
                                    color='white' if matrix[i,j]>=31 else INK)
    hx.set_yticks(np.arange(7),targets);hx.set_xticks(np.arange(8),channels,rotation=90)
    hx.tick_params(length=0,pad=3)
    for s in hx.spines.values():s.set_visible(False)
    hx.set_xticks(np.arange(-.5,8,1),minor=True);hx.set_yticks(np.arange(-.5,7,1),minor=True)
    hx.grid(which='minor',color='white',lw=.5);hx.tick_params(which='minor',length=0)
    fig.text(.699,.922,'Selected outer folds / 44',size=7.5)
    fig.text(.699,.526,'Shared: pSYK 43/44; pERK1/2 1/44',size=7.5)
    # D: all 44 deletion contrasts, in source patient order; no outlier inference.
    dx=fig.add_axes([.096,.224,.216,.196]);clean(dx,grid=True)
    x=np.arange(1,45)
    for c,color,m in zip(CONTRASTS,CONTRAST_COLORS,['o','s','^']):
        y=[float(r[c]) for r in deletion]
        dx.axhline(float(boot[c]['estimate']),color=color,lw=.9,ls=(0,(3,2)),alpha=.7)
        dx.plot(x,y,color=color,linestyle='none',marker=m,ms=2.7)
    dx.axhline(0,color='#ADB5BD',lw=.65)
    dx.set_xlim(0,45);dx.set_ylim(-11.5,5.5);dx.set_xticks([1,11,22,33,44]);dx.set_yticks([-10,-5,0,5])
    dx.set_ylabel('Risk contrast',labelpad=3);dx.set_xlabel('Omitted patient (source order)',fontsize=7.5,labelpad=4)
    fig.text(.045,.132,'Points: one patient removed\nDashed: full-cohort contrast',size=7.5,linespacing=1.5)
    for i,(text,color,m) in enumerate(zip(CONTRAST_LABELS,CONTRAST_COLORS,['o','s','^'])):
        fig.text(.045,.087-i*.022,m.replace('o','●').replace('s','■').replace('^','▲'),color=color,size=8)
        fig.text(.063,.087-i*.022,text,size=7.5)
    # E: a visibly restricted numerical axis for frozen points, not zero-based bars.
    ex=fig.add_axes([.464,.248,.174,.172]);clean(ex)
    groups=[('Overall',te)]+[(r['target'],{p:float(r['risk_'+p]) for p in POLICIES}) for r in target_risks]
    for i,(group,v) in enumerate(groups):
        for j,(p,m) in enumerate(zip(POLICIES,['o','s','^'])):
            ex.plot(v[p],2-i+(1-j)*.18,marker=m,color=COLORS[PRETTY[p]],ms=5,linestyle='none',mec=edge(COLORS[PRETTY[p]]),mew=.55)
    ex.set_ylim(-.45,2.5);ex.set_xlim(.315,.38);ex.set_xticks([.32,.34,.36,.38]);ex.set_yticks([2,1,0],['Overall','Dasatinib','IL-7'])
    ex.tick_params(axis='y',length=0,pad=3);ex.set_xlabel('Mean squared\nstandardized error',fontsize=8,labelpad=5)
    fig.text(.363,.151,'Training-frozen scaling; points only',size=7.5)
    fig.text(.363,.116,'Dasatinib: Shared = Exact (pSYK)\nIL-7: Shared pSYK; Exact pSTAT5',size=7.5,linespacing=1.65)
    for i,(p,m) in enumerate(zip(POLICIES,['●','■','▲'])):
        fig.text(.363+i*.089,.065,m,color=COLORS[PRETTY[p]],size=8)
        fig.text(.377+i*.089,.065,PRETTY[p],size=7.5)
    # F: ordering labels are derived from the frozen values; no magnitude bridge.
    fx=fig.add_axes([.699,.054,.284,.365]);fx.axis('off')
    for yy,name,v in [(.94,'Training cohort',tr),(.62,'Stanford transport cohort',te)]:
        fx.text(0,yy,name,size=8.5,weight='bold',va='top')
        for i,p in enumerate(sorted(v,key=v.get)):
            fx.text([0,.44,.82][i],yy-.16,PRETTY[p],size=8.5,color=edge(COLORS[PRETTY[p]]),weight='bold',va='top')
            if i<2:fx.text([.36,.73][i],yy-.16,'<',size=8.5,va='top')
    fx.text(0,.31,'Cohort-specific realized\npoint estimates',size=8,va='top',linespacing=1.6)
    fx.text(0,.10,'No population-optimum or\nsite-interaction claim',size=7.5,va='top',linespacing=1.6)
    # Check that every explicit label stays at the submission-density floor.
    for ax in fig.axes:
        for t in ax.texts+ax.get_xticklabels()+ax.get_yticklabels():
            assert t.get_fontsize()>=7.5
    save(fig,5)
    print('Figure 5 rebuilt: canonical values, all 44 deletions, 7×8 count map; PNG/PDF/SVG.')

if __name__ == '__main__':main()
