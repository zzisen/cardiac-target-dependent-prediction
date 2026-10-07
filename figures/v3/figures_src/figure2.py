"""Rebuild Figure 2 from frozen native policy estimates; no uncertainty added."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from style import ROOT, INPUT, COLORS, INK, edge, clean, save

SYSTEMS = ['Awinda', 'RLC-1', 'Radbill', 'Tanner']

def read(name):
    with (INPUT/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def source_data():
    adaptive = {r['system']:r for r in read('canonical_adaptive.csv')}
    points=[]
    for system in SYSTEMS:
        r=adaptive[system]
        for label, field in [('C0','fixed_C0_risk'),('Shared','fixed_shared_risk'),
                ('Family','fixed_family_risk'),('Exact','fixed_exact_risk')]:
            if r[field]:
                points.append(dict(system=system,policy=label,value=r[field],
                    metric=r['native_metric'],role='fixed resolution' if label!='C0' else 'reference',
                    source='canonical_adaptive.csv',source_field=field,
                    biological_n=r['outer_biological_n'],support=r['risk_support_label']))
        if system=='Radbill':
            points.append(dict(system=system,policy='Conventional',value=r['fixed_conventional_comparator'],
                metric=r['native_metric'],role='separate comparator',source='canonical_adaptive.csv',
                source_field='fixed_conventional_comparator',biological_n=r['outer_biological_n'],support=r['risk_support_label']))
        if system=='Tanner':
            ledger=read('canonical_master.csv')
            for label,evidence_id,policy in [('Nearest-rate','EV-0102','NEAREST'),('Random','EV-0103','RANDOM')]:
                row=next(x for x in ledger if x['evidence_id']==evidence_id)
                assert row['system']=='Tanner' and row['policy_or_comparator']==policy
                assert row['n_biological_units']=='7' and row['n_target_cells_or_observations']=='28'
                points.append(dict(system=system,policy=label,value=row['value'],metric='native MAE (mN/mm²)',
                    role='separate comparator',source='canonical_master.csv',source_field=evidence_id,
                    biological_n=row['n_biological_units'],support='28 trabecula × target predictions; frozen A25 native comparator'))
    return points

def main():
    points=source_data()
    assert len(points)==17
    assert [p['policy'] for p in points if p['system']=='Awinda']==['C0','Shared','Exact']
    assert not any(p['system']=='Tanner' and p['policy']=='Family' for p in points)
    with (INPUT/'figure2_native_points.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
    fig,axs=plt.subplots(1,4,figsize=(7.2,4.75))
    fig.subplots_adjust(left=.075,right=.985,bottom=.34,top=.73,wspace=.66)
    subtitles=['Mouse frequency design\n10 mice\n114 spectra','Regulatory light chain\n7 rats\n42 target cells',
        'Human pacing\n17 participants\n126 target cells','Intact trabeculae\n7 preparations\n28 target cells']
    metrics=['NRMSE (%)','Standardized MAE','Standardized MAE','MAE (mN/mm²)']
    takes=['Exact has\nlowest risk','Family nearly\nmatches exact','Family captures most\nof the shared-to-exact\ngain','Little gain\nbeyond shared']
    for i,(system,ax) in enumerate(zip(SYSTEMS,axs)):
        rows=[p for p in points if p['system']==system]
        labels=[p['policy'] for p in rows];values=np.array([float(p['value']) for p in rows])
        x=np.arange(len(rows),dtype=float)
        sep=next((j for j,p in enumerate(rows) if p['role']=='separate comparator'),None)
        if sep is not None:
            x[sep:]+=.65
            ax.axvline(x[sep]-.8,color='#C9D0D5',lw=.65,ls=(0,(2,3)))
        for xpos,label,val in zip(x,labels,values):
            color=COLORS[label]
            ax.bar(xpos,val,width=.69,color=color,edgecolor=edge(color),linewidth=.85,
                hatch='//' if label in ('Conventional','Nearest-rate','Random') else None)
            ax.text(xpos,val+values.max()*.025,f'{val:.2f}' if system=='Awinda' else f'{val:.3f}',
                fontsize=7.5,ha='center',va='bottom',rotation=0 if system=='Awinda' else 90)
        ax.set_ylim(0,values.max()*1.22)
        ax.set_xticks(x,labels,rotation=58,ha='right',rotation_mode='anchor')
        ax.set_ylabel(metrics[i],labelpad=4)
        ax.set_title(system,loc='left',pad=55)
        ax.text(0,1.10,subtitles[i],transform=ax.transAxes,ha='left',va='bottom',fontsize=7.5,linespacing=1.5)
        ax.text(-.05,1.56,'ABCD'[i],transform=ax.transAxes,fontsize=11,fontweight='bold',ha='left')
        ax.text(.5,-.55,takes[i],transform=ax.transAxes,ha='center',va='top',fontsize=7.5,linespacing=1.4)
        clean(ax,grid=True)
        ax.tick_params(axis='x',length=0,pad=5)
    save(fig,2)
    qa=ROOT/'qa';qa.mkdir(exist_ok=True)
    (qa/'figure2_build_notes.md').write_text('''# Figure 2 build notes

Panels A–D show Awinda, RLC-1, Radbill and Tanner native frozen risk point estimates. Full-precision input strings are retained in `figure2_native_points.csv`; artwork rounds Awinda to two decimals and other systems to three. Canonical A25 fixed values resolve strict Radbill support (126 cells /17 contributors). Tanner nearest-rate and random are frozen canonical A25 EV-0102/EV-0103 separate comparator rows, not resolution classes. These full-precision exports supersede rounded A23 comparator strings for the final artwork inputs. Hatched bars and a divider separate comparators. No error bars, bootstrap intervals, retrospective oracles, historical images, or scientific refitting are used.

Checks: 17 allowed plotted points; Awinda and Tanner Family absent; source roles and supports exported; native Tanner unit mN/mm²; all risk axes start at zero. Rendered PNG, PDF and SVG use shared styling, width7.2in,600dpi.
''',encoding='utf-8')
    provenance={name:hashlib.sha256((INPUT/name).read_bytes()).hexdigest() for name in ['canonical_adaptive.csv','canonical_master.csv']}
    (qa/'figure2_input_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')

if __name__=='__main__': main()
