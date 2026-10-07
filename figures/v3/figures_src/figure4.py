"""Rebuild adaptive risk ratios and observed scored-fold resolution frequencies."""
import csv
import hashlib
import json
from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from style import ROOT,INPUT,COLORS,INK,edge,clean,save

SYSTEMS=['Awinda','RLC-1','Radbill','Tanner']
IDS=dict(zip(SYSTEMS,['AWINDA_MOUSE','RLC1_RAT','RADBILL_HUMAN_PACING','TANNER_RAT_TRABECULA']))

def read(name):
    with (INPUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    canonical={r['system']:r for r in read('canonical_adaptive.csv')}
    frequencies=read('resolution_frequencies.csv')
    ratios=[]
    for s in SYSTEMS:
        r=canonical[s]
        fields=['fixed_shared_risk','fixed_exact_risk']+(['fixed_family_risk'] if s in ['RLC-1','Radbill'] else [])
        best=min(Decimal(r[f]) for f in fields)
        assert best==Decimal(r['best_fixed_risk'])
        ratio=Decimal(r['adaptive_heldout_risk'])/best
        ratios.append(dict(system=s,fixed_resolution=r['best_fixed_resolution_in_candidate_set'].title(),
            best_fixed_native_risk=r['best_fixed_risk'],adaptive_native_risk=r['adaptive_heldout_risk'],
            native_metric=r['native_metric'],best_fixed_normalized='1',adaptive_normalized=str(ratio),
            reference_role='descriptive realized best eligible fixed resolution; not oracle',
            comparator_exclusion='C0 and Conventional and nearest/random are not resolution candidates'))
    with (INPUT/'figure4_risk_ratios.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ratios[0]));w.writeheader();w.writerows(ratios)
    freq_rows=[]
    for s in SYSTEMS:
        for r in frequencies:
            if r['system_id']==IDS[s]:
                freq_rows.append(dict(system=s,resolution=r['resolution_id'].title(),
                    selected=r['n_scored_outer_units_selected'],total=r['n_scored_outer_units'],
                    frequency=r['frequency_scored_outer_units'],shown_in_artwork=str(int(r['n_scored_outer_units_selected'])>0)))
    with (INPUT/'figure4_resolution_counts.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(freq_rows[0]));w.writeheader();w.writerows(freq_rows)
    expected={'Awinda':[1,9],'RLC-1':[2,3,2],'Radbill':[0,2,15],'Tanner':[3,4]}
    for s in SYSTEMS:
        rs=[r for r in freq_rows if r['system']==s]
        assert [int(r['selected']) for r in rs]==expected[s]
        assert sum(int(r['selected']) for r in rs)==int(rs[0]['total'])
    fig,axs=plt.subplots(2,4,figsize=(7.2,5.45),sharey='row',gridspec_kw={'height_ratios':[1.1,1]})
    fig.subplots_adjust(left=.095,right=.985,bottom=.22,top=.85,wspace=.40,hspace=.67)
    for i,s in enumerate(SYSTEMS):
        top,bot=axs[:,i];ratio=float(ratios[i]['adaptive_normalized'])
        top.plot([0,1],[1,ratio],color='#B9C1C8',lw=1.2,zorder=1)
        top.scatter([0],[1],s=35,color=COLORS['Exact'],edgecolor=edge(COLORS['Exact']),lw=.8,zorder=3)
        top.scatter([1],[ratio],s=38,marker='D',color=COLORS['Adaptive'],edgecolor=INK,lw=.7,zorder=3)
        top.axhline(1,color='#B3BCC4',ls=(0,(3,3)),lw=.75,zorder=0)
        top.text(0,1.014,'1.000',ha='center',fontsize=7.5)
        top.text(1,ratio+.014,f'{ratio:.3f}',ha='center',fontsize=7.5)
        top.set_ylim(.97,1.22);top.set_xlim(-.3,1.3)
        top.set_yticks([1.0,1.1,1.2]);top.set_xticks([0,1],['Best fixed\n(Exact)','Adaptive'])
        top.tick_params(axis='x',length=0,pad=5)
        top.set_title(s,loc='left',pad=18)
        top.text(-.16,1.18,'ABCD'[i],transform=top.transAxes,fontsize=11,fontweight='bold')
        clean(top,grid=True)
        rs=[r for r in freq_rows if r['system']==s and int(r['selected'])>0]
        bottom=0
        for r in rs:
            height=100*int(r['selected'])/int(r['total']);color=COLORS[r['resolution']]
            bot.bar(0,height,width=.62,bottom=bottom,color=color,edgecolor=edge(color),lw=.8)
            bot.text(0,bottom+height/2,f"{r['selected']}/{r['total']}",ha='center',va='center',fontsize=8,
                color='white' if r['resolution']=='Shared' else INK)
            bottom+=height
        bot.set_xlim(-.65,.65);bot.set_ylim(0,100);bot.set_yticks([0,50,100])
        bot.set_xticks([0],[f"{rs[0]['total']} scored folds"]);bot.tick_params(axis='x',length=0,pad=5)
        clean(bot,grid=True)
    axs[0,0].set_ylabel('Risk / best fixed resolution',labelpad=7)
    axs[1,0].set_ylabel('Selected resolution (%)',labelpad=7)
    fig.legend(handles=[Patch(facecolor=COLORS[c],edgecolor=edge(COLORS[c]),label=c) for c in ['Shared','Family','Exact']],
        loc='lower center',bbox_to_anchor=(.55,.09),ncol=3,frameon=False,handlelength=1.3,columnspacing=2.2)
    save(fig,4)
    qa=ROOT/'qa';qa.mkdir(exist_ok=True)
    (qa/'figure4_build_notes.md').write_text('''# Figure 4 build notes

A–D are Awinda,RLC-1,Radbill,Tanner columns. Upper rows normalize adaptive risk by each system's best eligible fixed resolution; Exact is that realized minimum in all four. Decimal arithmetic is exported at full precision in `figure4_risk_ratios.csv`; artwork rounds to3decimals. Fixed eligibility excludes C0,Conventional,nearest-rate and random comparators. The minimum is a descriptive realized reference, not population oracle, unbiased regret or a policy chosen with holdout data. No uncertainty is shown.

Lower rows show actual scored-fold frequencies, with count fractions inside segments. Radbill Shared was eligible and selected0/17times; its zero row is retained in the CSV but no Shared segment is drawn. Family is absent from Awinda/Tanner. Asserted counts:1/9of10;2/3/2of7;0/2/15of17;3/4of7. PNG/PDF/SVG600dpi,width7.2in.
''',encoding='utf-8')
    hashes={name:hashlib.sha256((INPUT/name).read_bytes()).hexdigest() for name in ['canonical_adaptive.csv','resolution_frequencies.csv']}
    (qa/'figure4_input_provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')

if __name__=='__main__':main()
