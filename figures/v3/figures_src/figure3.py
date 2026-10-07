"""Rebuild natural-family structures and descriptive deletion-rank controls."""
import csv
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from style import ROOT, INPUT, COLORS, INK, edge, clean, save

def natural(name):
    with (INPUT/name).open(encoding='utf-8-sig',newline='') as f:
        return [r for r in csv.DictReader(f) if r['is_natural_partition']=='1']

def block(ax,y,height,title,targets):
    color=COLORS['Family']
    ax.add_patch(Rectangle((.01,y),.97,height,facecolor='#F0F7F4',edgecolor=edge(color),lw=.8))
    ax.add_patch(Rectangle((.01,y),.012,height,color=color,lw=0))
    ax.text(.055,y+height-.045,title,fontsize=8.5,fontweight='bold',va='top')
    for j,t in enumerate(targets):
        row=j//2;col=j%2
        ax.text(.055+col*.48,y+height-.16-row*.12,t,fontsize=8,va='top')

def ranks(ax,rows,denom,unit,letter):
    x=np.arange(len(rows));rank=[int(r['rank_lower_is_better']) for r in rows]
    ids=[r['deleted_biological_unit'].replace('Rat_','') for r in rows]
    ax.vlines(x,.55,rank,color='#BDDAD0',lw=1)
    ax.scatter(x,rank,s=26,color=COLORS['Family'],edgecolor=edge(COLORS['Family']),lw=.7,zorder=3)
    ax.set_ylim(3.6,.45);ax.set_yticks([1,2,3])
    ax.set_xlim(-.6,len(rows)-.4)
    ax.set_xticks(x,ids,rotation=55 if denom==35 else 0,ha='right' if denom==35 else 'center')
    ax.set_ylabel(f'Natural-family rank\namong {denom} partitions',labelpad=7)
    ax.set_xlabel(f'Omitted {unit}',labelpad=6)
    ax.set_title('Natural-family deletion rank',loc='left',pad=16,fontsize=9)
    ax.text(-.15,1.20,letter,transform=ax.transAxes,fontsize=11,fontweight='bold')
    ax.text(0,1.035,'1 = best',transform=ax.transAxes,fontsize=7.5,ha='left')
    if denom==35:
        ax.text(.5,.08,'Rank 1 in all 17 deletions',transform=ax.transAxes,ha='center',fontsize=8)
    clean(ax,grid=True)

def main():
    rlc=natural('rlc_natural_ranks.csv');rad=natural('radbill_natural_ranks.csv')
    assert [int(r['rank_lower_is_better']) for r in rlc]==[1,1,2,2,2,3,1]
    assert len(rad)==17 and all(r['rank_lower_is_better']=='1' for r in rad)
    assert not {'9','19'} & {r['deleted_biological_unit'] for r in rad}
    combined=[dict(system='RLC-1' if den==15 else 'Radbill',rank_denominator=den,**r) for rows,den in [(rlc,15),(rad,35)] for r in rows]
    with (INPUT/'figure3_natural_deletion_ranks.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)
    families=[('RLC-1','Peak',['3 µM Peak','10 µM Peak']),('RLC-1','TTP',['3 µM TTP','10 µM TTP']),
        ('RLC-1','RT50',['3 µM RT50','10 µM RT50']),('Radbill','QT',['110 QT','120 QT','130-first QT','130-last QT']),
        ('Radbill','PP',['110 PP','120 PP','130-first PP','130-last PP'])]
    with (INPUT/'figure3_family_membership.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['system','family','condition_label','semantic_source'])
        for system,family,targets in families:
            for target in targets:w.writerow([system,family,target,'owner full-rebuild contract; canonical A24 family definitions'])
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.4),gridspec_kw={'height_ratios':[1.25,1]})
    fig.subplots_adjust(left=.105,right=.985,bottom=.15,top=.90,wspace=.42,hspace=.54)
    a,c=axs[0];b,d=axs[1]
    for ax,letter,title in [(a,'A','RLC-1 natural target families'),(c,'C','Radbill natural target families')]:
        ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
        ax.text(0,1.08,title,fontsize=9,fontweight='bold',transform=ax.transAxes)
        ax.text(-.15,1.09,letter,fontsize=11,fontweight='bold',transform=ax.transAxes)
    for y,title,targets in [(0.68,'Peak family · 2 matched targets',families[0][2]),(.355,'TTP family · 2 matched targets',families[1][2]),(.03,'RT50 family · 2 matched targets',families[2][2])]:
        block(a,y,.29,title,targets)
    block(c,.535,.435,'QT family · 4 matched targets',families[3][2])
    block(c,.03,.435,'PP family · 4 matched targets',families[4][2])
    ranks(b,rlc,15,'rat','B');ranks(d,rad,35,'participant','D')
    save(fig,3)
    qa=ROOT/'qa';qa.mkdir(exist_ok=True)
    (qa/'figure3_build_notes.md').write_text('''# Figure 3 build notes

A/C are structural family blocks, not measurement waveforms. Target names are exactly the owner contract: Peak/TTP/RT50 at3/10µM and QT/PP at110/120/130-first/130-last. B/D contain only the natural partition rows from accepted A24 leave-one-unit rank exports. These deletion analyses remove one unit from frozen held-out loss summaries; they are not a new policy fit or the rank of that unit's individual error. The artwork therefore labels x as omitted rat/participant. Y displays observed rank range1–3 (1best), while labels explicitly give the full comparison spaces15/35. Ranks are descriptive, not p-values.

Checks: RLC vector1,1,2,2,2,3,1; Radbill17rank-one points; actual IDs1–8,10–18; no9/19; natural-only rows, no Shared/Family/Exact rank curves; membership table retained. PNG/PDF/SVG600dpi,width7.2in.
''',encoding='utf-8')
    hashes={name:hashlib.sha256((INPUT/name).read_bytes()).hexdigest() for name in ['rlc_natural_ranks.csv','radbill_natural_ranks.csv','canonical_natural_family.csv']}
    (qa/'figure3_input_provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')

if __name__=='__main__':main()
