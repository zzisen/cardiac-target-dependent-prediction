"""Rebuild Figure 1 from mathematical definitions and schematic geometry.

All coordinates in this script are layout-only. No risk curve is empirical or
fitted to a biological dataset, and no numerical risk/sample-size axis is used.
"""
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from style import INPUT, COLORS, INK, panel, save

def arrow(ax,start,end,**kwargs):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=9,
        color=INK,lw=.9,**kwargs))

def box(ax,xy,width,height,text,face='#F3F5F6',edge=INK,fontsize=8):
    ax.add_patch(Rectangle(xy,width,height,facecolor=face,edgecolor=edge,lw=.8))
    ax.text(xy[0]+width/2,xy[1]+height/2,text,ha='center',va='center',fontsize=fontsize)

def main():
    INPUT.mkdir(exist_ok=True)
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.45))
    fig.subplots_adjust(left=.075,right=.98,bottom=.13,top=.90,wspace=.31,hspace=.48)
    for ax in axs.flat:
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')

    ax=axs[0,0]
    panel(ax,'A','Target partitions',x=-.12,y=1.09)
    rows=[('Shared','One action for all targets',.73,'#DFEDF3',[range(6)]),
          ('Family','One action per family',.43,'#DCEDE7',[range(0,2),range(2,4),range(4,6)]),
          ('Exact','One action per target',.13,'#F4E2D8',[[i] for i in range(6)])]
    token_rows=[]
    for name,definition,y,fill,groups in rows:
        ax.text(0,y+.18,name,fontweight='bold',fontsize=9)
        ax.text(.25,y+.18,definition,fontsize=7.9)
        for group in groups:
            inds=list(group)
            x=.02+min(inds)*.163
            width=len(inds)*.163-.015
            ax.add_patch(Rectangle((x,y),width,.13,facecolor=fill,
                edgecolor=COLORS[name],lw=1))
            for idx in inds:
                ax.text(.02+(idx+.5)*.163,y+.065,f'T{idx+1}',ha='center',va='center',fontsize=8)
                token_rows.append({'panel':'A','resolution':name,'illustrative_target':f'T{idx+1}',
                    'illustrative_block':groups.index(group)+1,'semantic':'generic partition token; not a dataset target'})
    ax.text(.5,.015,'Coarser pooling → finer target conditioning',ha='center',fontsize=7.5)

    ax=axs[0,1]
    panel(ax,'B','Oracle gain versus selection cost',x=-.12,y=1.09)
    ax.text(.5,.94,'Gaussian estimated-risk model',ha='center',fontsize=7.8,color='#687783')
    ax.text(.5,.77,r'$\bar R_n(\mathcal{P})=R^{*}(\mathcal{P})+E_n(\mathcal{P})$',
        fontsize=12.0,ha='center')
    ax.text(.23,.61,'Oracle gain',ha='center',fontsize=9,fontweight='bold')
    ax.text(.75,.61,'Selection-cost change',ha='center',fontsize=8.5)
    ax.text(.5,.33,r'$\Delta R^{*}>\Delta E_n$',ha='center',fontsize=18)
    ax.text(.5,.13,'Refine when oracle gain exceeds\nthe selection-cost change.',ha='center',va='center',fontsize=8)

    ax=axs[1,0]
    panel(ax,'C','Possible learned-risk profiles',x=-.12,y=1.09)
    ax.text(.5,.96,'SCHEMATIC — no empirical values',ha='center',fontsize=7.5,color='#687783')
    ax.text(-.09,.60,'Expected learned risk\n(schematic)',rotation=90,
        ha='center',va='center',fontsize=7.5)
    resolutions=['Shared','Family','Exact']
    x=np.arange(3)
    profiles=[('Monotonic improvement',[.75,.54,.29],'#303B49','o'),
              ('Intermediate minimum',[.60,.24,.41],'#788999','s'),
              ('Non-monotone ordering',[.28,.66,.50],'#A5AFB8','^')]
    coordinates=[]
    child=ax.inset_axes([.10,.36,.85,.46])
    child.set(xlim=(-.18,2.18),ylim=(.10,.85),xticks=x,xticklabels=resolutions,yticks=[])
    child.spines[['left','bottom']].set_visible(False)
    child.tick_params(axis='x',length=0,labelsize=7.5,pad=3)
    for label,y,color,marker in profiles:
        child.plot(x,y,color=color,lw=1.0,marker=marker,ms=3.7,label=label)
        for resolution,xp,yp in zip(resolutions,x,y):
            coordinates.append({'profile':label,'resolution':resolution,'display_x':float(xp),
                'display_y':float(yp),'role':'arbitrary schematic geometry; NOT empirical data'})
    child.set_xlabel('Target resolution',fontsize=8.5,labelpad=4)
    child.legend(loc='upper center',bbox_to_anchor=(.5,-.53),frameon=False,
        fontsize=7.5,ncol=1,handlelength=1.8,labelspacing=.28,borderaxespad=0)

    ax=axs[1,1]
    panel(ax,'D','Nested biological-unit evaluation',x=-.12,y=1.09)
    box(ax,(.01,.76),.53,.16,'Training biological units',face='#ECF3F6',edge=COLORS['Shared'])
    box(ax,(.65,.76),.34,.16,'Whole biological-\nunit holdout',face='white',edge='#778691',fontsize=7.5)
    box(ax,(.01,.47),.53,.17,'Select action + resolution\nby inner validation',face='#ECF3F6',edge=COLORS['Shared'],fontsize=7.8)
    box(ax,(.01,.16),.53,.17,'Refit selected policy',face='#ECF3F6',edge=COLORS['Shared'])
    box(ax,(.65,.16),.34,.17,'Evaluate once\non unseen unit',face='white',edge=INK,fontsize=7.8)
    arrow(ax,(.275,.75),(.275,.65))
    arrow(ax,(.275,.46),(.275,.34))
    arrow(ax,(.55,.245),(.64,.245))
    arrow(ax,(.82,.75),(.82,.34),linestyle=(0,(3,2)))
    ax.text(.81,.53,'Never used\nfor selection',ha='center',va='center',fontsize=7.5,
        bbox={'facecolor':'white','edgecolor':'none','pad':2})

    with (INPUT/'fig1_schematic_profiles.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(coordinates[0]));w.writeheader();w.writerows(coordinates)
    with (INPUT/'fig1_partition_tokens.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(token_rows[0]));w.writeheader();w.writerows(token_rows)
    content={'all_panels':'schematic; no empirical uncertainty or calibrated risk axes',
        'A':'Definitions in owner-confirmed P2 V3 Main and explicit rebuild contract',
        'B':'canonical_theory.md and canonical_master.csv EV-0001/EV-0002',
        'C':'Three arbitrary discrete profiles at Shared/Family/Exact; straight segments guide the eye, no continuous resolution theorem, no empirical values or benchmark reconstruction',
        'D':'Nested evaluation methods; outer unit excluded from all selection and fitting',
        'policy':'curve coordinates are layout-only inputs; never scientific observations'}
    (INPUT/'fig1_content_contract.json').write_text(json.dumps(content,indent=2),encoding='utf-8')
    save(fig,1)
    print('Figure 1 PNG/PDF/SVG rebuilt; all curve axes schematic; no historical assets read.')

if __name__=='__main__':main()
