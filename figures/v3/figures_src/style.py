"""Shared, code-native publication styling for the V3.0 rebuild."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path(__file__).resolve().parent / 'inputs'
COLORS = {'C0':'#ABB3BD', 'Shared':'#3D8EAD', 'Family':'#65AC99',
          'Exact':'#D58A68', 'Adaptive':'#303B49', 'Conventional':'#7A8190',
          'Nearest-rate':'#7A8190', 'Random':'#A7ABB4'}
INK = '#263440'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,
    'axes.labelsize':8.5,'axes.titlesize':9,'axes.titleweight':'bold',
    'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,
    'axes.linewidth':0.7,'lines.linewidth':1.2,'text.color':INK,
    'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
    'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path',
    'svg.hashsalt':'P2_V3_SOL_V3_0',
    'savefig.facecolor':'white','figure.facecolor':'white',
    'savefig.dpi':600})

def edge(color):
    return tuple(v * 0.70 for v in to_rgb(color))

def panel(ax, letter, title='', x=-0.16, y=1.09):
    ax.text(x,y,letter,transform=ax.transAxes,fontweight='bold',fontsize=11,
            ha='left',va='bottom',clip_on=False)
    if title:
        ax.set_title(title,loc='left',pad=10)

def clean(ax, grid=False):
    ax.tick_params(length=3,width=0.65)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis='y',color='#E9EDF0',lw=.5)

def save(fig, number):
    stem=ROOT / f'P2_V3_Figure{number}_V3_0'
    for ext in ('png','pdf','svg'):
        options={}
        if ext=='pdf':
            options['metadata']={'CreationDate':None,'ModDate':None}
        elif ext=='svg':
            options['metadata']={'Date':None}
        fig.savefig(stem.with_suffix('.'+ext),dpi=600,**options)
    plt.close(fig)
