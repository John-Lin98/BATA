# -*- coding: utf-8 -*-
"""Redraw four BATA figures from verified CSVs; compatible with matplotlib 3.3.4.
No bootstrap is recomputed. CI endpoints are read from the frozen source tables.
"""
from pathlib import Path
import csv, json, hashlib
import numpy as np
import pandas as pd
# Pillow's type annotations expect numpy.typing.NDArray (NumPy >=1.21).
# Supply only this annotation alias in-process for the pinned NumPy 1.20 runtime.
import numpy.typing as npt
if not hasattr(npt, 'NDArray'):
    class _NDArrayAnnotation:
        def __class_getitem__(cls, item):
            return np.ndarray
    npt.NDArray = _NDArrayAnnotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'generated'
QA = ROOT / 'qa'
OUT.mkdir(exist_ok=True)
QA.mkdir(exist_ok=True)
TASKS = ['GB1', 'PABP', 'TrpB']
COLORS = {'GB1':'#3979A5','PABP':'#BA7C3E','TrpB':'#786398','HIS7':'#5D7676','GRB2':'#72777D'}
OBJS = ['G-Rank','T-Uniform-96','T-DCG-96','T-DCG-192']
ALTS = ['G-Rank','T-Uniform-96','T-DCG-192']
font_path = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'), fallback_to_default=False)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
 'font.size':8.5,'axes.titlesize':9,'axes.titleweight':'bold','axes.labelsize':8.5,
 'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
 'axes.linewidth':.65,'xtick.major.width':.6,'ytick.major.width':.6,
 'xtick.major.size':2.5,'ytick.major.size':2.5,'pdf.fonttype':42,'ps.fonttype':42,
 'mathtext.fontset':'stix','axes.unicode_minus':True,'savefig.dpi':300,
 'figure.facecolor':'white','axes.facecolor':'white'})

FILENAMES=['Figure5_paired_effects.csv','Figure5_objective_ranks.csv','Figure5_objective_per_run.csv',
 'Figure6_configuration_effects.csv','Figure6_configuration_pairs.csv',
 'Figure6_same_capacity_effects.csv','Figure6_same_capacity_pairs.csv']
RAW={}
for name in FILENAMES:
    with (DATA/name).open(encoding='utf-8-sig',newline='') as f:
        RAW[name]=list(csv.DictReader(f))
EF=RAW[FILENAMES[0]]; RANK=RAW[FILENAMES[1]]; OP=pd.read_csv(DATA/FILENAMES[2])
FS=RAW[FILENAMES[3]]; FP=pd.read_csv(DATA/FILENAMES[4])
CT=RAW[FILENAMES[5]]; CP=pd.read_csv(DATA/FILENAMES[6])
AUDIT=[]; OUTPUTS=[]; MEAN_CHECKS=[]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def record_mean(name,task,comparison,expected,observed):
    err=abs(float(expected)-float(observed)); assert err<1e-12,(name,task,comparison,err)
    MEAN_CHECKS.append({'source':name,'task':task,'comparison':comparison,'source_mean':float(expected),'recomputed_mean':float(observed),'absolute_error':err})

assert len(EF)==9 and len(RANK)==4 and len(OP)==288
assert len(FS)==5 and len(FP)==188 and len(CT)==6 and len(CP)==144
assert set(OP['group'])==set(range(24)) and (OP['unique_queries']==480).all()
PIVOTS={}
for task in TASKS:
    p=OP[OP.benchmark==task].pivot(index='group',columns='objective',values='Final').reindex(columns=OBJS)
    assert p.shape==(24,4) and not p.isnull().any().any()
    PIVOTS[task]=p
    for alt in ALTS:
        row=next(r for r in EF if r['benchmark']==task and r['rhs']==alt)
        assert row['cohort']=='matched_objective24' and int(row['n_pairs'])==24
        record_mean(FILENAMES[0],task,alt,row['paired_mean_difference'],(p['T-DCG-96']-p[alt]).mean())
for r in FS:
    task=r['benchmark']; pairs=FP[FP.benchmark==task]
    assert len(pairs)==int(r['n']) and set(pairs.cohort)=={r['cohort']}
    assert np.allclose(pairs.difference,pairs.fine_Final-pairs.bata_Final,rtol=0,atol=1e-12)
    record_mean(FILENAMES[3],task,'Fine-only M20 minus BATA',r['mean_difference'],pairs.difference.mean())
for r in CT:
    pairs=CP[(CP.benchmark==r['benchmark'])&(CP.alternative==r['alternative'])]
    assert len(pairs)==24 and set(pairs.cohort)=={'same_capacity_ablation24'}
    assert r['direction']=='BATA minus control'
    assert np.allclose(pairs.difference,pairs.bata_Final-pairs.control_Final,rtol=0,atol=1e-12)
    record_mean(FILENAMES[5],r['benchmark'],r['alternative'],r['mean_difference'],pairs.difference.mean())
trpb=next(r for r in CT if r['benchmark']=='TrpB' and r['alternative']=='c0_fine_only')
assert float(trpb['ci95_low'])<0<float(trpb['ci95_high'])
assert abs(float(trpb['ci95_low'])-(-.0006083352999999793))<1e-15


def clean(ax, grid='x'):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(grid!='x')
    ax.set_axisbelow(True); ax.grid(axis=grid,color='#E3E6E8',linewidth=.5)
    if grid=='x': ax.tick_params(axis='y',length=0,pad=5)


def ci(ax,row,y,figure,source,color,marker='o',mean_key='mean_difference'):
    mean=float(row[mean_key]); lo=float(row['ci95_low']); hi=float(row['ci95_high'])
    assert lo<=mean<=hi
    ax.errorbar(mean,y,xerr=np.array([[mean-lo],[hi-mean]]),fmt=marker,
      color=color,markersize=4.4,markeredgewidth=.7,elinewidth=1.1,capsize=2.8,zorder=3)
    low,high=ax.get_xlim(); assert low<lo and hi<high,(figure,lo,hi,low,high)
    AUDIT.append({'figure':figure,'csv':source,'benchmark':row['benchmark'],
      'comparison':row.get('rhs',row.get('alternative','Fine-only M20 minus BATA')),
      'cohort':row['cohort'],'n':int(row.get('n_pairs',row.get('n'))),
      'mean_source_text':row[mean_key],'ci95_low_source_text':row['ci95_low'],
      'ci95_high_source_text':row['ci95_high'],'mean':mean,'ci95_low':lo,'ci95_high':hi,
      'axis_low':low,'axis_high':high,'ci_in_axis':True,'contains_zero':lo<=0<=hi})


def save(fig,name):
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); bbox=fig.bbox
    text_sizes=[]; outside=[]
    for obj in fig.findobj(matplotlib.text.Text):
        if not obj.get_visible() or not obj.get_text():continue
        text_sizes.append(float(obj.get_fontsize()))
        bb=obj.get_window_extent(renderer)
        if bb.x0 < -1 or bb.x1 > bbox.width+1 or bb.y0 < -1 or bb.y1 > bbox.height+1:
            outside.append(obj.get_text())
    assert min(text_sizes)>=8,(name,min(text_sizes))
    assert not outside,(name,'text outside figure',outside)
    pdf=OUT/(name+'.pdf'); png=OUT/(name+'.png')
    fig.savefig(pdf,metadata={'Creator':'BATA figure refinement','CreationDate':None,'ModDate':None})
    fig.savefig(png,dpi=300)
    OUTPUTS.append({'name':name,'width_inches':float(fig.get_size_inches()[0]),
      'height_inches':float(fig.get_size_inches()[1]),'minimum_font_points':min(text_sizes),
      'pdf_sha256':sha(pdf),'png_sha256':sha(png),'text_outside_canvas':outside})
    plt.close(fig)


# Figure 5: separate native-unit axes for all nine effects; compact two-by-four rank matrix.
plt.rcParams.update({'font.size':9,'axes.titlesize':9.5,'xtick.labelsize':8.5,'ytick.labelsize':8.5})
fig=plt.figure(figsize=(5.5,3.3))
fig.text(.025,.972,'(a) Paired Final@480 differences',fontsize=9,fontweight='bold',va='top')
lefts=[.225,.485,.745]; width=.235
bounds={'GB1':(-.045,.075),'PABP':(-.145,.37),'TrpB':(-.009,.052)}
ticks={'GB1':[-.04,0,.04],'PABP':[-.1,0,.1,.2,.3],'TrpB':[0,.02,.04]}
for task,left in zip(TASKS,lefts):
    ax=fig.add_axes([left,.545,width,.285]);clean(ax)
    ax.set_xlim(*bounds[task]);ax.set_xticks(ticks[task]);ax.set_ylim(-.45,2.45)
    ax.axvline(0,color='#6A7178',ls='--',lw=.75)
    ax.set_yticks([2,1,0]);ax.set_yticklabels(ALTS if task=='GB1' else ['', '', ''])
    ax.set_title(task+' (n=24)',color=COLORS[task],pad=4)
    for j,alt in enumerate(ALTS):
        row=next(r for r in EF if r['benchmark']==task and r['rhs']==alt)
        ci(ax,row,2-j,'Figure5',FILENAMES[0],COLORS[task],['o','s','D'][j],'paired_mean_difference')
fig.text(.60,.40,'T-DCG-96 − alternative',ha='center',fontsize=8.5)
fig.text(.025,.335,'(b) Mean task ranks',fontsize=9,fontweight='bold',va='top')
ax=fig.add_axes([.225,.04,.755,.225]);ax.set_xlim(-.5,3.5);ax.set_ylim(-.5,1.5)
ax.axis('off')
for j,obj in enumerate(OBJS):
    row=next(r for r in RANK if r['objective']==obj)
    ax.text(j,1.65,obj,ha='center',va='bottom',fontsize=8.5,fontweight='bold' if obj=='T-DCG-96' else 'normal',clip_on=False)
    for y,key in [(1,'Final_mean_rank'),(0,'Query_AUC_mean_rank')]:
        val=float(row[key]); color=plt.cm.Blues(.10+.37*(4-val)/3)
        ax.add_patch(Rectangle((j-.48,y-.44),.96,.88,facecolor=color,edgecolor='white',lw=.7))
        ax.text(j,y,'%.2f'%val,ha='center',va='center',fontsize=8.5,fontweight='bold' if obj=='T-DCG-96' else 'normal')
ax.text(-.57,1,'Final@480',ha='right',va='center',fontsize=8.5)
ax.text(-.57,0,'Query-AUC',ha='right',va='center',fontsize=8.5)
save(fig,'Figure5_Layout')


(ROOT/'qa/figure5_layout_audit.json').write_text(json.dumps({'outputs':OUTPUTS},indent=2),encoding='utf-8')
