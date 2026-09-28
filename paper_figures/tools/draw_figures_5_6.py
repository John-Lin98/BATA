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
fig=plt.figure(figsize=(5.5,2.5))
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
    ax.text(j,1.65,obj,ha='center',va='bottom',fontsize=8,fontweight='bold' if obj=='T-DCG-96' else 'normal',clip_on=False)
    for y,key in [(1,'Final_mean_rank'),(0,'Query_AUC_mean_rank')]:
        val=float(row[key]); color=plt.cm.Blues(.10+.37*(4-val)/3)
        ax.add_patch(Rectangle((j-.48,y-.44),.96,.88,facecolor=color,edgecolor='white',lw=.7))
        ax.text(j,y,'%.2f'%val,ha='center',va='center',fontsize=8.5,fontweight='bold' if obj=='T-DCG-96' else 'normal')
ax.text(-.57,1,'Final@480',ha='right',va='center',fontsize=8.5)
ax.text(-.57,0,'Query-AUC',ha='right',va='center',fontsize=8.5)
save(fig,'Figure5')

# Figure 6: full-width, independent main-text configuration comparison.
fig=plt.figure(figsize=(5.5,2.2));ax=fig.add_axes([.23,.25,.74,.63]);clean(ax)
ax.set_xlim(-.325,.083);ax.set_xticks([-.3,-.2,-.1,0,.05]);ax.set_ylim(-.6,5.4)
ax.axvline(0,color='#626B73',ls='--',lw=.8)
order=[('GB1',4.8,35),('PABP',3.8,35),('HIS7',2.8,24),('GRB2',1.8,24),('TrpB',.1,70)]
ax.axhspan(-.4,.6,color='#F1EDF5',zorder=0);ax.axhline(1.05,color='#D4D8DC',lw=.6)
for task,y,n in order:
    row=next(r for r in FS if r['benchmark']==task)
    ci(ax,row,y,'Figure6',FILENAMES[3],COLORS[task],'D' if task=='TrpB' else 'o')
ax.set_yticks([x[1] for x in order]);ax.set_yticklabels([t+' (n='+str(n)+')' if t!='TrpB' else 'TrpB (n=70)\nsensitivity' for t,y,n in order])
ax.set_xlabel('Fine-only M20 − BATA: Δ Final@480',labelpad=5)
fig.text(.025,.974,'Stronger task-specific predictor',fontsize=9,fontweight='bold',va='top')
save(fig,'Figure6')

# Independent appendix capacity figure: task-specific axes, no pooled effect scale.
fig=plt.figure(figsize=(5.5,2.0));fig.text(.025,.96,'Matched task-specific predictor capacity',fontsize=9,fontweight='bold',va='top')
lefts=[.25,.50,.75];width=.225
cbounds={'GB1':(-.080,.082),'PABP':(-.25,.37),'TrpB':(-.008,.080)}
cticks={'GB1':[-.05,0,.05],'PABP':[-.2,0,.2],'TrpB':[0,.03,.06]}
for task,left in zip(TASKS,lefts):
    ax=fig.add_axes([left,.265,width,.50]);clean(ax)
    ax.set_xlim(*cbounds[task]);ax.set_xticks(cticks[task]);ax.set_ylim(-.65,1.65)
    ax.axvline(0,color='#626B73',ls='--',lw=.8)
    ax.set_yticks([1,0]);ax.set_yticklabels(['Task-specific-only','Equal-rank'] if task=='GB1' else ['', ''])
    ax.set_title(task+' (n=24)',color=COLORS[task],pad=5)
    for j,alt in enumerate(['c0_fine_only','c22_equal_rank']):
        row=next(r for r in CT if r['benchmark']==task and r['alternative']==alt)
        ci(ax,row,1-j,'Appendix_Capacity',FILENAMES[5],COLORS[task],'o' if j==0 else 's')
fig.text(.61,.055,'BATA − control: Δ Final@480',ha='center',fontsize=8.5)
save(fig,'Appendix_Capacity')

# All 24 paired groups/task, unchanged order and raw Final@480 values.
fig=plt.figure(figsize=(5.5,2.45))
lefts=[.09,.407,.724];width=.255
for task,left in zip(TASKS,lefts):
    ax=fig.add_axes([left,.315,width,.535]);clean(ax,'y')
    mat=PIVOTS[task].values
    for values in mat:
        ax.plot(range(4),values,color=COLORS[task],alpha=.19,lw=.55,marker='o',ms=1.8,mew=0,zorder=1)
    mean=mat.mean(axis=0)
    ax.plot(range(4),mean,color=COLORS[task],lw=1.4,marker='s',ms=4.0,mew=.5,zorder=4)
    ax.set_xlim(-.22,3.22)
    span=mat.max()-mat.min();ax.set_ylim(mat.min()-.045*span,mat.max()+.075*span)
    ax.set_xticks(range(4));ax.set_xticklabels(OBJS,rotation=42,ha='right',fontsize=8)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.set_title(task+' (n=24)',color=COLORS[task],pad=5)
    if task=='GB1':ax.set_ylabel('Final@480',labelpad=4)
handles=[Line2D([0],[0],color='#93999F',lw=.7,marker='o',markersize=2,label='Matched group'),Line2D([0],[0],color='#48525A',lw=1.4,marker='s',markersize=4,label='Mean')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.56,.01),ncol=2,frameon=False,handlelength=1.8,columnspacing=1.8)
save(fig,'Appendix_Objectives')

numeric={'source_files':{n:sha(DATA/n) for n in FILENAMES},'font':{'family':'Times New Roman','file':font_path},
 'ci_checks':AUDIT,'mean_regression_checks':MEAN_CHECKS,'outputs':OUTPUTS,
 'rank_matrix':[{'objective':r['objective'],'Final_mean_rank':float(r['Final_mean_rank']),'Query_AUC_mean_rank':float(r['Query_AUC_mean_rank']),'cohort':r['cohort'],'n_per_task':int(r['n_per_task'])} for r in RANK],
 'objective_raw_groups':{t:{str(g):[float(v) for v in PIVOTS[t].loc[g].values] for g in PIVOTS[t].index} for t in TASKS},
 'objective_column_order':OBJS,'no_recomputed_bootstrap':True,'native_units_not_pooled':True}
with (QA/'figures_5_6_source_numeric.json').open('w',encoding='utf-8') as f:json.dump(numeric,f,ensure_ascii=False,indent=2)

lines=['# Figure 5/6 与两个附录图重绘记录','',
'## Before / After','',
'已实际查看 baseline/visual/page-08.png、page-09.png、page-18.png、page-22.png。未从PDF估算任何数据。',
'', '| 图 | Before | After |', '|---|---|---|',
'| Figure5 | 9个效应竖排，共享宽范围；右侧rank表占高较大；原图高3.18in。 | 3个task各用原生fitness效应轴、保留全部9个contrast；底部2×4rank矩阵；5.5×2.5in。 |',
'| Figure6 | 从双panel文件裁左半，主文仅0.46linewidth，左右留白多、字号偏小。 | 独立5.5×2.2in全宽forest；35/35/24/24/70逐行标出，TrpB sensitivity分隔。 |',
'| Appendix capacity | 从双panel裁右半，重复label与窄图限制可读性。 | 独立5.5×2.0in，3任务原生单位facet；两个对照名称只在共享y轴说明。 |',
'| Appendix objective groups | 窄panel、7.1pt斜标签和额外脚注。 | 独立5.5×2.45in，全部24组轨迹和均值保留，canonical objective名称≥8pt、正常两项图例。 |','',
'## 数据与科学不变量','',
'- Figure5：Figure5_paired_effects.csv全部9行；Figure5_objective_ranks.csv全部4×2个rank；各任务native units分开显示，不pool原始fitness。',
'- Figure6：Figure6_configuration_effects.csv全部5行；Figure6_configuration_pairs.csv的188 matched pairs核对均值。GB1/PABP=35，HIS7/GRB2=24，TrpB independent sensitivity=70。方向Fine-only M20−BATA。',
'- Appendix capacity：Figure6_same_capacity_effects.csv全部6行；144条paired contrast来自每任务每对照24对；方向BATA−control。没有把same-capacity24与TrpB70合并。',
'- Objective groups：Figure5_objective_per_run.csv共288 runs，3任务×24组×4objectives；没有删组、按outcome重排或从PDF读数。',
'- 原CI直接读取冻结CSV，没有重新抽bootstrap或选择更有利CI。TrpB same-capacity task-specific-only的既定20,000-resample CI下界-0.0006083352999999793、上界0.07160071914895835，仍跨0。',
'- 所有20个paired mean由源pairs/group表重新求均值回归，误差<1e-12；这是绘图一致性检查，不是新科学实验。',
'', '## 全部CI显示范围核验','', '| 图 | task / comparison | n | mean | 95% CI | 轴范围 | 跨0 |', '|---|---|---:|---:|---|---|---|']
for q in AUDIT:
 lines.append('| {figure} | {benchmark} / {comparison} | {n} | {mean:.10f} | [{ci95_low:.12f}, {ci95_high:.12f}] | [{axis_low:.4f}, {axis_high:.4f}] | {z} |'.format(z='是' if q['contains_zero'] else '否',**q))
lines+=['','全部CI端点严格位于各自轴范围内，未截断；原始字符串与full precision值保存在 `qa/figures_5_6_source_numeric.json`。','',
'## 排版与输出','',
'- Times New Roman，字体真实路径：`'+font_path+'`。',
'- 所有字体≥8pt；固定5.5in画布；PDF矢量+PNG300dpi；未用bbox=tight改变画布尺寸。',
'- 黑灰虚线只标0；task颜色固定GB1 #3979A5、PABP #BA7C3E、TrpB #786398；不同CI/组合标记增加区分，不仅依靠颜色。',
'- 本脚本没有改manuscript、上传Overleaf、改caption或运行新实验。','',
'## Caption和引用建议（供主任务整合，不自动改稿）','',
'1. Figure5的(a)仍是全部9个paired effects，现为3个task facet，各自native x-axis；(b)仍是两指标的mean task ranks。现caption内容基本保留，只可将(a)说明补成“shown on separate task-specific axes”。',
'2. Figure6独立只有一个panel，正文 Figure6a 改为 Figure6；删任何旧panel(a)定位，保留配置比较三因素和TrpB70独立cohort解释。PDF以width=linewidth无trim直接插入。',
'3. Capacity独立图无(b)panel编号，沿用Appendix fig:capacity_controls标签，caption明确95% paired CI、24 matched initializations、positive values favor BATA；不要称该对照为Fine-only M20。',
'4. Objective groups沿用完整24组matched group图注；light lines=matched groups、square mean已由简短图例说明，可保留caption原解释。','',
'## 检查状态','',
'已完成：CSV/paired-mean/CI边界/字体下限/图内文字画布边界的程序核验。待脚本运行后的实际PNG目视复核，并由主任务做最终LaTeX整合与9页验收。','']
with (QA/'figures_5_6_design.md').open('w',encoding='utf-8') as f:f.write('\n'.join(lines))
print(json.dumps({'outputs':OUTPUTS,'ci_checks':len(AUDIT),'paired_mean_checks':len(MEAN_CHECKS),'font':font_path},indent=2))
