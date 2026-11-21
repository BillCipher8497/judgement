"""Static scientific figures from saved, separately checked results only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent;out=H/'figures';out.mkdir(exist_ok=True)
r=json.loads((R/'M3/results/held_out_results.json').read_text())
d=json.loads((R/'M3/results/label_decode_audit.json').read_text())['conditions']['S3R4n3_t-1']['held_out']
fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True,layout='constrained')
for ax,split in zip(axes,('discovery','confirmation')):
    labels=['Hard CE','Uniform CE','Set BCE']; keys=['hard_ce','uniform_ce','set_bce']; q=r['summaries'][f'S3R4n3_t-1/mlp/{split}']['objectives']
    means=[q[k]['mean'] for k in keys];half=[q[k]['half95'] for k in keys]
    ax.errorbar(range(3),means,yerr=half,fmt='o',capsize=4,label='Fitted MLP: 5 seeds, t95% CI')
    ax.scatter(range(3),[d[split]['label_excess'][k] for k in ('hard','uniform','set')],marker='x',s=65,label='Exact label decoder cost')
    ax.set_xticks(range(3),labels);ax.set_title(f"{split}: {d[split]['actual_own_hands']} reached hands\n{d[split]['n_worlds']:,} exact root worlds")
    ax.axhline(0,color='grey',lw=.6);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Avoidable hero cost above conditional J1')
axes[0].legend(fontsize=8)
fig.suptitle('Set supervision avoids label-induced cost\nLowest-index teacher/decoder ties; seed intervals exclude hand-sampling uncertainty',fontsize=11)
fig.savefig(out/'label_misalignment.png',dpi=200);fig.savefig(out/'label_misalignment.svg');plt.close(fig)
path=R/'M10/results/verified_summary.json'
if path.exists():
    m=json.loads(path.read_text());fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    contrasts=['R-L','R-N','R-canonical','L-canonical','N-canonical']
    for offset,(stage,values) in zip([-.12,.12],m['stages'].items()):
        stats=[values['contrasts'][c]['score'] for c in contrasts]
        ax.errorbar(np.arange(5)+offset,[s['mean'] for s in stats],yerr=[s['halfwidth'] for s in stats],fmt='o',capsize=4,label=f'{stage}: n=2000 paired games')
    ax.axhline(0,color='grey',lw=.8);ax.set_xticks(range(5),contrasts);ax.set_ylabel('Hero points per full game (difference)')
    ax.set_title('Matched fixed bidder families: paired seat-stratified 95% CIs\nR: MLP offset; L: linear offset; N: MLP without explicit offset')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'matched_controls.png',dpi=200);fig.savefig(out/'matched_controls.svg');plt.close(fig)
