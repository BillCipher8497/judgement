"""All-cells report after separate audit; no winning-cell selection."""
from n9lib import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def fmt(c):return f"{c['mean']:.8f} [{c['ci95'][0]:.8f}, {c['ci95'][1]:.8f}]"
def main():
    e=json.loads((O/'evaluation.json').read_text());v=json.loads((O/'verification.json').read_text());assert v['n_selected_models']==240 and v['all_independent_forward_DP_normalizers_selection_splits_raw_and_CIs_pass']
    lines=['# N9 — matched normalized-joint versus marginal training','', '**Codex separately checked, pending Claude external; local/uncommitted.**', '',
    'PRIMARY largeNT/summaryMLP/80:joint−marginal held jointKL '+fmt(e['primary'])+' nats; n5paired optimizer seeds,t95%df4. Negative favors joint. This is a KNOWN toy-hand benchmark/deterministic key subset, not new-hand replication or a generalization guarantee. Exact support sums use floating logs, not rational certificates.','',
    'All120trajectories/240selected20-or80recipes frozen before held evaluation. Three configurations with245input columns:summarylinear,summaryMLP,richMLP. Summaryzero-pads beyond unchanged35features; richadds strict own/public fullcontext/card/holder features. MLPnominal7905coefficients inboth; summaryhas1185input-active coefficients before accounting for normalization gauges. Linear246nominal/36input-active. No optimal NNclass/factor-family projection floor solved.','',
    'Marginal and joint losses share data,per-key1/Uscaling,reachweights,initialization,batches,Adam/L2and JOINTvalidation selection. New recipe omits oldN2mixture/alpha inBOTHarms. Short/longselectedcheckpoints share trainingprefix;80budget also adds selectioncandidates, not pure training-duration causality. Richinput changes which coefficients receive data signal, not a certified information-only mechanism. No unique cause of old N2failure inferred.','',
    '| condition | split | selectedkeys | reachedownhands | exactreachweight |','|---|---|---:|---:|---:|']
    for tag in TAGS:
        m=json.loads((O/tag/'data.json').read_text())
        for s,d in m['selection'].items():lines.append(f"| {tag} | {s}(0train/1val/2test) | {d['n_keys']} | {d['n_hands']} | {d['reach']} |")
    lines+=['','CIs below are paired n5optimizer t95%df4 ONLY, not chance/hand/population intervals. Every per-fit mean covers the full selected exact reach support; no world-IID samplingCI. All4conditions are not independent replications. Allsecondary cells descriptive/unadjusted; no selected winner. Null/adverse cells kept.','']
    for title,group in (('Joint minus marginal:held jointKL','joint_minus_marginal'),('Long80 minus short20:selected-pipeline effect','long_minus_short'),('RichMLP minus summaryMLP:matched loss/budget','rich_minus_summary')):
        lines += ['## '+title,'','| fixedcell | mean and95%CI(n5pairedseeds) |','|---|---|']
        lines += [f'| {key} | {fmt(c)} |' for key,c in e[group].items()];lines+=['']
    lines+=['## Independent checks and execution','',
    'Separate checker reconstructs the hand-balanced SHAselector and new210rich feature columns, exact structuralFsets/allRmemberships/targets, disjoint own-hand splits, originalBIreference values, all240selected forward factors/heldKLs using independent DPnormalizers, allcandidate validation costs/earliest-tie selection and112reportedindividual/paired intervals. Fresh policy-reach reenumeration covers ALLselected small keys and32fixed large keys/condition; remaininglargeRrelies on the unchanged previously audited N2posterior/hash, not a fresh complete reimplementation of all2million keys. Same-investigator check, not external approval.','',
    'Preflight known-state collision0,3241roots plus5children,2170M/domain0x4e39; gradientFDmax1.398e-11. Firstdata export failed before any row/fit because the frozenencoder returns4values but caller unpacked5. Originalscientificsourcefreeze/selectionUTC retained; newapi_resume.py adds only an unusedNone slot, public-array identity fixture/hashguard. No existing science source changed. First3hwindow stopped after hostclockgap with40fits; predeclaredwindow2resumed fixedcounts/checkpoints. Timings are not equal-compute/efficiency results. Allerrors/prefixes/states retained.','',
    'Rawper-contextKL/logits,models/candidates/epochstates,allper-hand means and BIsecondary differences are in results/. Original N2data/models/sources/2100Mseeds untouched. No52card/game-score/fusion/optimalposterior/novelty claim. Broader literature review incomplete; globalnormalization is established structuredprediction (RELATED_WORK.md).','',
    'Commands(cd engine): preflight.py,self_check.py; recorded bounded api_resume.py(samefrozen run.py), then separately bounded verify_v2.py,report.py. The first separate reader stopped on uint8 multiplication by256; verify_v2 casts both bytes before decoding. Failed reader/log retained; fitted data/models unchanged. Exact absolute stops in STATUS/launches. No commit/push before Claude validation.']
    (H/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,budget in zip(axs,(20,80)):
        for k,cfg in enumerate(CONFIGS):
            y=np.arange(4)+(k-1)*.18;c=[e['joint_minus_marginal'][f'{tag}/{cfg}/{budget}'] for tag in TAGS];means=np.array([v['mean'] for v in c]);lo=np.array([v['ci95'][0] for v in c]);hi=np.array([v['ci95'][1] for v in c]);ax.errorbar(means,y,xerr=np.array([means-lo,hi-means]),fmt='o',label=cfg)
        ax.axvline(0,c='gray',lw=1);ax.set_yticks(range(4),['smallS','smallNT','largeS','largeNT']);ax.set_xlabel('JointCE − marginalCE:held jointKL(nats)');ax.set_title(f'Selectionbudget{budget}epochs');ax.grid(alpha=.2)
    axs[0].legend(fontsize=8);fig.suptitle('Known hand benchmark; n5paired optimizerseeds,t95%CIs ONLY\nMatchedrecipe,not population/generalNN guarantee;pending external audit',fontsize=11)
    for ext in ('png','pdf'):fig.savefig(O/f'objective_controls.{ext}',dpi=180)
    plt.close(fig)
    save(O/'report_manifest.json',{'utc':stamp(),'evaluation_sha256':sha(O/'evaluation.json'),'verification_sha256':sha(O/'verification.json'),'report_sha256':sha(H/'RESULTS.md'),'figure_sha256':sha(O/'objective_controls.png'),'source_sha256':sha(Path(__file__))})
    log(H,'READY-FOR-REVIEW: N9newprotocol/source+adapter/preflight/caches/all120trajectories/240selectedmodels/rawlogits/KLs/verification/RESULTS/plots. Commands recordedpreflight,self_check,bounded api_resume.py,verify_v2.py,report.py.2170M/domain0x4e39,n5pairedoptimizer t95%CIs; primary '+json.dumps(e['primary'])+'. Allnull/adversecells retained; exact finite knownsubset/nochanceCI,feature/capacity/duration/selection limits stated; nooldsource/core/policy/seed changes, no game/causalundertraining/noveltyclaim. Separately self-checkedCodex,pendingClaudeexternal/local,no commit/push.')
if __name__=='__main__':main()
