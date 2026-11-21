"""Saved-result N1 report; all requested sensitivities and nulls retained."""
import json,math,statistics,hashlib
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;O=H/'results'
def read(name):return json.loads((O/name).read_text(encoding='utf-8'))
def ci(v):
    v=np.asarray(v,float);parts=[v[s::176] for s in range(176)];m=statistics.mean([statistics.mean(a) for a in parts]);h=1.96*math.sqrt(math.fsum(statistics.variance(a)/len(a) for a in parts))/176
    return {'mean':m,'ci95':[m-h,m+h],'n_decisions':len(v)}
def fmt(s):return f"{s['mean']:.6f} [{s['ci95'][0]:.6f}, {s['ci95'][1]:.6f}]"
def main():
    r=read('summary.json');s=read('sparsity_diagnostics.json');v=read('separate_verification.json');p=read('fresh_prefix_sampler_replay.json');models=read('frozen_models.json')
    losses=r['raw_decision_losses'];contrasts={a+'-'+b:ci(np.array(losses[a])-np.array(losses[b])) for a,b in [('net_full','net_bids'),('net_full','prop'),('net_full','uniform'),('net_bids','bi')]}
    (O/'reported_secondary_contrasts.json').write_text(json.dumps({'contrasts':contrasts,'n_decisions':3520,'method':'paired equal176stratum normal95%CI; fixed five-fit ensembles','report_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    lines=['# N1 completed — self-reviewed (Codex), pending external result audit','',
       'The fixed source was Claude-approved and committed as4c610703 before any data. The completed **results** have separate Codex raw/source/forward/selection/CI and manual prefix/sampler checks; no external N1result verdict is claimed. New results remain local and uncommitted pending later external review. No PyTorch, frozen source/data/seed edits or tuning.',
       '', '## Primary and design','',
       'Four holders include undealt stock. Every unseen card is scored; no oracle filter. One independent round/decision per record, hero/observer g%4,44balanced n3..13 × S/NT × TA/TA+ cells and four observer seats. Counts30800training/4400validation/3520confirmation, plus176resource pilot. Namespaced2100Mtrain/validation/pilot and2110Mconfirmation; exact entropy addresses in PROTOCOL/STATUS and seed_audit. Five paired fit seeds per feature arm,50epochs, frozen epoch/blend choices before confirmation.',
       '', 'Primary full-history ensemble minus K256BI unconditional log-loss/decision: **'+fmt(r['primary_net_minus_BI'])+' nats**, n=3520, paired equal176stratum normal95%CI. Negative is better. This is a fixed-ensemble prediction comparison on the specified synthetic policy/deal distribution, not a posterior exactness, arbitrary-hand, human or game-score guarantee. Every interval below is conditional on the selected fixed five-fit ensemble; it does not quantify refitting uncertainty.',
       '', '|method|mean unconditional nats/decision [95%CI]|n independent rounds|','|---|---:|---:|']
    for name,q in r['unconditional'].items():lines.append(f"|{name}|{fmt(q)}|{q['n_decisions']}|")
    lines+=['','Bids-only retains structural capacities/voids/public card removal; full adds the declared behavioral-history summaries. Full is not a recurrent/injective full-history encoder. Both were selected on the same fixed validation distribution; compare whole trained pipelines, not Bayes information value.','', '|secondary paired contrast|difference [95%CI]|n|','|---|---:|---:|']
    for name,q in contrasts.items():lines.append(f"|{name}|{fmt(q)}|3520|")
    lines+=['','## Predeclared Monte Carlo sparsity sensitivity','',
       'The declared primary floors all finite-sampler probabilities with .001prop. Its effective pseudocount is256*.001/.999. Additional **pre-confirmation** analyses use q_s=(count+a*prop)/(256+a), a1/4/16, with the same paired stratum CI. They are secondary; the primary is unchanged. No new deals, training or tuning.',
       '', '|sampler|smoothing|net_full minus sampler [95%CI]|n|','|---|---|---:|---:|']
    robust=[]
    for method,rows in s['sensitivity'].items():
        for a,q in rows.items():
            x=q['net_minus_sampler'];lines.append(f"|{method}|{a}|{fmt(x)}|3520|")
            if method=='bi' and a!='declared_floor':robust.append(x['ci95'][1]<0)
    conclusion='The nominal advantage survives all three predeclared BI smoothing settings.' if all(robust) else ('The advantage does not survive every BI smoothing setting; finite-K sparsity/shrinkage materially affects interpretation. No general learned-posterior superiority is promoted.')
    lines+=['',conclusion,'', '|sampler|true-holder zero count, round-equal fraction [95%CI]|pooled-card fraction [round-cluster95%CI]|n rounds|n card labels|','|---|---:|---:|---:|---:|']
    for method,q in s['zero_count'].items():lines.append(f"|{method}|{fmt(q['round_equal_fraction'])}|{fmt(q['pooled_card_fraction'])}|3520|{q['pooled_card_fraction']['n_cards']}|")
    lines+=['','Zero fractions describe the **unfloored K256sampler counts**; they do not assert zero true posterior probability. Pooled-card estimates use a round-cluster delta method, not card independence. Normal intervals are approximate; no equality is inferred from a null.',
       '', '## Calibration and conditional diagnostics','', '|method|ECE [stratified bootstrap95%CI]|conditional-active card log-loss [95%CI]|n rounds|','|---|---:|---:|---:|']
    for name,q in r['calibration'].items():
        e={'mean':q['ECE'],'ci95':q['ci95']};lines.append(f"|{name}|{fmt(e)}|{fmt(r['conditional_active'][name])}|3520|")
    lines+=['','Conditional-active scoring uses truth **only for this labelled diagnostic** and renormalizes every method over opponents identically. Primary inputs/scoring remain all unseen cards/four holders. Reliability-bin masses/confidence/accuracy CIs and supporting round/card counts are in sparsity_diagnostics.json; ten fixed bins,2000paired stratified round resamples. Empty bins have no estimate, sparse-bin ratios use nonempty-resample intervals and are exploratory; intervals are marginal, not simultaneous.',
       '', '## BI resource/degeneracy by n','', '|n|ESS mean [95%CI]|M mean [95%CI]|n rounds|','|---|---:|---:|---:|']
    for size,q in s['BI_by_n'].items():lines.append(f"|{size}|{fmt(q['ess']['mean'])}|{fmt(q['M']['mean'])}|320|")
    lines+=['','Full ESS/M observed ranges and quartiles with2000stratified bootstrap95%CIs are retained in sparsity_diagnostics.json. Observed min/max are exact log-sample extrema, not population bounds. New secondary bootstrap namespace91was declared before confirmation and collision-checked against123544known roots; ECE/bin diagnostics replay the predeclared stream90.',
       '', '## All predeclared cells (marginal intervals)','', '|cell|net_full minus BI nats/decision [95%CI]|n rounds|','|---|---:|---:|']
    for name,q in r['per_cell_net_minus_BI'].items():lines.append(f"|{name}|{fmt(q)}|{q['n_decisions']}|")
    lines+=['','All44cells shown; no selected-cell or multiplicity-adjusted superiority claim. No extension of confirmation, source/model changes or threshold/seed retuning.',
       '', '## Separate validation and limitations','',
       f"Independent numeric/raw-label audit covers{v['raw_label_rounds']['train']}training,{v['raw_label_rounds']['validation']}validation,{v['raw_label_rounds']['confirmation']}confirmation rounds and both model selections; primary CI reproduced exactly. Manual replay checks{p['n_manual_prefix_replays']}prefixes without RoundState/make_rngs/deal, plus{p['n_sampler_replay_decisions']}K256uniform/BIdecisions; all saved marginals/stats match. The frozen bidder/play functions and sampler algorithms are reused as definitions, not independently reimplemented. This is not an exact posterior certificate. All source/model/raw manifests and original results are retained.",
       '', 'Proper four-class prediction loss can improve without improving search or score. Sampling budgets, smoothing and the known fixed table type matter. Observational histories come from controlled TA/TA+ tables with canonical bidders, not humans or arbitrary opponents. The protocol and source preceded N1data; additional diagnostics were declared before confirmation. Separate diagnostic modules are post-source-freeze additions and have their own hashes; original approved source stays untouched. New tests/verification and later external audit are separate gates. No publication/push.']
    (H/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('PRIMARY',fmt(r['primary_net_minus_BI']),'n3520')
    for a,q in s['sensitivity']['bi'].items():print('BI',a,fmt(q['net_minus_sampler']))
    print('full-minus-bids',fmt(contrasts['net_full-net_bids']));print(conclusion)
if __name__=='__main__':main()
