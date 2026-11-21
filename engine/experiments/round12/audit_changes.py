"""Documentation/diagnostic corrections; no training arrays or frozen code writes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json, hashlib
from pathlib import Path
from fractions import Fraction
from datetime import datetime, timezone
import numpy as np
R=Path(__file__).resolve().parent; ROOT=R.parents[2]
if (R/'AUDIT_FIXES/APPLIED.json').exists():
    raise SystemExit('One-off correction record already applied. Reproduce diagnostics with M2/diagnose.py and M3/report_audited.py; do not reapply documentation patches.')
def append(p,s):
    with p.open('a',encoding='utf-8') as f:f.write('\n'+s+'\n')
def edit(p,a,b):
    t=p.read_text(encoding='utf-8'); assert a in t,(p,a);p.write_text(t.replace(a,b),encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat()
derived={'utc':stamp,'conditions':{},'scope':'Additional diagnostics; original scientific arrays/freezes untouched.'}
for folder in sorted((R/'M1/results').iterdir()):
    if not (folder/'manifest.json').is_file():continue
    m=json.loads((folder/'manifest.json').read_text());d=dict(np.load(folder/'root_data.npz'))
    obs=[json.loads(s)['observation'] for s in (folder/'private_contexts.jsonl').read_text().splitlines()]
    variants={'hard':d['hard_counts'],'uniform':d['uniform_optimal_units6'],'set':d['optimal_counts']}
    risks={}
    for name,v in variants.items():
        tied=(v==np.where(d['mask'],v,-1).max(1)[:,None])&d['mask']
        lo=np.where(tied,d['q_sum'],2**60).min(1);hi=np.where(tied,d['q_sum'],-1).max(1)
        risks[name]={'tied_group_count':int((tied.sum(1)>1).sum()),'best_excess':str(Fraction(int(lo.sum())+m['terminal_root_score_sum']-m['J1_sum'],m['n_exact_deals'])),
                     'worst_excess':str(Fraction(int(hi.sum())+m['terminal_root_score_sum']-m['J1_sum'],m['n_exact_deals']))}
    held={}
    for s,label in ((2,'discovery'),(3,'confirmation')):
        ids=np.flatnonzero(d['split']==s);N=int(d['count'][ids].sum());j1=int(np.where(d['mask'][ids],d['q_sum'][ids],2**60).min(1).sum())
        held[label]={'n_worlds':N,'n_keys':len(ids),'actual_own_hands':len({tuple(obs[i]['hand']) for i in ids}),
                     'information_floor':(j1-int(d['pi_by_group'][ids].sum()))/N,'label_excess':{}}
        for name,v in variants.items():
            a=np.where(d['mask'][ids],v[ids],-1).argmax(1)
            held[label]['label_excess'][name]=(int(d['q_sum'][ids,a].sum())-j1)/N
    derived['conditions'][folder.name]={'decoder_risks':risks,'held_out':held,'n_exact_deals':m['n_exact_deals']}
(R/'M2/results/audit_diagnostics.json').write_text(json.dumps(derived,indent=2)+'\n')
diag=R/'M2/results/diagnostics.json'; old=json.loads(diag.read_text());old['external_audit_addendum']=derived
# Retain original diagnostic record alongside its additive, explicitly dated update.
backup=diag.with_name('diagnostics_pre_external_audit.json')
if not backup.exists():backup.write_bytes(diag.read_bytes())
diag.write_text(json.dumps(old,indent=2)+'\n')
p=R/'M2/PROOF.md'
edit(p,'# M15 candidate — certified root decision risk and teacher ties','# M15 — Privileged root supervision and decomposed private decision risk')
edit(p,'**Status: pending external audit.**','**Status: externally validated (Claude,2026-11-17), documentation corrections applied; see root audit.**')
edit(p,'For an observable representation phi with legal action semantics constant in\neach representation class,','For phi=phi(I), a function of I alone, with the same legal index set in\neach representation class (actions are original-hand indices),')
edit(p,'optimum estimates those probabilities;','optimum equals those conditional probabilities at each observed private key;')
append(p,'J_student is explicitly a root-relaxation score: a private root action followed by omniscient continuation. It is not a feasible full-team score and must not be compared as an achieved policy to the M13 feasible uppers. J_phi depends on the original-hand-index action encoding; public_mask excess is not an intrinsic information measure. No guarantee applies to held-out own-hand keys.\n\nLoad-bearing assumptions: single-round hit/miss scoring without penalty; deterministic fixed hero; public hero bid; hero score as sole team objective; perfect-information continuation. Label and decoder ties use lowest original-hand index in the reported convention. Decoder tie sensitivity is in results/audit_diagnostics.json.')
for path in [R/'M2/RESULTS.md',ROOT/'LOCAL_REVIEW_LOGBOOK.md',ROOT/'AGENTS.md']:
    append(path,'Claude external audit2026-11-17 correction: hard labels and hard/uniform decoders use lowest original-hand index. SmallNT hard Bayes excess19/105 must accompany best decoder case1/42. Fixed-label decoder best/worst risks and tied-group counts are in round12/M2/results/audit_diagnostics.json; these ranges differ from changing the teacher priority. Exact complete populations2520/369600deals, no sampling CI. M3 supports avoiding label-induced cost, not a learning-dynamics advantage.')
append(R/'M2/STATUS.txt',stamp+' Audit1–8: convention/ranges stated, additive diagnostic fields saved with original backup; proof definitions/assumptions/title corrected. No frozen scientific source, labels, seed or policy changed.')
# Recompute the synthetic counterexamples rather than asserting bare inequalities.
p=R/'M2/diagnose.py'
t=p.read_text();start=t.index('    # Exact supervision counterexamples');end=t.index('    save(HERE/',start)
t=t[:start]+'''    cases = [([Fraction(1,5),Fraction(3,10),Fraction(1,2)], [(0,1),(1,0),(0,0)]),
             ([Fraction(3,5),Fraction(2,5)], [(0,0,1),(1,1,0)])]
    for index,(mass,costs) in enumerate(cases):
        risks=[sum(p*c[a] for p,c in zip(mass,costs)) for a in range(len(costs[0]))]
        labels=[Fraction(0) for _ in risks]
        for p,c in zip(mass,costs):
            good=[a for a,v in enumerate(c) if v==min(c)]
            for a in (good[:1] if index==0 else good): labels[a]+=p/(1 if index==0 else len(good))
        chosen=max(range(len(labels)),key=lambda a:(labels[a],-a))
        assert risks[chosen]>min(risks)
'''+t[end:];p.write_text(t)
append(p,'\n# External decoder-risk extension is reproducible with round12/audit_changes.py; original science arrays are read-only.\n')
notes=ROOT/'engine/math/NOTES.md';t=notes.read_text();t=t.replace('Status: pending external audit; local Round12 candidate only.','Status: externally validated (Claude,2026-11-17); audit documentation corrections applied.');notes.write_text(t)
large=derived['conditions']['S3R4n3_t-1']['held_out']
(R/'M3/results/label_decode_audit.json').write_text(json.dumps(derived,indent=2)+'\n')
np_info={'numpy':np.__version__,'configuration':str(np.__config__.CONFIG),'note':'Claude independent forward audit minimum top-two logit gap1.6e-7. Exact integer score totals require the same numerical stack; different BLAS/hardware can flip near ties.'}
(R/'M3/results/numerical_stack.json').write_text(json.dumps(np_info,indent=2)+'\n')
append(R/'M3/STATUS.txt',stamp+' Claude external M3 PASS-with-changes: supported primary framed as set supervision avoids label-induced cost. Label-only held costs/information floors/actual hand counts recomputed in results/label_decode_audit.json; numerical stack recorded. Seed CIs do not cover dominant hand-sampling uncertainty; no learning-dynamics claim.')
append(R/'REVIEW.md','## Claude external — audit2026-11-17\n\nRoot CLAUDE_AUDIT_2026-11-17.md externally validates M3 primary/split/leakage (PASS-with-changes). Changes9–14 applied through data-driven report/diagnostics and numerical-stack record. Primary supported as avoiding label-induced cost, not learning-dynamics benefit. This entry records Claude’s supplied verdict; it is not a new Codex external audit. Other ready tasks still require Claude external entries before local commits.')
board=R/'BOARD.md';t=board.read_text();t=t.replace('| Codex | ready-for-review | Completed200fits;','| Codex | validated | Claude external2026-11-17; label-misalignment framing; Completed200fits;');board.write_text(t)
proof=ROOT/'engine/experiments/round9/D5/proof.md'
append(proof,'## Claude external correction2026-11-17\n\nStatus: proved (externally checked): MH sweep invariance under strictly positive L. Any allowed swap has an allowed reverse; a non-null move exchanges distinct cards and changes the world.\n\nPrecisely L(w)=product over observed opponent bids j of [(1-eps)*1{b_j=argmax over hook-legal b EV_model(b|initial hand_j,public prior bids)}+eps/number_legal_bids_j], eps=.02,tau=0. Initial hands include reconstructed played cards; lowest-bid ties are used. The uniform feasible-world prior is a modelling choice, not the TA-play posterior; no opponent-play likelihood is included.\n\nSurvivor-filter lemma: restricting an equal-weight finite empirical bank to worlds consistent with new public evidence leaves equal weights among its survivors. Equal-weight resampling samples that finite empirical law only; duplicates induce dependence. It does not yield exact posterior draws or independent particles, nor remove initial SIR error. For an iid exact input bank, surviving values have the conditional law given evidence, but finite resampling still shares ancestors; this assumption is unavailable for the persistent MH bank.\n\nConcrete disconnected example: three opponents each hold one card, no stock. Card suits S,H,D; holder1 allows S/H, holder2 H/D, holder3 D/S. Exactly two feasible worlds are(S,H,D)and(H,D,S). Every pair exchange violates a holder constraint, so each world reaches only itself. Strictly positive L does not restore irreducibility. This says nothing about useful mixing or finite-bank posterior accuracy.')
append(ROOT/'engine/experiments/round9/D5/README.md','## Claude external2026-11-17\n\nMH sweep invariance and the completed300/300 primary null are externally checked, with the corrected proof above. The missing round/bid guard is an initialization-bias risk only. It is ruled out in this run: fresh policy per game, sampling after bids, only rounds2/22 share(n,trump,dealer), and zero cross-round carries. Any future module must include explicit round index and the full bid vector in its signature. The audit reports approximately78% fresh-SIR resets, leaving persistence active in approximately22% of sampler decisions. This is an audit-reported descriptive sampler diagnostic, not an iid-game effect estimate; denominator/cluster interval must be recovered before treating it as a new inferential result. No mixing/independence claim follows. Publication/export clearance remains separate.')
append(ROOT/'engine/experiments/round11/E3/RESULTS.md','## Claude external2026-11-17 corrections\n\nThe negative result is externally validated for this family on a ridge grid whose optimum sat at the1e-4 boundary; the grid did not bracket an optimum. Selection already showed B−canonical−0.883±0.130points/NTcontext,n8000,95%CI(descriptive, selected/optimistic). The protocol had no canonical non-inferiority gate. pipeline.py and predictor.py were first committed after the run: freeze provenance rests on in-run hashes/timestamps, not a pre-run commit. No new causal or universal family limit follows.\n\nFuture bidder protocols require a predeclared canonical non-inferiority gate at selection before spending screen/confirmation seeds, plus pipeline code committed before running. New source commits still require Claude external source approval under the current protocol. These requirements do not retroactively change frozenM4/M10selection or their predeclared counts.')
(ROOT/'engine/AGENTS.md').write_bytes((ROOT/'AGENTS.md').read_bytes())
print(json.dumps(derived['conditions']['S3R4n3_t-1'],indent=2))
