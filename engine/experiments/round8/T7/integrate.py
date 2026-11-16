"""Integrate Claude-validated round-8 results; no new estimates or simulations."""
import hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
ENGINE=HERE.parents[2];PROJECT=ENGINE.parent;ROUND=HERE.parent
board=(ROUND/'BOARD.md').read_text(encoding='utf-8')
for task in ['T1','T2','T3','T4','T5','T6']:
    row=next(x for x in board.splitlines() if x.startswith('| '+task+' |'))
    assert row.split('|')[-3].strip()=='validated',task

def read(relative):return json.loads((ENGINE/relative).read_text(encoding='utf-8'))
def w(v):return f"{100*v['p']:.2f} [{100*v['low95']:.2f}, {100*v['high95']:.2f}]"
def m(v):return f'{v[0]:.3f} ± {v[1]:.3f} (n={v[2]})'
def gap(v):return f"{v['point']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
t3=read('experiments/round8/T3/results/summary.json')
t4=read('experiments/round8/T4/results/summary.json')
fit=read('experiments/round8/T6/results/fit.json')
assert fit['config']['weights']==[0,0,0,0]

section='''## Round 8 — Claude-validated additions (2026-11-16)

Only results validated in `experiments/round8/REVIEW.md` enter this section. The raw files, exact commands, seed design and limitations are in each task's append-only STATUS.txt. All ± and bracketed intervals below are 95%; exact complete enumerations have no sampling CI.

### Audit correction: prefer the same-hero information comparison

At five cards, the heuristic-hidden/omniscient surplus ratio is **0.296 [0.264, 0.327]**, n=4,000 paired deals with the same TA+ hero and fixed bids. The hidden PIMC/omniscient ratio is **0.356 [0.313, 0.398]**, n=2,000 paired deals. Source: `experiments/round7_audit/results/a1_surplus.json`. These supersede the cross-policy M11 ratio as the preferred information contrast. Neither computes the optimum hidden coalition; keep the old M11 ratio above as descriptive history only.

### T2: equal-depth top-pool replication

Pool: 3×Hybrid-BI K32c6. Fresh seed base 1,000,000,000; challenger seat g%4; n=300 paired games per comparison. Game-level paired normal intervals:

| Challenger / contrast | Score difference, points/game | Ties-as-win probability difference |
|---|---|---|
| BI K32c6 sanity − pool | 0.00 ± 0.00 | 0.0000 ± 0.0000 |
| RB K32c6 − pool | −0.74 ± 2.96 | +0.0167 ± 0.0566 |
| RB K32c7 − pool | +4.64 ± 3.23 | +0.0333 ± 0.0569 |
| RB c7 − RB c6 | +5.38 ± 3.12 | +0.0167 ± 0.0589 |

Source: `experiments/round8/T2/results/summary.json` and 1,200 per-game JSON records. Equal-depth score improvement is not detected; the positive depth contrast does not establish an opponent-modelling benefit or a victory-probability improvement. The sanity arm is a deterministic plumbing check: all total vectors match; it does not calibrate independent Monte Carlo noise.

### T3: mode match under target-aware play

Both grids use 40 situations × 13 sizes × 2 trump types × 4 bid positions = **4,160 situations**, R=300 hidden worlds per legal bid. Every table entry is a percentage with a Wilson 95% CI, n=4,160 situations. Earlier bids use the original Greedy context design; later bidders use the policy's fitted model. The 4,242 grid reproduces every original Greedy distribution exactly. Its policy MC streams are independent; confirmatory 4,243 uses strictly common hidden worlds across Greedy, TA, TA+ and all candidate bids.

| Grid / play | Oracle = mode | = floor(mean) | = ceil(mean) | = round(mean) | = floor or ceil |
|---|---|---|---|---|---|
'''
for seed in [4242,4243]:
    for pol,label in [('greedy','Greedy'),('ta','TA'),('ta_plus','TA+')]:
        d=t3[f'raw_{seed}'][pol]
        section+=f'| {seed} / {label} | '+' | '.join(w(d[k]) for k in ['mode','floor','ceil','round','floor_or_ceil'])+' |\n'
section+='''
Source: `experiments/round8/T3/results/summary.json`, complete raw laws in `raw_4242.jsonl` and `raw_4243.jsonl`; diagnostic-only `unpaired_prefix_4243.jsonl` excluded. For TA/TA+, each bid changes play: the mode and mean are evaluated in **the law at the oracle-selected bid**, not a single exogenous trick law. Thus the high match is a conditional diagnostic and may be self-targeting; it is not an implementable rule to bid an unknown distribution's mode. Finite-rollout selection and argmax tie handling remain. These dependent laws do not inherit the independent-card PB theorem. Discovery has no stored hand/prior metadata; the fresh replication records and fully checks both.

### T4: dependence without the opening lead

NT, TA play, fixed prior/own bids and hook-legal diagnostic bids. Discovery base 1,100M; independent replication 1,110M. Each cell has 500 raw hands × R=300 rollouts, split 150/150 for cross-fitting; degenerate hands are excluded from the affected estimates. δ = Var(total tricks)/PB variance; the cross-suit term is covariance contribution divided by PB variance, not a fraction of total excess dispersion. Persistence excludes the first trick. Parentheses give each estimate's actual hand count; gap CIs use 4,000 paired-hand bootstrap replicates.

| Base / cards / position | δ | Cross-suit dispersion | Lead persistence | LC0 gap closed | TPB gap closed | Gap n |
|---|---|---|---|---|---|---|
'''
for key,d in t4.items():
    seed,n,pos=key.split('|')
    section+=f'| {int(seed)//1000000}M / {n} / {pos} | {m(d["delta"])} | {m(d["cross_suit_dispersion"])} | {m(d["persistence"])} | {gap(d["LC0_gap_closed"])} | {gap(d["TPB_gap_closed"])} | {d["n_hands"]} |\n'
section+='''
Source: `experiments/round8/T4/results/summary.json` and 6,000 per-hand NPZ files. Dependence persists without an opening lead. LC0's positive gap closure replicates at n=5 in positions 1 and 2; the dealer effect fails replication, and every n=8 LC0 interval includes zero. TPB has little explanatory power. Compared with M10's opening leader, these are descriptive differences across seeds and bidding contexts, not a paired leader intervention. These remain per-hand rollout diagnostics rather than deployable bidders.

### T5: exact hidden-team certificate in a small toy

Four players, 2 suits × 4 ranks, n=2; all myopic bids and TA hero play frozen. Each regime enumerates all **2,520 ordered deals**. Spoilers keep separate private information sets and optimize play only.

| Regime | TA baseline hero score | Independent-BR heuristic | Certified hidden team | PI minimum |
|---|---|---|---|---|
| Trump | 9.6024 | 8.9786 | 1257/140 | 1257/140 |
| NT | 9.9889 | 9.1032 | 946/105 | 946/105 |

All are full-population values, so no statistical CI applies. Exact integer support DP proves zero pointwise excess over the PI lower bound on every deal: the saved feasible hidden strategy has **joint optimality gap exactly zero**, in both regimes. This is stronger than the unilateral BR residual; multiplayer CFR convergence alone would not supply the certificate. Source: `experiments/round8/T5/results/certificate.json`, certified NPZ profiles, `T5/certify.py`, and M12 in `math/NOTES.md`. The independent-BR heuristic is the toy-specific construction, not the 52-card spoiler. This tiny fixed-bid play-only toy cannot bound the full-game hidden/PI ratio; a larger toy is a later-round question.

### T6: guide shape/lead feature grid gave no gain

Training seed 1,250M, n=60,000 S1-style true-deal counterfactual situations: none of 143 nonzero integer weight vectors beats the unchanged guide in this finite grid. The selected weights are (0,0,0,0). The best nonzero candidate's training difference is −0.002 ± 0.006 points/round (n=60,000); this is a selected, descriptive training estimate, not an independent replication of a feature effect.

The frozen zero-weight rule was screened at 1,300M (n=2,000 paired games) and validated unchanged at 1,400M (n=4,000). Both score and ties-as-win deltas are 0 ± 0 by identity; these games confirm implementation identity and supply no evidence for improved features. At validation, either guide scores 159.74 ± 0.85 points/game with ties-as-win probability 0.2423 ± 0.0133, n=4,000 against 3×TA, hero g%4. Source: `experiments/round8/T6/results/fit_sweep.json`, `fit.json`, screen/validation JSONL and summaries, and `verification.json` (all 60,000 vector/deployed bids agree; all 6,000 paired game records check). The original guide stays unchanged. This result excludes neither other encodings of shape/lead nor benefits under other opponent pools; no human playtest occurred.

### Reproduction and validation

From `engine/`, use `python experiments/round8/run_tasks.py T3` for the original/fresh grids (completed statuses skip by design; direct `T3/run.py` also resumes). Run each task's `analyze.py` where present, and `T3/verify.py`, `T5/certify.py`, `T6/verify.py` for the saved-data audits. T4 and T5 each provide a standalone `worker.py`; T6 `run.py` enforces its one-candidate freeze. Historical runners retain the overnight 07:30 ET hard deadline and therefore deliberately will not start compute after that date; use saved raw data for analysis, and preregister a fresh deadline and seeds for a new experiment. Exact commands and all seed bases are recorded in each STATUS.txt.

The final integrated suite passed **256 tests, 1 optional OpenSpiel module skipped** using `python -m pytest -q -p no:cacheprovider` (16.08 s). Existing core modules and existing policies are unchanged; only the new experimental guide-v2 policy was added.
'''
path=ENGINE/'paper/EVIDENCE.md';doc=path.read_text(encoding='utf-8')
marker='## Round 8 — Claude-validated additions (2026-11-16)'
if marker in doc:doc=doc.split(marker)[0].rstrip()+'\n'
path.write_text(doc.rstrip()+'\n\n'+section,encoding='utf-8')

sources=['experiments/round7_audit/results/a1_surplus.json']+[
    f'experiments/round8/{t}/results/{f}' for t,f in [('T2','summary.json'),('T3','summary.json'),('T3','verification.json'),('T4','summary.json'),('T5','certificate.json'),('T6','fit.json'),('T6','validation_summary.json'),('T6','verification.json')]]
manifest={'no_new_randomness':True,'validated_tasks':['T1','T2','T3','T4','T5','T6'],
          'sources_sha256':{p:hashlib.sha256((ENGINE/p).read_bytes()).hexdigest() for p in sources},
          'evidence_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(HERE/'results/evidence_sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
with (HERE/'STATUS.txt').open('a',encoding='utf-8') as f:
    f.write(datetime.now(timezone.utc).isoformat()+' Evidence integration complete; source hashes saved; no new estimates.\n')
print(json.dumps(manifest,indent=2))
