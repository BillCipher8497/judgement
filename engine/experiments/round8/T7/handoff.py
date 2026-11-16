"""Update both handoff copies using validated findings only."""
from pathlib import Path
HERE=Path(__file__).resolve().parent;ENGINE=HERE.parents[2];PROJECT=ENGINE.parent
root=PROJECT/'AGENTS.md';inner=ENGINE/'AGENTS.md'
doc=root.read_text(encoding='utf-8');assert doc==inner.read_text(encoding='utf-8')
board=(HERE.parent/'BOARD.md').read_text(encoding='utf-8')
for task in ['T1','T2','T3','T4','T5','T6']:
    assert next(x for x in board.splitlines() if x.startswith('| '+task+' |')).split('|')[-3].strip()=='validated'
doc=doc.replace('Last updated: 2026-11-15 (round 7 paper and welfare session)',
                'Last updated: 2026-11-16 (round 8 overnight, Claude-validated research)')
doc=doc.replace('pytest + hypothesis (250 passing as of 2026-11-15)',
                'pytest + hypothesis (256 passed, 1 optional OpenSpiel skip on 2026-11-16)')
doc=doc.replace('`spoiler`, `human_guide` |','`spoiler`, `human_guide`; experimental `human_guide_v2` (zero fitted adjustments; no improvement) |')
doc=doc.replace('`guide`, `round6`.','`guide`, `round6`, `round7_paper`, `round7_audit`, `round8`.')
lines=doc.splitlines()
for i,line in enumerate(lines):
    if line.startswith('- **Bidding:**'):
        lines[i]='- **Bidding:** Greedy oracle = mode: 86.8% [85.8, 87.8] (n=4,160; Wilson 95% CI), reproduced distribution-by-distribution. TA 94.6% [93.9, 95.3] and TA+ 95.6% [95.0, 96.2], each n=4,160; fresh replication TA 94.7% [94.0, 95.3], TA+ 95.6% [95.0, 96.2] (same n). For TA/TA+, mode means the law at the oracle-selected bid: a conditional diagnostic, not a bid-free recommendation. Under independent-card PB laws the unconstrained optimum is the mode or one above and lies in {floor μ, ceil μ}; hook and dependent laws require separate treatment.'
    elif line.startswith('- **Robustness:**'):
        lines[i]=line.split(' Top-pool equal-depth')[0].split(' Top pool (3×Hybrid-BI K32c6),')[0]+' Top pool (3×Hybrid-BI K32c6), fresh 1,000M seeds, n=300 paired: RB c6 −0.74 ± 2.96 points/game (no detected equal-depth gain); RB c7 +4.64 ± 3.23; c7−c6 +5.38 ± 3.12. Depth contrast does not establish opponent-modelling benefit or a win-rate gain.'
    elif line.startswith('- **Human guide:**'):
        lines[i]=line.split(' Round 8: none of 143')[0]+' Round 8: none of 143 nonzero shape/NT-lead weightings beat the current rule on 60,000 training situations (best selected difference −0.002 ± 0.006 points/round; descriptive, selection-biased). Zero weights were frozen; screen n=2,000 and fresh validation n=4,000 give paired Δ=0 ± 0 by identity. No improvement promoted; other feature encodings remain open.'
doc='\n'.join(lines)+'\n'
extra='''- **Non-leading NT heroes (round 8):** dependence persists; replication n=5 pos1 δ=1.575 ± 0.030 and lead persistence 0.237 ± 0.013 (n=500 hands). LC0 gap closed: discovery pos1 0.314 [0.153, 0.455], pos2 0.295 [0.117, 0.442] (n=499 each), replication 0.426 [0.271, 0.561] and 0.286 [0.064, 0.458] (n=500 each). Dealer effect fails replication; no n=8 LC0 interval excludes zero. Per-hand rollout diagnostics; comparison with the opening leader is descriptive across designs.
- **Hidden-team toy certificate (round 8):** an 8-card, 4-player, n=2 toy with fixed TA hero and myopic bids has feasible private-infoset spoiler strategies attaining PI punishment exactly on every deal. Mean hero scores are 1257/140 (trump) and 946/105 (NT), joint gap exactly zero; complete enumeration of 2,520 deals/regime, no sampling CI. Proved support-certificate lemma in M12. This toy does not bound the 52-card information gap.
'''
if '- **Non-leading NT heroes (round 8):**' not in doc:
    doc=doc.replace('\n## Current status', '\n'+extra+'\n## Current status')
start=doc.index('## Current status');end=doc.index('## Session log',start)
doc=doc[:start]+'''## Current status
- Rounds 0–8 compute are complete. Claude validated round-8 T1–T7, including the final evidence/handoff integration, in `engine/experiments/round8/REVIEW.md`. The round is closed; all simulations have finished, with no remaining queued compute.
- The integrated suite passes: 256 tests, 1 optional OpenSpiel module skipped on local Python 3.12 (`python -m pytest -q -p no:cacheprovider`). Core modules and existing policies are unchanged. The new experimental guide-v2 module stays at zero adjustments.
- New work: top-pool depth control; bid-conditional mode replication; non-leading NT dependence with fresh replication; exact hidden-team toy certificates (M12); negative guide-feature result. Raw outcomes, CIs, exact commands, seed bases and limitations are saved under `engine/experiments/round8/T1/` through `T7/`. The paper evidence map contains validated findings; Claude remains head of research and manuscript writer.
- Fresh seeds: top-pool 1,000M; NT dependence 1,100M discovery / 1,110M replication; guide training 1,250M, screen 1,300M, validation 1,400M. Mode grid uses original 4242 and fresh 4243. T5 enumerates the complete toy chance space, so no stochastic seed is used.
- All validated task commits are local; no push. The local Git history is newer than `engine_snapshot.zip`, which remains the pre-round-7 source snapshot. The cloud copy at `/home/claude/judgement-engine` may contain unsynced work; the living Claude report still needs reconciliation.

## Next steps (priority order)
1. Claude develops the manuscript from `paper/OUTLINE.md` and the validated `paper/EVIDENCE.md`; reconcile the living report and review literature. Prefer the audit's same-hero paired information contrast; distinguish optimal punishment from the implemented spoilers and the tiny toy certificate.
2. Decompose the human guide's bidding loss against ModelBidder by cards, trump/NT and seat before proposing more features. The tested finite shape/lead grid gave no improvement; do not retune it on the screening or validation seeds.
3. In a later round, extend the hidden-team support certificate to a larger fixed-bid toy where information may matter. Keep unilateral BR residuals separate from a joint coalition gap.
4. Playtest the existing guide with real players and prepare a one-page printable card; no human playtest has occurred yet.
5. Rejuvenate particles with card-swap MCMC and weight worlds by opponents' play likelihoods, under a newly assigned protocol and the shared compute limit.
6. Build deployable hand-feature lead models and per-suit control for larger NT rounds; current LC diagnostics are fitted per-hand rollout explanations. Study equilibrium selection in larger multiplayer toys separately.

'''+doc[end:]
row='| 2026-11-16 | Round 8 overnight (Claude head, Codex compute/integration) | Integrated audit; equal-depth top-pool null and positive depth contrast; mode-match replication with conditional-law caveat; non-leading NT dependence and limited LC0 replication; exact fixed-bid toy hidden-team certificate; negative guide-feature grid. All research results Claude-validated; 256 tests passed, 1 optional skip; raw data and local task commits saved. |'
if row not in doc:doc=doc.rstrip()+'\n'+row+'\n'
root.write_text(doc,encoding='utf-8');inner.write_text(doc,encoding='utf-8')
assert root.read_bytes()==inner.read_bytes()
print('Updated both AGENTS.md copies identically with validated results and session log.')
