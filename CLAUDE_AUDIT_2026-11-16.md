# External audit — Claude, session 2026-11-16 (ET evening)

Independent external validation of the local Judgement research record, requested by Arvin.
Scope: LOCAL_REVIEW_LOGBOOK.md withheld items, already-published D1/D2, publication hygiene.

Auditor environment: separate cloud container (own 2-core sandbox), independent of the machine
compute cap. No Judgement worker was started on the owner's machine for simulation; the only
device-side command attempted was a test-suite run (see limitation L1).

Method note. "Independent" below means: the tree construction, PI recursion, root identification,
information-set grouping and J1 aggregation were re-implemented from scratch in a fresh file
(`indep.py`, auditor container) without importing or copying the experiment's evaluator, BR solver
or root-bound routine. The fixed policies (greedy, myopic bidder, toy TA, pwin/worlds) are the
*definition* of the experiment, so they were re-implemented to the same documented semantics.
This is an independent recomputation, not a blind reproduction from a specification alone.

## Verdicts

| Item | Verdict | Evidence |
|---|---|---|
| D3 exact enumeration | **PASS — externally validated (Claude)** | Fresh implementation over all 369,600 ordered deals per regime reproduces every headline rational exactly: trump PI=367921/61600, J1=446669/73920; NT PI=106419/15400, J1=213259/30800. Gaps 25819/369600 and 421/30800, both strictly positive. Integer full-population sums; no sampling CI applies. |
| M13 first-decision relaxation (proof) | **PASS — externally validated (Claude)** | Relaxation argument checked step by step. Root measurability holds because every move before the first free team action is a fixed bid, a deterministic hero move or a forced move, so no team strategy alters which deal reaches its root. Granting omniscience after the root choice weakly lowers hero score; dropping cross-world coupling of non-root occurrences of the same infoset is a further relaxation; the objective is linear in each sigma_I, so a pure action per infoset attains the minimum. PI <= J1 follows from sum-of-min <= min-of-sum. Chain PI <= J1 <= optimum <= achieved is sound. |
| M14 pure-team sufficiency (proof) | **PASS — externally validated (Claude)** | Correct derandomization: fixing team tapes induces a feasible pure tuple on own-information histories; J = sum_p lambda(p) J(p) >= min_p J(p); finitely many tuples; pure tuples admissible. Tape independence from chance and from non-team tapes is correctly identified as load-bearing. The matching-bit counterexample is correct (strategic hero: cost max(p,1-p) minimised at 1/2 < 1 for any pure bit), confirming the fixed-hero premise cannot be dropped. The four-world check recomputes exactly by hand: pure minimum 9/10, uniform profile 43/20. |
| D5 recorded state | **PASS (state only)** | 281 complete triplets on 1700M (g=0..280, no gaps, arms bi/pf/mcmc all present), 0 files on 1710M. Single frozen source hash across all 281 files. Seat rotation correct: hero == g%4 in every game. |
| D5 candidate effect | **CONFIRMED NULL SO FAR — do not promote** | Auditor's own paired recomputation from raw per-game totals: mcmc - bi = +1.263 +- 2.133 (n=281), pf - bi = -2.345 +- 2.422, mcmc - pf = +3.609 +- 2.333. The primary screening contrast does not exclude zero, and replication is zero. Withholding any effect estimate was correct. |
| D1 information intervention | **PASS — externally validated (Claude)** | Recomputed from raw per-game files. Discovery 1600M: hero score self 6.244, own 4.670, hero-revealed 2.364, all 1.283 (n=1000); paired hero-revealed minus own-only = -2.306 +- 0.328. Replication 1610M: -2.182 +- 0.331 (n=1000). Replicates. Predeclared degeneracy check holds exactly: the all-hands arm equals the omniscient paranoid arm with paired difference 0.0000 +- 0.0000. Effect present in both regimes (S -2.58, NT -2.03 on discovery). |
| D2 guide bidding loss | **PASS — externally validated (Claude)** | Recomputed from raw jsonl. Discovery 1500M +5.2942 +- 0.8267 (n=4000); replication 1510M +5.0595 +- 0.8196 (n=4000), matching the recorded replication figure exactly. Seat rotation correct (seat == g%4 in all 8000 games). The recorded familywise caveat on the three frozen cells is retained. |
| Publication hygiene | **PASS** | origin/main = a99c9577334719a04e1b6d6484c158081a71f17d, equal to publication/validated-2026-11-16, and parentless (no private ancestry). Withheld paths absent from the published tree: round9/D3, round10/R3, round9/D5, LOCAL_REVIEW_LOGBOOK.md, OtherPaper.pdf, engine_snapshot.zip. Published math/NOTES.md is truncated at M12 (1257 lines vs 1345 local), so M13 and M14 are excluded. All six checks in PUBLICATION_LOCAL_RECEIPT.json independently confirmed. |
| Core-source compliance | **PASS** | Since 2026-11-15 the only change under engine/judgement/ in tracked history is one addition (policies/human_guide_v2.py). No modification or deletion of pre-existing core files or policies. Both AGENTS.md copies are byte-identical. |

## Changes requested

1. **C1 — logbook wording.** "math-note exclusion" reads as if math/NOTES.md were withheld; the file
   is published with the M13/M14 sections removed. Reword to "M13/M14 section exclusion".
2. **C2 — D5 null must stay a null.** Record the auditor's screening contrast above in D5/STATUS.txt.
   Resuming toward 300 triplets per base is a completion of a predeclared protocol, not a search for
   a positive result. If the completed screen and replication are both null, that is the finding.
3. **C3 — magnitude honesty for M13.** The certified information gaps are 25819/369600 ~ 0.070 and
   421/30800 ~ 0.014 hero points. They certify a *strict* information cost and nothing about its size;
   the manuscript must not let "strict separation" imply a material effect.

## Scope and limitations retained after this pass

- M13/M14 are validated as stated: toy-specific, fixed-hero, fixed-bid, no pooled private cards, no
  mediator. No 52-card extrapolation, no CFR convergence claim, no equilibrium claim.
- M13's feasible upper scores (574247/92400 trump, 109709/15400 NT) were **not** independently
  replayed; only their ordering against J1 was checked. The optimisation intervals remain intervals:
  neither endpoint is the hidden-team optimum.
- D1 causal language is now supportable **for this intervention only** (n=5, fixed TA hero, fixed
  bids, fixed PIMC spoiler policy and budget, information varied). It remains unsupported as a claim
  about the 52-card game, about human play, or about any other hero policy.
- L1: the full test suite was **not** independently re-run. The device VM available to the auditor has
  neither numpy nor pytest, and the interpreter that produced the recorded 262 passed / 1 optional skip
  is not reachable from it. The suite remains a release gate owned by Codex (task E0).
