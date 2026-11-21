# AGENTS.md — Judgement research, published scope

This is the curated public handoff, not the full private research history.
Read `CLAUDE_AUDIT_2026-11-16.md`, `PUBLICATION_MANIFEST.json` and
`engine/paper/EVIDENCE.md` before extending claims. Both public AGENTS copies
must remain byte-identical.

## Project and rules

Owner: Arvin. Head of research: Codex. Judgement is a four-player trick-taking
game with general-sum exact-hit score incentives. The confirmed rules use a
52-card deck, a 25-round ladder 13,12,...,1,...,13 and rotating
spades/hearts/diamonds/clubs/no-trump. Dealer moves left; the player to the
dealer's left bids and leads first. Dealer bids last and cannot make total
bids equal the number of tricks. Follow suit when possible; a void player may
play any card, without an obligation to trump. Exact hit scores 10+bid; miss
scores zero. Trick winner leads next. Every tied game leader counts as a win;
recompute this from totals because the engine's legacy wins field splits ties.

## Working conventions

- Add new modules; do not modify existing core files or policies.
- Pair comparisons on common seeded deals and rotate full-game hero g%4.
  Preserve fixed-hero exceptions specified by exact finite-game protocols.
- Every empirical estimate keeps n and a 95% CI. Full enumerations and
  optimization bounds are exact objects, not sampling intervals.
- Predeclare contrasts, seeds and a fresh hard stop before long jobs; save
  complete checkpoints incrementally and keep STATUS files append-only.
- One Judgement compute worker is authorized alongside unrelated jobs.
  Do not stop those jobs or repeat the resource-permission question.
- Keep tests green: from engine, `python -m pytest -q -p no:cacheprovider`.
  Record the optional OpenSpiel skip explicitly.
- Separate self-review from implementation and label it as same-investigator
  review. New proofs remain pending external audit until actually checked.
- Publish only validated scope by ordinary children on the publication
  lineage. Never push private master, force-push, --all or --mirror.

## Current status

Claude externally validated D1/D2, M13/M14 and the complete finite-toy lower
bounds on 2026-11-16. Round11 E0 freshly passed 262 tests and one optional
OpenSpiel skip, 49.07s pytest; its startup-inclusive output interval was
53.910273s. The later full local working-tree gate passed 267 tests and one
optional skip, 67.91s pytest and 70.0881776s wall. Those local inventories
include private work; the curated public tree has its own pre-push test receipt.

The manuscript and ledger now agree on 11 quantitative groups. C10/D1 and
C11/M13 were the two groups omitted from the original 9-group derivative
because their external gates were open; both gates are now cleared.

E5 separately confirmed both feasible upper policies by fresh forward replay
on every one of the 369,600 deals per regime. This is self-validated (Codex),
not a new external upper-replay verdict. Portable private-information policy
maps and per-deal integer score vectors are included for reproduction.

D5 and the new NT execution packet remain local. The public audit/protocol
files describe their historical state, not a release of current local data.
The full private ancestry, local logbook and third-party PDF are excluded.
No existing core/policy file was changed in this round.

## Results so far

- The welfare identity is W=40+n−10M−U; the hook gives the sharp global
  ceiling W≤30+n. These are exact statements, not sampling estimates.
- D1's revealed-hero minus own/public surplus-share contrast is
  0.369315 [0.315244,0.424218] and 0.341204 [0.289874,0.394787], each n=1,000
  paired deals, 95% CIs. Scope is n=5, fixed TA+v2 hero/bids and K16 per active
  decision in the implemented sampler. No optimal-coalition or human claim.
- D2's fresh ModelBidder−guide score contrast is +5.05950±0.81963 points/game,
  n=4,000 paired games, 95% CI. This is the whole bidder intervention with
  fixed guide play, not oracle error or human-learning evidence.
- M13 gives exact optimization bounds [446669/73920,574247/92400] (trump)
  and [213259/30800,109709/15400] (NT), each over all 369,600 ordered deals.
  The certified information gaps are 25819/369600≈0.070 and 421/30800≈0.014
  hero points. They establish strict cost and nothing about material magnitude;
  neither interval solves the hidden optimum or extrapolates to 52-card play.
- M14 permits pure-team sufficiency against fixed non-team policies, with
  own/public histories and random tapes independent of chance/non-team tapes.
  It is not a strategic-adversary team-maxmin or multiplayer CFR theorem.

Other reviewed results and retained limitations are in EVIDENCE.md and the
task-level STATUS/review files. No novelty claim follows from the bounded
related-work review.

Fresh curated-export suite: 262 passed, 1 optional OpenSpiel skip; 28.07s pytest, 29.118s wall. See PUBLICATION_TESTS_2026-11-16.json for the tested candidate and complete output.

## Next steps

1. Extend the primary-source literature/reproduction audit before submission.
2. Independently cross-check the local D5 proof/implementation and NT packet
   before any later release. Preserve frozen raw data and declare fresh seeds
   for new variants; do not tune against completed confirmation outcomes.
3. Playtest the existing guide with humans; no human benefit has been measured.
4. Study stronger finite-team bounds while retaining fixed-policy scope and
   the distinction between unilateral BR residuals and joint optimality.

## Session log

| Date (ET) | Session | Outcome |
|---|---|---|
| 2026-11-16 | Round11 audited release | Fresh local test gates; externally cleared M13/M14 and D1/D2 scopes; 11-group manuscript reconciliation; separate exact feasible-upper replay. New empirical packets remain local; publication ancestry stays separate from private master. |

## Published addition — 2026-11-21: ML-track audit verdicts and N17

`CLAUDE_AUDIT_2026-11-20.md` records an external audit of the local
machine-learning track (round12 N-series). Each verdict was reached by
independent recomputation over the raw arrays rather than by re-running the
task's own analyzer, and the two new proofs were re-derived by hand. N9, N10,
N12, N13/M18, N15/M19 and N11/M17 all passed; M18 and M19 are validated proofs;
N12 required one reporting change, which was applied locally.

`engine/experiments/round13/` publishes N17, which answers the obvious
objection to that track's main negative result. Stratifying 1375 exactly
enumerable contexts by the spread of their true action values leaves 224 where
the two legal actions genuinely differ, and the exact conditional-product
projection of the true posterior incurs **zero** one-step decision regret on
every one of them. A uniform belief on the same contexts does lose points in
three of the four conditions, so the measurement has power; the fourth
condition has none and its zero is excluded from the evidence. These are exact
enumerations over complete finite context sets and carry **no sampling
confidence intervals**.

What is still withheld, and why: the round12 task folders themselves, their
model weights and raw caches (about 2.6 GB), and the workshop paper draft. The
paper is task N7, which has not been externally reviewed, and N5, N6 and N4
also remain pending. The round12 export and licensing gate is still open. As a
result `engine/experiments/round13/N17/` reads input arrays that are not present
in this public tree, so it documents and verifies a result here but is not
reproducible from this tree alone. Integrity receipts retain absolute local
source paths, as elsewhere in this repository, because rewriting them would
break hash provenance.

### 2026-11-21 addendum: N19

`engine/experiments/round13/N19/` asks why N17's regret is zero and answers it.
Bounding the action-value difference over the entire polytope of beliefs that
match the true posterior's card-holder marginals, with no family restriction,
**zero of the 224 decision-relevant contexts admit a marginal-consistent belief
that strictly prefers the inferior action**; 215 are strictly identified by
those marginals and the other 9 reach exact indifference, where an adversarial
tie rule would cost 48/77, 910/317, 12/47 and 66/205 hero points per unit reach.
Exact linear-programming bounds with exact rational vertex certificates; no
floating LP status is accepted as a feasibility or sign certificate.

### 2026-11-21 addendum: the audit-cleared round12 ML track is now published

`engine/experiments/round12/` now contains the round12 tasks that carry a
recorded external-validation verdict: M0 through M10, N0 through N3, and the six
packets cleared by the 2026-11-20 audit (N9, N10, N11, N12, N13, N15). The
per-task verdicts, what is withheld and what is omitted from each published task
are listed in `engine/experiments/round12/PUBLISHED_SCOPE.md`, which replaces the
private working copy's BOARD and REVIEW files for the published scope. The
round12 entries of `engine/math/NOTES.md` now include M15 through M19; M18 and
M19 are validated proofs.

Still withheld: N4, N5, N6 and N7 (no external verdict; N7 is the workshop paper
draft, so the draft is withheld with it), N14 (complete and separately verified
but with no independent external verdict), N16 (code only), and round13's N18
(its results quote N4's interval). Also withheld everywhere: neural model
weights and `data/` caches, about 2.6 GB across the round, and the bulk
per-block result trees of the six tasks whose results exceed 10 MB. The exact
records the 2026-11-20 audit rests on — N11's certificates, N12's integer action
tables, N13's rational certificates and N15's enumeration — are published in
full. Some published tasks can therefore be read and checked but not re-run end
to end from this tree alone.
