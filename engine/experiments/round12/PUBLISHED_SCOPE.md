# Round 12 — published scope

This folder is a **curated derivative**, like the rest of this repository. It
contains the round12 tasks that carry a recorded external-validation verdict,
and nothing else. The private working copy's `BOARD.md` and `REVIEW.md` are not
published, because they carry results for tasks that are still pending review;
this file replaces them for the published scope.

## What is here

| task | what it is | verdict |
|---|---|---|
| M0 | Primary-literature map, positioning, resource and seed inventory, preregistration | validated (self, Codex) |
| M1 | Portable exact root counterfactual datasets for four fixed games | validated (self, Codex) |
| M2 | Exact tie/representation diagnostics and the decomposition proof note | validated (code/data self; proof externally checked) |
| M3 | Paired linear/MLP learners across five supervision losses | externally validated, Claude 2026-11-17 |
| M4 | Fresh no-trump dataset and baseline-preserving neural residual bidders | externally validated, Claude 2026-11-17 |
| M5 | Frozen bidder shift tests against human and greedy bidder pools | externally validated, Claude 2026-11-17 |
| M6 | Independent verification of raw results, leakage and seed CIs; paper draft | externally validated, Claude 2026-11-17 |
| M7 | Learning curves at 25/50/100% of training groups | externally validated, Claude 2026-11-17 |
| M8 | Logbook, AGENTS, session row, source and resource cleanup | externally validated, Claude 2026-11-17 |
| M9 | Frozen root-student deployment at every private team decision | externally validated, Claude 2026-11-17 |
| M10 | Matched linear/MLP and explicit-offset ablations on the M4 labels | externally validated, Claude 2026-11-17 |
| N0 | Belief-net pilot | **pilot only, explicitly not a result** |
| N1 | Learned belief model at scale | validated, Claude 2026-11-18, with the 2026-11-19 review corrections governing |
| N2 | Exact-posterior check in the toys | validated, Claude 2026-11-18, same corrections |
| N3 | Privileged-teacher distillation for full-game card play | validated-**uninformative**, Claude 2026-11-18: the selected policies are inactive baseline identities |
| N9 | Structured-joint versus marginal belief objective | externally validated, Claude 2026-11-20 |
| N10 | Activated residual late-play students | externally validated as a **scope failure**, Claude 2026-11-20: zero games ran |
| N11 | Stage-stratified exact fusion diagnostic, with the M17 witness | externally validated on the certificate algebra, Claude 2026-11-20 |
| N12 | Belief KL to exact one-step decision regret | externally validated, Claude 2026-11-20, one reporting change applied |
| N13 | Oracle conditional-product projection diagnostic, with the M18 lemma | externally validated, Claude 2026-11-20 |
| N15 | Exact single-context product-family KL infimum, the M19 proof | externally validated, Claude 2026-11-20 |

The 2026-11-20 verdicts and the evidence behind them are in
`CLAUDE_AUDIT_2026-11-20.md` at the repository root. The round13 follow-ups
N17 and N19, which build directly on N11/N12/N13/N15, are in
`engine/experiments/round13/`.

## What is withheld, and why

- **N4** (belief-in-search, fixed-K): ready for review, external scope review
  still open. Its equal-wallclock and SIR reset/ESS diagnostics are missing;
  round13's N18 measures part of that gap and is itself held, because its
  results quote N4's interval.
- **N5, N6, N7**: ready for review, no external verdict. N7 is the workshop
  paper draft, so the draft is withheld too.
- **N14** (actual-reward late-play learning): complete under its frozen protocol
  and separately verified, but it has no independent external verdict, so it is
  held to the same standard as the rest.
- **N16**: code only, no data, no seeds.
- **N8**: an owner decision, not an experiment.

## What is omitted from the published tasks

Everything substantive is here, but two classes of file are not:

1. **Model weights and `data/` caches.** Neural checkpoints and the raw
   training/validation/held caches run to roughly 2.6 GB across the round and
   do not belong in a git repository. Every protocol, source file, STATUS log,
   RESULTS write-up, proof note, figure and verification receipt is published.
2. **Bulk per-block result trees, for the six tasks whose results exceed
   10 MB.** For M4, M5, M10, N1, N3 and N10 the top-level results files
   (summaries, manifests, receipts, `verification.json`, `complete.json`) are
   published and the per-fit or per-game subdirectories are not. The omitted
   subdirectories are, by task: M4 and M5 per-seed fit directories and their
   screen/confirmation game blocks; M10 the `L_s*`, `N_s*`, `R_s*` fit
   directories and its screen/confirmation blocks; N1 its per-record arrays;
   N3 its per-block game trees (about 40,000 files on their own); N10 its
   per-fit candidate directories.

   The exact scientific records that the 2026-11-20 audit actually rests on —
   N11's certificates, N12's per-key integer action tables, N13's per-key
   rational certificates and N15's enumeration — are **all published in full**,
   because each of those trees is small.

Consequently some published tasks can be read and checked but not re-run end to
end from this tree alone. That is a deliberate curation, not an oversight, and
it is the same convention the earlier published rounds follow.

## Seed namespaces

Seed base integers appear inside the published collision-check code (for
example `N9/preflight.py` and `N10/n10lib.py` enumerate previously consumed
namespaces in order to prove that a new namespace does not collide). Those are
reserved-namespace identifiers, not results, and they are published
deliberately so that seed discipline can be checked.
