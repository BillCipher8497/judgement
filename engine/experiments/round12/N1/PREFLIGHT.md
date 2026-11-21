# N1 source preflight — ready for Claude external source review

Prepared 2026-11-17T21:18:06.908031+00:00. No experimental cases, training labels or outcomes generated.
This requests a **source-only** verdict so reviewed code can be committed before
experiment execution. It is not a validation claim for N1 results.

Source fingerprint: `327251dcd38e12740a43769bce83e4f5378895e414a6f5ea3e654e3716f9a86e`. Per-file raw/canonical-LF hashes, NumPy/BLAS
metadata are in source_preflight.json. Reviewed dependencies include the exact
canonical bidder model. Run `python N1/run.py --preflight` from round12 to
recompute metadata only. Do not start case generation before approval.

Three synthetic tests pass as part of the actual full suite: **283 passed,
1optional OpenSpiel skip**,14.35s pytest/15.0119943s wall; M6/results/review_full_tests.json.
They check stock/no-oracle-filtering and strict observable schema, the zero
residual/proportional baseline and history ablation, and finite-difference
masked proper-mixture CE/ReLU/L2 gradients.

Seed preflight checks 123544 distinct128-bit
root states, zero collisions, including the known N0two-integer streams.
Namespaces/stages/tails and bounded tracked-tree inventory are in
preflight/seed_audit.json. This is not a claim about unknowable external draws.
N0 is untouched. Bases2100M/2110M only, remaining reservations unused.

PROTOCOL.md declares four holders including stock, all unseen cards,3520fixed
confirmation decisions, full-history/bids-only paired arms, K256BI/uniform,
strictly positive identical smoothing, round-weighted proper loss,176stratum
paired95%CIs, secondary conditional diagnostics/reliability/ECE, reused
selection noninferiority filter, resource-only pilot gate, exact atomic
checkpoint/resume rules, count-only null/scope failures, one numpy worker.
Predeclaration UTC is in append-only STATUS.txt, before any outcomes.

Proposed scientific cap2026-11-18T03:00Z; a fresh launch time/deadline must be
recorded after source approval. No job now. External source approval must
identify the exact fingerprint in REVIEW.md; the runner enforces it and an
identical pre-run Git commit, then fingerprints all generated chunks/models.

The independent verifier reconstructs labels from original hands minus public
history, re-encodes strict observations, forwards all selected candidates,
checks epochs/blends and recomputes the paired primary CI. It is not an
external verdict; fresh prefix/sampler replays and ECE bootstrap checks remain
part of later external review. Reviewed code first, approved source-only local
commit second, pilot third. N1 stays running until result review; N2 not started.

Approval syntax for the runner: write the literal prefix `N1_SOURCE_APPROVED` followed by a space and the reviewed source fingerprint, in your Claude external verdict. This file/request deliberately contains no assembled approval token. Source-onlyapproval permits a pre-run source commit and resource pilot, not release of future unvalidated results.
