# N11 — stage-stratified exact fusion follow-up

**Externally validated by Claude, 2026-11-20, on the certificate algebra**; the reduction of the structural assignments to the exact posterior support is inherited from the previously audited N2 machinery rather than independently re-derived. See CLAUDE_AUDIT_2026-11-20.md. New protocol after N5–N7 and prompted by N6null/coverage. It does not replace the N6null or constitute fresh chance replication. No RNG/new seeds.

Fixed selector: all3hero cards remaining, at least2legal actions, reach support>=2;64evenly-spaced lexicographic keys per actor/first-trick-position,256per condition. Same unmodified N2posteriors/private TAopponents and same frozen N6solver/direct-table verifier. Every primary key finished before separate full verification. No outcome-based extension or changed tie rule.

| condition | keys / reach weight | positive fixed-decoder action-loss keys | positive max-value-optimism keys | positive any-action PI-gap keys | exact action loss / reach | exact max-value optimism / reach |
|---|---|---:|---:|---:|---|---|
| S3R4n3_t0 | 256 / 7645 | 1 | 2 | 4 | 8/1529 (0.005232178) | 64/7645 (0.008371485) |
| S3R4n3_t-1 | 256 / 9154 | 0 | 2 | 3 | 0 (0.000000000) | 110/4577 (0.024033210) |

Exact deterministic-subset diagnostic means, NOT all-infoset/population/prevalence estimates. No sampling CIs; denominators are selected posterior reach weights. Each key is a different unilateral intervention, not one global policy. Max-value optimism and current-action loss are different measurements. NTnulls and all other tested keys retained.

## Explicit finite certificates and post-run tie sensitivity

Every key with ANY action-level PI_Q>BR_Q has a certificate containing the complete exact posterior worlds, future own/public infoset score matrices and column sums. A PI continuation maximizes separately by world; a private continuation picks one common column. These tables can be checked without the recursive solver. Their existence/algorithm interpretation remains pending external audit.

| condition / key | posterior worlds | actions | PI Q sums | private BR Q sums | fixed-tie loss | best PI-tie loss | worst PI-tie loss | tie-robust? |
|---|---:|---|---|---|---|---|---|---|
| S3R4n3_t0 / 1434 | 192 | [2, 4, 9] | [1920, 1480, 1480] | [1920, 1470, 1480] | 0 | 0 | 0 | False |
| S3R4n3_t0 / 507870 | 14 | [4, 6, 10] | [140, 140, 140] | [100, 140, 140] | 20/7 | 0 | 20/7 | False |
| S3R4n3_t0 / 1614307 | 7 | [1, 2, 6] | [60, 60, 0] | [36, 36, 0] | 0 | 0 | 0 | False |
| S3R4n3_t0 / 1641696 | 12 | [6, 7, 10] | [120, 120, 40] | [80, 80, 40] | 0 | 0 | 0 | False |
| S3R4n3_t-1 / 1802 | 381 | [3, 5, 10] | [3696, 2453, 2849] | [3608, 2453, 2849] | 0 | 0 | 0 | False |
| S3R4n3_t-1 / 3113 | 128 | [4, 7, 9] | [154, 1342, 330] | [154, 1210, 330] | 0 | 0 | 0 | False |
| S3R4n3_t-1 / 492695 | 178 | [4, 6] | [638, 748] | [616, 748] | 0 | 0 | 0 | False |

The predeclared PIMC decoder breaks PI ties by the lower card. If best-PI-tie loss is zero, its current-action error is TIE-SENSITIVE and cannot establish that every PI-optimal decoder loses. We do not change the frozen decoder after this diagnostic. Strict future optimism can occur with no current-action loss; do not conflate them.

Independent verification reenumerated every512posterior support and matched all integer action values against direct contingent-action tables. The certificate pass reenumerated the positive-gap supports again and reconstructs worldwise/column sums explicitly. Both paths share the fixed private policy definition and established N2reachable-chance law; no claim of a second independent TApolicy definition.

No MaxN opponent-model substitution, approximate belief, joint team optimality, solved large-game prevalence,52-card effect, novelty or material magnitude claim. New computational witnesses/correctness argument are pending Claude external. Primary raw values, quotas, selectors, sources and all nulls stay unchanged/local.

Commands (cd engine): recorded JUDGEMENT_JOB_DEADLINE python experiments/round12/bounded_job.py experiments/round12/N11/run.py; then separate certificates.py and report.py under a fresh logged bound. Sources/results/checkpoints/certificates retained.
