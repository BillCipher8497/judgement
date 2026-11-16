# Round 7 audit (Track C): proof assumptions and headline claims vs raw results

Date: 2026-11-15 (ET). Auditor: Claude Code session, independent of Codex. Scope: `math/NOTES.md` (M1–M10, H1–H3), the
"Results so far" list in `AGENTS.md`, and the round STATUS logs through round 6. No existing file was edited: every
proposed change is listed in **§5 Proposed edits** for Codex to integrate. Recomputations read the saved raw arrays
only. They are in `scripts/`, outputs in `results/`, checkpoints in `STATUS.txt`.

Verdict legend: **OK** (reproduces / proof checks out) · **QUALIFY** (true as measured but the headline wording
claims more than the evidence) · **STALE** (text disagrees with the table it summarises) · **WRONG** (not supported).

---------------------------------------------------------------------------------------------------------------------

## 1. Headline claims in AGENTS.md: verdicts

| # | AGENTS.md claim | verdict | what the raw results say |
|---|---|---|---|
| 1 | "bid the mode, not the mean (86.8% match)" | **QUALIFY** | 86.8% is from **Greedy** play (phase1 `phase1_tables.md` l.358: oracle bid == mode 0.868, == round(E[t]) 0.637, n = 4,160 situations, Wilson ±0.010). No mode-match figure under TA/TA+ play exists. The proved statement (M6) is "floor or ceil of the mean", so "not the mean" needs rewording (see §5). |
| 2 | "Proved for independent cards that the best bid is always the mean rounded down or up" | **OK** (with scope) | T6 + L3′ lower side checked line by line (§2). Scope: unconstrained bid under the independent-card (PB) law. A hooked dealer can be forced outside {⌊μ⌋, ⌈μ⌉}, and real laws are not PB (especially NT, δ ≈ 1.5–1.8). |
| 3 | "A model of who holds the lead closes 98%, 82% and 30% of the NT bid gap at n = 5, 8, 13" | **QUALIFY** | Numbers reproduce, but each n uses a different model (LC1 at 5, LCX/LC2 at 8, LCX at 13), all for TA play only (TA+ at n=13: LCX −0.08), with no CI. A near-saturated control **without** a lead state (LCU) reaches the floor (0.99–1.00) at every n, including n = 13, so reaching the floor does not single out the lead. The discriminating evidence is LC0 (3 parameters) 0.58 vs trick-indexed TPB (n parameters) 0.04, plus persistence +0.31. CIs and details in §3.3. |
| 4 | "Hook tax … 8–10 points per round with all cards visible; 5.4 (n=1) falling to 0.27 (n=13) with hidden cards" | **QUALIFY** | All-cards-visible (phase2 F3, rational bids) dealer no-hook − hook: 10.17, 9.64, 8.92, 8.21, 7.82, 6.96 at n = 1..6 (n = 2000/2000/2000/1200/600/90). So "8–10" covers n ≤ 4 only, and n ≥ 7 was never measured. At n ≥ 3 the comparison is between PI rational bidding and **TA self-play**, so information is confounded with policy. Information and policy are both held fixed only at n = 1 (10.17 vs exact 5.48) and n = 2 (9.64 vs equilibrium 3.29, M8). Also, with all cards visible the hook costs **non-dealers** too (pos1 +1.48 ± 0.29, pos2 +1.19 ± 0.27 at n = 5), so "a tax on the dealer" holds for hidden cards (others ≤ 0.09, E5) but not fully for visible cards. |
| 5 | "Cooperation surplus … 81% at n=5 with all cards visible; with hidden cards about a third (26–31% at n=5; 31% of a full game)" | **OK numerically, QUALIFY framing** | The 81% (phase2 scheme R) uses a PI max^n hero with PI-rational bids, while the hidden shares use a TA+ hero. Holding the TA+ hero fixed, the paired ratio hidden/omniscient surplus is **0.296 [0.264, 0.327]** at n = 5 (heuristic, 4,000 deals), 0.356 [0.313, 0.398] (PIMC), 0.329 [0.268, 0.386] (MC), and 0.30–0.32 for the heuristic at every n = 3..7 (§3.1). So "about a third" survives the fairer comparison, now with a CI. Full game 0.311 [0.306, 0.317] reproduces. Spoilers bid like everyone else, so this is a **play-only** surplus. |
| 6 | "The gap comes from information, not from how strong the spoilers are" | **WRONG as stated** | What was shown: (a) the same paranoid solver given the true hands reaches the PI curve; (b) three hidden-information spoilers (heuristic, PIMC-paranoid K=16, policy-aware MC) plateau at about ⅓. H3.1 only brackets the truly attainable hidden-information surplus: at n = 5 it lies between ≈0.31 (best spoiler found) and ≈0.84 (omni). No result rules out a stronger hidden-information coalition (e.g. signalling conventions, CFR-style coordination). NOTES itself says "under the caveat that a better hidden-information spoiler may exist"; the headline drops that caveat. |
| 7 | "Play rules checked against the solver, two proved: highest sure loser at target; highest spare winner at need 1" | **QUALIFY** | Proved **under Assumption O** (open-loop opponents) and **for same-suit pairs** only (M9). The checks were against TA opponents (`m9_play.py`), not the exact solver. The TA+3 component uses the any-suit version, which is heuristic (+0.030 ± 0.010/round, fresh seeds). |
| 8 | "Bots vs 3×TA: TA+3 +7.8; Hybrid +17; Hybrid-BI +20.7" | **OK numbers, QUALIFY ranking** | Recomputed exactly: TA+3 **+7.98 ± 0.43** (n = 10,000; "+7.8" is the guide-run figure 7.79 ± 0.55, n = 6,000), Hybrid c5 +17.07 ± 1.32 (1,500), Hybrid-BI +20.74 ± 1.62 (1,000). Against 3×TA, Hybrid-BI is **not separable** from Hybrid+oracle bidder (+0.54 ± 1.73) or Hybrid K32 (+0.96 ± 1.11) on the same 1,000 games. It is separable from Hybrid c5 (+3.37 ± 1.14). Hybrid-BI's edge shows in the 3×Hybrid pool (+5.31 ± 1.59), not against 3×TA. |
| 9 | "Fixed bid model fragile (−4.0 vs honour counters); Hybrid-RB +2.4 to +6.5 over Hybrid in every pool" | **OK, QUALIFY scope** | Reproduces from r5/r6 tables (+6.51, +2.45, +2.72, +3.58, +2.37). Four of the five pools are **inside RB's type library**: model, noisy (pool p = .3 vs library .1/.25/.5), honour (exact), and a mix of these. The one out-of-library pool (guide bidders) gives the smallest gain, +2.37 ± 2.28 (lower bound ≈ +0.1). |
| 10 | "Top pool (3×Hybrid-BI K32 c6) still beatable by +3.4 ± 2.7" | **QUALIFY (weak)** | Pooled 422 games include the 142 discovery games, where the gap was first seen. On the **replication-only** games (g ≥ 142, n = 280): RB K32c7 − pool **+2.94 ± 3.28** and BI K32c7 − pool **+1.21 ± 3.23**, both not significant. The challenger also searches deeper (c7 vs the pool's c6), so this measures a compute advantage, not exploitability at equal strength. |
| 11 | "CFR+ reaches NashConv ≈ 0 for 2, 3, 4 players, but equilibria are not unique" | **OK (scope)** | NashConv is computed exactly by best response, so ε is verified rather than relying on a CFR guarantee (multiplayer CFR has none). 3p_nt plateaus at 0.0056 from a uniform start (warm starts reach 2–4e-4), and 3p_n3_t stopped at 0.0061 after T = 100. Non-uniqueness is material only in 3p_nt and 4p_nt (payoff spread up to 0.21 / 0.17). Other games spread ≤ 0.016. All toys are tiny (2×4 or 3×3 decks, n ≤ 3). |
| 12 | "Human guide keeps 81% of TA+3's edge over greedy; bidding is the main loss" | **OK numbers, QUALIFY framing** | 0.808 [0.794, 0.822] reproduces from the S3 log. The denominator is dominated by greedy's −43 pts/game. In absolute terms the guide is **−2.01 ± 0.78 vs TA**, i.e. worse than the plain TA bot, which keeps 0.848. The bidding > play loss split holds in both decomposition orders (bidding 5.50–7.72, play 2.08–4.31). |

---------------------------------------------------------------------------------------------------------------------

## 2. Proof audit (math/NOTES.md)

| item | verdict | notes |
|---|---|---|
| P0 dispersion identity | OK | Exact algebra; exact for plug-in estimators. |
| L1 PB log-concave ⇒ EV log-concave, maximisers ≤ 2 adjacent | OK | Hoggar / Newton correctly cited; strictness from log(H+b) strictly concave on the support. |
| L2 (i) b* ≥ m_lo any law; (ii) ratio rule; (iii) b* ∈ {m_hi, m_hi+1} under ULC | OK | (iii) uses ULC(∞): r_{j+1} ≤ r_j·j/(j+1), then (m+1)/(m+2) < (H+m+1)/(H+m+2). Correct. |
| L3 Darroch bracket | OK | |
| **M6 Lemma R / T6** (upper side of L3′, now "proved") | **OK** | Checked every step. Hoeffding (1956) Thm 4 reduction to p ∈ {0, a, 1}; the case j < m′ is trivial; (j−m′)a ≤ m′(1−a) ⇔ ja ≤ m′; ratio ≤ m′/(m′+1) ≤ m/(m+1); T6 contradiction uses H > 0 and no internal zeros (support [#ones, #nonzero] contains m). Sharp constant: d/da of (λ − m′a)/(1−a) has sign λ − m′ > 0, so j = m′+1 is extremal and s_N = max_m (m+1)/(1 + c_m(m+1)). Checked N = 1: 11/21; N = 13: 13/(1 + 13·22/23) = 0.9676. **Stale status line:** the round-1 status table (NOTES l.379) still says the upper side is only "verified numerically". The round-2 table supersedes it, but a reader of round 1 alone gets the old status. |
| **M2 P1 "unique PBE"** | **QUALIFY** | In **NT** the followers never bid 1 (NOTES l.228), so every history with a follower's 1 is off the equilibrium path. There, PBE beliefs (and hence later seats' actions) are unrestricted. Uniqueness holds **on path**, which covers all values and on-path strategies. In trump (S), every bid history has positive probability (thresholds ♠4+/♠T+/♠K+/♠A all occur), so S is unique outright. Suggested status: "proved: unique on the equilibrium path (unique outright for trump)". |
| M2 closed forms, P2 monotone thresholds | OK | Exhaustive integer enumeration; follower posterior in NT ≤ 12/51 < 10/21 checked. |
| M3 P3 "at most N−1 hit" | OK | Pure conservation. The **sentence after it** ("so it is the dealer's cost to bear") is an interpretation, not part of the proof. PI F3 shows non-dealers also lose 0.24–2.10 pts/round from the hook at n = 2..6. |
| M4 **P4 "tent"** | **QUALIFY** | "D = 0 maximises the joint mass on the all-hit vector" is true, but it does not imply that the *per-player* hit rate P(e_i = 0 \| Σe = D) is unimodal in D with peak at 0. That needs an extra property of π (e.g. symmetric log-concave errors). It is supported empirically (tent within .02–.04), so the label should be "heuristic, verified numerically", not Proposition. |
| M7 A1 calibration | OK | Even/odd split = proper out-of-sample. |
| M8 2-card equilibrium | OK (MC) | Correctly labelled "verified (MC)", with a 0.35% near-tie limit cycle. |
| **M9 Lemma E, Prop W, Prop L** | **OK under O** | The divergence after τ in Prop W is harmless because the keep-a world has already overshot (certain miss), so "pathwise" holds. Prop L reduces to W after an identical trick. Both need **Assumption O** and a **same-suit** pair; the "4 worse rounds" are attributed to O-violations without a check. |
| H3.1 sandwich | OK | Per deal, hero-vs-coalition with all hands known is a finite two-person zero-sum game of perfect information. A possibly randomised π satisfies min_τ S(π, τ) ≤ V_par. omni ≠ best response to π, so the omni surplus is a *lower* bound on the PI-attainable surplus against π, as stated. |
| H3.2 hero-led tricks information-free | OK | Key step: the current best stays the hero's card until some spoiler beats it, so each spoiler's ability to beat depends only on its own hand. |
| H3 counterexample | OK | Solver-verified in tests. |
| H1 finding 2 text | **STALE** | The text quotes omni 20/47/71/85/92%, PIMC 11/17/23/31/35%, "3.57 ± 0.23", "(0.854 − 0.305)/0.854 = 64%". The table directly above it gives omni 21/48/68/85/92, PIMC 13/17/20/32/35, omni − PIMC 3.44 ± 0.32 at n = 5. The text comes from the interim full-deal-set run (STATUS 00:32). Recomputed on the full deal sets (§3.1): omni 0.204/0.470/0.692/0.843/0.912, PIMC 0.111/0.170/0.231/0.305/0.346. "At least 64% … not attainable by a **search-based** coalition" should say "not attained by the three spoilers tried". |

---------------------------------------------------------------------------------------------------------------------

## 3. Recomputations from raw results

### 3.1 Hidden-information surplus, hero held fixed (`scripts/a1_surplus.py` → `results/a1_surplus.json`)
CRN check: the self-interested arrays are bit-identical across the main, omni, PIMC and MC files on shared deal
indices (all True). Ratio = mean(hidden surplus)/mean(omni surplus) on the same deals; 95% paired deal bootstrap
(B = 4,000).

| n | deals (omni) | omni share | heuristic share | heur/omni | PIMC/omni (deals) | MC/omni (deals) |
|---|---|---|---|---|---|---|
| 2 | 4000 | 0.204 [0.189, 0.220] | 0.094 | 0.459 [0.398, 0.516] | 0.563 [0.483, 0.637] (2000) | 0.543 [0.418, 0.653] (1000) |
| 3 | 4000 | 0.470 [0.451, 0.490] | 0.147 | 0.313 [0.269, 0.353] | 0.363 [0.302, 0.421] | 0.329 [0.246, 0.410] |
| 4 | 4000 | 0.692 [0.674, 0.710] | 0.219 | 0.316 [0.281, 0.350] | 0.327 [0.280, 0.374] | 0.283 [0.211, 0.353] |
| 5 | 4000 | 0.843 [0.828, 0.857] | 0.250 | 0.296 [0.264, 0.327] | 0.356 [0.313, 0.398] | 0.329 [0.268, 0.386] |
| 6 | 4000 | 0.912 [0.900, 0.923] | 0.284 | 0.311 [0.280, 0.341] | 0.378 [0.333, 0.418] | – |
| 7 | 2000 | 0.960 [0.948, 0.970] | 0.311 | 0.324 [0.280, 0.367] | 0.333 [0.270, 0.394] (1000) | – |
| 8 | 1000 | 0.971 [0.957, 0.984] | 0.378 | 0.389 [0.328, 0.446] | – | – |

Full games (4,000, ties counted as wins): share s1 0.091 [0.086, 0.097], s2 0.201 [0.195, 0.207], s3 0.311
[0.306, 0.317]; hero P(win) self 0.246 ± 0.013, s1 0.212, s2 0.154, s3 0.170 (split-ties values 0.242/0.207/0.150/
0.167). No conclusion changes. Two-spoiler P(win) is still the lowest.

### 3.2 Bot ladder (`scripts/a2_bots.py` → `results/a2_bots.json`; Δ = challenger − pool policy, same game and seat)

| comparison | n | Δ pts/game | ΔP(win) ties-as-win |
|---|---|---|---|
| TA+3 − TA | 10000 | +7.98 ± 0.43 | +0.088 ± 0.009 |
| Hybrid c5 − TA | 1500 | +17.07 ± 1.32 | +0.209 ± 0.027 |
| Hybrid+oracle − TA | 1500 | +20.46 ± 1.57 | +0.241 ± 0.030 |
| Hybrid K32 c5 − TA | 1000 | +19.78 ± 1.56 | +0.251 ± 0.033 |
| Hybrid-BI − TA | 1000 | +20.74 ± 1.62 | +0.272 ± 0.032 |
| Hybrid-BI − Hybrid c5 (g < 1000) | 1000 | +3.37 ± 1.14 | +0.055 ± 0.029 |
| Hybrid-BI − Hybrid+oracle (g < 1000) | 1000 | +0.54 ± 1.73 | +0.016 ± 0.035 |
| Hybrid-BI − Hybrid K32 c5 (g < 1000) | 1000 | +0.96 ± 1.11 | +0.021 ± 0.027 |
| C2: RB K32c7 − pool, all | 422 | +3.39 ± 2.72 | +0.024 ± 0.050 |
| C2: RB K32c7 − pool, discovery g < 142 | 142 | +4.29 ± 4.86 | −0.014 ± 0.085 |
| C2: RB K32c7 − pool, **replication g ≥ 142** | 280 | **+2.94 ± 3.28** | +0.043 ± 0.063 |
| C2: BI K32c7 − pool, discovery g < 142 | 142 | +5.82 ± 4.04 | +0.021 ± 0.084 |
| C2: BI K32c7 − pool, **replication g ≥ 142** | 281 | **+1.21 ± 3.23** | +0.075 ± 0.061 |

Data note: games 142–159 of the C2 pool reference exist twice (round 4 `f_g2_ref_*` and round 6 `f_c2x_ref_*`).
The 18 copies are **bit-identical** (same seed base 600M, seat, totals), so they are harmless for the round-6
analyser, which pairs by unique id. They also confirm cross-round reproducibility. My loader drops identical
duplicates and asserts on conflicts.

### 3.3 Lead-state model "gap closed" with CIs (`scripts/a3_lead.py` → `results/a3_lead.json`, per-hand arrays `results/a3_lead_*.npz`)
Re-runs `math/m10_analyze.analyze_hand` unchanged on the stored rollouts and keeps the per-hand cross-fit bid
matches. gap = (xmatch_M − xmatch_PB)/(floor − xmatch_PB), paired hand bootstrap (B = 4,000). The pmf of T has n
free values, while LC1 has 2n−1 parameters and LCX n+4.

| play | n | hands | PB xmatch → floor | LC0 (3 par.) | TPB (n par.) | LC1 (2n−1) | LC2 | LCX (n+4) | LCS | SB | **LCU (no lead state)** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TA | 5 | 1000 | .739 → .982 | .58 [.49, .65] | .04 [.02, .07] | **.98 [.94, 1.02]** | .99 [.96, 1.02] | .98 [.94, 1.01] | .73 [.66, .80] | .50 [.43, .57] | 1.00 [.99, 1.01] |
| TA+ | 5 | 1000 | .648 → .962 | .64 [.58, .70] | .05 [.03, .08] | 1.00 [.98, 1.02] | 1.00 [.98, 1.02] | .99 [.96, 1.01] | .75 [.69, .80] | .45 [.39, .50] | .99 [.98, 1.00] |
| TA | 8 | 1000 | .727 → .949 | −.07 [−.24, .09] | .08 [.03, .12] | .71 [.62, .79] | **.82 [.74, .89]** | **.82 [.75, .89]** | .50 [.41, .59] | .38 [.31, .45] | 1.00 [.98, 1.02] |
| TA+ | 8 | 1000 | .701 → .938 | .11 [−.04, .24] | .05 [.01, .09] | .86 [.78, .93] | .92 [.86, .98] | .81 [.74, .89] | .50 [.40, .59] | .37 [.30, .44] | 1.00 [.98, 1.01] |
| TA | 13 | 996 | .580 → .954 | .07 [−.02, .16] | .09 [.06, .13] | .19 [.11, .27] | .41 [.34, .48] | **.30 [.23, .36]** | .66 [.59, .72] | .20 [.14, .25] | 1.00 [.99, 1.01] |
| TA+ | 13 | 999 | .707 → .958 | −.56 [−.77, −.38] | −.04 [−.09, .00] | −.26 [−.42, −.12] | .06 [−.07, .17] | −.08 [−.20, .03] | .33 [.22, .43] | .20 [.13, .26] | .99 [.98, 1.00] |

All point estimates reproduce `math/results/m10_tables.txt` exactly. Readings:
- The headline's 98/82/30% are TA-only, and the model changes with n (bold cells). Under TA+ at n = 13 every lead
  chain except LCS is at or below PB.
- **LCU, which conditions on tricks-won-so-far and has no lead state, reaches the floor at every n, including n = 13
  where the lead chains fail.** So "closes x% of the gap" with flexible models measures flexibility, not the lead
  mechanism. The mechanism evidence is the parsimonious contrast at n = 5 (LC0, 3 parameters, 0.58 vs TPB, n
  parameters, 0.04), the persistence statistic, and the LL/LF covariance decomposition (M5).
- Runtime n = 8 and 13, both policies: ~3 min (one process).

---------------------------------------------------------------------------------------------------------------------

## 4. Method and process observations

1. **Seed reuse for selection.** TA+3's variant ("L any suit" over "same suit") was selected on the same 71M F1 games
   that report its +7.98 (a two-way choice; M9's fresh-seed check supports the direction). The guide's defaults
   were simplified after reading the S3 results on the 71M evaluation seeds and then re-evaluated on those same seeds.
   The score went down (0.823 → 0.808), so this did not inflate the result, but the final number is not a held-out
   validation. Guide bidding-line selection itself used TRAIN loss only (`s1_back.py`), which is correct.
2. **Discovery vs replication.** The round-4 C2 claim (+5.82, n = 142) shrank to +1.21 on new games (§3.2). Future
   "beatable" claims should report the replication sample separately.
3. **Hero position.** All dependence results (M1, M5, M10) use hero = first bidder = first leader (L₁ = 1). The
   lead-state findings are untested for seats that do not lead trick 1.
4. **Descriptive vs usable.** PB/SB/LC models are fitted per hand from that hand's own 150–300 rollouts. "Closes x%
   of the bid gap" measures representational adequacy, not a bidder a human or bot could run without rollouts.
5. **Missing CIs on headline ratios.** "share", "gap closed", "keeps x%" were reported without CIs in several places
   (conventions rule 2). §3 supplies them for the surplus ratio and gap-closed fractions. The guide's 0.808 already has one.
6. **Ties.** Recomputing ties-as-win changes nothing material anywhere it was checked (H1 games, bot ladder).

---------------------------------------------------------------------------------------------------------------------

## 5. Proposed edits (for Codex; not applied)

**AGENTS.md "Results so far"** — suggested replacements:
- Bidding: "Bid at the mode or one above it (L2). Under Greedy play the oracle bid equals the mode in 86.8% ± 1.0% of
  4,160 situations vs 63.7% for the rounded mean. Proved for independent cards: the unconstrained best bid is ⌊μ⌋ or ⌈μ⌉."
- Lead: "In NT, wins in different suits move together because winning gives the lead. A per-hand lead chain fits the
  trick law to the noise floor at n = 5 (LC1, 0.98 [0.94, 1.02] of the PB→floor bid gap), 0.82 [0.75, 0.89] at n = 8
  (LCX) and 0.30 [0.23, 0.36] at n = 13 (LCX; TA+ −0.08) (TA play, hero = opening leader). The parsimonious evidence:
  a 3-parameter lead chain closes 0.58 at n = 5, a trick-indexed control without the lead 0.04; lead persistence +0.31."
- Hook: "Dealer's hook cost, all cards visible (PI rational bids): 10.2, 9.6, 8.9, 8.2, 7.8, 7.0 at n = 1..6. Hidden
  cards: exact 5.48 at n = 1, equilibrium 3.29 at n = 2; TA self-play 5.4 → 0.27 at n = 1..13. With visible cards
  non-dealers also pay (up to 1.5/round)."
- Surplus: "… With hidden cards the best spoilers we built recover 0.30 [0.26, 0.33] of what a cheating spoiler with
  the same solver recovers at n = 5 (0.30–0.32 at n = 3..7; same TA+ hero). Whether any hidden-information coalition
  could do better is open: at n = 5 the attainable share lies between ≈0.31 and ≈0.84."
- Play rules: "… two proved for same-suit pairs against open-loop opponents (M9), and confirmed against TA opponents."
- Bots: "TA+3 +8.0 ± 0.4 (n = 10,000); Hybrid +17.1 ± 1.3; Hybrid-BI +20.7 ± 1.6 (tied with Hybrid+oracle and
  Hybrid K32 against 3×TA; ahead by +5.3 ± 1.6 against 3×Hybrid)."
- Top pool: "3×Hybrid-BI K32c6 was beaten by a deeper-search (c7) RB challenger by +3.4 ± 2.7 (n = 422), but the
  replication-only games give +2.9 ± 3.3 (n = 280): not established."
- Guide: append "(−2.0 ± 0.8 pts/game vs TA, i.e. below the plain TA bot)".

**math/NOTES.md status tables:**
- Round-1 table, L3′ row: point to M6 ("upper side proved in round 2, T6").
- P1: "unique on the equilibrium path; unique outright in trump rounds (NT has off-path follower histories)".
- P4: relabel from Proposition to "heuristic, verified numerically (tent within .02–.04)".
- P3: move "dealer's cost to bear" out of the proposition. Cite F3 (PI non-dealer cost) and E5 (hidden ≤ 0.09).
- H1 finding 2: replace the interim numbers with the table values or the §3.1 full-deal-set values; "search-based
  coalition" → "the three hidden-information spoilers tried".
- Prop W/L rows: add "same-suit pair" explicitly next to "under open-loop opponents".

---------------------------------------------------------------------------------------------------------------------

## 6. Files, commands, limitations, next steps

**New files (all under `engine/experiments/round7_audit/`):** `AUDIT.md`, `STATUS.txt`, `scripts/a1_surplus.py`,
`scripts/a2_bots.py`, `scripts/a3_lead.py`, `results/a1_surplus.json`, `results/a2_bots.json`,
`results/a3_lead.json`, `results/a3_lead_{ta,ta_plus}_n{5,8,13}.npz`, `results/a3_lead.log`.
No existing file was modified, and nothing was staged or committed. Full test suite after the audit:
`python -m pytest -q -p no:cacheprovider` → 249 passed, 1 skipped (`test_openspiel.py`: `pyspiel` not installed on
this machine; it is an optional dependency, so the OpenSpiel cross-validation was not re-run here).

**Commands** (from `engine/experiments/round7_audit`):
```
python -I scripts/a1_surplus.py          # < 10 s
python -I scripts/a2_bots.py             # < 10 s
python scripts/a3_lead.py --cfg 5        # ~2 min, one process
python scripts/a3_lead.py --cfg 8,13     # ~3 min, one process (background); reuses saved per-hand npz if present
```

**Limitations.** The proof audit is a careful read, not a machine check. Hoeffding's theorem was checked against its
statement, not re-derived. I did not re-run any simulation that generates games; all numbers come from saved arrays.
Phase-2 PI values and phase-1 E5 were taken from their summary tables, not recomputed. I did not audit the engine
code for rule bugs; that is covered by existing tests and the OpenSpiel cross-validation.

**Next steps (suggested).**
1. Run a fresh-seed replication of the top-pool claim at equal search depth (c6 vs c6) before it goes in the paper.
2. Build a stronger hidden-information spoiler (e.g. CFR or a signalling convention in a small toy where
   PI/hidden values can be computed exactly) to tighten the [0.31, 0.84] bracket. This is the paper's central number.
3. Measure the lead effect for a hero who does not lead trick 1 (positions 1–3).
4. Report a mode-match figure under TA/TA+ play, so the bidding headline uses the same play policy as the rest.
5. Add a short "conventions" check to future rounds: discovery and replication games reported separately; CIs on all ratios.
