# Math layer: lemmas, proofs, and tests against simulation

Rules as in the repo (4 players, score (10+b)·1{t=b}, hook on dealer, must follow suit,
no obligation to trump). H = 10 is the hit base. EV(b) = (H+b)·P(b). Every empirical
number is reported with its count and a 95% CI (± = 1.96·SE over hands/rounds, or Wilson
for proportions where stated). Scripts and seeds: see the end of this file. Result
tables are in `math/results/*.txt|json`. Status of each claim: **proved**, **verified
numerically** (exhaustive or randomized), or **refuted**.

---------------------------------------------------------------------------------

## M1: the independent-card (Poisson-binomial) model

Let X_i = 1{hero's card i wins a trick}, T = Σ X_i, p_i = E X_i. The *independent-card
model* replaces the law of T with PB(p) = law of a sum of independent Bernoulli(p_i).
It matches E[T] by construction.

**Identity P0 (dispersion = covariance). Proved.**
Var T − Var_PB = 2 Σ_{i<j} Cov(X_i, X_j), with Var_PB = Σ p_i(1−p_i). So the dispersion
index δ = Var_emp/Var_PB is above 1 exactly when the average pairwise covariance between
card wins is positive, and below 1 exactly when it is negative. The identity holds exactly for the
plug-in estimators used on the rollouts, because the sample variance of X_i equals p̂_i(1−p̂_i).

**Lemma L1 (log-concavity, unique bid). Proved.**
If T ~ PB(p) then P is log-concave with no internal zeros. It is in fact ultra-log-concave:
P(j)² ≥ P(j−1)P(j+1)·(1+1/j)(1+1/(k−j)) (Newton's inequalities), because the generating
polynomial Π(1−p_i+p_i z) has only real roots. b ↦ H+b is positive and linear, so it is
log-concave. A product of log-concave sequences is log-concave, so EV(b) is log-concave.
EV is in fact strictly log-concave, since log(H+b) is strictly concave. Hence the maximisers of EV form a
set of at most two adjacent integers ("unique up to ties"). EV(b+1)/EV(b) is strictly
decreasing, so b* = min{b : EV(b+1) ≤ EV(b)}. Local search is global, and when the
hook forbids b* the best legal bid is b*−1 or b*+1.
Proof of log-concavity: each Bernoulli pmf (1−p, p) is log-concave, and the convolution of log-concave
sequences is log-concave (Hoggar 1974).

**Lemma L2 (mode bracket and ratio rule).**
(i) *Any* distribution (not just PB): b* ≥ m_lo, the smallest mode. **Proved.** For b < m_lo,
EV(b) = (H+b)P(b) < (H+m_lo)P(m_lo).
(ii) b+1 beats b iff P(b+1)/P(b) > (H+b)/(H+b+1) (= 10/11, 11/12, … ). **Proved** (rearrange).
(iii) For PB (indeed any ultra-log-concave law): b* ∈ {m, m+1} with m = m_hi the largest
mode, and b* = m+1 iff P(m+1)/P(m) > (H+m)/(H+m+1). **Proved.** Write r_j = P(j)/P(j−1).
ULC gives r_{j+1} ≤ r_j · j/(j+1). Since m is the largest mode, r_{m+1} < 1, so
r_{m+2} < (m+1)/(m+2) < (H+m+1)/(H+m+2) (as H > 0). Hence EV(m+2) < EV(m+1), and by L1, EV
keeps decreasing after that. If two modes m−1, m exist, EV(m) > EV(m−1) by (i).
So bidding above the mode needs a near-tie: P(m+1) within a factor (H+m)/(H+m+1) of P(m).
That is a 9% band at m=0 and a 5% band at m=8.

**Lemma L3 (mean bracket).**
Darroch (1964): every mode of PB satisfies |m − μ| < 1. With L2: ⌊μ⌋ ≤ b* ≤ ⌈μ⌉+1.
**Proved.**
*Tightening L3′:* |b* − μ| < 1, i.e. b* ∈ {⌊μ⌋, ⌈μ⌉}, and b* = μ when μ is an integer.
The lower side is **proved**: b* ≥ m_lo > μ−1. The upper side is **proved for the binomial**:
if μ = kp ≤ m then r_{m+1} = (k−m)p/((m+1)(1−p)) ≤ μ/(m+1) ≤ m/(m+1) < (H+m)/(H+m+1), so
b* = m+1 forces μ > m. For general PB the upper side is **verified numerically**:
39,150 vectors (random families plus 150 hill-climbs maximising b*−μ), with
sup(b*−μ) = 0.94 random and 0.85 adversarial, and no violation (`results/m1_lemmas.json`).

**Corollary C1 (certificates of dependence). Proved.**
Under independence, P is log-concave and ULC. So any of the following, when it holds
beyond MC noise, *certifies* that card outcomes are dependent:
(a) an interior dip (bimodality);
(b) b*_emp ≥ m_hi+2;
(c) δ ≠ 1.
Note the converse fails: dependent outcomes can still look log-concave.

### M1 empirical test
**Design.** The script is `m1_sim.py` and the seeds are fixed. It deals 1,000 hands for each of n ∈ {5, 8, 13} × {trump S, NT}, so
6,000 hands in all. The hero is the first bidder (it bids first and leads). For each hand and each play
policy, the round is played 300 times: hidden hands are resampled, every seat bids with its policy's
fitted bidder model, and every seat then plays with that policy. TA and Greedy are paired on the same deals.
The script records P_emp(t) and p_i = P(card i wins a trick). Under TA the law depends on the hero's bid. The
main run uses the model bid. A bid-conditional oracle check on a subsample is reported in M1b below.
The noise floor is the agreement between argmax EV on rollout halves A and B (150 rollouts each).

**Fit and bid accuracy by n and trump type** (`results/m1_tables.txt`):

| play | n | type | hands | TV(emp, PB) | δ mean | frac δ<1 | PB bid = emp-opt | A/B noise floor | EV loss PB bid |
|---|---|---|---|---|---|---|---|---|---|
| ta | 5 | trump | 1000 | 0.072 ± 0.003 | 1.137 ± 0.015 | 0.29 | 0.878 ± 0.020 | 0.90 | 0.11 ± 0.02 |
| ta | 5 | NT | 1000 | 0.185 ± 0.006 | 1.776 ± 0.029 | 0.02 | 0.750 ± 0.027 | 0.98 | 0.83 ± 0.11 |
| ta | 8 | trump | 1000 | 0.084 ± 0.003 | 1.005 ± 0.015 | 0.54 | 0.879 ± 0.020 | 0.89 | 0.11 ± 0.03 |
| ta | 8 | NT | 1000 | 0.175 ± 0.005 | 1.718 ± 0.030 | 0.05 | 0.717 ± 0.028 | 0.95 | 0.72 ± 0.09 |
| ta | 13 | trump | 1000 | 0.132 ± 0.005 | 0.786 ± 0.017 | 0.83 | 0.867 ± 0.021 | 0.92 | 0.16 ± 0.03 |
| ta | 13 | NT | 1000 | 0.191 ± 0.006 | 1.549 ± 0.031 | 0.10 | 0.583 ± 0.031 | 0.95 | 1.38 ± 0.12 |
| ta | all | trump | 3000 | 0.096 ± 0.002 | 0.976 ± 0.010 | 0.55 | 0.875 ± 0.012 | 0.90 | 0.13 ± 0.02 |
| ta | all | NT | 3000 | 0.184 ± 0.003 | 1.681 ± 0.018 | 0.06 | 0.683 ± 0.017 | 0.96 | 0.98 ± 0.06 |
| greedy | 5 | trump | 1000 | 0.087 ± 0.005 | 1.288 ± 0.018 | 0.14 | 0.765 ± 0.026 | 0.84 | 0.22 ± 0.03 |
| greedy | 5 | NT | 1000 | 0.143 ± 0.009 | 1.566 ± 0.033 | 0.06 | 0.752 ± 0.027 | 0.87 | 0.65 ± 0.09 |
| greedy | 8 | trump | 1000 | 0.098 ± 0.004 | 1.319 ± 0.020 | 0.14 | 0.758 ± 0.027 | 0.82 | 0.22 ± 0.03 |
| greedy | 8 | NT | 1000 | 0.115 ± 0.006 | 1.404 ± 0.028 | 0.13 | 0.735 ± 0.027 | 0.86 | 0.48 ± 0.07 |
| greedy | 13 | trump | 1000 | 0.116 ± 0.005 | 1.226 ± 0.028 | 0.34 | 0.764 ± 0.026 | 0.83 | 0.33 ± 0.05 |
| greedy | 13 | NT | 1000 | 0.167 ± 0.006 | 1.726 ± 0.030 | 0.05 | 0.612 ± 0.030 | 0.86 | 0.69 ± 0.08 |
| greedy | all | trump | 3000 | 0.100 ± 0.003 | 1.278 ± 0.013 | 0.21 | 0.762 ± 0.015 | 0.83 | 0.26 ± 0.02 |
| greedy | all | NT | 3000 | 0.142 ± 0.004 | 1.565 ± 0.018 | 0.08 | 0.700 ± 0.016 | 0.87 | 0.61 ± 0.05 |

**Hand classes** (pooled n):

| play | class | hands | δ mean | frac δ<1 | PB bid = emp-opt | EV loss |
|---|---|---|---|---|---|---|
| ta | long_trump>=5 | 184 | 0.854 ± 0.041 | 0.75 | 0.734 ± 0.064 | 0.31 ± 0.12 |
| ta | trump_len<=2 | 1883 | 1.047 ± 0.013 | 0.44 | 0.925 ± 0.012 | 0.07 ± 0.02 |
| ta | void>=1 | 2146 | 1.513 ± 0.025 | 0.20 | 0.805 ± 0.017 | 0.55 ± 0.06 |
| ta | no_void | 3854 | 1.227 ± 0.015 | 0.37 | 0.764 ± 0.013 | 0.55 ± 0.04 |
| ta | long_trump>=5&void | 36 | 0.931 ± 0.142 | 0.67 | 0.639 ± 0.159 | 0.82 ± 0.51 |
| greedy | long_trump>=5 | 184 | 0.960 ± 0.051 | 0.62 | 0.799 ± 0.058 | 0.37 ± 0.16 |
| greedy | trump_len<=2 | 1883 | 1.339 ± 0.015 | 0.13 | 0.750 ± 0.020 | 0.26 ± 0.03 |
| greedy | void>=1 | 2146 | 1.521 ± 0.021 | 0.10 | 0.716 ± 0.019 | 0.54 ± 0.05 |
| greedy | no_void | 3854 | 1.366 ± 0.014 | 0.17 | 0.739 ± 0.014 | 0.37 ± 0.03 |
| greedy | long_trump>=5&void | 36 | 1.234 ± 0.157 | 0.44 | 0.667 ± 0.156 | 0.85 ± 0.58 |

**Paired TA − Greedy on the same hands** (ΣCov = Σ_{i<j}Cov(X_i, X_j)):

| config | Δ ΣCov | Δ δ | hands |
|---|---|---|---|
| n=5|S | -0.060 ± 0.008 | -0.152 ± 0.024 | 996 |
| n=5|NT | 0.035 ± 0.012 | 0.211 ± 0.029 | 994 |
| n=8|S | -0.131 ± 0.011 | -0.313 ± 0.024 | 996 |
| n=8|NT | 0.096 ± 0.014 | 0.316 ± 0.032 | 996 |
| n=13|S | -0.218 ± 0.014 | -0.439 ± 0.030 | 990 |
| n=13|NT | -0.053 ± 0.016 | -0.177 ± 0.034 | 985 |

**Findings.**
1. *Independence is the wrong default. Positive dependence dominates.*
   Pooled δ = 1.33 ± 0.01 (TA) and 1.42 ± 0.01 (Greedy). NT is strongly over-dispersed in both
   (δ ≈ 1.55–1.78, with 87–98% of hands having δ > 1). The mechanism is lead control: winning a trick in NT gives the lead, and the lead runs
   a suit, so the cards of a long suit win together.
2. *Headline hypothesis (TA ⇒ under-dispersion): supported for trump, growing with n.
   Refuted for NT at small n.* With trump, TA lowers δ against Greedy on the same hands by −0.15, −0.31 and −0.44
   at n = 5, 8, 13. At n = 13, TA trump is under-dispersed: δ = 0.79 ± 0.02, with 83% of hands below 1.
   That is negative dependence, i.e. the hero steering toward its bid. Long-trump hands (≥5) are under-dispersed
   under TA (0.85 ± 0.04). They carry the control (trumps) needed to throw or take tricks at will. In NT at n = 5 and 8,
   TA is *more* dispersed than Greedy (+0.21, +0.32). TA reduces dispersion in NT only at 13 (−0.18).
   Greedy is never close to independent (δ ≈ 1.2–1.7).
3. *The PB bid is good, not exact.* It equals the empirical-optimal bid in 78% (TA) and 73% (Greedy) of hands,
   against an MC noise floor of 93% and 85%. EV loss is 0.55 and 0.43 points per round. Nearly all of it comes
   from NT (TA NT: 0.98/round; TA trump: 0.13).
4. *L2 generalises.* The empirical-optimal bid lies in [m_lo, m_hi+1] in 99.7% of TA hands and 98.4% of Greedy hands,
   although these laws are not PB. Certificates C1 are rare but present: bimodal laws occur in 2.8% ± 0.4% (TA) and 5.5% ± 0.6% (Greedy) of hands,
   and b*_emp ≥ m_hi+2 in 0.3% and 1.6%.
5. *Bid shading (E3) is half explained by PB.* Shading is s = b* − round(μ), binned by frac(μ):

| bin | emp s | PB s | hands |
|---|---|---|---|
| ta|frac[0,0.25) | -0.038 ± 0.015 | 0.001 ± 0.001 | 1908 |
| ta|frac[0.25,0.5) | 0.047 ± 0.016 | 0.084 ± 0.013 | 1796 |
| ta|frac[0.5,0.75) | -0.777 ± 0.026 | -0.436 ± 0.028 | 1283 |
| ta|frac[0.75,1.0) | -0.372 ± 0.035 | -0.014 ± 0.007 | 1013 |
| greedy|frac[0,0.25) | 0.010 ± 0.024 | 0.000 ± 0.000 | 1576 |
| greedy|frac[0.25,0.5) | 0.175 ± 0.026 | 0.091 ± 0.014 | 1553 |
| greedy|frac[0.5,0.75) | -0.489 ± 0.031 | -0.292 ± 0.024 | 1467 |
| greedy|frac[0.75,1.0) | -0.243 ± 0.030 | 0.000 ± 0.000 | 1404 |

   PB reproduces the sign pattern: up for frac ∈ [.25, .5), down for [.5, .75). It does so purely through Darroch and the (11+b)/(10+b)
   tilt. It gives about half the magnitude, and nothing for frac ∈ [.75, 1). The residual down-shading is right skew from
   positive dependence. The empirical third central moment is 3–4× PB's (`results/m1_skew.json`).
   Shaded-down hands have third moment 1.0–2.7, against 0.0–0.4 for unshaded hands.
6. *Cheap closed-form bidder.* It looks up p_i by (n, NT, is_trump, rank, suit length), with backoff, fits on even hands, and is tested
   on odd hands against the empirical-optimal bid:

| play | lookup | lookup + context key | fitted bidder model | PB with true p_i | lookup EV loss | model EV loss | hands |
|---|---|---|---|---|---|---|---|
| greedy | 0.545 ± 0.018 | 0.551 ± 0.018 | 0.623 ± 0.017 | 0.731 ± 0.011 (all hands) | 1.19 ± 0.07 | 0.90 ± 0.07 | 3000 |
| ta | 0.534 ± 0.018 | 0.548 ± 0.018 | 0.888 ± 0.011 | 0.779 ± 0.010 (all hands) | 2.09 ± 0.11 | 0.16 ± 0.02 | 3000 |

   The lookup is clearly worse than the bidder model. Its loss comes from estimating p_i, not from independence:
   with Greedy, PB fed the *true* p_i (0.73) beats the fitted bidder model (0.62).
   Under TA, the model's 0.89 is inflated, because the TA law at the played bid is pulled toward that bid, which makes it self-fulfilling. See M1b.
   Phase 1 measured the bidder's oracle match at 0.72 / 0.70 (TA / Greedy). The like-for-like figure here (Greedy, first bidder) is 0.62.

**M1b: the bid-conditional oracle under TA** (`m1_ta_oracle.py`, `results/m1_ta_oracle.txt`). The hands are the first 300 of
each m1_sim config (1,800 in total). For each candidate bid b ∈ model bid ± 2, the hero is forced to bid b, with 200 CRN worlds per b. The
oracle is argmax_b (10+b)P_b(b). The independent-card bidder applies PB bid by bid: argmax_b (10+b)·PB(p^(b))(b).

| group | hands | PB == oracle | bidder model == oracle | EV loss PB | EV loss model | δ at oracle bid |
|---|---|---|---|---|---|---|
| all | 1800 | 0.688 ± 0.021 | 0.597 ± 0.023 | 0.36 ± 0.04 | 0.63 ± 0.05 | 1.350 ± 0.027 |
| trump | 900 | 0.730 ± 0.029 | 0.639 ± 0.031 | 0.22 ± 0.03 | 0.46 ± 0.06 | 0.975 ± 0.020 |
| NT | 900 | 0.647 ± 0.031 | 0.554 ± 0.032 | 0.50 ± 0.06 | 0.80 ± 0.09 | 1.727 ± 0.035 |
| n=5|S | 300 | 0.777 ± 0.047 | 0.750 ± 0.049 | 0.17 ± 0.05 | 0.20 ± 0.06 | 1.117 ± 0.032 |
| n=5|NT | 300 | 0.613 ± 0.055 | 0.693 ± 0.052 | 0.62 ± 0.13 | 0.38 ± 0.10 | 1.752 ± 0.054 |
| n=8|S | 300 | 0.760 ± 0.048 | 0.643 ± 0.054 | 0.20 ± 0.06 | 0.33 ± 0.07 | 0.994 ± 0.028 |
| n=8|NT | 300 | 0.683 ± 0.053 | 0.587 ± 0.056 | 0.42 ± 0.10 | 0.75 ± 0.14 | 1.800 ± 0.056 |
| n=13|S | 300 | 0.653 ± 0.054 | 0.523 ± 0.057 | 0.30 ± 0.06 | 0.85 ± 0.16 | 0.812 ± 0.032 |
| n=13|NT | 300 | 0.643 ± 0.054 | 0.383 ± 0.055 | 0.47 ± 0.10 | 1.26 ± 0.17 | 1.627 ± 0.070 |

The oracle sits at a truncated edge of the ±2 window in 6.3% of hands (21.7% at n = 13 NT), so those cells are lower bounds.
The PB bid is estimated from the same rollouts as the oracle, which shares noise in the mean. Its match rate is somewhat optimistic.

**Finding.** The independent-card model, given the true per-bid p_i, chooses better than the fitted bidder model under TA:
0.69 vs 0.60 match, and 0.36 vs 0.63 EV loss. The gain is largest at n = 13 (NT: 0.64 vs 0.38). At its oracle
bid, TA play is close to independent for trump (δ = 0.98 ± 0.02) and under-dispersed at n = 13 (0.81). NT
stays at 1.63–1.80. Bid-conditional p_i plus independence is therefore a strong bidder wherever p_i can be
estimated. The bottleneck is estimating p_i cheaply (finding 6).



---------------------------------------------------------------------------------

## M2: the 1-card round in closed form

Positions: pos0 = left of dealer (bids first, leads), pos3 = dealer.

**Proposition P1 (myopia and equilibrium). Proved on path; unique outright in trump.**
With one card, play is forced, so W_i = {i's card wins} is a function of the deal alone.
Player i's payoff is (H+1)·1{b=1, W_i} + H·1{b=0, not W_i}. It depends on nobody else's
action, except that the hook restricts the dealer's action set. Hence, at any information set
(card c, bid history h), the best response is myopic: bid 1 iff (H+1)q > H(1−q), i.e.
q := P(W_i | c, h) > H/(2H+1) = **10/21 ≈ 0.4762**. The dealer is forced when Σprior ∈ {0, 1}
(Σprior = 0 forbids 1, Σprior = 1 forbids 0) and otherwise uses the same rule.
Earlier players' optimal strategies do not depend on later ones, so forward induction
gives unique equilibrium play on the equilibrium path (unique outright for trump). NT follower-bid-1 histories are off path, so unrestricted PBE beliefs preclude an off-path uniqueness claim. No exact ties 21W = 10T occur
(exhaustive check).

**Proposition P2 (threshold strategies). Verified exhaustively.**
In every reachable information cell (S and NT), q(c, h) is nondecreasing in rank within
each suit. Hence every best response is a per-suit rank threshold. The check runs over all 6,497,400
ordered deals in exact integer arithmetic.

**Closed forms (proved).** Let r' = 0..12 be the rank index (2..A). The leader wins iff no
other hand holds a higher card of the relevant kind:
- trump card: a_{r'} = C(39+r', 3)/C(51, 3) = C(51−(14−r), 3)/C(51, 3). This gives a('3') = .4744 < 10/21 < a('4') = .5119.
  The leader bids 1 iff the trump is 4 or higher.
- side suit: b_{r'} = C(26+r', 3)/C(51, 3) ≤ C(38,3)/C(51,3) = .4051 < 10/21, so the leader never bids 1.
- NT: same a_{r'} in any suit, so the threshold is again 4 or higher.
- Leader value (S): V0 = (1/52)[Σ max(11a, 10(1−a)) + 3·Σ 10(1−b)] = 1638607/216580 = 7.5658.
  NT: V0 = (1/13)Σ max(11a, 10(1−a)) = 418063/54145 = 7.7212.
- NT followers. P(leader wins) = [C(52,4) − C(39,4)]/(13·C(51,3)) = 188474/270725 (hockey stick).
  A follower's posterior is at most 1/4, so followers never bid 1.
  V1 = V2 = 10(1 − (1−P_L)/3) = 37432/4165 = 8.9873. The dealer is always forced (bid 1 iff the leader bid 1): 111934/54145 = 2.0673.

**Exact equilibrium (`m2_onecard.py`, `results/m2_onecard.json`).** The script enumerates all 6,497,400 deals.
Its output matches the Phase-2 solver (`experiments/phase2/results/onecard.json`) exactly: max |seat-value
difference| = 0.0, and every strategy cell is identical, for both S and NT.

| | pos0 (leads) | pos1 | pos2 | dealer | table | P(Σbids=0) |
|---|---|---|---|---|---|---|
| S, value | 1638607/216580 = 7.566 | 30024697/3248700 = 9.242 | 30507739/3248700 = 9.391 | 3734399/928200 = 4.023 | 30.22 | 0.436 |
| S, hit rate | .741 | .907 | .923 | .396 | | |
| NT, value | 418063/54145 = 7.721 | 37432/4165 = 8.987 | 8.987 | 111934/54145 = 2.067 | 27.76 | 0.154 |
| S, no hook | 7.566 | 9.242 | 9.391 | 9.501 | 35.70 | |
| NT, no hook | 7.721 | 8.987 | 8.987 | 8.987 | 34.68 | |

Bid-1 card sets in equilibrium (S). Pos0: ♠4+. Pos1: after 0, any ♠; after 1, ♠T+. Pos2: 00 → any ♠, 01 → ♠9+,
10 → ♠T+, 11 → ♠K+. Dealer: 000 → forced 0; Σ = 1 → forced 1 (any card); 011 → ♠Q+;
101 and 110 → ♠K+; 111 → ♠A. The leader's bid is informative: a 1 means it holds ♠4+, so followers need
♠T+ to beat it, while a 0 makes any spade a likely winner. Followers' thresholds depend on the history, e.g. ♠2 bids 1 after a leader's 0. Their values (9.24, 9.39) beat the leader's (7.57).
Information, not position in the trick, makes followers the best seats.

**Exact dealer hook cost at n = 1:** 5.478 (S), 6.920 (NT). Phase-1 E5 measured +5.41 ± 0.11 in TA self-play.
In this ladder n = 1 is always a trump round, so the exact and measured values agree.


---------------------------------------------------------------------------------

## M3: hook theory

**Proposition P3 (at most N−1 hit). Proved.**
Σ_i t_i = n in every round. If all N players hit, then Σ b_i = Σ t_i = n, which the hook forbids.
So at most N−1 players hit, whatever the information structure or play. The bound needs no perfect
information. Incidence is empirical: E5 finds small hidden-information non-dealer costs, whereas phase2 F3 finds material PI non-dealer costs; conservation alone does not assign the cost to the dealer.
Check on the 250,000 Phase-1 all-TA rounds: max hits = 3, Σbids ≠ n and Σtricks = n always
(test_math_onecard.py).

**Approximation A1 (imperfect-information hook cost).**
Let f = n − Σprior be the forbidden bid, EV the dealer's (bid-conditional) expected score,
and b* = argmax EV. The best legal bid is b2 = argmax_{b≠f} EV, which by L1 is b*±1 when the
law is log-concave. Then
  C(n) ≈ E[ 1{f = b*}·(EV(b*) − EV(b2)) ] = P(f = b*) · E[gap | f = b*].
This ignores how the other players' (target-aware) play reacts to the dealer's changed bid.
E5's "others' mean Δ" (≤ 0.09) bounds that effect as small.
At n = 1 the exact game value (M2) gives C(1) = **5.478** (trump), or 6.920 (NT; but n=1 is
always a trump round in this ladder).

**Test of A1 (`m3_hook.py`, `m3_decay.py`).** The script re-creates all 250,000 dealer situations of the Phase-1
all-TA games. The deal stream depends only on the seed. As a check, the stored first-bidder bids equal the model's bids in 5000/5000 spot checks.
EV comes from the fitted TA bidder model P_b(t). "Measured" is the CRN difference between dealer scores in
games_ta_nohook and games_ta (= E5).

| n | P(f = b*) | E[gap \| f=b*] | Gaussian gap g(σ) | σ (dealer) | C_pred | measured (E5) | meas / pred |
|---|---|---|---|---|---|---|---|
| 1 | 0.554 | 8.25 | 8.28 | 0.29 | 4.57 | 5.41 ± 0.11 | 1.18 |
| 2 | 0.428 | 5.33 | 6.01 | 0.45 | 2.28 | 2.84 ± 0.08 | 1.24 |
| 3 | 0.399 | 4.21 | 2.56 | 0.67 | 1.68 | 1.85 ± 0.08 | 1.10 |
| 4 | 0.294 | 2.95 | 1.97 | 0.75 | 0.87 | 1.15 ± 0.07 | 1.32 |
| 5 | 0.264 | 1.66 | 2.01 | 0.75 | 0.44 | 0.70 ± 0.07 | 1.60 |
| 6 | 0.257 | 1.33 | 1.80 | 0.81 | 0.34 | 0.51 ± 0.07 | 1.49 |
| 7 | 0.243 | 1.11 | 1.48 | 0.88 | 0.27 | 0.42 ± 0.06 | 1.55 |
| 8 | 0.238 | 1.08 | 0.96 | 1.05 | 0.26 | 0.39 ± 0.06 | 1.51 |
| 9 | 0.215 | 0.97 | 0.85 | 1.10 | 0.21 | 0.35 ± 0.06 | 1.66 |
| 10 | 0.241 | 0.91 | 0.97 | 1.04 | 0.22 | 0.34 ± 0.07 | 1.54 |
| 11 | 0.251 | 0.93 | 0.92 | 1.10 | 0.23 | 0.36 ± 0.07 | 1.55 |
| 12 | 0.277 | 0.88 | 0.84 | 1.14 | 0.24 | 0.32 ± 0.07 | 1.33 |
| 13 | 0.239 | 0.80 | 0.63 | 1.27 | 0.19 | 0.27 ± 0.06 | 1.43 |

Rounds: 10,000 for n = 1 and 20,000 for every other n. CIs: C_pred ±0.01–0.09, measured ±0.06–0.11.

**Findings.**
1. P(f = b*) reproduces E5's "no-hook: frac Σbids = n" column exactly (.554, .428, .399, …, .239).
   This is an identity: without the hook, the dealer bids b*.
2. *Decay decomposition.* From n = 1 to 13, C_pred falls ×24 (measured: ×20). P(f = b*) accounts for ×2.3 (.55 → .24); it plateaus near .25 from
   n ≈ 5. The conditional EV gap accounts for ×10 (8.25 → 0.80). The gap is explained by the spread of the dealer's
   trick law alone. A one-parameter discretised Gaussian with the dealer's mean σ(n) gives g(σ) = 8.28 → 0.63,
   which tracks the model gap (largest misfit 2.6 vs 4.2 at n = 3). Mechanism (L1 + L2): EV(b*) − EV(b*±1) ≈
   (H+b)·P(b*)·(1 − e^{−(1±2δ)/2σ²}). This is O(H) when σ ≪ 1 (n = 1: only one trick to misplace) and
   O(H/σ³) once σ ≳ 1. The plateau near 0.3 at large n is σ growing slowly, from 1.0 to 1.3.
3. *Level.* A1 under-predicts by a factor 1.1–1.7, a consistent gap of +0.1 to +0.8 points.
   Binned calibration: realized gap = predicted + 0.5…1.4 in every bin (`m3_hook.json`). The cause is model
   miscalibration exactly where the hook acts. When forced *up* (b2 = b*+1, n = 42,323), the model expects
   P(hit) = .310 but the realized rate is .284 ± .004. At the natural bid the model expects .594 against .610 ± .002 realized.
   Forced *down* is calibrated (.430 vs .434 ± .006). The TA bidder's bid-pull terms overstate how well a player
   can climb to a bid above its natural one.
4. The dealer is forced up more often than down (42,323 vs 30,131). That is L2(ii): the (11+b)/(10+b) prize tilt makes
   b*+1 the better legal alternative. Forced-up is also the more costly direction, which feeds M4.


---------------------------------------------------------------------------------

## M4: conservation model of hit rates

**Model M4 (constrained independent errors).** The script is `m4_conservation.py`. Let e_i = t_i − b_i. The errors are independent,
with pmf π_c depending on bid class c ∈ {0, 1, 2, 3+}, support e ∈ [−min(b,4), min(n−b,4)] (a bid-0 player can only
over-take), *conditioned* on the conservation law Σe_i = D = n − Σb_i. Then

  P(hit_i | D) = π_i(0) · (∗_{j≠i} π_j)(D) / (∗_j π_j)(D).

The model has no D-specific parameters. It is fit by MLE on all 250,000 all-TA rounds (33 rounds with |e| > 4 dropped).
**Heuristic P4 (tent). Verified numerically, not a general theorem.** For any π with mode at 0, D = 0 maximises the joint mass on the all-hit vector. That holds by
construction. This does not prove unimodality of conditional individual hit rates. The observed near-tent and peak near |D|=1 are empirical; additional distributional assumptions would be needed for a theorem.

| Σb−n | rounds | empirical hit [95% CI] | model: symmetric π | one π | π by bid class | non-dealer emp | dealer emp |
|---|---|---|---|---|---|---|---|
| ≤−5 | 1364 | 0.393 ± 0.013 | 0.321 | 0.319 | 0.350 | – | – |
| −4 | 4830 | 0.443 ± 0.007 | 0.405 | 0.403 | 0.433 | 0.401 ± 0.008 | 0.568 ± 0.014 |
| −3 | 20037 | 0.510 ± 0.003 | 0.467 | 0.467 | 0.489 | 0.481 ± 0.004 | 0.596 ± 0.007 |
| −2 | 50114 | 0.563 ± 0.002 | 0.526 | 0.527 | 0.537 | 0.547 ± 0.003 | 0.613 ± 0.004 |
| −1 | 99387 | 0.617 ± 0.002 | 0.615 | 0.613 | 0.611 | 0.630 ± 0.002 | 0.578 ± 0.003 |
| +1 | 67425 | 0.570 ± 0.002 | 0.614 | 0.616 | 0.604 | 0.629 ± 0.002 | 0.394 ± 0.004 |
| +2 | 6255 | 0.487 ± 0.006 | 0.517 | 0.511 | 0.495 | 0.491 ± 0.007 | 0.476 ± 0.012 |
| +3 | 566 | 0.421 ± 0.020 | 0.453 | 0.453 | 0.433 | 0.426 ± 0.024 | 0.406 ± 0.040 |
| ≥+4 | 22 | 0.375 ± 0.101 | 0.387 | 0.399 | 0.375 | 0.349 ± 0.118 | 0.429 ± 0.212 |

No-hook games (games_ta_nohook) give the same law without the hook:

| Σb−n | −2 | −1 | 0 | +1 | +2 |
|---|---|---|---|---|---|
| empirical | 0.564 ± 0.002 | 0.621 ± 0.002 | 0.668 ± 0.002 | 0.565 ± 0.003 | 0.486 ± 0.007 |
| class model | 0.534 | 0.609 | 0.701 | 0.579 | 0.486 |

**Findings (why the peak is at −1).**
1. The constrained-error model gets the *tent*: the level and the fall-off with |Σb−n| come out within about .02–.04 for |D| ≤ 3.
   The peak at |D| = 1 is an empirical feature, not a consequence of P4 alone. A single π, even an asymmetric one, gives an almost
   symmetric tent (.613/.616). The class model gives only .611/.604. **The ±1 asymmetry (.617 vs .570) is not
   a conservation effect. The model is refuted as its explanation.**
2. *It is a hook effect on the dealer.* Non-dealers are symmetric at ±1: .630 ± .002 vs .629 ± .002.
   The dealer is not: .578 vs .394. 61% of Σb−n = +1 tables (41,383 of 67,425) are dealers forced *up* off b*,
   and those dealers hit only .284. Σb−n = −1 tables are mostly unhooked (73%).
   Chain: L2(ii) prize tilt ⇒ the dealer prefers to be forced up. Up is the harder direction to make under TA (M3 finding 3).
   So +1 tables carry a lot of dealer misses.
3. Without the hook the peak moves to Σb−n = 0 (.668). A smaller intrinsic over-bid penalty remains: −1 .621 vs +1
   .565, and −2 .564 vs +2 .486. The class model fitted on no-hook games captures about half of it (.609 vs .579).
   The source is bid-0 skew: zero-bidders can only over-take, so spare tricks have somewhere to go.


---------------------------------------------------------------------------------

## What the math explains, and what it does not

**Status of the claims**

| claim | status |
|---|---|
| P0 dispersion = 2ΣCov/ΣVar | proved (exact for the plug-in estimators) |
| L1 PB log-concave, so EV log-concave, so the bid is unique up to ties and the hooked alternative is b*±1 | proved; checked on 39k vectors |
| L2(i) b* ≥ smallest mode, for any law | proved |
| L2(ii) ratio rule P(b+1)/P(b) > (10+b)/(11+b) | proved |
| L2(iii) b* ∈ {m_hi, m_hi+1} for PB/ULC | proved; also holds for 98.4–99.7% of the non-PB simulated laws |
| L3 ⌊μ⌋ ≤ b* ≤ ⌈μ⌉+1 | proved (Darroch + L2) |
| L3′ \|b*−μ\| < 1 | proved for PB in round 2, M6 T6 (round-1 numerical checks superseded) |
| C1 bimodal ⇒ dependent | proved; empirical certificates in 2.8% (TA) and 5.5% (Greedy) of hands |
| "TA makes trick counts under-dispersed" | **supported for trump at n=13** (δ = 0.79) and long-trump hands; **refuted for NT** (δ ≈ 1.55–1.78; TA > Greedy at n=5, 8) |
| P1 1-card threshold 10/21 and equilibrium | proved: unique on path; unique outright in trump |
| P2 monotone thresholds in the 1-card game | verified exhaustively (all 6.5M deals) |
| 1-card closed forms (leader, NT seats) | proved; exact match to Phase 2 |
| P3 at most N−1 hit under the hook | proved (any information); holds in 250k rounds |
| A1 hook cost ≈ P(f=b*)·E[gap] | shape and decay verified; level under-predicted ×1.1–1.7 (cause identified) |
| P4 / M4 conservation tent | tent verified within .02–.04; refuted as an explanation of the ±1 asymmetry (a dealer-hook effect) |

**What the math explains.**
- The whole 1-card round, exactly. That includes why followers are the best seats: the leader's bid is a signal they read.
- The exact n=1 hook cost (5.48 vs 5.41 measured).
- The structure of optimal bids. The bid sits at the mode or one above it, the threshold for going above is a 5–9% near-tie
  band, and a hooked dealer steps to b*±1. All of this holds far beyond PB.
- The decay of the hook cost with n. It is mostly a ×10 shrinking EV gap, and one parameter (the spread σ of the dealer's
  trick law) predicts that gap. A ×2.3 fall in P(forbidden = optimal) contributes the rest.
- Half of the E3 bid shading.
- The tent shape of hit rate versus Σbids − n, and why the peak is at |D| = 1.

**What it does not explain.**
- Positive dependence between card outcomes (lead control, suit runs), which is largest in NT. The independent-card
  model costs about 1 point per NT round. Right skew from that dependence drives the rest of the down-shading.
- The *level* of the hook cost. That is a calibration defect of the TA bidder for forced-up bids, not a modelling
  error in A1.
- The ±1 hit-rate asymmetry. It is a dealer-hook selection effect plus an intrinsic over-bid penalty that is only half
  explained by bid-0 skew.


## Reproduce
```
python math/m2_onecard.py         # ~5 s, exact
python math/m1_lemmas.py          # ~8 s
python math/m1_sim.py --ns 5,8,13 --hands 1000 --R 300   # ~25 min, one core
python math/m1_analyze.py
python math/m1_ta_oracle.py && python math/m1_ta_oracle_analyze.py   # ~10 min
python math/m3_hook.py --games 10000 && python math/m3_decay.py
python math/m4_conservation.py && python math/m4_conservation.py games_ta_nohook.npz
pytest -q -p no:cacheprovider tests/test_math_*.py
```


=================================================================================
# ROUND 2 (M5–M9)
=================================================================================

---------------------------------------------------------------------------------

## M6: the upper side of L3′ — proved, with the sharp constant

**Theorem T6.** Let T ~ PB(p_1..p_N), μ = Σp_i, H > 0, b* any maximiser of EV(b) = (H+b)P(b). Then
b* < μ + 1. Together with the lower side (round 1), |b* − μ| < 1, i.e. b* ∈ {⌊μ⌋, ⌈μ⌉}, and b* = μ when μ ∈ ℤ.
Moreover the constant is sharp:
  sup_{p ∈ [0,1]^N} (b* − μ) = s_N := max_{0≤m≤N−1} (m+1)/(1 + c_m (m+1)),  c_m = (H+m)/(H+m+1),
approached (not attained) by Bin(m+1, a) with a ↓ a*_m = c_m(m+1)/(1 + c_m(m+1)). **Proved** (using Hoeffding's
extremal theorem); **verified numerically**.

**Lemma R (ratio bound).** If m ∈ ℤ and μ ≤ m, then P(m+1) ≤ (m/(m+1))·P(m).
*Proof.* Fix N and μ. g(T) = 1{T=m+1} − (m/(m+1))1{T=m} is a function of T, so by Hoeffding (1956,
Ann. Math. Statist. 27, extremal theorem: E g(T) over {p : Σp_i = μ} is extremised at a p taking at most three
values, of which only one lies outside {0,1}) it suffices to check p = (1^k, a^j, 0^{N−k−j}), i.e.
T = k + Bin(j, a), k + ja = μ. Put m′ = m − k ≥ 0 (k ≤ μ ≤ m) and λ = ja ≤ m′. If j < m′ both terms vanish.
Otherwise P(m+1)/P(m) = (j−m′)a / ((m′+1)(1−a)) and (j−m′)a ≤ m′(1−a) ⇔ ja ≤ m′, so the ratio is
≤ m′/(m′+1) ≤ m/(m+1). ∎
*Proof of T6.* Suppose b* ≥ μ+1 and put m = b*−1 ≥ μ. EV(b*) > 0 (EV at the mode is positive), so P(b*) > 0, and by
no-internal-zeros (support of PB is an interval [#ones, #nonzero] ∋ μ ≤ m) P(m) > 0. By Lemma R,
(H+m+1)P(m+1) ≤ (H+m+1)·m/(m+1)·P(m) < (H+m)P(m), since m(H+m+1) < (m+1)(H+m) ⇔ 0 < H. So EV(m) > EV(b*),
contradiction. ∎
*Sharp constant.* b* = m+1 needs EV(m+1) > EV(m), i.e. P(m+1)/P(m) > c_m. By the same extremal reduction, for fixed μ
the largest value of P(m+1) − c_m P(m) is attained at k + Bin(j, a). With λ = μ − k ∈ (m′, m′+1) the ratio
(λ − m′a)/((m′+1)(1−a)) is increasing in a (derivative sign λ − m′ > 0), so the best choice is the fewest trials,
j = m′+1, a = λ/(m′+1): ratio a/((m′+1)(1−a)) > c_m ⇔ a > c_m(m′+1)/(1+c_m(m′+1)). Then
m+1−μ = (m′+1)(1−a) < (m′+1)/(1+c_m(m′+1)), increasing in m′, so k = 0 is worst. Maximise over m ≤ N−1. ∎

| N (cards) | 1 | 2 | 3 | 5 | 8 | 10 | 13 |
|---|---|---|---|---|---|---|---|
| s_N = sup(b* − μ) | 11/21 = .524 | .706 | .796 | .882 | .935 | .952 | .968 |
| worst case | Bern(a), a ↓ 10/21 | Bin(2, a↓.647) | Bin(3,·) | Bin(5,·) | Bin(8,·) | Bin(10,·) | 13 cards each p ↓ .9256, μ = 12.03, bid 13 |

The worst case is always "m+1 equally likely winners", i.e. every card is a near-sure trick; s_N → 1 as N → ∞
because the (H+b) tilt toward the higher bid is worth relatively less and less but the binomial top ratio
a/((m+1)(1−a)) can be pushed to it with μ ever closer to m. For mixed hands the gap is much smaller (round 1's random
families: 0.94 max).
**Numerical check** (`m6_bound.py`, `results/m6_bound.json`, seed 606): for each N = 1..13, 3,000 random vectors
(4 families) + 40 hill-climbs (20 maximising b* − μ, 20 maximising the Lemma R violation). Max Lemma R violation
= 0 (no positive value); max b* − μ stays below s_N for every N (hill-climb gets within 0.002 of s_N at N = 1, 2, 3, 5, 7);
Bin(m+1, a*+10⁻⁹) attains s_N to 10⁻⁶. Tests: `tests/test_math2_bound.py`.

Consequence for play: under independence the optimal bid is *always* ⌊μ⌋ or ⌈μ⌉, so a bidder that only knows μ
is never more than one trick off; any empirical b* outside {⌊μ⌋, ⌈μ⌉} certifies dependence (adds to C1).

---------------------------------------------------------------------------------

## M7: hook-cost formula, level fixed

Same 250,000 dealer situations as M3 (10,000 Phase-1 all-TA games, hook vs no-hook CRN). In a hooked round
(f = b*) the no-hook dealer bids b* and the hooked one bids b_h ≠ f, so the measured cost is exactly
(10+b*)1{hit_nh} − (10+b_h)1{hit_h} (check: b_nh = b* in 100% of hooked rounds). A1 replaces both indicators by the
TA bidder model's P(hit). **Recalibration:** multiply the model P(hit) by k_cell = realized/predicted, cell =
direction (natural b* | forced up | forced down) × {pooled, n-bucket {1–2, 3–4, 5–8, 9–13}, per n},
**fit on even games, evaluated on odd games** (`m7_hookcal.py`, `results/m7_hookcal.json`, `m7_tables.txt`).

| n | measured (odd games, ±95%) | raw A1 | 3 pooled factors | by n-bucket | by n | natural-only fix | forced-only fix |
|---|---|---|---|---|---|---|---|
| 1 | 5.46 ± 0.15 | 4.59 | 4.85 | 4.94 | 5.15 | 4.99 | 4.61 |
| 2 | 2.88 ± 0.11 | 2.28 | 2.54 | 2.74 | 2.65 | 2.55 | 2.32 |
| 3 | 1.89 ± 0.11 | 1.73 | 2.01 | 1.95 | 1.91 | 1.97 | 1.78 |
| 4 | 1.22 ± 0.10 | 0.87 | 1.07 | 1.04 | 1.08 | 1.03 | 0.91 |
| 5 | 0.75 ± 0.10 | 0.44 | 0.60 | 0.58 | 0.65 | 0.57 | 0.47 |
| 7 | 0.50 ± 0.09 | 0.28 | 0.42 | 0.40 | 0.35 | 0.39 | 0.31 |
| 10 | 0.30 ± 0.09 | 0.22 | 0.37 | 0.32 | 0.37 | 0.34 | 0.25 |
| 13 | 0.26 ± 0.09 | 0.19 | 0.36 | 0.30 | 0.28 | 0.33 | 0.22 |
| pooled meas/pred | | 1.315 | 1.063 | 1.080 | 1.081 | | |

(prediction CIs ±0.01–0.13; all n in m7_tables.txt.) Pooled factors (even games): natural ×1.080, forced-up ×0.936,
forced-down ×1.014.

**Findings.**
1. *Level fixed.* Three pooled calibration factors take meas/pred from 1.32 to 1.06 out of sample. With n-bucket
   factors, every n ≥ 2 is inside the measured CI (n=2: 2.74 vs 2.88 ± 0.11); only n = 1 stays low (4.94 vs 5.46;
   the exact game value 5.48 is known from M2 anyway).
2. *Round 1's diagnosis was incomplete — the main error is at the natural bid, not forced-up.* Correcting only the
   forced directions moves C(n) by ≤ 0.05; correcting only the natural bid gives essentially all of the gain.
   The model under-predicts P(hit at b*) by 3–10 points at every n (n=2: .753 vs .857 ± .007; n=5: .557 vs .612).
   Forced-up is over-predicted by 1–5 points (n=4–13), forced-down is calibrated except at n ≤ 2.
3. *Mechanism: unused bid information.* At n = 1 play is forced, so any miscalibration is pure inference. There the
   model gives P(hit at b*) = .887 vs .979 realized, and forced-down .128 vs .000 (n = 1,130: after 0-0-0 a dealer
   whose model b* is 1 holds a trump that cannot lose, because earlier 0 bids certify no trump ≥ 4). The dealer's
   posterior given the three prior bids is sharper than the bidder model's linear-in-features use of them, so the
   *natural* option is worth more than the model thinks and the hook takes more away. (This matches Phase 3 P1:
   informed − uniform oracle +0.39/round at the dealer.)
4. Formula status: A1 with a calibrated P(hit) is **verified** (shape and level, n ≥ 2, out of sample);
   the residual at n = 1 is information the bidder model does not encode.

---------------------------------------------------------------------------------

## M5: where the dependence lives

**Design** (`m5_sim.py`, `m5_analyze.py`, `m5_lead.py`; `results/m5_raw_*.npz`, `m5_summary.json`, `m5_tables.txt`,
`m5_lead.json`). Same 6,000 deals and (for TA) the same rollout seeds as M1 — TA n=5 S reproduces M1 exactly
(TV .072, PB match .878, loss .11). Hero = first bidder, model bid; 1,000 hands × R = 300 per (n, type), TA and
TA+ (ta_plus_v2) play. Per rollout the full bitmask of the hero's winning cards is stored, so the full covariance
matrix C of the card-win indicators is available. By P0, δ − 1 = Σ_{i≠j} C_ij / Var_PB, split exactly by pair type:
TT (both trump), SS (same side suit), TX (trump × side), XX (different side suits); in NT same / cross suit.
Shares are per hand, then averaged (± 95% CI over hands).

| play | n | type | δ | within-suit share (TT+SS / same) | cross-suit share (TX+XX / cross) | TT | SS | TX | XX |
|---|---|---|---|---|---|---|---|---|---|
| TA | 5 | S | 1.138 ± .015 | +.033 | +.106 | −.016 ± .004 | +.049 ± .007 | +.018 ± .007 | +.088 ± .008 |
| TA | 8 | S | 1.005 ± .015 | +.012 | −.008 | −.035 ± .004 | +.047 ± .006 | −.061 ± .008 | +.054 ± .008 |
| TA | 13 | S | 0.789 ± .017 | −.030 | −.181 | −.071 ± .007 | +.041 ± .008 | −.169 ± .009 | −.012 ± .009 |
| TA+ | 5 | S | 1.047 ± .014 | +.002 | +.046 | −.028 ± .004 | +.030 ± .006 | −.002 ± .007 | +.047 ± .007 |
| TA+ | 8 | S | 0.869 ± .014 | −.041 | −.090 | −.049 ± .004 | +.008 ± .004 | −.081 ± .007 | −.009 ± .007 |
| TA+ | 13 | S | 0.728 ± .016 | −.057 | −.215 | −.077 ± .007 | +.020 ± .006 | −.164 ± .009 | −.051 ± .009 |
| TA | 5 | NT | 1.773 ± .029 | +.241 ± .021 | **+.532 ± .018** | | | | |
| TA | 8 | NT | 1.717 ± .030 | +.202 ± .016 | **+.516 ± .021** | | | | |
| TA | 13 | NT | 1.547 ± .031 | +.240 ± .016 | **+.307 ± .024** | | | | |
| TA+ | 5 | NT | 1.815 ± .029 | +.243 ± .020 | **+.571 ± .018** | | | | |
| TA+ | 8 | NT | 1.736 ± .030 | +.209 ± .015 | **+.527 ± .020** | | | | |
| TA+ | 13 | NT | 1.572 ± .032 | +.228 ± .015 | **+.344 ± .024** | | | | |

Within-suit covariance grows with suit length (TA, pooled n, Σ_{i≠j} C_ij per suit): NT length 2/3/4/5+ = .017/.052/.103/.242;
side suits in trump rounds .004/.013/.026/.082; the trump suit is *negative* and grows with length: −.026/−.057/−.143/−.231.

**Lead check** (`m5_lead.py`, TA, n = 8, 300 hands × 300, same seeds): each win is tagged as won on the hero's own
lead (L) or while following/ruffing/discarding (F); cross-suit covariance = LL + (LF+FL) + FF, as a share of δ − 1:

| n=8 | cross-suit total | LL | LF + FL | FF | share of wins taken on own lead |
|---|---|---|---|---|---|
| NT | +.527 ± .036 | +.142 ± .018 | **+.432 ± .023** | −.047 ± .014 | .60 |
| S | +.009 ± .020 | +.059 ± .014 | +.080 ± .018 | **−.130 ± .016** | .40 |
| 5 NT | +0.529 ± 0.035 | +0.270 ± 0.032 | +0.310 ± 0.031 | -0.051 ± 0.005 | 0.67 |
| 13 NT | +0.299 ± 0.043 | -0.019 ± 0.028 | +0.449 ± 0.025 | -0.131 ± 0.026 | 0.61 |
| 5 S | +0.098 ± 0.020 | +0.111 ± 0.021 | +0.091 ± 0.021 | -0.105 ± 0.011 | 0.41 |
| 13 S | -0.178 ± 0.023 | -0.012 ± 0.016 | +0.083 ± 0.032 | -0.249 ± 0.024 | 0.54 |

(rows for n = 5, 13: `results/m5_lead_extra.json`, same design, 300 hands each.) In NT the follow→lead term LF is the
largest positive term at every n (+.31, +.43, +.45), while follow–follow FF is negative and grows with n. In trump rounds FF is
strongly negative at n = 13 (−.25): ruffing and following winners substitute for each other.

**Suit-block model** SB: T = Σ_s T_s with each suit's empirical trick law, suits independent. Var_SB = Var_PB +
within-suit covariance exactly, so its dispersion error is minus the cross-suit share.

| play | n | type | δ err PB | δ err SB | TV PB | TV SB | xTV PB / SB / floor | match PB | match SB | A/B floor | EV loss PB | EV loss SB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TA | 5 | S | −.138 | −.106 | .072 | .058 | .089 / .080 / .067 | .878 ± .020 | .897 ± .019 | .90 | .11 ± .02 | .08 ± .02 |
| TA | 8 | S | −.005 | +.008 | .084 | .070 | .100 / .092 / .076 | .879 ± .020 | .892 ± .019 | .89 | .11 ± .03 | .08 ± .02 |
| TA | 13 | S | +.211 | +.181 | .133 | .107 | .143 / .122 / .073 | .867 ± .021 | .894 ± .019 | .92 | .16 ± .03 | .10 ± .02 |
| TA | 5 | NT | −.773 | −.532 | .185 | .135 | .190 / .144 / .068 | .750 ± .027 | .870 ± .021 | .98 | .83 ± .11 | .32 ± .06 |
| TA | 8 | NT | −.717 | −.516 | .175 | .133 | .182 / .144 / .076 | .717 ± .028 | .813 ± .024 | .95 | .72 ± .09 | .35 ± .06 |
| TA | 13 | NT | −.547 | −.307 | .192 | .157 | .199 / .168 / .080 | .581 ± .031 | .657 ± .030 | .95 | 1.39 ± .12 | 1.03 ± .11 |
| TA+ | 5 | S | −.047 | −.046 | .064 | .051 | .082 / .074 / .064 | .919 ± .017 | .932 ± .016 | .91 | .07 ± .02 | .05 ± .02 |
| TA+ | 8 | S | +.131 | +.090 | .095 | .078 | .109 / .098 / .073 | .904 ± .018 | .917 ± .017 | .92 | .08 ± .02 | .06 ± .02 |
| TA+ | 13 | S | +.272 | +.215 | .142 | .115 | .151 / .129 / .071 | .872 ± .021 | .891 ± .019 | .91 | .15 ± .03 | .11 ± .03 |
| TA+ | 5 | NT | −.815 | −.571 | .201 | .150 | .207 / .159 / .070 | .655 ± .029 | .799 ± .025 | .96 | 1.07 ± .11 | .50 ± .07 |
| TA+ | 8 | NT | −.736 | −.527 | .179 | .136 | .185 / .147 / .080 | .718 ± .028 | .807 ± .024 | .94 | .71 ± .09 | .38 ± .06 |
| TA+ | 13 | NT | −.572 | −.344 | .187 | .157 | .195 / .169 / .079 | .705 ± .028 | .761 ± .026 | .96 | .91 ± .10 | .69 ± .09 |

(δ err = δ_model − δ_emp; TV/KL full data, KL in m5_tables.txt: NT KL halves, e.g. .178 → .087; xTV = model fit on
rollouts 1–150 vs empirical on 151–300, floor = the two empirical halves; all ± are 95% over 1,000 hands; TV CIs ±.003–.007.)

**Findings.**
1. *NT dependence is mostly cross-suit, not within-suit.* Cross-suit pairs carry 69% (n=5), 72% (n=8) and 56% (n=13)
   of δ − 1 under TA (same under TA+). "Long-suit runs" (within-suit) are the smaller part (.20–.24), though per suit
   they grow steeply with length.
2. *Mechanism (one sentence): in NT, winning any trick gives the lead and the lead is what converts the hero's other
   suits into tricks, so wins in different suits are complements* — at n = 8 NT the lead-involving terms (LL + LF)
   carry 109% of the cross-suit covariance while follow–follow is negative (−.05). The same holds at n = 5 and 13
   (LL + LF = 110% and 144% of cross; LF alone is the largest term).
3. *Trump rounds: lead complementarity is cancelled by trump-vs-side substitution.* TX < 0 and grows with n
   (−.17 at n=13); within the trump suit TT < 0 and grows with trump length. A trump that ruffs and the side winner it
   replaces are substitutes for the same trick, and the target-aware player spends whichever is needed. Under TA this
   flips δ from 1.14 (n=5) to 0.79 (n=13); TA+ is more under-dispersed (0.73 at n=13, 0.87 at n=8) — its shed/win-high
   components deliberately convert surplus winners into losers, i.e. *more* negative dependence.
4. *Suit-block model: a real but partial fix.* In NT it halves KL, cuts TV by 18–27%, raises bid accuracy by 6–14 points
   (TA n=5: .75 → .87) and cuts EV loss by 24–61% (TA n=5: .83 → .32; n=13: 1.39 → 1.03). In trump it gains 1–3
   points of match and ≈.03–.06 EV. It still misses the cross-suit share, so NT remains far from the A/B noise floor
   (.95–.98); a model of the hero's trick law in NT needs a lead-state variable (e.g. a Markov chain on
   'hero holds the lead'), not more suit detail.

---------------------------------------------------------------------------------

## M8: the 2-card round — where myopic bidding breaks

**Design** (`m8_sim.py`, `m8_eq.py`, `m8_index.py`, `m8_robust.py`; `results/m8_raw_ta.npz`, `m8_eq.json`, `m8_tables.txt`,
`m8_index.json`, `m8_robust.json`). n = 2 is always a trump round in this ladder; trump = ♠, seat 0 bids first and
leads, seat 3 deals. Play policy fixed = TA, which plays on its *own* bid only and deterministically, so a deal plus
a bid vector fixes the trick counts. 300,000 random deals (seed 8080) × all 81 bid vectors were played out.
Information = own hand class (2 cards up to permutation of the side suits: 416 classes) + prior bids. Best responses
group the deals of half A by (class, history), which is the exact MC posterior (the history is a deterministic
function of earlier hands under the current profile), and include the reaction of later bidders. Gauss–Seidel sweeps
dealer → seat 0 until no change. All values are on the held-out half B (150,000 deals, ± 95%).
"Myopic" = the same iteration, but later bidders do not see my deviation (they respond to my profile bid).
Sanity check: the Phase-1 ModelBidder profile on the same deals gives a dealer hook cost of **2.84 ± 0.03**, against
E5's measured **2.84 ± 0.08**.

| profile (hook unless noted) | pos0 (leads) | pos1 | pos2 | dealer | table |
|---|---|---|---|---|---|
| equilibrium (iterated BR) | 7.09 ± .02 | 7.89 ± .02 | 8.21 ± .02 | 5.35 ± .03 | 28.54 ± .03 |
| myopic iteration | 7.07 ± .02 | 7.88 ± .02 | 8.21 ± .02 | 5.37 ± .03 | 28.53 ± .03 |
| TA ModelBidder (Phase 1) | 6.86 ± .02 | 7.86 ± .02 | 8.11 ± .02 | 5.62 ± .03 | 28.45 ± .03 |
| equilibrium, no hook | 7.25 ± .02 | 7.99 ± .02 | 8.26 ± .02 | 8.64 ± .02 | 32.14 ± .04 |
| ModelBidder, no hook | 6.91 ± .02 | 7.93 ± .02 | 8.13 ± .02 | 8.47 ± .02 | 31.44 ± .04 |

Hit rates (eq, hook): .682 / .766 / .795 / .515. Seat-0 bid distribution (eq): 0 66%, 1 14%, **2 20%**.

**Convergence / uniqueness.** The iteration settles within 7 sweeps into a limit cycle that flips 58 near-tie information sets per sweep
(0.35% of the 16,640). Starting from all-0, all-1, all-2 or a random profile gives the same equilibrium up to
0.2–1.6% of deals' bids (dealer largest) and seat values within 0.03; fitting on half B and valuing on half A gives
7.10 / 7.87 / 8.25 / 5.35. **Verified** (MC; the equilibrium is unique up to near-ties).

**Findings.**
1. *Myopia breaks, but only slightly.* At n = 1 the payoff depends on nobody's action (P1). At n = 2 the later bids
   change the later players' play, so a bid also acts as a signal. Best responses with and without this reaction differ on
   4.5 / 4.2 / 3.3% of deals at seats 0–2 (7.3% at the dealer, through the different upstream profile). A unilateral
   switch to the myopic best response costs **0.024 ± .012, 0.032 ± .009 and 0.018 ± .008** pts/round at seats 0, 1, 2, and
   0.002 ± .002 at the dealer (who bids last, so is myopic by construction). Equilibrium and myopic profiles have the same seat
   values to ±0.02.
2. *Equilibrium vs the Phase-1 bidder.* Equilibrium bidding earns the leader +0.23 and seats 1–2 +0.03/+0.10, and the dealer
   **−0.27**. The table gains only +0.09. The leader bids 2 with ♠A/♠K + any side card and with most ♠♠ hands (EV ♠A+♥6: bid 2
   5.48 vs bid 1 5.32). Followers who bid 0 duck, so the leader's side card often survives. Two side aces bid 0 (EV 5.09 vs 4.57
   for bid 1): with 2-card hands someone is usually void and ruffs (in 51% of deals the hand wins *no* trick under bid 0).
3. *Threshold / index rule: approximately, with the right index.* In every (seat, history) cell, take the best monotone
   rule (two cut points along one history-free hand index computed at seat 0), fit it on A and value it on B. Against the
   equilibrium it costs:

| index | pos0 | pos1 | pos2 | dealer | mass consistent (pos0..3) |
|---|---|---|---|---|---|
| E[t \| hand, bid 0] ("tricks you cannot avoid") | 0.17 | 0.07 | 0.08 | 0.02 | .88 / .94 / .94 / .95 |
| E[t \| hand, bid 1] | 0.24 | 0.26 | 0.32 | 0.08 | .84 / .87 / .87 / .93 |
| E[t \| hand, bid 2] ("tricks you can force") | 0.24 | 0.28 | 0.34 | 0.13 | .83 / .86 / .86 / .93 |

   (pts/round on 150k held-out deals; per-round SE ≈ 0.01). The n = 1 threshold structure (P2, exact) survives only
   approximately. The best single index is the *unavoidable*-trick count, not the makeable-trick count, and even it
   leaves 1–2% of seat value. Trump-ace hands behave like 2-trick hands while side-ace hands behave like 0-trick hands,
   which a one-dimensional strength order cannot capture.
4. *Hook cost at n = 2.* Equilibrium: **3.29 ± 0.03** (no-hook dealer 8.64 vs 5.35). ModelBidder profile: 2.84 ± 0.03,
   equal to E5's measured 2.84. When the hook is removed the others gain too: +0.16, +0.10, +0.05. So the hook costs *more*
   between equilibrium bidders, because earlier bidders exploit the dealer's restricted options better than the Phase-1 model
   does. The decay chain is now: 5.48 (n=1, exact) → 3.29 (n=2, equilibrium; 2.84 for the Phase-1 bidders).

---------------------------------------------------------------------------------

## M9: why "win-high" and "lead low at target" work, as exchange arguments

**Setting.** Two cards a ≻ b of the *same suit* (a higher). Comparing "I hold a" with "I hold b" (everything else
equal) is a swap. Legal moves are identical in the two worlds, because the follow-suit constraint only sees suits.

**Assumption O (open-loop opponents).** The opponents' cards in each trick are a function of the deal, the trick index,
who leads and the led suit. In particular they do not react to *which* of a/b I showed. Double-dummy endgames with forced
replies satisfy O. Real (TA) opponents violate it mildly: their beliefs about unseen cards change.

**Lemma E (first divergence). Proved under O.** Let σ be any continuation strategy holding a, and let σ′ hold b and play b
wherever σ plays a. The two plays coincide up to the trick τ in which a or b is played. In τ the other three cards are the
same (by O). If a does not win τ, then b does not win it either, and the winner is the same card, because anything that beats
a beats b. Play then continues identically. If a wins τ, b may or may not win it. So the *only* possible difference is
that I win τ with a and lose it with b, and every trick before τ is identical.

**Proposition W (win-high when at need 1). Proved under O.** Suppose I am to play with need = 1, and two same-suit
cards a ≻ b both win this trick for sure. Playing a now, so that I keep b, weakly dominates playing b now. *Proof:* after this
trick need = 0. By Lemma E the worlds differ only at τ. There the keep-a world wins a trick at need 0, which is an overshoot
and hence a certain miss, while the keep-b world loses it. Hit(keep b) ≥ hit(keep a) pathwise. Optimising over the continuation
with b only helps. ∎
*Mirror (need = r, must win every trick): keep the high card, i.e. win with the cheapest sure winner.* At τ the keep-b
world loses a trick, which is a certain miss. This is TA's R4|need=rem rule, so it is **proved** too.
*In between there is no dominance.* With 1 ≤ need′ < r′ after the trick, winning τ can be good or bad, and both signs occur.
So TA+'s margin rule (spare a winner only if E_w − lead_pw(c) ≥ need − 1 + margin) is a heuristic, not a theorem.
*Scope:* when following, the sure winners are always in one suit (the led suit, or trumps if I am void), so W covers every
following decision at need 1. Across suits (when leading) there is no order, and that part of win_high stays heuristic.
The proposition has content only when an outstanding card lies *between* b and a. Touching cards are equivalent.

**Proposition L (lead at target). Proved under O.** At need = 0 on lead, let a ≻ b be same-suit cards that both lose this
trick for sure when led. Under O the trick has the same winner either way. Then leading a (keeping b) weakly dominates leading b.
*Proof:* the trick result is identical, and afterwards the worlds differ by holding b vs a at need 0, which Proposition W's
argument settles in favour of b. ∎
So the dominance principle at target is **"dispose of the highest card that is still certain to lose"**, not "lead low".
"Lead low" (TA's R2: lead the card of least P(win), ties → lowest) is the right *myopic* rule when no card is a sure loser,
because the immediate trick risk is the only first-order loss. Among several sure losers it picks the wrong one.

**Empirical test** (`m9_play.py`, `m9_signs.py`; `results/m9_play.json`, `m9_signs.json`). There are 30,000 rounds per
variant, with the round index cycling through the 25-round ladder and the hero seat rotating. The other three seats play
the base policy, and each pair is CRN-paired on the same deal and seeds. Variants:
W = following at need 1 with ≥ 2 sure winners, win with the highest. L = leading at need 0 with ≥ 2 sure losers (pw = 0),
lead the highest-value one (any suit). L_same = the same, but only within the suit of TA's own choice (exactly Prop. L).

| variant (base) | Δ score / round | Δ P(hit) | decisions changed | rounds better / worse |
|---|---|---|---|---|
| W (TA) | **+0.035 ± 0.008** | +0.0029 ± 0.0006 | 842 | 90 / 4 |
| L_same (TA) | +0.012 ± 0.006 | +0.0009 ± 0.0005 | 1,338 | 41 / 15 |
| L any suit (TA) | +0.031 ± 0.013 | +0.0023 ± 0.0010 | 1,764 | 156 / 86 |
| L_same (TA+) | +0.011 ± 0.007 | +0.0008 ± 0.0005 | 1,349 | – |
| L any suit (TA+) | +0.037 ± 0.014 | +0.0028 ± 0.0011 | 1,844 | – |

**Findings.**
1. W is **verified** as almost dominant against real TA opponents: 90 rounds better and 4 worse. The 4 are violations of O
   (opponents react to the revealed card). It is worth +0.87 pts/game (×25 rounds). That is the provable part of TA+'s win_high
   (+1.54/game as a single component in experiments/play).
2. The theory predicts that "highest sure loser on lead at target" beats TA's "lowest", and it does. Same-suit: +0.012/round,
   41 vs 15. Cross-suit, where no theorem applies: +0.031–0.037/round. **New candidate TA+ component** (~+0.9 pts/game
   on TA+ base). **Confirmed on fresh seeds** (seed base 9191, 60,000 paired rounds, `results/m9_play_seed9191.json`):
   L any-suit on TA+ +0.030 ± 0.010/round (+0.75/game), L_same +0.013 ± 0.006. No parameter was tuned, so this is not
   a selection effect; a full-game 1v3 in experiments/play remains to be run.
3. The play rules that work are all the same exchange: *spend your dangerous cards on tricks whose outcome is already
   decided*. That covers shed (following, when losing), win-high (following, when winning) and highest-sure-loser (leading).
   "Lead low" is only the fallback when no trick outcome is decided.

---------------------------------------------------------------------------------

## Round 2: status of claims

| claim | status |
|---|---|
| Lemma R: μ ≤ m ⇒ P(m+1) ≤ m/(m+1)·P(m) for PB | proved (Hoeffding extremal reduction); 0 violations in 39k vectors incl. hill-climbs |
| T6 (L3′ upper side): b* < μ+1 for PB, hence b* ∈ {⌊μ⌋, ⌈μ⌉} | **proved**; verified numerically |
| sharp constant sup(b*−μ) = max_{m<N}(m+1)/(1+c_m(m+1)) (.968 at N=13), extremal Bin(m+1, a*) | proved; attained to 1e-6 |
| NT over-dispersion is mainly cross-suit (56–72% of δ−1), lead-borne (LL+LF = 109% of cross at n=8) | verified (6k hands × 300, TA & TA+) |
| trump-round under-dispersion at large n comes from trump×side substitution (TX < 0) and negative TT | verified |
| suit-block model fixes the within-suit part only; NT bid match +6–14 pts, EV loss −24–61%, still far from floor | verified |
| M3's "forced-up miscalibration" explains the hook-cost level gap | **refuted as the main cause**: the natural-bid P(hit) under-prediction is; 3 pooled factors close meas/pred 1.32 → 1.06 out of sample |
| A1 with calibrated P(hit) predicts hook cost by n | verified for n ≥ 2 (n-bucket factors, inside CI); n = 1 still low (4.94 vs 5.46) |
| 2-card equilibrium unique up to near-ties | verified (4 starts, both halves) |
| myopic bidding at n = 2 | breaks slightly: unilateral myopic BR loses 0.02–0.03/round at seats 0–2 |
| threshold/index rule at n = 2 | approximately: best 1-D index (E[t\|bid 0]) loses 0.02–0.17/round; exact at n = 1 only |
| equilibrium hook cost n = 2 | 3.29 ± 0.03 (Phase-1 bidder profile 2.84 ± 0.03 = E5) |
| Lemma E / Prop W (keep-low at need 1, keep-high at need = r) | proved for same-suit pairs under open-loop opponents; verified vs TA (90 better / 4 worse) |
| Prop L (lead highest sure loser at target, same suit) | proved for same-suit pairs under open-loop opponents; verified +0.012 ± 0.006/round (fresh seeds +0.013 ± 0.006); cross-suit variant +0.030 ± 0.010 on TA+ (heuristic, fresh seeds) |

## Reproduce (round 2)
```
python math/m5_sim.py --ns 5,8,13 --hands 1000 --R 300   # ~38 min, one core
python math/m5_analyze.py && python math/m5_lead.py       # ~2 min + ~2 min
python math/m6_bound.py                                   # ~10 s
python math/m7_hookcal.py --games 10000                   # <1 min
python math/m8_sim.py --N 300000 --pol ta                 # ~22 min
python math/m8_eq.py && python math/m8_index.py && python math/m8_robust.py   # ~3 min
python math/m9_play.py 30000 && python math/m9_signs.py   # ~6 min
pytest -q -p no:cacheprovider tests/test_math2_*.py
```


# ROUND 3 (H1–H3)

Rules as confirmed (4 players, hook on dealer, left of dealer bids and leads first, hit = 10 + bid,
miss = 0). All numbers: mean ± 95% CI (normal, deal- or hand-level), n = number of deals/hands/games
stated; CRN = identical deal and identical per-seat rng streams across conditions; fixed seeds.

## H1: the cooperation surplus with hidden cards

**Question.** Phase 2 (F1) measured, with perfect information (PI), that 81% of a seat's score at n = 5
depends on the opponents playing for themselves (max^n − paranoid). How much of that survives when the
opponents cannot see the cards?

**Design** (`experiments/surplus_hidden/sim.py`, `judgement/policies/spoiler.py`). Hero = TA+
(`ta_plus_v2`) bidding with the TA bid model, seat 0. Opponents: self-interested TA+ (same bidder), or
`SpoilerPolicy(target = hero)`: ignores its own score while the hero's outcome is open; DENY (minimise
P(hero wins this trick)) while the hero needs tricks, GIVE (maximise it) when the hero is at its bid,
using the hero's bid, tricks won and TA's card-tracking estimates (unseen cards, known voids,
hypergeometric beat probabilities); ties: DENY keeps strength (cheapest card), GIVE sheds it (most
valuable card); once the hero has missed it reverts to TA+ for itself. Spoilers bid with the same model
as everyone else, so bids are identical across conditions on a CRN deal (only play differs).
Conditions per deal: self (0 spoilers), s1 (1 spoiler at hero offset 1 + (d//4)%3), s2 (pairs
{1,2},{1,3},{2,3} by (d//4)%3), s3 (3 coordinated spoilers). Round mode: n = 1..13, 10,000 deals each,
seed SeedSequence([131000000, n, d]), trump cycle[d%5], dealer d%4 (hero position cycles). Game mode:
4,000 full games (seed 133000000 + g, hero seat g%4, spoilers target the hero all game).
surplus = score(self) − score(spoilers); share = surplus / score(self) (delta-method CI).

Stronger spoilers (same hero, same deals, conditions self/s3; `analyze_strong.py`):
* PIMC-paranoid (`PIMCSpoilerPolicy`, K = 16 sampled worlds consistent with the spoiler's information;
  each candidate card solved exactly with `Paranoid(target)`; hidden information);
* MC with a hero model (`MCSpoilerPolicy`, K = 8 worlds, rollouts with the hero's TRUE policy TA+ and
  heuristic spoiler mates; hidden information but knows how the hero plays);
* omniscient (`OmniscientSpoilerPolicy`): the same Paranoid solver on the TRUE hands — a cheating
  reference that isolates what hidden information costs the spoilers, holding the algorithm fixed.

## H1 per-round: hero TA+ (model bid, seat 0, dealer d%4, trump cycle d%5); CRN deals; 95% CI (n = deals)
| n | deals | self | 1 spoiler | 2 spoilers | 3 spoilers | surplus (s3) | share s3 | share s1 | share s2 | PI share | PI surplus | hit self | hit s3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 10000 | 7.41 ± 0.09 | 7.41 ± 0.09 | 7.41 ± 0.09 | 7.41 ± 0.09 | 0.00 ± 0.00 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 | 0.00 | 0.725 ± 0.009 | 0.725 ± 0.009 |
| 2 | 10000 | 7.05 ± 0.09 | 6.84 ± 0.10 | 6.64 ± 0.10 | 6.41 ± 0.10 | 0.64 ± 0.08 | 0.091 ± 0.010 | 0.030 ± 0.007 | 0.059 ± 0.009 | 0.230 | 1.82 | 0.683 ± 0.009 | 0.621 ± 0.010 |
| 3 | 10000 | 6.77 ± 0.10 | 6.42 ± 0.10 | 6.11 ± 0.10 | 5.72 ± 0.10 | 1.05 ± 0.10 | 0.155 ± 0.014 | 0.052 ± 0.010 | 0.097 ± 0.013 | 0.496 | 4.02 | 0.646 ± 0.009 | 0.545 ± 0.010 |
| 4 | 10000 | 6.62 ± 0.10 | 6.15 ± 0.10 | 5.67 ± 0.11 | 5.21 ± 0.11 | 1.40 ± 0.11 | 0.212 ± 0.016 | 0.071 ± 0.012 | 0.144 ± 0.015 | 0.706 | 5.88 | 0.621 ± 0.010 | 0.487 ± 0.010 |
| 5 | 10000 | 6.49 ± 0.10 | 5.99 ± 0.11 | 5.42 ± 0.11 | 4.82 ± 0.11 | 1.67 ± 0.13 | 0.257 ± 0.017 | 0.077 ± 0.014 | 0.164 ± 0.017 | 0.810 | 6.93 | 0.600 ± 0.010 | 0.441 ± 0.010 |
| 6 | 10000 | 6.28 ± 0.11 | 5.79 ± 0.11 | 5.26 ± 0.11 | 4.53 ± 0.11 | 1.75 ± 0.13 | 0.278 ± 0.019 | 0.078 ± 0.016 | 0.162 ± 0.018 | 0.889 | 7.82 | 0.569 ± 0.010 | 0.405 ± 0.010 |
| 7 | 10000 | 6.37 ± 0.11 | 5.70 ± 0.11 | 5.11 ± 0.11 | 4.33 ± 0.11 | 2.04 ± 0.14 | 0.320 ± 0.019 | 0.104 ± 0.017 | 0.198 ± 0.019 | nan | nan | 0.566 ± 0.010 | 0.376 ± 0.009 |
| 8 | 10000 | 6.42 ± 0.11 | 5.71 ± 0.11 | 4.87 ± 0.11 | 3.97 ± 0.11 | 2.45 ± 0.14 | 0.382 ± 0.019 | 0.110 ± 0.018 | 0.241 ± 0.019 | nan | nan | 0.561 ± 0.010 | 0.337 ± 0.009 |
| 9 | 10000 | 6.29 ± 0.12 | 5.64 ± 0.12 | 4.65 ± 0.12 | 3.76 ± 0.11 | 2.53 ± 0.15 | 0.402 ± 0.020 | 0.103 ± 0.019 | 0.260 ± 0.020 | nan | nan | 0.540 ± 0.010 | 0.310 ± 0.009 |
| 10 | 10000 | 6.42 ± 0.12 | 5.64 ± 0.12 | 4.70 ± 0.12 | 3.74 ± 0.11 | 2.68 ± 0.16 | 0.418 ± 0.020 | 0.121 ± 0.019 | 0.268 ± 0.021 | nan | nan | 0.540 ± 0.010 | 0.302 ± 0.009 |
| 11 | 10000 | 6.65 ± 0.12 | 5.61 ± 0.12 | 4.70 ± 0.12 | 3.71 ± 0.11 | 2.94 ± 0.16 | 0.442 ± 0.019 | 0.155 ± 0.019 | 0.292 ± 0.020 | nan | nan | 0.549 ± 0.010 | 0.292 ± 0.009 |
| 12 | 10000 | 6.87 ± 0.12 | 5.86 ± 0.12 | 4.80 ± 0.12 | 3.84 ± 0.12 | 3.03 ± 0.17 | 0.441 ± 0.019 | 0.147 ± 0.020 | 0.302 ± 0.020 | nan | nan | 0.555 ± 0.010 | 0.296 ± 0.009 |
| 13 | 10000 | 6.98 ± 0.13 | 5.92 ± 0.13 | 4.67 ± 0.12 | 3.52 ± 0.12 | 3.46 ± 0.17 | 0.496 ± 0.018 | 0.152 ± 0.020 | 0.331 ± 0.020 | nan | nan | 0.552 ± 0.010 | 0.265 ± 0.009 |

Cost of punishment (spoiler's own score: self-interested minus spoiler, same seat, CRN) and bystander gain (non-spoiler opponents)
| n | spoiler own s3 | cost s3 | cost s2 | cost s1 | bystander gain s1 | bystander gain s2 |
|---|---|---|---|---|---|---|
| 1 | 7.36 ± 0.04 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 2 | 6.80 ± 0.05 | 0.33 ± 0.04 | 0.29 ± 0.05 | 0.29 ± 0.06 | -0.01 ± 0.03 | -0.00 ± 0.06 |
| 3 | 6.09 ± 0.05 | 0.67 ± 0.05 | 0.64 ± 0.06 | 0.62 ± 0.08 | 0.01 ± 0.04 | 0.01 ± 0.08 |
| 4 | 5.52 ± 0.06 | 1.03 ± 0.06 | 1.05 ± 0.07 | 1.10 ± 0.10 | 0.02 ± 0.05 | 0.09 ± 0.10 |
| 5 | 5.11 ± 0.06 | 1.28 ± 0.07 | 1.35 ± 0.08 | 1.30 ± 0.11 | 0.05 ± 0.06 | 0.14 ± 0.11 |
| 6 | 4.77 ± 0.06 | 1.63 ± 0.08 | 1.70 ± 0.09 | 1.75 ± 0.12 | 0.02 ± 0.07 | 0.05 ± 0.12 |
| 7 | 4.61 ± 0.06 | 1.72 ± 0.08 | 1.73 ± 0.09 | 1.83 ± 0.13 | 0.08 ± 0.07 | 0.14 ± 0.13 |
| 8 | 4.45 ± 0.07 | 1.95 ± 0.08 | 2.09 ± 0.10 | 2.20 ± 0.13 | 0.03 ± 0.08 | 0.13 ± 0.14 |
| 9 | 4.25 ± 0.07 | 2.07 ± 0.09 | 2.21 ± 0.10 | 2.39 ± 0.14 | 0.10 ± 0.08 | 0.24 ± 0.14 |
| 10 | 4.29 ± 0.07 | 2.19 ± 0.09 | 2.40 ± 0.11 | 2.51 ± 0.14 | 0.06 ± 0.09 | 0.20 ± 0.15 |
| 11 | 4.25 ± 0.07 | 2.26 ± 0.09 | 2.40 ± 0.11 | 2.65 ± 0.15 | 0.20 ± 0.10 | 0.26 ± 0.15 |
| 12 | 4.30 ± 0.07 | 2.52 ± 0.09 | 2.72 ± 0.11 | 2.92 ± 0.16 | 0.11 ± 0.10 | 0.33 ± 0.16 |
| 13 | 4.36 ± 0.07 | 2.71 ± 0.10 | 3.00 ± 0.12 | 3.16 ± 0.16 | 0.08 ± 0.10 | 0.33 ± 0.16 |

Surplus (s3) by hero bidding position (0 = first bidder/leader, 3 = dealer)
| n | pos0 | pos1 | pos2 | pos3 |
|---|---|---|---|---|
| 1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 2 | 1.04 ± 0.17 | 0.72 ± 0.16 | 0.66 ± 0.15 | 0.14 ± 0.13 |
| 3 | 1.09 ± 0.20 | 1.31 ± 0.20 | 1.11 ± 0.21 | 0.69 ± 0.20 |
| 4 | 1.31 ± 0.22 | 1.66 ± 0.23 | 1.59 ± 0.24 | 1.05 ± 0.23 |
| 5 | 1.68 ± 0.24 | 1.62 ± 0.26 | 1.73 ± 0.25 | 1.64 ± 0.25 |
| 6 | 1.61 ± 0.26 | 2.02 ± 0.27 | 1.70 ± 0.27 | 1.66 ± 0.27 |
| 7 | 2.03 ± 0.27 | 1.99 ± 0.29 | 1.91 ± 0.29 | 2.23 ± 0.27 |
| 8 | 2.53 ± 0.28 | 2.41 ± 0.29 | 2.45 ± 0.30 | 2.41 ± 0.29 |
| 9 | 2.75 ± 0.29 | 2.64 ± 0.30 | 2.33 ± 0.30 | 2.40 ± 0.31 |
| 10 | 2.93 ± 0.31 | 2.82 ± 0.31 | 2.74 ± 0.31 | 2.23 ± 0.32 |
| 11 | 3.43 ± 0.32 | 2.67 ± 0.33 | 2.97 ± 0.32 | 2.67 ± 0.32 |
| 12 | 3.30 ± 0.32 | 2.96 ± 0.33 | 3.01 ± 0.33 | 2.84 ± 0.33 |
| 13 | 3.88 ± 0.32 | 3.39 ± 0.33 | 3.11 ± 0.34 | 3.48 ± 0.34 |

## H1 full games (hero seat g%4, spoilers target hero all game), 4000 games, CRN
| cond | hero total | P(win) | surplus | share |
|---|---|---|---|---|
| self | 166.0 ± 0.9 | 0.242 ± 0.013 | – | – |
| s1 | 150.8 ± 0.9 | 0.207 ± 0.012 | 15.16 ± 0.90 | 0.091 ± 0.005 |
| s2 | 132.6 ± 0.9 | 0.150 ± 0.011 | 33.34 ± 1.03 | 0.201 ± 0.006 |
| s3 | 114.3 ± 0.9 | 0.167 ± 0.011 | 51.66 ± 1.10 | 0.311 ± 0.006 |
spoilers' own total (s3) 125.8 ± 0.5, cost vs self-interested 40.0 ± 0.6; P(hero outscores every spoiler) 0.164 ± 0.011


## H1 spoiler strength vs information (same TA+ hero, same CRN deals; share = surplus / self score)
heuristic = SpoilerPolicy (rows restricted to the same deals as each comparison); PIMC = PIMC-paranoid K=16 (hidden info); omni = cheating PI paranoid solver on the true hands; PI = Phase-2 max^n − paranoid (PI hero AND PI opponents, scheme R)
MC = hidden-info Monte-Carlo spoiler that KNOWS the hero's policy (rollouts: hero TA+, mates heuristic spoilers)
| n | deals | self | heuristic s3 | PIMC s3 | MC s3 | omni s3 | share heur | share PIMC | share MC | share omni | PI share | PIMC−heur (pts) | omni−PIMC (pts) | hit s3 heur/PIMC/MC/omni | spoiler cost PIMC | spoiler cost omni | CRN self check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1000 | 7.04 ± 0.30 | 6.33 ± 0.31 | 6.14 ± 0.31 | 6.26 ± 0.31 | 5.59 ± 0.32 | 0.101 ± 0.032 | 0.128 ± 0.032 | 0.112 ± 0.033 | 0.206 ± 0.032 | 0.230 | -0.19 ± 0.12 | -0.55 ± 0.15 | 0.615/0.598/0.609/0.546 | 0.35 ± 0.14 | 0.28 ± 0.14 | True |
| 3 | 1000 | 6.75 ± 0.31 | 5.78 ± 0.33 | 5.59 ± 0.33 | 5.70 ± 0.33 | 3.54 ± 0.31 | 0.145 ± 0.047 | 0.173 ± 0.045 | 0.157 ± 0.043 | 0.476 ± 0.040 | 0.496 | -0.19 ± 0.23 | -2.05 ± 0.27 | 0.550/0.534/0.544/0.340 | 0.69 ± 0.18 | 0.54 ± 0.18 | True |
| 4 | 1000 | 6.23 ± 0.33 | 4.90 ± 0.33 | 4.99 ± 0.33 | 5.02 ± 0.33 | 1.97 ± 0.26 | 0.213 ± 0.054 | 0.199 ± 0.052 | 0.193 ± 0.053 | 0.684 ± 0.039 | 0.706 | 0.09 ± 0.28 | -3.02 ± 0.31 | 0.457/0.468/0.471/0.186 | 0.96 ± 0.20 | 0.94 ± 0.20 | True |
| 5 | 1000 | 6.53 ± 0.33 | 4.71 ± 0.34 | 4.44 ± 0.33 | 4.71 ± 0.33 | 1.00 ± 0.19 | 0.278 ± 0.054 | 0.320 ± 0.052 | 0.278 ± 0.052 | 0.847 ± 0.029 | 0.810 | -0.28 ± 0.32 | -3.44 ± 0.32 | 0.435/0.413/0.437/0.095 | 1.25 ± 0.23 | 0.95 ± 0.24 | True |
| 6 | 2000 | 6.30 ± 0.24 | 4.38 ± 0.24 | 4.12 ± 0.24 | – | 0.52 ± 0.10 | 0.304 ± 0.041 | 0.346 ± 0.040 | – | 0.917 ± 0.016 | 0.889 | -0.26 ± 0.25 | -3.60 ± 0.23 | 0.393/0.369/nan/0.049 | 1.59 ± 0.17 | 1.45 ± 0.17 | True |
| 7 | 1000 | 6.38 ± 0.35 | 4.60 ± 0.35 | 4.34 ± 0.35 | – | 0.26 ± 0.11 | 0.279 ± 0.062 | 0.320 ± 0.060 | – | 0.959 ± 0.017 | nan | -0.26 ± 0.36 | -4.08 ± 0.35 | 0.397/0.377/nan/0.023 | 1.63 ± 0.25 | 1.67 ± 0.25 | True |
| 8 | 1000 | 6.66 ± 0.36 | 4.15 ± 0.35 | – | – | 0.19 ± 0.09 | 0.378 ± 0.057 | – | – | 0.971 ± 0.014 | nan | – | – | 0.351/nan/nan/0.017 | – | 1.84 ± 0.27 | True |


**Findings.**
1. With hidden cards the heuristic spoiler surplus share is 9% (n = 2), 16% (3), 21% (4), 26% (5), 28% (6),
   rising to 50% at n = 13 — about one third of the PI curve (23/50/71/81/89%) at n = 2..6. In full games
   three spoilers cost the hero 51.7 ± 1.1 points (31.1 ± 0.6% of its 166.0 self-interested total, 4,000
   games); one spoiler 15.2 (9.1%), two 33.3 (20.1%): roughly additive per spoiler.
2. **More informed spoilers attain more punishment; optimal hidden punishment is open.** On the common n=5 subset (1,000 deals), omni minus PIMC hero score is −3.44 ± 0.32 points/round; shares are omni 0.847 ± 0.029 and PIMC 0.320 ± 0.052. These replace stale interim values. On the larger paired same-hero sample, heuristic/omni surplus is 0.296 [0.264, 0.327] (4,000 deals); PIMC/omni 0.356 [0.313, 0.398] (2,000), MC/omni 0.329 [0.268, 0.386] (1,000). The punishment gap was not attained by the three hidden-information spoilers tried; it is not proved unattainable by a stronger hidden coalition. See round7_audit/AUDIT.md §3.1.
3. **Punishment is expensive and does not convert into wins.** Spoilers give up 1.0–2.7 points/round of
   their own (n = 4..13; ≈ 75–80% of the damage they do) and 40.0 ± 0.6 points/game (s3). Against three
   spoilers the hero's P(win) falls only from .242 to .167 (games): the spoilers' own totals (125.8) drop
   below the self-interested level, so the hero still outscores all three in 16.4% of games. P(win) is
   lowest with TWO spoilers (.150 ± .011): the remaining self-interested opponent is the main beneficiary.
   Bystanders gain little (≤ 0.3 points/round).
4. Position: the hook protects the dealer at small n (n = 2 surplus 0.14 ± 0.13 at the dealer seat vs
   1.04 ± 0.17 for the first bidder/leader); by n ≥ 5 position no longer matters (all four seats within CI).
5. n = 1: surplus exactly 0 in every condition (forced play; Corollary H3.3a).

## H3: propositions

Setting: one deal, bids fixed (in H1 all seats bid with the same model in every condition, so bids
coincide across conditions on a CRN deal), hero policy π, and a coalition of the other three seats
minimising the hero's score. S(π, σ) = hero's expected score against coalition strategy σ. Σ_H =
coalition strategies in which each spoiler's action is measurable w.r.t. its own information (hidden
cards, no communication beyond the play); Σ_PI ⊃ Σ_H = strategies that may depend on all hands.

**Proposition H3.1 (information sandwich).** For every hero policy π and every σ ∈ Σ_H,
S(π, σ) ≥ β(π) := min_{τ∈Σ_PI} S(π, τ), and β(π) ≤ V_par (the PI paranoid value). Hence
hidden surplus(π, σ) ≤ S(π, self) − β(π), and S(π, self) − β(π) ≥ S(π, self) − V_par.
*Proof.* Σ_H ⊂ Σ_PI gives the first inequality. With all hands known, hero vs coalition is a finite
two-person zero-sum game of perfect information with value V_par = max_ρ min_τ S(ρ, τ) over hero
strategies ρ that may see everything; π is one such ρ (it just ignores the extra information), so
min_τ S(π, τ) ≤ V_par. ∎
*Use.* Any concrete PI coalition (omni) gives S(π, omni) ≥ β(π), so its surplus is a lower bound on
the PI-attainable surplus against π; any concrete hidden-information spoiler's surplus is a lower
bound on sup_{σ∈Σ_H}. The gap "omni − best hidden spoiler found" is the information cost of spoiling
under the caveat that a better hidden-information spoiler may exist (PIMC and the policy-aware MC
spoiler are the strongest tried).

**Proposition H3.2 (hero-led tricks are information-free, trick by trick).** In a trick the hero leads,
for either objective (DENY: the hero loses this trick; GIVE: the hero wins it), the coalition achieves
the objective under hidden information iff it can under perfect information, and the myopic rule
(DENY: beat the current best if you can; GIVE: never beat it unless forced) achieves it.
*Proof.* The three spoilers act after the hero's card is public, and each sees the current best card
when it acts; whether it holds a legal beating (DENY) or non-beating (GIVE) card depends on its own hand
only. DENY: the hero loses iff some spoiler beats the then-current best (the hero has already played);
if any spoiler can, the first one able to does so under the rule; if none can, no strategy can. GIVE:
the hero wins iff no spoiler beats, and a spoiler beats only when all its legal cards beat — forced
under every strategy. ∎
So hidden information can only bite through (i) tricks in which some spoiler commits before the hero
plays and (ii) WHICH beating / non-beating card is used (resource choice for later tricks).

**Corollary H3.3.** (a) n = 1: all play is forced, surplus ≡ 0 against every coalition (measured 0.00
on 10,000 CRN deals in all four conditions; PI Phase 2: 0.00). (b) The last trick of every round is
forced. (c) Combined with H3.2, a hero who leads every trick it must win cannot be denied by hidden
information more than by PI *within* those tricks; the measured PI–hidden gap must come from
spoiler-first tricks and resource choice.

**Counterexample (H3.1 is strict; "three coordinated spoilers can reproduce PI punishment" is false).**
NT, n = 2, dealer seat 0, hero = seat 1 (leads), bids hero 1, all others 0. Seat 2 holds K♥ 3♣; seats
3 and 0 hold diamonds (4♦5♦, 6♦7♦). Deal A: hero A♠ 2♥; deal B: hero A♠ 2♣. In both deals the PI
paranoid value is a hero MISS (0): the hero must lead A♠ (leading the deuce first loses both tricks),
and seat 2 must then discard so that the hero's deuce is unbeatable — K♥ in deal A, 3♣ in deal B.
Seat 2's information at that moment (own hand K♥ 3♣, bids, A♠ led) is identical in A and B, so any
hidden-information strategy (discard K♥ with probability q) lets the hero hit (11 points) with
probability 1 − q in A and q in B: with A and B equally likely the hero scores 5.5 vs 0 under PI.
Verified with the exact solver in `tests/test_round3_math_h3.py`.

**Status of the "lower bound from the hook/conservation" idea.** No useful deal-free lower bound on the
hidden surplus follows from conservation: the spoilers do not care about their own bids, so Σ tricks = n
does not constrain them, and the hook only constrains the dealer's bid (identical across conditions in
our design). What H3.1–H3.2 give is the sandwich and the localisation of the information cost.


## H2: a lead-state model for NT trick counts

**Data** (`math/m10_sim.py`): same 1000 deals and rollout seeds as m5 (card masks reproduce m5 exactly),
NT, n = 5, 8, 13, TA and TA+ play, R = 300 rollouts per hand, plus the trick-indexed win sequence
W_1..W_n of the hero (first bidder = first leader). Because the trick winner leads next, the hero's lead
state is L_1 = 1, L_{j+1} = W_j: the lead is a deterministic function of the hero's own wins, so a
"lead-state" model for T is a binary chain on the hero's win sequence.

**Models** (`math/m10_analyze.py`; all fitted per hand from rollouts, as PB and SB are):
PB (cards independent), SB (suit blocks, round 2), TPB (control: tricks independent with trick-indexed
marginals), LC0 (homogeneous lead chain, 3 numbers), **LC1** (time-inhomogeneous lead chain:
a_j = P(W_j | lead), b_j = P(W_j | not lead), 2n − 1 numbers, shrunk with κ = 2 pseudo-rollouts),
LC2 (second order), LCX (logit α_j + lead × phase + lead-specific slopes in the budget
T_{<j} − E T_{<j}; n + 4 numbers), LCS/LCSP (on-lead win rate indexed by on-lead tricks already won =
finite supply), LSB (suit blocks split by trick-1 outcome); LCT/LCU are near-saturated (state includes
tricks so far) and reproduce the empirical pmf in sample by construction — reported only as cross-fit
checks. Cross-fit: fit on rollouts 1–150, test against the empirical pmf of 151–300; the floor is
empirical-A vs empirical-B (the noise floor: bid match .94–.98).

H2 NT (1000 hands × R=300 per row; gap closed = fraction of the PB→floor gap, cross-fit; full tables incl. LCB/LB/LCSP/LCT in math/results/m10_tables.txt)
| play | n | trump | hands | model | TV | δ err | match (in-sample) | xTV | xmatch | xloss | gap closed: xTV | gap closed: xmatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ta | 5 | NT | 1000 | PB | 0.185 ± 0.006 | -0.773 ± 0.029 | 0.750 ± 0.027 | 0.190 ± 0.006 | 0.739 ± 0.027 | 0.91 ± 0.11 | 0.00 | 0.00 |
| ta | 5 | NT | 1000 | SB | 0.135 ± 0.005 | -0.532 ± 0.018 | 0.870 ± 0.021 | 0.144 ± 0.005 | 0.860 ± 0.022 | 0.38 ± 0.07 | 0.38 | 0.50 |
| ta | 5 | NT | 1000 | TPB | 0.181 ± 0.006 | -0.750 ± 0.030 | 0.760 ± 0.026 | 0.186 ± 0.006 | 0.749 ± 0.027 | 0.88 ± 0.11 | 0.03 | 0.04 |
| ta | 5 | NT | 1000 | LC0 | 0.112 ± 0.007 | +0.094 ± 0.092 | 0.882 ± 0.020 | 0.131 ± 0.006 | 0.879 ± 0.020 | 0.43 ± 0.09 | 0.48 | 0.58 |
| ta | 5 | NT | 1000 | LC1 | 0.039 ± 0.002 | -0.084 ± 0.013 | 0.981 ± 0.008 | 0.073 ± 0.002 | 0.978 ± 0.009 | 0.02 ± 0.01 | 0.95 | 0.98 |
| ta | 5 | NT | 1000 | LC2 | 0.021 ± 0.001 | -0.028 ± 0.006 | 0.989 ± 0.006 | 0.068 ± 0.002 | 0.979 ± 0.009 | 0.01 ± 0.01 | 1.00 | 0.99 |
| ta | 5 | NT | 1000 | LCX | 0.025 ± 0.002 | +0.029 ± 0.003 | 0.983 ± 0.008 | 0.071 ± 0.002 | 0.976 ± 0.009 | 0.02 ± 0.01 | 0.97 | 0.98 |
| ta | 5 | NT | 1000 | LCS | 0.113 ± 0.004 | +0.122 ± 0.022 | 0.933 ± 0.016 | 0.125 ± 0.004 | 0.917 ± 0.017 | 0.14 ± 0.04 | 0.53 | 0.73 |
| ta | 5 | NT | 1000 | LSB | 0.129 ± 0.005 | -0.537 ± 0.016 | 0.875 ± 0.021 | 0.139 ± 0.005 | 0.870 ± 0.021 | 0.35 ± 0.07 | 0.42 | 0.54 |
| ta | 5 | NT | 1000 | LCU | 0.007 ± 0.000 | -0.110 ± 0.008 | 0.999 ± 0.002 | 0.067 ± 0.002 | 0.982 ± 0.008 | 0.01 ± 0.01 | 1.00 | 1.00 |
| ta | 5 | NT | 1000 | floor (emp A vs B) | – | δ_emp 1.77 | – | 0.068 | 0.982 ± 0.008 | 0.01 | 1 | 1 |
| ta | 5 | NT | | persistence P(W\|lead)−P(W\|no lead) = +0.311 ± 0.013; acf lag 1/2/3 = +0.330 / +0.140 / +0.057 | | | | | | | | |
| ta_plus | 5 | NT | 1000 | PB | 0.201 ± 0.007 | -0.815 ± 0.029 | 0.655 ± 0.029 | 0.207 ± 0.007 | 0.648 ± 0.030 | 1.15 ± 0.12 | 0.00 | 0.00 |
| ta_plus | 5 | NT | 1000 | SB | 0.150 ± 0.005 | -0.571 ± 0.018 | 0.799 ± 0.025 | 0.159 ± 0.005 | 0.789 ± 0.025 | 0.55 ± 0.08 | 0.35 | 0.45 |
| ta_plus | 5 | NT | 1000 | TPB | 0.197 ± 0.007 | -0.793 ± 0.030 | 0.664 ± 0.029 | 0.203 ± 0.007 | 0.664 ± 0.029 | 1.09 ± 0.11 | 0.03 | 0.05 |
| ta_plus | 5 | NT | 1000 | LC0 | 0.112 ± 0.006 | +0.041 ± 0.058 | 0.872 ± 0.021 | 0.133 ± 0.006 | 0.850 ± 0.022 | 0.43 ± 0.08 | 0.54 | 0.64 |
| ta_plus | 5 | NT | 1000 | LC1 | 0.038 ± 0.002 | -0.107 ± 0.011 | 0.986 ± 0.007 | 0.075 ± 0.002 | 0.962 ± 0.012 | 0.03 ± 0.01 | 0.97 | 1.00 |
| ta_plus | 5 | NT | 1000 | LC2 | 0.022 ± 0.001 | -0.044 ± 0.005 | 0.994 ± 0.005 | 0.071 ± 0.002 | 0.962 ± 0.012 | 0.03 ± 0.01 | 1.00 | 1.00 |
| ta_plus | 5 | NT | 1000 | LCX | 0.025 ± 0.002 | +0.028 ± 0.003 | 0.988 ± 0.007 | 0.073 ± 0.002 | 0.958 ± 0.012 | 0.04 ± 0.01 | 0.98 | 0.99 |
| ta_plus | 5 | NT | 1000 | LCS | 0.124 ± 0.004 | +0.167 ± 0.022 | 0.898 ± 0.019 | 0.136 ± 0.004 | 0.884 ± 0.020 | 0.20 ± 0.04 | 0.52 | 0.75 |
| ta_plus | 5 | NT | 1000 | LSB | 0.143 ± 0.005 | -0.557 ± 0.016 | 0.798 ± 0.025 | 0.154 ± 0.005 | 0.792 ± 0.025 | 0.54 ± 0.08 | 0.39 | 0.46 |
| ta_plus | 5 | NT | 1000 | LCU | 0.007 ± 0.000 | -0.104 ± 0.007 | 1.000 ± 0.000 | 0.070 ± 0.002 | 0.960 ± 0.012 | 0.03 ± 0.01 | 1.00 | 0.99 |
| ta_plus | 5 | NT | 1000 | floor (emp A vs B) | – | δ_emp 1.81 | – | 0.070 | 0.962 ± 0.012 | 0.03 | 1 | 1 |
| ta_plus | 5 | NT | | persistence P(W\|lead)−P(W\|no lead) = +0.320 ± 0.013; acf lag 1/2/3 = +0.330 / +0.149 / +0.079 | | | | | | | | |
| ta | 8 | NT | 1000 | PB | 0.175 ± 0.005 | -0.717 ± 0.030 | 0.717 ± 0.028 | 0.182 ± 0.006 | 0.727 ± 0.028 | 0.74 ± 0.10 | 0.00 | 0.00 |
| ta | 8 | NT | 1000 | SB | 0.133 ± 0.004 | -0.516 ± 0.021 | 0.813 ± 0.024 | 0.144 ± 0.004 | 0.811 ± 0.024 | 0.39 ± 0.06 | 0.36 | 0.38 |
| ta | 8 | NT | 1000 | TPB | 0.168 ± 0.005 | -0.659 ± 0.033 | 0.745 ± 0.027 | 0.175 ± 0.006 | 0.744 ± 0.027 | 0.67 ± 0.09 | 0.06 | 0.08 |
| ta | 8 | NT | 1000 | LC0 | 0.171 ± 0.009 | +0.735 ± 0.123 | 0.721 ± 0.028 | 0.184 ± 0.009 | 0.712 ± 0.028 | 1.24 ± 0.17 | -0.02 | -0.07 |
| ta | 8 | NT | 1000 | LC1 | 0.086 ± 0.004 | -0.006 ± 0.032 | 0.889 ± 0.019 | 0.107 ± 0.004 | 0.885 ± 0.020 | 0.14 ± 0.03 | 0.71 | 0.71 |
| ta | 8 | NT | 1000 | LC2 | 0.066 ± 0.003 | +0.020 ± 0.020 | 0.934 ± 0.015 | 0.095 ± 0.004 | 0.909 ± 0.018 | 0.09 ± 0.02 | 0.82 | 0.82 |
| ta | 8 | NT | 1000 | LCX | 0.047 ± 0.003 | +0.047 ± 0.006 | 0.932 ± 0.016 | 0.086 ± 0.003 | 0.910 ± 0.018 | 0.12 ± 0.03 | 0.91 | 0.82 |
| ta | 8 | NT | 1000 | LCS | 0.148 ± 0.005 | +0.670 ± 0.045 | 0.857 ± 0.022 | 0.155 ± 0.005 | 0.839 ± 0.023 | 0.38 ± 0.06 | 0.25 | 0.50 |
| ta | 8 | NT | 1000 | LSB | 0.130 ± 0.004 | -0.532 ± 0.019 | 0.826 ± 0.024 | 0.142 ± 0.004 | 0.811 ± 0.024 | 0.38 ± 0.06 | 0.38 | 0.38 |
| ta | 8 | NT | 1000 | LCU | 0.007 ± 0.000 | -0.137 ± 0.011 | 0.999 ± 0.002 | 0.075 ± 0.002 | 0.949 ± 0.014 | 0.04 ± 0.01 | 1.01 | 1.00 |
| ta | 8 | NT | 1000 | floor (emp A vs B) | – | δ_emp 1.72 | – | 0.076 | 0.949 ± 0.014 | 0.04 | 1 | 1 |
| ta | 8 | NT | | persistence P(W\|lead)−P(W\|no lead) = +0.290 ± 0.010; acf lag 1/2/3 = +0.240 / +0.091 / +0.033 | | | | | | | | |
| ta_plus | 8 | NT | 1000 | PB | 0.179 ± 0.005 | -0.736 ± 0.030 | 0.718 ± 0.028 | 0.185 ± 0.006 | 0.701 ± 0.028 | 0.81 ± 0.10 | 0.00 | 0.00 |
| ta_plus | 8 | NT | 1000 | SB | 0.136 ± 0.004 | -0.527 ± 0.020 | 0.807 ± 0.024 | 0.147 ± 0.005 | 0.788 ± 0.025 | 0.47 ± 0.07 | 0.36 | 0.37 |
| ta_plus | 8 | NT | 1000 | TPB | 0.177 ± 0.006 | -0.697 ± 0.033 | 0.718 ± 0.028 | 0.183 ± 0.006 | 0.712 ± 0.028 | 0.79 ± 0.10 | 0.01 | 0.05 |
| ta_plus | 8 | NT | 1000 | LC0 | 0.164 ± 0.009 | +0.605 ± 0.098 | 0.734 ± 0.027 | 0.180 ± 0.008 | 0.727 ± 0.028 | 1.09 ± 0.15 | 0.05 | 0.11 |
| ta_plus | 8 | NT | 1000 | LC1 | 0.081 ± 0.004 | -0.074 ± 0.027 | 0.917 ± 0.017 | 0.107 ± 0.004 | 0.904 ± 0.018 | 0.11 ± 0.03 | 0.74 | 0.86 |
| ta_plus | 8 | NT | 1000 | LC2 | 0.061 ± 0.003 | -0.039 ± 0.016 | 0.949 ± 0.014 | 0.095 ± 0.003 | 0.920 ± 0.017 | 0.08 ± 0.02 | 0.86 | 0.92 |
| ta_plus | 8 | NT | 1000 | LCX | 0.049 ± 0.003 | +0.045 ± 0.005 | 0.923 ± 0.017 | 0.090 ± 0.003 | 0.894 ± 0.019 | 0.15 ± 0.03 | 0.91 | 0.81 |
| ta_plus | 8 | NT | 1000 | LCS | 0.148 ± 0.004 | +0.619 ± 0.039 | 0.830 ± 0.023 | 0.157 ± 0.005 | 0.819 ± 0.024 | 0.41 ± 0.07 | 0.27 | 0.50 |
| ta_plus | 8 | NT | 1000 | LSB | 0.134 ± 0.004 | -0.541 ± 0.019 | 0.810 ± 0.024 | 0.145 ± 0.004 | 0.796 ± 0.025 | 0.45 ± 0.07 | 0.38 | 0.40 |
| ta_plus | 8 | NT | 1000 | LCU | 0.008 ± 0.000 | -0.132 ± 0.009 | 0.996 ± 0.004 | 0.079 ± 0.002 | 0.937 ± 0.015 | 0.05 ± 0.02 | 1.01 | 1.00 |
| ta_plus | 8 | NT | 1000 | floor (emp A vs B) | – | δ_emp 1.74 | – | 0.080 | 0.938 ± 0.015 | 0.05 | 1 | 1 |
| ta_plus | 8 | NT | | persistence P(W\|lead)−P(W\|no lead) = +0.287 ± 0.010; acf lag 1/2/3 = +0.230 / +0.091 / +0.045 | | | | | | | | |
| ta | 13 | NT | 996 | PB | 0.192 ± 0.006 | -0.547 ± 0.031 | 0.581 ± 0.031 | 0.199 ± 0.007 | 0.580 ± 0.031 | 1.45 ± 0.13 | 0.00 | 0.00 |
| ta | 13 | NT | 996 | SB | 0.157 ± 0.006 | -0.307 ± 0.024 | 0.657 ± 0.030 | 0.168 ± 0.006 | 0.654 ± 0.030 | 1.07 ± 0.11 | 0.26 | 0.20 |
| ta | 13 | NT | 996 | TPB | 0.188 ± 0.006 | -0.289 ± 0.046 | 0.630 ± 0.030 | 0.196 ± 0.006 | 0.614 ± 0.030 | 1.26 ± 0.12 | 0.03 | 0.09 |
| ta | 13 | NT | 996 | LC0 | 0.300 ± 0.013 | +3.667 ± 0.633 | 0.607 ± 0.030 | 0.309 ± 0.012 | 0.607 ± 0.030 | 1.99 ± 0.21 | -0.93 | 0.07 |
| ta | 13 | NT | 996 | LC1 | 0.192 ± 0.009 | +0.865 ± 0.090 | 0.651 ± 0.030 | 0.205 ± 0.008 | 0.652 ± 0.030 | 1.05 ± 0.11 | -0.05 | 0.19 |
| ta | 13 | NT | 996 | LC2 | 0.162 ± 0.008 | +0.648 ± 0.064 | 0.750 ± 0.027 | 0.181 ± 0.007 | 0.734 ± 0.027 | 0.69 ± 0.09 | 0.15 | 0.41 |
| ta | 13 | NT | 996 | LCX | 0.107 ± 0.006 | +0.034 ± 0.014 | 0.689 ± 0.029 | 0.134 ± 0.006 | 0.692 ± 0.029 | 0.91 ± 0.10 | 0.55 | 0.30 |
| ta | 13 | NT | 996 | LCS | 0.242 ± 0.009 | +2.398 ± 0.134 | 0.826 ± 0.024 | 0.245 ± 0.009 | 0.825 ± 0.024 | 0.56 ± 0.09 | -0.39 | 0.66 |
| ta | 13 | NT | 996 | LSB | 0.157 ± 0.006 | -0.307 ± 0.024 | 0.657 ± 0.030 | 0.168 ± 0.006 | 0.655 ± 0.030 | 1.06 ± 0.11 | 0.26 | 0.20 |
| ta | 13 | NT | 996 | LCU | 0.009 ± 0.000 | -0.103 ± 0.011 | 0.999 ± 0.002 | 0.079 ± 0.002 | 0.954 ± 0.013 | 0.04 ± 0.01 | 1.01 | 1.00 |
| ta | 13 | NT | 996 | floor (emp A vs B) | – | δ_emp 1.55 | – | 0.080 | 0.954 ± 0.013 | 0.04 | 1 | 1 |
| ta | 13 | NT | | persistence P(W\|lead)−P(W\|no lead) = +0.408 ± 0.010; acf lag 1/2/3 = +0.287 / +0.064 / -0.042 | | | | | | | | |
| ta_plus | 13 | NT | 999 | PB | 0.187 ± 0.006 | -0.572 ± 0.032 | 0.705 ± 0.028 | 0.195 ± 0.006 | 0.707 ± 0.028 | 0.89 ± 0.10 | 0.00 | 0.00 |
| ta_plus | 13 | NT | 999 | SB | 0.157 ± 0.006 | -0.344 ± 0.024 | 0.761 ± 0.026 | 0.169 ± 0.006 | 0.757 ± 0.027 | 0.69 ± 0.09 | 0.22 | 0.20 |
| ta_plus | 13 | NT | 999 | TPB | 0.194 ± 0.006 | -0.348 ± 0.045 | 0.709 ± 0.028 | 0.202 ± 0.006 | 0.697 ± 0.029 | 0.93 ± 0.10 | -0.06 | -0.04 |
| ta_plus | 13 | NT | 999 | LC0 | 0.305 ± 0.013 | +4.289 ± 0.573 | 0.571 ± 0.031 | 0.315 ± 0.013 | 0.566 ± 0.031 | 2.36 ± 0.22 | -1.04 | -0.56 |
| ta_plus | 13 | NT | 999 | LC1 | 0.191 ± 0.009 | +0.810 ± 0.093 | 0.626 ± 0.030 | 0.204 ± 0.009 | 0.642 ± 0.030 | 1.13 ± 0.11 | -0.08 | -0.26 |
| ta_plus | 13 | NT | 999 | LC2 | 0.159 ± 0.008 | +0.590 ± 0.066 | 0.730 ± 0.028 | 0.177 ± 0.007 | 0.722 ± 0.028 | 0.78 ± 0.10 | 0.15 | 0.06 |
| ta_plus | 13 | NT | 999 | LCX | 0.115 ± 0.007 | +0.075 ± 0.020 | 0.699 ± 0.028 | 0.142 ± 0.006 | 0.687 ± 0.029 | 1.04 ± 0.11 | 0.46 | -0.08 |
| ta_plus | 13 | NT | 999 | LCS | 0.231 ± 0.009 | +2.136 ± 0.122 | 0.812 ± 0.024 | 0.238 ± 0.009 | 0.790 ± 0.025 | 0.71 ± 0.10 | -0.37 | 0.33 |
| ta_plus | 13 | NT | 999 | LSB | 0.157 ± 0.006 | -0.344 ± 0.024 | 0.761 ± 0.026 | 0.169 ± 0.006 | 0.757 ± 0.027 | 0.69 ± 0.09 | 0.22 | 0.20 |
| ta_plus | 13 | NT | 999 | LCU | 0.009 ± 0.000 | -0.114 ± 0.012 | 1.000 ± 0.000 | 0.078 ± 0.002 | 0.956 ± 0.013 | 0.04 ± 0.01 | 1.01 | 0.99 |
| ta_plus | 13 | NT | 999 | floor (emp A vs B) | – | δ_emp 1.57 | – | 0.079 | 0.958 ± 0.012 | 0.03 | 1 | 1 |
| ta_plus | 13 | NT | | persistence P(W\|lead)−P(W\|no lead) = +0.417 ± 0.011; acf lag 1/2/3 = +0.287 / +0.067 / -0.038 | | | | | | | | |

Contrast: trump (S) rounds, 500 hands × R=300 (gap-closed for match omitted when PB is within .02 of the floor)
| play | n | trump | hands | model | TV | δ err | match (in-sample) | xTV | xmatch | xloss | gap closed: xTV | gap closed: xmatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ta | 5 | S | 500 | PB | 0.075 ± 0.004 | -0.133 ± 0.021 | 0.884 ± 0.028 | 0.091 ± 0.005 | 0.866 ± 0.030 | 0.14 ± 0.04 | 0.00 | 0.00 |
| ta | 5 | S | 500 | SB | 0.061 ± 0.004 | -0.107 ± 0.015 | 0.902 ± 0.026 | 0.082 ± 0.004 | 0.882 ± 0.028 | 0.12 ± 0.04 | 0.40 | 0.33 |
| ta | 5 | S | 500 | LC1 | 0.049 ± 0.003 | +0.006 ± 0.015 | 0.922 ± 0.024 | 0.076 ± 0.004 | 0.902 ± 0.026 | 0.09 ± 0.03 | 0.65 | 0.75 |
| ta | 5 | S | 500 | LCX | 0.031 ± 0.003 | +0.005 ± 0.004 | 0.936 ± 0.021 | 0.071 ± 0.003 | 0.914 ± 0.025 | 0.07 ± 0.02 | 0.88 | 1.00 |
| ta | 5 | S | 500 | floor (emp A vs B) | – | δ_emp 1.13 | – | 0.068 | 0.914 ± 0.025 | 0.07 | 1 | 1 |
| ta | 5 | S | | persistence P(W\|lead)−P(W\|no lead) = +0.008 ± 0.013; acf lag 1/2/3 = +0.016 / -0.013 / +0.010 | | | | | | | | |
| ta_plus | 5 | S | 500 | PB | 0.066 ± 0.005 | -0.047 ± 0.020 | 0.922 ± 0.024 | 0.085 ± 0.005 | 0.876 ± 0.029 | 0.13 ± 0.04 | 0.00 | nan |
| ta_plus | 5 | S | 500 | SB | 0.053 ± 0.004 | -0.051 ± 0.015 | 0.924 ± 0.023 | 0.077 ± 0.004 | 0.878 ± 0.029 | 0.12 ± 0.04 | 0.47 | nan |
| ta_plus | 5 | S | 500 | LC1 | 0.060 ± 0.004 | +0.006 ± 0.020 | 0.938 ± 0.021 | 0.083 ± 0.004 | 0.898 ± 0.027 | 0.08 ± 0.03 | 0.15 | nan |
| ta_plus | 5 | S | 500 | LCX | 0.026 ± 0.002 | -0.011 ± 0.003 | 0.956 ± 0.018 | 0.069 ± 0.003 | 0.904 ± 0.026 | 0.07 ± 0.02 | 0.93 | nan |
| ta_plus | 5 | S | 500 | floor (emp A vs B) | – | δ_emp 1.05 | – | 0.068 | 0.884 ± 0.028 | 0.09 | 1 | 1 |
| ta_plus | 5 | S | | persistence P(W\|lead)−P(W\|no lead) = -0.109 ± 0.015; acf lag 1/2/3 = -0.104 / +0.027 / +0.006 | | | | | | | | |
| ta | 8 | S | 499 | PB | 0.082 ± 0.005 | -0.006 ± 0.020 | 0.884 ± 0.028 | 0.100 ± 0.005 | 0.866 ± 0.030 | 0.17 ± 0.06 | 0.00 | 0.00 |
| ta | 8 | S | 499 | SB | 0.068 ± 0.004 | +0.003 ± 0.016 | 0.900 ± 0.026 | 0.092 ± 0.004 | 0.882 ± 0.028 | 0.11 ± 0.04 | 0.38 | 0.67 |
| ta | 8 | S | 499 | LC1 | 0.090 ± 0.005 | +0.219 ± 0.030 | 0.912 ± 0.025 | 0.108 ± 0.005 | 0.888 ± 0.028 | 0.09 ± 0.03 | -0.38 | 0.92 |
| ta | 8 | S | 499 | LCX | 0.046 ± 0.003 | -0.016 ± 0.006 | 0.894 ± 0.027 | 0.083 ± 0.003 | 0.876 ± 0.029 | 0.11 ± 0.04 | 0.79 | 0.42 |
| ta | 8 | S | 499 | floor (emp A vs B) | – | δ_emp 1.01 | – | 0.078 | 0.890 ± 0.028 | 0.08 | 1 | 1 |
| ta | 8 | S | | persistence P(W\|lead)−P(W\|no lead) = +0.049 ± 0.012; acf lag 1/2/3 = +0.008 / -0.040 / -0.043 | | | | | | | | |
| ta_plus | 8 | S | 500 | PB | 0.094 ± 0.006 | +0.131 ± 0.019 | 0.910 ± 0.025 | 0.109 ± 0.006 | 0.878 ± 0.029 | 0.12 ± 0.04 | 0.00 | 0.00 |
| ta_plus | 8 | S | 500 | SB | 0.078 ± 0.005 | +0.089 ± 0.015 | 0.928 ± 0.023 | 0.098 ± 0.005 | 0.878 ± 0.029 | 0.11 ± 0.04 | 0.31 | 0.00 |
| ta_plus | 8 | S | 500 | LC1 | 0.106 ± 0.006 | +0.253 ± 0.024 | 0.926 ± 0.023 | 0.122 ± 0.006 | 0.870 ± 0.030 | 0.13 ± 0.04 | -0.37 | -0.20 |
| ta_plus | 8 | S | 500 | LCX | 0.043 ± 0.003 | -0.037 ± 0.005 | 0.934 ± 0.022 | 0.077 ± 0.004 | 0.908 ± 0.025 | 0.08 ± 0.03 | 0.88 | 0.75 |
| ta_plus | 8 | S | 500 | floor (emp A vs B) | – | δ_emp 0.87 | – | 0.073 | 0.918 ± 0.024 | 0.07 | 1 | 1 |
| ta_plus | 8 | S | | persistence P(W\|lead)−P(W\|no lead) = -0.014 ± 0.011; acf lag 1/2/3 = -0.055 / -0.026 / -0.051 | | | | | | | | |

**Findings.**
* The lead persistence is large: P(win | lead) − P(win | not lead) on tricks 2..n = .31/.29/.41 (TA,
  n = 5/8/13). Trick-win autocorrelation decays roughly geometrically (n = 5: .33/.14/.06 at lags 1/2/3),
  i.e. Markov-like, not the flat profile an exchangeable "good-world luck" mixture would give; at n = 13
  it turns negative at lag 3 (−.04): finite supply of winners.
* Trick indexing alone (TPB) does nothing (≈ PB). The lead does.
* n = 5: LC1 closes essentially the whole gap: TV .185 → .039 (79%), bid match .750 → .981 in sample
  (92% of the gap to 1), cross-fit match .978 vs floor .982 (98% of the PB → floor gap; TA+ 100%), EV
  loss .83 → .01. δ error −.77 → −.08.
* n = 8: LC1 closes 71% (TA) / 86% (TA+) of the cross-fit bid-match gap; LCX (lead × phase plus
  lead-specific budget slopes, n + 4 numbers) closes 91% / 90% of the xTV gap and 82% / 81% of the
  match gap (xmatch .910 / .894 vs floor .949 / .938).
* n = 13: no parsimonious chain reaches the floor. LC1 over-disperses (δ err +.8: runs of lead wins
  are not geometric — the hand runs out of winners); LCX fixes the dispersion and halves the TV gap but
  bid match improves only modestly; LCS (finite on-lead supply) gets the bid right more often but the
  shape wrong. Fractions of the cross-fit gap closed at n = 13 (TA / TA+): LCX xTV .55 / .46, match
  .30 / −.08; LCS match .66 / .33; SB .20 / .20. The n = 13 dependence needs a per-suit control state (which suits the hero has
  established), not just who holds the lead.
* Trump contrast (S, n = 5/8, 500 hands): lead persistence ≈ 0 (TA +.008 ± .013 / +.049 ± .012; TA+
  −.109 ± .015 / −.014 ± .011) and lag-1 autocorrelation ≈ 0 or negative: the lead mechanism is an NT
  phenomenon (trumps make the lead worth little), consistent with round 2 (TX < 0 cancels it).
* Summary for the bidder: in NT the hero's trick count is, to the noise floor at n ≤ 8, a lead-state
  chain; PB's NT bid errors at n = 5 (25–35% mismatches, 0.8–1.1 points/round) are almost entirely the
  missing lead persistence.

## Round 3: status of claims

| claim | status |
|---|---|
| hidden-info cooperation surplus (3 heuristic spoilers): 9/16/21/26/28% at n = 2..6, 50% at n = 13; games 31% | measured (10,000 deals/n, 4,000 games, CRN) |
| PI surplus is information-borne: same paranoid solver on true hands reproduces the PI curve vs TA+ | measured (omni 4,000 deals n ≤ 6, 2,000 n = 7, 1,000 n = 8) |
| stronger hidden-info spoilers (PIMC-paranoid K=16; MC with exact hero model) do not beat the heuristic | measured (2,000 / 1,000 deals per n ≤ 7 / ≤ 5) |
| punishment costs the spoilers ~75–80% of the damage they inflict; hero P(win) .242 → .167 (s3), .150 (s2) | measured |
| H3.1 sandwich S(π,σ_H) ≥ β(π), β(π) ≤ V_par | proved |
| H3.2 hero-led tricks information-free (per trick) | proved |
| PI paranoid punishment not attainable with hidden info (strict) | counterexample, solver-verified |
| deal-free lower bound on hidden surplus from hook/conservation | not found; conservation does not bind spoilers (argued) |
| NT trick count ≈ first-order lead chain, n = 5 | verified to the noise floor (cross-fit match gap 98–100% closed) |
| n = 8 | 81–91% of the gap closed (LCX) |
| n = 13 | open: best parsimonious chain closes ≤ 55% of xTV gap, ≤ 66% of match gap; needs per-suit control |

## Reproduce (round 3)
```
cd experiments/surplus_hidden
python sim.py rounds 1,2,3,4,5,6,7,8,9,10,11,12,13 10000 main   # ~7 min
python sim.py games 4000 main                                    # ~5 min
./run_r3.sh   # omni / PIMC spoilers + m10_sim NT (~65 min);  ./run_r3b.sh  # m10 analyze, trump m10_sim, MC spoiler (~25 min)
python analyze.py main && python analyze_strong.py
python ../../math/m10_analyze.py && python ../../math/m10_analyze.py --cfg 5:S,8:S --tag _S
pytest -q -p no:cacheprovider tests/test_round3_math_*.py
```

## M11 Exact welfare accounting and the paper evidence audit

**Status: proved** for four players, hit base H = 10 and zero score on a miss.
The numerical reanalysis below is **verified numerically**, using saved paired H1
outcomes rather than newly simulated games. Source and reproduction:
`math/m11_welfare.py`, `experiments/round7_paper/results/welfare.json`.

**Proposition M11.1 (welfare identity). Proved.** Let A be the set of players
whose actual tricks equal their bid, M = 4 − |A| the number who miss, and
U = sum_{i not in A} t_i the number of tricks taken by players who miss.
Then table score is

    W = sum_i (10+b_i) 1{b_i=t_i} = 40+n − 10M − U.

Proof: successful bidders contribute 10 each and b_i = t_i; their total tricks
are n − U by conservation. Consequently total score varies with both the number
of exact hits and which players take the tricks, even though total tricks is n.
With general nonnegative hit base H, the identity is W = H(4−M)+n−U.

**Proposition M11.2 (sharp hook ceiling). Proved.** The hook implies M ≥ 1:
if everybody hit, sum b_i = sum t_i = n, forbidden. Thus

    0 ≤ W ≤ 30+n,       (30+n)−W = 10(M−1)+U.

Both endpoints are attainable on legal histories for every n in 1..13, under
spades trump or NT. Deal each seat n cards of its own distinct suit; seat 0
holds spades, leads first (dealer 3), and wins every trick. Bids (n,0,0,1)
are hook-legal and attain 30+n. On the same deal and play, bids (n−1,1,1,1)
are hook-legal and make everybody miss, giving zero. The dealer's last bid is
legal in both cases. Without the hook, bids (n,0,0,0) attain 40+n. These are
global bounds and witnesses, not assertions that the bound is attainable for
every fixed deal and bid vector. The ceiling difference of ten is not the
measured dealer hook tax. Summing the round bound over the standard ladder
gives a table-score ceiling of 931 with the hook, versus 1181 without it.
These are exact bounds, not empirical estimates.

**Fixed-bid witness. Verified by the engine.** With NT, n=2, dealer 3,
hands (AS 2H, KH 3C, 4D 5D, 6D 7D), bids (1,0,0,0), and AS led,
seat 1 can discard KH or 3C. All other seats discard their lowest legal card.
Discarding KH leaves seat 0 winning both tricks and table score 30;
discarding 3C leaves KH winning the second trick and table score 31.
The deal and bids are unchanged. Hence even the fixed-bid play subgame need
not have a constant sum of scores. This concerns score utility; a unique
tournament winner or split-tie win shares define a different objective.

**Proposition M11.3 (paired loss decomposition). Proved.** For self-interested
and hostile policy outcomes on the same deal,

    W_self − W_hostile = 10(M_hostile−M_self) + U_hostile−U_self.

This is an accounting identity. Neither component is a separately identified
causal mechanism. Components and total can be negative on individual deals.
Sum them within a game before computing full-game intervals.

**Exhaustive and engine checks. Verified numerically.** All hook-legal
bid/trick count pairs were enumerated through n=6: counts by n=1..6 are
48, 710, 4,720, 20,650, 69,440 and 194,628. These count vectors include a
superset of realizable card histories; every identity and bound holds and
the maximum is 30+n in each case. Separate legal engine witnesses establish
sharpness at n=1,5,13, S and NT. These are deterministic checks, not samples.

**H1 welfare reanalysis. Verified numerically.** Three spoilers, same H1
TA+ hero and fitted bidder, with all components paired by original seed:

| sample | independent units | hero loss | mean loss per spoiler | table loss | 10 times additional misses | additional tricks held by missers |
|---|---|---|---|---|---|---|
| n=5, all trump states | 10,000 deals | 1.669 ± 0.126 | 1.283 ± 0.071 | 5.517 ± 0.237 | 5.201 ± 0.210 | 0.316 ± 0.035 |
| full ladder | 4,000 games | 51.661 ± 1.105 | 40.015 ± 0.605 | 171.708 ± 1.985 | 154.425 ± 1.628 | 17.282 ± 0.440 |

Units are points per round or per game respectively; ± denotes 1.96 SE over
the independent units. Table loss equals hero loss plus three times mean
spoiler loss exactly per paired unit. Spoilers reduce their own expected score
as well as the hero's. This supports the general-sum score interpretation but
does not measure equilibrium inefficiency or an optimal coalition value.

**Standardized information comparison. Verified numerically, descriptive.**
At n=5, retain H1 deals with d mod 5 in {0,4}, giving equal S/NT representation.
Their surplus share is 0.22984 ± 0.02751 (4,000 deals). Scheme R in phase2,
averaging seats within each deal, gives 0.81095 ± 0.01465 (1,000 deals, equal
S/NT). The ratio of shares is 0.28342 ± 0.03431. Ratio intervals use paired
numerator/denominator covariance; the ratio across experiments combines
independent influence variances. This standardizes trump composition but
still compares different hero/bidding regimes. It is not a randomized causal
estimate of information alone. The original H1 all-trump estimates remain
valid for their own sampling population.

**Victory metric correction. Verified numerically.** The legacy H1 analyzer
uses split-tie wins saved in the `wins` field. Recomputing from totals with ties
as wins gives baseline P(win)=0.24600 ± 0.01335 and s3 P(win)=0.16975 ± 0.01164;
paired change −0.07625 ± 0.01697, all with 4,000 games. Scores and score-surplus
results are unaffected. Historical files are preserved; use the M11 victory
values in new reporting.

Reproduce from `engine/`:

```bash
python math/m11_welfare.py
python -m pytest -q -p no:cacheprovider tests/test_round7_welfare.py
```

The script validates all included H1 scores, trick conservation and hook
legality, saves paired components by deal/game, and records source hashes.
Paper outline and source map: `paper/OUTLINE.md`, `paper/EVIDENCE.md`.

## M12 Hidden team support certificate in the round 8 toy

**Support certificate lemma. Proved.** Fix the hero's deterministic policy and
all bidding policies. For each complete deal d, let m(d) be the minimum hero
score attainable by fully informed spoilers. Choose a nonempty support S(I)
of legal actions at each spoiler's private information set I. This specifies
a feasible hidden-information policy, for example uniform choice within S(I).
Let M_S(d) be the maximum hero score over paths restricted to those supports,
with the hero and bids still fixed. If M_S(d)=m(d) for every deal, every path
of the hidden policy has hero score m(d). Thus it attains E[m(d)], a lower
bound for every hidden coalition, and its remaining joint best-response gain
is exactly zero. The argument does not use multiplayer CFR convergence or
equate unilateral NashConv with joint team optimality.

**Finite toy application. Verified exhaustively by an integer certificate.**
Four players, two suits with four ranks, two cards each; hero seat zero uses
TA-toy, all myopic bids are frozen, and spoilers optimize play only. The
complete chance space has 2,520 equally likely ordered deals per regime.
After CFR training, supports are taken from the saved current profile above
1e-12 and made uniform. The verifier independently reconstructs the fixed TA
profile, then evaluates both bounds using integer terminal scores and min/max
dynamic programming. Both bounds coincide on every deal:

| regime | exact minimum expected hero score | deals with support excess | joint gap |
|---|---|---|---|
| trump | 1257/140 | 0 of 2520 | 0 exactly |
| no trump | 946/105 | 0 of 2520 | 0 exactly |

These are exact finite-population values, so sampling CIs do not apply. The
certified strategy is the saved support-rounded policy, not a claim that every
floating-point CFR average is optimal. The toy exhibits no information loss
for this fixed-hero, fixed-bid intervention. It does not bound the 52-card
hidden/PI surplus ratio or resolve optimal punishment with unrestricted bids.

Source: `experiments/round8/T5/certify.py`, `results/certificate.json`, and
`results/certified_trump_{0,-1}.npz` under that task folder. Claude validated
the certificate logic and scope in `experiments/round8/REVIEW.md`, T5.
Reproduce: `python experiments/round8/T5/certify.py` from engine (night's hard
compute cutoff applies). The saved current and average profiles and exact BR
diagnostics remain in the per-regime JSON/NPZ checkpoints.

## M13 First-decision private-information relaxation (round 9)

**Status: proved and externally validated (Claude, 2026-11-16).** Task D3 was
separately self-validated (Codex) in round9/REVIEW.md; the proof and the exact
enumeration were then independently rechecked and reproduced by Claude. See
../../CLAUDE_AUDIT_2026-11-16.md. Claude checked only the feasible uppers' ordering against J1.
Round11/E5 subsequently confirmed both saved uppers through a fresh forward
replay, separately self-validated (Codex), not an external replay verdict.

Fix the hero policy and all bids. For each uniformly likely deal d, collapse
hero actions and forced spoiler moves until the first free spoiler choice or a
terminal. No earlier free spoiler action affects this root's probability.
Let Q_d(a) be the PI-minimum hero score after root action a. Group roots by the
acting spoiler's private information set I (seat, original own hand, complete
public history); D_I is the set of deals rooted in I. Legal actions are the
same within I. Define

    J1 = [sum of terminal-root scores + sum_I min_a sum_{d in D_I} Q_d(a)] / N.

For any feasible hidden team, the root action distribution sigma_I is common
to all roots in I. Grant the team omniscience after this choice. Its expected
hero score is at least the displayed expression before minimization over
sigma_I. That expression is linear in each sigma_I, so minimizing selects a
pure action separately for each I. Any coupling with decisions that are
non-root in other deals is dropped, which only relaxes the problem. Therefore

    PI minimum <= J1 <= optimal hidden-team hero score <= any feasible score.

The first inequality also follows from min(sum) >= sum(min). This is a bound,
not an equilibrium or a solution of the decentralized team problem. It covers
the fixed-bid, fixed-hero play intervention only; no private-card pooling or
mediator that observes all cards is allowed.

**Exact enumeration externally validated (Claude,2026-11-16):** four players, three cards,
3 suits x 4 ranks, no undealt stock, all 369,600 ordered deals per regime.
Trump: PI = 367921/61600, J1 = 446669/73920, difference = 25819/369600 > 0.
NT: PI = 106419/15400, J1 = 213259/30800, difference = 421/30800 > 0.
These are integer full-population sums, so sampling CIs do not apply. The
certified gaps are approximately 0.070 (trump) and 0.014 (NT) hero points. They
certify a strict information cost and nothing about its magnitude: these lower
bounds neither establish a material cost nor estimate the actual hidden-team
optimum minus PI. No CFR convergence or 52-card information-cost claim follows.

Sources: `experiments/round9/D3/ROOT_BOUND.md`, `root_bound.py`,
`results/summary_final.json`. Saved feasible profiles were replayed against the
complete cached trees; the raw PI/achieved interval remains separate from the
tighter J1/achieved interval. Small-tree regression compares full, reduced and
compact representations, baseline and private-BR policies/values, and checks
PI <= J1 <= baseline. Claude's verdict belongs in round9/REVIEW.md.

Separate reviewer scripts `round9/reviews/audit_d3.py` and
`exact_policy_d3.py` reproduce all headline values from saved tree/profile
arrays without importing the experiment evaluator, BR solver or root-bound
routine. The latter integer-replays the already evaluated one-hot checkpoint
50 policies: feasible upper scores are exactly 574247/92400 (trump) and
109709/15400 (NT), over all 369,600 deals/regime, no sampling CI. Thus exact
optimization bounds are [446669/73920,574247/92400] and
[213259/30800,109709/15400], respectively. Neither interval solves the hidden
team optimum. The review reuses source-fingerprinted derived tree caches;
Claude subsequently independently reconstructed the complete lower-bound populations; round11/E5 separately confirms the feasible uppers by fresh forward replay, with unchanged fixed hero/bidder definitions and no optimizer.


Round11/E5 fresh forward replay gives integer hero-score sums2296988(trump)
and2633016(NT), each over all369600ordered deals, hence exactly574247/92400
and109709/15400. Portable private-key policies and per-deal score vectors are
saved under experiments/round11/E5/results/. Independent integer-vector,
checkpoint/source and map checks pass. This is same-investigator
self-validation, not Claude replaying the uppers, extra optimization, a
solved hidden optimum or a sampling confidence interval.

## M14 Pure-team sufficiency against fixed non-team policies

**Status: proved and externally validated (Claude, 2026-11-16).** Elementary
clarification, not a novelty claim; exploratory derivation documented in
round10/R3/STATUS.txt. The derandomization argument, the matching-bit
counterexample and the four-world arithmetic were rechecked independently; see
../../CLAUDE_AUDIT_2026-11-16.md.

In a finite full-history private-information game, fix the hero and every
non-team policy. Team random tapes (possibly shared) are independent of the
deal/chance and non-team tapes. Fixing the team tapes induces a feasible pure
policy tuple on the members' own information histories. There are finitely
many distinct pure tuples. Hence any randomized team's expected hero score is

    J = sum_p lambda(p) J(p) >= min_p J(p).

Pure tuples are admissible, so the global minimum over randomized private team
policies equals an attained pure-team minimum. Legal public signaling is still
allowed; no pooled private cards or deal-correlated random device is added.

This is a fixed-policy team decision problem, not a strategic-adversary
team-maxmin equilibrium theorem. It does not certify CFR convergence, efficient
global optimization, a restricted heuristic family, or arbitrary profile
rounding. Full-history/perfect-recall information and independence from chance
are material assumptions. Every proposed achieved policy still needs replay.
Internal team coins affect non-team decisions only via legal observed histories;
no extra hero-observed random device is added outside the modeled game. A
matching-bit strategic-response counterexample in R3/PROOF.md illustrates why
the fixed-hero premise cannot be dropped.

Proof/scope: experiments/round10/R3/PROOF.md. A four-world exact arithmetic
sanity check is in R3/check.ps1 (16 pure plans,81 half-grid profiles); it is not
a new card-game experiment or a substitute for the proof. No sampling CI or
stochastic seed applies. External proof audit remains required.
