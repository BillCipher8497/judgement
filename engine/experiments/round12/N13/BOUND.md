# Finite conditional-product lower-bound lemma

Status: derived by Codex; **externally validated by Claude, 2026-11-20**
(re-derived by hand; see CLAUDE_AUDIT_2026-11-20.md). This is a standard
finite entropy variational argument, not a novelty claim.

Let F be a finite allocation set, f(w) its vector of card-holder indicators,
P uniform on nonempty R subset F, and mu=E_P f. For any distribution q on F
with E_q f=mu, and Q_theta(w)=exp(theta.f(w))/Z(theta),

    KL(P || Q_theta) = log Z(theta) - theta.mu - H(P)
                    >= H(q) - H(P).

The inequality follows from KL(q||Q_theta)>=0, expanded as
log Z(theta)>=theta.E_q f+H(q). No optimization of q is necessary. Any
exact-feasible q with a certified positive entropy gain supplies a lower
bound for EVERY finite theta; taking liminf preserves it along limits of
finite-potential distributions. If P gives mass where a limit Q has zero
mass, KL is infinite and the inequality still holds. No equality/attainment
or fitted-feature/optimizer convergence is established here.

The code builds exact rational q from a separately checked LP proposal, then
mixes with P. Floating LP feasibility is never accepted as exact feasibility.
The log series uses z in[0,1/3], all positive terms. Its remainder after N
terms is between0 and2*z^(2N+1)/((2N+1)*(1-z^2)). Powers of2 scale arbitrary
positive rationals to[1,2]; multiplying a negative scaling exponent reverses
which log2 endpoint contributes. Outward dyadic rounding preserves enclosure.
Thus a positive rational lower entropy endpoint does not rely on a floating
logarithm or a numerical optimizer. Source/arithmetic checks are separate. The Claude audit re-derived the lemma
independently and reproduced every per-key certified lower bound as
H(q) - log|R| from the stored rational mixtures, including the reach-weighted
largeNT value 0.0660799372 nats.

Audit note: because P is uniform on R, H(P) = log|R| is the maximum entropy on
R, so a useful witness must place mass outside R. The bound is in fact tight —
the max-entropy moment-matching distribution lies in the exponential family, so
inf_theta KL(P||Q_theta) = H(q*) - H(P) by the standard I-projection identity.
This note states only the inequality, which is the conservative direction;
M19 exhibits the matching upper limit in one context.
