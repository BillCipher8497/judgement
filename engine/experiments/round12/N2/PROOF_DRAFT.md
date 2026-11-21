# N2 mathematical notes — pending external audit

Status: **pending external audit**. Conditional finite-model statements for
Claude to check; no N2data has been generated. These are elementary finite
Bayes/counting facts, not claimed novel theorems or results about game score.
No headline identifier is assigned here; the head of research can assign one.

## Complete reaching histories

Let W be the finite set of equally likely ordered chance deals, and fix a
known deterministic private/public policy profile, including bidding and all
play tie rules. Record every pre-play decision along its full trajectory.
An infoset I includes actor, original own hand and the complete public action
history, so a given deal visits I at most once. Define R(I) as the chance deals
whose trajectory reaches I. For every w in R(I),

    P(w | I reached) = 1/|R(I)|.

Indeed, the reach indicator is zero or one and the chance weights are equal.
Recording all decisions weights every occurrence equally, and there is at
most one occurrence of this particular I per world. No hidden first-free-
team-decision condition is added. Root-relaxation groups cannot be substituted
unless their selection event is itself determined by I. Stochastic policies,
nonuniform chance or unmodeled stopping rules require likelihood weights,
not this simple support-count formula.

## A defined proportional product joint equals uniform

Let F(I) be the structurally feasible deals given own cards, observed played
cards/ownership, capacities and follow-suit/void constraints. Assume F is
nonempty and contains R. Let C be unseen cards, h_j the known remaining
capacity of holder j, and A_c the allowed holders for card c. Define raw
capacity probabilities p(c,j)=h_j*1[j in A_c]/z_c, with
z_c=sum_{j in A_c} h_j. Positive feasibility gives z_c>0.

Define a **joint law** by multiplying these card factors and renormalizing
over F. Every w in F assigns exactly h_j cards to j, hence

    product_{c in C} p(c, holder_w(c))
      = product_j h_j**h_j / product_c z_c,

constant in w (0**0 means1 for zero-capacity holders). Thus this specifically
defined constrained product joint is exactly uniform on F. This does not
assert that p(c,j) is the marginal of that joint. Actual card marginals can
change after conditioning on the capacities. A raw-marginal loss and a
normalized-joint KL must not be interchanged.

## Matched deterministic bids and soft likelihood

The proposed N2data bidder is the same deterministic myopic argmax used by the
BI-style likelihood. Let m_j be the public number of legal bids when opponent
j bid, and let e_j(w) indicate a matching argmax on its reconstructed original
hand and public bid prefix. The declared noise is epsilon=1/50, giving

    L_j(w) = (49*m_j*e_j(w)+1)/(50*m_j).

The denominator is constant across F. Define integer weight
B(w)=product_j(49*m_j*e_j(w)+1), B_max=product_j(49*m_j+1), and
Z=sum_{w in F} B(w). Let F_bid contain worlds where every opponent bid matches.
The fixed matched-profile premise implies R is a subset of F_bid. Each
reachable world has weight B_max; therefore

    |R| <= |F_bid| <= |F|,
    |F_bid|*B_max <= Z <= |F|*B_max.

For P uniform on R, U_bid uniform on F_bid, U_F uniform on F, and q_B=B/Z,

    KL(P || U_bid) = log(|F_bid|/|R|),
    KL(P || q_B) = log(Z/(|R|*B_max)),
    KL(P || U_F) = log(|F|/|R|),

so

    0 <= KL(P||U_bid) <= KL(P||q_B) <= KL(P||U_F).

The prop-induced joint above equals U_F. Preserve the exact integer ratios
before taking floating logarithms. The ordering needs the matched deterministic
bid premise; it need not hold with unknown/stochastic/misspecified bidders.
The soft-BI gap includes both epsilon softening and missing fixed-play-action
likelihood. These are defined posterior-model comparisons, not an additive
information/label/belief/strategy-fusion game-cost theorem. A net-induced
joint is not ordered by this argument and may lose to every baseline.

## Finite allocation normalizers

Order holders0..2 in these all-dealt toys. For remaining unseen-card subset M,
capacity h_j and positive or nonnegative per-hand factor b_j(H), the recursion

    Z(j,M)=sum_{H subset M, |H|=h_j, allowed} b_j(H)*Z(j+1,M\\H)
    Z(3,empty)=1; Z(3,nonempty)=0

enumerates each feasible allocation once. Choosing H for the next holder is
a disjoint partition of the allocation set, establishing the recurrence by
finite induction. Unit factors give integer feasible counts; bid indicators
give hard-bid counts; integer B factors give Z exactly. Card-product neural
factors use floating arithmetic, with direct-enumeration reference checks,
raw unclipped KL saved and a numerical tolerance; they are not rational
optimization certificates.

## Boundaries

These nats-valued posterior statements do not establish an expected-score
information floor, a full private-team optimum, a policy-value improvement,
a mixing guarantee, or a neural generalization/sample-complexity theorem.
The 35-feature summaries plus the structural support F define the effective
net input/decoder; representation claims cannot ignore F. Hand/key splits
share latent deals across actors; no independent-deal or future-hand guarantee.
The exact posterior always uses the full declared chance prior, never a
private train/test marker. All proposed empirical checks and all proof
validation remain pending; no source/experiment approval is asserted here.
