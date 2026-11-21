# M17 — A PI-optimal action need not be private-optimal

Status: **verified numerically by separate exact Codex paths; pending external
audit**. This is a finite computational witness of a known strategy-fusion
phenomenon, not a novelty claim or universal PIMC failure theorem.

Game:4players,S3,R4,n3,trump0,dealer3, fixed myopic bids/private TA profile
before the current decision. Cards c have suit floor(c/4), rank c mod4.
Hero seat1 has original hand[4,6,10]. Observed bids[0,0,2,0], followed by
seat0's lead1. Hero has all3cards, is void in the led trump suit and can
legally discard4,6or10. The ordinary profile-conditioned posterior contains
exactly14equiprobable original deals, not M1root-selected worlds. All three
other players retain the same private policies in both continuations.

There are560structural original-hand assignments: the known lead belongs to
seat0, leaving8unknown cards for capacities2/3/3; C(8,2)C(6,3)=560. Independent
myopic-bid and TAlead replay leaves14. See the complete world list and future
own/public infoset matrices in
results/certificates/S3R4n3_t0_0507870.json.

| Current card | PI-future-hero score sum | Exact private-BR score sum | PI mean | private mean |
|---|---:|---:|---:|---:|
|4|140|100|10|50/7|
|6|140|140|10|10|
|10|140|140|10|10|

Both methods share ONE current action across worlds. PI-future permits
world-specific future hero choices; privateBR requires a common card at every
future own/public history. The certificate partitions each current-action
branch by these future histories, lists each world's terminal score for each
available future card, and compares sum(row maxima) with max(column sums).
The latter is feasible and optimal here because at most one further nonforced
hero choice remains (the last own card is forced). Fixed opponents are replayed
with independent legal/winner transitions. All posterior worlds/reach choices
were independently enumerated; integer sums match the recursive privateBR and
PI solver. A third algebra-only checker matches all world partitions, terminal
constants, row maxima, column sums and tie ranges for all7positive-gap cases.

Consequently, the PI-optimal current set{4,6,10} strictly contains the private
optimal set{6,10}. The predeclared lower-card PIMC tie rule chooses4and loses
40/14=20/7hero points conditionally. The best PI tie loses0; no claim that every
PIMC decoder loses, or that PI ranks a unique inferior action. The maximum
current PI and private values both equal10in this case. The error is not an
intrinsic lower bound on the best current private policy's value.

For THIS unilateral maximizing teacher, every per-world PI action value is at
most10(hero bid0). Each action sums to140over14worlds, so every current action
is PI-optimal in every posterior world. Hard lowest-card labels select4;
uniform-optimal, membership and PI-value targets provide no private sequential
preference among the three actions at this context. This is not a refutation
of M15: that theorem evaluates its SAME PI-continuation root objective, while
the private sequential objective here changes continuation constraints. It
does not quantify why the separate M9/N3learners failed or rule out learning
useful preferences from other information/contexts.

No sampling CI applies to this complete finite posterior. It establishes
neither prevalence, material52-card cost, joint team optimality nor a new
strategy-fusion theory. The broader N11fixed256-key/regime diagnostics and all
nulls are in RESULTS.md; no selector, quota, decoder or source was retuned.
External audit must check reach conditioning, fixed-policy scope, future
infoset grouping and tie sensitivity before this claim is released.
