# Optional sequential deployment stress — beforeM3 held-out outcomes

Research question: does a frozen one-decision privileged-teacher student remain
useful when every later team decision must also use private information? This
is a distinct evaluation axis, not a repair of theM3 primary, a new fit, or an
estimate of the irreducible full-team information cost.

Run only afterM4/M5 and before07:00ET, one worker. Four original exact games;
MLP hard_ce andset_bce only, same five2010Mfits/condition, frozen validation-only
selected models. At every free spoiler action use the unchanged241private
encoder and legal masked decoder; forced cards remain forced; hero remains
original TA and bids original myopic. No teacher, hidden cards or Qvalues are
used for policy decisions. No fine-tuning, action gates, best-model selection
or altered history representation after outcomes.

Primary descriptive contrast: setBCE−hardCE full-private-team mean hero cost,
all369600largeNTdeals perfit, paired five-seed t95%CI. Other conditions report
the same contrast descriptively, not selected favorable results. Complete
population means and per-deal vectors are exact, no deal-sampling CI. Training
and held-out hands mix across different teammates in this evaluation, so it is
not an independent held-out full-team generalization test. Root-held-outM3
and full-deploymentM9 results must be reported separately.

Use the audited read-only tree to map each private key to neural action and
evaluate the resulting pure profile. Save all action mappings and per-deal
scores incrementally permodel, integer sums, source/model/cache hashes.
Fresh manual forward play on128fixed evenly-spaced deals percondition/model
must agree, reconstructing hands/legalcards/winners and querying only private
keys for actions. If any policy beats the existing feasible upper, require a
fresh full-population forward replay of that policy before calling it a new
feasible upper; never infer optimality or external proof clearance.

Expected teacher-to-deployment degradation is a valid negative finding. Even
an improved achieved value is only a feasible upper against fixed policies,
not the global hidden optimum, an equilibrium, or a52-card result. Hard stop
06:45ET17Nov2026, checkpoint each model and every10000replayed deals, within
07:00ET final scientific stop. New theory remains pending external audit.

Reach-semantics audit added before anyM9 outcomes: count root private keys that
also occur at non-root nodes of the full tree. Whether another teammate earlier
had multiple legal cards need not be publicly known. A root-only override is
therefore not presumed executable merely because its action uses a private key.
The deployed network acts at every matching private key/history with no hidden
"first free team decision" indicator. Root/later overlap and conditional reach
are diagnostic exact counts, not sampling estimates or a new fitted family.
