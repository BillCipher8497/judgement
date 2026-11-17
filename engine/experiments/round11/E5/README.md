# Fresh forward replay of saved feasible policies

This evaluates the existing saved upper policies; it does not optimize,
select a new checkpoint, rebuild a value recurrence or change the hero/bids.
The full chance populations are all369600ordered deals in each regime.
Expected exact hero means are574247/92400(trump) and109709/15400(NT).
These are feasible optimization upper bounds, not sampling intervals or
solutions of the hidden-team optimum.

```
python engine/experiments/round11/E5/replay.py
```

The default command resumes the recorded run; a fully completed checkpoint
checks the saved vector rather than doing a second fresh traversal. For a
new traversal, set `JUDGEMENT_REPLAY_OUT` to a new empty directory and declare
a fresh authorized deadline. Published portable policy maps are read as
inputs while new score vectors/checkpoints go to the empty output directory.

Run only with a recorded unexpired hard stop and the sole Judgement worker
slot free. The default cutoff is23:00ET on2026-11-16; a later authorized
window may set `JUDGEMENT_REPLAY_DEADLINE` in Unix seconds after appending its
bound and checkpoint cadence to STATUS. Full1000-deal score-sum checkpoints
resume without altering the saved policy or enumerated population. An
interrupted final line is preserved separately before retaining complete
checkpoint records.

The lexicographic deal order also indexes a one-byte-per-deal hero-score
vector. Each1000-deal block is appended before its sum checkpoint. Any
uncheckpointed tail is preserved separately on resume; the retained vector
length and sum must match the last complete checkpoint. Final acceptance
checks both the exact sum and the complete score vector's length/hash.

The first run needs D3's source-fingerprinted derived cache solely to export
private keys and the action in each saved pure row. The export is a portable
`private_policy_<trump>.bin.gz`, with its own SHA256 and metadata. It never
reads the cache's edges, roots, terminal payoffs or optimizer state. With
portable maps present, the cache is unnecessary for subsequent forward
evaluation. Line-ending-normalized builder fingerprints permit equivalent
Git checkouts; all substantive source changes still fail verification. Raw
source hashes are retained alongside the normalized checkpoint hashes.

Historical checkpoints record Windows-style relative source paths. Resume
them on the same platform; cross-platform fresh reproduction should use the
new empty output directory described above. The separate verify.py audit
normalizes source paths while checking the archived score vectors/maps.

Forward evaluation constructs deals, follow-suit legality, trick winners,
remaining-hit impossibility and scores independently in this script. The
unchanged myopic bidder and toy TA definitions are imported as the fixed
policies defining the experiment. Using those policies does not invoke the
original optimizer or tree evaluator. Each nonforced spoiler lookup contains
only its own initial hand and the public action sequence. Once an exact hit
is mathematically impossible, scoring can terminate at zero without any
further policy lookup. Every other requested action is checked legal.

Rule tests independently compare every ordered distinct-card four-card trick,
all three-card follow-suit cases, and the complete deal enumeration count.
The ultimate acceptance check is the exact integer score sum over both
complete populations. This is a fresh evaluation path by Codex, the same
investigator; it must not be described as Claude independently replaying the
uppers or solving the remaining optimization interval.
