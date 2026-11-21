# N2 review addendum — Codex independent cross-check

2026-11-18T03:56:18.731165+00:00

Status: self-checked(Codex), pending external implementation/results review. The actual Claude M16 verdict covers the conditional proof, not this packet.

The initial verification attempt was stopped because frozen verify.py repeatedly decompressed complete NPZ members inside its record loop. Its logs and pre-cache progress/manifests remain saved. Verification restarted with n2_cached_audit.py, a separately tested read-only eager raw-chunk cache. All arrays/dtypes/string keys/empty arrays/context API and source bytes matched; the frozen verifier/scientific sources/data were unchanged. Runtime adapter SHA: 7c2598822466bd59133d080139b5b528d43b304bbb641625bf5e2b2d2901a6ab.

All original raw/grouping/model/CI/direct-posterior tests passed in all four conditions. Extra n2_policy_actions.py independently re-evaluated the fixed policy function at every private key and matched every raw action (the heuristic definition is reused, not independently reimplemented). Counts:

|Condition|Private keys|Recomputed nonforced choices|Raw decision-world actions matched|
|---|---:|---:|---:|
|S2R4n2_t0|11464|1587|20160|
|S2R4n2_t-1|12613|1695|20160|
|S3R4n3_t0|2148865|492602|4435200|
|S3R4n3_t-1|2228532|523351|4435200|

Fresh direct normalizer/reach enumeration remains all small keys and64fixed large contexts per condition; it is a separate route from the complete raw/profile/CSR audit. No second independently coded TA heuristic or independent entire-population large normalizer is claimed.

NT seed0 optimization was additionally regenerated: interrupted epoch2orphan and final epoch20parameters/Adam states matched bit for bit. It reuses the frozen gradient/optimizer definitions with vectorized batch gathering; no extra fit/model/sample or retuning. Windows-I/O adapter provenance also passed; one memory-backed checkpoint read, zero bounded replace retries in the final run.

All four held-out primary net-minus-BI joint-KL contrasts are positive (worse net). n5optimization-seed t95%CIs only, fixed reached test-hand populations; small controls have6hands, large44. Full values/limits are in RESULTS.md. New results remain uncommitted/local pending external review.
