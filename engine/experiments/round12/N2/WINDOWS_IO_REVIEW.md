# Windows checkpoint continuation — source-only self-review (Codex), external pending

2026-11-18T02:52Z, after the first resource continuation exited at NT seed0
epoch2 on WinError5 replacing checkpoint.tmp -> checkpoint.npz. The original
stderr and previous epoch1 checkpoint are retained. Three complete conditions,
all NT raw data/posteriors and the original failed resource gate stay intact.

The frozen resume function holds an np.load(checkpoint.npz) NpzFile across
later replacements. A new runtime adapter reads only that checkpoint into a
memory-backed NpzFile, preserving arrays and lazy NPZ API while closing the
disk handle. It also retries result-file PermissionError for at most5seconds.
This prevents the identified resume self-lock; it does not establish the cause
of the first error, which occurred before resume. No hidden numerical patch.

Scientific producers and resource continuation sources remain byte-unchanged.
The adapter has its own source-only pre-run commit/manifest and synthetic
equivalence/open-handle/finite-retry fixture. Window2 is the already allocated
second and final additional window, fresh absolute cap no longer than3h.
Original checkpoint.tmp is archived by the frozen atomic writer on resume;
completed conditions must hash-verify without rewrite. Result packet must carry
both original sources plus scheduling and Windows-I/O extensions for auditing.
External implementation/result review remains pending; no release/push.

Synthetic checks PASS 2026-11-18T02:52:48.875738+00:00: exact arrays/NPZ API preserved, memory NPZ open during real Windows disk replace, bounded injected retry. Epoch1 checkpoint finite/readable (Adam235); all original scientific/scheduling manifests still match. Adapter rawSHA e36aaf61a28c906205bd2b89a0e9bab3446f984f690e238ebe6aef1d29ccec86. Self-reviewed I/O-only extension, source-only local commit before second window. No source/scientific-result external validation asserted.
