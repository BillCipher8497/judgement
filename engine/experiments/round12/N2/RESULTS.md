> Current interpretation, 2026-11-19: [independent review corrections](REVIEW_CORRECTIONS_20261119.md) govern. The fixed-population negative reproduces; small-trump hand sensitivity, oracle-family limits and unresolved cause must accompany it.

# N2 — exact policy-conditioned belief benchmark

Status: self-checked (Codex), **pending external implementation/results audit**. New result packet remains local. M16 conditional deterministic/matched-bid proof was externally checked by Claude on 2026-11-17; that proof verdict does not validate this execution.

## Held-out primary comparison

Positive net-minus-BI means the learned factorized joint is worse. Each fixed fit is evaluated over the entire reached population in its held-out own-hand split. Intervals below are t95% over five paired optimization seeds only, not sampling CIs over chance deals/hands or a second independent test population. Small conditions are descriptive controls.

|Condition|Reached test hands|Private test keys|Exact decision-world weight|Uniform KL|Soft-BI KL|Hard-bid KL|Net − soft-BI, n=5 fits, t95% CI|
|---|---:|---:|---:|---:|---:|---:|---|
|S2R4n2_t0|6|2172|4320|0.79240963|0.21717673|0.21184059|0.09737651 [0.08682473, 0.10792828]|
|S2R4n2_t-1|6|2563|4320|0.88229011|0.16503717|0.15846909|0.25012586 [0.23882551, 0.26142621]|
|S3R4n3_t0|44|427094|887040|1.43467836|0.28659626|0.27721491|0.50043539 [0.49191439, 0.50895638]|
|S3R4n3_t-1|44|431510|887040|1.43827423|0.28550155|0.27669907|0.49341417 [0.48481549, 0.50201285]|

All baseline entries are deterministic complete-population mean joint KL values in nats, computed with floating logarithms of exact support counts/integer bid weights. They do not receive sampling intervals. The net has five individual models rather than a post-hoc pooled primary ensemble.

## All conditions and nulls retained

|Condition|Split|Net − soft-BI, n=5 fits, t95% CI|Net share of uniform KL removed, n=5 fits, t95% CI|
|---|---|---|---|
|S2R4n2_t0|all|0.15460574 [0.14626689, 0.16294459]|0.61715200 [0.60656131, 0.62774269]|
|S2R4n2_t0|train|0.18868709 [0.18003377, 0.19734040]|0.62575208 [0.61496679, 0.63653737]|
|S2R4n2_t0|validation|0.12095139 [0.11481666, 0.12708612]|0.60742860 [0.59916606, 0.61569114]|
|S2R4n2_t0|test|0.09737651 [0.08682473, 0.10792828]|0.60304213 [0.58972607, 0.61635819]|
|S2R4n2_t-1|all|0.28509799 [0.27524635, 0.29494964]|0.51848303 [0.50746352, 0.52950254]|
|S2R4n2_t-1|train|0.30116128 [0.29191058, 0.31041198]|0.51506482 [0.50490241, 0.52522723]|
|S2R4n2_t-1|validation|0.27723470 [0.26612265, 0.28834676]|0.51688606 [0.50400050, 0.52977163]|
|S2R4n2_t-1|test|0.25012586 [0.23882551, 0.26142621]|0.52944839 [0.51664042, 0.54225637]|
|S3R4n3_t0|all|0.49802000 [0.48867845, 0.50736155]|0.45198521 [0.44562830, 0.45834212]|
|S3R4n3_t0|train|0.49876805 [0.48904797, 0.50848814]|0.45444089 [0.44788986, 0.46099191]|
|S3R4n3_t0|validation|0.49336045 [0.48412051, 0.50260038]|0.44505856 [0.43873689, 0.45138024]|
|S3R4n3_t0|test|0.50043539 [0.49191439, 0.50895638]|0.45142294 [0.44548363, 0.45736224]|
|S3R4n3_t-1|all|0.52080168 [0.51163286, 0.52997051]|0.45961259 [0.45336962, 0.46585555]|
|S3R4n3_t-1|train|0.52751663 [0.51848671, 0.53654654]|0.45935196 [0.45324384, 0.46546009]|
|S3R4n3_t-1|validation|0.52804437 [0.51784969, 0.53823905]|0.46154882 [0.45461375, 0.46848388]|
|S3R4n3_t-1|test|0.49341417 [0.48481549, 0.50201285]|0.45843727 [0.45245880, 0.46441574]|

All/train/validation rows are descriptive, not additional replications. Shares may be negative, are not additive, and are not a game-score fraction. Raw capacity-factor card log loss is separately saved and must not be confused with constrained-joint marginal log loss. Full per-seed KL, every raw key/chance support and model-selection candidate are retained.

## Selection and provenance

|Condition|Selected epochs (five fits)|Selected blends (five fits)|
|---|---|---|
|S2R4n2_t0|20, 20, 20, 20, 20|1.0, 1.0, 1.0, 1.0, 1.0|
|S2R4n2_t-1|20, 20, 20, 20, 20|1.0, 1.0, 1.0, 1.0, 1.0|
|S3R4n3_t0|20, 20, 20, 20, 20|1.0, 1.0, 1.0, 1.0, 1.0|
|S3R4n3_t-1|20, 20, 20, 20, 20|1.0, 1.0, 1.0, 1.0, 1.0|

At most30000training keys and4000validation keys per condition, selected by fixed SHA order within disjoint own-hand partitions; test uses every reached key. All/train summaries include uncapped keys as descriptive evaluations, not claims of full-key training. Fixed20epochs and five shared fit seeds; selection among checkpoints/blends uses validation card-marginal loss, not held-out joint KL. Thus these are selected belief pipelines, not an isolated architecture or joint-KL training comparison. Fixed SHA own-hand splits share latent deals across actor observations and have few distinct test hands. Seed intervals exclude hand-split uncertainty.

Original scientific source commit: `1c0c7192f65464b79afa7a525e121f76543218c1`; fingerprint `7c7ba88609951b1b6a4be672984bf7adc3dc1bc7ababaeff7afa5d9ab10173b7`. The original three-hour forecast **failed before model/outcome generation**; original failed gate and scope-failure receipt remain unchanged. Resource-only continuation source commit `ba6505d43fa41e1c6a2e00d3d6ed5abe32c405dc` predeclared at most two further three-hour windows, preserving all scientific sources, counts, seeds and original pilot. Additional windows actually used: 2. This is an explicit resource-budget amendment, not a successful original gate.

## Separate checks and limits

The verifier checks every saved raw world/label/grouping, deterministic matched-bid support premise, every model selection and all reported CIs. A fresh manual rules/posterior/normalizer path covers all small infosets and64fixed large contexts per condition; it is not a second complete enumeration of every large posterior. Existing policy functions are reused as the profile definition. Resource-source and scientific-source manifests are checked separately.

The benchmark concerns an all-dealt tiny game, fixed private TA play and matched deterministic bids, with every pre-play observation including forced moves. It is neither the M1 root relaxation nor a 52-card/game-score benchmark. Summary features and a normalized product family may lose history/dependence information. Finite epoch/blend grids may select boundaries and do not bracket an optimum. Do not attribute errors solely to optimization or neural capacity. N1 uses another game/profile, stock and a different predictive metric; comparisons between N1 and N2 are descriptive, not a controlled scaling effect.

Windows I/O extension: first continuation failed replacing NT epoch2checkpoint; epoch1 and original traceback/tmp retained. Source-only adapter1d4eafb8e2652175288725e825484fe1ebbc8402 reads only resume checkpoints into memory-backed NPZ objects (identical arrays/API) and retries result-file PermissionError for at most5seconds. This prevents the identified resume file-handle self-lock, but first-error cause is unestablished. Second/final preallocated continuation uses this explicit runtime adapter; no frozen science/scheduler source changed. Include its manifest and separate provenance receipt in any future release.

No publication release is implied by this report.


## Claude external audit addendum (2026-11-18)
See root `CLAUDE_AUDIT_2026-11-18.md`, section N2, for the verdict, the post-hoc diagnostics and the requested changes. That section governs the interpretation of this packet: where it conflicts with the text above, the audit wording applies.
