# N9 — matched normalized-joint versus marginal training

**Externally validated by Claude, 2026-11-20** (independent recomputation over the raw arrays; see CLAUDE_AUDIT_2026-11-20.md at the repo root). Model weights and bulk per-block result trees remain local.

PRIMARY largeNT/summaryMLP/80:joint−marginal held jointKL -0.01704725 [-0.02449983, -0.00959466] nats; n5paired optimizer seeds,t95%df4. Negative favors joint. This is a KNOWN toy-hand benchmark/deterministic key subset, not new-hand replication or a generalization guarantee. Exact support sums use floating logs, not rational certificates.

All120trajectories/240selected20-or80recipes frozen before held evaluation. Three configurations with245input columns:summarylinear,summaryMLP,richMLP. Summaryzero-pads beyond unchanged35features; richadds strict own/public fullcontext/card/holder features. MLPnominal7905coefficients inboth; summaryhas1185input-active coefficients before accounting for normalization gauges. Linear246nominal/36input-active. No optimal NNclass/factor-family projection floor solved.

Marginal and joint losses share data,per-key1/Uscaling,reachweights,initialization,batches,Adam/L2and JOINTvalidation selection. New recipe omits oldN2mixture/alpha inBOTHarms. Short/longselectedcheckpoints share trainingprefix;80budget also adds selectioncandidates, not pure training-duration causality. Richinput changes which coefficients receive data signal, not a certified information-only mechanism. No unique cause of old N2failure inferred.

| condition | split | selectedkeys | reachedownhands | exactreachweight |
|---|---|---:|---:|---:|
| S2R4n2_t0 | 0(0train/1val/2test) | 2048 | 16 | 4666 |
| S2R4n2_t0 | 1(0train/1val/2test) | 512 | 6 | 1330 |
| S2R4n2_t0 | 2(0train/1val/2test) | 1024 | 6 | 2520 |
| S2R4n2_t-1 | 0(0train/1val/2test) | 2048 | 16 | 3920 |
| S2R4n2_t-1 | 1(0train/1val/2test) | 512 | 6 | 1006 |
| S2R4n2_t-1 | 2(0train/1val/2test) | 1024 | 6 | 2125 |
| S3R4n3_t0 | 0(0train/1val/2test) | 2048 | 132 | 5412 |
| S3R4n3_t0 | 1(0train/1val/2test) | 512 | 44 | 1427 |
| S3R4n3_t0 | 2(0train/1val/2test) | 1024 | 44 | 2840 |
| S3R4n3_t-1 | 0(0train/1val/2test) | 2048 | 132 | 5133 |
| S3R4n3_t-1 | 1(0train/1val/2test) | 512 | 44 | 1428 |
| S3R4n3_t-1 | 2(0train/1val/2test) | 1024 | 44 | 2617 |

CIs below are paired n5optimizer t95%df4 ONLY, not chance/hand/population intervals. Every per-fit mean covers the full selected exact reach support; no world-IID samplingCI. All4conditions are not independent replications. Allsecondary cells descriptive/unadjusted; no selected winner. Null/adverse cells kept.

## Joint minus marginal:held jointKL

| fixedcell | mean and95%CI(n5pairedseeds) |
|---|---|
| S2R4n2_t0/summary_linear/20 | -0.00432423 [-0.00443501, -0.00421345] |
| S2R4n2_t0/summary_linear/80 | -0.01370963 [-0.01381196, -0.01360730] |
| S2R4n2_t0/summary_mlp/20 | -0.02995507 [-0.03441827, -0.02549188] |
| S2R4n2_t0/summary_mlp/80 | -0.04703796 [-0.06158063, -0.03249529] |
| S2R4n2_t0/rich_mlp/20 | -0.04554750 [-0.05814699, -0.03294800] |
| S2R4n2_t0/rich_mlp/80 | -0.04166677 [-0.05148148, -0.03185205] |
| S2R4n2_t-1/summary_linear/20 | -0.01157681 [-0.01173422, -0.01141940] |
| S2R4n2_t-1/summary_linear/80 | -0.01547900 [-0.01555841, -0.01539959] |
| S2R4n2_t-1/summary_mlp/20 | -0.04579833 [-0.05867603, -0.03292063] |
| S2R4n2_t-1/summary_mlp/80 | -0.04435193 [-0.05387110, -0.03483276] |
| S2R4n2_t-1/rich_mlp/20 | -0.03070824 [-0.04608037, -0.01533610] |
| S2R4n2_t-1/rich_mlp/80 | -0.01851260 [-0.04214080, 0.00511561] |
| S3R4n3_t0/summary_linear/20 | -0.00316773 [-0.00329614, -0.00303933] |
| S3R4n3_t0/summary_linear/80 | -0.00166348 [-0.00178663, -0.00154032] |
| S3R4n3_t0/summary_mlp/20 | -0.03518822 [-0.04385392, -0.02652251] |
| S3R4n3_t0/summary_mlp/80 | -0.02958132 [-0.05034404, -0.00881859] |
| S3R4n3_t0/rich_mlp/20 | -0.03689239 [-0.04578945, -0.02799533] |
| S3R4n3_t0/rich_mlp/80 | 0.00288864 [-0.02360986, 0.02938714] |
| S3R4n3_t-1/summary_linear/20 | 0.00557319 [0.00544771, 0.00569867] |
| S3R4n3_t-1/summary_linear/80 | 0.00734880 [0.00719706, 0.00750055] |
| S3R4n3_t-1/summary_mlp/20 | -0.03126846 [-0.05514175, -0.00739518] |
| S3R4n3_t-1/summary_mlp/80 | -0.01704725 [-0.02449983, -0.00959466] |
| S3R4n3_t-1/rich_mlp/20 | -0.02618286 [-0.04382652, -0.00853920] |
| S3R4n3_t-1/rich_mlp/80 | -0.03270862 [-0.05468578, -0.01073146] |

## Long80 minus short20:selected-pipeline effect

| fixedcell | mean and95%CI(n5pairedseeds) |
|---|---|
| S2R4n2_t0/summary_linear/marginal | -0.01748612 [-0.01757801, -0.01739423] |
| S2R4n2_t0/summary_linear/joint | -0.02687152 [-0.02696072, -0.02678233] |
| S2R4n2_t0/summary_mlp/marginal | -0.24245656 [-0.25677884, -0.22813428] |
| S2R4n2_t0/summary_mlp/joint | -0.25953944 [-0.27229764, -0.24678125] |
| S2R4n2_t0/rich_mlp/marginal | -0.18649335 [-0.19338003, -0.17960667] |
| S2R4n2_t0/rich_mlp/joint | -0.18261262 [-0.18687757, -0.17834767] |
| S2R4n2_t-1/summary_linear/marginal | -0.07312110 [-0.07328337, -0.07295883] |
| S2R4n2_t-1/summary_linear/joint | -0.07702328 [-0.07710888, -0.07693768] |
| S2R4n2_t-1/summary_mlp/marginal | -0.20184483 [-0.21769242, -0.18599724] |
| S2R4n2_t-1/summary_mlp/joint | -0.20039844 [-0.21068888, -0.19010800] |
| S2R4n2_t-1/rich_mlp/marginal | -0.21139600 [-0.22898305, -0.19380895] |
| S2R4n2_t-1/rich_mlp/joint | -0.19920036 [-0.20919611, -0.18920460] |
| S3R4n3_t0/summary_linear/marginal | -0.06333810 [-0.06361848, -0.06305772] |
| S3R4n3_t0/summary_linear/joint | -0.06183385 [-0.06197983, -0.06168786] |
| S3R4n3_t0/summary_mlp/marginal | -0.20801051 [-0.22814933, -0.18787169] |
| S3R4n3_t0/summary_mlp/joint | -0.20240361 [-0.21160493, -0.19320228] |
| S3R4n3_t0/rich_mlp/marginal | -0.22025836 [-0.23535081, -0.20516591] |
| S3R4n3_t0/rich_mlp/joint | -0.18047733 [-0.19876380, -0.16219086] |
| S3R4n3_t-1/summary_linear/marginal | -0.11725379 [-0.11729845, -0.11720913] |
| S3R4n3_t-1/summary_linear/joint | -0.11547818 [-0.11572016, -0.11523619] |
| S3R4n3_t-1/summary_mlp/marginal | -0.21526732 [-0.23394032, -0.19659432] |
| S3R4n3_t-1/summary_mlp/joint | -0.20104610 [-0.21712808, -0.18496412] |
| S3R4n3_t-1/rich_mlp/marginal | -0.13249370 [-0.14792967, -0.11705773] |
| S3R4n3_t-1/rich_mlp/joint | -0.13901946 [-0.16843433, -0.10960459] |

## RichMLP minus summaryMLP:matched loss/budget

| fixedcell | mean and95%CI(n5pairedseeds) |
|---|---|
| S2R4n2_t0/marginal/20 | -0.16391497 [-0.17680550, -0.15102444] |
| S2R4n2_t0/marginal/80 | -0.10795176 [-0.12033356, -0.09556997] |
| S2R4n2_t0/joint/20 | -0.17950739 [-0.19716109, -0.16185370] |
| S2R4n2_t0/joint/80 | -0.10258057 [-0.11647545, -0.08868569] |
| S2R4n2_t-1/marginal/20 | -0.18991008 [-0.20315709, -0.17666306] |
| S2R4n2_t-1/marginal/80 | -0.19946124 [-0.22362336, -0.17529913] |
| S2R4n2_t-1/joint/20 | -0.17481999 [-0.18885577, -0.16078420] |
| S2R4n2_t-1/joint/80 | -0.17362191 [-0.18339683, -0.16384698] |
| S3R4n3_t0/marginal/20 | -0.09924231 [-0.10734165, -0.09114296] |
| S3R4n3_t0/marginal/80 | -0.11149016 [-0.13694829, -0.08603202] |
| S3R4n3_t0/joint/20 | -0.10094648 [-0.10876765, -0.09312531] |
| S3R4n3_t0/joint/80 | -0.07902020 [-0.09505703, -0.06298337] |
| S3R4n3_t-1/marginal/20 | -0.17893524 [-0.20390716, -0.15396332] |
| S3R4n3_t-1/marginal/80 | -0.09616162 [-0.11359441, -0.07872882] |
| S3R4n3_t-1/joint/20 | -0.17384964 [-0.18322030, -0.16447897] |
| S3R4n3_t-1/joint/80 | -0.11182299 [-0.13724478, -0.08640120] |

## Independent checks and execution

Separate checker reconstructs the hand-balanced SHAselector and new210rich feature columns, exact structuralFsets/allRmemberships/targets, disjoint own-hand splits, originalBIreference values, all240selected forward factors/heldKLs using independent DPnormalizers, allcandidate validation costs/earliest-tie selection and112reportedindividual/paired intervals. Fresh policy-reach reenumeration covers ALLselected small keys and32fixed large keys/condition; remaininglargeRrelies on the unchanged previously audited N2posterior/hash, not a fresh complete reimplementation of all2million keys. Same-investigator check, not external approval.

Preflight known-state collision0,3241roots plus5children,2170M/domain0x4e39; gradientFDmax1.398e-11. Firstdata export failed before any row/fit because the frozenencoder returns4values but caller unpacked5. Originalscientificsourcefreeze/selectionUTC retained; newapi_resume.py adds only an unusedNone slot, public-array identity fixture/hashguard. No existing science source changed. First3hwindow stopped after hostclockgap with40fits; predeclaredwindow2resumed fixedcounts/checkpoints. Timings are not equal-compute/efficiency results. Allerrors/prefixes/states retained.

Rawper-contextKL/logits,models/candidates/epochstates,allper-hand means and BIsecondary differences are in results/. Original N2data/models/sources/2100Mseeds untouched. No52card/game-score/fusion/optimalposterior/novelty claim. Broader literature review incomplete; globalnormalization is established structuredprediction (RELATED_WORK.md).

Commands(cd engine): preflight.py,self_check.py; recorded bounded api_resume.py(samefrozen run.py), then separately bounded verify_v2.py,report.py. The first separate reader stopped on uint8 multiplication by256; verify_v2 casts both bytes before decoding. Failed reader/log retained; fitted data/models unchanged. Exact absolute stops in STATUS/launches. No commit/push before Claude validation.
