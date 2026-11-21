# N13 — conditional-product belief family limits

**Externally validated by Claude, 2026-11-20**; the M18 lemma was re-derived by hand and every per-key certified bound recomputed independently. See CLAUDE_AUDIT_2026-11-20.md.

There are15 strictly positive entropy witnesses among the256 prospectively selected contexts. These counts describe the complete selected diagnostic grid, not an estimate of prevalence on all contexts. Each witness lower-bounds KL for every finite card-holder conditional-product potential, including an oracle with unrestricted per-context potentials. It therefore concerns a factor-family limitation, beyond finite shared neural features or optimization in those cases. It is not a universal claim that the family fails on every context or that improved KL improves play.

For P uniform on R, and any exact-moment distribution q on structural F, KL(P||Q_theta) >= H(q)-H(P). The finite proof is in BOUND.md, pending external audit; the variational idea is established, not a novelty claim (RELATED_WORK.md). Rational LP proposals are accepted only after exact normalization/moment checks. Every reported lower endpoint uses outward-rounded rational log enclosures, independently checked with a longer series.

## All four fixed diagnostic subsets

| condition | selected keys / exact reach | positive witnesses | certified mean KL lower bound, approx nats | numerical free-potential KL value | optimizer status failures |
|---|---:|---:|---:|---:|---:|
| S2R4n2_t0 | 64 / 123 | 1 | 0.0081165408 | 0.0135341478 | 0 |
| S2R4n2_t-1 | 64 / 135 | 1 | 0.0032926298 | 0.0053339449 | 1 |
| S3R4n3_t0 | 64 / 236 | 7 | 0.0881641563 | 0.3418880293 | 1 |
| S3R4n3_t-1 | 64 / 170 | 6 | 0.0660799372 | 0.2030068841 | 1 |

PRIMARY is the largeNT reach-weighted certified lower bound. Its exact rational nats endpoint is:

119456011386581294114109985962707667077/1807750074267485587149177601981268623360

No sampling95%CI applies to these deterministic finite-support bounds, witness counts or optimization outputs. Decimal lower values above approximate exact rational endpoints, not exact optima. The numerical column is a floating objective at saved feasible finite potentials, an upper estimate of the infimum subject to roundoff, NOT a rigorous interval or solved floor. Three optimizer status failures are retained; gradient residuals/status/messages are saved for every key. No restart or count extension was used to erase them. All256 LP proposals reconstructed exact rational marginals; zero reconstruction failures.

All four subsets use64 held keys sorted by the predeclared SHA selector before optimization, reusing known N9/N2 supports. No new chance or optimizer seeds, fresh-hand replication or deployment result. Some selected keys have posterior size1 or otherwise admit no positive witness; all zero certificates are retained. The fixed mixing grid supplies feasible witnesses, not maximum-entropy solutions. More sophisticated entropy optimization could tighten bounds but was not retroactively run.

## Per-key complete results

| condition / key | reach / structural worlds | certified lower endpoint, approx nats | numerical potential value | optimizer success |
|---|---:|---:|---:|---|
| S2R4n2_t0/4909 | 1 / 12 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/298 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/10015 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/8940 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/8614 | 2 / 6 | 0.0000000000 | 0.0000000004 | True |
| S2R4n2_t0/2729 | 2 / 30 | 0.0000000000 | 0.0000000002 | True |
| S2R4n2_t0/9249 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/942 | 1 / 1 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t0/6541 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/8125 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/5800 | 4 / 12 | 0.0000000000 | 0.0000000006 | True |
| S2R4n2_t0/8035 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/9470 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/8594 | 4 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/6087 | 2 / 6 | 0.0000000000 | 0.0000000004 | True |
| S2R4n2_t0/10150 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/5579 | 1 / 3 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/8037 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/6542 | 1 / 1 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t0/8965 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/2896 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/8638 | 1 / 4 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/934 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/6534 | 1 / 1 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t0/3056 | 2 / 4 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/2877 | 4 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/940 | 1 / 1 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t0/250 | 2 / 2 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t0/2983 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/235 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/6022 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/3153 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/3417 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/8128 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/3148 | 2 / 2 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t0/9899 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/8916 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/5960 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/147 | 8 / 90 | 0.1247918150 | 0.2080875201 | True |
| S2R4n2_t0/5557 | 2 / 9 | 0.0000000000 | 0.0000000006 | True |
| S2R4n2_t0/5586 | 1 / 12 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/8919 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/5814 | 2 / 12 | 0.0000000000 | 0.0000000011 | True |
| S2R4n2_t0/542 | 2 / 2 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t0/8048 | 4 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/8908 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/5014 | 4 / 9 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t0/4980 | 1 / 12 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/4976 | 5 / 12 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/5946 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/9261 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/6651 | 2 / 2 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t0/6083 | 2 / 6 | 0.0000000000 | 0.0000000004 | True |
| S2R4n2_t0/3542 | 1 / 1 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t0/9538 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/9896 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/316 | 2 / 2 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t0/2959 | 6 / 6 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t0/2908 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/408 | 1 / 4 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/5027 | 6 / 12 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t0/9253 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/4916 | 1 / 3 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t0/8165 | 2 / 2 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t-1/3184 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/163 | 4 / 90 | 0.0000000000 | 0.0000000000 | False |
| S2R4n2_t-1/6596 | 1 / 1 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t-1/3696 | 2 / 2 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t-1/6796 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/1321 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/3208 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/243 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/9382 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/8662 | 2 / 6 | 0.0000000000 | 0.0000000004 | True |
| S2R4n2_t-1/9783 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/8685 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/2273 | 3 / 30 | 0.0000000000 | 0.0000000014 | True |
| S2R4n2_t-1/10157 | 4 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/3967 | 1 / 1 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t-1/6338 | 2 / 4 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/8686 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/4111 | 2 / 2 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t-1/1315 | 2 / 2 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t-1/827 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/355 | 2 / 2 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t-1/6806 | 2 / 2 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t-1/357 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/342 | 4 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/872 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/598 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/842 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/2285 | 2 / 30 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/10586 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/263 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/220 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/5648 | 9 / 12 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t-1/3096 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/5244 | 1 / 3 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/8792 | 4 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/10535 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/3100 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/6261 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/6250 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/8800 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/4096 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/861 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/1025 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/8780 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/868 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/4097 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/2546 | 18 / 30 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t-1/2790 | 1 / 30 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/3097 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/5269 | 2 / 3 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/16 | 11 / 90 | 0.0404095474 | 0.0654620504 | True |
| S2R4n2_t-1/4103 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/2929 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/2845 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/6249 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/2849 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/10643 | 2 / 2 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t-1/10829 | 1 / 1 | 0.0000000000 | 0.0000000000 | True |
| S2R4n2_t-1/8787 | 2 / 6 | 0.0000000000 | 0.0000000004 | True |
| S2R4n2_t-1/3181 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/6146 | 1 / 12 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/5254 | 3 / 12 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/9820 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S2R4n2_t-1/6163 | 1 / 3 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/53059 | 1 / 24 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t0/77305 | 2 / 5 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t0/1008813 | 1 / 60 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t0/1753809 | 2 / 4 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t0/61630 | 1 / 9 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t0/1661511 | 1 / 60 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t0/247318 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/1104097 | 2 / 12 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t0/563982 | 3 / 30 | 0.0000000000 | 0.0000000023 | True |
| S3R4n3_t0/751663 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/1101428 | 12 / 30 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t0/1906836 | 1 / 1 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t0/580616 | 1 / 3 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/1940995 | 1 / 1 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t0/734142 | 1 / 3 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/741885 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/1614962 | 6 / 60 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t0/1231174 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/1569680 | 3 / 90 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t0/613471 | 2 / 9 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t0/1759623 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/1801225 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/1075647 | 9 / 30 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t0/1736209 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/23082 | 5 / 60 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t0/1296443 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/1300672 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/787095 | 1 / 1 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t0/1693384 | 2 / 24 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t0/1302017 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/1270979 | 1 / 4 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/1305037 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/1906828 | 1 / 1 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t0/1300771 | 1 / 1 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t0/25471 | 1 / 18 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t0/1554308 | 18 / 90 | 0.1753305591 | 0.2703100721 | True |
| S3R4n3_t0/1558039 | 1 / 36 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t0/615366 | 1 / 9 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t0/1750044 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/172479 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/510106 | 4 / 560 | 0.3890483495 | 0.4054651081 | True |
| S3R4n3_t0/1374660 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/1009028 | 18 / 210 | 0.3505401728 | 1.6094379124 | True |
| S3R4n3_t0/693620 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/1747951 | 1 / 4 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/765763 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/687712 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/561808 | 3 / 18 | 0.0000000000 | 0.0000000033 | True |
| S3R4n3_t0/1045347 | 1 / 150 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t0/1054721 | 54 / 210 | 0.0772849420 | 0.6287985876 | True |
| S3R4n3_t0/79058 | 1 / 9 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t0/1873404 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/1740626 | 2 / 9 | 0.0000000000 | 0.0000000006 | True |
| S3R4n3_t0/1642166 | 2 / 60 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t0/1579281 | 1 / 60 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t0/60696 | 5 / 12 | 0.0998334520 | 0.1833321793 | True |
| S3R4n3_t0/1669141 | 6 / 30 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t0/94972 | 1 / 3 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/501344 | 16 / 560 | 0.2590832531 | 0.4739923368 | True |
| S3R4n3_t0/1740527 | 1 / 4 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/1020883 | 1 / 210 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t0/1043417 | 2 / 210 | 0.0000000000 | 0.0000000028 | False |
| S3R4n3_t0/571580 | 1 / 12 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t0/1029829 | 16 / 210 | 0.0604367581 | 0.1732867951 | True |
| S3R4n3_t-1/775263 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1165841 | 3 / 6 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t-1/278049 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1592476 | 9 / 36 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t-1/1208788 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/502271 | 1 / 60 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t-1/1618764 | 2 / 60 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1214464 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1661596 | 11 / 60 | 0.0404095474 | 0.0654620504 | True |
| S3R4n3_t-1/1246692 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/598540 | 4 / 12 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t-1/993969 | 1 / 100 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t-1/1926991 | 1 / 4 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1098356 | 1 / 4 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/199442 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/550029 | 1 / 18 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t-1/1939020 | 1 / 4 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1227869 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/18120 | 6 / 60 | 0.0831945433 | 0.1800673768 | True |
| S3R4n3_t-1/486874 | 7 / 560 | 0.0000000000 | -0.0000000000 | False |
| S3R4n3_t-1/1700043 | 1 / 12 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/666521 | 1 / 4 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1975615 | 1 / 1 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t-1/1799364 | 2 / 4 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t-1/145306 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1961137 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1731698 | 1 / 4 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1730651 | 1 / 4 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/85880 | 2 / 12 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t-1/1156374 | 1 / 12 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1272970 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1867188 | 2 / 4 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1718530 | 1 / 12 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/495963 | 24 / 560 | 0.2737103305 | 0.7513022102 | True |
| S3R4n3_t-1/1870244 | 1 / 4 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1771107 | 1 / 4 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/487438 | 1 / 560 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t-1/1109632 | 2 / 12 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t-1/1083785 | 1 / 12 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/278046 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/5022 | 2 / 30 | 0.0000000000 | 0.0000000020 | True |
| S3R4n3_t-1/661014 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/236273 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1180757 | 1 / 3 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/621955 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/612424 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/676109 | 1 / 4 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/600197 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1198703 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1678352 | 24 / 90 | 0.0372578030 | 0.3639559152 | True |
| S3R4n3_t-1/1978151 | 1 / 1 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t-1/1078482 | 6 / 12 | 0.0000000000 | 0.0000000000 | True |
| S3R4n3_t-1/487474 | 11 / 560 | 0.2115922346 | 0.4928400296 | True |
| S3R4n3_t-1/515345 | 1 / 30 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1374872 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/280581 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/133156 | 1 / 4 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/105013 | 3 / 12 | 0.1663890866 | 0.1744160479 | True |
| S3R4n3_t-1/1660667 | 1 / 6 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1041705 | 1 / 60 | 0.0000000000 | 0.0000000002 | True |
| S3R4n3_t-1/1827935 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1080174 | 2 / 6 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/1799087 | 1 / 4 | 0.0000000000 | 0.0000000001 | True |
| S3R4n3_t-1/775424 | 1 / 2 | 0.0000000000 | 0.0000000001 | True |

## Separate checks, provenance and limits

Separate verify.py checks all256 rational mixture normalizations/moments, exact construction from LP proposals, alternate40-term rational log bounds/containment/strict positivity, independent finite-potential forward KL/gradients and every reach-weighted aggregate fraction. Synthetic tests cover a four-state entropy example and scaled rational logs. These checks share the finite lemma and previously validated N9/N2 support definitions; they are NOT external mathematical approval.

The illustrative smallest positive witness is exported with its full structural owner set, exact posterior indices, rational weights, private/public key and lower enclosure in results/smallest_positive_example.json. It was selected posthoc for readability, not as a new confirmatory result.

The pre-data synthetic check initially imported an older N2 verify.py through a frozen helper path. Original helper/error retained; current-task import precedence was fixed BEFORE any game LP/projection or source freeze. No source/data/model/seeds changed after outcomes. SciPy warned that threads=1 is passed verbatim to HiGHS; that is intentional single-worker control, not a failed solve. All other optimizer status failures are retained.

Files: PROTOCOL.md, BOUND.md(pending external), append-onlySTATUS, lib13.py/run.py/self_check.py/verify.py/report.py, complete per-key potentials/LP proposals/mixtures/log certificates and summaries. Commands(cd engine): bounded_job.py N13/self_check.py; bounded_job.py N13/run.py; separately bounded_job.py N13/verify.py; report.py. Absolute stops/checkpoints inSTATUS/launch records. Original N2/N9/core/policies unchanged; no game-score/52-card/equilibrium/generalNN guarantee, novelty claim, commit or push.
