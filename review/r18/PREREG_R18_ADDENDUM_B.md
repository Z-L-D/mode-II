# R18 addendum B (2026-11-09, before any R18 capture): activation baseline (CKA, Procrustes)

Zac chose option A on 2026-11-09 12:17Z: add activation sketches to every Round 18 arm and rerun the four Round 17
models with them. Reviewers (GLM M6/M9, DeepSeek) asked for an activation-based similarity baseline on the same pairs;
the archive stores only the spectral triple, so none could be computed before. The frozen predictions R18-1 to R18-4,
the gates R18-0a to R18-0d (PREREG_R18.md, sha 6db32be7) and addendum A are unchanged.

## Instrument
Capture node option `act_sketch` (R8; `capture_code_r4/nodes.py`, test `tests/test_r8_sketch.py`, all pass). At step 0
and at the steps whose flow time is nearest 0.75, 0.5 and 0.25 (t = sigma for flow models, sigma / (1 + sigma)
otherwise), for every attention and MLP hook of the conditional branch, it stores in float16:
- the token mean over image rows (caption rows and padding excluded);
- 36 image tokens at fixed normalised image positions ((i + 0.5) / 6, (j + 0.5) / 6), frame 0, row-major.
The main telemetry CSV is byte-identical with the option on or off (tested). Expected disk use about 25 GB for all
sessions.

## Sessions
All Round 18 arms (R18_SD35L, R18_SD35LT, R18_ZIT2, R18_ZB6) with sketches, plus at their Round 17 settings, prompts
`prompts_r3_cond60.tsv`, seed 1001: R18S_PIXART, R18S_COSMOS2B, R18S_SD3M, R18S_SD35M, and R18S_SD35M1 (SD3.5 Medium at
CFG 1, the Round 11 setting) as a same-weights reference. The R18 arms run first on each GPU.

## Pairs
- Same weights: R18S_SD35M vs R18S_SD35M1.
- Related: R18_SD35L vs R18_SD35LT (documented teacher and student, 38 blocks each); R18S_SD3M vs R18S_SD35M
  (successor, 24); R18_ZB6 vs R18_ZIT2 (siblings, 30; common prompts only).
- Unrelated (different developers): every pair across the families {SD3.5 L, SD3.5 L Turbo, SD3 M, SD3.5 M},
  {Z-Image Base, Turbo}, {PixArt-Sigma}, {Cosmos-Predict2 2B}: 21 pairs, one of them depth-matched (PixArt vs Cosmos).
- Same vendor, relation undocumented (descriptive only): the four remaining SD3-family pairs.

## Statistics (script `tier_controls_r18/cka_r18b.py`, frozen with this file)
Primary: linear CKA on token rows (common prompts x 36 tokens), per layer pair; summary = symmetric best match
((mean_i max_j + mean_j max_i) / 2, no layer correspondence assumed, so defined across depth); averaged over the four
states matched by label. Null: the second model's prompts permuted (token rows move with their prompt), 200 draws, the
same permutation for every layer and state; p = (k + 1) / (N + 1). Attention is primary; MLP is reported.
Secondary (reported, no rule): the same at matched normalised depth; prompt-level linear CKA (1,000 draws) and debiased
CKA (200 draws); orthogonal Procrustes similarity on prompt rows (no null).

## Predictions and decision rules (attention, primary statistic)
- **B1 (prediction).** The same-weights pair and each of the three related pairs lie above their null p95.
- **B2 (decision rule, no direction).** CKA *separates* related from unrelated pairs if the lowest of the three related
  pairs is above the highest of the 21 unrelated pairs; otherwise it does not. The rank AUC of related against
  unrelated is reported either way.
- **B3 (decision rule, no direction).** On the depth-matched pairs measured with both instruments (SD3.5 L vs Turbo,
  SD3 M vs SD3.5 M, Z-Image Base vs Turbo, PixArt vs Cosmos, SD3.5 M CFG 4.5 vs 1), CKA *agrees with the tangent
  ordering* if the Spearman correlation between the primary CKA value and the native-grid attention tangent of the same
  sessions is at least 0.8 (with five pairs, 0.9 or 1.0).
- **B4 (prediction).** The same-weights pair has the highest primary value of all pairs.

Every result is reported whatever it is. A CKA that separates the pairs where the tangent does not, or the reverse, is
reported as such in both papers; neither instrument is declared the reference.

## Amendment B.1 (2026-11-09 15:55Z, before any capture; Zac asked whether Base needs all 120 prompts)
R18_ZB6 stores sketches only for the 60 prompts of `prompts_r3_cond60.tsv` (the prompts it shares with R18_ZIT2; the
telemetry still covers all 120 for R18-4). CKA uses common prompts only, so no statistic or rule changes. Driver option
`m3_comfy.py --sketch_only`; disk for all sketches about 22 GB.
