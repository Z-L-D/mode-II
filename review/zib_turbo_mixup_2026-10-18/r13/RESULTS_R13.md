# Round 13 results: patterns in the real-Base data (2026-10-27)

Discipline: prompts split 60/60 before looking (SPLIT.md, sha 3973063e). Patterns were found on the EXPLORE half only, written down as predictions with thresholds (PREDICTIONS_R13.md sha 47cf1e67; ACE addendum sha 79631bc3), and only then computed on the sealed HOLDOUT half. 10 of 10 predictions held. Verdicts: out/r13_verdicts.csv, out/r13_ace_verdicts.csv.

Caveat: the holdout is new prompts from the same runs, not a new capture. The spectral-arm comparisons (H1, H2, H3a, H4) compare Base at 25 steps with Turbo at 8 steps, matched by noise level. H3b uses identical settings.

| ID | Finding (HOLDOUT) | Numbers |
|---|---|---|
| H1 | Base image tokens share a bigger common "mean direction" than Turbo's | Base > Turbo in 30/30 attn and 29/30 MLP layers |
| H2 | That mean direction is most of what separates Base from Turbo | Base−Turbo gap 4.3× (attn) and 2.8× (MLP) smaller once the mean is removed |
| H3 | Base is more concentrated (lower g) than Turbo everywhere | image rows: 30/30 layers in both; identical settings: all 25 steps in both |
| H4 | Distillation changed the first and last layers, not the middle | middle (L17–22) difference 0.16 / 0.11 vs early 0.60 / 1.09 and late 1.06 / 0.93 (attn / MLP) |
| H5 | Guidance has a fixed signature in Base attention | cond−uncond: raised in the first and last layers, lowered in L6–21, lowered at every step 2–15 |
| H6 | The all-rows Base statistic is set by the prompt from the very first step | prompt share 0.94 at step 0 (expected from the caption rows dominating) |
| A1 | ACE on the image prompts has no family resemblance to any image model | tangents −0.12 to +0.29 (all in the unrelated range) |
| A2 | ACE doesn't rank prompts the way the image models do | per-prompt correlations 0.08–0.17 |
| A3 | ACE's attention fingerprint is set by its weights, not by the prompt type | ACE on image prompts vs on music prompts: attn 0.95, MLP 0.77 |
| A4 | ACE's statistic carries no image-domain information | domain share 0.16 / 0.13, about chance |
