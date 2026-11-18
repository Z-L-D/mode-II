# R3 Part B results: ACE-Step 1.5 turbo (audio flow transformer)

Scored 2026-10-09 against `review/r3_prereg/R3_PREREGISTRATION.md` (SHA efa613eb, 2026-10-08 15:27Z, Amendment 1),
under `R3B_SCORING_PLAN.md` (SHA 7fa54aab, frozen 2026-10-09T06:36:23Z; the SHA file holds the true time).
Analyzer and lens code used unchanged. Post hoc items are labelled.

## Capture (upload `AS15-attn.mlp-TE1-CFG1.0-R180-SRAND-STUDYPRIMARY-AUDIO-G120.zip`, 2026-10-09 06:33Z)
- ACE-Step 1.5 **turbo** (6-step distilled), CFG 1, cond-only (passive branch, cond), 180 s audio.
- 24 layers x {attn, mlp}, seq 2250, D 2048; 6 steps, sigma = 1.0, 0.9378, 0.8573, 0.75, 0.6007, 0.3756.
- 120 generations, 10 domain CSVs, 29 prompts (jazz TSV idx 5 missing; one jazz prompt run 8 times), 114 unique random seeds.
- Deviations from protocol (turbo, 6 steps, 29 prompts) are recorded in the scoring plan.

## Scorecard: 4 confirmed, 4 refuted, B1 gate passed, B7 not run

| ID | Result | Verdict |
|---|---|---|
| B1 | 24 layers x attn/mlp on every step; N 2250; D 2048 | PASS |
| B2 | g_MP 0.7505; cells >= 0.10: attn 71.5%, MLP 95.1%; median ratio 0.148 / 0.303 | tested, not supported |
| B3 | PC1+PC2 attn 0.9878, MLP 0.9758 (< 0.98) | tested, not supported |
| B4 | dR2 attn 0.596 (0.325 -> 0.920), MLP 0.773 (0.016 -> 0.789) | CONFIRMED |
| B5 | even > odd log g in 100% of (gen, step) cells; attn median diff 0.580 (MLP 0.118, descriptive) | CONFIRMED |
| B6 | attn one crossing (+-, g* 0.090, slope -0.90); 1 crossing in 99/100 prompt bootstraps. Descriptive: even-only 3 crossings, odd-only 1 | tested, not supported |
| B7 | conditional on B6 | NOT RUN |
| B8 | onset sigma attn 0.898, MLP 0.804, inside [0.73, 0.92]; plateau speed from 1 interval | CONFIRMED |
| B9 | chi_logit argmax step 4 (sigma 0.60), interior, stability 1.0, margin over last step < 2 SE: no peak passes | CONFIRMED |
| B10 | prompt share 0.96 at sigma 1, dips to 0.86 / 0.82 at sigma 0.60, steepest rise in the last interval | tested, not supported |

## Preregistered consequences
- B2, B3 and B10 fail, so "not specific to image data" is NOT established for those properties. B4 and B8 hold in audio.
- **B6 fails while B4 holds: the layer-axis crossing pattern is image-specific (image and video in this corpus), and the "coarse envelope recurrence" depth item in Mode-II stays image/video-only.**
- B6/B7 did not both hold: no cross-modality recurrence claim.

## Post hoc (labelled post hoc wherever reported)
- P1 backbone: attn PC1 0.775, rank 2.10 (null 3.86, p 0.001); MLP PC1 0.564, rank 2.73 (null 4.08, p 0.011). 6 states, rank <= 5.
- P3 prompt share: naive 0.86-0.97, ANOVA 0.82-0.96 (`out/P3_prompt_share_by_step.csv`).
- P5 decoding: uninformative (2-3 prompts per domain; TF-IDF at chance 0.10); telemetry 0.11 / 0.12.
- P7 clock envelopes: -+- in 100% of 1000 prompt bootstraps for both submodules, max at sigma 1, first-step drop 19% / 16%. Descriptive (6 points).
- A2/A4 analyzer replay: attn 1 crossing in 100% of generation bootstraps and 6/6 steps; MLP 1 crossing in 95%, 3 in 5%, 3 crossings at 2/6 steps; surrogates A and B reproduce the single crossing 30/30.

## Caveats
One audio model; distilled turbo; 6 steps; cond-only at CFG 1; 29 prompts. ZIT attention (distilled, cond-only) also has a single crossing,
so modality and distillation are confounded for B6. An undistilled audio model with more steps would separate them.

## Where it is in the papers
Mode-I: new Section 12 (`sec:ace`, Fig I-F14), rows in Table 1, Table 3, Table 4 (tab:beta), I-F1, I-F2, I-F5a, I-F7; abstract, synthesis, limits, prereg table.
Mode-II: Sections 5, 9.2 (layer axis, prereg consequence), 9.1, 13 (universality item 1 scoped), 14; II-F5c curves; II-F12 audio column; prereg table.

Outputs: `out/` (see scripts a0-a3). Ledger v0.27 section `r3-ace`.
