# Round 6 (Claude): preregistration for the prepublication sweep

This was frozen before any Round 6 calculation by Claude. The SHA-256 and UTC time are in `R6C_PREREG_SHA256.txt`.

The reviewer's Round 6 package is in `reviewer_r6/mode12_round6/` and uses the labels R6-1 to R6-6. To avoid a clash, Claude's predictions are labelled **S1–S5** and audit items **A1–A6**.

Before freezing, Claude looked only at metadata, never at results:
- the archive cubes in the scratch folder have shape [G, T, L, (g, gap, H)];
- 117 prompts are shared by ZIT, ZIB, Mochi 4.5, SD35M and SD35L, with 4 generations each;
- the R4 ZIB explicit-branch arms have 20 prompts (CFG 1, cond) and 19 prompts (CFG 4, cond and uncond), with attn and mlp;
- the R3 ZIB CFG 1 arm (`F3_Z10`) is cond-only with 60 prompts;
- the reviewer's files and the 2026-09-15 Mode-I session summary.

Results that are already known are marked as such and are not predictions.

## A. Audit of the reviewer's Round 6 (checks, not predictions)
- **A1. Provenance of every number in `R6_METRIC_HIERARCHY.csv`.** Trace each one to a file in the project and recompute it from the raw cubes where possible.
- **A2. Integrity of their preregistration.** Were the decision rules for R6-1, R6-3 and R6-4 fixed before the inputs were known? Their SHA must match their file.
- **A3. `R6_ESTIMATOR_CALIBRATION.csv` against `round5/out/H1_seedfate_vs_sample_size.csv`.** Rows must match and labels must be unique.
- **A4. Their ledger v0.18.** Which v0.17 build was it based on? Does it carry the later v0.17 edits (Z-Image shift resolved; rider moved to Mode-III)? Are the nav and section structure intact?
- **A5. Their gate table** compared with v0.17. List every difference.
- **A6. The 2026-09-15 Mode-I session summary.** Build a release claim register: every headline claim, with its ledger status as of v0.17, and use / use with wording / do not use.

## S1. Observable hierarchy on one pipeline from raw cubes (attention, archive)

Models: ZIT, ZIB, MOCHI (4.5), SD35M, SD35L, all attn and archive. The lineage pair is ZIT↔ZIB. The unrelated pairs are the 8 cross-family pairs: ZIT/ZIB × MOCHI/SD35M/SD35L, plus MOCHI × SD35M/SD35L. SD35M↔SD35L is reported separately as the same-family pair.

Levels:
- **L0 (scalar envelope).** E_m(σ) is the mean over generations and layers of g at each native σ. The pair r is Pearson, with the denser model's E interpolated (linear in σ) onto the sparser model's σ grid inside the overlapping σ range. **D1** uses g and **D2** uses log g. D1 is primary.
- **L1 (level field).** Mean over generations of log g, depth interpolated to 24 bins, matched σ (nearest), Pearson over all (σ, depth) cells.
- **L2 (registered shape).** This is the Round 2 E10 shape r: depthnorm (log g, log gap, H) per σ, 24 bins, matched σ, mean over σ.
- **L3 (tangent).** This is the Round 2 E10 tangent cosine.

margin_k = r_k(ZIT, ZIB) − max over the 8 unrelated pairs of r_k.

Predictions:
- **S1a.** At L0 (D1 or D2), the lineage is not separated: margin_L0 ≤ 0.05.
- **S1b.** At L0 (D1), r ≥ 0.6 for at least 6 of the 8 unrelated pairs. This is the "generic envelope" test.
- **S1c.** margin_L3 ≥ 0.5 and margin_L3 > margin_L2 > margin_L0.
- **S1d.** Capture regime. Replace archive ZIB attn by the R4 ZIB CFG 4 *cond* branch (explicit, 19 prompts). The ordering statistic Δ = r_L0(ZIB, ZIT) − r_L0(ZIB, MOCHI) then changes by more than 0.008 in absolute value. That is, the historical 0.008 ordering margin is inside the capture-regime sensitivity.
- **Reproduction (not a prediction).** Report how close D1, D2 and per-prompt-median variants come to the historical 0.9405 / 0.9325 / 0.822 (09-15 summary §16).

## S2. Content geometry as a separate axis (117 shared prompts, attn, archive)

Features per prompt, per model, at τ ∈ {first step, middle step T//2, last step}. Take the mean over the prompt's generations of [log g, log gap, H] across depth, interpolate it to 24 bins (72 features), then z-score each feature across prompts within the model. This removes the model backbone.

- **Theme residual.** Average the residuals over each of the 10 domains. The same-theme r is the mean over domains of Pearson(theme vector A, theme vector B) on matched features.
- **RSA.** For each model and τ, build a 117×117 correlation-distance matrix. The Mantel Spearman between two models is tested against 2000 prompt permutations.
- **Text RSA.** Use TF-IDF cosine distance of the prompt texts.

Predictions:
- **S2a.** Lineage same-theme residual r > 0.45 at ≥ 2 of the 3 τ. This is the reviewer's R6-3 rule, applied to raw data.
- **S2b.** For every unrelated pair, the same-theme residual r is ≤ 0.20 at ≥ 2 of the 3 τ.
- **S2c.** Lineage RSA (ZIT↔ZIB) is ≥ 0.30 at all 3 τ, each with p < 0.01.
- **S2d.** Shared text encoders. At the first step, SD35M↔SD35L RSA is ≥ 0.20, although their tangent is −0.25.
- **S2e.** Encoder-sharing order at the first step: ZIT↔ZIB > SD35M↔SD35L > max(MOCHI↔SD35M, MOCHI↔SD35L) > max over the 6 pairs with no shared encoder (Z-Image × MOCHI/SD35M/SD35L). Encoders are Qwen3 for Z-Image, T5-XXL for Mochi, and CLIP-L/G + T5-XXL for SD3.5; `LIT_R6C.md` verifies these.
- **S2f.** The text RSA correlates positively with the model RSA at the first step (Mantel > 0, p < 0.05) in ≥ 4 of 5 models.
- **S2g.** Content and dynamics are different axes. Over the 10 model pairs, Spearman(RSA at first step, L3 tangent) < 0.5.

## S3. Estimator re-statement (the "now" gate; frozen procedure)

Channels: ZIT attn, ZIT mlp, ZIB attn, ZIB mlp, MOCHI, SD35M, SD35L (archive).

- **Seed fate.** Use the Round 2 E5 / Round 5 `curve` estimator: within-prompt residuals, ridge α = 100, prompt-grouped 5-fold CV, uncentred. Run it at matched n = 60 prompts (× their gens) over 50 random prompt subsets, and also at native n.
- **Reported quantities.** σ50 (R² crosses 0.5) and relative onset (R²/R²_final crosses 0.5), as median with p5–p95.
- **VR4.** Use the Round 5 `VR` estimator. Simulated walk, AR(0.8) and iid references are generated at each channel's own shape (T, n gens, prompts). Report VR4 / VR4_walk.

Predictions:
- **S3a.** Relative onset at matched n lies in [0.80, 0.97] σ for ≥ 6 of 7 channels.
- **S3b.** Across the 7 channels, the range of relative onset is smaller than the range of σ50 (both at matched n).
- **S3c.** The partition spread (p95 − p5) is smaller for relative onset than for σ50 in ≥ 6 of 7 channels.
- **S3d.** VR4 / VR4_walk ≥ 1 in ≥ 5 of 7 channels.

## S4. Three-regime topology on clean captures

Fit a continuous three-segment piecewise-linear model in σ to the population-mean layer-mean g, by least squares, with two breakpoints searched on the native σ grid. Each segment must contain at least 2 σ points; ZIT is exempt from that minimum. Slopes are signed in time, so σ decreasing is forward. The topology is the sign pattern; "+−+" means expand, contract, re-expand. Use 1000 prompt bootstraps.

- **S4a.** Archive ZIT mlp and ZIB mlp give +−+ in ≥ 95% of bootstraps. This is a reproduction of the 09-15 result of 1000/1000.
- **S4b.** Clean ZIB mlp gives +−+ in ≥ 95% of bootstraps, on both R4 Z40 cond (CFG 4, 19 prompts) and R3 Z10 cond-only (CFG 1, 60 prompts).
- **S4c.** The R4 Z40 cond mlp turning points are within 0.03 (peak) and 0.05 (trough) of the archive ZIB mlp values from this pipeline.
- **S4d.** The R4 Z40 cond mlp peak-to-trough contraction is within ±10 percentage points of archive ZIB mlp (same pipeline).
- **S4e.** Cross-architecture recurrence: +−+ in ≥ 80% of bootstraps for ≥ 4 of the 5 archive attn channels (ZIT, ZIB, MOCHI, SD35M, SD35L).

## S5. Synchronization R(σ): what survives a stronger null

R is defined as in the 09-15 summary §19, with per-layer z-scores over time.

There are two nulls, each with 500 draws per channel on the population-mean field:
- **Null 1, the warp null.** Each layer's trajectory (all 3 variables together) gets its own random monotone time warp: cumulative Dirichlet(1) increments, with endpoints fixed. This keeps each layer's shape and monotonicity but desynchronizes timing across layers.
- **Null 2, the trend surrogate.** Each layer×variable is replaced by its own least-squares line in time. This is Round 1's surrogate.

Predictions:
- **S5a.** Observed R_peak > p95 of the warp null in ≥ 5 of 7 channels. Cross-layer timing is coordinated beyond what independent monotone drift gives.
- **S5b.** Observed R_peak ≤ the trend surrogate R_peak in 7 of 7 channels. A peak in R is therefore not evidence of an interior event.

## Scoring

Every row is scored CONFIRMED, REFUTED or NOT SCORABLE, and the reason is given for any row that is NOT SCORABLE. Post hoc work is labelled as such. The reviewer's rows R6-1 to R6-6 are re-scored under A2 on their own rules, using recomputed inputs where possible.
