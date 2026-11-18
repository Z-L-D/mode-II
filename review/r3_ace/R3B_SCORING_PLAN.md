# R3 Part B (ACE-Step 1.5 audio arm): scoring plan, frozen before any outcome is computed

Written 2026-10-09 around 06:45 UTC; SHA-256 and UTC time in `R3B_SCORING_PLAN_SHA256.txt`.
Predictions B1-B10 were frozen in `review/r3_prereg/R3_PREREGISTRATION.md` (SHA efa613eb, 2026-10-08 15:27Z, with Amendment 1).
This file fixes the operational choices that file left open. Nothing in it changes a prediction or a threshold.

**Looked at so far:** file list, CSV header, the first 9 rows of one file (gen 1, step 0, layers 0-4), and a profile of
counts: rows, gens, steps, layers, seq_len/hidden_dim, sigma per step, prompts and seeds. No ratio, R2, crossing, peak,
share or curve has been computed.

## Data (as captured; recorded, not chosen, per the prereg)
- Upload 2026-10-09 06:33Z: `AS15-attn.mlp-TE1-CFG1.0-R180-S(RAND)-STUDY[PRIMARY-AUDIO]-G120.zip`, 10 domain CSVs -> `data/`.
- Model: **ACE-Step 1.5 turbo, 6-step distilled** (Zac). The prereg says "ACE-Step 1.5"; the turbo checkpoint is reported as
  such everywhere, and no claim is made about the base checkpoint.
- 6 steps, CFG 1.0, so only cond rows exist (hook_mode passive-branch, branch cond). sigma 1.000 -> 0.376 over the 6 recorded
  evaluations (flow-matching sigma, used as the clock directly).
- 120 complete gens (6 steps x 24 layers x {attn, mlp}), seq_len 2250, hidden_dim 2048.
- Prompts: 29 of the 30 in `ace_prompts_30.tsv` (jazz idx 5 absent; one jazz prompt ran 8 times), 4 random seeds per
  prompt otherwise. Domains = the TSV domain of each prompt text (equals the file name for every gen).
- Gens keyed by (file, gen_idx). Loading for B2-B10 uses the analyzer DataManager unchanged, exactly as
  `standing_wave/sw_load.py`, into a cache with the same keys (g_gen, gap_gen, ent_gen, sigma, prompts).
  If DataManager renumbers or collides gen_idx across files, the cache is built directly from the CSVs with the same pivots.

## Per prediction
- **B1 gate:** 24 layers x {attn, mlp} on every step of every gen, seq_len 2250, hidden_dim 2048.
- **B2:** g_MP(2250, 2048) = mean over 20 white-Gaussian draws of g (singular values of X/sqrt(N)), as in the U-Net arm.
  Cell = (step, layer), value = median over gens of g / g_MP; 6 x 24 = 144 cells per channel.
  Refuted if more than 5% of cells are >= 0.10 in either channel.
- **B3:** `pca_shares` (copied verbatim from `lens_L1_triple.py`) on (H1/lnD, H2/lnD, ln gap) over all rows, H2 = ln(g D),
  H1 = sv_entropy. Holds if PC1+PC2 >= 0.98 in both channels.
- **B4:** `lens_L2_rg.r1` logic copied unchanged (23 adjacent pairs, 12 inner knots, 5 prompt-grouped folds, rng seed 1).
  Holds if dR2 >= 0.30 in both channels.
- **B5:** attn; cell = (gen, step); d = mean of log g over even layer indices minus mean over odd indices (0-based index as
  captured). Share = fraction of cells with the modal sign. Holds if >= 0.80. MLP reported descriptively.
- **B6:** analyzer `plot_beta_layer` unchanged on the ACE attn rows (pooled over sigma); crossings parsed with
  `lens_L2_rg.curve`. 100 **prompt** bootstraps (rng seeds 3000+k; resample the 29 prompts with replacement, take all gens of
  each draw, relabel gen_idx). Holds if the point estimate has 3 crossings in the + - + - order and 3 is the modal count in
  >= 50% of bootstraps. If B5 holds: even-only and odd-only sequences (layers renumbered consecutively), descriptive.
- **B7 (only if B6 holds):** y3 = slope of the binned curve at the third crossing (`curve`); SE over bootstraps with the same
  pattern. Refuted if y3 +- 2 SE lies wholly outside [-0.80, -0.15].
- **B8:** E3 H3b code unchanged on the cond mean of (log g, log gap, H), clock = logged sigma. Holds if the onset lies in
  [0.73, 0.92] in both channels. With 6 steps only one interval has midpoint < 0.6, so the plateau speed is one interval;
  stated with the result.
- **B9:** `chi_logit` (lens_posthoc_lit.py) with the Amendment-1 `interior_peak` rule; 6 steps <= 8, so edge 1 and tol 1;
  100 prompt bootstraps (rng 20261008). Refuted if a peak passes in either channel.
- **B10:** E6 code unchanged (ANOVA prompt share on the standardised triple), as the archive statement it copies.
  Holds if the steepest rise is the first step interval in both channels. A g-only share is reported as a sensitivity.

## Post hoc (labelled post hoc wherever reported), for the paper tables and figures, as for SD1.5
P1 backbone (999 depth-shift nulls), P2 plateau curve, P3 prompt share by step (naive and ANOVA), P5 ten-way domain decoding
(I-F5 decoder, 99 scrambles, first state and along steps), P7 clock-axis topology (R6 S4 fit3, 1000 prompt bootstraps; with
6 points the fit has at most 2 points per segment and is stated as such). Not run: prompt RSA against the image models
(no shared prompts), tangent/shape comparisons (no partner model), seed predictability.
