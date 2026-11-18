# R3 Part C (UNet arm): scoring plan, frozen before any outcome is computed

Written 2026-10-08 around 21:55 UTC. The SHA-256 and UTC time are in `R3C_SCORING_PLAN_SHA256.txt`.

**What has been looked at so far:** the file list, the CSV header, the first 8 data rows of one file (gen 1, step 0, blocks 0–1), and the (seq_len, hidden_dim) of each block. No g statistic, ratio, R² or curve has been computed.

**Why this file exists:** the predictions C1–C3 were frozen in `review/r3_prereg/R3_PREREGISTRATION.md` (SHA efa613eb, 15:27Z). That file does not fix the cell definition for C1, which branch is used, or exactly which model comparison C3 means. Those choices are fixed here, before scoring.

## Data

- **Source:** 10 CSVs, `SD1.5-attn.mlp-TE1-CFG7.5-R512-S(RAND)-STUDY[PRIMARY-UNET]-G480-*.csv`, unpacked into `review/r3_unet/data/`. The run is SD1.5, CFG 7.5, 512 px, 50 steps and random seeds, captured with R3 nodes.py and the `UNetModel` entry.
- **Blocks:** 16 transformer blocks in forward order, with these (N tokens, D) per level:
  - down: (4096, 320) ×2, (1024, 640) ×2, (256, 1280) ×2
  - middle: (64, 1280)
  - up: (256, 1280) ×3, (1024, 640) ×3, (4096, 320) ×3
- **Channels:** attn (`attn1.to_out.0`, self-attention output) and MLP (`ff.net.2`).
- **Gens:** only complete ones are kept: 50 steps × 16 blocks × 2 submodules × 2 branches. Gens are keyed by (file, gen_idx).
- **Primary branch:** **cond**. This is the closest match to the archive's cond-only regime. Uncond is reported as a sensitivity check and is never the verdict.
- **Prompt groups:** prompt_text, propagated within each gen. It is only written on the first row of a gen.

## C1. Anisotropy against each block's own isotropic baseline

- **g_MP(N, D):** the mean over 20 draws of g for a white Gaussian X (N×D), computed as in `lens_posthoc_isotropy.py`: singular values of X/√N, then g = (Σs)²/(D Σs²). Same g definition as the node.
- **Cell:** (step, block) within a channel, giving 50 × 16 = 800 cells per channel. The cell value is the median over gens of cond-branch g, divided by g_MP at that block's (N, D).
- **Verdict:**
  - C1 holds if, in each channel, at most 5% of cells have a ratio ≥ 0.10.
  - It is refuted if more than 5% of cells reach ≥ 0.10 in either channel.
  - Also reported: the same fraction over all rows, which is descriptive.

## C2. R1 autonomy fails

- **Method:** exactly `lens_L2_rg.r1()`, applied to the array g[gen, step, block] (cond).
  - β_L = g_{L+1} − g_L over all 15 adjacent block pairs, level boundaries included.
  - f(g) is a cubic B-spline with 12 inner quantile knots.
  - 5-fold cross-validation grouped by prompt (fold assignment rng seed 1).
  - ΔR² = R²(f(g) + pair fixed effect) − R²(f(g)).
- **Verdict:** C2 holds if ΔR² ≥ 0.30 in both channels.

## C3. Depth effect beyond the U-Net level

- **Pairs used:** only within-level pairs, i.e. adjacent blocks with the same (N, D) at the same resolution stage. There are 9: (0,1), (2,3), (4,5), (7,8), (8,9), (10,11), (11,12), (13,14), (14,15).
- **Statistic:** ΔR²_within = R²(f(g) + pair FE) − R²(f(g) + level FE), using the same spline, folds and cross-validation as C2.
  - Levels are down-4096, down-1024, down-256, up-256, up-1024, up-4096.
  - A level with a single pair is fully absorbed by its level effect.
- **Verdict:** C3 holds if ΔR²_within ≥ 0.15 in both channels.
- **Sensitivity (not the verdict):** R²(f(g) + pair FE) − R²(f(g)) on the same within-level pairs.

## Post hoc analyses (not preregistered; labelled post hoc wherever reported)

Each one uses an archive method unchanged, with cond rows. Where the archive uses a rectified-flow σ ∈ [0, 1], SD1.5's sigma is mapped to the same signal-to-noise ratio: t = σ/(1+σ).
- **P1. Backbone.** PC1 share and exp-entropy effective rank of the states × (block, {log g, log gap, H}) matrix. Each feature is z-scored across depth at each state. The null is 999 independent circular depth shifts per state. The implementation is first checked on SD3.5 Medium and Z-Image Base from the archive parquet, and is used only if it reproduces their ledger values within 0.05.
- **P2. Plateau onset.** `e3_sigma_registration.py` H3b code on the t clock.
- **P3. Prompt share by step.** `e6_varcomp.py`, ANOVA share.
- **P4. Triple dimension.** `lens_L1_triple.pca_shares` on (H₁/lnD, H₂/lnD, ln gap) over rows.

Not run, as the prereg states: the layer-axis β(g) crossing count, because 29-bin pooling across UNet levels with different D and N is not meaningful.
