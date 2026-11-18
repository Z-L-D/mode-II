# R3 Part C: SD1.5 U-Net arm, results (2026-10-08)

## Data

- **Source:** Zac's upload at 21:42Z, `SD1.5-attn.mlp-TE1-CFG7.5-R512-S(RAND)-STUDY[PRIMARY-UNET]-G480.zip`, unpacked to `data/`.
- **Size:** 10 domain CSVs, 476 complete generations, 119 prompts with about 4 random seeds each. 116 of those prompts are shared with every archive model.
- **Settings:** 50 steps, CFG 7.5, 512 px.
- **Instrument:** R3 `nodes.py` (after the 17:02Z phi fix), `UNetModel` registry entry, 16 transformer blocks, attn (`attn1.to_out.0`) and MLP (`ff.net.2`) hooks, explicit cond and uncond rows (`passive-branch`).
- **Data quality:** one cond attn value is missing (gen weather_atmosphere:46, step 27, block 10). phi is blank exactly at the five width boundaries, as designed.

**Order of work:**
1. Predictions C1–C3 were frozen at 15:27Z (`review/r3_prereg/`, SHA efa613eb).
2. The scoring plan was frozen at 21:46:11Z (`R3C_SCORING_PLAN.md`, SHA 7a3b852f). Its prose says "around 21:55"; the SHA file holds the true time.
3. The P5–P6 addendum was written at 21:58:36Z.

Unless stated otherwise, every number below uses the cond branch.

## Preregistered (C1–C3)

| ID | Prediction | Result (attn / MLP) | Verdict |
|---|---|---|---|
| C1 | Per-block g / g_MP < 0.10, with each block judged against its own isotropic value at (N, D); refuted if ≥ 0.10 in more than 5% of (step, block) cells | Cells ≥ 0.10: **30.8% / 94.6%**. Median cell ratio 0.077 / 0.262; max 0.22 / 0.59. Uncond: 32.5% / 93.9% | **Refuted** in both channels |
| C2 | R1 autonomy fails as in the DiTs: ΔR² ≥ 0.30 for g + block over g alone | R²(g) 0.27 / 0.24; R²(g + block) 0.73 / 0.85; **ΔR² 0.46 / 0.61**. Uncond 0.47 / 0.61. DiTs: 0.49–0.65 | **Confirmed** in both channels |
| C3 | Depth effect beyond U-Net level: within-level ΔR² ≥ 0.15 | ΔR²(block over level) **0.22 / 0.06**. R² for g, g + level, g + block: attn 0.25, 0.34, 0.56; MLP 0.05, 0.76, 0.83 | **Confirmed for attn, refuted for MLP** |

Score: 2 confirmed (C2, C3 attn) and 2 refuted (C1, C3 MLP), counting C3 per channel. Counting each ID once: C2 confirmed, C1 refuted, C3 split.

### What it means

- **C1.** The DiT archive has g at 0.1–7% of its isotropic value (median ratios: attn 0.018–0.071, MLP 0.0014–0.0019). SD1.5 attention lies at the edge of that range (median 0.077). The SD1.5 MLP is about 100 times less anisotropic than the DiT MLPs (median 0.26). So strong anisotropy of the uncentred second moment is not a fixed property of denoisers. It depends on the architecture, and most of all on the MLP. The analysis cannot say why. One candidate is that SD1.5 lacks the few massive activation channels that dominate uncentred DiT spectra (the papers already list this as an open robustness test). That candidate is untested.
- **C2.** The depth finite difference β_layer is not a function of g in a U-Net either. It is as non-autonomous as in every DiT channel.
- **C3.** In attention, block identity carries depth information beyond the level's width and token count. In the MLP, the level (D, N) explains almost all of what depth explains (R² 0.76 of 0.83).

The prereg's consequence table names no U-Net outcomes, so nothing else changes automatically.

## Post hoc (labelled; not preregistered)

| ID | Analysis (archive method unchanged) | SD1.5 attn / MLP | Archive |
|---|---|---|---|
| P1 | Backbone PC1 and exp-entropy effective rank; null of 999 depth circular shifts | PC1 **0.897 / 0.615**; rank 1.46 / 2.24; null means 0.22 / 0.19 and 11.1 / 12.1; p = 0.001 in both | PC1 0.71–0.90, rank 1.5–2.5. The implementation reproduces SD3.5 M (0.904 vs 0.897) and ZIB (0.676 vs 0.710) |
| P2 | Plateau onset (E3 H3b) on the SNR-matched clock t = σ/(1+σ) | **0.58 / 0.67**; with the SD3 resolution shift (a = 2 for 512 px), 0.73 / 0.80 | 0.78–0.87 at 1024 px; SD3.5 M at 512 px is 0.72, or 0.84 resolution-corrected |
| P3 | Prompt share of variance by step (E6 ANOVA) | 0.56 → max 0.67 / 0.63 → 0.66; steepest rise in the first interval | 0.36–0.95 |
| P4 | Triple PCA, PC1 + PC2 | **0.995 / 0.987** | 0.980–0.994 |
| P5 | Ten-way domain decoding at the first state (I-F5 decoder) | Ordered profile **0.38 / 0.42**; layer mean 0.19 / 0.18; prompt length 0.13; TF-IDF on the text 0.40; 99 scrambles max 0.17 / 0.19 (p = 0.01). Along steps: 0.36–0.39 / 0.30–0.42 | Ordered 0.45–0.61, above TF-IDF 0.38–0.42 |
| P6 | Prompt RSA with Mantel tests (R6 S2), 116 shared prompts | SD1.5 attn vs the five DiTs: **0.11–0.35** (MLP 0.12–0.32); every p = 0.0005 | Recomputed on the same 116 prompts: unrelated DiT pairs 0.14–0.43; Base↔Turbo 0.56–0.62 |

**Reading:**
- **Holds in the U-Net:** a low-dimensional backbone that beats the depth-shift null, a 2-D triple, the non-autonomous depth step, ordered-depth content decoding that beats the depth scramble, and prompt-relational geometry shared with the transformer models. The RSA values sit at the low end of the unrelated-DiT range.
- **Weaker or different:**
  - Domain decoding reaches only text-decoder level, where the DiTs exceed it.
  - The plateau onset is later on the SNR-matched clock.
  - The resolution correction moves the MLP onset into the archive band and leaves attention below it.
  - The onset depends on how SD1.5's VP/ε schedule is mapped. Its first recorded t is 0.936, so the σ ≈ 1 transient is truncated.
  - Anisotropy (C1) differs most.

## Post hoc, added at 22:45Z for the figure integration (R3C_ADDENDUM_P7.md, SHA 6251b69f, frozen 22:45:30Z before running)

| ID | Analysis | SD1.5 attn / MLP | Archive |
|---|---|---|---|
| P7 | Clock-axis topology of the layer-mean g (R6 S4 fit3, 1000 prompt bootstraps) on t, raw σ_VP and the resolution-shifted t′ | **+++ / +++** in 100% of bootstraps on every clock. No contraction; the MLP ends 0.6% below its maximum | Attention ++− in all five DiTs; Z-Image MLP +−+ |
| P3b | Naive between-prompt share (E6), for Fig. II-F5c | 0.67–0.75 / 0.72–0.74 | 0.51–0.96 |

Outputs: `out/P7_clock_topology.csv`, `out/P7_clock_curves.csv`, `out/P3_prompt_share_by_step_naive.csv`. Scripts: `scripts/u4_clock_topology.py`, `scripts/u5_naive_share.py`. Ledger v0.26.

## Not run

- Layer-axis β(g) crossings: per the prereg, pooling across levels with different (N, D) is not meaningful.
- Seed predictability.
- Tangent and shape similarity: there is no lineage partner.

## Files

- **Scripts:**
  - `scripts/u0_load.py` builds the cache.
  - `scripts/u1_score_C.py` scores C1–C3.
  - `scripts/u2_posthoc.py` runs P1–P4.
  - `scripts/u3_content.py` runs P5–P6.
- **Outputs:**
  - `out/C_scores.csv` and `out/C1_cells.csv`;
  - `out/P1_backbone.csv`, `P2_plateau_onset.csv`, `P3_*.csv`, `P4_triple_pca.csv`, `P5_decoding.csv`, `P6_rsa.csv`;
  - logs `out/u*.log`.
