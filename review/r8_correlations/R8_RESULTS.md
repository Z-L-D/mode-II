# Round 8: missed correlations — results

Preregistration: `R8_PREREGISTRATION.md`, SHA-256 `acc578ac26cdaf8e58a6f036dfe6defc81b7aae12a1d63e779a512ca5db52916`, frozen 2026-10-09T08:30:59Z.
Computed 2026-10-09 08:31–08:46Z. Only existing captures were used, with the unchanged estimators.

Scripts are in `scripts/`:
- `r8_a_grid.py`: R8-2, R8-8; then `r8_a2_tabulation.py` (the R8-2 step-count tabulation, added to the same JSON).
- `r8_b_fast.py`: R8-1, 3, 4, 5, 6, 9, 10, 11, 13.
- `r8_c_analyzer.py`: R8-7, using the analyzer unchanged.
- `r8_d_mantel.py`: R8-12.

Outputs are in `out/`. Items marked **[tab]** re-tabulate numbers that were already recorded and are not counted as confirmations. Items marked **post hoc** were not in the prereg.

## Scorecard

| Test | Rule (short) | Result | Verdict |
|---|---|---|---|
| R8-1a | rel. onset inside S4 contraction window, ZIT & ZIB MLP | ZIT MLP 0.673 vs [0.833, 0.955]; ZIB MLP 0.887 vs [0.818, 0.972] | **tested, not supported** (the bottleneck stays an envelope description) |
| R8-1b | pairwise ρ ≥ 0.75 under LOO among plateau onset, rel. onset, share stabilisation, decoding stabilisation (7 ch.) | max ρ 0.56 (plateau~rel. onset, p 0.20); no pair passes LOO | **tested, not supported** |
| R8-2a | onset drops ≥ 0.10 on ZIT's 8-state grid in ≥ 3/5 | 0/5; largest drop 0.023 (ZIB MLP) | **not a grid artifact** |
| R8-2b | VR(4)/walk ≥ 1 on the 8-state grid in ≥ 3/5 | 3/5: ZIB attn 0.54→1.13, SD3.5 M 0.61→1.35, SD3.5 L 0.87→1.26 (ZIB MLP 0.88, Mochi 0.95) | **grid artifact**: the ZIT attention VR outlier (1.13) is what an 8-state grid gives |
| R8-2 [tab] | steps vs relative onset | ρ 0.80 over 7 channels; 0.46 without ZIT (native: 0.71 → 0.21) | the ρ≈0.80 that GLM cites is real but is carried by ZIT |
| R8-3a | ρ(log g/g_MP, decoding) ≤ −0.6, all *and* within transformers | all 9: −0.68 (p 0.049, LOO ok); transformers: −0.36 (p 0.44) | **tested, not supported**: the all-channel value is carried by the U-Net |
| R8-3b | \|ρ(log g/g_MP, onset)\| ≥ 0.6 under LOO (11 ch.) | 0.12 (p 0.74). Transformers alone 0.68, LOO ok, p 0.10 (descriptive) | **tested, not supported** |
| R8-3c | \|ρ(log g/g_MP, ΔR²)\| < 0.5 and the tercile ranges overlap | ρ 0.55 (p 0.084); ranges overlap | **tested, not supported** as worded. ΔR² ≥ 0.46 in all 11 channels, but it tends to be larger where g is less anisotropic |
| R8-4a/b | resolution FE adds held-out ΔR² < 0.02 to the spline in 1−PI | spline 0.678 → +FE 0.689; ΔR² **0.011**. The isotropic N-effect would be 0.053 / 0.105 / 0.376 (1024/768/512); the observed median excess is 0.040 / 0.050 / 0.088 | **branch difference explains the resolution trend**; no N/D component at this threshold |
| R8-4c | excess predicts 1−PI with held-out R² ≥ 0.5 at every resolution | 0.640 / 0.617 / 0.632 | **supported**: the pooling excess is a usable proxy for branch divergence |
| R8-5a | early-third tangent − late-third ≥ 0.05, both submodules (archive) | attn 0.969 / 0.961 / 0.863 (+0.106); MLP 0.950 / 0.985 / 0.978 (−0.028) | **tested, not supported** (MLP fails) |
| R8-5b | R8-5a and Mistral late > early | Mistral \|Δlog g\| by third 0.0005 / 0.0018 / 0.0029 (rises with depth) | **tested, not supported** (R8-5a fails) |
| R8-6 [tab] | \|Δ_guidance\| < \|Δ_capture\|/3, both submodules | attn 0.0072 vs 0.0486; MLP 0.0071 vs 0.0246 | holds: the archive tangent drop is the capture functional, not guidance |
| R8-7 | g\* inside vs outside the S4 window (MW) or 3-crossing count (Fisher), p < 0.05 | g\* MW p **0.0036** (inside median 0.012, outside 0.0045); Fisher p 0.40 | **coupled** (by g\*, not by count) |
| R8-8 | VR/walk above vs below onset differs by ≥ 0.2 in the same direction in ≥ 4/5 | 2/5 (SD3.5 M, SD3.5 L); windows per side 1–3 | **tested, not supported** |
| R8-9 [tab] | shape_r vs tangent | ρ 0.74 over 13 pairs (p 0.005); 0.43 over the 10 non-lineage pairs (p 0.22) | the association is driven by the related pairs |
| R8-10 | per-layer CFG4.5/CFG1 seed-spread ratio vs mean 1−PI: ρ ≤ −0.5, p < 0.05 | ρ −0.16 (p 0.45); per-layer ratio 0.27–0.90 | **tested, not supported** |
| R8-11 | descriptive | argmax layer of 1−PI: SD3.5 M layer 9 (steps 0–6), then 5; ZIB attn layer 17 from step 1; ZIB MLP 24 → 12 at step 18 (σ 0.54) | shown, not claimed |
| R8-12 | partial Mantel controlling \|len_i−len_j\|: each unrelated cell keeps p ≤ 0.01 and partial ≥ 0.8 × full | 24/24 cells; min partial/full 0.998; \|r(RDM, length)\| ≤ 0.053 | **survives the length control** |
| R8-13 | tangent at n = 60: median within 0.02 of full, and 5% > 0.24 | attn 0.942 (full 0.942, 5–95% 0.937–0.946); MLP 0.967 (full 0.967, 0.965–0.968) | **n-stable** |

There are 16 preregistered decision rules (excluding [tab] items and the descriptive R8-11):
- Seven came out on the side the reviewers proposed: 2b, 4a/b, 4c, 7, 12, 13, and 2a (the onset is not a grid artifact, which was GLM's question).
- Nine were tested and not supported: 1a, 1b, 3a, 3b, 3c, 5a, 5b, 8, 10.

### Notes per test
- **R8-1.** The full event table is in `out/R8_1_events.csv`. Two caveats on the event estimates:
  - The decoding stabilisation σ rests on the 5 decoded states per channel (I-F5), so its resolution is coarse.
  - For ZIT attn, the prompt share is within 0.02 of its late median from σ = 1, so its stabilisation σ is 1.0.
- **R8-3.** The channel table is in `out/R8_3_channel_table.csv`. Two deviations to note:
  - The U-Net onset is on the t clock (as preregistered), so its scale differs from σ.
  - The U-Net ratio is the C1 median over cells against each block's own g_MP, while the transformers use the lens PH baseline.
- **R8-4.**
  - Cells: (generation, step ≥ 2, layer). There are 54,720 at 1024 px (60 pairs) and 26,448 each at 768 and 512 px (29 pairs). Folds are grouped by prompt.
  - The resolution FE coefficient is +0.021 for 512 px and +0.001 for 768 px (relative to 1024). That is an order of magnitude below the isotropic finite-sample prediction.
  - Logged pooled seq_len: 8192 / 4608 / 2048, so N = 4096 / 2304 / 1024 with D = 1536.
- **R8-5.**
  - Thirds are the diffs inside points 0–7, 8–15 and 16–23. The 7→8 and 15→16 steps are excluded, so the "full" value is the unchanged E10 tangent (0.942 / 0.967 reproduced).
  - In the clean R4 CFG4 cond capture, every third is ≥ 0.984 (attn 0.990 / 0.994 / 0.993; MLP 0.984 / 0.999 / 0.996). The archive attention drop sits in the late third and goes away with the clean capture, which fits R8-6.
  - DeepSeek's thirds (0.995 / 0.920 / 0.874) are the ledger's terminal-state registration, not a tangent.
- **R8-6.** The S4 contraction is nearly invariant across the same captures:
  - MLP: archive −0.497, R4 CFG4 cond −0.505, R4 CFG1 cond −0.557, R3 CFG1 −0.552.
  - Attention: −0.066 / −0.197 / 0.0 / 0.0.
  - Meanwhile the tangent moves 0.94 → 0.99. This fits GLM §6: the contraction is invariant while the tangent is capture-sensitive.
- **R8-7.**
  - The per-step table is in `out/R8_7_per_step.csv`. 9 steps lie inside the window and 16 outside.
  - **Post hoc:** g\* scales with σ (ρ 0.51). After dividing g\* by the per-step median g, inside is still above outside (MW p 0.011; `out/R8_7_per_step_posthoc_normalised.csv`).
  - Also post hoc: the 3-crossing per-step pattern occurs at steps 9–11 (σ 0.84–0.79) and 17–22 (σ 0.59–0.29), i.e. mostly on the plateau.
- **R8-10.** In the CFG1 capture, 9 of 28,800 (prompt, step, layer) cells have zero two-seed spread: the two seeds logged identical g to 6 decimals. Those ratios are left undefined. The per-layer median over all layers is 0.595, against the paper's 0.637, which used the layer-mean definition.
- **R8-12.**
  - The full Mantel values reproduce S2 exactly (e.g. SD3.5 M–L first 0.3078).
  - Prompt lengths are 130–188 characters (median 164), and length carries almost no RDM structure.
  - The lineage and family pairs are also unchanged (ratio ≥ 0.998).
- **R8-13.** The subsets draw n/4 shared prompts and take all of each model's generations for them, 50 draws (rng 64). On the 117 shared prompts the full value is 0.943 / 0.967.

## Fact-check of the reviewers' premises

| Reviewer item | What the archive says | Test → outcome |
|---|---|---|
| DeepSeek §1 pooling excess as observable | The per-cell branch_div values quoted are not in any output. ρ 0.83 is real (Q1f, 1024 px cells) | R8-4c → supported |
| DeepSeek §2 depth-localised inheritance | The Mistral vector is invented. 0.995 / 0.920 / 0.874 is terminal-state registration on pooled ZIB, not a depth-resolved tangent | R8-5 → not supported |
| DeepSeek §3 MLP anisotropy vs tangent | n = 2 | not tested |
| DeepSeek §4 onset vs N/D | N/D is not a model constant; the SD3.5 M resolution sweep confounds N with the schedule shift | not tested; needs a capture |
| DeepSeek §5 tangent residual as guidance sensitivity | The "guidance strength" ordinal is invented. Pooled vs branch is a capture difference, not a guidance level | R8-6 [tab] → capture, not guidance |
| DeepSeek §6 contraction vs tangent | No model carries both | not tested (see the R8-6 note) |
| DeepSeek §7 ZIT VR 1.13 | Real (S3: 1.130) | R8-2b → grid artifact |
| DeepSeek §8 RSA vs encoder | Already in Mode-I §6 (S2e) | not a miss |
| Qwen §1 anisotropy vs decodability | The table mixes attention and MLP values and invents the Mochi value | R8-3a → not supported |
| Qwen §2 onset vs anisotropy | | R8-3b → not supported |
| Qwen §3 PI depth profile | | R8-11 descriptive |
| Qwen §4 VR along σ | | R8-8 → not supported |
| Qwen §5 clock × depth | | R8-7 → coupled (g\*) |
| Qwen §6 prompt share vs seed predictability | | R8-1b → not aligned |
| Qwen §7 H–gap under guidance | No prediction offered | not tested |
| Qwen §8 decoding derivative vs onset | | R8-1b → not aligned |
| Qwen §9 contraction vs PI | | R8-10 → not supported |
| Qwen §10 parity from alternating submodules | Attention and MLP are already analysed separately; each is +−+− | not a miss |
| GLM §1 σ-event alignment | | R8-1 → not aligned |
| GLM §2 steps vs onset, subsampling | ρ 0.80 is correct | R8-2a → not a grid artifact; R8-2 [tab]: carried by ZIT |
| GLM §3 anisotropy vs decoding | | R8-3a → not supported |
| GLM §4 pooling increment vs N/D | The arithmetic (D 1536; N 1024/2304/4096; pooled 2N) checks out | R8-4 → branch difference, ΔR² 0.011 |
| GLM §5 non-autonomy robust to anisotropy | | R8-3c → not supported as worded (ΔR² > 0.46 everywhere, trend ρ 0.55) |
| GLM §6 contraction invariant vs tangent sensitive | | R8-6 note → consistent |
| GLM §7 shape vs tangent | | R8-9 [tab] → 0.74 / 0.43 |
| GLM §8 depth-resolved tangent | | R8-5a → not supported |
| GLM §9 incomplete intervention grid | A design point for Mode-III, not a correlation | noted |
| GLM §10 RSA length control; tangent n | | R8-12 → survives; R8-13 → n-stable |

## What could go into the papers (not applied; awaiting Zac)
1. **Mode-I, seed-fate / VR paragraph:** ZIT's VR(4)/walk of 1.13 is reproduced by subsampling 25–40-step models to its 8 states (3 of 5 channels exceed 1). The relative onset does not move under subsampling (largest drop 0.023).
2. **Mode-I, pooling section:**
   - At fixed resolution, the pooled-over-branch-mean excess predicts 1−PI with held-out R² 0.62–0.64.
   - Resolution adds ΔR² 0.011, against an isotropic finite-sample effect of 0.05–0.38. The 512/768/1024 trend is branch difference, not N/D.
3. **Mode-I, robustness of lineage results:**
   - The lineage tangent is n-stable down to 60 generations (5–95% 0.937–0.946 attn).
   - The cross-model RSA is unchanged when controlling for prompt length (partial/full ≥ 0.998).
   - The archive tangent drop is a capture effect: Δ_guidance 0.007 vs Δ_capture 0.025–0.049, and in attention it sits in the deepest third.
4. **Mode-I §7.2 (layer axis):** per-step g\* is higher inside the MLP contraction window (p 0.004; 0.011 after normalising by per-step g, post hoc). The three-crossing per-step pattern sits mostly on the plateau.
5. **Negative results worth one line each:**
   - Anisotropy does not predict decoding, onset or ΔR² within the transformers.
   - The σ-events do not align across channels.
   - Guidance contraction is not strongest in the most divergent layers.
