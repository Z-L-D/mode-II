# Round 8: missed correlations (preregistration)

**Source of the ideas:** three external "missed correlations" readings that Zac posted on 2026-10-09 at 08:27Z (DeepSeek v4.1, Qwen 3.8 Max, GLM 5.3). This file fixes, before any new number is computed, which of their suggestions are tested, how, and what result counts as support. Tests use only existing captures and the unchanged estimators of Rounds 2–6, R3 and the lens round.

**What was already seen.** Recorded outputs were in view when this file was written: Table 5 relative onsets, E3 plateau onsets, E10 tangents and shapes, Q1/Q2 values, S4 breakpoints for ZIB, and the per-channel numbers of the papers. Tests that only re-tabulate those numbers are marked **[tabulation]**. They are reported, but not counted as preregistered confirmations. Every other test produces numbers that did not exist before this file.

**Global rules**
- n is small in every cross-channel test. A cross-channel association is called *supported* only if it meets the stated rule *and* survives leaving out each channel in turn (the sign is kept and |ρ| ≥ 0.5 in every leave-one-out).
- Two-sided permutation p-values use 10,000 permutations, rng seed 2026100908.
- Bootstraps over prompts use 1,000 resamples, rng 64, unless a reused script fixes its own seed.
- Wording: "tested, not supported", never "excluded". No separatrix, RG or geometry→quality wording.

## Claims not tested here, and why
- **DeepSeek §3 (MLP anisotropy vs. tangent conservation):** n = 2 (one lineage, two submodules). A correlation cannot be estimated.
- **DeepSeek §6 (seed contraction vs. stage-wise tangent):** the contraction is measured on SD3.5 M and the tangent on the Z-Image lineage. No model carries both.
- **DeepSeek §8 (RSA vs. text encoder):** already stated in Mode-I §6 ("shared text encoders do not explain it", S2e). Not a miss.
- **Qwen §10 (+−+− from alternating submodules):** attention and MLP are hooked and analysed separately. The pattern is reported per submodule (ZIB attn and MLP each +−+−). Not a miss.
- **Qwen §7 (H–gap fingerprint under guidance):** no prediction was offered and no decision rule can be stated. Left for a later round.
- **DeepSeek §4 (onset vs N/D across models):** N/D is not a model-level constant (the U-Net has 4 levels). The only within-model N sweep (SD3.5 M at 512/768/1024 px) changes N and the resolution-dependent schedule shift together. Not separable with existing captures, so it is listed as a needed capture.

## Tests

### R8-1 Alignment of σ-events (GLM §1, Qwen §6, Qwen §8)
For each of the 7 archive channels:
- (i) plateau onset (E3, recorded);
- (ii) relative seed-predictability onset (S3 matched-60 median, recorded);
- (iii) prompt-share stabilisation σ: the largest σ after which the E6 naive prompt share stays within 0.02 of its median over σ < 0.6 (new);
- (iv) decoding stabilisation σ: the largest σ after which the ordered-profile decoding accuracy (I-F5 along-steps data) stays within 0.05 of its median over σ < 0.6 (new).
- Z-Image MLP also has (v): the S4 contraction window [break2, break1] (recorded for ZIB; computed for ZIT MLP with the S4 code unchanged).

Rules:
- **R8-1a:** ZIT MLP and ZIB MLP relative onsets both lie inside their own contraction windows → "aligned". Otherwise the E→C→C bottleneck stays an envelope description.
- **R8-1b:** pairwise Spearman among (i)–(iv) over the channels. A pair is "aligned across channels" if ρ ≥ 0.75 under the global LOO rule.

### R8-2 Step grid vs. relative onset and VR(4) (GLM §2, DeepSeek §7)
**[tabulation]** Spearman of steps vs relative onset over the 7 channels, with and without ZIT.

New test: subsample ZIB attn, ZIB MLP, Mochi attn, SD3.5 M attn and SD3.5 L attn to the 8 states whose σ are nearest ZIT's 8 σ values (no repeats). Then recompute, with the S3 code unchanged:
- the matched-60 relative onset (50 subsets);
- VR(4)/walk (walk reference re-simulated at T = 8).

Rules:
- **R8-2a:** onset is a grid artifact if the subsampled onset falls by ≥ 0.10 in ≥ 3 of 5 channels.
- **R8-2b:** the ZIT attention VR outlier is a grid artifact if subsampled VR(4)/walk ≥ 1.0 in ≥ 3 of 5 channels.

### R8-3 Anisotropy against the other channel-level quantities (Qwen §1–2, GLM §3, GLM §5)
Channel table:
- median g/g_MP (lens PH_isotropy_baseline; U-Net C1 medians; ACE B2 medians);
- first-state ordered decoding (Table 3; ACE excluded as uninformative);
- plateau onset (E3; U-Net on the t clock; ACE B8);
- R1 ΔR² (lens L2; U-Net C2; ACE B4).

Spearman of log(g/g_MP) against each quantity:
- **R8-3a:** decoding. Reviewers predict ρ < 0, i.e. more anisotropic means better decoding. Supported only if ρ ≤ −0.6 over all channels with decoding *and* ρ ≤ −0.6 within the 7 transformer channels.
- **R8-3b:** onset. Associated if |ρ| ≥ 0.6 under LOO.
- **R8-3c:** ΔR². GLM predicts no association. "Non-autonomy is robust to anisotropy" if |ρ| < 0.5 and the ΔR² range in the least anisotropic third overlaps that of the most anisotropic third.

### R8-4 Pooling excess: branch difference vs. N/D (GLM §4, DeepSeek §1)
Use the paired SD3.5 M CFG 4.5 trajectories at 512, 768 and 1024 px (Q1c/Q1d pairing code unchanged). Per (generation, step ≥ 2, layer) cell:
- excess e = log g_pool − log g_BM;
- d = 1 − PI (logged prompt influence).

Fit e on a 5-knot spline in d. Then add resolution fixed effects. Held-out ΔR² uses 5-fold CV grouped by prompt.

Rules:
- **R8-4a:** branch difference explains the resolution trend if the resolution FE adds ΔR² < 0.02.
- **R8-4b:** an N/D (finite-sample) component exists if it adds ≥ 0.05. Between the two thresholds, "partly".
- Reference: isotropic g_MP(2N, D) − g_MP(N, D) in log, at each resolution's logged seq_len and D = 1536 (20 Gaussian draws).
- **R8-4c [reframing test, DeepSeek §1]:** excess is a usable proxy for 1 − PI if, at fixed resolution, held-out R² of 1 − PI from excess (spline) is ≥ 0.5 at all three resolutions.

### R8-5 Depth-localised deformation (DeepSeek §2, GLM §8)
- **(a)** ZIB↔ZIT tangent cosine (E10 `tangent`, unchanged) restricted to each depth third of the 24-point normalised grid (points 0–7, 8–15, 16–23), for the archive attn and MLP. For the clean ZIB CFG 4 cond capture vs ZIT, where the cube exists.
- **(b)** Mistral abliteration: mean |Δlog g| per depth third from E8_ablit_prefill_by_layer.csv **[tabulation]**.

Rules:
- **R8-5a:** "dynamical inheritance is depth-localised" if the early-third tangent exceeds the late third by ≥ 0.05 in both submodules (archive).
- **R8-5b:** "one depth law across interventions" only if R8-5a holds *and* Mistral's late third exceeds its early third.

### R8-6 Capture vs. guidance in the lineage tangent (DeepSeek §5, GLM §6) [tabulation]
From Q2 and E10:
- Δ_guidance = tangent(ZIB CFG 1 cond, ZIT) − tangent(ZIB CFG 4 cond, ZIT);
- Δ_capture = tangent(ZIB CFG 4 cond, ZIT) − tangent(ZIB archive pooled CFG 4, ZIT).

Rule: "the drop is the capture functional, not guidance" if |Δ_guidance| < |Δ_capture| / 3 in both submodules. The S4 contraction depth across the same captures is tabulated beside it (GLM §6).

### R8-7 Clock axis × depth axis (Qwen §5)
ZIB MLP archive (25 steps): per-step layer-axis β with the analyzer's `plot_beta_layer` (r6c replay code, unchanged). Record the crossing count and the first +→− crossing g* per step.

Rule: "coupled" if g* inside the S4 contraction window differs from g* outside it (Mann–Whitney p < 0.05) *or* the crossing-count distribution differs (Fisher exact on 3 vs not-3, p < 0.05).

### R8-8 VR(4) along σ (Qwen §4)
For channels with ≥ 25 steps, compute VR(4)/walk in sliding windows of 9 states (step 4). Walk reference re-simulated at T = 9. Descriptive.

Rule: "persistence changes at the plateau onset" if the mean VR/walk of windows wholly above the onset differs from that of windows wholly below by ≥ 0.2 in the same direction in ≥ 4 of 5 channels.

### R8-9 Shape vs. tangent across pairs (GLM §7) [tabulation]
Spearman of shape_r vs tangent_cos over the 13 E10 pairs, and over the 10 unrelated/family pairs.

### R8-10 Guidance contraction vs. branch divergence by layer (Qwen §9)
SD3.5 M 1024 px R4 captures, seeds 1001/1002 shared across CFG 1 and 4.5. Per layer:
- the across-seed spread ratio CFG 4.5 / CFG 1 (cond branch; same estimator as the paper's 0.637 paired ratio, steps in the last three quarters);
- mean (1 − PI) at CFG 4.5 over the same steps.

Spearman across layers. Rule: "contraction is strongest where the branches diverge most" if ρ ≤ −0.5 (n = 24 layers, permutation p < 0.05).

### R8-11 Depth profile of prompt influence (Qwen §3)
Descriptive. Layer × σ map of 1 − PI for SD3.5 M CFG 4.5 (1024 px) and ZIB CFG 4 (R4). Report the layer of maximal divergence per step and whether it moves along σ (Spearman of argmax layer vs step). No decision rule. Shown, not claimed.

### R8-12 Prompt-length control for the shared RSA (GLM §10)
Partial Mantel (S2 code) of each model pair's prompt-dissimilarity matrices controlling for |len_i − len_j| (characters). First, middle and last step.

Rule: the shared structure "survives the length control" if every unrelated pair keeps p ≤ 0.01 and its partial ρ is ≥ 0.8 × the full ρ.

### R8-13 Sample-size dependence of the lineage tangent (GLM §10)
ZIB↔ZIT archive tangent (attn, MLP) recomputed on random subsets of n = 60, 120, 240 generations from each model (50 draws each; prompts matched).

Rule: "n-stable" if the median at n = 60 is within 0.02 of the full-n value and the 5–95% band at n = 60 stays above the largest unrelated-pair tangent (0.24).

## Outputs
`review/r8_correlations/out/R8_*.csv|json`, `R8_RESULTS.md`, ledger v0.28 section `r8-correlations`. Paper changes, if any, only after Zac sees the results.
