# Mode Series Round 2: preregistered predictions

Frozen 2026-10-05 ~11:10 UTC, before any Round-2 calculation was run. The SHA-256 of this file is recorded in `R2_PREREG_SHA256.txt` at freeze time. Amendments made after the literature pass go in §B. They are dated and carry their own hash, and they are applied only before the corresponding calculation runs.

**Data:**
- TELEMETRY.zip (SHA d7f5a82e…b391).
- `data/sd35m_cfg1.0_vs_5.5/`.
- Prior ZIT step-count CSVs from Drive `cos-dm.zip` (z-image-turbo-3/4/5/7/8).

Unit conventions follow Round 1: complete generations only; A = per-layer [log g, log gap, H]; σ is the exported sigma.

**Scoring words:** CONFIRMED = all stated criteria met. PARTIAL = direction right but a criterion missed. REFUTED = the stated falsifier was met.

---

## A. Predictions

### P1 — Synchronization redefined (E1)

H1a. Round 1 found that R tracks −mean|Z|² with r ≥ 0.85 in all 7 channels. Under a null that preserves each layer's monotone drift but randomizes its timing, the observed R_max will **not** exceed the null 95th percentile in at least 5 of 7 channels.
- Null: per layer, a random monotone time-warp of its own trajectory, i.e. the trajectory resampled on a random increasing map of [0,1] onto itself; 999 surrogates.

H1b. Two centering-free order parameters will **not** show an interior maximum (peak at normalized τ in [0.2, 0.8] that exceeds both endpoints' values by more than the bootstrap SE) in at least 5 of 7 channels:
- K(t), increment coherence (Round 1 definition).
- C(t), the mean pairwise cross-layer correlation of the per-layer increment vectors across generations at each step.

H1c. Real cross-layer co-movement exists. Integrated K exceeds its per-layer circular-shift null mean in 7 of 7 channels.

Falsifier for the Round-1 critique: R_max exceeds the monotone-warp null 95th percentile in 5 or more channels, **and** K or C shows an interior maximum in 5 or more channels.

### P2 — Capture regime effects (E2)

H2a. In SD35L, where pooled (8192) and branch-mean (4096) regimes occur at the same steps, the regime shifts layer-mean log g by a nonzero amount after matching domain and step. Predicted |Δ log g| ≥ 0.05, i.e. pooled g higher, as in SD35M step 1 (+24%). Gap and H shift in opposite directions to each other.

H2b. Domain decoding at matched steps within regime-homogeneous subsets stays ≥ 3× chance in SD35L and SD35M. Round 1 found 0.46–0.52 for SD35L.

H2c. Shape r (depth-normalized profile) between regimes at matched σ is ≥ 0.98. The regime moves A more than M.

### P3 — σ-indexed registration (E3)

H3a. For every prompt among the 105 shared ZIT/ZIB prompts, the equal-σ map loses ≤ 0.01 in mean registered r against that prompt's optimized monotone warp in at least 90% of prompts, for both attn and mlp.

H3b. The onset of the transient→plateau boundary, defined as the first σ where per-σ speed falls below 2× its plateau median, will cluster more tightly in log-SNR (λ = log((1−σ)²/σ²)) than in normalized step index across the 7 channels. Predicted SD ratio (log-SNR SD / step-fraction SD, both rescaled to [0,1] ranges) < 1. Low confidence.

### P4 — Step-count invariance at matched σ (E4)

H4a. Clean test: ZIT 8-step vs the 4 auxiliary ZIT 100-step generations in TELEMETRY.zip (same model, CFG 1.0, cond-only). Also the prior ZIT 8/16/32-step CSVs. At matched σ, the median absolute relative deviation of population-mean per-layer g is ≤ 5%, and gap and H are ≤ 3%.

H4b. Mochi 30-step CFG4.5 vs 100-step CFG1.0 differs in CFG and functional, so it is **not** a clean test. Per-σ speed curves nonetheless agree within 25% in every σ band below 0.9, as seen in Round 1.

### P5 — Seed-fate vs a diffusive baseline (E5)

H5a. Round 1's seed-fate R²(t) curves rise smoothly. Against a diffusive surrogate (per-step residual increments shuffled across generations within step, which preserves increment covariance and breaks per-generation persistence), the observed R²(t) **exceeds** the surrogate at mid-run (τ≈0.5) in at least 5 of 7 channels. Fate is more persistent than a random walk.

H5b. The excess has **no** sharp jump: max single-step increase of observed R² ≤ 0.25 in all channels. Round 1's largest was SD35M at 0.01→0.48 at step 1, which is the σ=1 transient, so H5b is evaluated excluding the first transition.

H5c. Bootstrap (prompt-cluster) 95% CI of the τ at which R² crosses 0.5 has width ≤ 0.25 in at least 5 of 7 channels.

### P6 — Within-model variance components by step (E6)

H6a. Prompt share of within-(model, step) variance is ≥ 0.5 at every step in every channel except possibly σ=1 for ZIB and the SD3.5 models. Round 1 found 0.51–0.96.

H6b. Seed share (1 − prompt share) is non-monotone in time in at least 3 channels. This repeats the ledger's "no universal seed collapse".

### P7 — Context-invariant LLM observable (E7)

H7a. PR/n has a much weaker monotone dependence on n than g. Median within-generation |Spearman ρ(n, PR/n)| ≤ 0.9 in every LLM, against 1.000 for g. Direction: decreasing.

H7b. A Marchenko–Pastur-referenced observable g̃ = PR / PR_MP(n, D) still decreases with n. This is low confidence, because PR_MP for white data is ≈ nD/(n+D) ≈ n when n ≪ D, making g̃ ≈ PR/n.

H7c. On the context-invariant observable, Qwen size-pair depth-derivative correlations at matched n stay low (|r| ≤ 0.4 for 1.5B↔7B and 3B↔7B). The ledger's "capacity breaks microscopic law" survives the context fix.

H7d. The Qwen β-ladder (3 crossings) computed with the PR/n observable loses at least one crossing or shifts ratios by more than 20%.

### P8 — Mistral ablation weight-only effect (E8)

H8a. On identical-text prefixes (prefill, plus generated tokens before the first divergence), median |Δ log g| between Instruct and ablit is ≤ 0.005. The ablation's direct geometric effect is tiny.

H8b. The per-layer profile of that effect is not flat. Effect is concentrated in the middle-to-late layers where abliteration directions are usually computed: argmax layer in the [0.3, 0.9] depth fraction.

### P9 — Llama-Base sector analysis (E9)

H9a. In Llama-3.1-8B-Base (dual hook), adding cross-type φ (stride-1) to [g, gap, H] improves held-out next-layer β_g prediction more than adding same-type φ_skip, in both attn and mlp channels. This replicates Z-Image's "cross-sector dominates". Moderate confidence.

### P10 — Unrelated-pair calibration (E10)

H10a. Tangent (depth-derivative of the normalized profile, matched σ) cosine for ZIT↔ZIB is ≥ 0.8 for both submodules. All unrelated diffusion pairs at matched σ and interpolated depth have |cosine| ≤ 0.3.

H10b. Shape r of unrelated pairs ranges up to ~0.8 (Round 1), so derivative alignment separates lineage from generic structure better than shape r. Predicted ZIT↔ZIB tangent minus best unrelated tangent ≥ 0.5, against ≤ 0.2 for shape r.

### P11 — φ fields and the ZIT step-flow loop (E11)

H11a. Preregistered loop criterion: plot population-mean (g, β_step) for each layer over steps, where β_step = g(t+1) − g(t). A loop is present if the trajectory has at least 2 sign changes of β_step and a signed-area winding with |winding| ≥ 0.5 turns. The prior S246 note reported the ZIT step-flow loop in 87/87 generations. Prediction: in this archive's ZIT at 8 steps, ≥ 50% of layers meet the criterion for at least one submodule. Other channels show it in < 50% of layers.

### P12 — β_layer(g) topology beyond one prompt (E12)

H12a. Using the frozen 24-bin estimator, the 120-prompt SD35M CFG4.5 campaign shows ≥ 3 zero crossings of binned β_layer(g), against 4 in the single-prompt resolution suite. The crossing g values differ by > 20% from the res-suite values, because the regime differs (CFG and branch-mean).

### P13 — CFG 1.0 vs 5.5 study re-run (E13)

H13a. TRI (terminal rebound index, ledger definition) in steps 0–9 reproduces the ledger's 0.866→0.874 within ±0.05.

H13b. The rebound-region complex eigenpair is present under both CFGs, with modulus within ±0.05 of the ledger (0.878 / 0.848).

H13c. Branch averaging alone predicts reduced generation-to-generation spread at CFG 5.5. Estimate: averaging two equally variable and weakly correlated branches gives a spread ratio ≥ 1/√2. The observed 4.6× reduction (Round 1) exceeds this, so it is **not** fully explained by averaging (ratio < 0.5).

---

## B. Amendments (post literature pass, pre-calculation)

Amended 2026-10-05 ~11:17 UTC, after the literature pass (`R2_LITERATURE.md`). E4, E6, E12 and E13 had already run when these were written, and nothing below touches them. E1, E2, E3, E5 and E7–E11 had not run. No prediction's direction or threshold is changed; the amendments fix formulas, add context or add unscored analyses.

- **B1 (E7, H7b).** The null-model PR for an iid Gaussian n×D matrix is exactly PR₀ = nD/(n+D) to leading order (Wishart moments; Chun et al. 2026), so g₀ = n/(n+D). H7b is evaluated on g/g₀. The prediction is unchanged (g/g₀ still decreases with n; low confidence).
- **B2 (E8, H8b).** Arditi et al. (2024) project the refusal direction out of every residual-writing matrix in every layer, so a layer-concentrated weight effect is not expected from the edit itself. H8b stays as written; confidence is lowered from moderate to low.
- **B3 (E1).** The literature has no direct statement of the z-scoring artifact. The agent's simulation puts the dispersion minimum anywhere in τ 0.2–0.8 depending on curve shape. The per-layer monotone-warp null keeps each layer's own shape, so no change.
- **B4 (E10).** Unrelated-model depth-profile similarity of about 0.8 is an expected baseline (Murphy et al. 2024; Cheng et al. 2025). Add an unscored layer-order permutation null for the tangent cosine, and report each pair's percentile.
- **B5 (E3/E5, exploratory, unscored).** Report the σ and log-SNR at which seed-fate R² crosses 0.5 and at which the speed plateau begins. Compare them descriptively with the speciation window of Biroli et al. (2024). Latent scaling and timestep shift differ, so no numeric prediction is made.
- **B6 (data note, post hoc).** While running E12, the legacy-schema files were found to come from a different capture path: SD3.5 res-suite, the CFG 1.0/5.5 study and the prior ZIT 8/16/32-step CSVs. They have no submodule or φ columns and carry hook_mode 'attn'. The current `CoSPAFS_EMAGGuiderImpl` records telemetry only inside the EMA-perturbed [uncond, cond] forward pass. Their g scale is about 10× below the passive campaign's at the same model and resolution, and its time trend is reversed. This is recorded here because it changes how E12 and E13 are interpreted. It does not change the scores.
