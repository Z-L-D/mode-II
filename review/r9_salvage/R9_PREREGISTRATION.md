# Round 9: Codex missed correlations and LLM salvage (preregistration)

**Sources.** Two sets of ideas, both posted by Zac on 2026-10-09 at 08:58Z:
- a Codex memo, `MISSED_CORRELATIONS_MEMO.md`, which was itself computed post hoc on the Round 6 summary tables;
- four replies (GLM 5.3, DeepSeek 4.1, Qwen 3.8 Max, Codex) to the question of whether any LLM salvage studies were being missed.

This file fixes, before any new number is computed, what is tested and what counts as support. It uses existing captures only, with the unchanged estimators of Round 2 (E7, E8, E10), Round 6 (S2) and the U-Net arm (u3).

**PSTF exclusion.** The PSTF/PTSF material and the pstf_llama captures are not used. Everything here is a Mode-I/II question or a Mode-III candidate. The PSTF arm stays out.

**What was already seen.** The following were in view when this file was written:
- the Round 2 E7 per-pair median/min/max (Qwen pairs) and the E7 context table;
- the E8 Mistral prefill tables;
- the Round 6 S1/S2 tables;
- the A4 per-step pattern rows for ZIB;
- R8-7.

The Codex memo's own numbers were reproduced before freezing (exact Spearman and permutation p match: RSA~theme ρ −0.405 / 0.810 / 0.976 and RSA~tangent −0.714 at the middle state). Re-tabulations are marked **[tab]** and are not counted as confirmations.

**Global rules.** As in Round 8:
- permutations: 10,000, rng 2026100908;
- bootstraps: 1,000 over prompts, rng 64;
- cross-channel associations need the leave-one-out rule (sign kept and |ρ| ≥ 0.5 in every LOO);
- wording: "tested, not supported";
- no geometry→quality, separatrix or RG wording.

## Premises checked before freezing (facts, not tests)
- **LLM v1 lives in N < D.** The max seq_len over all v1 LLM captures is 389 (Llama-3.1 instruct); Qwen and Mistral reach 342–355. D is 1536–4096. So g ≤ n/D holds by Cauchy–Schwarz (GLM's premise is correct).
  - However, g does not sit at that ceiling. Round 2 E7 gives g·D/n = 0.33–0.37 (median).
  - The ceiling-normalised PR/n is still monotone in n (median |ρ| ≥ 0.9995), and so is g/(n/(n+D)).
  - So the retirement is not *only* the ceiling. R9-B1 tests whether a model-free nested sample reproduces it.
- **The v1 captures log only scalars (g, gap, H per layer), not activations or covariances.** Per-layer null curves g_null(ℓ, n) against the real M(ℓ) (GLM N12, N15) cannot be computed. The isotropic finite-N reference that N12 was meant to supply for the 512 px pooling question was already computed in R8-4: resolution adds ΔR² 0.011, and the trend is branch difference.
- **Qwen chat template.** For each of the 24 prompts, the prefill length is identical across the 1.5B, 3B and 7B captures (24/24 checked), so all three share one template regime.
  - The gravity prompt's prefill is 38 tokens. The Qwen2.5 default template (system "You are Qwen, created by Alibaba Cloud…" plus 8–9 user tokens and role markers) gives about 38–40, so it is consistent with Qwen's own template.
  - A Llama-3.1 header tokenised in Qwen's vocabulary would be longer. This is an inference: the tokenizer was not reachable (the HF CONNECT was refused).
- **Mistral pairs.** 120 instruct and 120 ablit generations share 120 prompt texts, one generation each. There are no repeated seeds of one checkpoint, so GLM's "same checkpoint against itself across seeds" null does not exist in the archive.
- **Most LLM generations hit the 256-token cap** (median gen_len 255). Few end on EOS.
- **DeepSeek C2 (MP crossing at n = D):** never reached, since max n = 389 < D.

## Not tested, and why
- **DeepSeek B (increment whiteness).** Under nesting, the increment is the new row's contribution, so it carries token content by construction. Without the rows there is no model-free null. Its premise that "matched-n derivative correlations should all be high if pure clock" is wrong: the E7 derivative is across depth, not n.
- **DeepSeek C1 (asymptotes g∞, β):** n ≤ 389 ≪ D, so no asymptote is reachable and a fit would extrapolate.
- **DeepSeek D (Mistral depth-localisation vs diffusion interventions):** already in Mode-I §8 and R8-5b.
- **GLM N12/N15:** see above (no M(ℓ)).
- **Qwen 3.8 §1.2 (AR plateau vs diffusion spike):** recorded in the lens round (v0.24). An LLM↔diffusion comparison waits for a new capture (standing decision).
- **Qwen 3.8 §1.3 (prefill domain decoding):** the LLM prompts carry no domain labels. §1.4 (thermodynamic arrow) has no test. Track 2 needs the v2 capture (G4).
- **Codex 3 (context-residualised decoding):** prompt identity is decodable by construction, because the prompt rows are shared by all seeds of a prompt. The RSA version is tested in R9-B6.
- **Codex 7 (rebound census 6.9 / 71.7 / 3.5%) and the D08/D09 numbers quoted by Codex and GLM:** these come from documents outside this archive, and LLM β/TRI is retired.
- **Codex 1, 5 (depth atlas, scale localisation):** descriptive with no decision; not run.
- **Codex memo §4.6 (eigenmode vs tangent):** no rule can be stated.

## Part A: Codex memo

### R9-A1 [tab]
The memo's numbers, reproduced. Over the 8 unrelated pairs, Spearman(RSA, theme residual r):
- first −0.405 (p 0.33);
- middle 0.810 (p 0.022);
- last 0.976 (p 0.0004).

RSA~tangent: −0.714 (middle, p 0.058).

### R9-A2 Held-out replication of RSA ~ theme residual on the U-Net pairs
The SD1.5 arm gives 10 new unrelated pairs (SD1.5 attn and MLP × 5 transformers) on the 116 prompts shared by all six models.
- RSA: P6 values (recorded).
- Theme residual r: Round 6 S2 `theme_r`, unchanged, on the u3 features (new).

Rule: **replicates** if Spearman(RSA, theme_r) ≥ 0.7 at the last state in both 5-pair sets (attn, MLP), *and* ≥ 0 at the middle state in both. Otherwise "tested, not supported". The first state is reported.

### R9-A3 Three-crossing incidence vs plateau onset
Use the recorded A4 per-step layer-axis patterns. For each channel, the share of steps whose pattern has exactly three crossings (`+- -+ +-`) among steps with σ below the E3 plateau onset, and among steps at or above it.
- Channels: SD3.5 M, SD3.5 L, ZIT attn, ZIT MLP, ZIB attn, Mochi CFG 4.5.
- ZIB MLP is excluded because it was seen in R8-7.

Rule: **plateau-concentrated** if the below-onset share exceeds the above-onset share in ≥ 5 of 6 channels.

### R9-A4 Prompt share vs decoding and RSA (7 archive channels)
- (a) Spearman(E6 ANOVA prompt share at the first state, first-state ordered decoding).
- (b) Over the 5 attention models: Spearman(mean E6 ANOVA prompt share over all states, the model's mean RSA to the other four at the middle state).

Rule: associated if |ρ| ≥ 0.6 under LOO (a). (b) is descriptive, with n = 5.

### R9-A5 Anisotropy vs backbone strength
Spearman(log median g/g_MP, backbone PC1 share) over the 9 channels that have both: ZIT, ZIB, Mochi, SD3.5 M, SD3.5 L (attention ratios; backbone per model), SD1.5 attn/MLP, ACE attn/MLP.

Rule: associated if |ρ| ≥ 0.6 under LOO.

### R9-A6 Capture shift vs tangent degradation [descriptive]
Z-Image Base, per submodule. The capture functionals are cond (R4 CFG 4), uncond, branch-mean (R4) and pooled (archive).
- Shift: |median over (step, layer) of the population-median log g of the functional minus that of cond|.
- Tangent to ZIT: Q2 / E10.

Report Spearman(shift, 1 − tangent) over the 3 non-cond functionals.

## Part B: LLM salvage

### R9-B1 Does a model-free nested sample reproduce the retirement statistic? (GLM)
Simulate the v1 capture: rows h_1 … h_n drawn i.i.d., with the statistic recomputed on the nested prefix X_n for n = n0 … n0 + 255, where n0 is drawn from the observed Qwen prefill lengths. Use the instrument's own g, gap and H on the uncentred C = XᵀX/n.

Populations at D = 1536 and 4096:
- (i) isotropic Gaussian;
- (ii) power-law spectrum λ_k ∝ k^−1;
- (iii) (ii) plus a mean spike carrying 20% of the energy.

Each has 30 simulated generations. Compute per generation the Spearman of n with g, with PR/n = gD/n, and with H.

Rule: **the retirement statistic is model-free** if the median |ρ(n, g)| ≥ 0.99 *and* the median |ρ(n, PR/n)| ≥ 0.99 in every population. The observed value is ≥ 0.9995 for both (E7).

### R9-B2 Qwen tangent triple with uncertainty (GLM N11)
Use the E7 statistic unchanged: the Pearson r of the depth derivative of the log mean-g profile, 24-point grid, in 8 matched seq_len bins, median over bins.

Add:
- a prompt bootstrap (resample the 24 prompts, all their generations);
- a layer-order permutation null in the E10 manner (permute b's native layers, then renormalise, interpolate and differentiate; 1,000 permutations, rng 2026100908).

Rule:
- **above null** for a pair if the bootstrap 5th percentile is above the null 95th percentile;
- the 1.5B↔3B value is **separated** from the other two pairs if its 5th percentile exceeds both of their 95th percentiles.

Note: the diffusion bound (|cos| ≤ 0.24) is a different statistic on a different substrate and is not used as a reference.

### R9-B3 Mistral ablation tangent at prefill (GLM N14)
Instruct vs ablit on identical prefill text (E8 pairing).
- E10-style tangent cosine on the 24-point depth-normalised (log g, log gap, H) mean profile.
- E7-style derivative r on log g.
- Prompt bootstrap over the 120 prompts.

Rule: **the edit preserves the tangent** if the cosine is ≥ 0.9 with a bootstrap 5th percentile ≥ 0.9.

### R9-B4 Divergence position of the instruct/ablit texts (GLM N13) [Mode-III candidate]
The divergence token is the first generated token_idx at which layer-0 g differs between the paired generations. Layers 0–5 are identical in weights, so layer-0 g differs only once the input token differs.
- Descriptive: the distribution over 120 pairs.
- Test: Spearman(per-prompt prefill mean |Δlog g| from E8, divergence token), with the permutation p.

Rule: prefill deformation **predicts** divergence if |ρ| ≥ 0.3 and p < 0.01. No same-checkpoint null exists (see premises), so no claim is made about the edit shifting divergence relative to sampling noise.

### R9-B5 Mistral localisation along the shared prefix (Codex 6)
For every generated token before divergence, compute per-layer |Δlog g| between the instruct and ablit generations.

Rule: **localisation stable** if layers 0–5 are bit-identical *and* the argmax layer is the final layer in ≥ 95% of shared tokens (all pairs pooled).

### R9-B6 Prompt-relational geometry across LLMs at matched context (DeepSeek C3, Codex 2)
Prompt-mean depth profile per model (24-point grid of log g, log gap, H, as in S2) at generated-token index k ∈ {0 (prefill), 128, 255}. At fixed k a prompt has the same seq_len in every model.

Because n differs across prompts and every statistic rises with n, the features are first residualised on [1, log n] across prompts (per feature). Comparison is by partial Mantel on ranks, controlling |log n_i − log n_j| (R8-12 code).

Pairs:
- Qwen 1.5B↔3B, 1.5B↔7B, 3B↔7B (24 prompts);
- Mistral instruct↔Llama-3.1 instruct (118 shared prompts; an unrelated pair);
- Mistral instruct↔ablit (reference).

Rule: shared if partial p ≤ 0.01 at ≥ 2 of 3 k. The full (uncontrolled) Mantel is reported beside it.

### R9-B7 Derivative agreement vs context length across Qwen scale (Codex 4) [descriptive]
From the R9-B2 per-bin r: the Spearman of r against bin index for each pair.

### R9-B8 Is the Qwen "terminal collapse" an EOS signature? (Qwen 3.8 §1.1)
The ledger's branch is a final-layer β ≈ −0.876 g step. Per (generation, token), compute the last-layer ratio β_L/g_{L−1} = g_L/g_{L−1} − 1.

Rule:
- **EOS signature** if ratios ≤ −0.8 occur on the final generated token of EOS-terminated generations (gen_len < 255) at more than 3× their rate on other tokens;
- **readout** if ratios ≤ −0.8 occur in ≥ 80% of all (generation, token) rows.

Both are reported for each Qwen size.

## Outputs
`review/r9_salvage/out/R9_*.csv|json`, `R9_RESULTS.md`, ledger v0.29 section `r9-salvage`. Paper changes only after Zac sees the results. The supported Round 8 additions were approved at 08:58Z and applied separately.
