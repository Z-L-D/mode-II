# Round 15 preregistration: replication of the 2026-11-05 deep-dive findings on held-out captures

Frozen before any R15 statistic is computed. The exploratory analyses X1-X4 (2026-11-05, post hoc on Z1/ZIT/ACE cubes;
`exploratory/`) motivated these rules. Every R15 test below runs on captures or comparisons that X1-X4 did not use, or
adds a null that X1-X4 did not compute. Thresholds are fixed here; nothing is tuned after scoring.

Common definitions
- Depth profile at a state: per layer (log g, log gap, H); for RSA, prompt means over generations, each feature z-scored across prompts.
- Prompt RSA: Spearman correlation of the upper triangles of the two prompt RDMs (1 - Pearson r between prompt profiles);
  Mantel p from 999 prompt permutations (seed 15).
- Seed share: mean within-prompt variance (ddof 1) / total variance of the standardised depth profile at a state.
- Tangent: Round 5 E10 definition (depth-normalised mean profile, 24-point depth grid, cosine of depth differences,
  states matched by nearest sigma, averaged over states).

R15-1 Text-to-latent handoff (replicates X2 on held-out explicit-branch captures).
  Channels: SD3.5 M R4 CFG 4.5 1024 attn (seeds 1001+1002, 30 prompts); ZIT-25 R4 CFG 4 attn and mlp (seed 1001, 19 prompts);
  SD1.5 U-Net attn and mlp (cond/uncond rows, 119 prompts). Per step, RSA between the cond and uncond prompt RDMs.
  (a) step 0: RSA <= 0.25 (if the uncond step-0 features do not vary across prompts, (a) counts as met: no prompt information);
  (b) at step ceil(0.10 T): RSA >= 0.50; (c) final step: RSA >= 0.80.
  SUPPORTED if (a), (b) and (c) all hold in at least 4 of the 5 channels.

R15-2 Guidance does not raise seed share (addresses the CFG confound in X3: Base ran at CFG 4, Turbo at CFG 1).
  SD3.5 M R4, cond branch, 1024 px, 30 prompts x seeds 1001/1002, CFG 4.5 vs CFG 1, 40 steps, attn.
  SUPPORTED if seed share(CFG 4.5) <= seed share(CFG 1) + 0.02 at >= 75% of steps.

R15-3 Lineage content geometry in a second family (replicates X1b on ACE-Step).
  ACE Base cond vs ACE Turbo, shared music prompts, prompt RSA at the first state, the state nearest sigma 0.75, and the last state; attn and mlp.
  SUPPORTED if RSA >= 0.30 with Mantel p < 0.01 at >= 2 of 3 states in both submodules.

R15-4 Tiers survive a block-restricted layer-permutation null (new null).
  Null: layer order of model B permuted only within each contiguous quarter of depth (layers split into 4 blocks by index), 1000 draws, seed 15.
  Pairs: same weights ZIT-ZIB attn/mlp; lineage Z1-ZIT attn/mlp, ACEB-ACET attn/mlp; unrelated: the 8 archive pairs (attn),
  Z1-SD35M, Z1-SD35L, Z1-MOCHI (attn), ACET-ZIT attn/mlp, ACEB-Z1 attn/mlp.
  Predictions: (a) same-weights and Z-Image lineage channels and ACE attn: tangent - block p95 >= 0.20;
  (b) ACE mlp: tangent - block p95 < 0.20 (predicted to fail, as X4 suggests end-layer carriage);
  (c) every unrelated pair: tangent <= block p95 + 0.10. SUPPORTED if (a), (b) and (c) all hold.

R15-5 Lineage discrimination (ROC).
  For each pair, 200 draws of independent 30-generation subsets per side (seed 15); tangent per draw. Classes:
  same weights = ZIT-ZIB, Z3-ZIB, ACEB0-ACEB1; lineage = Z1-ZIT, Z1-Z3, ACEB-ACET; unrelated = the R15-4 unrelated pairs.
  (a) attention AUC(lineage vs unrelated) >= 0.95; (b) attention AUC(same weights vs lineage) >= 0.95;
  (c) MLP AUC(lineage vs unrelated) >= 0.90 (unrelated MLP pairs: ACET-ZIT, ACEB-Z1, ACET-Z1, ACEB-ZIT).
  SUPPORTED if (a), (b) and (c) hold. Z3 has 120 generations, so its subsets overlap more; reported, not corrected.

R15-6 LLM depth tangents across families (first lineage-style test on Mode-III captures).
  Cubes: token-mean over generated tokens of (g_w64, gap_w64, H_w64) per (gen, layer, submodule) for Qwen3.5-0.8B (24 layers),
  Qwen2.5-1.5B (28) and Gemma-3-1B short run (26). One state, so the tangent is a single cosine. Full layer-permutation null, 1000 draws.
  (a) attention: all three pairs tangent < null p95 + 0.30;
  (b) MLP: with the last depth step removed, all three pairs tangent < null p95 + 0.30.
  SUPPORTED if (a) and (b) hold. Full-depth MLP values are reported (end collapse is expected to raise them).

R15-7 LLM and DiT prompt geometry (descriptive, one-sided).
  Qwen2.5-1.5B and Qwen3.5-0.8B prompt RDMs (generated-token depth profile, attn and mlp) vs Z-Image Base cond first-state RDM (attn),
  on the shared image prompts. SUPPORTED if at least one LLM channel gives RSA > 0 with Mantel p < 0.01.

Reporting: every verdict, value and threshold goes to out/R15_verdicts.csv and RESULTS_R15.md. X1-X4 stay labelled post hoc
in the papers; R15 results are labelled preregistered.
