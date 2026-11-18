# Round 6 (Claude): prepublication sweep — results and synthesis

- **Preregistration:** `R6C_PREREGISTRATION.md`, SHA-256 c0279f03…ad612, frozen 2026-10-08T07:06:28Z, before any Round 6 calculation.
- **Scripts:** `scripts/r6c_*.py`, using the Round 2–5 estimators unchanged.
- **Outputs:** `out/`. **Figures:** `fig/`. **Literature:** `LIT_R6C.md` (77 sources, 76 verified on a primary page).
- **Audit of the reviewer's Round 6:** `R6C_AUDIT_OF_REVIEWER.md`.
- **Release claim register:** `R6C_RELEASE_CLAIM_REGISTER.md`.
- **Gates:** `R6C_GATES.md`.

**Scorecard:** 22 preregistered rows, 17 confirmed and 5 refuted (`out/R6C_scorecard.csv`). Post hoc checks P1–P4 and the analyzer replay A1–A4 (`R6C_ANALYZER_REPLAY.md`) are labelled as such.

## What this round changes for tonight's papers

1. **Use the three-regime result, scoped to Z-Image MLP along σ.** *(Revised after the analyzer replay: along the layer axis, the analyzer's β_layer + − + − pattern is cross-architecture. See item 8 and `R6C_ANALYZER_REPLAY.md`.)* Expand → contract → re-expand appears in 100% of prompt bootstraps on the archive, on the clean ZIB CFG 4 cond branch and on ZIB CFG 1 cond-only. The contraction is −49.7% in the archive and −50.5% on the clean branch. Capture regime and guidance do not move it (S4a–d). The time-order test is decisive on 25 steps (0/1000 permutations fit as well) but not on 8 steps (ZIT: 38/1000, post hoc P3).
2. **Attention follows a different curve, the same in all five architectures.** Layer-mean attention g rises through most of denoising and falls late in ZIT, ZIB, Mochi, SD3.5 M and SD3.5 L (92–100% of bootstraps; S4e refuted for "+−+", post hoc P1). This shared rise is the reason scalar curves correlate across unrelated models.
3. **The lineage statistic is the tangent, and the scalar curve should never be used for lineage.** On the curve that gave the historical numbers, ZIT↔SD3.5 L scores 0.981, above ZIT↔ZIB at 0.933. The lineage margin over the best unrelated pair grows with the observable: −0.05 (scalar g), +0.05 (scalar log g), +0.13 (depth × σ field), +0.15 (registered shape), **+0.70 (tangent)** (S1a–c).
4. **The "Mochi anomaly" comes from the capture.** All four historical numbers reproduce exactly from raw rows. The ordering ZIB↔Mochi > ZIB↔ZIT appears only with the pooled ZIB archive, on linear g, with the full prompt set. Changing any one of those reverses it. On the explicit cond branch the scores are 0.989 vs 0.831 (S1d).
5. **New Mode-I result: prompt-similarity structure is shared across architectures.** For each model, a 117×117 matrix records which prompts look alike in telemetry. These matrices agree between every pair of models: Mantel ρ is 0.13–0.43 for unrelated pairs and 0.56–0.62 for the lineage pair, with every p ≤ 0.0005 (S2c). Three things do not account for it:
   - **Theme blocks.** It persists within themes (29 of 30 pair×step cells have p < 0.05).
   - **Shared words.** It persists after removing prompt-text overlap.
   - **Shared text encoders.** Z-Image↔Mochi, with no shared encoder, scores above SD3.5 M↔L, which share all three (S2e refuted).

   Every Mantel value is tested against its own prompt-permutation null. This is the calibration that `LIT_R6C.md` 2.9 says global similarity metrics need; it found that uncalibrated "platonic" convergence is largely a scale artifact. Cite the convergence literature with that caveat.

   The theme-residual version of the lineage claim holds for the lineage pair (0.63–0.75). It does *not* make unrelated pairs near zero: those reach 0.52 by the middle and last steps (S2b refuted). Theme residuals are lineage-specific only at the first step.
6. **Drop the synchronization pulse as an event.** A straight-line surrogate gives a higher R peak in 7 of 7 channels. Letting each layer drift on its own random monotone clock matches the observed peak in 4 of 7 channels (S5). Some timing coordination beyond independent drift exists in ZIT (p 0.008–0.014) and marginally in ZIB attn (p 0.048). That is the most the data supports.
7. **Re-state the estimators.** At matched n, relative onset is 0.84–0.95 σ on six of seven channels. The exception is ZIT MLP at 0.67, on an 8-step grid. Relative onset varies less across channels and across fold partitions than σ50 does (S3a–c). On the calibrated persistence scale, archive seed deviations sit at **0.49–0.87 of a random walk**; only ZIT attn is above (1.13). Paper tables should quote "between AR(0.8) and walk" with the simulated references, not "walk-like" (S3d refuted).
8. **(Added after the analyzer replay.) The layer-axis three phases are cross-architecture.** Read with Zac's analyzer, unchanged, the layer-direction flow β = g(L+1) − g(L) shows + − + − with three crossings in 100% of bootstraps for SD3.5 M and L attention, 84–90% for Z-Image MLP and 67% for ZIB attention. At fixed σ it appears in all four architectures. It beats both 09-15 surrogates in SD3.5 M and Z-Image Base, and it is null-generic in SD3.5 L.

## Scorecard

| ID | Verdict | Result |
|---|---|---|
| S1a | CONFIRMED | Lineage margin at the scalar level is −0.049 (g) and +0.047 (log g) |
| S1b | CONFIRMED | All 8 unrelated pairs have r ≥ 0.6 on the scalar curve (0.73–0.98) |
| S1c | CONFIRMED | Margins −0.05 / +0.05 / +0.13 / +0.15 / +0.70 |
| S1d | CONFIRMED | Ordering margin −0.008 (archive) → +0.157 (explicit cond branch) |
| S2a | CONFIRMED | Lineage theme residual 0.678 / 0.750 / 0.626 |
| S2b | REFUTED | Unrelated theme residual reaches 0.517; 4 of 8 pairs exceed 0.20 at ≥ 2 time points |
| S2c | CONFIRMED | Lineage RSA 0.564 / 0.616 / 0.619, p 0.0005 each |
| S2d | CONFIRMED | SD3.5 M↔L first-step RSA 0.308 (tangent −0.25) |
| S2e | REFUTED | Encoder-sharing order fails: Z-Image↔Mochi 0.390 > SD3.5 M↔L 0.308 |
| S2f | CONFIRMED | Text RSA is positive in 5 of 5 models (0.08–0.15, p 0.0005) |
| S2g | CONFIRMED | Spearman(content RSA, tangent) = 0.30 over 10 pairs |
| S3a | CONFIRMED | Relative onset in [0.80, 0.97] in 6 of 7 channels |
| S3b | CONFIRMED | Range across channels: relative onset 0.28 vs σ50 0.50 |
| S3c | CONFIRMED | Narrower partition spread in 7 of 7 channels |
| S3d | REFUTED | VR4/walk ≥ 1 in only 1 of 7 (0.49–1.13) |
| S4a | CONFIRMED | +−+ in 100% of bootstraps (archive ZIT and ZIB MLP) |
| S4b | CONFIRMED | +−+ in 100% (clean CFG 4 cond, CFG 1 cond-only) |
| S4c | CONFIRMED | Breakpoints 0.972 / 0.818 (archive) vs 0.986 / 0.818 (clean) |
| S4d | CONFIRMED | Contraction −49.7% vs −50.5% |
| S4e | REFUTED | Attention is "++−", not "+−+", in all 5 models |
| S5a | REFUTED | The warp null is beaten in only 3 of 7 channels |
| S5b | CONFIRMED | The straight-line surrogate's peak is ≥ observed in 7 of 7 |

## Post hoc checks (labelled)

- **P1. Attention topology** (`out/posthoc_R6C.json`). The modal topology is "++−" in all five models: ZIT 92%, and the others 100% of 500 bootstraps. From its maximum, the late fall is −6% to −17%. ZIT's 8-step schedule stops at σ 0.30 and so never records the fall.
- **P2. Content RSA robustness** (`out/P2_rsa_robustness.csv`).
  - *Within-theme RSA* (theme means removed, same-theme pairs only, theme-preserving permutation null): lineage 0.47 / 0.63 / 0.56; unrelated 0.06–0.42; p < 0.05 in 29 of 30 cells.
  - *Partial Mantel controlling prompt-text RSA:* values change by ≤ 0.02.
- **P3. Time-order permutation of the three-segment fit** (`out/P3_order_perm.csv`).
  - The ZIB MLP channels (archive, clean CFG 4, clean CFG 1): 0 of 1000 permutations fit as well, with R² 0.94 against a null median of 0.27.
  - ZIT MLP (8 steps): 38 of 1000. The 09-15 statement "0/1000" holds for the 25-step channel only.
- **P4. The three phases along the layer axis** (`out/P4_depth_topology.csv`, `P4_depth_per_sigma.csv`, `P4b_depth_robustness.csv`, `fig/R6C_fig8_depth_profiles.png`; added after Zac asked, 2026-10-08 07:38Z). Same three-segment fit as S4, with layer index in place of σ, on the depth profile of g averaged over σ and generations.
  - *Z-Image MLP:* "+−+" in 100% of 500 bootstraps for archive ZIT, archive ZIB and the clean ZIB CFG 4 cond branch. g rises to a peak at layer 5 (depth 0.17), falls by about 1.5 decades to a floor across the middle layers, and rises again over the last layers. It survives log g and dropping the last one or two layers (100% each time). The second breakpoint moves with the scale (0.86 on g, 0.59–0.66 on log g), so the end of the floor is not sharply located; the first breakpoint is stable at 0.17–0.19.
  - *Attention:* **[withdrawn 2026-10-08, see the analyzer replay]** no shared depth shape. ZIT/ZIB attn give "+−+" on g, but it is a dip near depth 0.85 that disappears when the last two layers are dropped, and ZIB attn is "+++" on log g. Mochi's "+−+" comes from its last two layers (layer 46 near zero); without them it is "++−" or "+++". SD3.5 M and L are "+−−" with no re-expansion.
  - *Reading:* ~~the three-phase structure is real on both axes in Z-Image MLP, and on neither axis does it carry over to attention in the other architectures.~~ **Superseded by the analyzer replay.** A piecewise-linear fit to the time-averaged depth profile is not the analyzer's layer-axis measurement. The analyzer's β_layer = g(L+1) − g(L) shows + − + − in SD3.5 M/L attention (100% of bootstraps), Z-Image MLP and ZIB attention, and per σ in all four architectures.

## Analyzer replay A1–A4 (post hoc, 2026-10-08)

This section uses Zac's CoS-PAFS Live Analyzer v3 unchanged. Full write-up: `R6C_ANALYZER_REPLAY.md`.
- The layer-axis three-phase pattern (+ − + − in β_layer) is cross-architecture. Its strength by channel: SD3.5 M and L attention 100%, ZIT MLP 90%, ZIB MLP 84%, ZIB attention 67%, Mochi CFG 4.5 19% on the whole run but 60% of individual steps.
- It beats the 09-15 Surrogates A and B in SD3.5 M attention and ZIB attention and MLP. In ZIT MLP it beats A but comes from the mean field under B. In SD3.5 L it is null-generic.
- The analyzer's v3 three-crossing tools (mean-profile, interval decomposition, g₃/g₁) are LLM-only as written. The dashboard pools attn and mlp.

## Other ways to read this data, and what each gives

| Reading | Verdict | Why it matters for the papers |
|---|---|---|
| **Projection degeneracy.** A scalar summary shared across architectures is uninformative about lineage | **Supported and explained.** Every attention envelope rises (P1), so any two correlate (S1b) | Mode-I methods: report lineage only with the tangent. Mode-II: a one-number effective rank is many-to-one by construction |
| **Capture functional as a hidden variable** | **Supported.** It flips the historical ordering (S1d), on top of the Round 5 pooling theorem | Mode-I must name the capture regime for every number. Mode-II: the concavity bound |
| **Common scaffold plus lineage-specific direction** (reviewer) | **Supported, with three levels:** envelope shared by all; content geometry shared by all and strongest in the lineage; direction of motion shared only by the lineage | Gives Mode-II a concrete decomposition with a measured level for each part |
| **Content geometry tracks the text encoder** | **Refuted** (S2e). Unrelated encoders agree as much as shared ones | The shared structure lives in the prompts or the generation task, not in encoder weights. The literature reports convergent representations across models (`LIT_R6C.md` §2) |
| **The MLP three-regime curve as a capture or CFG artifact** | **Refuted.** It is identical on clean branches and at CFG 1 (S4b–d) | The strongest Mode-I curve result now rests on clean captures |
| **E→C→C as a cross-architecture law** | **Split by axis.** Along σ: refuted as stated (S4e); the MLP curve is Z-Image-specific and attention shares a different curve. Along the layer axis, in the analyzer's β_layer: **supported across architectures** (A2–A4), beyond both surrogates in SD3.5 M and Z-Image Base, null-generic in SD3.5 L | Mode-I scopes E→C→C to Z-Image MLP and reports the attention curve separately as the cross-architecture regularity |
| **Synchronization as coordinated timing rather than an event** | **Partly supported** (ZIT, marginal ZIB attn) | At most a supplementary statement, with the warp null |
| **Massive activations as a driver of g** (literature) | Not tested. Plausible: a few large AdaLN-linked channels can set s1/s2 in an uncentred matrix (`LIT_R6C.md` §3) | Mode-II caveat. A per-channel ablation of the top channels would test it (Mode-III) |

## Synthesis for Mode-I and Mode-II

**Mode-I (empirical) can state, with these numbers:**
- In Z-Image MLP, g expands, contracts by about 50%, and re-expands. This holds in Base and Turbo, under pooled and explicit-branch capture, and at CFG 1 and CFG 4 (S4; P3 for the order test). The same MLP stack also shows three phases along depth: a peak at layer 5, a floor through the middle layers, and a rise over the last layers (P4, post hoc).
- Attention g rises and then falls late in five DiT/MMDiT/AsymmDiT models (P1, descriptive).
- Lineage is carried by the direction of motion. The tangent is 0.94–0.99 for Base↔Turbo against ≤ 0.24 for unrelated pairs. Coarser observables cannot separate lineage (S1).
- Prompt-similarity structure is shared across five unrelated models beyond theme and text, and is strongest within the lineage (S2, P2).
- The correction trail, now with two more items: the scalar-curve ordering and the sync pulse.
- Calibrated estimators throughout (S3).

**Mode-I should not state:** a synchronization event or pinch; the σ-axis E→C→C curve as universal across architectures (the layer-axis β_layer pattern is cross-architecture; see the analyzer replay); theme residuals as lineage-specific at every time; σ50 or VR comparisons at unequal n or against a VR = 1 walk; the Qwen ladder as an anomaly (it is demoted in the ledger); any separatrix or commitment wording.

**Mode-II (theory) gains three things:**
- The concavity theorem for pooled PR (Round 5).
- A formal statement that a scalar envelope correlation cannot identify lineage when all envelopes share a monotone trend.
- A three-level decomposition, with a measured level for each part: shared envelope, then shared content geometry, then lineage-specific tangent field.
