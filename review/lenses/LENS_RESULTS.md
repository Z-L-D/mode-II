# Lenses on the spectral triple: results (2026-10-08)

**Preregistration:** `LENS_PREREGISTRATION.md`.
- Part A: SHA e3aba957…0939, frozen 12:45:12Z.
- Part B (ladders): SHA 23c90704…7037, frozen 12:50:41Z, after the project excavation and before the literature survey was read.

**Inputs:**
- `EXCAVATION.md`: what the project recorded on each lens.
- `LIT_LENSES.md`: about 60 verified papers, with BibTeX.

**Scripts:** `lens_*.py`, Python only. **Outputs:** `out/`. **Gate review:** `LENS_GATES_v024.md`. **Figure:** `out/LENS_summary.pdf`.

**Data:** the 7 archive diffusion channels cached by the analyzer's DataManager. LLM captures are used for L1 only. β crossings come from the analyzer's `plot_beta_layer`, unchanged.

## Short answer
- **Most of the lenses do not apply to this observable.**
  - No RG flow: β_layer is not a function of g alone.
  - No ladder (Fibonacci/Kepler or Coxeter): hits occur at chance rates.
  - No heat-kernel inheritance: internal spectra do not isotropise the way the forward noise does.
  - No spin-glass replica structure: overlaps are unimodal, and the only ultrametricity comes from the prompt set.
  - No robust symmetry-breaking susceptibility peak.
- **What the triple is.** It is an exact pair of Rényi entropies of one distribution plus the top-two ratio: **H₂ = ln(gD), H₁ = sv_entropy, Λ = ln gap**. It is effectively two-dimensional, extremely anisotropic (g is 0.1–7% of the Marchenko–Pastur isotropic value), and no simple spectral family reproduces it.
- **Two weak results deserve a scoped test:**
  - the slope at the third β crossing agrees across Z-Image and SD3.5;
  - Mochi has a susceptibility peak at σ ≈ 0.86.
- Nothing here changes the papers' claims. The literature supports a few wording additions (end of this file).

## What the triple is (definitions from the capture code)
- **Definitions.** σᵢ = √eig(XᵀX/N) and pᵢ = σᵢ/Σσ. g·D = 1/Σpᵢ² = exp H₂(p). sv_entropy = H₁(p), so exp H₁ is the Roy–Vetterli effective rank. spectral_gap = p₁/p₂.
- **This matches the papers.** Mode-I (l. 86–90) and Mode-II (l. 77–91) already define g as the amplitude participation ratio on singular values and distinguish it from the eigenvalue participation ratio and the effective rank.
- **L0 sanity:** H₁ ≥ H₂ and Λ ≥ 0 hold on every row of all 14 captures (0 violations).
- **Isotropy null (post hoc, LIT T1):** g_MP at each channel's N and D is 0.75–0.93. The archive medians are 0.0011–0.064, i.e. 0.14–7% of g_MP. MLP channels sit at 0.14–0.19%. No channel is anywhere near isotropic. (`out/PH_isotropy_baseline.csv`)

## Results against the preregistered predictions

| Test | Prediction | Result | Verdict |
|---|---|---|---|
| **P1a** dimension of the triple | PC1 ≥ 0.90 in ≥ 5/7; PC1+PC2 ≥ 0.99 in 7/7 | PC1 0.68–0.93 (only Mochi ≥ 0.90); PC1+PC2 0.980–0.994 (only SD3.5 L ≥ 0.99); PC3 0.6–2.0% | **Refuted**. The triple is close to 2-D, but not strictly |
| **P1b** power law best for H₁ | PL beats EXP and SB in ≥ 5/7 | PL wins in 5/7 (all attention channels; SB wins both MLP channels). PL H₁ error 3.6–7.0% of ln D for attention, 13.5–14.1% for MLP | **Confirmed** |
| **P1b** PL misses Λ by > 0.5 | in ≥ 4/7 | 3/7 (ZIT MLP 0.92, ZIB MLP 0.96, Mochi 1.04) | **Refuted** |
| **P1b** spike + power law fits H₁ to 0.02 | in ≥ 5/7 | 0/7 (errors 0.05–0.10). SPL has no solution for up to 55% of rows (0.2% Mochi to 55% SD3.5 L): σ₁/σ₂ is *smaller* than the bulk power law implies, so the top is a plateau of several large directions, not one spike | **Refuted** |
| **P1c** no universal bulk exponent | range > 0.2 across families | a_bulk: Z-Image attn 0.84–0.85, MLP 1.02–1.08; Mochi 0.749 ± 0.003; SD3.5 M 0.745 ± 0.005, L 0.787 ± 0.003 | **Confirmed** (range 0.33). Note: Mochi and SD3.5 M agree within 1 SE, a single cross-family coincidence |
| **R1** RG autonomy | ΔR² ≥ 0.30 in ≥ 5/7 | g alone explains 9–35% of β_layer out of sample; adding layer position gives 58–92% (ΔR² 0.49–0.65) in 7/7. Post hoc (PH2): ΔR² 0.42–0.76 in every σ tercile, smallest at high σ | **Confirmed**. β_layer is not a function of g; it is a depth finite difference |
| **R2** crossing slopes universal | no cross-family agreement | Crossings 1 and 2 disagree strongly (e.g. y₁ = −5.1/−5.5 in SD3.5 vs −0.04 to −1.0 in Z-Image). **Crossing 3 agrees in 4 of 6 Z-Image↔SD3.5 pairs** (e.g. ZIT MLP −0.53 ± 0.22 vs SD3.5 M −0.45 ± 0.05) | **Refuted by the letter, weakly**. SEs are large, and R1 means a crossing is a zero of a layer-pooled average, not a fixed point |
| **R3** PGL(2) cross-ratio invariance | not supported | The rule is met only in pairs involving ZIB, where the cross-ratio SE is 9–18 because two crossings nearly coincide. In the one well-determined pair (SD3.5 L vs ZIT MLP, CR₁ 1.49 vs 1.43) the raw r₂₁ also agrees. Post hoc in H₂ coordinates (LIT T5): 3/6 pairs agree, the same ill-conditioned ones | **Confirmed in substance**. The letter is met only where the test has no power |
| **L3** heat kernel vs thermal | TH beats HK in ≥ 4/7 | TH wins 7/7 under the frozen rule. HK is infeasible in 32–81% of cells: exactly the cells where the internal spectrum is *less* isotropic than at the end, which additive noise cannot produce. Where both are feasible, HK is better in 3/7. TH needs T ≈ 0.96–1.02 and leaves residuals of 0.20–0.28 | **Confirmed by rule**. Neither deformation describes the σ-trajectory, and the network does not inherit the forward process's isotropisation |
| **L4** heat capacity peak | interior C̄ peak in ≤ 3/7 | 1/7 (ZIB MLP, σ ≈ 0.70). C̄ = 2(H₁ − H₂) is 2.0–5.2. Attention channels fall early and have a shallow interior minimum; MLP channels rise and plateau | **Confirmed** |
| **L4** Z2 susceptibility | no interior χ_w peak in ≥ 5/7 | 4/7 without a peak; peaks in ZIT MLP (σ 0.95), ZIB attn (σ 0.97) and SD3.5 L (σ 0.40) | **Refuted**. Post hoc (LIT T7, logit-normalised): those three vanish, and only Mochi peaks (σ 0.86, stability 0.89) |
| **L5 U1** P(q) not bimodal | BC < 0.555 in ≥ 6/7 | BC > 0.555 in 5/7 | **Refuted by the letter**. Post hoc: every within-prompt P(q) has one KDE mode; BC is inflated by strong left skew (−1.0 to −2.9) |
| **L5 U2a** no within-prompt ultrametricity | no excess in ≥ 5/7 | no excess in 7/7 (p 0.22–1.0); seed triples are *less* ultrametric than the two-level Gaussian null | **Confirmed** |
| **L5 U2b** cross-prompt ultrametricity | excess in ≥ 4/7 | excess in 6/7 (p 0.001–0.018; ZIB MLP p 0.67) | **Confirmed**. As stated in advance, this is the content hierarchy of the prompt set, not replica symmetry breaking |
| **B: Fibonacci–Kepler ladder** (dynamical exponents z) | does not stand | Only 3 of 7 relaxation exponents are positive; z is not a stable power-law exponent (ZIT MLP flips sign with cadence; Mochi 0.31 → 0.00). Primary: 1/3 hits vs 0.48 expected, p = 0.48. The one hit (SD3.5 M z = 1.70 ≈ 5/3) fails cadence (1.98 at every second step, > 2 SE). All observables: 10/57 vs 10.2, p = 0.63 | **Does not stand** |
| **B: Coxeter ladder** 2cos(π/m) | does not stand | Primary (crossing ratios + z): 6/18 vs 4.6, p = 0.27. All observables: 18/57 vs 14.0, p = 0.056 (descriptive; it includes cross-ratios with SE > 9 that hit by default) | **Does not stand** |

**Tally:** 8 predictions confirmed (2 of them only in substance or by rule), 6 refuted (U1 and the χ peaks are explained by post hoc checks, and R2 rests on large SEs), and 2 ladders that do not stand.

## Descriptive (no prediction)
- **LLM triple (L1):** with rank = N tokens, a pure power law predicts H₁ from H₂ to within 0.5–1.4% of ln N in all 7 captures, far better than for diffusion. This is subject to the running-statistic caveat (G4).
- **Spike vs bulk across depth (post hoc, LIT T2, model-based):** at the final step, H₂ across layers follows the SPL bulk exponent (R² 0.49–0.93) more than the spike strength (R² 0.02–0.56). The depth shape is therefore not obviously a one-outlier effect. This needs the spectrum capture to confirm (`out/PH5_spike_vs_bulk.csv`).
- **E10, Z2 and PGL:** these come from the S246 cosmology track (`EXCAVATION.md` §4).
  - **E10:** its only telemetry-facing number is Lehmer's number, the E10 Coxeter element's spectral radius (McMullen 2002), and it gave no match last round.
  - **Z2:** the triple is invariant under X → XO, so a symmetric Z2 break is invisible by construction (LIT C2). Only asymmetric, content-dependent branching can show, and it does not, robustly.
  - **PGL:** R3 above.

## What, if anything, belongs in the papers

**Decision (Zac, 14:20Z): ledger only.** The papers are unchanged; the items below remain proposals.

1. **One sentence, Mode-II §flow (recommended).** "β_layer is a depth finite difference, not a renormalization-group beta function. Out of sample, g alone explains 9–35% of it and depth position a further 49–65%, at every noise level." This quantifies the non-autonomy Mode-II already states.
2. **One sentence, methods (recommended).** "g·D = exp H₂ and H = H₁ of the same normalised singular-value distribution, so H ≥ ln(gD) identically. For white Gaussian activations at our N and D, g would be 0.75–0.93; observed values are 0.1–7% of that."
3. **Optional sentence.** All three statistics are invariant under X → XO, so they cannot register symmetric symmetry breaking (Raya & Ambrogioni 2023; Biroli et al. 2024). This explains why the papers make no speciation claim.
4. **Leave out:** RG field-theory analogies, Fibonacci/Kepler, Coxeter, E10, Salem/Pisot, spectral triples, spin-glass/RSB, the heat-kernel analogy. All are tested here and none survives. They are recorded in the ledger as "tested, not supported", not as "excluded forever".

## Future tests scoped by this round (only where something was suggestive)

| # | Test | What it needs | Null and decision rule |
|---|---|---|---|
| F1 | **Spectrum capture.** The exact family of the spectrum, outliers vs bulk, and a massive-activation control | Per row: top-32 singular values, Σσ, Σσ², and g of the mean-centred matrix. Instrument change only | Recompute L1 exactly. "Outlier-driven depth shape" if removing the top-k changes the depth profile by > 50% of its variance |
| F2 | **Latent spectrum Λ per model.** Speciation-time prediction and the input-level heat-kernel baseline | About 1k encoded images per model. No generation | Feature at σ_S = √Λ/(1+√Λ) within ±1 step in ≥ 2 families vs a uniform-σ null. Model-generated σ-trajectory = observed minus input baseline beyond 3 SE |
| F3 | **G3 clones at the Mochi χ peak (σ ≈ 0.86) and at Z-Image σ 0.93–0.97** | Clone continuations, ≥ 16 futures each, fates labelled on outputs | Committor slope maximal at the χ-peak σ, beyond a permutation null, in ≥ 2 families |
| F4 | **Crossing-3 slope across families** | Per-σ analyzer β_layer on existing data (cheap), plus MLP hooks on SD3.5/Mochi | Slopes agree within 2 SE at matched σ in ≥ 2 families *and* autonomy holds within a σ bin. Without autonomy, report as shape agreement only |
| F5 | **Avalanches (G4)** | KV-cache per-unit activations, thousands of tokens, MR estimator | Must beat the position-matched random-event null and the synthetic running-capture null (audit) |
