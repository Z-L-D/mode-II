# Lenses on the spectral triple: preregistration (Part A)

**Asked by Zac, 2026-10-08 12:38Z.** Look at the data through the geometry of the spectral triple, the Fibonacci (Popkov) and Coxeter (Ruiz) ladders, Z2 / PGL / E10, spin glasses, heat kernels and thermodynamics, and the renormalization group (RG), especially for g as a coarse-grained observable. Rule nothing out. Scope future tests for anything suggestive.

**Order of work (Zac's paradigm).** Predictions frozen here → project excavation and literature research (running in parallel, not read before freezing) → synthesis → calculation. Part B (ladder target sets) is frozen separately after the project excavation defines what "Popkov's Fibonacci ladder" and "Ruiz's Coxeter ladder" meant in this project. Part B is frozen before the literature survey is read.

**Status.** Post hoc with respect to the archive: these data have been analysed for months. The tests below are new and their outcomes have not been computed.

## Data
- **Diffusion (primary):** the 7 archive channels cached by the analyzer's DataManager in `review/standing_wave/sw_cache/*.npz`. Per generation × step × layer they hold g, spectral_gap and sv_entropy, plus σ and prompt.
  - ZIT attn/MLP: 480 gens, 8 steps, 30 layers.
  - ZIB attn/MLP: 476 gens, 25 steps, 30 layers.
  - Mochi attn: 428 gens, 30 steps, 48 layers.
  - SD3.5 M attn: 480 gens, 40 steps, 24 layers.
  - SD3.5 L attn: 485 gens, 28 steps, 38 layers.
- **Families** for any "recurs in unrelated architectures" rule: Z-Image (ZIB, ZIT), SD3.5 (M, L) and Mochi.
- **Hidden width D:** Z-Image 3840, Mochi 3072, SD3.5 M 1536, SD3.5 L 2432. Token count N exceeds D in every diffusion channel, so XᵀX/N is full rank.
- **LLM (secondary, L1 only):** the 7 archive captures, generated tokens only. Here N < D, so the rank is N. The capture is a running statistic over the growing context.

## The triple, from the capture code (`capture_code_r2/cos_pafs_llm_measure.py`, lines 671–681)
- **Definitions.** σᵢ = √eig(XᵀX/N) and pᵢ = σᵢ/Σσ.
- **The three observables:**
  - g·D = (Σσ)²/Σσ² = 1/Σpᵢ², so **H₂ := ln(g·D)** is the Rényi-2 entropy of p;
  - **H₁ := sv_entropy** is its Shannon entropy;
  - **Λ := ln(spectral_gap) = ln(p₁/p₂)**.
- **Consequences.** The triple is a coarse description of one distribution p. Every member is invariant under X → XO for any orthogonal O, which includes the Z2 sign flip X → −X. A broken Z2, or any broken O(D) direction, is therefore invisible in the triple's mean and can only appear in fluctuations.

## L0. Sanity (not a hypothesis)
- Share of rows violating H₁ ≥ H₂ (tolerance 10⁻³, for 4-decimal rounding) or Λ ≥ 0.
- Expected: 0.

## L1. Geometry of the triple
- **Coordinates:** h₁ = H₁/ln D, h₂ = H₂/ln D and Λ per row (generation × step × layer).
- **P1a, dimension.** PCA on the standardised (H₁, H₂, Λ) over all rows of a channel.
  - *Prediction:* PC1 ≥ 0.90 of the variance in ≥ 5/7 diffusion channels, and PC1+PC2 ≥ 0.99 in 7/7.
- **P1b, spectral family.** Spectra over i = 1..D:
  - **PL:** σᵢ ∝ i^(−a);
  - **EXP:** σᵢ ∝ exp(−b·i/D);
  - **SB:** spike plus flat bulk (σ₁ = s, σᵢ₍≥₂₎ = 1);
  - **SPL:** spike plus power-law bulk (σ₁ multiplied by s ≥ 1, the rest PL).

  The one-parameter families are fixed by H₂ in each row; their predictions of H₁ and Λ are then scored. SPL is fixed by (H₂, Λ) and its prediction of H₁ is scored. The score is the median |error| over rows, with H₁ errors in units of ln D.
  - *Predictions:*
    - PL predicts H₁ better than EXP and SB in ≥ 5/7.
    - PL misses Λ (median |ΔΛ| > 0.5, i.e. the top singular value is an outlier) in ≥ 4/7.
    - SPL predicts H₁ to within 0.02 (median, in ln D units) in ≥ 5/7.
- **P1c, bulk exponent.** The SPL bulk exponent a at the final step (median over layers of the channel-mean) differs across the three families by more than 0.2.
  - *Prediction:* no universal exponent.
- **LLM:** the same metrics, reported descriptively with no verdict.

## L2. RG lens: is β_layer a beta function of the coarse coupling g?
- **R1, autonomy.** An RG flow in one coupling requires β to depend on g alone.
  - Pairs: (g_L, β_L = g_{L+1} − g_L), the analyzer's definition, for every generation and step.
  - Models: β = f(g), a cubic spline with 12 quantile knots, against f(g) plus a layer fixed effect.
  - Scoring: out-of-sample R² from 5-fold cross-validation grouped by prompt. ΔR² = R²(g + layer) − R²(g).
  - Autonomy holds if ΔR² < 0.05.
  - *Prediction:* ΔR² ≥ 0.30 in ≥ 5/7, so the flow is not autonomous in g.
- **R2, crossing slopes (critical exponents).**
  - At each analyzer crossing g*ₖ, y_k = dβ/dg is taken from the analyzer's binned curve (`plot_beta_layer`, unchanged), by linear interpolation across the two bins that straddle the crossing.
  - SEs come from 100 generation bootstraps, analyzer style (seeds 3000+k, drawn copies relabelled, only bootstraps with the same crossing pattern).
  - Universality requires the slope of the matching crossing to agree within 2 SE in ≥ 2 families.
  - *Prediction:* it does not.
- **R3, PGL(2,ℝ) invariance.** For channels with 3 crossings, the cross-ratios CR₁ = CR(0, g*₁; g*₂, g*₃) and CR₂ = CR(g*₁, g*₂; g*₃, 1) are invariant under any Möbius reparametrisation of g. The cross-ratio is defined as CR(a, b; c, d) = (c−a)(d−b)/((c−b)(d−a)).
  - The test: agreement of CR across ≥ 2 families within 2 SE, while the raw ratios r₂₁ and r₃₁ do not agree.
  - SEs come from the R2 bootstraps.
  - *Prediction:* not supported.

## L3. Heat kernel vs thermal deformation along σ
- **Reference spectrum.** For each channel and layer, take the final-step channel-mean (H₂, Λ). The SPL fit gives a reference eigenvalue spectrum λᵢ = σᵢ².
- **Two one-parameter deformations,** each fitted per earlier step to match H₂ exactly:
  - **HK** (additive isotropic heat flow): λᵢ + c·mean(λ), with c ≥ 0. This is infeasible if H₂ at that step is below the reference.
  - **TH** (thermal tempering): λᵢ^(1/T), with T > 0.
- **Score:** |ΔH₁|/ln D + |ΔΛ| for the predicted H₁ and Λ, median over steps and layers. An infeasible fit counts as a failure.
- *Prediction:* TH beats HK in ≥ 4/7.
- Also reported: the HK infeasible share, and whether early steps move toward isotropy (h₂ → 1).

## L4. Thermodynamics and Z2 symmetry breaking
- **Spectral heat capacity.** C̄ := 2(H₁ − H₂). The exact identity is H₁ − H₂ = ∫₁² (2−q) Var_q(ln p) dq, where Var_q is the variance of ln p under the escort distribution p^q/Σp^q. So C̄ is a weighted mean of the heat capacity of p regarded as a Gibbs state at temperature 1.
- **Interior peak criterion:**
  - the layer-mean channel-mean C̄(step) has its argmax away from the first 2 and last 2 steps (1 and 1 for ZIT);
  - the argmax is stable to within ±2 steps (±1 for ZIT) in ≥ 80% of 200 prompt bootstraps;
  - the peak exceeds both endpoints by more than 2 bootstrap SE.
  - *Prediction:* an interior C̄ peak in ≤ 3/7.
- **Susceptibility.** χ_w(step) is the within-prompt (across-seed) variance of layer-mean g, averaged over prompts with ≥ 2 generations. A Z2-breaking (speciation) signature is an interior χ_w peak under the same criterion.
  - *Prediction:* no interior χ_w peak in ≥ 5/7, because the triple is O(D)-invariant.
  - If both C̄ and χ_w have interior peaks in a channel, report whether they coincide within ±2 steps.
- The seed share χ_w/(χ_w + χ_b) is reported descriptively.

## L5. Spin-glass lens: overlaps and ultrametricity
- **Fields and overlaps.**
  - Field per generation: the g field over steps × layers, minus the channel-mean field, with each feature z-scored.
  - Overlap: q_ab = Pearson correlation of two fields. Distance: d = √(2(1 − q)).
- **U1, shape of P(q) for within-prompt pairs.** Bimodality coefficient BC = (skew² + 1)/(excess kurtosis + 3(n−1)²/((n−2)(n−3))).
  - *Prediction:* BC < 0.555, i.e. no replica-symmetry-breaking-like bimodality, in ≥ 6/7.
- **U2, ultrametricity.**
  - For triples sorted d₁ ≤ d₂ ≤ d₃, K = (d₃ − d₂)/sd(all distances).
  - Null: 1,000 replicates of Gaussian fields with the same covariance (top 50 PCs, matched variances), with the same triple design.
  - Excess ultrametricity: observed median K below the 5th percentile of the null medians.
  - **(a)** Within-prompt triples (3 generations, same prompt). *Prediction:* no excess in ≥ 5/7.
  - **(b)** Cross-prompt triples (3 distinct prompts, one generation each). *Prediction:* excess in ≥ 4/7.
  - Interpretation, stated in advance: excess in (b) alone reflects the content hierarchy of the prompt set (data hierarchy), not replica symmetry breaking.
- **Caveat:** seeds may be shared across prompts in the archive; within-prompt variance includes seed identity.

## L6. Ladders and named groups (Part B, frozen separately)
- **Method,** fixed now:
  - Dimensionless exponents get bootstrap SEs: L1 bulk a, the R2 slopes y, and z, the σ-relaxation exponent.
    - z is the slope of log|G_lm(step) − G_lm(final)| against log σ over all non-final steps with a positive difference, where G_lm is the layer-mean channel-mean g.
    - Its SE comes from 200 prompt bootstraps.
    - Cadence check: every second step.
  - These exponents are matched to a frozen target set.
  - A hit is a target within ±1 SE. Chance coverage uses log-uniform values in [x/1.25, 1.25x], and p is Poisson-binomial, as in the Salem/Pisot round.
  - A ladder "stands" only if p < 0.05, the matched exponents are cadence-invariant, and the match recurs in ≥ 2 families.
- **Target sets:** defined in Part B from the project records.
- **E10:** already covered. Lehmer's number, the leading eigenvalue of the E10 Coxeter element, gave no match in the Salem/Pisot round.
- **Z2:** covered by L4.
- **PGL:** covered by R3.

## Reporting
- Each test reports its outcome against its prediction (confirmed or refuted). Nothing is dropped.
- Anything suggestive gets a scoped future test (data, design, null, decision rule).
- Outputs go to `review/lenses/out/`.

---

# Part B: ladder target sets (frozen after the project excavation, before the literature survey is read)

**What the project records say** (`EXCAVATION.md` §2–3):
- Both ladders come from the S246 cosmology track.
- On Sept 25 S246 re-categorised them: "Popkov–Schütz and Ruiz give dynamical exponents z … not amplitude ratios".
- No Ruiz paper and no explicit Coxeter-ladder list exist in the project.
- On the telemetry side, the legacy analyzer v3 compared β crossing ratios to "Coxeter class" lines at √2, φ, √3 and 2.

## Target sets
- **K (Popkov–Schütz Fibonacci/Kepler ladder):** z_n = F_{n+1}/F_n for n = 1..15, which gives 1, 2, 3/2, 5/3, 8/5, 13/8, … → φ. The set includes the diffusive z = 2 and the KPZ z = 3/2.
- **C (Coxeter ladder):** 2cos(π/m) for m = 3..30, together with its limit 2. This gives 1, √2 (m = 4), φ (m = 5), √3 (m = 6), 1.8019 (m = 7), … → 2. These are the spectral radii of the Coxeter graphs A_{m−1} and the rotation numbers of the (2,3,m) and (2,4,m) triangle groups. The set contains all four "Coxeter class" lines of analyzer v3.

## Observables
- **Primary for K (dynamical exponents):** z, the σ-relaxation exponent (L6), for the 7 diffusion channels at native cadence.
- **Primary for C:** the 15 β crossing ratios already measured with the analyzer (`salem_avalanche/out/SA_O1_crossing_ratios.csv`), together with z.
- **Secondary (both sets, descriptive):** all other dimensionless quantities from this round and the last: a_bulk (L1), the R2 slopes |y|, the R3 cross-ratios, and DMD ω/γ.

## Rule
- The method of Part A L6 applies: a hit is a target within ±1 SE; chance coverage uses log-uniform values in [x/1.25, 1.25x]; p is Poisson-binomial.
- A ladder **stands** only if:
  - the primary p < 0.05;
  - the matched exponents pass the every-second-step cadence check;
  - the match recurs in ≥ 2 families.
- If the ±2 SE windows typically contain a target, the result is reported as **uninformative**. Uninformative is not the same as excluded.

## Predictions
- K does not stand: the σ-relaxation exponents are not dynamical exponents of a driven-diffusive system, and nothing ties them to Kepler ratios.
- C does not stand.
- Both are expected to be at least partly uninformative, because the sets are dense in [1.4, 2].
