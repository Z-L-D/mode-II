# Standing-wave re-test: preregistration

**Asked by Zac, 2026-10-08 10:40Z:** did the apparent standing wave across the layers hold up?

**Source of the claim:** `review/round6/CoS_PAFS_ModeI_Session_Summary_2026-09-15.md`, sections 38–50. The ledger and the R6C register file it under "L15 throat / downstream (Mode IV–VI), DO NOT USE". Rounds 1–7 never re-tested it.

## The historical claim, restated

The analyzer's 3D depth trajectory is X(L,σ) = (⟨g_micro⟩, ⟨spectral_gap⟩, L), taken per step and averaged over generations (`plot_3d_phase_trajectory`).

- **Transverse field.** Take one fixed PCA axis across all steps. The transverse field is z(L,σ) = x⊥ + i·y⊥, with phase θ = arg z.
- **Stationary.** Allowing a depth shift picks zero shift at every step.
- **Separable.** z ≈ A(σ)Ψ(L). Mode-1 energy is 83% for ZIT MLP and 88% for ZIB MLP, and about 96% post-pinch.
- **Nodes.** Local minima of |Ψ| fall at L = 9, 12, 15, 18, 21 (ΔL = 3), with the global minimum at L15. 500/500 prompt bootstraps put it at L15. A phase-randomised exploratory p was about 0.016.
- **Phase gradient.** It reverses sign about 13 times, with nonuniform spacing (CV 0.86).
- **No wave operator.** A strict shared real potential fit fails, with residuals of 0.93–0.95.

## Reconstruction (frozen)

**Data.** Load each channel with the analyzer's `DataManager.load_paths`, unchanged. Use the same folders and submodule filters as `papers/mode-I/scripts/extract_I_F08_beta_curves.py`. Keep only complete generations: modal step count × layers. This drops the 4 auxiliary 100-step ZIT generations.

**Per-step mean trajectory.** For each step t, take the mean over generations of g_micro and spectral_gap at each layer L.

**Geometry.**
- Primary, **G-z:** each of the three columns (g, gap, L) is z-scored over all pooled (L, t) points of the channel.
- Variant **G-raw:** raw units, as the analyzer plots them.
- Variant **G-ent:** sv_entropy in place of spectral_gap, z-scored. This is the analyzer's own fallback secondary.

**Fixed axis.** Run PCA on all pooled (L, t) points of the channel. PC1 is the axis. z = (PC2 score) + i·(PC3 score).

**Modes.** Form the complex SVD of the matrix Z[L, t]. Ψ is the leading left singular vector and A(t) = s1·(leading right singular vector)*. Mode-1 energy is s1²/Σs². Post-pinch means σ ≤ 0.75.

## Tests (frozen)

**P1. Reproduction (descriptive).** For ZIT MLP and ZIB MLP under G-z and G-raw, report:
- mode-1 energy, full and post-pinch;
- local minima of |Ψ| and the global minimum;
- sign reversals of ∂_L unwrap(arg Ψ).

**P2. Wave versus fixed shape.** A standing wave needs an amplitude that oscillates with fixed nodes. Measure the phase of A(t) relative to A(t0). Then project A(t) on its principal real direction and count sign changes.
- **Oscillation (W)** holds if the phase excursion is ≥ π OR the real projection changes sign at least once.
- **Node stability:** take the local minima of the per-step |z(L,t)| for the 5 central nodes in post-pinch steps. Stability holds if the median absolute displacement from the Ψ nodes is ≤ 1 layer.

**P3. Node regularity against nulls.** Statistic R3 = the longest run of consecutive interior |Ψ|-minimum spacings equal to 3 (±0, integer layers).
- **Null 1:** randomise the phase of Ψ. Keep |FFT| of the complex Ψ, apply uniform random phases, and use 10,000 draws.
- **Null 2:** randomise the profiles. In each depth profile ḡ(·,t) and gap(·,t), apply the same random Fourier phases to every step, after removing the per-layer linear trend. This keeps separability and smoothness. Rebuild the full pipeline, with 1,000 draws.
- p = P(R3_null ≥ R3_obs).
- Also report CV of interior minimum spacing against the same nulls.

**P4. Coordinate robustness.** Count how many of the central nodes {9, 12, 15, 18, 21} are recovered within ±1 layer under G-raw and G-ent. The pass mark is ≥ 4 of 5 in each variant, for each of ZIT and ZIB MLP.

**P5. L15 bootstrap.** Resample prompts 500 times with replacement under G-z and G-ent, and record the location of the global minimum of |Ψ|.

**P6. Generality (descriptive with nulls).** Run P1 and P3 on ZIT attn, ZIB attn, Mochi attn (CFG 4.5, analyzer loader), SD3.5 M attn and SD3.5 L attn.

## Verdict rule (frozen)

- **STANDS as a standing wave:** in both ZIT and ZIB MLP, W holds, nodes are stable, P3 gives p < 0.05 under both nulls, and P4 passes.
- **Stationary depth mode, not a wave:** separable with stable nodes, but W fails. A fixed profile shape with a changing amplitude, which the papers already report as registration.
- **Node lattice does not stand:** P3 fails either null in either channel, or P4 fails.
  - If P4 fails, the nodes depend on the coordinate choice.
  - If P3 fails, the nodes are null-generic.

Everything here is post hoc with respect to the Sept 15 analysis. The criteria are frozen before any number is computed in this session.
