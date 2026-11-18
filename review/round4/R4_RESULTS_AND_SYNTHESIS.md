# Mode Series Round 4: results and synthesis

2026-10-06. Predictions frozen in `R4_PREREGISTRATION.md` (SHA-256 82ef210a5bbad88fdf77c5386f538664cc19f06a7e26aa6c6d4f800c30c79634, 2026-10-06T23:01:29Z) before any section P calculation ran. Scripts: `scripts_p_calcs.py` (section P exactly as frozen), `scripts_posthoc.py` (labelled post hoc), `scripts_figures.py`. Raw outputs: `out/P_results.json`, `out/posthoc_results.json`. The audit of the reviewer's Round 3 work is in `R3_AUDIT_OF_REVIEWER.md`; gates are in `R4_GATES.md`; literature is in `LIT_R4.md`.

**Filters:** one nonfinite SD35M 512 generation dropped; the duplicate S1001 CFG 4.5 1024 zip skipped (byte-identical); no other exclusions. Section P uses all 60 R3 prompts. The ZIB comparisons use the 57 prompts shared with the ZIB archive where the row says so.

## Scorecard (section P, 16 rows): 11 confirmed, 5 refuted

| ID | Result | Verdict |
|---|---|---|
| P1a | SD35M pooled vs branch-mean, steps ≥ 2: median log g ratio **+0.071** (86% of cells positive) | CONFIRMED |
| P1b | ZIB pooled vs branch-mean: **+0.265** attn, **+0.282** mlp | CONFIRMED |
| P1c | Gap: SD35M −0.020, ZIB attn **+0.011**, ZIB mlp −0.004 | REFUTED (pooling raises g but leaves the gap essentially unchanged) |
| P1d | Control, SD35M step 0 (pooled in both): **+0.017** | CONFIRMED (the two builds agree when the regime matches) |
| P2a | Tangent ZIB CFG 4 branch-mean ↔ ZIT: **0.988** attn, 0.991 mlp (pooled archive: 0.942 / 0.967) | CONFIRMED: pooling |
| P2b | ZIB CFG 1 ↔ ZIB CFG 4 branch-mean: tangent 0.988 / 0.992, shape r 0.992 / 0.992 | CONFIRMED |
| P2c | Offset ratio branch-mean / pooled: **0.276** attn, 0.224 mlp | CONFIRMED |
| P3a | Seed-centred σ50, CFG 1: **0.742** (95% CI 0.69–0.97) | REFUTED (later than the predicted 0.85–0.96) |
| P3b | Seed-centred step-0 R², CFG 1: **0.127** (uncentred 0.445) | CONFIRMED |
| P3c | Seed-centred VR(4), CFG 1: **1.79** (uncentred 1.84) | REFUTED (the shared seed does not make it walk-like) |
| P3d | Uncentred VR(4), CFG 4.5 pooled: **0.74** | REFUTED (not > 1; between the two decision bands) |
| P4a | Legacy res suite / passive at matched σ and layer: median log ratio **−2.37** (10.7×); ρ(step, g) −0.99 legacy, +1.00 passive | CONFIRMED |
| P4b | −+−+− kept in ≥ 82% of draws even at **20%** RMS smooth perturbation (threshold > 20%) | CONFIRMED |
| P4c | First sign '+' in **60/60** single passive prompts; '−' in **30/30** single legacy generations. −+−+− in 0/60 vs 30/30 | CONFIRMED |
| P4d | Legacy-schema CFG study (256 px, same pyramids prompt, 100 steps, CFG 1): **+−+−**, 3 crossings | REFUTED |
| P5 | Onset σ: SD35M CFG 4.5 1024 0.810, 768 0.810, 512 0.777; ZIB CFG 4 0.805 attn, 0.779 mlp | CONFIRMED |

## Post hoc checks (labelled, run after scoring)

- **H1 (instrument behaviour by file).**
  - Res suite at 512/768/1024: median g 0.005–0.006, falls over steps (ρ −0.87 to −0.99), −+−+− at every resolution.
  - CFG study at 256 px: median g 0.018, *rises* (ρ +0.89), +−+−.
  - Passive at 512–1024, CFG 1 or 4.5: median g 0.064–0.073, rises (ρ ≥ +0.99), +−+−.
  - So "legacy schema" names two different behaviours.
- **H2 (the reviewer's 0.278 / 0.250).** Full-population offset ratio against the pooled archive:
  - ZIB CFG 1: 0.081 attn / 0.059 mlp;
  - ZIB CFG 4 branch-mean: 0.264 / 0.232.
  
  The reviewer's figures match the CFG 4 arm.
- **H3 (design check).** Archive SD35M with 4 runs per prompt gives σ50 0.936. Subsampled to 2 runs per prompt (20 draws) it gives σ50 0.90 (range 0.83–0.93) and VR(4) 0.51. Two seeds per prompt does not by itself produce the R3 CFG 1 values (σ50 0.74, VR(4) 1.8).

## Synthesis

**1. The Base→Turbo lineage result is now fully explained, and it is the strongest result in the series.**
- At cond-only capture, ZIB and ZIT have practically the same depth geometry (tangent 0.999).
- The old 0.943 came from pooled capture, not from guidance or distillation:
  - branch-mean capture of the same guided runs gives 0.988;
  - pooling alone, inside one model, opens a 0.946 gap (ZIB CFG 4 branch-mean vs pooled).
- The absolute-g offset breaks down as about 5% Base→Turbo difference, about 23% guidance plus the uncond branch, and about 72% pooling (Fig. 1). Per the Z-Image report, Turbo was distilled from the SFT checkpoint and then RL-tuned, so the 5% bundles distillation, RL and any checkpoint mismatch.
- For Mode-I: report lineage at cond-only capture, and state that the earlier 0.94 was a capture artifact.
- For Mode-II: the separation between trajectory geometry (inherited, about 0.999) and operating point (moved by guidance and capture) is now quantitative.

**2. Capture regime is a first-class variable.**
- Pooling raises g (+7% SD35M, +30% ZIB) without moving the gap (P1).
- When regimes match, two independent builds agree to 1.7% (P1d).
- Every Mode-I table of absolute g has to carry its regime. This is the new regime-map gate.

**3. The four-crossing topology belongs to the res-suite functional (B7).**

| Factor | Evidence |
|---|---|
| Level and trend | 10.7× below passive, and falling where passive rises |
| Robustness | Survives perturbations 10× larger than measured step-count effects |
| Prompt | Never appears for any single passive prompt |
| Traversal | Absent from the legacy-schema CFG study, which shared the prompt and the 100-step traversal |

- No dataset shows −+−+− except the one with that functional.
- The traversal confound is not formally excluded until the 100-step passive run (prereg R2, expected +−+−). It is now an unlikely explanation. The literature gives no general guarantee of step-count invariance at matched noise: Ning et al. 2024 find step-count dependence that grows late in sampling. So the ≤ 2% bound from ZIT may not transfer to SD3.5-M, and the confirmatory run keeps its value.
- Mode-I should not present four crossings as a model property. If it is mentioned, it is a property of the legacy capture.
- The passive three-crossing +−+− topology holds at 512, 768 and 1024, at CFG 1 and 4.5, and per prompt.

**4. Seed fate and persistence: the clean picture differs from the archive, and the cause is not yet identified.**
- Removing the shared-seed shortcut cuts step-0 predictability from 0.45 to 0.13. The fate onset moves to σ≈0.74 (Fig. 3).
- Seed deviations in the clean arm are persistent (VR(4) 1.8, unchanged by seed-centring). The pooled CFG 4.5 arm with the same seeds is jitter-like (0.74), and the archive branch-mean arm more so (0.51) (Fig. 4).
- A design artifact is ruled out (H3, P3c).
- The pattern matches Round 2. ZIT-attn, the only cond-only channel there, was also walk-like (VR 0.97/0.92) and committed late (σ50 0.66). Every cond-only channel measured so far is walk-like with a late onset; every mixed-capture channel is jitter-like with an early onset. All the cond-only channels are also CFG 1, so guidance remains confounded.
- The difference tracks the arm: cond-only CFG 1 vs mixed-capture CFG 4.5. Mixed capture adds the uncond branch, which is prompt-independent and noise-dominated early. That could inflate early predictability and add step-to-step jitter.
- Guidance itself could also do both. Only the per-branch capture (prereg R1d, R1e) separates them.
- Mode-I wording: "passive future-state predictability", with its regime stated. No commitment or speciation language before G3.

**5. The plateau onset at σ≈0.8 holds on every arm (P5).**
- All arms span 0.75–0.84 except the one ZIB CFG 1 attention value, one grid cell below 0.75.
- Report the onset with the schedule's grid resolution.

## What the literature changes (LIT_R4.md, 26 of 32 sources verified on a primary page)

- **Commitment needs continuations.** Biroli et al. 2024 (two clones from the same x_t, P(same class), speciation at φ≈0.775), Sclocchi et al. 2025 (re-noise to t and regenerate) and Li & Chen 2024 all measure commitment by launching continuations. G3 should copy that design. Our passive R² is not that quantity.
- **Guidance acts early in the literature.** Kynkäänniemi et al. 2024, CADS (Sadat et al. 2024) and Wang et al. 2024 place guidance's diversity loss at high noise. Our spread contraction shows up in the later three quarters (B6). It may be a delayed readout of an early cause. A guidance-interval rider (CFG only in the first quarter vs only in the last three quarters, at the R1 settings) would decide this. It is cheap, but Mode-III unless the CFG card stays in Mode-I.
- **Pooling and participation ratio.** Pooling two conditions can raise or lower PR. It rises when the conditions occupy partly different subspaces and falls when one large mean shift dominates. P1 found g up and the gap unchanged, which fits the first case: cond and uncond differ in subspace, not by one dominant offset.
- **Distilled models and commitment.** Gandikota & Bau 2025 report that distilled SDXL variants fix structure at the first step, while the base takes about 30% of steps. Round 2 found ZIT's seed-fate late (σ50 0.66), but with only 5–8% seed variance. These are compatible: little seed variance is left to commit. A matched clone test on ZIB vs ZIT (G3) would make this a clean Mode-III result.
- **Seed and layout.** Mao et al. 2023 and Li et al. 2025 support a seed-to-layout tendency, while Prompt-to-Prompt shows it is not a law. P3's shared-seed component (34% of residual variance at CFG 1) quantifies it for SD3.5-M.
- **Variance ratio.** Lo & MacKinlay 1988 is the citation for VR(k). Their own finding was VR > 1 (positive autocorrelation), the same direction as the clean arm here.

## What this means for the papers now

- **Mode-I can state now:**
  - clean lineage inheritance (0.999), with the capture decomposition;
  - plateau onset σ≈0.8 across 3 resolutions, 2 models and 2 guidance levels;
  - passive +−+− topology;
  - the regime map;
  - the step-count invariance from Round 2.
- **Mode-I should hold or hedge:**
  - seed-fate onset values;
  - persistence (walk vs jitter);
  - CFG spread contraction;
  - four crossings.
- **Mode-II gains** a worked example of how a measurement functional (pooling over a mixture) moves the PR-based operating point while leaving geometry intact. P1c is the formal handle: pooling raises g but not the gap.
