# Mode Series Round 5: results and synthesis

2026-10-08. Predictions were frozen in `R5_PREREGISTRATION.md` (SHA-256 78b250a2…7018, 2026-10-08T05:24:22Z) before any section Q calculation ran.

**Scripts** (all Python, in `scripts/`, run in this order):
1. `r5_load.py`
2. `r5_scoreR.py`
3. `r5_audit_v016.py`
4. `r5_q_calcs.py`, which runs section Q exactly as frozen
5. `r5_scoreQ.py`
6. `r5_posthoc.py` and `r5_vr_null.py`, both labelled post hoc
7. `r5_figures.py`

**Outputs:** `out/Q_results.json`, `out/Q_scorecard.csv`, `out/posthoc_R5.json`, `out/H1_seedfate_vs_sample_size.csv`, `out/VR_null.json`. Figures are in `fig/`.

**Related files:**
- The reviewer audit and the Round 4 section R scorecard: `V016_AUDIT_AND_R_SCORING.md`
- Gates: `R5_GATES.md`
- Literature: `LIT_R5.md` (35 of 40 sources verified on a primary page)

**Filters:**
- One ZIB CFG 4 generation and one CFG 4.5 40-step pyramid generation are dropped (one nonfinite cell each).
- One Round 3 768 generation with 4 non-pooled steps is dropped from Q1d.

**Implementation note on Q3c (disclosed).** The rule ran as frozen. The estimator is undefined in 93.5% of partitions, because R² never reaches 0.5 even at the final step. That outcome is reported as REFUTED, not as missing.

## Scorecard (22 rows): 13 confirmed, 9 refuted

| ID | Result | Verdict |
|---|---|---|
| Q1a | The new CFG 1 run is **identical** to the Round 3 CFG 1 run in 100% of cells (to 6 decimals, 60 paired generations) | CONFIRMED |
| Q1b | The new ZIB CFG 4 synthetic branch-mean equals the Round 3 "branch-mean" to 2e-16 (attn and mlp); ZIB CFG 1 retest is identical | CONFIRMED |
| Q1c | Same trajectory, 1024: pooled/BM median **+0.040** (band 0.04–0.10, met by 0.0003); 100% of cells > 0; gap 1.5e-5 | CONFIRMED (marginal) |
| Q1d | 768 **+0.050**, 512 **+0.088** | CONFIRMED |
| Q1e | √g_pooled ≥ (√g_c+√g_u)/2 in **100%** of cells (median slack 0.022 log) | CONFIRMED |
| Q1f | Pooled excess vs (1 − prompt_influence) across step × layer: ρ **0.83** | CONFIRMED |
| Q2a | ZIB CFG 4 **cond** ↔ ZIT tangent **0.992 / 0.992** | CONFIRMED |
| Q2b | uncond ↔ ZIT 0.981 / 0.989, below cond | CONFIRMED |
| Q2c | ZIB CFG 1 ↔ CFG 4 cond 0.991 / 0.993 | CONFIRMED |
| Q3a | Seed-centred σ50 partition 5–95% width: 0.063 with 60 prompts vs 0.111 with 30; medians **0.733 vs 0.524** | CONFIRMED |
| Q3b | Paired σ50(4.5 cond) − σ50(1 cond): median +0.030, positive in 66.5% of partitions | CONFIRMED (weak) |
| Q3c | Archive on the 30 core prompts, 2 runs: σ50 undefined in 93.5% of partitions | REFUTED |
| Q4a | Unique seeds, pyramid CFG 1: VR(4) **1.95** | CONFIRMED |
| Q4b | Pyramid CFG 4.5: cond 0.86 > uncond 0.66 > BM 0.64 (predicted BM > uncond) | REFUTED |
| Q4c | VR(10) at 100 steps / VR(4) at 40: 1.89 and 2.22 | REFUTED |
| Q5a | CFG 1 40→100 shift 0.0326 > seed-noise null p95 0.0266 (p 0.023); 100-step g higher in 88% of cells | REFUTED |
| Q5b | CFG 4.5 cond shift 0.0068 < null p95 0.0216 | CONFIRMED |
| Q6a | Low-g lobe z: 2.13 (CFG 1 1024), 1.96, 0.05, 1.82, 0.43, 0.15 | REFUTED (one arm at 2.13) |
| Q6b | Round 3 pooled 512 on the 29 shared prompts: +−+− | REFUTED |
| Q7a | Pyramid (unique seeds) contraction, last three quarters: 0.73 (40 steps), 1.35 (100 steps) | REFUTED |
| Q7b | First quarter: 0.45 (40), 0.59 (100). Contraction is early here | REFUTED |
| Q8 | Onsets 0.72–0.89; 512 cond **0.721** is below 0.75 | REFUTED |

## Post hoc checks (labelled, run after scoring)

**H1. Seed-fate estimator vs sample size** (`out/H1_seedfate_vs_sample_size.csv`, Fig. 2). In the archive, σ50 rises steadily with the number of generations n in the ridge regression:

| Archive subsample | n | σ50 |
|---|---|---|
| 30 prompts × 2 runs | 60 | undefined in 98% of partitions |
| 60 × 2 | 120 | 0.720 |
| 30 × 4 | 120 | 0.867 |
| 120 × 2 | 240 | 0.843 |
| 60 × 4 | 240 | 0.917 |
| 120 × 4 | 480 | 0.936 |

The final-step R², which is the ridge ceiling, rises with n in the same way (0.48 → 0.89). A ceiling-free version is the σ at which R²(t)/R²(final) crosses 0.5, called the relative onset. It barely moves:
- archive: 0.87–0.96 at every n;
- every new seed-centred arm at n = 60: 0.896–0.914;
- Round 3 CFG 1 centred: 0.917.

**H2. VR and step granularity.** Thinning the 100-step arms to the 40-step σ grid gives these VR(4) values:

| Arm | 100 steps, thinned | 40 steps |
|---|---|---|
| CFG 1 | 2.00 | 1.95 |
| CFG 4.5 cond | 1.41 | 0.86 |
| CFG 4.5 uncond | 1.24 | 0.66 |

So for CFG 1, VR is a property of the σ span, and Q4c failed only because a 1-step increment at 100 steps is a different unit. The CFG 4.5 arms differ by more than that. They use different seed sets (13 vs 16 seeds), so this is not resolved.

**VR calibration** (`out/VR_null.json`, Fig. 3). With the same estimator and data shape, the reference values are:
- a pure random walk: VR(4) **0.82**
- AR(1) with ρ 0.8: 0.72
- independent jitter: 0.25

The estimator standardises every step, so a walk does not land at 1. On this scale:
- cond CFG 1 (1.85–1.95) and cond CFG 4.5 core (1.25) are **trending**: increments are positively autocorrelated, beyond a walk.
- branch-mean (0.79) is walk-like.
- uncond (0.59) and the archive branch-mean (0.41–0.51) are mean-reverting.

The "≤ 1.0 vs > 1.2" bands used in Rounds 3–4 were not calibrated to this estimator. The literature step independently flags small-sample VR bias (Kim, Nelson & Startz; Chen & Deo, in `LIT_R5.md`).

**H3. Resolution shift** (Fig. 6). The SD3 paper's resolution shift is t' = a·t/(1+(a−1)t), with a = 1024/res. Applying it to the cond-branch onsets moves 512 / 768 / 1024 from 0.721 / 0.759 / 0.810 to **0.838 / 0.808 / 0.810**, so the spread shrinks from 0.09 to 0.03. The Round 3 pooled onsets overshoot after the shift (0.874 / 0.851 / 0.810), so the realignment is partial. ComfyUI's ModelSamplingSD3 applies a fixed shift of 3.0 at every resolution (verified in source, `LIT_R5.md` §2). Equal σ is therefore not equal effective SNR across resolutions.

**H4. Contraction timing** (Fig. 5).
- In the paired core, the median ratio is about 1 through σ ≈ 0.8 and then falls to 0.43–0.45. Only 27% of prompts fall below 0.8 in the first quarter.
- In the unpaired pyramid arms, the first quarter is 0.45 (95% seed bootstrap 0.32–0.66) at 40 steps and 0.59 (0.33–0.93) at 100 steps. The late quarters are wide (0.44–2.28) and include 1.
- Pairing forces the ratio toward 1 at the first steps, because both arms start from the same noise. The pyramid is one 1,370-character prompt.
- The timing difference is therefore real for this prompt, but it is not established in general.

**H5. Round 4 P1a decomposed.** On the 30 core prompts:
- pooled (Round 3) vs archive branch-mean: +0.070, which reproduces P1a;
- pooled vs new branch-mean on the same trajectory: +0.040;
- new branch-mean vs archive branch-mean (different runs and build): +0.018.

About 0.04 of P1a is pooling on one trajectory. The rest is the difference between runs and builds.

## Synthesis

**1. The instrument is now validated end to end, and pooling has an explanation that is both theorem and measurement.**
- *Deterministic replay.* The new CFG 1 run reproduces Round 3 bit for bit (Q1a). The new per-branch ZIB run reproduces Round 3's "branch-mean" as exactly the mean of the two branches (Q1b).
- *What pooling does.* On a single trajectory (same prompts, seeds, sampler and latents), pooling the two branches raises g by +0.040 at 1024, +0.050 at 768 and +0.088 at 512. It leaves the spectral gap unchanged to 1e-5 (Q1c, Q1d).
- *Why.* Pooling averages the uncentred second-moment matrices, and tr √M is concave. So Σs_pooled ≥ (Σs_c + Σs_u)/2, with equality only when the branches have identical second moments. The bound holds in every cell (Q1e), and the excess follows how different the branches are (ρ 0.83 with 1 − prompt_influence, Q1f).
- *A caveat.* The inequality on g itself assumed equal branch norms. In general it holds against the energy-weighted branch mean, and branch norms are not logged (`LIT_R5.md` §1).
- *What it means for Mode-II.* This gives a worked, provable example of a measurement functional moving the PR operating point while leaving the gap and the depth geometry untouched.
- *The excess is largest at the first steps* (+0.27 at σ = 1, Fig. 1). That is where cond and uncond differ most, which agrees with RAAG/MAMBO-G's measurements on SD3.5 (`LIT_R5.md` §3).

**2. Lineage: the cond branch of guided ZIB carries ZIT's geometry (0.992), and the ladder is clean** (Fig. 4).

| Comparison with ZIT | Tangent |
|---|---|
| ZIB CFG 1 | 0.999 |
| ZIB CFG 4 cond | 0.992 |
| ZIB CFG 4 branch-mean | 0.988 |
| ZIB CFG 4 uncond | 0.981 |
| ZIB CFG 4 pooled | 0.942 |

- Guidance moves cond geometry by under 1%.
- The uncond branch is the part least like Turbo.
- Pooling is the only large step.
- Mode-I can state lineage at cond-only or cond-branch capture as 0.99–0.999.

One open check: the official Z-Image Base config uses a sampling shift of 6.0, while ComfyUI's default is 3.0 (`LIT_R5.md`). Zac should confirm which was used. This does not touch tangents of 0.99 or higher, but it matters for any σ-matched Base vs Turbo level.

**Update 2026-10-08 (after Zac's answer, no shift node used):** resolved. The logged σ schedules fit shift = 3.000 exactly at every step in the ZIB archive, the ZIT archive and both R4 ZIB arms (`scripts/r5_shift_fit.py`). Base and Turbo share one σ map, so σ-matched Base vs Turbo comparisons stand as captured. Base ran below its official 6.0, so describe it as "Z-Image Base at shift 3.0".

**3. Seed-fate onset numbers so far are mostly sample-size numbers. This changes how Rounds 2–4 should be read.**
- σ50 is the σ at which ridge R² crosses 0.5. With ridge α = 100 and 60–480 generations, the R² ceiling itself depends on n, so σ50 rises with n (H1).
- The "early onset 0.89–0.94 in mixed arms vs late 0.74 or 0.52 in cond-only arms" pattern of Rounds 2–4 was largely a comparison across different n:
  - archive n = 480 vs Round 3 n = 120 vs new n = 60;
  - on identical data, 60 prompts give 0.73 and 30 prompts give 0.52 (Q3a).
- With the relative onset, every arm sits at 0.90–0.92 at matched n: cond, uncond, branch-mean, CFG 1 and CFG 4.5.
- Guidance shifts onset by +0.03 at most, below the partition noise (Q3b).
- The archive's 2-run subsample on the core prompts cannot even reach R² 0.5 (Q3c).
- Practical rules:
  - Report fate onset only at matched n, as a distribution over at least 50 random prompt-grouped partitions (Varoquaux 2018).
  - Prefer the relative onset.
  - Never compare σ50 across datasets of different size.
  - Passive predictability is still not commitment; G3 is unchanged.

**4. Persistence: single-branch seed deviations trend, and mixing toward uncond pulls them back. Unique seeds do not change this.**
- On the calibrated scale (walk = 0.82):
  - cond CFG 1 is strongly trending: 1.85 shared seeds, 1.95 unique seeds;
  - cond CFG 4.5 trends less: 1.25 core, 0.86 pyramid;
  - the uncond branch is mean-reverting: 0.59–0.66;
  - the scalar branch-mean sits between them: 0.64–0.79.
- The shared-seed explanation (Round 4 P3c) is now dead twice over: seed-centring and unique seeds both leave CFG 1 trending.
- The archive's jitter-like 0.41–0.51 came from branch-mean capture under guidance, consistent with this ordering.
- Q4b's predicted order (BM above uncond) failed narrowly in the pyramid arm (0.64 vs 0.66).
- VR is a σ-span property for CFG 1 (H2). Compare arms only at matched step grids.

**5. Step count matters slightly without guidance and not at all with it.**
- At CFG 1, the same-prompt 40→100 shift is +3.1% in g. 88% of cells move up, and the shift exceeds seed noise (Q5a, p = 0.023).
- At CFG 4.5 the shift is 0.7% and inside noise (Q5b).
- Both are far below the 10× gap between the legacy res suite and passive capture.
- The legacy four-crossing card stays retired as a model property (R2a). Ning et al. (exposure bias) and Karras et al. explain why a small, late, step-count effect is expected.

**6. Crossing count is the wrong summary. Report the main crossing plus a lobe flag.**
- Every 30-prompt passive arm has the same main +→− crossing at g* = 0.074–0.085 (the single pyramid prompt: 0.058–0.059).
- The extra pair in +−+− is a lobe 0.1–6% as tall as the main peak. It sits within 0.00004–0.003 in g of the first crossing and is at most 2.1 bootstrap SE from zero (Q6a). It appears or disappears with the prompt sample and the functional: pooled 512 shows it on the same 29 prompts where the cond branch does not (Q6b).
- The pyramid prompt never shows it.
- Agreed with v0.16 that crossing count is not a universal invariant. The reason is that it is a noise-floor feature, not that resolution changes the topology.
- The H3 shift suggests that resolution comparisons should be made on a resolution-corrected clock in any case.

**7. Guidance contraction is real on the cond branch, and its timing is not settled.**
- Paired core (R1f): 0.64 over the last three quarters, about 1 early.
- Unpaired unique-seed pyramid: contraction at the first steps (0.45), then partial recovery at 40 steps and expansion at 100 steps.
- Two things could explain the difference: pairing pins the early ratio near 1, and the pyramid is a single, very long prompt.
- The literature allows both readings. Guidance's diversity loss is early (Kynkäänniemi 2024, CADS). Jin et al.'s stage-wise analysis adds a late within-mode contraction.
- Only the guidance-interval rider, run with unique seeds across 30 prompts, separates them. Its priority rises for the Mode-I CFG card.
- **Update 2026-10-08 (Zac):** the rider moves to Mode-III, since it is a perturbation experiment. Mode-I states that guidance contracts g (R1f 0.637) and does not state when; timing is named as open.

**8. Plateau onset holds on the cond branch at 1024 and 768 and on ZIB, and moves to 0.72 at 512** (Q8 refuted). On the SD3 resolution-corrected clock the cond onsets are 0.81–0.84 (H3). Mode-I should give the onset with its resolution and schedule shift, not as a single σ.

## What this means for the papers

- **Mode-I can now state:**
  - capture regime as a methods requirement, with the deterministic-replay validation;
  - lineage of 0.99–0.999 at cond capture;
  - cond-branch guidance contraction (0.64, paired, late), with timing flagged as prompt-dependent;
  - single-branch persistence ordering on a calibrated VR scale;
  - plateau onset at about 0.8 on a resolution-corrected clock;
  - the main β crossing g*.
- **Mode-I should drop or rewrite:**
  - every σ50 comparison across datasets of different size;
  - "early vs late commitment" contrasts;
  - crossing-count statements;
  - the four-crossing card;
  - VR thresholds at 1.0.
- **Mode-II gains** a theorem (concavity of tr √M) that explains the pooling shift exactly, plus an empirical confirmation on identical trajectories.
- **Mode-III** gets the guidance-interval rider, the Z-Image shift check, G3 clones and G4/PSTF, unchanged. (Update 2026-10-08: the Z-Image shift check is resolved at 3.0, and the rider is confirmed as Mode-III only.)
