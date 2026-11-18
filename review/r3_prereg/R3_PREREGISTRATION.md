# R3 preregistration: latent Λ (speciation), ACE-Step audio arm, UNet arm

Written 2026-10-08, around 15:27 UTC. The SHA-256 and UTC time are in `R3_PREREG_SHA256.txt`. None of the data these predictions concern exists yet: no latent spectrum has been computed, and no ACE-Step or UNet capture has been made.

**Disclosed prior looks:** the archive's E6 prompt-share curves (`review/round2/out/E6_varcomp_by_step.csv`) and E3 plateau onsets (`E3_plateau_onset.csv`) were read while writing Parts A and B. Every image-model number used below comes from earlier rounds (R2, lens round v0.24).

**Wording rule (standing):** a failed prediction is reported as "tested, not supported", never "excluded".

---

## Part A: latent Λ and speciation σ (no generation)

**Data:** `capture_code_r3/latent_spectrum.py`, run on images the models already generated:
- ZIT and ZIB with the Flux VAE;
- SD3.5 M and SD3.5 L with the SD3 VAE;
- up to 1000 images per model, at the most common resolution.

Λ is the top eigenvalue of the covariance of the flattened, `process_in`-scaled latents. For the rectified-flow path, the speciation σ is s_spec = √Λ / (1 + √Λ).

| ID | Prediction | Refuted if |
|---|---|---|
| L1 | Speciation is early: s_spec ≥ 0.95 for every image model | Any model has s_spec < 0.95 |
| L2 | Λ is stable: the split-half s_spec values differ by ≤ 0.005 in each model | Difference > 0.005 |
| L3 | **Link test.** s_spec falls inside the σ interval of the steepest rise in prompt share of g variance (E6, ANOVA share, by step). Seen intervals: ZIB 1.000→0.986; SD3.5 M 0.992→0.983; SD3.5 L 1.000→0.988; ZIT 1.000→0.955 | s_spec is outside the interval for 2 or more of the 4 models |
| L4 | Where ≥ 3 prompts have ≥ 8 images each, the within-prompt s_spec is below the pooled s_spec | within ≥ pooled |

**Caveat fixed in advance (L3):** the activation prompt share is already 0.36–0.92 at σ = 1 (step 0), because text conditioning acts before the image exists. So L3 tests timing only. Even a pass does not show that the activations "speciate".

**Consequence for G3 (decided now):**
- If the within-prompt s_spec lies outside the clone grid {0.97, 0.93, 0.86, 0.85, 0.79}, add it to the grid.
- If L1 holds and the within-prompt value is below 0.95, cloning at 0.97 straddles the predicted within-prompt speciation, so the clone design keeps 0.97.

---

## Part B: ACE-Step 1.5 audio arm (protocol `capture_code_r3/ace_step/ACE_STEP_PROTOCOL.md`)

**Data:**
- 30 prompts × 2 seeds, 180 s each, instrumental;
- `generate_audio_codes` off;
- attn + MLP on 24 layers.

**Analysis:** each prediction is run exactly as on the image archive, with the same scripts unchanged:
- B3 and B4: the lens-round scripts (`review/lenses/lens_L1_triple.py`, `lens_L2_rg.py`, `lens_L34_thermo.py`), applied to new channels "ACE attn" and "ACE MLP" with D = 2048;
- B6 and B7: the analyzer's `plot_beta_layer` (29 bins, ≥ 10 points), via `lens_L2_rg.py`;
- B8: E3 plateau onset;
- B10: E6 variance components.

The step count and CFG are whatever Zac's ACE workflow uses; they are recorded, not chosen here.

| ID | Prediction | Refuted if |
|---|---|---|
| B1 | **Instrument (gate, not science).** The preflight passes: 24 layers × {attn, mlp} on every step, seq_len 2250, hidden_dim 2048 | Any fails. Then no science analysis runs until fixed |
| B2 | **Anisotropy.** g / g_MP < 0.10 in every (step, layer) cell, where g_MP is the Marchenko–Pastur isotropic value at N = 2250 and D = 2048. Images: 0.0014–0.07 | Any cell is ≥ 0.10 in more than 5% of cells |
| B3 | **The triple is 2-D.** PCA of (H₁/lnD, H₂/lnD, ln gap) over rows gives PC1+PC2 ≥ 0.98. Images: 0.980–0.994 | < 0.98 in either channel |
| B4 | **β_layer is not autonomous in g.** Same R1 test: out-of-sample ΔR² (spline f(g) + layer vs f(g), 5-fold CV grouped by prompt) ≥ 0.30 in both channels. Images: 0.49–0.65 | ΔR² < 0.30 in either channel |
| B5 | **Layer parity.** Even layers use sliding-window self-attention (128 tokens) and odd layers full attention. In attn, the sign of (mean log g over even layers − mean over odd layers) is the same in ≥ 80% of (gen, step) cells. Direction not predicted | < 80% |
| B6 | **Layer-axis recurrence.** `plot_beta_layer` on ACE attn (pooled over σ) gives the image archive's 3 crossings (+,−,+,−), and 3 is the modal count in ≥ 50% of 100 prompt bootstraps. If B5 holds, the test is also run on even-only and odd-only sequences, and the verdict uses the raw sequence; the parity versions are descriptive | Point estimate ≠ 3, or modal-count share < 50% |
| B7 | **Crossing-3 slope** (only if B6 holds): y₃ lies inside the image envelope widened by 0.05, i.e. [−0.80, −0.15]. Images: ZIB attn −0.20, ZIB MLP −0.28, SD3.5 M −0.45, ZIT MLP −0.53, SD3.5 L −0.74 | y₃ ± 2 SE lies wholly outside that interval |
| B8 | **Plateau onset.** E3's onset σ lies in [0.73, 0.92]. Images: 0.78–0.87 | Outside |
| B9 | **No susceptibility peak.** The within-prompt variance of logit g has no interior maximum along σ that survives the PH3 rule: a peak more than 10% above both end values, with the step on each side lower. This held in 6/7 image channels | Present in either channel |
| B10 | **Prompt share rises first.** In E6's prompt share of g variance, the steepest rise is in the first step interval, as in every image channel | The steepest rise is later |

**What each outcome would mean (fixed now):**
- If B2–B4, B8 and B10 hold, those properties are not specific to image data.
- If B6 fails while B4 holds, the layer-axis crossing pattern is image-specific, and the "coarse envelope recurrence" scope in Mode-II stays image-only.
- If B6 and B7 both hold, the recurrence crosses modality. That is still not an RG fixed point (B4), but it is the strongest result available for the F4 future test.

---

## Part C: UNet arm (only if SD1.5 or SDXL is captured with R3 nodes.py)

| ID | Prediction | Refuted if |
|---|---|---|
| C1 | Per-block g / g_MP < 0.10, each block judged against its own MP value at its own (N, D) | ≥ 0.10 in more than 5% of cells |
| C2 | R1 autonomy fails as in the DiTs: ΔR² ≥ 0.30 for g + block index over g alone | < 0.30 |
| C3 | U-Net level (D) explains part of the depth effect: within-level ΔR² is still ≥ 0.15 | < 0.15 |

The UNet arm has no layer-axis crossing prediction, because D and N change across blocks and the 29-bin β(g) pooling across levels is not meaningful.

---

## Not preregistered here

- The per-token block coarse-graining (real RG) test: it needs a new capture mode and gets its own preregistration.
- The optional generate_audio_codes arm.
- ACE-Step v1.

---

## Amendment 1 (15:27 UTC, before any data): B9 rule corrected

B9 described the peak rule wrongly; the rule as written there was never used. The rule that applies is the one actually used for L4 and PH3: `interior_peak()` in `review/lenses/lens_L34_thermo.py`. A peak counts only when all three of these hold:
- its argmax is interior (at least 2 steps from either end; 1 step when a run has 8 steps or fewer);
- it is stable (the bootstrap argmax lands within ±2 steps, or ±1, in ≥ 80% of 100 prompt bootstraps);
- it beats both end values by more than 2 bootstrap SE.

It is applied to the logit-normalised within-prompt variance, `chi_logit()` in `lens_posthoc_lit.py`. Nothing else in the file changes.
