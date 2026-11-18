# Round 17 preregistration: two more depth-matched tier controls for Mode-I/II

Frozen 2026-11-07, before any capture exists. Zac approved these two tests on 2026-11-07 (21:57Z). They answer review asks that
were set aside for lack of data: (1) the papers have only one depth-matched unrelated pair (ACE-Step <-> SD3.5 M, across modalities);
(3) both lineage pairs are distillation pairs, so it is open whether the middle tier detects distillation or just a shared vendor/recipe.

## Data to be captured
`capture_code_r4/tier_controls/run_tier_controls.bat`, driver `m3_comfy.py capture`, CoS-PAFS Passive Telemetry Guider with
split_cfg_branches and spectral_variants on, the 60 prompts of `prompts_r3_cond60.tsv` (the R3/R11 spectral prompts), seed 1001,
one image per prompt, 1024 x 1024, empty negative prompt. Every model runs at its own recommended settings (Zac's rule):

| Session | Model | Blocks | Steps | CFG | Sampler / scheduler | Clock |
|---|---|---|---|---|---|---|
| R17_PIXART | PixArt-Sigma XL-2 1024-MS (Huawei/PixArt) | 28 | 20 | 4.5 | dpmpp_2m / normal | VP-eps sigma -> t = sigma/(1+sigma) |
| R17_COSMOS2B | Cosmos-Predict2 2B text-to-image (NVIDIA) | 28 | 35 | 7.0 | euler / simple | EDM sigma -> t = sigma/(1+sigma) |
| R17_SD3M | SD3 Medium (Stability) | 24 | 28 | 7.0 | dpmpp_2m / sgm_uniform, shift 3 | flow sigma (raw) |
| R17_SD35M | SD3.5 Medium (Stability, MMDiT-X) | 24 | 40 | 4.5 | dpmpp_2m / sgm_uniform, shift 3 | flow sigma (raw) |

Reference set already on file: R11 SD3.5 M spectral (40 steps, CFG 1, attention, 59 of the 60 prompts; `RESULTS_R11_R12.md`).

Why these pairs: PixArt-Sigma and Cosmos-Predict2 2B are both 28-block text-to-image DiTs with T5-family encoders from unrelated
labs (Wan was rejected: Alibaba, like Z-Image, and video). SD3 Medium and SD3.5 Medium are both 24-block Stability MMDiTs with the same
three text encoders; SD3.5 Medium is a new model (MMDiT-X, new training), not documented as distilled from SD3 Medium.

## Methods (unchanged from earlier rounds)
Cubes: the R11/R12 cube builder (`review/zib_turbo_mixup_2026-10-18/r12/scripts/build_cubes.py`), branch cond, gen-mean.
Tangent: Round 6 S1 `tangent()` (`review/round6/scripts/r6c_common.py`) as used by `score_r12.py`: canonical 24-point clock grid
(primary), nearest matched clock value in the other run; null = layer-order permutation (same permutation at every clock value),
1000 draws, rng seed 17, p95. Native grid: secondary, reported only. Identity gate: R12-0 (step-0 depth-profile Pearson r, gen-mean;
>= 0.85 = same checkpoint). Hook sites: attention output projection for all four; MLP output (PixArt fc2, Cosmos mlp.layer2) for the
two test-1 models. SD3/SD3.5 are attention only (registry).

## Gates
| ID | Test | Pass if |
|---|---|---|
| R17-0a | Complete arms | >= 58 of 60 gens with full step grids per session; a failing session is reported and not scored |
| R17-0b | Every block hooked | attn: 28/28 (PixArt, Cosmos), 24/24 (SD3 M, SD3.5 M); MLP 28/28 for PixArt and Cosmos, else MLP is not scored for that pair |
| R17-0c | Right checkpoints | step-0 r: PixArt vs Cosmos < 0.85, SD3 M vs SD3.5 M < 0.85 (attn); R17_SD35M vs R11 SD3.5 M >= 0.85 |

## Predictions and fixed decision rules
| ID | Statement | Scored as |
|---|---|---|
| R17-1 | PixArt <-> Cosmos is in the unrelated tier | Supported if the tangent (PixArt first) is below the pair's null p95, attn and MLP (the R11-3 rule). Consequence fixed now: if supported, the papers have a second depth-matched unrelated pair, and the first one within one modality. If the tangent is >= 0.47 (the RC3 threshold) in either channel, the papers say the bottom tier did not replicate on a same-modality depth-matched pair. Between p95 and 0.47: reported as above null but below the lineage band. |
| R17-2 | SD3 M <-> SD3.5 M (attn): decision rule, no predicted direction | Middle tier: tangent >= null p95 + 0.30 and <= 0.95 (the R12-6 rule). Then the papers say the middle tier also appears for a same-vendor pair with no distillation link (a second case after the Qwen pair), so it marks shared lineage in the broad sense, not distillation. Unrelated tier: below null p95. Then the papers say the middle tier was not produced by shared vendor, encoders and depth alone. Anything else (between p95 and p95 + 0.30, or > 0.95) is reported as it is, with no tier claim. |
| R17-3 | Same-weights references stay in the top tier | R17_SD35M <-> R11 SD3.5 M (different CFG, 4.5 vs 1, as R12-4/5) >= 0.94; cond <-> uncond within each new arm >= 0.90 (attn; MLP where hooked) |
| R17-4 | Ordering inside the Stability family | SD3.5 M recapture <-> R11 SD3.5 M  >  SD3 M <-> SD3.5 M (attn) |

Descriptive only (no prediction): SD3 M <-> R11 SD3.5 M (CFG 1); both new pairs on the native grid; beta_layer sign pattern and plateau
onset for the four new arms; image-row vs caption-row split where the capture separates rows.

## What changes in the papers
R17-1 and R17-2 enter the tier table of Mode-I (§ on depth-matched tiers) and the Mode-II pooling/tier cross-references, marked as
Round 17 preregistered results, whatever the outcome. The claim register and scorecard get the new rows.
