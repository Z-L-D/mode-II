# Round 18 preregistration: documented distillation pair, Z-Image session band on image rows, Base at shift 6

Frozen 2026-11-09, before any capture exists. Zac chose "Build R18 for these papers" on 2026-11-09 (10:19Z), after the round-5
reviews. R18 answers asks that the round-5 reviewers listed as score-changers and that existing data cannot answer:
- GLM M1 and DeepSeek Q3: a documented teacher→student pair. Both distillation pairs on file (Z-Image, ACE-Step) are vendor-labelled, and the student's initialisation is not published.
- Kimi M4 and GLM: the Z-Image image-row lineage value (0.928) has no image-row same-weights session band to compare with (RC22c).
- GLM Q2: Base was captured at shift 3 only.

Not in R18 (stays with Mode-III): more depth-matched unrelated pairs (no second 38-block model was available cheaply), the 30-prompt
held-out domain set, Base/Turbo at matched guidance, and Mochi/SD3.5 L centred and pruned row-split rechecks.

## Data to be captured
Capture uses `capture_code_r4/tier_controls_r18/run_r18.bat`, driver `m3_comfy.py capture`, and the CoS-PAFS Passive Telemetry Guider with `split_cfg_branches` on.
- Images are one per prompt at 1024 × 1024, with an empty negative prompt.
- Every model runs at its own recommended settings (Zac's rule).
- `r18_preflight.py` writes the settings each arm actually ran to `r18_settings.json`, and the SHA-256 of every denoiser file to `r18_weight_hashes.json`.

| Session | Model | Blocks | Prompts | Steps | CFG | Shift | Sampler / scheduler | spectral_variants |
|---|---|---|---|---|---|---|---|---|
| R18_SD35L | SD3.5 Large (Stability) | 38 | `prompts_r3_cond60.tsv`, seed 1001 | 28 | 3.5 | 3 | euler / sgm_uniform | off |
| R18_SD35LT | SD3.5 Large Turbo (Stability; ADD distillation of SD3.5 Large, documented on the model card) | 38 | same | 4 | 1 | 3 | euler / sgm_uniform | off |
| R18_ZIT2 | Z-Image Turbo, second session | 30 | same (as the R11 Turbo session ZITSV) | 8 | 1 | 3 | Zac's R11/R12 Z-Image workflow (template euler / simple only if that is absent) | on (image rows) |
| R18_ZB6 | Z-Image Base | 30 | `prompts_primary120.txt` (the Z1 prompts), seed 1001 | 25 | 4 | **6** | as Z1 (Zac's workflow) | off |

The SD3.5 Large settings are those on the Stability model cards (28 steps, guidance 3.5; Turbo 4 steps, guidance 0, which is CFG 1 in ComfyUI). MMDiT hooks read the image stream only, so for the SD3.5 arms image rows are all rows and spectral_variants is not needed. SD3/SD3.5 are hooked in attention only (registry).

Reference sets already on file:
- archive SD35L (485 generations, 28 steps, 38 blocks);
- R11 ZITSV (Turbo, CFG 1, 8 steps, spectral, 58 generations; image-row cubes V_ZITSV);
- R12 Z1 (Base, CFG 4, shift 3, 479 generations), Z2 (Base, CFG 1, cond60, with image rows V_Z2) and Z3 (Turbo at the Z1 settings);
- archive ZIT (Turbo, 480 generations).

## Methods (unchanged from earlier rounds)
- **Cubes**: the R11/R12 cube builder (`review/zib_turbo_mixup_2026-10-18/r12/scripts/build_cubes.py`), branch cond, mean over generations. Image-row cubes (V_*) come from the Round 11 row-split builder, using the columns g_img, gap_img and H_img.
- **Primary tangent**: the Round 6 S1 `tangent()` (`review/round6/scripts/r6c_common.py`), as in `score_r17.py`.
  - Canonical 24-point depth grid; the nearest matched σ in the second run; the first-named model supplies the states.
  - Null: layer-order permutation, 1000 draws, rng 17, p95.
  - **Lineage rule**: tangent ≥ null p95 + 0.30.
  - **Middle tier**: lineage rule met and tangent ≤ 0.95.
- **Secondary statistics** (reported, not scored): the RC22 smoothness-preserving surrogate p95 (`rc22_round5_checks.py`, 300 draws, rng 22) and the 90% interval from 200 draws of 30 generations.
- **Identity gate**: R12-0, the step-0 depth-profile Pearson r on gen-means. ≥ 0.85 means the same checkpoint; the §5.5 reading band is ≥ 0.98 for same weights.
- **σ clock**: raw flow σ for all four arms.

## Gates
| ID | Test | Pass if |
|---|---|---|
| R18-0a | Complete arms | ≥ 58 of 60 generations with full step grids (SD35L, SD35LT, ZIT2), ≥ 116 of 120 (ZB6). A failing session is reported and not scored. |
| R18-0b | Every block hooked | attn 38/38 for SD35L and SD35LT; attn and MLP 30/30 for ZIT2 and ZB6. For ZIT2: `n_img_rows` = 4096 on every row, and `n_txt_rows` is 32 or 64. |
| R18-0c | Right checkpoints | SHA-256 differ between SD35L and SD35LT and between ZIT2 and ZB6. The SD3.5 hashes equal the Hugging Face LFS sha256 of `sd3.5_large.safetensors` / `sd3.5_large_turbo.safetensors`. Step-0 r: ZIT2 vs ZITSV ≥ 0.98 (attn and MLP); ZB6 vs Z1 ≥ 0.85 and ZB6 vs ZIT < 0.85 (attn; the Round 12 Base check); R18 SD35L vs archive SD35L ≥ 0.85. |
| R18-0d | Schedules | ZB6: every σ equals the shift-6 image of Z1's grid, σ6 = 6u/(1+5u) with u = σ3/(3 − 2σ3), within 1e-3. ZIT2: σ grid equals ZITSV's within 1e-3. |

## Predictions and fixed decision rules
| ID | Statement | Scored as |
|---|---|---|
| R18-1 | The documented distillation pair SD3.5 Large ↔ Large Turbo sits in the middle tier | SD35L first, attention. **Supported** if the lineage rule holds and the tangent is ≤ 0.95. If supported, the papers gain a third distillation lineage pair, and the first one where the teacher→student link is documented. If the tangent is > 0.95, the papers say that a documented distillation pair can sit at the same-weights level, so the middle tier is not a property of distillation as such. If it is below p95 + 0.30, the papers say that a documented distillation pair did not meet the lineage rule, so the rule detects relatedness for some distillation recipes only. |
| R18-2 | Same weights across sessions stay at the top | R18 SD35L ↔ archive SD35L (attn) ≥ 0.95 (different session and prompt set); cond ↔ uncond within R18_SD35L ≥ 0.90 (attn) |
| R18-3a | Z-Image Turbo has a narrow image-row session band | ZIT2 ↔ ZITSV, image rows, ≥ 0.97 in attention and in MLP; all rows ≥ 0.98 in both |
| R18-3b | The image-row lineage value replicates with a new Turbo session | Z2 ↔ ZIT2, image rows: attention in [0.90, 0.95], MLP in [0.70, 0.80] (on file: Z2 ↔ ZITSV 0.928 and 0.754) |
| R18-3c | Lineage stays below the session band on image rows | (R18-3a attention) − (R18-3b attention) ≥ 0.03. If it holds, the papers say the Z-Image image-row lineage value lies below a two-session image-row band for the same weights. If it fails, the papers say that on image rows the Z-Image lineage pair cannot be told apart from a session change, and the identity gate's 0.85–0.98 band is read as undecided for it. |
| R18-4 | The Z-Image lineage result does not depend on Base's shift | ZB6 first, attention and MLP. **Supported** if ZB6 ↔ ZIT is within 0.10 of Z1 ↔ ZIT (on file: 0.707 attention, 0.719 MLP) and meets the lineage rule, and ZB6 ↔ Z1 (same weights, shift 6 vs 3) is ≥ 0.90. If supported, the papers answer GLM Q2 with the shift-6 value. Otherwise they report the shift dependence and limit the Z-Image lineage numbers to shift 3. |

**Descriptive only** (no prediction):
- Turbo-first order and native grid for every pair.
- RC22 surrogate p95 and margins for every new pair.
- SD35LT ↔ archive SD35L.
- ZB6 ↔ Z3 (Turbo at the Z1 settings).
- ZIT2 all rows vs image rows.
- RC24 field transfer on SD35L ↔ SD35LT.
- β-layer sign pattern and relative onset for the new arms.

## What changes in the papers
R18-1 to R18-4 enter the Mode-I tier table (Table 4) and §5.3, and the Mode-II §7.2 cross-references, marked as Round 18 preregistered results, whatever the outcome. The claim register and scorecard get the new rows. The triage list of round-5 deferred items is updated.
