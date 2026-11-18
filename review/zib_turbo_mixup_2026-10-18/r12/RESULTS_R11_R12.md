# Round 12 and Round 11 results (real Z-Image Base, 2026-10-27)

Scored exactly as frozen in `../rerun/PREREG_R12.md` (sha ad06a1bb) and `../../peer_review_2026-10-17/capture/PREREG_R11.md` (sha 3834abb0) with addendum 1. Anything not in those files is labelled post hoc. Scripts: `scripts/` here and `review/peer_review_2026-10-17/r11/scripts/`. Outputs: `out/` in both places.

## Data received and gates

| Arm | Run | Identity gate (step-0 r with Turbo, attn/MLP) | Preflight | Used |
|---|---|---|---|---|
| Z1 | Base, 120 × 4, 25 steps, CFG 4, random seeds | 0.649 / 0.586: **not Turbo** | 5 gens with one blank g cell, 1 interrupted gen (20 steps), 4 surplus gens of one prompt | cond: 479 attn / 478 MLP gens (first 4 complete per prompt), all 120 prompts, all at 1024 px |
| Z3 | Turbo matched to Z1, 120 × 1 | 1.000 / 1.000: Turbo, as intended | pass | 120 gens |
| Z2 | Base spectral, 60 prompts, seed 1001, CFG 1 | 0.651 / 0.585: **not Turbo** | pass (variant columns present, row split correct) | 60 gens |
| ZIT spectral | Turbo, 60 prompts, 8 steps, CFG 1 | 1.000 / 1.000: Turbo | 2 prompts missing (jumping spider, ancient tree) | 58 gens |
| SD3.5 M spectral | 40 steps, CFG 1, attn | n/a | one prompt swapped (vertical farm for the Soviet pool); **bit-identical to Round 3** on the other 59 | 59 of the 60 prereg prompts |
| AS15T audio matched | ACE-Step 1.5 **Turbo** on the 120 image prompts, 6 steps | 0.985 / 0.994: Turbo | pass | Not part of R11/R12 (Mode-III). ACE Base is not run yet, so R12-6/7 are not scored. |

The two "prompt not in the frozen list" flags on Z1/Z3/ACE are the sea-otter and rusted-chain prompts, which start with a newline in the TSV (known quirk). The 4128 vs 4160 sequence lengths are the 32- vs 64-row caption padding; every image is 4096 rows (1024 px).

## Round 12: lineage (all preregistered tests that could be scored hold)

| Pair (cond branch) | attn tangent | MLP tangent | Null p95 (attn / MLP) |
|---|---|---|---|
| Base Z1 ↔ Turbo-8 archive (lineage) | **0.707** | **0.719** | 0.292 / 0.198 |
| Base Z1 ↔ Turbo matched Z3 (lineage, identical settings) | **0.713** | **0.697** | 0.288 / 0.213 |
| Base Z2 (CFG 1) ↔ Turbo-8 | 0.715 | 0.718 | 0.273 / 0.204 |
| Turbo matched Z3 ↔ Turbo-8 (same weights, different steps and CFG) | 0.990 | 0.992 | |
| Largest unrelated (Z-Image vs Mochi / SD3.5 M / SD3.5 L) | 0.212 | n/a | |

- R12-1 (above null), R12-2 (below same weights), R12-3 (above unrelated + 0.30), R12-4 (same at identical settings), R12-5 (CFG 1): **all supported.** The paper consequence fixed in advance applies: the lineage claim is restated as a middle tier, same weights (0.99) > distillation lineage (0.70–0.72) > unrelated (≤ 0.21).
- Identity cross-checks (descriptive): Z1 ↔ May Base run 0.970 / 0.991, so the May run was Base. Z3 ↔ archive "ZIB" 0.948 / 0.977, so the archive was Turbo. Z1 cond ↔ uncond 0.996 / 0.963.
- R12-6 and R12-7 (ACE Base): not scored, because ACE Base has not been run.

## R12-8 (descriptive): the layer-axis +−+− is not a Base property

Analyzer `plot_beta_layer` unchanged, 100 prompt bootstraps.

| Data | attn point / modal count (share) | MLP point / modal count (share) |
|---|---|---|
| **Base Z1 cond** | 1 crossing / 1 (0.50; 3 in 0.11) | 1 / 1 (1.00) |
| Turbo matched Z3 | 1 / 1 (1.00) | 1 / 1 (1.00) |
| Turbo-8 archive | 1 / 1 (1.00) | 3, +−+− / 3 (0.65) |
| Archive "ZIB" (Turbo, 25 steps, pooled) | 3, +−+− / 3 (0.59) | 3, +−+− / 3 (0.69) |

Post hoc sensitivity (`out/r12_8_crossings_posthoc_sensitivity.csv`): with 1 gen per prompt the archive "ZIB" attn is still modal 3 (0.66), and Base stays modal 1. The Base **uncond** branch gives a point estimate of +−+− in attn (bootstrap 0.44). The Z-Image layer-axis three crossings are fragile, and they do not appear in the real Base cond branch. The papers' Z-Image +−+− statements came from the Turbo archive.

## Round 11: spectral variants (2 of 7 hold)

Base = Z2, Turbo = ZIT spectral, cond, the 60 preregistered prompts.

| ID | Prediction | Result | Verdict |
|---|---|---|---|
| R11-1 | Base↔Turbo V1 (image rows) ≥ 0.95, within 0.03 of V0 | attn 0.928 (V0 0.720), MLP 0.754 (V0 0.714) | REFUTED |
| R11-2 | V3 (centred) and V4 (top-8 removed) ≥ 0.90 | V3 attn 0.762, MLP 0.615; V4 attn 0.679, MLP 0.435 | REFUTED |
| R11-3 | Unrelated (vs SD3.5 M, attn) below permutation p95 in V1/V3/V4 | all −0.13 to 0.02, p95 0.22–0.25 | SUPPORTED |
| R11-4 | Lineage margin ≥ 0.50 in V3 and V4 (attn) | 0.738 and 0.754 | SUPPORTED |
| R11-5 | Backbone PC1 ≥ 0.70, beats all 999 depth-shift nulls | every cell beats its nulls (p = 0.001); PC1 below 0.70 in Base attn V0 (0.638) and Base MLP V3 (0.676) | REFUTED (2 of 20 cells) |
| R11-6 | Base MLP clock-axis +−+ in ≥ 90% of bootstraps | Base +++ in V0, V1, V3; ++− in V4; +−+ in 0%. Turbo V0 +−+ in 100%, but +++ in V1/V3/V4. | REFUTED |
| R11-7 | Caption rows < 10% of energy in ≥ 90% of Z-Image cells | ≥ 10% in 56% (Base attn), 79% (Base MLP), 50% (Turbo attn), 76% (Turbo MLP) of cells | REFUTED |

**The thresholds in R11-1 and R11-2 were set when the archive "Base↔Turbo" pair was believed to be lineage (about 0.95). It was really Turbo against Turbo.** Against the real lineage level (0.72 in V0), image rows and centring do not reduce Base↔Turbo agreement in attention (0.93 and 0.76). They reduce it in MLP (0.75 → 0.62 → 0.44). Lineage stays far above unrelated in every variant (R11-3, R11-4). The frozen §7 text ("R11-2 fails: the lineage separation is a property of the uncentred moment") is triggered by the letter of R11-2. The data say the separation from unrelated models survives centring, and what changes is the MLP lineage level.

**The largest finding is R11-7.** From MLP layer 8 onward, the 32–64 caption rows carry 99–100% of the uncentred second moment, in both Base and Turbo (median share by layer in `out/r11_diagnostics.json`). Deep attention layers reach 40–100%. Where that share is ≥ 0.99, the all-rows g equals the caption-only g (Base MLP ratio 1.08). Where the share is < 0.10, the all-rows g equals the image-row g (ratio 1.01–1.05). So **the Z-Image MLP numbers in both papers (V0) are, from layer 8 down, a statistic of the caption tokens, not the image.** The Turbo clock-axis +−+ (R11-6) is carried by those rows, because it disappears in image rows.

Descriptive (prereg, `out/r11_beta_layer_by_variant.csv`): SD3.5 M attn has the layer-axis +−+− in V0 (0.98 of bootstraps) and V4 (1.00), and loses it when centred (V3: 0.00). In Z-Image Base the MLP V0 pattern (+−+−, 0.74) vanishes in image rows (1 crossing, 1.00).

## What this means for the papers (not edited; needs Zac's go-ahead)

1. Lineage: restate as a middle tier with the Z1/Z3 numbers (pre-agreed consequence of R12-1 to R12-3).
2. Every Z-Image "Base" number is replaced from Z1, or relabelled as Turbo-25 (IMPACT_AUDIT.md).
3. Z-Image layer-axis +−+− and the Z-Image MLP clock-axis +−+: no longer supported for Base. They are scoped to Turbo, or dropped.
4. Z-Image MLP statistics: add the caption-row dominance as a scope statement, or report image-row (V1) values for Z-Image where available. This is a new caveat on every Z-Image MLP result.
5. SD3.5 M: the layer-axis +−+− survives removing massive channels but not centring. It is a property of the uncentred moment.
