# R18 addendum B results: activation CKA baseline (scored 2026-11-11)

Prereg `PREREG_R18_ADDENDUM_B.md` (sha 24a739d4), amendments B.1 and B.2 / B.2a (`PREREG_R18_ADDENDUM_B2.md`). Zac ran
`cka_r18b.py` (B.2a version) on his machine; outputs in `cka_out/` (29 pairs in attention, 6 in MLP). Primary statistic: token
CKA, symmetric best match, mean of four states, attention; null 200 prompt permutations (smallest possible p = 0.005).

Data note (B.2): Z-Image Base sketches overflow float16 in 12 of 30 attention layers at step 0 (2 layers at t = 0.75 and 0.5,
none at 0.25) and in up to 9 MLP layers; Turbo in up to 2 MLP layers. Those layers are left out of every Z-Image Base value.

## Rules
| Rule | Verdict | Numbers |
|---|---|---|
| B1 related and same-weights pairs above null p95 | **Supported** | SD3.5 M CFG 4.5 vs 1: 0.612 (p95 0.385); SD3.5 L vs L Turbo 0.625 (0.352); SD3 M vs SD3.5 M 0.462 (0.266); Z-Image Base vs Turbo 0.344 (0.132). All p = 0.005. |
| B2 CKA separates related from unrelated | **Does not separate** | Lowest related 0.344 (Z-Image) is below the two highest unrelated pairs, SD3.5 M vs PixArt 0.364 and SD3.5 L vs PixArt 0.353. Rank AUC 0.968. |
| B3 CKA agrees with the tangent ordering (Spearman ≥ 0.9 with five pairs) | **Does not agree** | Spearman 0.70. CKA order: L/L Turbo 0.625 > same weights 0.612 > SD3 M/SD3.5 M 0.462 > Z-Image 0.344 > PixArt/Cosmos 0.338. Native tangent order: same weights 0.984 > SD3 M/SD3.5 M 0.926 > L/L Turbo 0.904 > Z-Image 0.521 > PixArt/Cosmos 0.196. Both put Z-Image and PixArt/Cosmos at the bottom. |
| B4 same-weights pair highest | **Refuted** | SD3.5 L vs L Turbo (0.625) is above the same-weights pair (0.612). |

## Descriptive
- **All 21 unrelated pairs are above their own null p95** (p = 0.005 each, 0.23 to 0.36). The prompt-permutation null tests
  whether two models order prompts alike, not whether they are related, so for CKA "above null" carries no relatedness
  information. The tangent's layer-permutation null does not have this property (unrelated attention pairs sit at or below it).
- Every secondary statistic separates related from unrelated in attention (AUC 1.0): token CKA at matched depth (lowest related
  0.326 vs highest unrelated 0.281), prompt CKA (0.700 vs 0.683), debiased prompt CKA (0.636 vs 0.610), Procrustes (0.863 vs 0.851).
  Margins are 0.01 to 0.05.
- By state (post hoc): token best-match separates at t = 0.75, 0.5 and 0.25 (AUC 1.0, margins 0.012, 0.000, 0.013) and fails at
  step 0 (AUC 0.78), where the input is noise and Z-Image Base loses 12 layers to overflow.
- Same-vendor pairs with no documented relation (SD3.5 L or L Turbo vs SD3 M or SD3.5 M) score 0.37 to 0.45, between the related
  and unrelated pairs.
- MLP (only the four non-SD3 models have MLP hooks): Z-Image Base vs Turbo 0.239 is below PixArt vs Cosmos 0.302 and both
  Z-Image vs PixArt pairs.

## Reading for the papers
On the preregistered primary statistic, activation CKA confirms that the related pairs are above chance (B1) but neither
separates them from unrelated pairs (B2) nor reproduces the tangent's ordering (B3, B4). It ranks the distillation pair above the
same-weights pair, which the tangent does not. The two instruments agree on which pairs are at the bottom. Neither is declared the
reference (as preregistered). The secondary CKA variants separate the groups with small margins, which is post hoc.
