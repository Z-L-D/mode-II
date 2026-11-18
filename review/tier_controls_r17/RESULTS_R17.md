# Round 17 results, 2026-11-09 (complete)

The tests are scored exactly as frozen in `PREREG_R17.md` (sha 16ccbe54).
- Scripts: `scripts/score_r17.py` and `scripts/posthoc_r17.py` (post hoc).
- Outputs: `out/`.
- Data: `data/full/` (Zac, 2026-11-09 05:29Z). PixArt, Cosmos and SD3 M are complete. SD3.5 M CSV arrived separately (09:13Z); 60/60 gens, 24/24 blocks.

## Verdicts
| ID | Result | Verdict |
|---|---|---|
| R17-0a/b | PixArt 60 gens, 28/28 blocks (attn, mlp); Cosmos 60, 28/28 (attn, mlp); SD3 M 60, 24/24 (attn) | PASS |
| R17-0c | step-0 r PixArt vs Cosmos 0.098 / 0.668 (< 0.85: pass); SD3.5 M recapture vs R11 1.000 (pass); **SD3 M vs SD3.5 M 0.940 (needed < 0.85)** | FAIL on the SD3 pair: the identity gate cannot tell SD3 M from SD3.5 M. The files are different checkpoints (same-weights recapture gives 1.000 at different CFG and steps; SD3 M gives 0.940), so this is a gate false positive, the second after the Qwen pair. |
| R17-1 | PixArt <-> Cosmos: attn 0.303 (p95 0.251), mlp **0.687** (p95 0.206) | **REFUTED: the bottom tier did not replicate** (mlp >= 0.47; attn above null, below 0.47) |
| R17-2 | SD3 M <-> SD3.5 M attn **0.926** (p95 0.260; middle band 0.560-0.95; reverse order 0.926) | **MIDDLE TIER** (by 0.024 below the 0.95 ceiling) |
| R17-3 | cond <-> uncond: PixArt 0.996 / 0.998, Cosmos 0.995 / 0.993, SD3 M 0.995, SD3.5 M 0.995; recapture vs R11 (CFG 4.5 vs 1) 0.980 | SUPPORTED |
| R17-4 | recapture vs R11 0.980 > SD3 M vs SD3.5 M 0.926 | SUPPORTED |

Descriptive (preregistered as descriptive): SD3 M <-> R11 SD3.5 M (CFG 1) attn tangent 0.898 (p95 0.254).

## Post hoc: where the PixArt <-> Cosmos value comes from (`out/posthoc_r17_pixart_cosmos.csv`)
- **Clock mapping.** The t = sigma/(1+sigma) clock and the raw sigma give the same value, because the nearest matches are unchanged.
- **MLP.** Most of the 0.687 comes from the last depth interval:
  - Without it the tangent is 0.389, and without the first two and last two intervals it is 0.395. Both are still above the null p95 of 0.206.
  - Cosmos MLP puts 50% of its depth-difference energy in the two first and two last intervals; PixArt puts 77% there.
  - The per-step range is 0.45-0.78.
  - By statistic: H 0.90, log g 0.65, log gap 0.43.
- **Attention.** Per step the tangent runs from 0.18 to 0.45. Without the first interval it is 0.15.

## Reading
- **Attention.** An unrelated pair at equal depth within one modality sits just above its permutation null (0.30). That is far below both lineage pairs (Z-Image 0.71, ACE 0.80 at 8 steps), so the attention tiers keep a margin of about 0.4.
- **MLP.** An unrelated pair reaches the lineage band: PixArt <-> Cosmos is 0.69, against Z-Image lineage 0.70-0.72 and ACE lineage 0.57-0.59. MLP tangents therefore do not separate lineage from unrelated models at equal depth. The shared end-of-network change in the last block does much of this.
- **Consequence, fixed in the preregistration.** The papers say the bottom tier did not replicate on a same-modality depth-matched pair (MLP), and they scope the tier claim to attention.
- **SD3 family (R17-2).** SD3 M and SD3.5 M are a same-vendor successor pair, not a distilled student and teacher. Whether SD3.5 M was initialised from SD3 M is not documented; its MMDiT-X architecture changes the first 13 blocks. Their tangent is 0.93, which is above both distillation pairs (Z-Image 0.71, ACE 0.80) and level with ACE same-weights across capture sessions (0.92, R16 post hoc). Fixed consequence: the middle tier also appears without distillation, so it marks shared lineage in the broad sense (same vendor, recipe, encoders and depth), not distillation.
- **Ordering in attention.** Same weights within a session (0.98-0.999) >= same weights across sessions (0.92) ~ same-vendor successor (0.93) > distillation (0.71-0.80) >> unrelated at equal depth (0.30, -0.07). The tangent grades relatedness. There is no clean boundary between "same weights" and "close relative".
- **Identity gate.** It passes SD3 vs SD3.5 as the same checkpoint (0.94). Its false-positive list now has two same-family pairs.
