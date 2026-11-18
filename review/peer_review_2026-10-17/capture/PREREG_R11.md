# Round 11 preregistration: spectral-variant control capture

Frozen 2026-10-17T13:22:42Z, before any Round 11 capture exists. Approved by Zac on 2026-10-17 ("proceed with node update for captures"). Arms, prompts, gates and methods are CAPTURE_SPEC.md §3–§5 (k = 8). The section below is copied verbatim from CAPTURE_SPEC.md §6, and §7 fixes what each outcome does to the paper. No prediction may be changed after the first Round 11 CSV arrives; any later analysis is labelled post hoc.

## 6. Analyses and predictions (to freeze before the capture)

Methods are unchanged from the paper: Round 6 S1 (`review/round6/scripts/r6c_S1_hierarchy.py`) for shape and matched-σ tangent with its layer-permutation null; the depth-shift null for the backbone (§4); S4 for the clock-axis fit; the analyzer `plot_beta_layer` for the layer axis. Each statistic is run once per variant V0, V1, V3, V4 (V2 is descriptive only: 64 rows < D, so g is rank-bounded).

| ID | Test | Prediction (supports the paper) | Refuted if |
|---|---|---|---|
| R11-1 | Base↔Turbo tangent, attn and MLP, V1 | ≥ 0.95 in both, and within 0.03 of V0 | either < 0.95 |
| R11-2 | Base↔Turbo tangent, V3 and V4 | ≥ 0.90 in both submodules | either < 0.90 |
| R11-3 | Unrelated tangents (ZIB↔SD3.5M, ZIT↔SD3.5M attn), V1/V3/V4 | below that pair's layer-permutation p95 | any above its p95 |
| R11-4 | Lineage margin (lineage tangent − max unrelated), V3 and V4 | ≥ 0.50 | < 0.50 |
| R11-5 | Backbone PC1 per arm × submodule × variant | ≥ 0.70 and beats all 999 depth-shift nulls (ZIT: rank ≤ 7 caveat) | any fails |
| R11-6 | ZIB MLP clock-axis +−+ (S4 fit, 1000 prompt bootstraps), V1/V3/V4 | ≥ 90% of bootstraps | < 90% in any |
| R11-7 | Diagnostics | `txt_energy_share` < 0.10 in ≥ 90% of Z-Image cells; `mean_energy_share` reported per layer (no prediction) | share ≥ 0.10 in > 10% of cells |

Descriptive only (no prediction): the analyzer β-layer sign pattern per variant for ZIB attn/MLP and SD3.5 M attn; whether `topk_idx` is the same channel set across layers and between Base and Turbo.

## 7. What each outcome does to the paper

- **R11-1 to R11-4 pass:** one sentence in §5.2 and one in §3.1 reporting the image-row and centred tangent and margin. The Q1/Q2 scope sentences (added now) become "tested". The limitations item on massive activations is reduced to "other models".
- **R11-1 fails, R11-2 passes:** the lineage tangent is partly carried by shared caption rows. §5.2 and the abstract restate the lineage result for image rows with the new numbers. The margin in Table 2 gets an image-row column.
- **R11-2 or R11-4 fails:** the lineage separation is a property of the uncentred moment. The abstract and synthesis say so, and the hierarchy is reported for both moments. That is a real change to the headline, which is why it is worth running.
- **R11-5 or R11-6 fails** for a variant: that result is scoped to the uncentred statistic in the text.

