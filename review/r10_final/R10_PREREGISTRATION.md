# Round 10: final correlation sweep over Mode-I and Mode-II (preregistration)

Frozen at 2026-10-09T22:27:59Z, before any computation. Zac's request (2026-10-09 22:24Z): "another quick correlation look at mode-I and mode-II and call the papers and repos done". Scope: recorded outputs only (Round 2, Round 6, Round 8) plus the archive cubes; no new captures; PSTF and pstf_llama stay out. Everything not listed here is post hoc and will be labelled so.

**What was already seen:** every recorded output under review/*/out, both papers, and the Clark 2026 Lyapunov paper (the source of R10-1).

## R10-1 [descriptive] How far apart are g, gap and H in the archive?
Mode-II's dimension-measures remark states that g, the effective rank exp(H) and related quantities "are not monotone transformations of each other", and the new Lemma states that the triple does not fix the spectrum. Tabulate, per archive cube (F_ZIT_attn, F_ZIT_mlp, F_ZIB_attn, F_ZIB_mlp, F_MOCHI_attn, F_MOCHI1_attn, F_SD35M_attn, F_SD35L_attn), the Spearman ρ over all (generation, step, layer) cells between log g and H, log g and log gap, and H and log gap; and the mean over sampling states of the depth-profile correlation corr_u(z̃_g, z̃_H). No decision rule. The reading is how close the triple is to one-dimensional in practice.

## R10-2 [rule] Plateau onset vs seed-predictability onset across channels
Across the 7 channels with both statistics (E3 `onset_sigma`; S3 `matched60_rel_median`), Spearman ρ with leave-one-out. **Associated** if |ρ| ≥ 0.7 and all 7 LOO values share its sign. Otherwise **tested, not supported**. (R8-3b tested anisotropy vs onset; the two onsets have not been compared.)

## R10-3 [rule] Persistence vs seed-predictability onset across channels
Across the same channels, Spearman of S3 `VR4_over_walk` against `matched60_rel_median`, with LOO. Because R8-2b showed ZIT attention's VR to be an 8-state-grid artifact, the rule is applied to the 6 channels without ZIT-attn (and the 7-channel value is reported). **Associated** if |ρ| ≥ 0.7 and all 6 LOO values share its sign. Otherwise **tested, not supported**.

## R10-4 [tabulation] Cross-paper numeric consistency
A fixed list of quantities quoted in both papers (tangent values and capture ladder, hierarchy margins, Qwen derivative values and intervals, abliteration tangent, D10 derivative correlations, LLM RSA ranges, pooling held-out R², ZIT VR, divergence token, n-stability ranges, prereg-table SHA prefixes). The script searches each number in both main.tex files and reports any that is present in only one. Rule: every number is in both, or the discrepancy is recorded and fixed before release.

Scripts: `scripts/r10_a_sweep.py` (R10-1 to R10-3) and `scripts/r10_b_consistency.py` (R10-4). Outputs in `out/`.
