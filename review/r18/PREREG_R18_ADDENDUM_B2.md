# R18 addendum B.2 (2026-11-11 06:35Z): non-finite sketches

Written after Zac's first `cka_r18b.py` run stopped at R18_ZB6 vs R18_ZIT2 (`LinAlgError` in the Procrustes step, after
"invalid value" warnings), and before any Z-Image CKA value was seen. The three pairs that had finished (SD3.5 M CFG 4.5 vs 1
0.612, SD3.5 L vs L Turbo 0.625, SD3 M vs SD3.5 M 0.462, attention) are unaffected.

Cause: the node stores sketches in float16 (addendum B), so any activation above 65504 is stored as inf. Z-Image has
very large activations in some hooks.

Rule: a layer whose token rows or prompt rows hold a non-finite value for any prompt at a state is left out of every statistic
of that state, in the observed value and in every null draw alike. Its row or column of the layer-by-layer matrix is NaN, and
the best-match and matched summaries skip NaN. On data with no non-finite value the result is the same as the frozen script
(tested on synthetic sketches: values agree to 1e-7, p-values identical). Output rows gain `nonfinite_a` / `nonfinite_b`
(layers left out). The script now writes its CSVs after every pair, and `--resume` skips pairs already written.
Frozen script: `cka_r18b_frozen.py`; amended: `cka_fix/cka_r18b_v2.py` (= `capture_code_r4/tier_controls_r18/cka_r18b.py`).
Rules B1 to B4 are unchanged. Every pair with a left-out layer is reported with the count.

## B.2a (2026-11-11 06:55Z)
The second run finished eight pairs, then stopped in `eigvalsh` (Procrustes step) at R18_SD35L vs R18_ZB6, state t0.25, where no
layer was non-finite. Procrustes is now computed from thin SVDs: Xa = U s Vᵀ gives |Xaᵀ Xb|_* = |diag(s) Uᵀ Xb|_*. This is the
same quantity as the frozen form (agreement 4e-10 on synthetic data) but needs no eigen-solver. A constant layer, or a layer whose
SVD fails, is left out of the Procrustes statistic only. Procrustes is secondary and has no null; the primary token CKA is unchanged.
The eight finished pairs are kept (`--resume`).
