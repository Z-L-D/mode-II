# Round 16 results: ACE-Step 1.5 turbo at 8 steps and the 1-10 step panel (2026-11-07)

Scored exactly as frozen in `PREREG_R16.md` (sha a3b59c92). Scripts: `scripts/score_r16.py` (preregistered tests, estimators copied
from R14) and `scripts/posthoc_r16.py` (post hoc). Outputs: `out/`. Cubes: R14 `build_cubes.py`, cond branch.

## Data
Upload `ACE15T_180s_steptest.zip` (Zac, 2026-11-07 23:29Z), in `data/`. Every arm is complete: T8 60 gens (30 prompts x seed_a, seed_b),
T1-T5, T7, T9, T10 30 gens each (seed_a), 24 layers x attn/mlp at every step, N 2250, D 2048. The sigma grids are shift 3 (T6's grid
is the same schedule at 6 steps). k = 6 was not captured.

## Verdicts
| ID | Test | Result | Verdict |
|---|---|---|---|
| R16-0 | right checkpoint in every arm | step-0 r with T6 0.863-0.927, with Base 0.169-0.383 | PASS |
| R16-1 | T8 <-> T6 >= 0.94 | attn 0.918, mlp 0.821 | REFUTED |
| R16-2 | Base <-> T8 lineage rule (R12-6) | attn 0.795 (p95 0.337), mlp 0.594 (p95 0.229; bound 0.529) | SUPPORTED |
| R16-3 | Base <-> T8 within 0.10 of Base <-> T6 | attn 0.795 vs 0.850 (-0.055), mlp 0.594 vs 0.572 (+0.022) | SUPPORTED |
| R16-4 | T_k <-> T8 >= 0.94, k in {3,4,5,7,9,10} (12 tests) | 11 of 12 pass (attn 0.993-0.999, mlp 0.952-0.989); T3 mlp 0.906 | REFUTED |
| R16-5 | one layer-axis crossing at 8 steps | point estimate 1 (+); 1 crossing in 94/100 bootstraps (3 in 6) | SUPPORTED |
| R16-6 | MLP end collapse at every k >= 2 | drop 1.61-1.83 (k1 1.55) | SUPPORTED |
| R16-7 | attn even > odd at every k | 100% of cells at every k | SUPPORTED |
| R16-8 | T8 final-step prompt share >= 0.70 | attn 0.692, mlp 0.593 | REFUTED |

## Exploratory (preregistered as exploratory, no prediction)
- k = 1 and 2 vs T8: attn 1.000 / 0.995, mlp 0.999 / 0.948. With one sigma point the tangent is the step-0 profile, which is the
  same computation in every arm.
- Base <-> T_k over k: attn 0.77-0.82, mlp 0.52-0.59, with no trend: no step count brings turbo closer to Base.

## Post hoc (not preregistered): why R16-1 fails while the panel agrees with itself
`out/posthoc_r16_prompt_vs_steps.csv`.
- T6's 29 prompts are all in the 30-prompt panel set (the prereg called them "other prompts"; the texts match, but T6 used random seeds and 4-8 gens per prompt).
- Every panel arm sits at the same distance from T6 (attn 0.917-0.938, mlp 0.70-0.83), including T5 and T7, while the panel arms are
  within 0.95-0.999 of T8. The gap is between the two capture sessions, not between 6 and 8 steps.
- It is present at step 0, where the computation does not depend on the step count: step-0 profile r between panel arms is 1.000, and
  between T8 and T6 0.926 (attn) / 0.863 (mlp).
- Noise ceilings: T8 seed_a <-> seed_b 0.989 / 0.974; prompt-half splits 0.990 / 0.958 (T8) and 0.981 / 0.931 (T6).
- Reading: the T6 session differed in something fixed at step 0: the conditioning metadata (bpm, key, lyrics, language fields of
  the text encoder node), node version, or the seed design. This cannot be settled from the CSVs. It is a capture-condition
  effect within one checkpoint.

## Consequences (as fixed in the prereg)
- R16-1 refuted: turbo results are restated at 8 steps (T8) wherever T8 can replace T6. The 6-step values are kept as a secondary arm.
  The post hoc finding is added: the T6/T8 difference does not come from the step count.
- R16-2/3 supported: the ACE lineage pair is stated at the release setting with matched prompts and seeds (0.80 / 0.59).
- R16-4 refuted (one cell of twelve, T3 MLP 0.906): the "at most 2-3% across step counts at matched sigma" sentence is scoped to the
  models it was measured on. The panel is reported as the counterexample at 3 steps in MLP; attention stays at 0.99 or higher at every k.
- R16-5..7 supported: the crossing count, end collapse and parity are stated at the release setting.
- R16-8 refuted: at 8 steps with two fixed seeds per prompt, turbo is not prompt-dominated at the 0.70 level (0.69 / 0.59; T6 had
  0.82-0.96 with 4-8 random seeds). The scope of the T6 statement is narrowed accordingly.

## Implications for the papers (noted 2026-11-07, for the joint R16+R17 pass)
- Same-weights tangent across capture sessions (same checkpoint and prompts) is 0.92 / 0.82. That is below the 0.94 top-tier line, and the margin over ACE lineage (0.80 / 0.59) is about 0.12 in attention. State this as a limit of the tiers.
- The step-0 identity gate across sessions gives 0.86-0.93, close to its 0.85 cutoff.
- Within one session the fingerprint is step-count invariant (k = 1-10).
