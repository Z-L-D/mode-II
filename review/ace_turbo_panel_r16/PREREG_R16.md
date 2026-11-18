# Round 16 preregistration: ACE-Step 1.5 turbo at its release setting (8 steps) and a step-count panel

Frozen 2026-11-06, before any capture exists. Mode-III follow-up (review questions that need new data go to Mode-III).

## Data to be captured
`capture_code_r4/ace_step/run_ace_turbo_panel.bat` with `workflows/ace15_turbo_api.json`: ACE-Step 1.5 turbo, the
CoSPAFS_Sampler / Passive guider structure of the Base run (R14), CFG 1 (cond only), shift 3 (model default, verified from the logged
sigma grid), 180 s, the 30 prompts of `ace_prompts_30.tsv` (the Base prompts). Sessions `ACE15T_180s_S<k>`:
- **T8 (corrected arm):** 8 steps, seed_a and seed_b, 60 gens: same prompts, seeds and size as ACE Base (R14).
- **Panel:** k = 1, 2, 3, 4, 5, 7, 9, 10 steps, seed_a only, 30 gens each. T8 seed_a is the k = 8 panel point. k = 6 is optional
  (the existing 6-step capture, T6 = `review/r3_ace/data`, used other prompts and random seeds).

Comparison sets: T6 (120 gens, 29 other prompts), ACE Base (R14, `review/ace_base_r14`, cond), T8, panel T_k.

## Methods (unchanged from earlier rounds)
Cubes: `ace_base_r14/scripts/build_cubes.py`, branch cond. Tangent: Round 6 S1 `tangent()`, denser sigma grid iterated, nearest
sigma in the other run; null = layer-order permutation (same permutation at every sigma), 1000 draws, rng seed 16, p95.
Identity gate: R12-0 (step-0 depth-profile Pearson r, gen-mean, >= 0.85 = same checkpoint). Crossings: B6 method of
`R3B_SCORING_PLAN.md` (cond attn, pooled over sigma, 100 prompt bootstraps, rng 3000+k). End collapse, parity and prompt share:
R14-4, R14-5, R14-6 code. Sample size: RC2 found every tangent channel within 0.05 of its full value from 10 gens; 30 gens per
arm is taken as adequate for tangents and gates. Prompt share needs replicate seeds, so it is scored on T8 only.

## Gate
| ID | Test | Pass if |
|---|---|---|
| R16-0 | Every arm ran the turbo checkpoint | step-0 r with T6 >= 0.85 and with Base < 0.85, attn and mlp, for T8 and every T_k. A failing arm is dropped and reported. |

The first sigma is 1.0 at every step count, so step 0 is the same computation in all arms up to prompt and seed.

## Predictions
| ID | Prediction | Supported if |
|---|---|---|
| R16-1 | 6 vs 8 steps does not change the turbo fingerprint | tangent T8 <-> T6 >= 0.94 (lower edge of the same-weights tier), attn and mlp |
| R16-2 | The ACE lineage pair holds at turbo's release setting | Base <-> T8 tangent (Base first): >= p95 + 0.30 and <= 0.95, attn and mlp (the R12-6 rule) |
| R16-3 | ...with the same values as at 6 steps | Base <-> T8 tangent within 0.10 of Base <-> T6 (0.850 attn, 0.572 mlp) in both |
| R16-4 | The depth fingerprint does not depend on step count (k >= 3) | tangent T_k <-> T8 >= 0.94 for every k in {3,4,5,7,9,10}, attn and mlp (12 tests; all must pass) |
| R16-5 | One layer-axis crossing at the release setting | T8 attn: 1 crossing modal in >= 50% of prompt bootstraps (as T6 99/100 and Base 100/100) |
| R16-6 | End collapse at every step count >= 2 | MLP L23 gen-mean log g below the median of L12-L22 by > 0.20, for T8 and every T_k with k >= 2 |
| R16-7 | Parity sign at every step count | attn even > odd mean log g in >= 80% of (gen, step) cells, T8 and every T_k |
| R16-8 | Turbo stays prompt-dominated at the release setting (unlike Base at CFG 4, 0.48/0.52) | T8 final-step ANOVA prompt share >= 0.70, attn and mlp (T6: 0.82-0.96) |

## Exploratory (no prediction, reported as such)
- k = 1 and 2 versus T8 tangents (with one or two sigma points, the tangent has little to average).
- Crossing count, plateau onset (B8 2x rule) and step-0-to-final speed as functions of k.
- Base <-> T_k tangent as a function of k: does any step count bring turbo closer to Base?
- Final-step state at matched sigma across k (the 2-3% matched-sigma claim of Mode-I, now with nine schedules).

## Consequences, fixed now
- R16-1 supported: Mode-I keeps the 6-step turbo results and states that 8 steps gives the same fingerprint. Refuted: turbo results
  are restated at 8 steps (T8) wherever T8 can replace T6, and the 6-step values kept as a secondary arm.
- R16-2/3: supported -> the ACE lineage pair is stated at the release setting and with matched prompts and seeds. R16-2 refuted ->
  the ACE lineage tier is reported as operating-point dependent.
- R16-4 refuted -> the "geometry changes by at most 2-3% across step counts at matched sigma" sentence is scoped to the models it was
  measured on, and the panel is reported as the counterexample.
- R16-5..8: each verdict replaces the corresponding T6 statement's scope (one checkpoint at one step count -> release setting).
