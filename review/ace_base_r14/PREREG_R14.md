# Round 14 preregistration: ACE-Step 1.5 Base (undistilled audio), scoring R12-6/R12-7 and replaying R3 Part B

Frozen 2026-11-05, before any statistic beyond the identity gate is computed on the ACE Base capture.

**Data (as received, 2026-11-05 upload `ACE15B_180s_CFG4.zip`):** `ACE15B_180s_CFG4_gpu0.csv` and `_gpu1.csv` (+ layer JSONs).
ACE-Step 1.5 Base, 50 steps, CFG 4, 180 s (seq 2250, D 2048), passive-branch rows (cond and uncond), 24 layers x {attn, mlp},
30 prompts (`capture_code_r4/ace_step/ace_prompts_30.tsv`) x 2 seeds: seed_a = 1000+2i on GPU 0, seed_b = 1001+2i on GPU 1.
60 generations, every one complete (4800 rows each). Generations are keyed by (prompt_hash, seed); gen_idx repeats across the two files.
This is the RERUN_SPEC fallback size (2 gens per prompt).

**Seen before freezing:** row counts, columns, seeds, sigma grid head, summary of prompt_influence (min/median/max), layer JSON
(even layers sliding:128, odd full) and the R12-0 identity gate (below). No g, gap, entropy, tangent, crossing, share or profile
was computed or plotted.

**R12-0 identity gate (frozen rule in `../zib_turbo_mixup_2026-10-18/rerun/PREREG_R12.md`), already run:** step-0 depth-profile r
with ACE Turbo = 0.340 (attn) / 0.697 (mlp) over 60 gens (gpu0 0.342/0.700, gpu1 0.338/0.693). Both < 0.85: **NOT TURBO, pass.**

## Operational choices for the frozen R12 tests (nothing in R12 changes)
- Cubes: `build_cubes.py` from R12, unchanged, branch cond (uncond separately), cap 0, applied to both CSVs together.
  ACE Turbo = the 120-gen Round 3 run (`review/r3_ace/data`, all rows are cond).
- R12-6: Round 6 S1 `tangent()` with A = ACE Base cond, B = ACE Turbo (the R12 convention: Base first, the denser sigma grid
  iterated, nearest Turbo sigma). Null: Turbo's layer order permuted, the same permutation at every sigma, 1000 draws, rng seed 14,
  p95. Supported if tangent >= p95 + 0.30 **and** <= 0.95 in both attn and mlp. The reverse order (Turbo first) is reported as a
  sensitivity, not scored.
- R12-7: B6 method of `review/r3_ace/R3B_SCORING_PLAN.md` (analyzer `plot_beta_layer` via `lens_L2_rg.curve`, cond attn, pooled
  over sigma; 100 prompt bootstraps, rng 3000+k, all gens of each drawn prompt). Decision rule as frozen in R12: 3 crossings modal
  in >= 50% -> distillation; 1 crossing modal in >= 50% -> modality; anything else unresolved.

## R14-B: R3 Part B rules (B1-B10) replayed unchanged on ACE Base cond
The predictions and thresholds of `review/r3_prereg/R3_PREREGISTRATION.md` (SHA efa613eb, Amendment 1) and the operational
choices of `R3B_SCORING_PLAN.md` apply unchanged, with two forced differences: 30 prompts with 2 gens each (bootstraps resample the
30 prompts), and 50 steps, so B9 uses edge 2 and tol 2 (Amendment 1), and B8's plateau speed is the median over all intervals with
midpoint < 0.6. Each B is scored on Base. Reported alongside: whether the verdict equals the Turbo verdict.

## New predictions
| ID | Prediction | Supported if | 
|---|---|---|
| R14-1 | Gate: both GPUs ran the same model | gpu0 vs gpu1 cond cubes: tangent >= 0.95 and Pearson r of the (step x layer) mean log g field >= 0.98, attn and mlp. If it fails, every result is reported per GPU. |
| R14-2 | Base is more concentrated (lower g) than Turbo, as Z-Image Base is (R13 H3) | For each Turbo sigma, the nearest Base step; mean over those 6 matched steps of the gen-mean log g per layer; Base < Turbo in >= 18 of 24 layers, in both attn and mlp |
| R14-3 | cond and uncond share the fingerprint (same weights, same run) | Base cond <-> Base uncond tangent >= 0.90, attn and mlp |
| R14-4 | End collapse in MLP (the family of Q35, SD3.5 and ACE Turbo) | Base cond MLP: gen-mean log g at L23 (averaged over steps) below the median of L12-L22 by more than 0.20 |
| R14-5 | Layer parity is set by the architecture, so its sign carries over | attn, (gen, step) cells: sign of even-minus-odd mean log g equals the Turbo sign (even > odd) in >= 80% of cells |
| R14-6 | Prompts dominate the final state at CFG 4 | E6 ANOVA prompt share (B10 code) at the last step >= 0.70, attn and mlp |

## Consequences for the papers, fixed now
- R12-6 supported: a second distillation pair is added; lineage middle tier is stated for two model families. If the tangent is
  > 0.95: "lineage indistinguishable from same weights in ACE". If below p95 + 0.30: the lineage tier is stated for Z-Image only and
  the ACE failure is reported.
- R12-7: per the frozen rule; the papers' audio caveat ("confounded with distillation") is withdrawn or kept accordingly.
- R14-B: where the Base verdict differs from Turbo, the paper reports both and scopes the Turbo statement to the distilled checkpoint.
- R14-1 failing changes only the presentation (per GPU).
