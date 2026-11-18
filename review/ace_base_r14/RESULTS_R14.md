# Round 14 results: ACE-Step 1.5 Base (2026-11-05)

Scored exactly as frozen in `PREREG_R14.md` (sha dc997f6d) and, for R12-6/R12-7, `../zib_turbo_mixup_2026-10-18/rerun/PREREG_R12.md`
(sha ad06a1bb). Scripts: `scripts/score_r14.py` (R12-6, R14-1..6), `scripts/r14_B_replay.py` (B1-B10, R12-7), `scripts/posthoc_r14.py`
(post hoc, labelled). Cubes: `scripts/build_cubes.py` (the R12 builder with a guarded domain parse). Outputs: `out/`.
Rerun from the raw CSVs on a clean copy: every output byte-identical.

## Data
ACE-Step 1.5 Base, 50 steps, CFG 4, 180 s (N 2250, D 2048), cond + uncond rows, 24 layers x {attn, mlp}. 30 prompts x 2 seeds
(seed_a 1000+2i on GPU 0, seed_b 1001+2i on GPU 1), all 60 generations complete. The two CSVs were joined by (file, gen_idx, seed),
which is the same as prompt_hash + seed (gen_idx restarts at 1 in each file).

## Preregistered results

| ID | Result | Verdict |
|---|---|---|
| R12-0 | step-0 depth-profile r with ACE Turbo 0.340 (attn) / 0.697 (mlp) | NOT TURBO (pass) |
| R12-6 | Base <-> Turbo tangent attn **0.850** (null p95 0.316; bound 0.616), mlp **0.572** (p95 0.242; bound 0.542); both <= 0.95 | SUPPORTED |
| R12-7 | ACE Base attn: one crossing (+), 1 crossing in 100/100 prompt bootstraps | frozen rule: MODALITY |
| R14-1 | gpu0 <-> gpu1: tangent 0.992 / 0.972, field r 0.998 / 0.998 | SUPPORTED (pooled) |
| R14-2 | Base log g below Turbo (sigma-matched) in 14/24 (attn), 10/24 (mlp) layers | REFUTED |
| R14-3 | cond <-> uncond tangent 0.997 / 0.964 | SUPPORTED |
| R14-4 | MLP end collapse: L23 below median(L12-22) by 2.30 (Turbo 1.87) | SUPPORTED |
| R14-5 | attn even > odd in 100% of (gen, step) cells (median 1.165); mlp 92% (median 0.077) | SUPPORTED |
| R14-6 | ANOVA prompt share at the last step 0.481 / 0.518 | REFUTED |

R14-B, R3 Part B rules replayed unchanged on Base cond (B9 edge 2, tol 2):

| ID | Base result | Base | Turbo |
|---|---|---|---|
| B1 | gate | PASS | PASS |
| B2 | cells >= 0.10: attn 48.2%, mlp 89.0%; median g/g_MP 0.095 / 0.276 | REFUTED | REFUTED |
| B3 | PC1+PC2 attn 0.9875, mlp 0.9794 | REFUTED | REFUTED |
| B4 | dR2 attn **0.254** (0.689 -> 0.943), mlp 0.671 | **REFUTED** | CONFIRMED |
| B5 | even > odd 100% (attn) | CONFIRMED | CONFIRMED |
| B6 | 1 crossing, 100/100 | REFUTED | REFUTED |
| B7 | not run | - | - |
| B8 | onset sigma attn 0.786, mlp 0.848 (16 plateau intervals) | CONFIRMED | CONFIRMED |
| B9 | attn peak at step 6 (sigma 0.957): interior, stability 0.92, beats both ends by > 2 SE | **REFUTED** | CONFIRMED |
| B10 | steepest rise in the first interval (attn 0.34 -> 0.46, mlp 0.22 -> 0.41) | **CONFIRMED** | REFUTED |

## Preregistered consequences (applied in the papers)
- **Second distillation pair.** Lineage is a middle tier in two families: Z-Image (0.71/0.72) and ACE-Step (0.85/0.57). For ACE the
  same-weights references are cond<->uncond 0.997/0.964 and gpu0<->gpu1 0.992/0.972, and the unrelated floor is <= 0.27 (attn) /
  <= 0.20 (mlp) (post hoc, below). The ACE MLP clears its bound by only 0.03.
- **R12-7 fires "modality":** the single ACE crossing is not caused by distillation, so the "confounded with distillation" caveat is
  withdrawn. Read together with R12-8 (real Z-Image Base cond also has one crossing), the +−+− is not an image property either:
  it is present in SD3.5 M/L, Mochi and Z-Image Turbo at 25 steps/CFG 4/pooled, and absent in Z-Image Base, Z-Image Turbo at 8 steps
  (attn) and both ACE checkpoints. The papers state it as model- and capture-specific.
- B4, B9, B10 differ between the two ACE checkpoints; each Turbo statement is scoped to the distilled 6-step checkpoint and the Base
  result is given beside it. With 50 steps the B9 attention peak sits at sigma 0.957 inside the first-step transient.

## Post hoc (labelled post hoc where reported)
- ACE Base vs image/video models (`out/posthoc_ace_vs_image_tangents.csv`): attn -0.11 to 0.27, mlp 0.13 to 0.20 (unrelated range).
- ACE Base MLP layer axis: 2 crossings (−+) in 100/100; uncond attn 1 (100/100); uncond mlp 2 (84/100).
- Backbone (999 depth-shift nulls, R11 implementation; `out/posthoc_backbone_clock.csv`): ACE Base PC1 0.811 (attn) / 0.707 (mlp), effective
  rank 1.87 / 2.27, p = 0.001 both. Real Z-Image Base Z1 for reference: PC1 0.636 / 0.762, rank 2.28 / 2.10, p = 0.001.
- Clock-axis three-segment fit (S4 fit3, point estimate): ACE Base attn −++, mlp ++− (+−+ in 0 of 1000 bootstraps); Z1 attn ++−, mlp +++.
