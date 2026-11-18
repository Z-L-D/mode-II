# Round 18 results (scored 2026-11-11)

Prereg `PREREG_R18.md` (sha 6db32be7), addendum A. Upload: `/mnt/project-files/R18.zip` (27 files). Scoring script `scripts/score_r18.py`,
extras `scripts/r18_extra.py`, outputs in `out/` (`r18_verdicts.csv`, `r18_tangents.csv`, `score_r18.log`). Estimators are the frozen ones
(24-point tangent, permutation null 1000 draws rng 17). Native-grid values and surrogate nulls are secondary and are in `r18_tangents.csv`.

## Gates
| Gate | Result | Detail |
|---|---|---|
| R18-0a complete arms | PASS (after the fix run) | SD35L 60/60, SD35LT 60/60, ZIT2 60/60, ZB6 120/120. The first upload had SD3.5 Large Turbo at 56 of 60 generations (needs 58). The first four prompts failed with `HeaderTooLarge`: the hash cache shows the Turbo file was a 152-byte placeholder when the run started and the full 16.46 GB file arrived later (`out/m3_comfy_state/R18_SD35LT.state.json`). Zac ran `capture_code_r4/tier_controls_r18/r18_fix_sd35lt.bat` on 2026-11-11 (06:01Z); the four prompts appended as gen 57 to 60, and the updated CSV is in `data_fix/`. |
| R18-0b hooks and row counts | PASS | 38/38 and 30/30 blocks; ZIT2 image rows 4096, caption rows 64 everywhere. |
| R18-0c checkpoints | PASS on step-0 r and on distinct hashes; Hugging Face comparison not done here (huggingface.co is blocked from this container) | ZIT2 vs ZITSV 1.0000 (attn and MLP); ZB6 vs Z1 0.9999; ZB6 vs ZIT 0.648; R18 SD35L vs archive 0.994. SHA-256 (`out/r18_weight_hashes.json`): SD3.5 L ffef7a27…8f93, L Turbo fb64610b…489c, Z-Image Turbo 24076130…74a6, Base 996a67d3…3dee. |
| R18-0d schedules | PASS | ZB6 σ equals the shift-6 image of Z1's grid to 8e-7; ZIT2 grid equals ZITSV's exactly. |

## Predictions
| ID | Verdict | Numbers |
|---|---|---|
| R18-1 SD3.5 L ↔ L Turbo in the middle tier | **Supported** | 0.818, null p95 0.301 (rule needs ≥ 0.601 and ≤ 0.95); 90% interval 0.795 to 0.837; Turbo first 0.815; against archive SD3.5 L 0.817 / 0.821; native grid 0.904 (surrogate p95 0.345). On the first 56 prompts it was 0.813. By the prereg, the papers gain a third distillation lineage pair and the first whose teacher→student link is documented. |
| R18-2 same weights stay at the top | Supported | R18 SD35L ↔ archive SD35L 0.994 (different session and prompt set); cond ↔ uncond 0.994. |
| R18-3a narrow image-row session band | Supported, but see the replay note | ZIT2 ↔ ZITSV 1.000 on image rows and all rows, attention and MLP. |
| R18-3b image-row lineage value replicates | Supported, but see the replay note | Z2 ↔ ZIT2 image rows 0.928 attention, 0.754 MLP (on file 0.928, 0.754). |
| R18-3c lineage below the session band | Supported, but see the replay note | gap 0.072 attention (0.246 MLP). |
| R18-4 lineage does not depend on Base's shift | Supported | ZB6 ↔ ZIT 0.649 attention (ref 0.707, p95 0.277) and 0.682 MLP (ref 0.719, p95 0.201); ZB6 ↔ Z1 (shift 6 vs 3) 0.9996 in both. Native grid 0.520 / 0.760 (Z1: 0.547 / 0.771). |

## Replay note (not anticipated by the prereg)
ZIT2 used the same prompts, seed and sampler as the R11 session ZITSV. Its 58 shared generations are identical to ZITSV in every
cell at the CSV's printed precision (`out/r18_replay_check.csv`). The same holds for all four R18S reruns against Round 17
(PixArt-Σ, Cosmos 2B, SD3 M, SD3.5 M: 60 of 60 generations identical each). So:
- A capture is exactly reproducible across sessions and machines at fixed prompt, seed and settings. This is a provenance
  statement the papers did not have.
- R18-3a therefore measures a replay, not a band, and R18-3b repeats the on-file number. They pass by the frozen rule but add
  no independent evidence. The design fault is in the prereg (same seed and prompts for the second session).
- Because a session by itself contributes nothing, the only between-session variation is the prompt and seed set. Post hoc
  (defined after seeing the data): 200 random 30 / 30 prompt splits of ZIT2 on image rows give a split-half tangent with 5th
  percentile 0.996 (attention and MLP), against a 95th percentile of 0.931 (attention) and 0.759 (MLP) for Z2 ↔ ZIT2 on 30
  generations each (`out/r18_splithalf_image_band.csv`). The image-row lineage value lies 0.065 below the prompt-set band in
  attention. This is the informative form of R18-3c, and it is post hoc.

## Descriptive
- ZB6 ↔ Z3 0.674 / 0.658; ZB6 ↔ ZIT2 0.657 / 0.678; ZB6 cond ↔ uncond 0.995 / 0.939.
- Step-0 gate SD3.5 L vs L Turbo 0.986 (inside the "same weights" reading band of ≥ 0.98; the gate does not separate this
  distillation pair, the tangent does).
- Addendum A, tr M_cond / tr M_uncond per cell (`out/r18_trace_ratio.csv`): SD3.5 L median 1.004 (5th to 95th percentile 0.955 to
  1.227; 89% of cells within 10%); Z-Image Base shift 6 attention 1.021 (0.870 to 1.166; 72% within 10%), MLP 1.013 (0.924 to
  1.057; 96% within 10%). Equal energy holds at the median and fails in a tail of cells (extremes 0.62 to 2.17 for SD3.5 L).

## Open
1. Done: the four SD3.5 Large Turbo prompts were recaptured and R18-1 is scored.
2. Received 2026-11-11 (`out/r18_settings.json`, `out/r18_weight_hashes.json`). Settings match the prereg. ZIT2 and ZB6 ran from the Z-Image templates (CoS-PAFS, simple), not Zac's own workflow files; ZIT2 reproduces ZITSV exactly, so the template matches his settings.
3. Addendum B: the `_sketch` folders stay local; `cka_r18b_*.csv` / `.npz` from `cka_r18b.py` are not uploaded yet.

## Native grid (descriptive; `scripts/r18_native.py`, `out/r18_native_grid.csv`, RC33 statistics)
SD3.5 L ↔ L Turbo 0.904 (perm95 0.326, surrogate 0.335, subset .893-.913); SD3.5 L sessions 0.998, cond ↔ uncond 0.997;
ZB6 ↔ ZIT 0.520 / 0.760 (perm95 0.299 / 0.225); ZB6 ↔ Z1 1.000 / 1.000; Z2 ↔ ZIT2 image rows 0.913 / 0.736.
