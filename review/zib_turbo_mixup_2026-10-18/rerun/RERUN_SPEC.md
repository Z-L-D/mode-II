# Base reruns after the Z-Image Base/Turbo mix-up (Round 12 captures)

2026-10-18. Every "Z-Image Base" capture in the project so far was the Turbo model. This includes the August archive, Round 3, Round 4 and Round 11 arm A. See `../IMPACT_AUDIT.md`. These runs replace them and add a second distillation pair in audio. The predictions are frozen in `PREREG_R12.md` before any of this data exists.

Zac's May 2026 Base run was a workflow test. It is used only to calibrate the identity gate and as an exploratory preview (`../out/may_base_hierarchy.csv`). It is not used in the papers.

## Node and checks for every arm

- Use **capture_code_r4** (copy the folder over the node folder; README_R4.md). It always writes per-branch rows (`passive-branch`, cond/uncond), so nothing is pooled.
- **`spectral_variants`: off for Z1 and Z3.** Round 12 uses only the standard columns, and the switch costs about 30 min per Base generation (four 3840² eigendecompositions per hook). Turn it on only for the Round 11 arms (Z2 here, plus Turbo CFG 1 and SD3.5 M per `README_R4.md`). Changed 2026-10-24; this spec first said "on for every arm".
- **Identity gate before every long run.** Do 1 prompt × 2 steps, then:
  ```
  python fingerprint_check.py --model zimage <csv>     # Z-Image arms
  python fingerprint_check.py --model ace15  <csv>     # ACE arm
  ```
  It compares the step-0 depth profile with the Turbo reference. Z-Image Turbo scores 0.99–1.00 against itself under every setting. Your May Base run scores 0.79 (attn) and 0.59 (MLP). **TURBO = stop: the wrong model file is loaded.** Run the gate again on the first finished CSV of each session.
- Then check the model file in the saved images: `python capture_code_r4/scan_png_models.py <output folder>`. Every row should name the Base file.
- Then run `preflight_check.py` as in README_R4, adding `--variants zimage` for the Z-Image arms.

## Arms

| Arm | Model | Prompts × gens | Steps | CFG | Other settings | Role |
|---|---|---|---|---|---|---|
| **Z1** (primary) | Z-Image **Base** | 120 primary prompts (`prompts_primary120.tsv`, 12 per domain) × 4 | 25 | 4 | 1024 px, shift 3.0, the same sampler and scheduler as the August run (its σ schedule: 1.000, 0.9863, 0.9718 … 0.2069, 0.1111). A different seed per generation. | Replaces the archive "ZIB" channel. Its cond branch is the Base side of the lineage pair. |
| **Z2** | Z-Image **Base** | the 60 Round 3 prompts (`capture_code_r4/prompts_r3_cond60.txt`), seed 1001 | 25 | 1 | as Z1 | Replaces Round 11 arm A and the Round 3/4 Base CFG 1 arms. Turbo vs Base at the same CFG 1. |
| **Z3** (recommended) | Z-Image **Turbo** | 120 primary prompts × 1 | 25 | 4 | exactly as Z1 | The same settings as Z1 with only the weights changed. This is the cleanest lineage contrast. |
| **A1** | ACE-Step 1.5 **Base** (non-turbo) | the 30 ACE prompts (`capture_code_r4/ace_step/ace_prompts_30.tsv`) × 4 | the Base model's usual step count | the Base model's usual CFG (per-branch rows are written, so the cond branch stays comparable with the Turbo run at CFG 1) | 180 s; tags, bpm, key and time signature from the TSV; `[Instrumental]`; `generate_audio_codes` off; the same text encoder and VAE as the Turbo run | Second distillation pair: ACE Base ↔ ACE Turbo. Also separates modality from distillation for the single ACE crossing. |

If time is short: Z1 first, then A1, then Z2, then Z3. If A1 is too slow at 4 gens, run 2 gens per prompt (60 gens). Never mix durations.

**Not part of this round:** ACE on the 120 image prompts. It goes to Mode-III, Turbo only at first.

## What comes back

- The CSVs.
- The `scan_png_models.py` output (for ACE, the model name from the workflow).
- The step count, sampler, scheduler and CFG actually used for each arm.
