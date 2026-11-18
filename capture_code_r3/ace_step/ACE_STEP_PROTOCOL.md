# ACE-Step audio arm: capture protocol (R3, 2026-10-08)

**Purpose.** All five models in the archive share natural-image (or video) data, a denoising objective and transformer blocks. ACE-Step changes both the modality and the training data while keeping a DiT denoiser. Whatever recurs here is not an artifact of image data. Whatever fails to recur shows how much of the archive's picture is image-specific. Predictions are frozen in `review/r3_prereg/R3_PREREGISTRATION.md` before any audio capture exists.

## Setup

1. Install `capture_code_r3` over the node folder (`nodes.py` and `model_registry.json` both changed) and restart ComfyUI.
2. Delete and re-add the Passive Telemetry Guider in the workflow, so no shifted widget values carry over.
3. Leave `hook_mode` on `auto`. The registry entry hooks attn (`self_attn.o_proj`) and MLP (`mlp.down_proj`) on all 24 layers.
4. **Preflight** (1 prompt, 1 seed, 2 steps):

       python preflight_check.py "<csv>" --steps 2 --layers 24 --subs attn,mlp --prompts ace_step/ace_prompts_30.txt

   With log_telemetry on, the console should show `Registry match: AceStepConditionGenerationModel → 24 blocks via 'decoder.layers'` and `seq_len` 2250. If it shows "No transformer blocks found", the old nodes.py is still loaded.

## Settings (identical for all 60 generations)

| Setting | Value | Why |
|---|---|---|
| Model | ACE-Step 1.5 (base or SFT if you have it; turbo otherwise) | Base/SFT gives more σ points. Turbo works, but 8 steps is coarse |
| Steps, sampler, scheduler, shift, CFG | Your working ACE workflow's values, unchanged across all gens | Write them down once. With turbo, CFG 1 gives cond rows only |
| Latent | `EmptyAceStep1.5LatentAudio`, **180 s** | 2250 tokens ≥ D = 2048, so g is not rank-limited (same regime as the image runs) |
| Text encode | `TextEncodeAceStepAudio1.5`. `tags` = the TSV `tags` column; `lyrics` = `[Instrumental]`; bpm, keyscale and timesignature from the TSV; language en; duration 180 | Fixed metadata, so nothing about the piece is inferred differently per seed |
| `generate_audio_codes` | **off** | When on, a Qwen3 LM samples a semantic plan with its own randomness, and the DiT mostly decodes that plan. Off keeps a pure text → DiT path, like the image models. The text-node `seed` then has no effect; leave it at 0 |
| Guider `prompt_text` | the same `tags` string | Gives the md5 prompt hash in the CSV |
| Seeds | `seed_a` and `seed_b` columns (1000+2i, 1001+2i) | Own pair per prompt (the R5 seed-design fix) |
| Output | Save the audio with a file name that carries prompt idx and seed, e.g. `ace_{idx:02d}_{seed}.flac` | D14 lesson: the output must be linkable to its telemetry rows |

**Run size:** 30 prompts × 2 seeds = **60 generations**, in one CSV session named e.g. `ACE15_180s_CFG<x>`. The 10 domains × 3 prompts mirror the 10 themes × 3 of `matched_prompts_30_v2`.

**If 180 s is too slow:** use 120 s for every generation (1500 tokens, N/D = 0.73), and say so. g is then rank-limited, and the analysis compares it with the isotropic baseline at that N and D. Never mix durations.

## Optional second arm (only after the first is in)

- 10 prompts (one per domain) × 2 seeds with `generate_audio_codes` **on**, and the text-node seed set equal to the sampler seed. This tests how much of the DiT's geometry is the LM plan.
- ACE-Step v1, if you have it: same prompts at 120 s. That is 1292 tokens, D = 1536. Self-attention is *linear* there, a different operator, so the results stay separate.

## What comes back

- The CSV(s) and the audio files.
- The console line `branch check gen 1: ...` from the first generation.
