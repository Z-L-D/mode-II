# capture_code_r2: Round 2 instrument updates

This folder is a copy of `capture_code/` with two changes. The originals are untouched.

## R5 note (2026-10-08): the fix worked; three protocol slips remain

The Round 4 captures (`data/r4_captures/`) are genuinely per-branch in every arm. `nodes.py` is unchanged this round. What still went wrong, and what changed:
- **Prompt list.** The core arms ran 26 of the 30 frozen prompts. `matched_prompts_30.tsv` stores the sea-otter prompt as a quoted field that starts with a newline, and `matched_prompts_30.txt` has a blank line before it. Use the clean copies **`matched_prompts_30_v2.tsv` / `.txt`** (same 30 prompts, no stray quote or blank line) and run the preflight with `--prompts matched_prompts_30_v2.tsv` *before* the long run.
- **Seeds.** Seeds 1001/1002 were reused for every prompt again. For seed-fate work, give each prompt its own seed pair (for example 1000+2i and 1001+2i for prompt i) and use the same pair in both CFG arms.
- **D14.** No output image name is saved. Add the image file name per generation (keyed to seed and prompt_hash).
- **`preflight_check.py`** now keys a generation by (gen_idx, seed), because one 40-step pyramid CSV holds two appended sessions that reuse gen_idx; reads a CSV with no header row (one 512 file) and warns; reports nonfinite g/gap/H cells; and warns when no output-image column exists.

## 0. Round 3 fix (2026-10-06): per-branch rows could not be trusted to switch on

**What went wrong.** Every Round 3 CSV says `hook_mode=passive` and `branch=all`. That column is a direct readout of the node's split flag at write time, so the flag reached the backend as False in all nine arms, even though the toggle looked on. The most likely route is the widget-position shift described below: a workflow saved with the build that had `split_cfg_branches` in the middle loads into the corrected build with every value one slot off, and the last slot (the split switch) receives the old `log_telemetry` value. If your console from those runs shows `layer_range 'true' matches no layer`, that confirms it.

**What changed in `nodes.py` (Passive guider only):**
- Per-branch capture is **always on**. `split_cfg_branches` stays as the last widget only so saved workflows load; a False value is ignored and printed as a warning that the node's widget values are shifted.
- A second branch channel: a pre-hook on the diffusion model reads `transformer_options['cond_or_uncond']`, so rows are still attributed if another node replaces `model_function_wrapper`.
- After step 0 of **every** generation the console prints one line whatever `log_telemetry` says, for example `branch check gen 3: cfg=4.5 branches=cond,uncond layers=24 forwards=1 (pooled) ... OK`. A failure prints `*** BRANCH CHECK FAILED ***`.
- `get_guider` flags shifted widget values (a boolean in `layer_range`, a string in `log_telemetry`, `True`/`False` in `csv_session`).
- `gen_len` now holds the real step count (from `sample_sigmas`) instead of -1, so 40- and 100-step runs are distinguishable in the CSV.
- Hooks are removed when each sample ends, so several guiders sharing one model in a multi-arm workflow never leave hooks firing on each other's runs.

**Before any long run (capture-preflight gate):** delete and re-add the Passive Telemetry Guider node, run 1 prompt x 1 seed x 2 steps at CFG 4.5, then

    python preflight_check.py "<csv>" --steps 2 --layers 24 --subs attn

It must print PASS (hook_mode passive-branch, branches cond+uncond, finite prompt_influence, seed, md5 prompt hash, complete layers). Run it again on every finished CSV, with `--prompts matched_prompts_30.tsv` to enforce the frozen prompt list. It also warns when one seed is reused across prompts.

**Tested:** `tests/test_passive_r3.py` (20 checks: flag forced off, shifted widgets, wrapper dropped with and without the second channel, separate vs pooled forwards, CFG 1, gen_len, hook cleanup across two guiders on one model) and the original `tests/test_passive.py`, all passing on the mock sampler. The mock now follows ComfyUI master's `calc_cond_batch` wrapper call and `transformer_options` hand-off (checked against comfy/samplers.py, 2026-10-06).

## 1. Passive Telemetry Guider: per-CFG-branch rows (`nodes.py`)

**Diff:** `nodes_passive_branch.patch` (diff against `capture_code/nodes.py`). Only `CoSPAFS_PassiveTelemetryGuider(Impl)` changed; every other guider is byte-identical.

**Install:** copy `nodes.py` over the one in `ComfyUI/custom_nodes/<cos-pafs>/` and restart ComfyUI. The node gets a new input, `split_cfg_branches`, which defaults to on. It sits last in the node's widget list so saved workflows load their old values in the right places (an earlier build put it before `layer_range`, which shifted every saved value by one and hooked 0 blocks). If you saved a workflow with that earlier build, delete and re-add the node, or check that `layer_range` is empty and `csv_session` holds the session name. A `layer_range` that matches no layer now prints a warning and captures all layers.

**What it does:**
- A `model_function_wrapper` records ComfyUI's `cond_or_uncond` list for every forward.
- The hook splits each activation batch by branch and computes g, gap, H and φ **per branch**. The result is the same whether ComfyUI batched cond and uncond together or ran them separately.

**New CSV columns** (appended at the end, so existing loaders still work):

| Column | Meaning |
|---|---|
| `branch` | `cond` / `uncond`; `pooled` if a batch couldn't be attributed; `all` when the split is off |
| `prompt_influence` | Cosine between the cond and uncond mean activations at that layer |
| `seed` | Sampler noise seed |

**Other changes:**
- `hook_mode` is `passive-branch`, which tags the new instrument.
- `prompt_hash` is now a stable md5, not Python's salted `hash()`.
- At CFG 1.0 ComfyUI skips the uncond pass, so only `cond` rows appear. They equal the cond rows of a CFG > 1 run at step 0.

**Tested** against a mock ComfyUI sampler (`tests/test_passive.py`, `tests/mock_comfy/`) with a small hooked transformer:
- Per-branch g, gap, H, φ and prompt_influence are **identical** (difference 0.0) for pooled vs separate forwards and for either branch order.
- The CSV value matches a direct recomputation (0.628728 vs 0.628727).
- Legacy split-off rows still differ, as expected.

It has **not** been run inside real ComfyUI. A 1-gen check on your machine: confirm the console shows `L0ac=... L0au=...` and the CSV has `cond` and `uncond` rows.

## 2. `cos_pafs_llm_measure_v2.py`: single-pass, context-invariant LLM telemetry (G4)

**Method:**
1. Generate with the KV cache (greedy by default) and save the ids and text to `.jsonl`.
2. Run one hooked forward pass.
3. For each generated position, compute:
   - **Fixed-window** g, gap and H over the trailing W rows. W = 64 is primary; 32 and 128 are also written. Columns `g_w64` etc. have constant n, so they are context-invariant.
   - **φ** from window means.
   - **Per-token novelty:** `nov_cos_mean` and `nov_out_frac`, i.e. the newest token's energy outside the top-8 subspace of the previous 64 tokens.
   - **Optional `--legacy_running`:** the v1 statistic over all rows, for a sanity check.
- `--teacher_force run.jsonl` scores another run's exact token ids. Use it for the Mistral ablit (G5) and Llama Base↔Instruct comparisons.

**Template:** v1 replaced *every* tokenizer's chat template with Llama-3.1's (`tokenizer.chat_template = LLAMA31_TEMPLATE`, unconditionally). If that line was live during the archived Qwen/Mistral runs, those models were prompted with foreign header text. v2 uses the model's own template and falls back to Llama-3.1 only when a tokenizer has none (base models). `--force_llama31_template` restores v1 behaviour.

**Tested** on CPU with a tiny randomly initialised Qwen2 model and an offline tokenizer (`tests/test_llm_window.py`):
- Windowed g and gap, running g and novelty match a direct recomputation from the saved ids to ≤ 4e-6. That is the CSV's 6-significant-figure rounding.
- `--verify_v1` agrees with a v1-style no-cache loop to ≤ 9e-5 relative. That is float32 noise between different-length forwards.
- Teacher forcing reproduces the generated run exactly (difference 0.0).
- Without `--legacy_running`, analysis takes 0.4 s per generation on CPU for the tiny model.

**Not yet run** on a real checkpoint or a GPU. First run on your machine:

    python cos_pafs_llm_measure_v2.py --model Qwen/Qwen2.5-1.5B-Instruct --n_prompts 2 --max_tokens 48 --legacy_running --verify_v1 --output_prefix smoke

You want to see `verify: max rel ... ≲ 1e-2` (bf16) and two generation lines with timings.

**G4 minimum** (greedy, the archive's 120 prompts):

    python cos_pafs_llm_measure_v2.py --model Qwen/Qwen2.5-1.5B-Instruct --output_prefix qwen15_v2
    python cos_pafs_llm_measure_v2.py --model Qwen/Qwen2.5-3B-Instruct   --output_prefix qwen3_v2
    python cos_pafs_llm_measure_v2.py --model Qwen/Qwen2.5-7B-Instruct   --output_prefix qwen7_v2

## 3. Matched diffusion study

See `R2_MATCHED_STUDY.md` and `matched_prompts_30.tsv` (3 prompts per theme). Tier 1 is 234 generations.
