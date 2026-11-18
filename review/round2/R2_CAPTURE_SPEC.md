# Round 2 capture spec: batch 1 (diffusion) and G4 (LLM)

Written 2026-10-05 in answer to Zac's questions in the Round 2 thread. Everything below is sized to be as small as possible while still settling the open question.

---

## 0. What the emag_scale = 0 answer settles, and what it doesn't

The current `nodes.py` only writes CSV rows when EMAG is active (`emag_active = phase_emag_scale > 1e-4`, CSV write at about line 2399). At emag_scale 0, the current Diagnostic guider would write nothing. So the legacy files came from an **older** build of the node, and we don't have that build.

In every build we do have, telemetry is computed inside `_make_emag_hook`. That hook **replaces the attention output with the previous step's EMA whenever it is registered**, regardless of emag_scale. The scale only weights the final guidance combination. So emag_scale = 0 does not by itself mean the measured pass was unperturbed. It depends on which hook the older build used, and that is why step **A** below exists.

---

## 1. Batch 1 (diffusion)

### A. Instrument A/B. Run this first. It is tiny and it decides whether B is needed at full size.

- **Goal:** one test showing whether the 10× g gap between the legacy SD3.5 files and the passive campaign comes from the instrument.
- **Model and settings:** SD3.5-Medium with exactly the res-suite settings. That means the pyramids prompt, 512 px, CFG 1.0, 100 steps, the same sampler, scheduler and text-encoder setup as the res-suite, and the same seeds.
- **Runs:** 3 seeds × 2 guiders, 6 generations in total:
  1. the **old build** of "CoS-PAFS EMAG Guider (Diagnostic)" at emag_scale 0, if you still have it;
  2. the **Passive Telemetry Guider**, hook_mode auto.
- **Readout:** the median g ratio between the two guiders at matched step and layer, plus whether the images are pixel-identical.
  - If the two guiders' g agree within a few percent, the instrument is cleared. The 10× gap then comes from settings: CFG 4.5, the 3-encoder setup, or branch-mean capture.
  - If the images differ between guiders, the old build was perturbing the forward pass.
- **If the old build is gone:** skip A and run B at full size.

### B. G0b: passive re-capture of the legacy SD3.5 studies

Use the Passive Telemetry Guider for both studies, with the same settings and seeds as the originals.

| Study | Prompt | Resolution | CFG | Steps | Gens |
|---|---|---|---|---|---|
| Resolution suite | pyramids | 512, 768, 1024 | 1.0 | 100 | 10 seeds per resolution (archive has 30) |
| CFG study | pyramids | 256 px (seq 256) | 1.0 and 5.5 | 100 | 30 per arm, same seeds as the original |

- **Scope if A clears the instrument:** the CFG-study 5.5 arm still needs per-branch rows (C). The resolution suite needs nothing more, because it is CFG 1.0 and therefore cond-only.
- **What this decides:** whether the four-crossing β(g), TRI, the complex eigenpair and the CFG spread result exist on the instrument used for every other Mode-I result.

### C. G0: per-branch rows for CFG > 1 campaigns

**Code change (Passive guider only, roughly 30 lines; I can draft the patch).**
1. In `predict_noise`, add a `model_function_wrapper` to `model_options`. It records `c['transformer_options']['cond_or_uncond']` on `self` before calling `apply_model`. This list says which batch rows are cond (0) and which are uncond (1) for the forward pass that follows.
2. In `_capture_one`, take the batch dimension B of `activation` (shape [B, N, D]). Split it into `len(cond_or_uncond)` chunks. Compute g, gap and H **per chunk**, and store each under key (layer, sub, branch).
   - This works the same whether ComfyUI pools the branches in one forward or runs two separate forwards, so the VRAM-dependent regime stops mattering.
3. **CSV changes:**
   - Add a `branch` column (`cond` or `uncond`), so each step writes 2 rows per layer and submodule.
   - Write `seq_len` per branch.
   - Compute φ per branch from the per-branch mean vectors.
4. **Free extras on the same pass:**
   - `prompt_influence` = cosine of the cond and uncond mean vectors per layer, which are already stored for φ.
   - A stable `prompt_hash`, sha1 of the text, replacing the salted `hash()`.

**Runs (minimum viable):**
- **Prompts:** 30 prompts (3 per domain) × 2 seeds, i.e. 60 gens per model. Reuse the archive's prompt text and seeds, so the cond rows can be laid next to the archive rows and we see how much each archived number moves.
- **Models and settings**, at the archive's settings (1024 px, same sampler and encoders):

| Model | Steps | CFG | Note |
|---|---|---|---|
| SD3.5-M | 40 | 4.5 | Cheapest, run first |
| ZIB | 25 | 4.0 | |
| ZIB | 25 | **1.0** | Extra arm: identical capture to ZIT, isolates distillation for G1 |
| SD3.5-L | 28 | 3.5 | |

- **Total:** about 240 diffusion generations. With A (6 gens) and B (60 + 60 gens) it comes to **roughly 370 generations**.

**What G0 decides:**
- Every CFG > 1 vs CFG = 1 absolute comparison: ZIB↔ZIT, Mochi CFG, and the SD3.5 CFG card.
- Whether the σ≈1 seed-share spike (Round 2 §3.4) and the step jitter in seed deviations (§3.3) are capture artifacts.

---

## 2. G4 (LLM): not a reproduction of the LLM archive

**The problem.** The current script recomputes the full context at every token, with no KV cache and with causal attention. Earlier tokens' hidden states therefore never change. Each step's matrix is the previous matrix plus one new row, so every per-step value is a running statistic over a growing sample. That is why everything is exactly monotone in context length. The existing generated text was never saved (only `prompt_text` is in the CSV), so the old runs can't be re-scored.

**The fix makes capture about 10× cheaper than before, not more expensive.** Because of causality, a single forward pass over the finished sequence already contains every token's hidden state, which is everything the per-token recompute loop produced.

1. **Generate with the KV cache on**, normally and fast. Use greedy decoding (temperature 0) so runs are reproducible, and **save the token ids and text**.
2. **Run one forward pass** over prompt + generation with the hooks on. Hold each layer's [n, D] activation on the GPU.
3. **Compute per-position statistics from that one pass:**
   - **Fixed-window** g, gap and H over the trailing W tokens, with W ∈ {32, 64, 128}. Because n = W is constant, this is context-invariant, which is the point of G4.
   - **Per-token novelty:** the new token's norm, its cosine to the running mean, and the fraction of its energy outside the running top-k subspace.
   - **Optional legacy running statistic,** reproduced exactly for a sanity check on a few gens. Use the n×n Gram matrix XXᵀ instead of D×D XᵀX: same nonzero eigenvalues, much cheaper when n < D.
4. **Cost.** The script's own docstring says about 2–3 min per 256-token 7B generation on a 3090. The new path is roughly generation with cache plus one forward plus small W×W eigensolves, about 10–20 s per generation.

**Minimum G4 run:**
- Qwen2.5 1.5B, 3B and 7B.
- The same 120 prompts as the archive, 1 greedy sample each: 360 generations, **on the order of 1–2 hours**.
- This decides whether any LLM per-token result (β crossings, the Qwen size ladder) survives on a context-invariant observable.

**Riders on the same harness (cheap, optional, each unblocks a gate):**
- **G5 (Mistral Instruct vs ablit):** generate with Instruct, then **teacher-force the identical token ids** through ablit in one pass each. Add 40 refusal-triggering prompts paired with 40 matched benign prompts. Round 2 found the ablit edit starts at layer 6 and peaks at the last layer; this tests whether that profile grows when refusal is engaged.
- **Llama Base↔Instruct:** run Llama-3.1-8B-Instruct with the dual attn+mlp hook on the same prompts, then teacher-force the Instruct text through Base. This removes the hook mismatch that blocks that pair.

**Script changes to `cos_pafs_llm_measure.py`:**
- Replace the `use_cache=False` loop with `model.generate(..., do_sample=False)`, followed by one hooked forward.
- Make the hook store activations instead of computing statistics per call.
- Add a post-pass that computes the windowed and per-token statistics and writes them in the existing CSV layout, plus a `window` column and `generated_ids` / `generated_text` columns.
- Replace the salted `prompt_hash` with sha1.

I can draft this as a new script (`cos_pafs_llm_measure_v2.py`). This sandbox has no GPU or torch, so it would arrive untested, with a 2-prompt smoke-test checklist for your machine.

---

## 3. Order

1. **A** (6 gens).
2. **G4 minimum** (360 short LLM gens).
3. **C**, starting with SD3.5-M, then ZIB at CFG 4.0 and 1.0, then SD3.5-L.
4. **B**: the full resolution suite and CFG study, needed only if A does not clear the instrument. The CFG-study 5.5 arm needs C's per-branch rows either way.
5. **G5 / Llama riders** whenever convenient.

---

## Update (2026-10-05, 14:40 UTC)

Zac approved the instrument changes. They are implemented and smoke-tested in `/mnt/project-files/capture_code_r2/` (see its `README_R2.md`).

The single-prompt 100-step reruns are replaced by the matched study in `capture_code_r2/R2_MATCHED_STUDY.md`: Tier 1 is 234 generations, 3 prompts per theme.
