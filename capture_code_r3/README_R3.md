# capture_code_r3: registry v3.0 and hooking fixes for the Mode-I/II release (2026-10-08)

This folder is a copy of `capture_code_r2/`. Only `nodes.py` and `model_registry.json` changed, plus the new files listed below. The R2 folder is untouched. Diff of `nodes.py`: `nodes_r3_blocks.patch`.

**Install:** copy the whole folder over the node folder (both `nodes.py` and `model_registry.json` changed), restart ComfyUI, and re-add the guider node in each workflow.

## Why SD1.5 and SDXL wrote nothing

There were two separate faults, and either one alone was enough to stop telemetry.
1. **Wrong registry key.** The node looks up the registry by the class name of ComfyUI's diffusion model. For SD1.5 and SDXL that class is `UNetModel` (`comfy.ldm.modules.diffusionmodules.openaimodel`). The registry keys were `UNet2DConditionModel_SD15/_SDXL`, which are diffusers names, so they never matched.
2. **No UNet path in the Passive guider.** Only the EMAG guider had UNet linearization. The Passive guider's `_find_blocks` tried the registry and then `transformer_blocks / joint_blocks / blocks / layers`. A UNet has none of those, so it returned `[]` and printed "No transformer blocks found — telemetry disabled" (visible only with log_telemetry on).

ACE-Step 1.5 failed the same way. Its blocks sit at `diffusion_model.decoder.layers`, and the R2 code checked `hasattr(dm, block_path)`, which cannot follow a dotted path. ACE-Step v1 was auto-detected through `transformer_blocks`, but its registry key (`AceStepTransformer`) never matched, so only the generic attn/MLP fallbacks were used.

## What changed

**`nodes.py`**
- A new `_resolve_block_list()` accepts a dotted path (`decoder.layers`), a composite (`double_blocks + single_blocks`) and `UNET_LINEARIZE`. Both guiders use it.
- A new shared `_linearize_unet()` returns blocks in forward order: input_blocks, then middle_block, then output_blocks.
- The Passive guider falls back to UNet linearization. It now always prints a line when no blocks are found, listing the model's children.

**`model_registry.json` v2.1**
- `UNetModel` replaces the two SD keys. It covers both SD1.5 (16 blocks; D 320/640/1280) and SDXL (70 blocks; D 640/1280).
  - attn hook: `attn1.to_out.0`
  - MLP hook: `ff.net.2`
- `AceStepConditionGenerationModel` (ACE-Step 1.5):
  - block_path `decoder.layers`, 24 layers, D 2048
  - attn hook: `self_attn.o_proj`; MLP hook: `mlp.down_proj`
- `ACEStepTransformer2DModel` (ACE-Step v1):
  - block_path `transformer_blocks`, 28 layers, D 1536
  - attn hook: `attn.to_out.0`; MLP hook: `ff`
- All three set `dual_hook_supported` true, so attn and MLP are both captured. `verified` stays false until one real sampler run.

## Tested

`tests/test_r3_blocks_real_comfy.py` uses the **real** ComfyUI classes from master 58b176f, not the mock. Output: `tests/test_r3_blocks_real_comfy.log`, 27/27 PASS.
- **Real-size configs:** the block counts and per-block D come out as listed above. ACE-Step 1.5 even layers use sliding-window attention.
- **Tiny configs with real forwards on CPU:** every block's attn and MLP hook fires exactly once per forward, the captured D equals that block's width, and the UNet block list is in forward order.
- **Regression:** R2 `nodes.py` finds 0 blocks on SD1.5 and on ACE-Step 1.5, which reproduces the report.
- **Old mock tests:** `test_passive_r3.py` passes 20/20, and `test_passive.py` gives the same numbers as before (0.628728 vs CSV 0.628727).

Run it yourself: `COMFY=/path/to/ComfyUI python tests/test_r3_blocks_real_comfy.py` from this folder.

## Fix: SD1.5 / SDXL crashed at step 0 (Zac's report, 17:02Z)

**Symptom:** `RuntimeError: The size of tensor a (640) must match the size of tensor b (320)` in `predict_noise`, at the first step.

**Cause:** phi_micro is the cosine between consecutive changes in the per-block mean activation. That needs three captures of the same width. UNet blocks change width at each level (SD1.5 320/640/1280), so the subtraction failed where two levels meet. The earlier tests only checked the hooks, and the crash happens after the hooks, in `predict_noise`.

**Fix (`nodes.py`):** phi_micro and phi_micro_skip are left **blank** in the CSV where the captures they use differ in width. Everywhere else they are computed as before, so DiT runs are unchanged. The EMAG guider writes nan in the same case.

**Test:** `tests/test_r3_sampler_e2e.py` runs real ComfyUI sampling (a BaseModel and ModelPatcher with SD1.5- and SDXL-shaped tiny UNets, CFG 4.5, a CSV session) through the Passive guider.
- The previous R3 `nodes.py` fails with the reported error on both SD1.5 and SDXL.
- The fixed one passes: 3 steps, both branches, finite g, and phi blank exactly where widths differ (`tests/test_r3_sampler_e2e.log`).

**For the analysis:** UNet phi only exists inside a level. g, gap and entropy are on every row.

## Caveats for the analysis

- **UNet:** D and token count both change with level. SD1.5 at 512 px has 4096, 1024, 256 and 64 tokens. Where tokens < D, g is rank-limited. Compare each block with its own isotropic (Marchenko–Pastur) baseline, never raw g across levels. The `seq_len` and `hidden_dim` CSV columns record both per row.
- **ACE-Step 1.5:** tokens = latent frames / 2, and the latent runs at 25 frames/s, so there are 12.5 tokens/s. N ≥ D = 2048 needs at least 164 s of audio. The protocol uses 180 s.

## Registry v3.0 (16:22Z request: new models, and fix the keys that never matched)

In v2.1 only 8 of 25 keys were ComfyUI class names. Flux, Chroma, HunyuanVideo and Hunyuan Image 2.1 wrote **no telemetry**; Wan, Mochi and LTX were captured only through generic fallbacks; Cosmos-1 hooked nothing.

v3.0 has **34 keys, every one a real class in ComfyUI master 58b176f** (checked by grep and by building each class). Every entry has a `class_checked` line saying what was tested. `verified` stays false everywhere until one real sampler run.

**New models**

| Model | Key | Blocks hooked | attn / MLP hook | D, blocks |
|---|---|---|---|---|
| Ideogram 4 | `Ideogram4Transformer2DModel` | `layers` | `attention.o` / `feed_forward.w2` | 4608, 34 |
| Krea 2 | `SingleStreamDiT` | `blocks` (text-fusion blocks not hooked) | `attn.wo` / `mlp.down` | 6144, 28 (class defaults) |
| MiniMax H3 | `MiniMaxH3Model` | `blocks` (token refiner not hooked) | `attn.out_proj` / `mlp` | 5376, 50 |
| Qwen-Image 2.1 | `QwenImage21Transformer2DModel` | `transformer_blocks` | `attn.to_out.0` / `img_mlp` | 4096, 32 |

**Fixed or added families:** Flux 1 and Flux 2 (`Flux`, also LongCat), `Chroma`, `ChromaRadiance`, `HunyuanVideo` (1.0, 1.5, Image 2.1 and variants), `WanModel` and 7 Wan subclasses (VACE, Camera, HuMo, S2V, Animate, SCAIL, SCAIL2), `LTXVModel` (LTX-Video 0.9.x), `LTXAVModel` (LTX-2 / 2.3), `CogVideoXTransformer3DModel`, `GeneralDIT` (Cosmos 1), `MiniTrainDIT` (Cosmos Predict2), `Anima`, `HiDreamImageTransformer2DModel`, `HiDreamO1Transformer`, `OmniGen2Transformer2DModel`, `Kandinsky5`, `QwenImageTransformer2DModel` (now with its image MLP), `ErnieImageModel`.

**Folded or renamed, archive unchanged:** SD3/3.5 Medium into `OpenAISignatureMMDITWrapper` (same class); Lumina 2 into `NextDiT` (same class); Mochi `MochiTransformer3DModel` renamed to its class `AsymmDiTJoint` with hook `attn.proj_x`. The tests show R3 hooks exactly the R2 modules for Z-Image and SD3.5, and gives identical g for Mochi. SD3.5 and Mochi stay attn-only, as in the archive.

**Removed:** every other diffusers-named key. Each one is replaced by an entry under the real class name.

### Node changes for v3.0 (`nodes.py`)

- **Single-branch forwards** are captured whole, whatever the tensor layout. MiniMax H3 packs one sample into a 2-D [tokens, D] tensor, which the batch split would otherwise cut along tokens.
- **`transformer_options_set`** (registry field): the Passive guider merges it into the transformer options. Used to turn Qwen-Image 2.1's prefix cache off.
- **`nn.ModuleDict` block lists** are iterated by value. Cosmos-1 keeps its blocks in a ModuleDict, and `list()` gave the key strings, so 0 hooks were registered with no error.
- **`batch_dim`** (registry field): for sequence-first models (Cosmos-1: attn output [THW, B, D], MLP [T, H, W, B, D]) the CFG batch axis is moved to the front before the cond/uncond split. Tested: each branch's g equals a B=1 forward of that sample.

### Traps (each one is in the entry's notes and tested)

- **Qwen-Image 2.1 prefix cache.** It is on by default. Step 0 then runs every block twice (text prefix, then image rows) and later steps see image rows only. The registry turns it off, so every step sees the full text+image sequence.
- **MiniMax H3** runs at batch size 1 and puts text, audio and video tokens in one sequence. g is over the mixed sequence.
- **Ideogram 4** uncond pass has no text, so it has fewer tokens than the cond pass. Compare cond with cond.
- **Flux, Chroma and Hunyuan**: only the double-stream blocks are hooked. Single blocks have no separate attn/MLP output modules (`linear2` mixes both).
- **Fused MLP kernels.** Wan, LTX, MiniMax and Qwen 2.1 compute the last MLP Linear through `comfy.ops.linear_input_act`, which skips that Linear's `forward()` on fused paths (INT8, fp16 accumulation). Their MLP hook is therefore on the parent module (`ffn`, `ff`, `mlp`, `img_mlp`), whose output is the same tensor.
- **MoE and chunked MLPs** (HiDream `ff_i`, Kandinsky 5 above 8192 tokens): the inner Linears fire several times, so the whole module is hooked.
- **Token sets differ by model.** HiDream single-stream blocks add text tokens (N jumps at block 16); OmniGen2, Ernie and CogVideoX include text rows; LTX-2 captures the video stream only; HiDream-O1's KV cache drops the text prefix on cache hits, including step 0 of a repeated prompt.
- **WanAnimate2Model** has no entry: its hooks fire twice per block when its pose cache is cold.

### Not verified

- No real checkpoint was loaded and no sampler was run. Dims marked "class defaults" or "public config" (Krea 2, Kandinsky 5 block counts, Anima, Lumina 2, LTX-2.3, Flux 2 Klein) were not read from a checkpoint.
- Real-size configs were built on the meta device; forwards used tiny configs of the real classes on CPU.

### Tests (`tests/registry_v3/`, all read the shipped `model_registry.json`)

| Test | Covers | Result |
|---|---|---|
| `test_D_mine.py` | Ideogram 4, Krea 2, MiniMax H3, Qwen-Image 2.1, archive identity (Z-Image, SD3.5, Mochi) | 29/29 PASS |
| `test_A_flux.py` | Flux 1/2, Chroma, ChromaRadiance, HunyuanVideo family | 106/106 PASS |
| `test_B_video.py` | Wan family, LTX, LTX-2, CogVideoX, Cosmos 1, Cosmos Predict2 | 184/184 PASS |
| `test_C_image.py` | HiDream I1 and O1, OmniGen2, Kandinsky 5, Anima, Qwen-Image, Ernie | 130/130 PASS |

Also rerun on the final code: `tests/test_r3_sampler_e2e.py` (real sampling, SD1.5 and SDXL UNets) 14/14, `tests/test_r3_blocks_real_comfy.py` (UNet, ACE-Step) 27/27, mock `test_passive_r3.py` 20/20, and `test_passive.py` unchanged (0.628728 vs CSV 0.628727).

Run: `COMFY=/path/to/ComfyUI python tests/registry_v3/test_D_mine.py` (add `NODES_R2=<capture_code_r2>` for the archive-identity checks). The v2.1 registry is kept as `tests/fixtures/model_registry_v2.1.json` for the before/after checks.

## New files

- `ace_step/ace_prompts_30.tsv`, `ace_step/ace_prompts_30.txt`, `ace_step/ACE_STEP_PROTOCOL.md`: the audio arm.
- `latent_spectrum.py`: the latent-Λ gate, run on images already generated (predictions in `review/r3_prereg/R3_PREREGISTRATION.md`).
