# Data: where the telemetry goes

The raw telemetry (about 0.7 GB) and the derived cubes (about 0.3 GB) are too large for GitHub. They are distributed as a
Hugging Face dataset: [LINK](https://huggingface.co/buckets/modeLIMINAL/Mode_I-II_Data). Every path below is relative to the repository root. SHA-256 values for every file are in
`telemetry_sha256.txt`.

## Raw telemetry

| File or folder (dataset) | Place it at | Contents |
|---|---|---|
| `TELEMETRY.zip` (140,472,929 bytes, 506,345,966 unpacked; SHA-256 `d7f5a82e…6d391`) | unzip to `cache/tel/`, then each nested zip in `cache/tel/DM/` into `cache/tel/DM/<name>/` (`review/scripts/load.py` does this) | the passive archive: Z-Image Turbo (including the files labelled Z-Image Base, which the model-identity gate shows are Turbo weights), Mochi CFG 1 and 4.5, SD3.5 Medium and Large, the SD3.5 Medium resolution suite (`DM/`), and the language-model controls (`LLM/`) |
| `data/zib_abstract_conceptual/` (1 CSV; in `raw_captures_r3_r4.zip`) | `data/zib_abstract_conceptual/` | the one abstract-conceptual domain file labelled Z-Image Base, supplied separately (48 generations; Turbo weights, as above) |
| `data/r3_captures/` (90 files) | `data/r3_captures/` | Round 3 captures (SD3.5 Medium and the arm labelled Z-Image Base, which is Turbo weights; CFG and resolution arms) |
| `data/r4_captures/` (84 files) | `data/r4_captures/` | Round 4 explicit-branch captures (instrument v2) |
| `data/sd35m_cfg1.0_vs_5.5/` (2 CSVs) | `data/sd35m_cfg1.0_vs_5.5/` | legacy-schema SD3.5 Medium CFG study (Appendix: legacy instrument) |
| `SD1.5-attn.mlp-TE1-CFG7.5-R512-S(RAND)-STUDY[PRIMARY-UNET]-G480.zip` | unzip the 10 CSVs to `review/r3_unet/data/` | SD1.5 U-Net arm (R3 Part C) |
| `AS15-attn.mlp-TE1-CFG1.0-R180-SRAND-STUDYPRIMARY-AUDIO-G120.zip` | unzip the 10 CSVs to `review/r3_ace/data/` | ACE-Step 1.5 turbo audio arm (R3 Part B) |
| `data/r12_captures/` (6 folders, about 300 MB; as `r12_captures.zip`) | `data/r12_captures/` | Rounds 11-13: `Z1_ZIB_CFG4_primary` (real Z-Image Base, CFG 4, explicit branches), `Z2_ZIB_CFG1_variants` (Base, CFG 1, row-split statistics), `Z3_ZIT_CFG4_matched` (Turbo at Z1 settings), `R11_ZIT_CFG1_variants`, `R11_SD35M_CFG1_variants`, `ACE_turbo_matched` (ACE-Step turbo on the image prompts, Mode-III) |
| `ACE15B_180s_CFG4.zip` (2 CSVs + 2 layer JSONs, 37 MB) | `review/ace_base_r14/data/` | Round 14: ACE-Step 1.5 Base, 30 prompts x 2 seeds, 50 steps, CFG 4, explicit branches, two GPUs (gpu0 = seed 1000+2i, gpu1 = seed_b); join key (file, gen_idx, seed) |
| `ACE15T_180s_steptest.zip` (9 CSVs + 9 layer JSONs) | `review/ace_turbo_panel_r16/data/` | Round 16: ACE-Step 1.5 turbo, 30 Base prompts, 8 steps (seed_a, seed_b) and 1-5, 7, 9, 10 steps (seed_a), CFG 1, shift 3. Cubes: `review/ace_base_r14/scripts/build_cubes.py <name> cond 0`; score with `SCR=<cube dir> python3 review/ace_turbo_panel_r16/scripts/score_r16.py` |
| `R17_full.zip` (4 CSVs + 4 layer JSONs) | `review/tier_controls_r17/data/full/` | Round 17: PixArt-Sigma, Cosmos-Predict2 2B, SD3 Medium, SD3.5 Medium; 60 prompts, seed 1001, explicit branches. Cubes: `review/zib_turbo_mixup_2026-10-18/r12/scripts/build_cubes.py <name> cond|uncond 0`; score with `review/tier_controls_r17/scripts/score_r17.py` |
| `R18.zip` (27 files: CSVs, per-branch spectra and layer JSONs) | `review/r18/data/` | Round 18: SD3.5 Large and Large Turbo (60 prompts, 38 blocks), a second Z-Image Turbo session with image rows, Z-Image Base at shift 6, and the Round 17 models replayed (`R18S_*`); settings in `review/r18/out/r18_settings.json`, weight-file SHA-256s in `review/r18/out/r18_weight_hashes.json`. Cubes: `review/zib_turbo_mixup_2026-10-18/r12/scripts/build_cubes.py <name> cond|uncond 0`; score with `SCR=<archive cubes> R18CUBES=<Round 18 cubes> python3 review/r18/scripts/score_r18.py`. The addendum B CKA outputs (pairs, values, per-pair matrices) are in `review/r18/cka_out/` |

On the dataset, `data/r3_captures/`, `data/r4_captures/`, `data/sd35m_cfg1.0_vs_5.5/` and `data/zib_abstract_conceptual/`
are packed in `raw_captures_r3_r4.zip`, and `data/r12_captures/` in `r12_captures.zip`; unzip both at the repository root.

## Weight files

`model_weight_sha256.csv` (in this repository and on the dataset) lists the SHA-256 of the weight file of every model
measured. Round 18 recorded eight of them at capture time; the same files hashed again on 2026-11-12 give the same values.
The ACE-Step 1.5 Base and turbo and Mochi 1 hashes were taken on 2026-11-12 from the files on the capture machine, so they
identify the files now on disk; no hash was recorded when those captures were made.

## Derived bundle

`mode-series-derived.zip`: unzip at the repository root. It adds:
- `cache/F_*.pkl` (archive cubes from `build_fields.py`), `cache/F3_*.pkl` (R3 arms, `r3_load.py`), `cache/R5_*.pkl` (R4 captures, `r5_load.py`);
- `cache/pq/*.parquet` (the archive as parquet, from `load.py`). It includes the Llama 3.1 8B Instruct run, which Round 9 (B6, B6b) uses; the Llama 3.1 8B base run, which no paper uses, is left out;
- `cache/fig/data/` (v11 figure-data tables read by Mode-I figures I-F1, I-F9 and I-S4);
- `review/r3_unet/out/u0_cache.npz`, `review/r3_ace/out/cache_{attn,mlp}.npz`, `review/standing_wave/sw_cache/*.npz`,
  and the small `E1_curves.pkl` / `E5_curves.pkl` caches of Rounds 2 and 3.

`mode-series-derived-r11-r14.zip`: unzip at the repository root. It adds the cubes of Rounds 11-14 to `cache/`:
`F_Z1_*`, `F_Z1U_*`, `F_Z2_*`, `F_Z3_*`, `F_ZITSV_*`, `F_SD3SV_attn`, `F_ACEB*_*`, `F_ACET_*` (cond/uncond branch cubes), `V_Z2_*`, `V_ZITSV_*`, `V_SD3SV_attn`
(row-split variant arrays) and the Round 13 holdout splits (`*H_*`). `review/build_r11_r14_cubes.sh` rebuilds them from the raw captures above.

The cubes of Rounds 16-18 are not in a bundle; build them from the raw captures with the commands in the table above.

Everything in it can be rebuilt from the raw telemetry with the loaders named in README.md. The `.pkl` files are Python
pickles: load them only from a source you trust.

## Capture schema

Each telemetry CSV row is one (generation, step, layer, submodule[, branch]) cell: `gen_idx, token_idx, token_type, layer,
submodule, g_micro, phi_micro, phi_micro_skip, spectral_gap, sv_entropy, sigma, seq_len, hidden_dim, hook_mode, prompt_hash,
prompt_len, gen_len, prompt_text, branch, prompt_influence, seed` (R3 schema; legacy files lack `submodule`, `phi`, `branch`).
Known loader traps (header-less Mochi file, auxiliary 100-step Turbo generations, salted `prompt_hash`, capture regimes)
are handled in `review/scripts/common.py` and described in Mode-I, Section 2 and Appendix A.
