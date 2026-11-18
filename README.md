# Mode-II: Projection, Pooling and Tangent Agreement

Zac Doyle, mode:LIMINAL. Companion code for the paper "Projection, Pooling and Tangent Agreement: Equivalence Relations for Layerwise Spectral Trajectories of Generative Transformers" (Mode-II) (`papers/mode-II/main.tex`).
The companion paper, Mode-I (the empirical atlas), is at https://github.com/Z-L-D/mode-I and carries the same analysis code.

This repository contains the paper source, the code that draws every figure, the analysis code that computes every number,
the preregistrations and scoring plans those numbers were scored against, and the analysis scripts with their
recorded outputs. The findings ledger and internal planning notes are not included; they are available from the author on
request for review. The raw telemetry is too large for GitHub and is distributed separately (see [DATA.md](DATA.md)).

## Model identity and Rounds 11-18

The archive campaign whose files are labelled "Z-Image Base" (476 generations, 25 steps, CFG 4, pooled capture) is
Z-Image Turbo: a preregistered model-identity gate (Round 12) matched it to Turbo at r = 0.99 and not to new captures of
the real Base. Both papers call it **ZIT-25** (internal key `ZIB` in the scripts and cubes), treat every result on it as a
Turbo result, and base every lineage statement on the real Base captures. The last eight preregistered rounds are:

| Round | Folder | What it tests |
|---|---|---|
| 11 | `review/peer_review_2026-10-17/` (`capture/PREREG_R11.md`, `r11/`) | uncentred-statistic controls: image rows, caption rows, centred, top-8 channels removed |
| 12 | `review/zib_turbo_mixup_2026-10-18/` (`rerun/PREREG_R12.md`, `r12/`, `IMPACT_AUDIT.md`) | model-identity gate; real Base vs Turbo lineage (middle tier; 0.55/0.77 on the native grid, 0.71/0.72 on the 24-point grid) |
| 13 | `review/zib_turbo_mixup_2026-10-18/r13/` | split-half holdout: what distillation changed in Z-Image |
| 14 | `review/ace_base_r14/` (`PREREG_R14.md`, `RESULTS_R14.md`) | ACE-Step 1.5 Base: second lineage pair (0.85/0.57), depth sign sequence, B1-B10 replay |
| 15 | `review/r15_deep_dive/` (`PREREG_R15.md`, `RESULTS_R15.md`, `exploratory/`) | held-out replications of the deep-dive findings: text-to-latent handoff, guidance and seed share, block-restricted layer null, lineage ROC, LLM tangents, LLM-DiT prompt geometry |
| 16 | `review/ace_turbo_panel_r16/` (`PREREG_R16.md`, `RESULTS_R16.md`) | ACE-Step turbo at its release setting (8 steps) and a 1-10 step panel on the Base prompts and seeds |
| 17 | `review/tier_controls_r17/` (`PREREG_R17.md`, `RESULTS_R17.md`) | tier controls: PixArt-Sigma vs Cosmos-Predict2 2B (unrelated, 28 layers) and SD3 Medium vs SD3.5 Medium (same vendor, 24 layers) |
| 18 | `review/r18/` (`PREREG_R18*.md`, `RESULTS_R18.md`, `RESULTS_R18_ADDENDUM_B.md`) | documented teacher-student pair SD3.5 Large vs Large Turbo, a second Z-Image Turbo session with image rows, Base at shift 6; addendum A trace ratio, addendum B activation CKA baseline (`cka_out/`) |

`review/r12_scoring/` holds the scoring helpers shared by Rounds 12 and 14, and `review/build_r11_r14_cubes.sh` rebuilds
the Round 11-14 cubes from the raw captures. `check_numbers.py` re-reads every Round 11-18 number from the output that
produces it, compares it with the printed value in the paper, and writes `NUMBERS.md` (213 of 213 pass on the release copy; set `SCR` to the cube folder for the one check that reads `F_ACEB_attn.pkl`).

`review/r15_deep_dive/review_checks/` holds the post hoc checks requested by four external reviews (DeepSeek, Qwen, Kimi, GLM):
grid size and sample size (`rc1_grid_and_n.py`), depth calibration of the tangent, asymmetry, the SD3.5 M-L phase, tiers by row
set, caption-row energy, identity-gate operating characteristics and per-family separation (`rc5_glm_kimi.py`, `rc5b_lowpass.py`),
and per-domain text-decoder recall (`rc12_tfidf_per_domain.py`). Questions that need new captures are Mode-III tests. The most consequential check: the depth tangent is not
invariant to depth, so its three tiers are stated for depth-matched pairs.

Round 2 of the reviews (DeepSeek, Qwen, Kimi, GLM) added five checks on existing data:
a step-joint null for the per-step layer-axis counts, replacing a binomial that assumed independent steps (`rc13_block_null.py`);
the aggregate surrogates with 1000 draws (`rc14_surrogates_1000.py`); the Round 12 tier criteria re-scored on every row set, where
image-row attention fails the 0.10 same-weights margin (`rc15_rowset_criteria.py`); prompt RSA on image-only and centred rows
(`rc16_rowset_rsa.py`); and an independent numpy recomputation of the layer-beta curves from the raw telemetry, which matches the
analyzer to 1e-16 (`rc17_beta_curves_recompute.py`). RC13, RC14 and RC17 need the raw archive with the separately supplied
Z-Image abstract-conceptual file in place (`DATA.md`); without it, 25-step Turbo attention has 434 instead of 482 generations
and its aggregate pattern changes.

Round 3 of the reviews added four more: the tangent on each coordinate alone and the
identity gate on g, gap and H separately (`rc18_round3_checks.py`, RC18-RC19); the relative onset at thresholds 0.3-0.7 and
bootstrap intervals for the pooling increment by resolution (`rc20_onset_pooling.py`, RC20-RC21). RC18-RC19 run on the derived
cubes; RC20-RC21 need the archive and Round 5 cubes.

Rounds 4-7 of the reviews added RC22-RC33 in the same folder: a smoothness-preserving
surrogate null, trimmed MLP, the same-weights session band, grid size and interpolation, the identity gate on every run, and the
native-grid tangent with its own nulls (`rc33_native_grid.py`), which the papers now lead with. `review/posthoc_2026-11-10/`
(`PLAN.md`, `RESULTS.md`, `scripts/`, `out/`) holds the post hoc sweep RC34-RC49 on existing captures: the Round 17 models replayed
through the earlier rules, the centred and pruned tangent, and the tangent per denoising state.
Round 8 of the reviews added RC53-RC55 in `review/r15_deep_dive/review_checks/`: an inventory
of every depth-matched unrelated pair with the exact false-positive bound (`rc53_fpr_inventory.py`), the tangent by depth
frequency, and MLP variants on the row-split captures (`rc54_rc55_round8.py`).
Round 9 added RC56 (`rc56_grid12_reconstruction.py`): the 12-point tangent
reconstructed from native-grid depth frequencies, with and without an ideal low-pass before resampling.

Warning for the v1 language-model scripts: the v1 LLM capture re-encodes the full context at every token, so every logged
value is a running statistic over a growing context. The analyzer's per-token views (phases, onsets, crossings) are not
meaningful on those files; use them only for fixed-context comparisons, as the papers do.

Series plan: a "Mode-IV" paper on renormalisation-group and black-hole analogies is not planned. It is not cited in Mode-I or Mode-II. Language-model telemetry and the text-encoder side of text-to-image
models move to Mode-III.

## Layout

| Path | What it holds |
|---|---|
| `papers/mode-II/` | `main.tex`, `references.bib`, `main.bbl`, built `main.pdf`; `figures/` (PDF + PNG + `CAPTIONS.md`); `scripts/` (one script per figure, `make_all_figures.py`); `data/` (the exact plotted values of every figure, with a `source` column) |
| `papers/common/figstyle.py` | shared plotting style |
| `papers/_sources/` | `FINAL_FIGURE_INVENTORY.csv` (the figure scripts check transcribed values against the internal ledger text when it is present and skip that check otherwise) |
| `papers/LIT_REVIEW.md`, `papers/CITATION_KEYS.md` | how each bibliography entry was checked |
| `review/scripts/` | archive loaders: `load.py` (unpack `TELEMETRY.zip`, fix the header-less Mochi file), `build_fields.py` (gens × steps × layers × [g, gap, H] cubes), shared estimators |
| `review/round2/` … `review/round7/` | review rounds 2–7: preregistrations (with SHA-256 files), scripts, recorded outputs (`out/`) and results |
| `review/r8_correlations/`, `review/r9_salvage/`, `review/r10_final/` | Rounds 8–10 (preregistered checks of external "missed correlation" readings; the language-model salvage round with its text-control addendum; the final sweep and cross-paper consistency check): preregistrations with SHA-256 files, scripts, outputs, results |
| `review/lenses/`, `review/standing_wave/` | preregistered lens round and standing-wave re-test (the lens baselines appear in the papers) |
| `review/r3_prereg/` | R3 preregistration (SHA-256 prefix `efa613eb`) for the SD1.5 U-Net (Part C) and ACE-Step (Part B) arms |
| `review/r3_unet/`, `review/r3_ace/` | U-Net and audio arms: scoring plans (with SHA files), scripts, outputs, results |
| `review/round6/analyzer/cos_pafs_live_analyzer.py` | the CoS-PAFS Live Analyzer, used unchanged for every layer-axis β reading |
| `capture_code_r3/` | ComfyUI capture node (`nodes.py`), model registry v3.0, prompt files, preflight check and tests |
| `review/peer_review_2026-10-17/`, `review/zib_turbo_mixup_2026-10-18/`, `review/ace_base_r14/`, `review/r12_scoring/`, `review/r15_deep_dive/`, `review/ace_turbo_panel_r16/`, `review/tier_controls_r17/`, `review/r18/` | Rounds 11-18 (see above) |
| `review/posthoc_2026-11-10/` | post hoc sweep RC34-RC49 on existing captures (see above) |
| `check_numbers.py`, `NUMBERS.md` | Round 11-18 number check and its generated table |

Every Python path is relative to the repository root. Two environment variables override the defaults:
`COSPAFS_ROOT` (repository root) and `COSPAFS_CACHE` (derived cubes; default `<root>/cache`).
The older `SCR`, `REVIEW`, `CUBES`, `TEL` variables used by individual scripts still work.

## Quick start: rebuild the paper and its figures (no telemetry needed for most figures)

    python3 -m pip install -r requirements.txt
    cd papers/mode-II
    python3 scripts/make_all_figures.py          # writes figures/*.pdf|png and data/*.csv
    pdflatex main && bibtex main && pdflatex main && pdflatex main
    cd ../.. && python3 check_numbers.py         # Round 11-18 numbers against their outputs

Two scripts check the mathematics numerically and need no telemetry: `papers/mode-II/scripts/pooling_bound_check.py` (the pooling theorem) and `papers/mode-II/scripts/lemma_rotation_invariance.py` (the rotation-invariance lemma, its corollary on which network symmetries the telemetry cannot see, and the counterexample showing that equal triples do not fix the spectrum; 39 checks, written to `papers/mode-II/data/II-lemma_rotation_check.csv`).

Figures that recompute from telemetry cubes need the derived bundle (below) unpacked at the repository root.
With it, `make_all_figures.py` rebuilds every figure, and the plotted values in `data/` reproduce byte for byte
(checked on the release copy).

## Reproducing the numbers

Three levels, from cheapest to complete:
1. **Recorded outputs.** `review/*/out/` holds every table the papers quote, as written by the scripts beside it.
2. **Derived bundle.** `mode-series-derived.zip` (Hugging Face, see DATA.md) holds the telemetry cubes
   (`cache/F_*.pkl`, `F3_*.pkl`, `R5_*.pkl`), the parquet archive (`cache/pq/`), the v11 figure-data tables (`cache/fig/data/`)
   and the per-arm caches (`review/r3_unet/out/u0_cache.npz`, `review/r3_ace/out/cache_*.npz`, `review/standing_wave/sw_cache/`).
   Unzip it at the repository root. With it the analysis scripts run without the raw telemetry.
3. **Raw telemetry.** Place the files as DATA.md describes and run the loaders: `review/scripts/load.py` and
   `build_fields.py` (archive cubes, run from `cache/`), `review/round3/audit/scripts/r3_load.py` (R3 arms),
   `review/round5/scripts/r5_load.py` (R4 captures), `review/r3_unet/scripts/u0_load.py`, `review/r3_ace/scripts/a0_load.py`.

Round 2 scripts expect to be run with `cache/` as the working directory (they open `F_*.pkl` relative to it);
the other rounds' scripts run from anywhere. Checked on the release copy: the ACE-Step arm rerun from its raw CSVs
(`a0`–`a3`), Round 6 S1 and S4, the U-Net arm (u1, u2, u4, u5), every Round 8, 9 and 10 script (R8-7 from the raw archive, the language-model tests from the bundled parquet), reproduce their recorded outputs byte for byte.
One known difference: Round 2 was scored before the separately supplied Z-Image Base domain file was added, so its
recorded outputs use 428 Base generations on 105 prompts, while the current cubes hold 476 on 117. Rerunning Round 2
on the current cubes reproduces every other channel exactly and moves the Base rows slightly (for example the minimum
seed-corrected prompt share 0.357 → 0.353; plateau onsets unchanged). The papers quote the recorded Round 2 values.

## Preregistration and scoring

Each round's preregistration is stored with a SHA-256 file and a UTC freeze time; the scorecards are in Appendix B of
Mode-I and the appendix of Mode-II. Files under `review/` are kept as they were when scored, so their SHA-256 values still
match. Documents in `review/` record the paths of the original working environment; the scripts have been made
repository-relative.

## Requirements

Python 3.11 (tested), with the packages in `requirements.txt`. The capture node in `capture_code_r3/` runs inside ComfyUI
(tested against ComfyUI master `58b176f`) and needs its own environment (PyTorch).

## Citation

See `CITATION.cff`.
