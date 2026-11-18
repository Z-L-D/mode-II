# Round 6 addendum: the telemetry read with the CoS-PAFS Live Analyzer v3, unchanged (A1–A4)

Zac asked on 2026-10-08 (08:13Z) for the data to be analyzed exactly as his analyzer does it, and for past discussions to be excavated first. All of this is post hoc. Nothing below was preregistered.

## How it was run

- **The analyzer itself, unchanged.** `analyzer/cos_pafs_live_analyzer.py` is a byte-for-byte copy of the uploaded file (SHA-256 4089062551a3c900…41f9). The scripts import it and call its functions. The one change is that `gradio` is replaced by an empty module, because only `make_app()` uses it.
- **Loading.** Data go through the analyzer's `DataManager.load_paths`, which runs `pd.read_csv` per file and then `_deduplicate_gen_idx`, the same as the dashboard's Load Paths button. The inputs are the raw per-domain CSVs from TELEMETRY.zip and the Round 4 captures.
- **Reading the results.** Crossings and turning layers are read from what the analyzer returns or draws: figure titles, and its marker lines.

**Scripts:**
- `scripts/r6c_analyzer_replay.py` → `out/A1_analyzer_replay.csv`, `.json`
- `scripts/r6c_analyzer_beta_bootstrap.py` → `out/A2_beta_layer_bootstrap.csv`
- `scripts/r6c_analyzer_beta_nulls.py` → `out/A4_beta_layer_per_step.csv`, `out/A4_beta_layer_surrogates.csv`
- `scripts/r6c_analyzer_figures.py` → `analyzer_exports/` (the analyzer's own depth-profile, β_layer and spiral figures for every set)
- `scripts/r6c_figure_A4.py` → `fig/R6C_fig9_analyzer_beta_per_step.png`

## What past discussions say the layer-axis three phases are

- **S246 ledger (Drive, 2026-09-29, 01:16–01:55).** The CoS-PAFS sign pattern is "+ > − > + > −" in β = g(L+1) − g(L), the layer-direction flow. Its three zero crossings separate the three stages (explore → converge → consolidate). Zac's correction at 01:20 adds that the fourth sign is the resting frame of the next cycle (ECC → ECC → ECC). The Qwen2.5-7B layer crossings were 0.0086 / 0.0143 / 0.0241.
- **09-15 Mode-I summary §7–10.** Pooled-cloud crossings are not fixed points in the RG sense:
  - one crossing is null-generic under Surrogate A;
  - in the ZIT 100-step run, the three crossings come back under Surrogate B, which keeps the mean field g(L, σ) ("the temporal E→C→C profile folded into (g, β)").
- **Round 1 carry-forward (CF8, CF13).** The old "ratio 2" crossing ladder does not replicate, and the depth profile is not stationary across steps.

These definitions are what is tested below. The Round 6 S4 and P4 fits were piecewise-linear fits of g along σ (S4) or along layer index (P4). They are not the analyzer's measurement, and the analyzer reading supersedes P4.

## A1. What each analyzer view gives on diffusion telemetry

- **The v3 three-crossing tools return nothing on diffusion data.** These are the mean-profile β(g) (`_get_mean_profile_crossings`), Key Results, the interval decomposition (Δ₁/span against the 46% prediction) and the bootstrap g₃/g₁ ratio. They keep only rows with `token_type == "generated"`. Every diffusion row is `token_type == "step"`, so on every diffusion set these views report 0 crossings or "insufficient data". They were built for the LLM corpus.
  - When the same function runs on the frame without its `token_type` column, the spline (s = 5e-5) is far too loose for diffusion g (about 10⁻³–10⁻²). SciPy reports "s too small" and returns 7–37 crossings. That is not a usable reading.
- **The dashboard β_layer tab (`plot_beta_layer`) is the analyzer's layer-axis three-phase measurement on diffusion data.** It bins every (generation, step, layer) point of β = g(L+1) − g(L) against g into 29 bins, needs at least 10 points per bin, and reports the zero crossings. The three-phase pattern is + − + −, with crossings pos→neg, neg→pos, pos→neg.
- **Most views pool the two submodules.** When a file has both attn and mlp, every view except the v3 ones averages them per (layer, step) cell. This applies to Z-Image, where pooled ZIT and ZIB show one crossing.
- **Two loading behaviours to know about:**
  - The Mochi CFG 4.5 `portraits_figures` CSV has no header row, so `read_csv` drops its 73,152 rows without a warning. The Mochi CFG 4.5 numbers here are therefore 9 of 10 domains.
  - The ZIT archive includes the 4 auxiliary 100-step generations. The step-axis views (`detect_phase_boundaries`, Step Trajectories) therefore run to step 99 on those 4 generations.

## A2. The β_layer pattern: generation bootstrap (100 per set, analyzer function)

| Set | Submodule | Gens | Observed | + − + − in bootstraps | 1 crossing | ≥ 5 |
|---|---|---|---|---|---|---|
| SD3.5 M archive (CFG 4.5) | attn | 481 | + − + − | **100%** | 0% | 0% |
| SD3.5 L archive (CFG 3.5) | attn | 486 | + − + − | **100%** | 0% | 0% |
| ZIT archive | mlp | 486 | + − + − | **90%** | 0% | 10% |
| ZIB archive (pooled CFG 4) | mlp | 482 | + − + − | **84%** | 16% | 0% |
| ZIB R4 CFG 1 cond | mlp | 20 | + − + − | 74% | 18% | 7% |
| ZIB archive (pooled CFG 4) | attn | 482 | + − + − | 67% | 32% | 1% |
| Mochi archive CFG 1 (100 steps) | attn | 60 | 5 crossings | 55% | 27% | 18% |
| ZIB R4 CFG 1 cond | attn | 20 | + − | 38% | 56% | 6% |
| ZIB R4 CFG 4 cond branch | attn | 20 | + − | 37% | 28% | 30% |
| Mochi archive CFG 4.5 (9 domains as loaded) | attn | 430 | + − | 19% | 81% | 0% |
| ZIT archive | attn+mlp pooled | 486 | + − | 17% | 83% | 0% |
| ZIT archive | attn | 486 | + − | 5% | 95% | 0% |
| ZIB R4 CFG 4 cond branch | mlp | 20 | + − | 4% | 96% | 0% |
| ZIB archive | attn+mlp pooled | 482 | + − | 1% | 99% | 0% |

## A3. Fixed σ: the layer axis alone (analyzer β_layer on one step at a time)

Fig. 9 (`fig/R6C_fig9_analyzer_beta_per_step.png`) shows the crossing pattern at every step.

| Channel | Steps with + − + − | Where |
|---|---|---|
| SD3.5 L attn | 24 of 28 (86%) | σ 0.96 → 0.19 |
| Mochi attn (CFG 4.5) | 18 of 30 (60%) | σ 1.00–0.98, 0.92 → 0.72 and 0.40 → 0.17; 1 crossing at σ 0.69–0.48 |
| ZIT MLP | 3 of 8 | σ 1.00–0.90 |
| ZIB MLP | 9 of 25 | σ 0.84–0.79 and 0.59–0.29 |
| SD3.5 M attn | 13 of 40 (33%) | σ 1.00 → 0.86 (early only); 1 crossing below |
| ZIB attn | 8 of 25 | early (σ 1.00–0.90), 0.82, 0.76, 0.54 |
| ZIT attn | 2 of 8 | σ 0.90 and 0.65 |

At fixed σ, the three-crossing layer-flow pattern appears in **every architecture** at some stage of denoising. Where it appears depends on the model and on σ.

## A4. Nulls (09-15 Surrogates A and B, 30 draws each, analyzer function)

| Channel | Observed (all steps) | Surrogate A reproduces it | Surrogate B reproduces it | Reading |
|---|---|---|---|---|
| SD3.5 M attn | + − + − | 0 of 30 (gives + −) | 0 of 30 | Needs within-generation layer coupling. Beats both nulls |
| ZIB MLP | + − + − | 0 of 30 | 0 of 30 | Beats both nulls |
| ZIB attn | + − + − | 0 of 30 | 0 of 30 | Beats both nulls |
| ZIT MLP | + − + − | 0 of 30 | 30 of 30 | Comes from the mean field g(L, σ), as in 09-15 §9. Beats Surrogate A |
| SD3.5 L attn | + − + − | 30 of 30 | 30 of 30 | **Null-generic**: per-layer marginals alone produce it |
| Mochi attn (CFG 4.5) | + − | 30 of 30 | 7 of 30 (B gives + − + − in 23 of 30) | One crossing is null-generic |
| ZIT attn | + − | 30 of 30 | 30 of 30 | One crossing is null-generic |

**Size of the middle segment.** The pattern is stable under resampling, but in some channels the middle segment is small. The table gives the analyzer's own interval split, Δ₁/span = (g*₂ − g*₁)/(g*₃ − g*₁), with the 46% prediction for reference. The 09-15 audit (§10) withdrew crossing ratios as invariants, so these numbers are descriptive only.

| Channel | g*₁ / g*₂ / g*₃ | Δ₁/span |
|---|---|---|
| SD3.5 M attn | 0.0178 / 0.0196 / 0.0765 | 3% (a narrow dip near the low-g edge) |
| ZIB attn | 0.0591 / 0.0621 / 0.0794 | 15% |
| SD3.5 L attn | 0.0230 / 0.0426 / 0.0735 | 39% |
| ZIT MLP | 0.0086 / 0.0150 / 0.0220 | 48% |
| ZIB MLP | 0.0136 / 0.0176 / 0.0188 | 77% |

In the analyzer's β_layer figures (`analyzer_exports/*__beta_layer.png`), the middle excursion of the binned mean is small compared with the spread of the cloud. In ZIT MLP, for example, the middle positive bump is about +0.001 against a late descent to −0.02. The robust statement is therefore the sign pattern and its bootstrap rate, not the interval sizes.

## What this changes

1. **E→C→C along the layer axis is not a Z-Image MLP result.** In the analyzer's own measurement, the + − + − layer flow is strongest in SD3.5 M and SD3.5 L attention (100% of bootstraps). It is also present in Z-Image MLP (84–90%) and ZIB attention (67%). At fixed σ it appears in all four architectures, Mochi included. Beyond both 09-15 surrogates, it holds in SD3.5 M attention and ZIB attention and MLP. Against Surrogate A, it also holds in ZIT MLP.
2. **What stays scoped:** the σ-axis curve (S4, layer-mean g expand → contract → re-expand in time) is still Z-Image MLP only. Attention layer-mean g rises and then falls in all five models.
3. **Corrections to Round 6 text:**
   - P4's statement "attention has no shared depth shape" is withdrawn. It used a piecewise-linear fit to the time-averaged depth profile, which is not the analyzer's measurement.
   - Register row 1 and the gate "E→C→C scope" are revised.
4. **Instrument facts that belong in Mode-I methods:**
   - The v3 mean-profile, interval and ratio tools are LLM-only as written.
   - The dashboard pools attn and mlp.
   - The pooled run-level crossing count depends on capture (pooled vs explicit branch), submodule, step mixing and sample size (20 vs 480 generations).
   - SD3.5 L's pattern is null-generic.
   - Report the pattern per σ, or with its bootstrap rate and surrogate result. Do not report a single crossing count.
