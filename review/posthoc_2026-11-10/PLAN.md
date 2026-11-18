# Post hoc follow-ups on existing data, defined before running (2026-11-10)

Zac asked (2026-11-10 11:45Z) for a sweep to find post hoc follow-ups that can be added to the papers while Round 18
is running. This file fixes what is computed and how each result is read **before any of it is computed**. Everything
here is post hoc: the data exist and were looked at for other purposes. Where a rule is listed it is the rule an earlier
preregistered round froze for the same statistic, replayed unchanged, in the way B1-B10 were replayed on ACE-Step Base.
No new capture is needed. Nothing here changes a frozen prediction of Rounds 1-18.

Seen before writing: the Round 17 tangents and gates (as in the papers); the shapes of the Round 17 cubes; and the
variant columns of PixArt-Sigma MLP at step 0 for four layers (last-layer `mean_energy_share` 0.9998, which prompted RC42).
Not seen: any backbone, decoding, RSA, handoff, envelope, crossing or centred-tangent value on the Round 17 captures.

Data: Round 17 cubes (60 prompts of `prompts_r3_cond60.tsv`, seed 1001, explicit branches, conditional branch unless
stated): PixArt-Sigma XL (28 blocks, attn + MLP, 20 steps), Cosmos-Predict2 2B (28, attn + MLP, 35 steps), SD3 Medium
(24, attn, 28 steps), SD3.5 Medium (24, attn, 40 steps, CFG 4.5); their variant cubes `V_*` (centred, top-8 channels
removed, mean-direction energy share); the Round 11 cubes (Z2, ZITSV, SD3SV) and the Round 12 cubes (Z1, Z1U, Z3); the
archive cubes. Clock: t = sigma / (1 + sigma) for PixArt-Sigma and Cosmos (as `score_r17.py`), sigma otherwise.

## A. Atlas results replayed on the Round 17 models (never analysed for these)

| ID | Statistic (code reused unchanged) | Rule replayed (source) |
|---|---|---|
| RC34 | Backbone: PC1 share and entropy effective rank against 999 per-slice depth-shift nulls (`backbone()` of `posthoc_r14.py`) | PC1 >= 0.70 and beats all 999 nulls (R11-5) |
| RC35a | Triple two-dimensional: PC1+PC2 of (H/ln D, ln(gD)/ln D, ln gap) (`pca_shares`) | >= 0.98 (B3) |
| RC35b | Anisotropy: g / g_MP(N, D) per (step, layer) cell, median over generations | < 0.10 in >= 95% of cells (B2/C1) |
| RC35c | Depth step not a function of g: held-out dR2 of a layer effect over a spline in g (`r1`) | >= 0.30 (B4) |
| RC36 | Plateau onset, 2x rule (B8 code) on each model's clock; also Z1 and Z3 | inside [0.73, 0.92] (B8) |
| RC37 | Clock-axis envelope: three-segment fit of layer-mean g, 1000 prompt bootstraps (`fit3`, `clock`) | attention: +,+,- in >= 90% of bootstraps (the archive pattern); MLP: +,-,+ in < 10% (it is a Z-Image Turbo caption-row property) |
| RC38 | Layer-axis crossings of the analyzer's binned beta (`L2.curve`), 100 prompt bootstraps; all statistics, centred and top-8 removed | descriptive: modal crossing count and its share; share with three crossings |
| RC39 | First-state ten-way domain decoding (decoder of `fig_I_F05_semantic.py`): ordered profile, layer mean, prompt length, TF-IDF, 99 depth scrambles; conditional and unconditional branch | conditional branch beats all 99 scrambles (p = 0.01); unconditional branch at the first state does not (built-in null) |
| RC40 | Shared prompt geometry: Mantel RSA (S2 features, 2000 permutations) among the seven single-seed 60-prompt captures (PixArt, Cosmos, SD3 M, SD3.5 M R17, SD3.5 M R11, Z-Image Base Z2, Z-Image Turbo R11) at each model's first, middle and last state; partial Mantel on TF-IDF text distance and on length | every pair of different models: rho > 0 with p <= 0.01 at all three states (S2); same weights (SD3.5 M in two sessions) highest in its row |
| RC41 | Text-to-latent handoff: RSA between conditional and unconditional branch per step (`score_r15.py` R15-1 code) | per channel: step 0 <= 0.25 (or unconditional sd ~ 0), >= 0.50 at step ceil(0.10 T), >= 0.80 at the last step (R15-1); 6 channels |

## B. Centred and channel-pruned statistics on the Round 17 captures (variant cubes never analysed)

| ID | Statistic | Reading fixed now |
|---|---|---|
| RC42 | Depth tangent on the native grid (RC33 code: 1000-draw permutation p95, 300-draw surrogate p95) on all rows, centred and top-8-removed statistics for: SD3 M <-> SD3.5 M (successor), SD3.5 M R17 <-> R11 (same weights, two sessions), PixArt <-> Cosmos attn and MLP (unrelated), conditional <-> unconditional of each model (same run) | Split robust to centring if every related pair stays >= permutation p95 + 0.30 and the unrelated attention pair stays below p95 + 0.30. MLP bottom tier: "restored on centred statistics" if PixArt <-> Cosmos MLP centred is below its permutation p95; "not restored" if >= p95 + 0.30; otherwise "reduced, not restored" |
| RC43 | Mean-direction energy share, top-1 channel share and count of channels > 100x median, by layer, per model and submodule (variant columns) | descriptive; in particular the last MLP layer |
| RC44 | Identity gate (step-0 depth-profile r of log g) on row sets and variants: Z2 <-> ZITSV on image rows, caption rows, centred, top-8 removed, with prompt-split halves of each as same-weights reference; SD3 M <-> SD3.5 M, SD3.5 M R17 <-> R11 and PixArt <-> Cosmos on centred and top-8 removed | descriptive against the gate's bands (>= 0.98 same; 0.85-0.98 undecided; < 0.85 different). Closes "identity gate: not recomputed on row sets" (Mode-I Table 6) |
| RC45 | First-state domain decoding by row set for Z2 and ZITSV (image, caption, centred, top-8 removed), same decoder and scrambles | descriptive; closes "first-state domain decoding: not recomputed on row sets" |

## C. Gaps the papers flag as not computed

| ID | Statistic | Reading |
|---|---|---|
| RC46 | Decoding table rows for the real Base (Z1 cond and uncond, attn and MLP; all five columns) | descriptive; replaces "decoding was not recomputed on the real Base captures" |
| RC47 | Successor pair by depth: SD3 M <-> SD3.5 M native tangent on the first 12 depth intervals (blocks 0-12, where MMDiT-X adds self-attention) and on the last 11, each with its own permutation null; the same halves for SD3.5 M across sessions | descriptive: is the successor's shortfall from same weights concentrated in the blocks that changed |
| RC48 | Multiple regression on distance matrices for the archive prompt RSA (the check Fig. 6 lists as open): rank distances of model A regressed on model B, TF-IDF text distance and length distance; 2000 permutations of B's prompt labels; 10 pairs x 3 states | partial coefficient of B positive with p <= 0.01 in every unrelated cell = the partial-Mantel result stands |
| RC49 | Depth tangent by sampling state for every depth-matched pair (native grid): per-state values and the share of states at which each related pair exceeds the largest unrelated value and its own permutation p95 | descriptive: does the split hold along the whole clock |
| RC50 | Seed predictability on the real Base (Z1 cond; Z3): relative onset at matched n = 60 prompts (50 subsets) and VR(4)/walk, Round 6 S3 code | descriptive rows for Mode-I Table 9 (if the S3 code runs unchanged on these cubes) |
| RC51 | State alignment against derivative alignment (D10) on depth-matched pairs including the real lineage pair, re-implemented (Procrustes and linear maps on state landmarks) | descriptive; replaces "has not been repeated on the real lineage pair" (if a faithful re-implementation is possible) |
| RC52 | Theme-residual alignment (Fig. S1 method) for Base <-> Turbo | descriptive (if the Fig. S1 code runs on the Base cubes) |

Outputs go to `out/`; `RESULTS.md` reports every row whatever it shows. Results enter the papers only after Zac agrees.
