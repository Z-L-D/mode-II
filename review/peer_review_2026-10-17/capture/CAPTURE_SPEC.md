# Control capture: text rows and uncentred spectra (Round 11 draft)

Status: approved by Zac 2026-10-17. §6 frozen as the Round 11 preregistration in `PREREG_R11.md` (SHA-256 in `PREREG_R11.sha256`). Node: `capture_code_r4/`.

## 1. What it has to answer

The two reviewers' main technical concerns cannot be tested on the archive. The node computes g, gap and H on the whole uncentred matrix and stores only those three numbers.

| Question | Why it matters | Where in the paper |
|---|---|---|
| Q1. Do the Z-Image text rows drive the Base↔Turbo agreement? | Both Z-Image models put the same text-encoder rows into every hooked matrix: 64 of 4160 rows per branch (ZIT `seq` = 4160, ZIB pooled 8320). SD3.5 and Mochi hooks read image rows only. | §5.2 tangent 0.94–0.999, Table 2 margin +0.70, lineage RSA 0.56–0.62 |
| Q2. Do a few large mean or "massive" channels drive the statistics? | The second moment is uncentred (`XᵀX/N`, no mean subtracted). A large shared mean direction or a few extreme channels can set g, gap and H by themselves. | §3.1 definition; every result |

Q1 matters only for the lineage results. Every unrelated pair is already at |cos| ≤ 0.24, and text rows can only add shared structure, not remove it. Q2 matters for everything, but most for the tangent and the backbone.

## 2. Where the text rows are (checked in ComfyUI source)

ComfyUI `comfy/ldm/lumina/model.py` (NextDiT, current master), `patchify_and_embed`:

- The caption is embedded, padded with a learned `cap_pad_token` to a multiple of `pad_tokens_multiple` (32), and run through `context_refiner`.
- The image patches are embedded and padded with `x_pad_token` to a multiple of 32.
- The joint sequence is `padded_full_embed = torch.cat(feats + (x,), dim=1)`, so **caption rows come first and image rows last**.

At 1024 px the latent is 128×128 and the patch size is 2, so there are exactly 4096 image rows (no image padding). A prompt of 33–64 tokens gives 64 caption rows, and ≤32 tokens gives 32 (4 archive generations have `seq` = 4128). **The caption block includes learned pad rows, which are identical vectors.** That is one more reason the text block could weigh more in an uncentred spectrum than its 1.5% row share suggests.

The hooked tensors (`attention.out`, `feed_forward.w2`) have shape `[B, S, D]` with `S = cap_len + 4096`. Image rows are `X[:, -n_img:, :]`, where `n_img = ceil(H_lat/2)·ceil(W_lat/2)` rounded up to a multiple of 32. Caption rows are `X[:, :-n_img, :]`.

For SD3.5 (MMDiT) the hook already returns the image stream only, so the image-rows variant equals the all-rows variant. It is kept as a check.

## 3. Arms

All arms: **CFG 1** (cond-only, so there is no pooling confound), 1024 px, sampling shift 3.0, and the same sampler, scheduler, text encoders and step count as the archive campaign of that model.

Prompts: the **60 prompts of the Round 3 cond-only arms**, seed **1001**: `capture/prompts_r3_cond60.tsv` (domain, prompt, seed) and `.txt`. That is 6 per domain. The same 60 prompts and seed exist in Round 3 for ZIB CFG 1 (`data/r3_captures/c9ad092f`) and SD3.5 M CFG 1 (`90c87dde`), so the unchanged statistic can be checked for bit-identity (§5). All 60 prompts are in the ZIT archive.

| Arm | Model | Steps | Hooks | Gens | Role |
|---|---|---|---|---|---|
| A | Z-Image Base | 25 | attn + MLP | 60 | lineage, teacher-side sibling |
| B | Z-Image Turbo | 8 | attn + MLP | 60 | lineage, distilled sibling |
| C | SD3.5 Medium | 40 | attn (image stream) | 60 | unrelated control, so that every variant compares like with like |

Optional, only if A–C pass and you want CFG 4 coverage: arm D, ZIB CFG 4 with per-branch rows (the always-on split), 60 gens. It would connect the variants to the archive's guided regime.

Mochi is left out: one unrelated control is enough for the margin, and Mochi is the most expensive capture.

## 4. Node change: what is recorded

A new boolean input, `spectral_variants`, added **last** in `INPUT_TYPES` (saved workflows restore widgets by position; see README_R2). Default off, so existing workflows are unchanged. When on, each passive hook (`_make_measurement_hook`, `_make_mlp_measurement_hook` and the per-branch hook) computes the following from the same activation, per batch element / branch row.

| Variant | Rows | Matrix | Stats | Cost |
|---|---|---|---|---|
| V0 all (existing) | all `S` | `XᵀX/N` | `g_micro`, `spectral_gap`, `sv_entropy` (unchanged columns) | 1 eig of D×D |
| V1 image rows | last `n_img` | `XᵀX/N` | `g_img`, `gap_img`, `H_img` | 1 eig of D×D |
| V2 caption rows | first `S−n_img` | via the `n×n` Gram matrix (n ≤ 64) | `g_txt`, `gap_txt`, `H_txt` | tiny |
| V3 image rows, centred | last `n_img` | `(X−μ)ᵀ(X−μ)/N`, μ = mean over rows | `g_ctr`, `gap_ctr`, `H_ctr` | 1 eig of D×D |
| V4 image rows, massive channels removed | last `n_img` | `XᵀX/N` with the k = 8 channels of largest max\|x\| deleted (D−8 columns) | `g_topk`, `gap_topk`, `H_topk` (normalised by D−8) | 1 eig of (D−8)² |

Diagnostics per hook (cheap, from quantities already computed):
- `n_img_rows`, `n_txt_rows`;
- `txt_energy_share` = ‖X_txt‖²_F / ‖X‖²_F, the caption rows' share of the second moment;
- `mean_energy_share` = ‖μ‖² / tr(XᵀX/N) on image rows, the part of the uncentred moment that centring removes;
- `top1_chan_share` = max_j (XᵀX)_jj / tr(XᵀX) on image rows;
- `n_massive_100x` = number of channels whose max|x| exceeds 100× the median |x| (Sun et al. 2024 style);
- `topk_idx` = the 8 removed channel indices, semicolon-separated, so it can be checked whether they are the same channels across layers, steps and models.

The new columns are appended **after** the existing 21, so the loaders, the preflight and the analyzer keep working on V0.

Why k = 8 is fixed: a fixed k keeps D−k equal across cells, so g stays comparable. `n_massive_100x` shows whether 8 is too few or too many. If it is routinely above 8, V4 is reported as "partial removal".

Runtime: each hook does four D×D eigendecompositions instead of one. Keep them on the GPU in float32 (as now).

## 5. Gates before the long run

1. **Unit test** (mock, `tests/test_passive_r3.py` style). With `spectral_variants` off, the CSV is byte-identical to the current node. With it on, V0 equals the old columns. V1/V3/V4 match a numpy reference on a synthetic `[B,S,D]` tensor with known caption rows; V3 equals `np.cov(..., bias=True)`-based stats; the slice puts the caption first.
2. **E2E test**: `tests/test_r3_sampler_e2e.py` with the toggle on (hook-only tests missed CSV bugs before).
3. **Preflight on real hardware**: 1 prompt × 2 steps per model. Check `n_img_rows` = 4096 and `n_txt_rows` ∈ {32, 64} for Z-Image; `n_txt_rows` = 0 and V1 = V0 for SD3.5.
4. **Bit-identity**: arm A V0 must reproduce Round 3 `c9ad092f`, and arm C V0 must reproduce `90c87dde`, to 6 decimals in every cell (the Round 4 CFG 1 rerun did this before). If they don't, stop. Something other than the toggle changed.

## 6. Analyses and predictions (to freeze before the capture)

Methods are unchanged from the paper: Round 6 S1 (`review/round6/scripts/r6c_S1_hierarchy.py`) for shape and matched-σ tangent with its layer-permutation null; the depth-shift null for the backbone (§4); S4 for the clock-axis fit; the analyzer `plot_beta_layer` for the layer axis. Each statistic is run once per variant V0, V1, V3, V4 (V2 is descriptive only: 64 rows < D, so g is rank-bounded).

| ID | Test | Prediction (supports the paper) | Refuted if |
|---|---|---|---|
| R11-1 | Base↔Turbo tangent, attn and MLP, V1 | ≥ 0.95 in both, and within 0.03 of V0 | either < 0.95 |
| R11-2 | Base↔Turbo tangent, V3 and V4 | ≥ 0.90 in both submodules | either < 0.90 |
| R11-3 | Unrelated tangents (ZIB↔SD3.5M, ZIT↔SD3.5M attn), V1/V3/V4 | below that pair's layer-permutation p95 | any above its p95 |
| R11-4 | Lineage margin (lineage tangent − max unrelated), V3 and V4 | ≥ 0.50 | < 0.50 |
| R11-5 | Backbone PC1 per arm × submodule × variant | ≥ 0.70 and beats all 999 depth-shift nulls (ZIT: rank ≤ 7 caveat) | any fails |
| R11-6 | ZIB MLP clock-axis +−+ (S4 fit, 1000 prompt bootstraps), V1/V3/V4 | ≥ 90% of bootstraps | < 90% in any |
| R11-7 | Diagnostics | `txt_energy_share` < 0.10 in ≥ 90% of Z-Image cells; `mean_energy_share` reported per layer (no prediction) | share ≥ 0.10 in > 10% of cells |

Descriptive only (no prediction): the analyzer β-layer sign pattern per variant for ZIB attn/MLP and SD3.5 M attn; whether `topk_idx` is the same channel set across layers and between Base and Turbo.

## 7. What each outcome does to the paper

- **R11-1 to R11-4 pass:** one sentence in §5.2 and one in §3.1 reporting the image-row and centred tangent and margin. The Q1/Q2 scope sentences (added now) become "tested". The limitations item on massive activations is reduced to "other models".
- **R11-1 fails, R11-2 passes:** the lineage tangent is partly carried by shared caption rows. §5.2 and the abstract restate the lineage result for image rows with the new numbers. The margin in Table 2 gets an image-row column.
- **R11-2 or R11-4 fails:** the lineage separation is a property of the uncentred moment. The abstract and synthesis say so, and the hierarchy is reported for both moments. That is a real change to the headline, which is why it is worth running.
- **R11-5 or R11-6 fails** for a variant: that result is scoped to the uncentred statistic in the text.

## 8. What I need from you

1. Approve or change the arms (A–C, optional D) and the k = 8 choice.
2. Then I implement the node patch with the tests in §5, and freeze §6 as the Round 11 preregistration before you run anything.
