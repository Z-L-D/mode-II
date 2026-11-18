# Audit of the reviewer's Round 6 package (A1–A5)

This audit is by Claude, dated 2026-10-08. It covers `reviewer_r6/mode12_round6/`, which is the extracted `ModeSeries_Round6_Prepublication_Sweep_2026-10-08.zip` (SHA-256 4278e19e…8bb2). The A6 claim register is in `R6C_RELEASE_CLAIM_REGISTER.md`.

The reviewer did not have the raw archive, so their package carries the historical numbers forward without recomputing them. Claude recomputed everything from the archive cubes with `scripts/r6c_S1_hierarchy.py` and `r6c_S2_content.py`.

## Verdict

The reviewer's reading holds: lineage is invisible at the coarse level and clear at the tangent level. Two points change the story they tell.

1. **The "Mochi/ZIB anomaly" comes from the capture, not the model.** Every historical number reproduces exactly from raw rows: ZIB↔Mochi 0.9405, ZIB↔ZIT 0.9325, ZIT↔Mochi 0.8217, and MLP ZIB↔ZIT 0.9715. The statistic is the Pearson r of the layer-mean g curves. Three things each reverse the ordering:
   - **The ZIB capture.** The archive ZIB pools both CFG branches. With the explicit cond branch, ZIB↔ZIT is 0.989 and ZIB↔Mochi is 0.831. The ordering margin moves from −0.008 to +0.157.
   - **The scale.** On log g the archive itself orders the lineage first: 0.973 against 0.860.
   - **The prompt set.** Restricted to the 19 Round 4 prompts, the archive gives +0.008 instead of −0.008.

   The ordering is therefore not "real at the level originally measured" in any useful sense. It is one draw from a quantity that capture, scale and prompts each flip.

2. **R6-3 fails on raw data.** The lineage theme residual passes, at 0.678 / 0.750 / 0.626. The unrelated pairs are not near zero: ZIT↔SD35M reaches 0.479 at the middle step and 0.517 at the last, and four of the eight pairs exceed 0.20 at two of the three time points. The reviewer's figures (0.611/0.794/0.649 and 0.119, cited as "publication v0.5") do not appear in any file in the project.

## A1. Where each number in `R6_METRIC_HIERARCHY.csv` comes from

| Their value | Their source | Actual source | Recomputed from raw (Round 6) |
|---|---|---|---|
| ZIB-Mochi attn 0.9405 | "recovered prior analysis" | 09-15 summary §16 | **0.9405** (scalar envelope, layer-mean g) |
| ZIB-ZIT attn 0.9325 | same | same | **0.9325** |
| ZIT-Mochi attn 0.822 | same | same | **0.8217** |
| ZIB-ZIT MLP 0.9715 | same | same | **0.9715** |
| Shape ZIB-ZIT ≥ 0.972 | "v0.17 ledger" | Round 1 review §2: 0.972–0.990 with an *optimized warp* | 0.981 at matched σ, no warp |
| Unrelated shape ≤ 0.83 | "v0.17 ledger" | Round 1 review §2 / §3.5 | 0.829 (ZIB↔Mochi) |
| Tangent 0.943 / 0.968 | "v0.17 ledger" | Round 2 E10 | 0.942 (attn; same estimator) |
| Unrelated tangent ≤ 0.24 | "v0.17 ledger" | Round 2 E10 (0.241) | 0.240 |
| Theme residual 0.611 / 0.794 / 0.649 | "publication v0.5" | **not found in the project** | 0.678 / 0.750 / 0.626 (S2, Claude's definition) |
| SD35M-Mochi early 0.119 | "publication v0.5" | **not found** | 0.215 at the first step (p 0.004); unrelated pairs reach 0.517 |

The tangent and shape values are on the archive's pooled ZIB. On explicit branches the lineage tangent is 0.992, from Round 5 Q2.

## A2. Integrity of their preregistration

- The SHA-256 of `R6_PREREGISTRATION.md` matches their file (93e876a5…2fd02968).
- Their R6-1 and R6-4 decision rules were evaluated only on numbers that already existed when they froze the file. R6-3 used values from a file Claude cannot find. These rows are consistency checks, not predictions, and the ledger labels them that way.

Here are the rows re-scored on their own rules, with Claude's recomputed inputs:

| Row | Their verdict | Re-scored | Note |
|---|---|---|---|
| R6-1 | PASS | **PASS (consistency check)** | 0.9405 ≥ 0.9325 reproduces; tangent margin 0.703 ≥ 0.50. The *reading* is incomplete: see S1d |
| R6-2 | SUPPORTED | **SUPPORTED, sharpened** | Every attention envelope rises through denoising in all five models (post hoc P1), so any two envelopes correlate. That is the mechanism of the projection degeneracy |
| R6-3 | PASS | **FAIL** | Lineage arm passes (0.678 / 0.750 / 0.626). Unrelated pairs are not ≤ 0.20: max 0.517 |
| R6-4 | PASS | **PASS** | SHA 78b250a2…7018 and 13 / 9 confirmed |
| R6-5 | PASS | **Agreed, with additions** | See `R6C_GATES.md` |
| R6-6 | — | **Order confirmed** | Margins at each level: −0.05, +0.05, +0.13, +0.15, +0.70 (S1c). The content axis is only weakly related to the dynamics axis (ρ 0.30, S2g) |

## A3. `R6_ESTIMATOR_CALIBRATION.csv`

The values are copied correctly from Round 5 `H1` and `VR_null.json`. There are three labelling slips:
- The column for runs per prompt was dropped. As a result, two rows read "archive SD35M CFG4.5 BM (60 prompts) centred=False" with different values; they are 2 and 4 runs per prompt.
- The VR reference columns are 2.5th and 97.5th percentiles but are labelled p5/p95.
- The core-30 rows and the CFG 4.5 branch rows are left out.

Section S3 below replaces this table with a full re-statement of all seven archive channels.

## A4. Their ledger v0.18

- It is built on Claude's final v0.17, and it carries the shift and rider edits.
- Their section is appended after `</footer>` and has no navigation link.
- Their cards cite "registered shape ≈ 0.972–0.990" without saying this is the optimized-warp range.

Ledger v0.19 keeps their section intact, adds a nav link to it, and annotates their four cards.

## A5. Their gate table

It agrees with v0.17 on every row they list. They leave out four rows: capture preflight, prompt lock, seed design and the Z-Image shift (which is resolved). There is no conflict to resolve. The gate determination for v0.19 is in `R6C_GATES.md`.
