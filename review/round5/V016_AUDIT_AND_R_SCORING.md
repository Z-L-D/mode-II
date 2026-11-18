# Audit of the reviewer's v0.16 analysis, and Claude's scoring of Round 4 section R

Audited 2026-10-08 by Claude against `review/round4/R4_PREREGISTRATION.md` (SHA-256 82ef210a…c79634) and the raw captures in `data/r4_captures/` (uploaded as `telemetry-10-7-26.zip`, SHA-256 0f820e1f…f4d1). The reviewer's package is in `reviewer_v016/`; their ledger v0.16 HTML hashes to db71ec7b…fd61, the same file as the separate ledger upload. Every number below was recomputed with Claude's own loader (`scripts/r5_load.py`) and the Round 2–4 estimators (`scripts/r5_common.py`), not with the reviewer's scripts. Scripts: `scripts/r5_scoreR.py`, `scripts/r5_audit_v016.py`. Outputs: `out/R_section_scorecard.csv`, `out/R_section_results.json`, `out/A_audit_v016_table.csv`.

## Verdict

The reviewer's audit is sound, and every number they report reproduces exactly. Four points need correcting or narrowing:

1. **Seed-fate σ50 values depend on the fold partition.** Their 0.631 is reproduced exactly with their row order. The same data and estimator give 0.484 with a different, equally valid prompt-to-fold assignment. Round 5 Q3 then shows that σ50 mostly measures sample size; see `R5_RESULTS_AND_SYNTHESIS.md`.
2. **The R1f definition changed.** v0.16 takes the log of the layer-mean g. The frozen B6 rule, and the reviewer's own v0.13 numbers, use the layer-mean of log g. Both are below 0.7, so the verdict stands, but the frozen value is **0.637**, not 0.575, and the second-quarter ratio is 0.90, not 0.77.
3. **R2 scoring.** v0.16 calls R2a and R2b NOT SCORABLE because 15 seeds were run instead of ≥ 30, yet scores R1b, R1f and R1g despite the 4/30 prompt substitution. Claude applies one rule to both cases:
   - A row is NOT SCORABLE when the missing design element changes what is estimated. Shared seeds make seed identity trivially decodable, so R1c–e are NOT SCORABLE.
   - A row is scored, with the deviation flagged, when the deviation only lowers precision. That covers the 15-seed R2 rows and the 26/30-prompt R1 rows.
   - On this rule **R2a is CONFIRMED** and **R2b is REFUTED as frozen**. See the scorecard below.
4. **prompt_influence is mis-described.** v0.16 calls it the "cosine of unconditional versus conditional noise predictions". The capture code (`nodes.py`, CSV writer, line ≈3513) computes the cosine between the token-mean hidden-state vectors of the two branches at each layer. It is a per-layer quantity, and its minimum (0.727) sits in the middle layers.

One minor intake-table error does not affect any analysis. The reviewer's `NEW_CAPTURE_INTAKE.csv` lists the 512 arm with 27 prompts and 57,599 rows, because one 512 CSV has no header row. Their analysis loader repairs that file, so every 512 result in v0.16 uses all 30 prompts.

## Reproduction table

| Reviewer claim | Recomputed | Status |
|---|---|---|
| R1b median \|log\| at step 0 = 0.000453 | 0.000453 (max 0.0043) | Exact |
| R1g min prompt_influence at σ ≤ 0.9 = 0.727 | 0.7272 | Exact (per-layer quantity) |
| R1h median log(g_BM/g_cond), steps ≥ 2 = 0.00077 | 0.00077 | Exact |
| R1f = 0.575; quarters 1.00/0.77/0.55/0.46 | 0.5746; 1.00/0.77/0.55/0.46 with the v0.16 definition | Reproduced. Frozen definition: **0.637**; 1.00/0.90/0.58/0.45 |
| VR(4): CFG 1 cond 1.85, CFG 4.5 cond 1.25, uncond 0.59, branch-mean 0.79 | 1.850, 1.250, 0.592, 0.787 | Exact |
| Seed-centred σ50: CFG 1 cond 0.631, CFG 4.5 cond 0.596, uncond 0.571 | Same with their row order. Claude's row order: 0.484, 0.571, 0.528 | Exact, but **partition-dependent** |
| R2 passive 100-step sequence +−, one crossing at 0.0593 | +− (0.05929) | Exact |
| 15/15 single seeds start '+', 0/15 are −+−+− | 15/15, 0/15 | Exact |
| 40→100 same-prompt median \|log g\| = 0.0326 | 0.0326 (0.0345 if mapped 100→40) | Exact |
| Topology map: 10 arm × functional cells | 10/10 sequences identical | Exact |
| Legacy and passive 100-step σ grids differ (max 0.1816) | 0.1816 | Exact |
| Legacy −+−+− survives regridding to the passive grid | −+−+− (nearest-σ thinning) | Reproduced |
| Round 4 P1–P5: 11 confirmed / 5 refuted | Not re-run here; it is Claude's own Round 4 output | Agreed |
| Guidance-lag correlations ρ −0.83 to −0.95 | Read, not recomputed | Flat across ±15 steps, so it carries no lead/lag information. Agreed it is descriptive only |
| D14: no output id or quality label | Confirmed. `preflight_check.py` now warns on it | Agreed |

## Intake facts (Claude's loader, `out/R5_intake.csv`)

- All 12 arms are `passive-branch`. Branches are {cond, uncond} at CFG > 1 and {cond} at CFG 1. There are no pooled rows, and prompt_influence is finite in every core CFG 4.5 generation.
- **Prompts.** The core arms ran 26 of the 30 frozen prompts. Missing: Japanese temple, Gothic cathedral, sea otter, geode. Added instead: brutalist parking structure, chameleon, antique instruments, Art Deco theater. The frozen `matched_prompts_30.tsv` stores the sea-otter prompt as a quoted field that starts with a newline, and the `.txt` copy has a blank line before it. Clean copies are now in `capture_code_r2/matched_prompts_30_v2.tsv` and `.txt`.
- **Every new prompt is one of the 60 Round 3 prompts, run with the same seeds 1001/1002.** This makes same-trajectory comparisons with Round 3 possible (Round 5 Q1).
- **Seeds.** 1001/1002 were reused across all prompts in the core. The pyramid arms use 13–17 random seeds.
- **Filters:**
  - One ZIB CFG 4 generation (one NaN cell) is dropped.
  - One CFG 4.5 40-step pyramid generation (one NaN g cell, two NaN prompt_influence rows) is dropped.
  - The CFG 4.5 40-step pyramid CSV holds two appended sessions that reuse gen_idx 3–9, so generations are keyed by (gen_idx, seed).
  - One 512 CSV has no header row and is read with the frozen schema.
- **Tooling.** `preflight_check.py` now handles all four of these cases (see `capture_code_r2/README_R2.md`, R5 note).

## Round 4 section R: Claude's scorecard

Rule for deviations: a row is scored when its decision rule can be computed on the delivered design and the deviation only reduces precision (the deviation is listed). It is NOT SCORABLE when the deviation changes the estimand.

| ID | Prediction | Verdict | Result | Design deviation |
|---|---|---|---|---|
| R1a | Preflight passes | **CONFIRMED** | passive-branch; {cond, uncond} at 4.5, {cond} at 1; 24 layers × 40 steps; finite prompt_influence | 26/30 prompts; shared seeds |
| R1b | cond at 4.5 = CFG 1 at step 0 (≤ 1e-3) | **CONFIRMED** | 0.000453 | same |
| R1c | Seed identity not decodable (unique seeds) | NOT SCORABLE | Post hoc: 0.92 accuracy (CFG 1) | Shared seeds change the estimand |
| R1d | σ50 in [0.85, 0.96] (unique seeds) | NOT SCORABLE | Post hoc seed-centred: 0.48–0.63 depending on partition | same; and Round 5 Q3 shows σ50 tracks n |
| R1e | VR(4) CFG 1 cond ≤ 1.0 (unique seeds) | NOT SCORABLE | Post hoc 1.85 (seed-centred 1.79). The unique-seed pyramid arm gives 1.95 (Round 5 Q4a) | same |
| R1f | Cond-branch spread ratio < 0.7 | **CONFIRMED** | 0.637 (prompt bootstrap 0.45–0.87) | 26/30 prompts |
| R1g | prompt_influence < 0.999 at σ ≤ 0.9 | **CONFIRMED** | 0.727 | 26/30 prompts |
| R1h | BM vs cond (report only) | REPORTED | median log(g_BM/g_cond) +0.0008 | — |
| R2a | 100-step passive: first '+', not −+−+−, ≤ 3 crossings | **CONFIRMED** | +−, one crossing. 15/15 seeds start '+'; 0/15 are −+−+−; seed bootstrap 100% +− | 15 of ≥ 30 seeds |
| R2b | \|log(g100/g40)\| ≤ 0.05 against Round 3 CFG 1 | **REFUTED** | 0.256 (Round 3 prompt-level yardstick 0.072). Same-prompt post hoc: 0.034 | 15 of ≥ 30 seeds |
| R3 | Instrument A/B | NOT SCORABLE | not run | — |
| R4 | G4 LLM / PSTF | NOT SCORABLE | no data | — |

Tally: 5 confirmed, 1 refuted, 5 not scorable, 1 report-only. No aggregate hit rate is given.

**R2b is a failure of the preregistration design, and Claude owns it.** The frozen rule compared the pyramid prompt against the 60 Round 3 prompts and set a 0.05 threshold. That threshold is below the prompt-to-prompt spread (median 0.072), and the pyramid prompt sits 0.26 log units below the Round 3 population. The rule could not separate step count from prompt. The same-prompt 40-step arm, which the user also captured, is the right control: 0.033, and Round 5 Q5 tests it against seed noise.

**R2a deviates from what Round 4 said to expect.** The R4 gates text expected +−+−. The decision rule did not require it, but the pyramid prompt gives +− at both 40 and 100 steps. The second crossing pair is prompt-dependent, as Round 5 Q6 shows.
