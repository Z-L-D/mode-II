# Round 5 (2026-10-08): v0.16 audit, Round 4 section R, Round 5 preregistered calculations, ledger v0.17

Read in this order:
1. `CoS-PAFS_Findings_Ledger_v0.17.html` is the ledger. It keeps all of v0.16, adds v0.17 notes to the cards that changed, and adds new sections: Claude audit v0.16, Round 5, gates v0.17, R5 prereg.
2. `V016_AUDIT_AND_R_SCORING.md` covers the audit of the reviewer's v0.16 analysis, and Claude's scoring of Round 4 section R.
3. `R5_PREREGISTRATION.md` and `R5_PREREG_SHA256.txt` hold the 22 predictions, frozen 2026-10-08T05:24:22Z before any section Q calculation.
4. `R5_RESULTS_AND_SYNTHESIS.md` has the scorecard (13 confirmed, 9 refuted), the post hoc checks and the synthesis.
5. `R5_GATES.md` is the gate table v0.17 and the next capture session.
6. `LIT_R5.md` is the literature step (40 sources, 35 verified on a primary page).

**Scripts** (Python, `scripts/`). Run them in this order, with `SCR` set to a scratch folder:

    r5_load.py → r5_scoreR.py → r5_audit_v016.py → r5_q_calcs.py → r5_scoreQ.py → r5_posthoc.py → r5_vr_null.py → r5_figures.py → ledger_build/build_v017.py. `r5_shift_fit.py <csv ...>` checks the sampling shift from the logged σ values

They need the Round 3 field cubes (`round3/audit/scripts/r3_load.py`) and the Round 2 archive cubes (`F_*.pkl`, built by `round2/scripts`). Estimators are in `scripts/r5_common.py`, copied unchanged from Rounds 2–4.

**Outputs** (`out/`):
- `R5_intake.csv`, `R5_prompt_membership.txt`
- `R_section_scorecard.csv`, `R_section_results.json`
- `A_audit_v016_table.csv`, `A_audit_v016.json`, `A_topology_map_recomputed.csv`
- `Q_results.json`, `Q_scorecard.csv`
- `posthoc_R5.json`, `H1_seedfate_vs_sample_size.csv`, `VR_null.json`

**Figures** are in `fig/` (R5_fig1–6).

**Data:** the raw captures are in `data/r4_captures/`, one folder per arm. The reviewer's package is in `reviewer_v016/`.

**Tooling changed this round** (`capture_code_r2/`):
- `preflight_check.py` keys generations by (gen_idx, seed), reads headerless CSVs, reports NaN cells, and warns on a missing output-image column.
- `matched_prompts_30_v2.tsv` and `.txt` are clean copies of the frozen prompt list.
- `README_R2.md` has an R5 note.
- `nodes.py` is unchanged.

`ModeSeries_Round5_outputs.zip` bundles everything in this folder except the reviewer package, and adds the changed tooling files.
