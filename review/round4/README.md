# Round 4 outputs (2026-10-06)

Read in this order:
1. `R3_AUDIT_OF_REVIEWER.md`: audit of the second reviewer's v0.13/v0.14 work.
2. `R4_PREREGISTRATION.md`: frozen before any section P calculation; hash and time in `R4_PREREG_SHA256.txt`.
3. `R4_RESULTS_AND_SYNTHESIS.md`: scorecard (16 rows: 11 confirmed, 5 refuted), post hoc checks, synthesis and paper consequences.
4. `R4_GATES.md`: gate table v0.15, plus the one capture session that clears the diffusion gates.
5. `LIT_R4.md`: literature check (26 of 32 sources verified on a primary page).
6. `CoS-PAFS_Findings_Ledger_v0.15.html`: the reviewer's v0.14 kept intact. v0.15 notes are appended to changed cards, and new sections cover the instrument fix, the Claude audit, Round 4, gates v0.15 and the R4 capture predictions. The v0.14 section, which sat after `</html>`, is moved inside the page.

Scripts (run from the scratchpad that holds the F_*.pkl archive fields):
- `../round3/audit/scripts/r3_load.py` builds the R3 field cubes.
- `../round3/audit/scripts/a1_reproduce.py` reproduces the reviewer's numbers.
- `scripts_p_calcs.py` runs section P as frozen; `scripts_posthoc.py` runs the post hoc checks; `scripts_figures.py` draws the figures.
- `ledger_build/build_v015.py` builds the ledger.

Raw outputs: `out/`, `../round3/audit/out/`. Figures: `fig/`.

Instrument fix: `capture_code_r2/` (`nodes.py`, `preflight_check.py`, `README_R2.md` §0, `tests/test_passive_r3.py`).
