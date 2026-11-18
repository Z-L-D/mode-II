# Round 6 (2026-10-08): prepublication sweep, audit of the reviewer's Round 6, release claim register, ledger v0.19

Read in this order:
1. `R6C_RELEASE_CLAIM_REGISTER.md` is the drafting checklist for Mode-I and Mode-II. Every headline claim from the 2026-09-15 session summary and the ledger is marked USE, USE WITH WORDING or DO NOT USE, with evidence and release wording.
2. `CoS-PAFS_Findings_Ledger_v0.19.html` is the ledger. It keeps all of v0.18 (the reviewer's build of v0.17), moves the reviewer's Round 6 section back inside the page with a nav link, adds v0.19 notes to the cards that changed, and adds new sections: Release register, Round 6 (Claude), Claude audit v0.18, gates v0.19, R6 prereg.
3. `R6C_AUDIT_OF_REVIEWER.md` covers A1–A5: where every number in the reviewer's package comes from, their preregistration, estimator table, ledger and gate table.
4. `R6C_PREREGISTRATION.md` and `R6C_PREREG_SHA256.txt` hold predictions S1–S5 (22 rows), frozen 2026-10-08T07:06:28Z before any Round 6 calculation.
5. `R6C_RESULTS_AND_SYNTHESIS.md` has the scorecard (17 confirmed, 5 refuted), the post hoc checks P1–P3, "other ways to read this data", and the Mode-I / Mode-II synthesis.
6. `R6C_GATES.md` is the gate table v0.19. No capture is needed for the release as scoped.
7. `LIT_R6C.md` is the literature step (77 sources, 76 verified on a primary page).
8. `R6C_ANALYZER_REPLAY.md` (added 2026-10-08, post hoc) reads the telemetry with Zac's CoS-PAFS Live Analyzer v3, unchanged (`analyzer/cos_pafs_live_analyzer.py`). The layer-axis + − + − pattern in β_layer is cross-architecture. The analyzer's own figures for every set are in `analyzer_exports/`.

**Scripts** (Python, `scripts/`). Run them in this order, with `SCR` set to the scratch folder that holds the cubes:

    r6c_S1_hierarchy.py → r6c_S2_content.py → r6c_S3_estimators.py → r6c_S4_topology.py → r6c_S5_sync.py → r6c_posthoc.py → r6c_posthoc_P3.py → r6c_posthoc_P4_depth.py → r6c_posthoc_P4b_depth_robust.py → r6c_scorecard.py → r6c_figures.py → r6c_figure_P4.py → r6c_analyzer_replay.py → r6c_analyzer_beta_bootstrap.py → r6c_analyzer_beta_nulls.py → r6c_analyzer_figures.py → r6c_figure_A4.py → ledger_build/build_v019.py

They import `round5/scripts/r5_common.py` (estimators copied unchanged from Rounds 2–5) and need the Round 2 archive cubes (`F_*.pkl`), the Round 3 cubes (`F3_*.pkl`) and the Round 4 explicit-branch cubes (`R5_*.pkl`). `r6c_S2_content.py` takes about 2 minutes (2000-permutation Mantel tests).

**Outputs** (`out/`):
- S1 observable hierarchy: `S1_hierarchy.csv`, `S1_historical_reproduction.csv`, `S1_results.json`
- S2 content geometry: `S2_theme_residual.csv`, `S2_rsa.csv`, `S2_text_rsa.csv`, `S2_results.json`
- S3 estimators at matched n: `S3_estimators.csv`, `S3_results.json`
- S4 three-regime topology: `S4_topology.csv`, `S4_results.json`
- S5 synchronization nulls: `S5_sync.csv`, `S5_results.json`
- Post hoc: `posthoc_R6C.json` (P1 attention topology), `P2_rsa_robustness.csv`, `P3_order_perm.csv`, `P4_depth_topology.csv`, `P4_depth_per_sigma.csv`, `P4b_depth_robustness.csv` (three phases along the layer axis)
- Analyzer replay: `A1_analyzer_replay.csv`/`.json`, `A2_beta_layer_bootstrap.csv`, `A4_beta_layer_per_step.csv`, `A4_beta_layer_surrogates.csv`
- `R6C_scorecard.csv`

**Figures** are in `fig/` (R6C_fig1–9).

**Inputs kept for reference:** the reviewer's package is in `reviewer_r6/` (not in the zip), and `CoS_PAFS_ModeI_Session_Summary_2026-09-15.md` is a copy of the uploaded session summary.
