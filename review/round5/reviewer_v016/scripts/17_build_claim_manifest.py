#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
BASE=Path('/mnt/data/r4_r5_audit_v016')
rows=[
('Archive intake / instrument + seed design','01_new_capture_intake.py','NEW_CAPTURE_INTAKE.csv; NEW_CAPTURE_INTAKE.json'),
('R1a/R1b/R1f/R1g/R1h branch-core score','scripts/02_R1_per_branch_core.py','results/R1_SCORECARD_AND_POSTHOC.json; results/R1b_cond_cfg45_vs_cfg1_step0.csv; results/R1f_cfg_spread_ratio_cells.csv; results/R1g_prompt_influence_by_step.csv; results/R1h_branchmean_vs_cond.csv'),
('R2 deepstep topology / 40-100 comparison','scripts/03_R2_deepstep.py','results/R2_DEEPSTEP_RESULTS.json; results/R2_beta_curve_*.csv; results/R2a_seed_bootstrap_topology.csv; results/R2b_same_prompt_sigma_match.csv'),
('Post-hoc seed-centred fate and VR','scripts/04_R1_seedcentred_posthoc.py','results/POSTHOC_R1_seedcentred_summary.csv; results/POSTHOC_R1_seedcentred_curves.csv'),
('Guidance/branch persistence decomposition','scripts/05_R1_branch_persistence_decomposition.py','results/POSTHOC_R1_branch_persistence_decomposition.csv'),
('Round-4 P1-P5 prereg audit','scripts/06_audit_round4_scorecard.py','results/ROUND4_SCORECARD_REAUDIT.csv; results/ROUND4_SCORECARD_REAUDIT.json'),
('ZIB CFG1/CFG4 branch audit','scripts/07_ZIB_branch_audit.py','results/POSTHOC_ZIB_cfg1_cfg4_cond*.csv; results/POSTHOC_ZIB_cond_uncond*.csv'),
('Deepstep seed overlap','scripts/08_deepstep_seed_overlap.py','results/DEEPSTEP_SEED_OVERLAP.csv'),
('Legacy vs passive 100-step pyramid audit','scripts/09_B7_matched_100step_instrument_audit.py','results/B7_MATCHED100_*.csv'),
('Legacy/passive sigma schedule comparison','scripts/10_B7_sigma_schedule_audit.py','results/B7_100STEP_SIGMA_GRID_*.csv'),
('Schedule reweight / passive 40-100 bootstrap','scripts/11_B7_schedule_reweight_and_bootstrap.py','results/B7_SIGMA_GRID_REWEIGHT_TOPOLOGY.csv; results/B7_PASSIVE_40_100_BOOTSTRAP.csv'),
('Current per-branch beta topology map','scripts/12_current_branch_topology_map.py','results/CURRENT_BRANCH_BETA_TOPOLOGY_MAP.csv'),
('Resolution topology prompt bootstrap','scripts/13_resolution_topology_bootstrap.py','results/CURRENT_RESOLUTION_TOPOLOGY_BOOTSTRAP.csv'),
('Prompt-influence / spectral / spread triangulation','scripts/14_branch_triangulation.py','results/BRANCH_TRIANGULATION_BY_STEP.csv; results/BRANCH_TRIANGULATION_CORRELATIONS.csv'),
('Guidance lag descriptive control','scripts/15_guidance_lag_descriptive.py','results/POSTHOC_GUIDANCE_LAG_CORRELATION.csv'),
('Prompt protocol and D14 linkage audit','scripts/16_protocol_and_output_linkage_audit.py','results/PROTOCOL_AND_D14_AUDIT.json'),
('R1/R2 formal audit scorecard','synthesis from scripts 01-16','results/R1_R2_AUDIT_SCORECARD.csv'),
('v0.16 gate review','synthesis from scripts 01-16','results/GATE_STATUS_V016.csv; results/GATE_STATUS_V016.md'),
('Figure F01','figure_scripts/F01_guidance_seed_contraction.py','figures/F01_guidance_seed_contraction.svg; figures/F01_guidance_seed_contraction.pdf; figures/F01_guidance_seed_contraction.png'),
('Figure F02','figure_scripts/F02_branch_persistence_decomposition.py','figures/F02_branch_persistence_decomposition.svg; figures/F02_branch_persistence_decomposition.pdf; figures/F02_branch_persistence_decomposition.png'),
('Figure F03','figure_scripts/F03_legacy_vs_passive_beta100.py','figures/F03_legacy_vs_passive_beta100.svg; figures/F03_legacy_vs_passive_beta100.pdf; figures/F03_legacy_vs_passive_beta100.png'),
('Figure F04','figure_scripts/F04_resolution_topology_bootstrap.py','figures/F04_resolution_topology_bootstrap.svg; figures/F04_resolution_topology_bootstrap.pdf; figures/F04_resolution_topology_bootstrap.png'),
]
pd.DataFrame(rows,columns=['claim_or_artifact','python_reproducer','outputs']).to_csv(BASE/'CLAIM_SCRIPT_MANIFEST_V016.csv',index=False)
