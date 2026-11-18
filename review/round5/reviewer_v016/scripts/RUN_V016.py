#!/usr/bin/env python3
"""Replay the v0.16 calculations in dependency order.

Requires the original input archives mounted under /mnt/data and the supplied
Round-4 package extracted to /mnt/data/r4_pkg/round4.
"""
from pathlib import Path
import subprocess,sys
BASE=Path('/mnt/data/r4_r5_audit_v016')
order=[
    BASE/'01_new_capture_intake.py',
    BASE/'scripts/02_R1_per_branch_core.py',
    BASE/'scripts/03_R2_deepstep.py',
    BASE/'scripts/04_R1_seedcentred_posthoc.py',
    BASE/'scripts/05_R1_branch_persistence_decomposition.py',
    BASE/'scripts/06_audit_round4_scorecard.py',
    BASE/'scripts/07_ZIB_branch_audit.py',
    BASE/'scripts/08_deepstep_seed_overlap.py',
    BASE/'scripts/09_B7_matched_100step_instrument_audit.py',
    BASE/'scripts/10_B7_sigma_schedule_audit.py',
    BASE/'scripts/11_B7_schedule_reweight_and_bootstrap.py',
    BASE/'scripts/12_current_branch_topology_map.py',
    BASE/'scripts/13_resolution_topology_bootstrap.py',
    BASE/'scripts/14_branch_triangulation.py',
    BASE/'scripts/15_guidance_lag_descriptive.py',
    BASE/'scripts/16_protocol_and_output_linkage_audit.py',
    BASE/'scripts/17_build_claim_manifest.py',
]
for p in order:
    print(f'\n=== {p.name} ===',flush=True)
    subprocess.run([sys.executable,str(p)],check=True)
for p in sorted((BASE/'figure_scripts').glob('*.py')):
    print(f'\n=== {p.name} ===',flush=True)
    subprocess.run([sys.executable,str(p)],check=True)
print('\nLedger update is intentionally excluded from automatic replay; run 18_update_ledger_v016.py only after reviewing all results.')
