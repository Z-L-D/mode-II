"""Rebuild every Mode-II figure (PDF + PNG in ../figures) and its plotted data (../data/<figure>.csv).

Usage:  python3 make_all_figures.py            # all figures
        python3 make_all_figures.py F2 F9      # a subset, by id suffix
Raw-cube figures (II-F1, F8, F10, S1) need the telemetry cubes F_*.pkl / R5_*.pkl; point CUBES=<dir> at them.
Other inputs: /mnt/project-files/review/round{2,5,6}/out, papers/_sources/ledger_v020_text.txt (read only).
"""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    ('II-F1', 'fig_II_F1_observable_stack.py'),
    ('II-F2', 'fig_II_F2_pooling_theorem.py'),
    ('II-F3', 'fig_II_F3_projection_degeneracy.py'),
    ('II-F4', 'fig_II_F4_closure_ladder.py'),
    ('II-F5', 'fig_II_F5_trajectory_decomposition.py'),
    ('II-F6', 'fig_II_F6_tangent_inheritance.py'),
    ('II-F7', 'fig_II_F7_content_vs_dynamics.py'),
    ('II-F8', 'fig_II_F8_two_axis_topology.py'),
    ('II-F9', 'fig_II_F9_differencing_nulls.py'),
    ('II-F10', 'fig_II_F10_local_mode_census.py'),
    ('II-F11', 'fig_II_F11_estimators.py'),
    ('II-F12', 'fig_II_F12_claim_boundary.py'),
    ('II-F13', 'fig_II_F13_lineage_controls.py'),
    ('II-S1', 'fig_II_S1_intrinsic_progress.py'),
    ('II-S2', 'fig_II_S2_legacy_rebound.py'),
]

if __name__ == '__main__':
    want = set(sys.argv[1:])
    failed = []
    for fid, script in SCRIPTS:
        if want and fid.split('-')[1] not in want and fid not in want:
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=HERE, capture_output=True, text=True)
        status = 'ok' if r.returncode == 0 else 'FAILED'
        print(f'{fid:7s} {status:6s} {time.time() - t0:5.1f}s  {script}')
        if r.returncode != 0:
            failed.append(fid); print(r.stderr[-2000:])
    sys.exit(1 if failed else 0)
