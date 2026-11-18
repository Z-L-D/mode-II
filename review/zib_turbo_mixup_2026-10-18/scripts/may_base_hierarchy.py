import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# EXPLORATORY (not preregistered): Zac's May 2026 Z-Image Base workflow-test run (120 prompts x 1 gen, 100 steps,
# CFG 4 pooled) against every archive channel, with the Round 6 S1 estimators unchanged (r6c_common / r5_common).
# BMAY = May Base. ZIT = Turbo 8 steps CFG 1. ZIB = archive labelled Base, actually Turbo 25 steps CFG 4 pooled.
# Layer-permutation null: B's layer order shuffled (same permutation at every sigma), 200 draws, p95 reported.
# Usage: SCR=<cubes cache with F_BMAY_*.pkl> python may_base_hierarchy.py
import os, sys
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round6/scripts')
from r6c_common import *
OUTD = f'{_COSPAFS_ROOT}/review/zib_turbo_mixup_2026-10-18/out'
M = ['BMAY', 'ZIT', 'ZIB', 'MOCHI', 'SD35M', 'SD35L']
rng = np.random.default_rng(0)
rows = []
for sub in ('attn', 'mlp'):
    ms = [m for m in M if sub == 'attn' or m in ('BMAY', 'ZIT', 'ZIB')]
    D = {m: arch(m, sub) for m in ms}
    E1 = {m: D[m][2][..., 0].mean((0, 2)) for m in ms}; E2 = {m: np.log(D[m][2][..., 0]).mean((0, 2)) for m in ms}
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            _, _, XA, sa = D[a]; _, _, XB, sb = D[b]
            tc, sr = tangent(XA, sa, XB, sb)
            null = []
            for _ in range(200):
                p = rng.permutation(XB.shape[2]); null.append(tangent(XA, sa, XB[:, :, p], sb)[0])
            rows.append(dict(sub=sub, a=a, b=b, L0_D1=env_r(E1[a], sa, E1[b], sb), L0_D2=env_r(E2[a], sa, E2[b], sb),
                             L1_field=field_r(XA, sa, XB, sb), L2_shape=sr, L3_tangent=tc, tangent_null_p95=float(np.percentile(null, 95))))
            print(rows[-1], flush=True)
H = pd.DataFrame(rows); H.to_csv(f'{OUTD}/may_base_hierarchy.csv', index=False)
print(H.round(3).to_string())
