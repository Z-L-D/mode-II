import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# POST HOC (version 2, 2026-11-05): the Round 6 S1 observable hierarchy (envelope g, envelope log g, field, shape, tangent;
# r6c_S1_hierarchy.py functions unchanged) with the real Z-Image Base (Z1, cond branch) in place of the mislabelled archive
# campaign, plus the same-weights pair ZIT-25 <-> ZIT and the matched Turbo rerun Z3 for reference. Not preregistered.
# Output: ../out/posthoc_hierarchy_realbase.csv and posthoc_hierarchy_margins.csv
import sys
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round6/scripts')
from r6c_common import *
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
MOD = ['ZIT', 'ZIB', 'Z1', 'Z3', 'MOCHI', 'SD35M', 'SD35L']
rows = []
for sub in ('attn', 'mlp'):
    D = {}
    for m in MOD:
        try: D[m] = arch(m, sub)
        except FileNotFoundError: pass
    E1 = {m: D[m][2][..., 0].mean((0, 2)) for m in D}; E2 = {m: np.log(D[m][2][..., 0]).mean((0, 2)) for m in D}
    pairs = [('ZIT', 'ZIB', 'same weights'), ('Z3', 'ZIT', 'same weights'), ('Z1', 'ZIT', 'lineage'), ('Z1', 'Z3', 'lineage')]
    pairs += [(a, b, 'unrelated') for a in ('ZIT', 'Z1') for b in ('MOCHI', 'SD35M', 'SD35L')] + [('MOCHI', 'SD35M', 'unrelated'), ('MOCHI', 'SD35L', 'unrelated'), ('SD35M', 'SD35L', 'same family')]
    for a, b, k in pairs:
        if a not in D or b not in D: continue
        _, _, XA, sa = D[a]; _, _, XB, sb = D[b]; tc, sr = tangent(XA, sa, XB, sb)
        rows.append(dict(sub=sub, a=a, b=b, kind=k, L0_D1=env_r(E1[a], sa, E1[b], sb), L0_D2=env_r(E2[a], sa, E2[b], sb),
                         L1_field=field_r(XA, sa, XB, sb), L2_shape=sr, L3_tangent=tc))
H = pd.DataFrame(rows); H.to_csv(f'{OUT}/posthoc_hierarchy_realbase.csv', index=False)
lv = ['L0_D1', 'L0_D2', 'L1_field', 'L2_shape', 'L3_tangent']; un = H[(H['sub'] == 'attn') & (H.kind == 'unrelated')]
M = []
for a, b in [('ZIT', 'ZIB'), ('Z1', 'ZIT')]:
    r = H[(H['sub'] == 'attn') & (H.a == a) & (H.b == b)].iloc[0]
    M.append(dict(pair=f'{a}-{b}', **{k: float(r[k] - un[k].max()) for k in lv}))
M.append(dict(pair='unrelated_max', **{k: float(un[k].max()) for k in lv}))
pd.DataFrame(M).to_csv(f'{OUT}/posthoc_hierarchy_margins.csv', index=False)
pd.set_option('display.width', 250); print(H.round(3).to_string()); print(pd.DataFrame(M).round(3).to_string())
