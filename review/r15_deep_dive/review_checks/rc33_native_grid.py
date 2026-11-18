# RC33 (post hoc, 2026-11-09, after round 7; Zac chose "native grid leads, 24 kept"). For every RC22 pair whose
# shallower model does not have 24 hooked layers (Z-Image 30, PixArt/Cosmos 28), the tangent on the native grid
# (n = hooked layers of the shallower model, so no block is averaged away by interpolation) with:
#   perm95   1,000-draw layer-order permutation p95 of the second model (rng 33)
#   surr95   300-draw RC22 smoothness-preserving surrogate p95 (shared random phase on the depth differences)
#   ci05/95  200-draw generation bootstrap (30 gens each side), as RC22
#   p10k     10,000-draw layer-order permutation p-value on the native grid, p = (k + 1) / (N + 1) (same rng stream)
#   rule     native - perm95 (lineage rule: >= 0.30)
# Pairs with 24 hooked layers are unchanged from RC22/RC30 (native grid = 24-point grid).
# Usage: python rc33_native_grid.py <cache> <r16cubes> <r17cubes>
import os, sys, numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rc22_round5_checks as R
from rc26_round6_tangent import prof, tan
OUT = R.OUT
def job(p):
    kind, a, b, sub, img = p; XA, sa = R.load(a, sub, img); XB, sb = R.load(b, sub, img); MA, MB = XA.mean(0), XB.mean(0)
    n = min(MA.shape[1], MB.shape[1]); rng = np.random.default_rng(33)
    r = dict(kind=kind, a=a, b=b, sub=sub, rows='image' if img else 'all', La=MA.shape[1], Lb=MB.shape[1], n=n)
    A = prof(MA, len(sa), n); B = prof(MB, len(sb), n); r['native'] = tan(A, sa, B, sb)
    r['perm95'] = float(np.percentile([tan(A, sa, prof(MB[:, rng.permutation(MB.shape[1])], len(sb), n), sb) for _ in range(1000)], 95))
    k = sum(tan(A, sa, prof(MB[:, rng.permutation(MB.shape[1])], len(sb), n), sb) >= r['native'] for _ in range(10000)); r['p10k'] = (k + 1) / 10001
    sur = np.array([R.tan_diff(A, sa, R.surrogate(B, rng), sb) for _ in range(300)]); r['surr95'] = float(np.percentile(sur, 95))
    bs = []
    for _ in range(200):
        ia = rng.choice(len(XA), min(30, len(XA)), replace=False); ib = rng.choice(len(XB), min(30, len(XB)), replace=False)
        bs.append(tan(prof(XA[ia].mean(0), len(sa), n), sa, prof(XB[ib].mean(0), len(sb), n), sb))
    r['ci05'], r['ci95'] = (float(x) for x in np.percentile(bs, [5, 95])); r['rule_margin'] = r['native'] - r['perm95']
    print({q: (round(v, 4) if isinstance(v, float) else v) for q, v in r.items()}, flush=True); return r
if __name__ == '__main__':
    todo = []
    for p in R.P:
        XA, _ = R.load(p[1], p[3], p[4]); XB, _ = R.load(p[2], p[3], p[4])
        if min(XA.shape[2], XB.shape[2]) != 24: todo.append(p)
    with Pool(os.cpu_count()) as pool: D = pd.DataFrame(pool.map(job, todo))
    D.to_csv(f'{OUT}/rc33_native_grid.csv', index=False); pd.set_option('display.width', 250)
    print(D.round(4).to_string(index=False))
