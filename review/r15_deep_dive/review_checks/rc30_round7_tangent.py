# RC30 (post hoc, round-7 reviews 2026-11-09: GLM M10a and Q4/M3b, DeepSeek 3.3). On every pair of the RC22 tier table,
# with the Round 6 estimator unchanged except where stated:
#   p10k     layer-order permutation p-value of the 24-point tangent with 10,000 draws of the second model (rng 30),
#            p = (k + 1) / (N + 1), k = draws at or above the observed value
#   native   tangent on a grid with as many points as the shallower model of the pair has hooked layers (a rule tied to
#            the comparison set rather than fixed), with its own 1,000-draw permutation p95
# Usage: python rc30_round7_tangent.py <cache> <r16cubes> <r17cubes>
import os, sys, numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rc22_round5_checks as R
from rc26_round6_tangent import prof, tan
OUT = R.OUT; N = 10000
def job(p):
    kind, a, b, sub, img = p; XA, sa = R.load(a, sub, img); XB, sb = R.load(b, sub, img); MA, MB = XA.mean(0), XB.mean(0)
    rng = np.random.default_rng(30); r = dict(kind=kind, a=a, b=b, sub=sub, rows='image' if img else 'all', La=MA.shape[1], Lb=MB.shape[1])
    A = prof(MA, len(sa)); obs = tan(A, sa, prof(MB, len(sb)), sb); r['full24'] = obs
    k = sum(tan(A, sa, prof(MB[:, rng.permutation(MB.shape[1])], len(sb)), sb) >= obs for _ in range(N))
    r['k_ge'] = int(k); r['p10k'] = (k + 1) / (N + 1)
    n = min(MA.shape[1], MB.shape[1]); r['n_native'] = n; An = prof(MA, len(sa), n); r['native'] = tan(An, sa, prof(MB, len(sb), n), sb)
    r['p95_native'] = float(np.percentile([tan(An, sa, prof(MB[:, rng.permutation(MB.shape[1])], len(sb), n), sb) for _ in range(1000)], 95))
    print({q: (round(v, 4) if isinstance(v, float) else v) for q, v in r.items()}, flush=True); return r
if __name__ == '__main__':
    with Pool(os.cpu_count()) as pool: D = pd.DataFrame(pool.map(job, R.P))
    D.to_csv(f'{OUT}/rc30_round7_tangent.csv', index=False); pd.set_option('display.width', 250)
    print(D.round(4).to_string(index=False))
