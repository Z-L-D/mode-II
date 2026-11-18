# RC26 (post hoc, round-6 reviews 2026-11-09: Qwen M3/Q3/4.2/4.3, Kimi M3/M4, GLM 5.3c/5.4/Q3).
# On every pair of the RC22 tier table this computes, with the Round 6 estimator otherwise unchanged (cosine of first
# differences of depth-standardised profiles, nearest matched clock, first-named model supplies the states):
#   grid      tangent at depth grids of 12 / 24 / 48 points x linear / cubic (not-a-knot spline) interpolation,
#             each with its own layer-order permutation p95 (200 draws, rng 26)
#   mid3      tangent on the middle third of depth only (24-grid intervals 7..15, u ~ 0.30-0.70) and its permutation p95
#   pboot     PROMPT-level bootstrap: prompts resampled with replacement within each model (all generations of a drawn
#             prompt kept), 200 draws, rng 26; 5th/95th percentiles and the share of draws below the full-sample value
# Usage: python rc26_round6_tangent.py <cache> <r16cubes> <r17cubes>
import os, sys, pickle, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.interpolate import CubicSpline
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rc22_round5_checks as R  # reads sys.argv[1:4]; pair list P, load(), depthnorm(), cos()
OUT = R.OUT
def prof(M, ns, n=24, kind='linear'):
    D = R.depthnorm(M); u = np.linspace(0, 1, D.shape[1]); G = np.linspace(0, 1, n)
    if kind == 'linear': return np.stack([[np.interp(G, u, D[t, :, k]) for k in range(3)] for t in range(ns)]).transpose(0, 2, 1)
    return np.stack([[CubicSpline(u, D[t, :, k])(G) for k in range(3)] for t in range(ns)]).transpose(0, 2, 1)
def tan(A, sa, B, sb, seg=None):
    v = []
    for t in range(len(sa)):
        j = int(np.argmin(np.abs(sb - sa[t]))); a = np.diff(A[t], axis=0); b = np.diff(B[j], axis=0)
        if seg is not None: a, b = a[seg], b[seg]
        v.append(R.cos(a, b))
    return float(np.mean(v))
MID = slice(7, 16)
def meta(n, sub, img):
    d, f = R.SRC.get(n, (R.CA, n))
    if img: return np.asarray(pickle.load(open(f'{d}/V_{f}_{sub}.pkl', 'rb'))['meta']['pt'].astype(str))
    return np.asarray(pickle.load(open(f'{d}/F_{f}_{sub}.pkl', 'rb'))[1].pt.astype(str))
def job(p):
    kind, a, b, sub, img = p; XA, sa = R.load(a, sub, img); XB, sb = R.load(b, sub, img); MA, MB = XA.mean(0), XB.mean(0)
    r = dict(kind=kind, a=a, b=b, sub=sub, rows='image' if img else 'all'); rng = np.random.default_rng(26)
    perms = [rng.permutation(MB.shape[1]) for _ in range(200)]
    for n in (12, 24, 48):
        for k in ('linear', 'cubic'):
            A = prof(MA, len(sa), n, k); r[f'g{n}_{k}'] = tan(A, sa, prof(MB, len(sb), n, k), sb)
            r[f'p95_g{n}_{k}'] = float(np.percentile([tan(A, sa, prof(MB[:, q], len(sb), n, k), sb) for q in perms], 95))
    A = prof(MA, len(sa)); B = prof(MB, len(sb)); r['mid3'] = tan(A, sa, B, sb, MID)
    r['p95_mid3'] = float(np.percentile([tan(A, sa, prof(MB[:, q], len(sb)), sb, MID) for q in perms], 95))
    pa, pb = meta(a, sub, img), meta(b, sub, img); ua, ub = np.unique(pa), np.unique(pb)
    ia = {u: np.where(pa == u)[0] for u in ua}; ib = {u: np.where(pb == u)[0] for u in ub}; bs = []
    for _ in range(200):
        ga = np.concatenate([ia[u] for u in rng.choice(ua, len(ua))]); gb = np.concatenate([ib[u] for u in rng.choice(ub, len(ub))])
        bs.append(tan(prof(XA[ga].mean(0), len(sa)), sa, prof(XB[gb].mean(0), len(sb)), sb))
    r['full24'] = r['g24_linear']; r['pb05'], r['pb95'] = np.percentile(bs, [5, 95]); r['pb_share_below'] = float((np.array(bs) < r['full24']).mean())
    r['n_prompts_a'], r['n_prompts_b'] = len(ua), len(ub)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True); return r
if __name__ == '__main__':
    with Pool(min(16, os.cpu_count())) as pool: D = pd.DataFrame(pool.map(job, R.P))
    D.to_csv(f'{OUT}/rc26_round6_tangent.csv', index=False)
    pd.set_option('display.width', 320); pd.set_option('display.max_columns', 40)
    cols = ['kind', 'a', 'b', 'sub', 'rows'] + [f'g{n}_{k}' for n in (12, 24, 48) for k in ('linear', 'cubic')] + ['mid3', 'p95_mid3', 'pb05', 'pb95', 'pb_share_below']
    print(D[cols].round(3).to_string(index=False))
