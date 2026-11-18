"""U1: score preregistered C1-C3 (R3_PREREGISTRATION.md Part C) as fixed in R3C_SCORING_PLAN.md (SHA 7a3b852f, 21:46:11Z).
Output: out/C1_cells.csv, out/C_scores.csv"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/lenses')
from lens_L2_rg import spline_basis   # unchanged spline basis
c = np.load(f'{BASE}/out/u0_cache.npz', allow_pickle=True); NL = c['NL']; pt = c['pt']
rng = np.random.default_rng(1)

def g_mp(N, D, reps=20):
    v = []
    for _ in range(reps):
        s = np.linalg.svd(rng.standard_normal((N, D)) / np.sqrt(N), compute_uv=False); v.append(s.sum() ** 2 / (D * (s ** 2).sum()))
    return float(np.mean(v))
GMP = {}
for N, D in sorted({tuple(x) for x in NL}):
    GMP[(N, D)] = g_mp(int(N), int(D))
print('g_MP', {k: round(v, 4) for k, v in GMP.items()})
gmp_block = np.array([GMP[tuple(x)] for x in NL])

LEVEL = np.array(['d4096', 'd4096', 'd1024', 'd1024', 'd256', 'd256', 'mid', 'u256', 'u256', 'u256', 'u1024', 'u1024', 'u1024', 'u4096', 'u4096', 'u4096'])
WITHIN = [(0, 1), (2, 3), (4, 5), (7, 8), (8, 9), (10, 11), (11, 12), (13, 14), (14, 15)]

def cv_r2(x, y, grp, designs):
    """5-fold CV grouped by prompt, fold assignment as lens_L2_rg.r1 (rng seed 1). designs: list of extra FE index arrays (or None)."""
    up = np.unique(pt); r = np.random.default_rng(1); fold_of = dict(zip(up, r.permutation(len(up)) % 5))
    fold = np.array([fold_of[p] for p in grp]); preds = [np.empty_like(y) for _ in designs]
    for f in range(5):
        tr, te = fold != f, fold == f
        lo, hi = x[tr].min(), x[tr].max(); kn = np.quantile(x[tr], np.linspace(0, 1, 14)[1:-1])
        Btr, Bte = spline_basis(x[tr], lo, hi, kn), spline_basis(x[te], lo, hi, kn)
        for k, fe in enumerate(designs):
            if fe is None:
                Mtr, Mte = Btr, Bte
            else:
                u, inv = np.unique(fe, return_inverse=True); E = np.eye(len(u))[inv][:, 1:]
                Mtr, Mte = np.hstack([Btr, E[tr]]), np.hstack([Bte, E[te]])
            cf = np.linalg.lstsq(Mtr, y[tr], rcond=None)[0]; preds[k][te] = Mte @ cf
    ss = np.sum((y - y.mean()) ** 2)
    return [1 - np.sum((y - p) ** 2) / ss for p in preds]

def pairs_xy(g, pairs):
    n, T, L = g.shape; X, Y, P, G = [], [], [], []
    for (a, b) in pairs:
        X.append(g[:, :, a].ravel()); Y.append((g[:, :, b] - g[:, :, a]).ravel()); P.append(np.full(n * T, a)); G.append(np.repeat(pt, T))
    x, y, p, grp = (np.concatenate(v) for v in (X, Y, P, G)); ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok], p[ok], grp[ok]

rows, cells = [], []
for sub in ('attn', 'mlp'):
    for br in ('cond', 'uncond'):
        g = c[f'{sub}_{br}_g_micro'].astype(float); n, T, L = g.shape
        # C1
        med = np.nanmedian(g, 0); ratio = med / gmp_block[None, :]; frac_cells = float((ratio >= 0.10).mean())
        frac_rows = float(np.nanmean(g / gmp_block[None, None, :] >= 0.10))
        if br == 'cond':
            for t in range(T):
                for l in range(L): cells.append(dict(channel=sub, step=t, block=l, N=NL[l, 0], D=NL[l, 1], g_median=med[t, l], g_MP=gmp_block[l], ratio=ratio[t, l]))
        # C2
        x, y, p, grp = pairs_xy(g, [(l, l + 1) for l in range(L - 1)])
        R2g, R2gl = cv_r2(x, y, grp, [None, p])
        # C3
        x, y, p, grp = pairs_xy(g, WITHIN)
        R2g_w, R2lev, R2pair = cv_r2(x, y, grp, [None, LEVEL[p], p])
        rows.append(dict(channel=sub, branch=br, gens=n,
                         C1_frac_cells_ge_0p10=frac_cells, C1_max_cell_ratio=float(np.nanmax(ratio)), C1_median_cell_ratio=float(np.nanmedian(ratio)), C1_frac_rows_ge_0p10=frac_rows,
                         C2_R2_g=R2g, C2_R2_g_pair=R2gl, C2_dR2=R2gl - R2g,
                         C3_R2_g=R2g_w, C3_R2_g_level=R2lev, C3_R2_g_pair=R2pair, C3_dR2_within=R2pair - R2lev, C3_sens_dR2_pair_vs_g=R2pair - R2g_w))
S = pd.DataFrame(rows)
S['C1_holds'] = S.C1_frac_cells_ge_0p10 <= 0.05; S['C2_holds'] = S.C2_dR2 >= 0.30; S['C3_holds'] = S.C3_dR2_within >= 0.15
pd.DataFrame(cells).to_csv(f'{BASE}/out/C1_cells.csv', index=False); S.to_csv(f'{BASE}/out/C_scores.csv', index=False)
pd.set_option('display.width', 250); print(S.T.to_string())
