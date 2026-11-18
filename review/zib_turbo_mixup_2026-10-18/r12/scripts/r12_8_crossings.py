import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# R12-8 (descriptive): layer-axis crossings with the analyzer's plot_beta_layer unchanged (via lens_L2_rg.curve),
# 100 prompt bootstraps (rng seeds 3000+k, all gens of each drawn prompt), as review/r3_ace/R3B_SCORING_PLAN.md B6.
# Input: cond-branch cubes (complete gens). Also run on Z3, the Turbo-8 archive and the old "ZIB" archive (Turbo-25 pooled) for reference.
import os, sys, types, pickle, json
sys.modules['gradio'] = types.ModuleType('gradio')
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/lenses')
import lens_L2_rg as L2
SCR = os.environ.get('SCR', _COSPAFS_CACHE); OUTD = f'{_COSPAFS_ROOT}/review/zib_turbo_mixup_2026-10-18/r12/out'
pat = lambda c: ''.join('+' if d == 'pos→neg' else '-' for _, d, _ in c)
def df_of(name):
    _, meta, X, s = pickle.load(open(f'{SCR}/F_{name}.pkl', 'rb')); G, T, L, _ = X.shape
    gi, t, l = np.meshgrid(np.arange(G) + 1, np.arange(T), np.arange(L), indexing='ij')
    return pd.DataFrame(dict(gen_idx=gi.ravel(), token_idx=t.ravel(), layer=l.ravel(), g_micro=X[..., 0].ravel())), np.asarray(meta.pt.astype(str).str.strip())
def pboot(args):
    df, pts, seed = args; rng = np.random.default_rng(seed); up = np.array(sorted(set(pts))); pick = rng.choice(up, len(up), replace=True)
    by = {p: np.where(pts == p)[0] + 1 for p in up}; parts, k = [], 0
    g = {gid: d for gid, d in df.groupby('gen_idx')}
    for p in pick:
        for gid in by[p]:
            k += 1; parts.append(g[gid].assign(gen_idx=k))
    return L2.curve(pd.concat(parts, ignore_index=True))
rows = []
for name in sys.argv[1:]:
    df, pts = df_of(name); obs = L2.curve(df)
    with Pool(8) as pool: bs = pool.map(pboot, [(df, pts, 3000 + k) for k in range(100)])
    c = pd.Series([len(b) for b in bs]); ps = pd.Series([pat(b) for b in bs])
    r = dict(cube=name, n_gens=int(df.gen_idx.nunique()), point_n=len(obs), point_pattern=pat(obs),
             point_g=[round(x, 5) for x, _, _ in obs], boot_mode=int(c.mode()[0]), boot_mode_share=float((c == c.mode()[0]).mean()),
             share3=float((c == 3).mean()), share_pmp=float((ps == '+-+').mean()), count_dist=c.value_counts().sort_index().to_dict())
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(f'{OUTD}/r12_8_crossings_{"_".join(sys.argv[1:])}.csv', index=False)
