import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# POST HOC (not preregistered), labelled so wherever used. ACE Base context numbers for the papers:
# (a) ACE Base cond <-> unrelated image/video archive models (attn and, where present, mlp), Round 6 S1 tangent() unchanged;
# (b) ACE Base MLP layer-axis crossings (B6 method, descriptive); (c) per-sigma B8 curve and B2 ratio summaries for the figure.
# Usage: SCR=<cube dir> python posthoc_r14.py
import os, sys, json, pickle, types
sys.modules['gradio'] = types.ModuleType('gradio')
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round6/scripts'); sys.path.insert(0, f'{_COSPAFS_ROOT}/review/lenses')
from r6c_common import *
import lens_L2_rg as L2
from multiprocessing import Pool
OUTD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out'); SCR = os.environ.get('SCR', _COSPAFS_CACHE)
def cube(n, sub):
    _, meta, X, s = pickle.load(open(f'{SCR}/F_{n}_{sub}.pkl', 'rb')); return meta, X, np.asarray(s, float)
rows = []
for sub, others in (('attn', ['Z1', 'Z3', 'ZIT', 'ZIB', 'MOCHI', 'SD35M', 'SD35L']), ('mlp', ['Z1', 'Z3', 'ZIT', 'ZIB'])):
    for a in ('ACEB', 'ACET'):
        _, XA, sa = cube(a, sub)
        for o in others:
            _, XB, sb = cube(o, sub); tc, sr = tangent(XA, sa, XB, sb); rows.append(dict(a=a, b=o, sub=sub, tangent=tc, shape=sr))
T = pd.DataFrame(rows); T.to_csv(f'{OUTD}/posthoc_ace_vs_image_tangents.csv', index=False); print(T.to_string())
pat = lambda c: ''.join('+' if d == 'pos→neg' else '-' for _, d, _ in c)
def frame(n, sub):
    meta, X, _ = cube(n, sub); G, T_, L_ = X.shape[:3]
    gi, t, l = np.meshgrid(np.arange(G) + 1, np.arange(T_), np.arange(L_), indexing='ij')
    return pd.DataFrame(dict(gen_idx=gi.ravel(), token_idx=t.ravel(), layer=l.ravel(), g_micro=X[..., 0].ravel())), meta.pt.astype(str).str.strip().values
def pboot(args):
    df, pts, seed = args; rng = np.random.default_rng(seed); up = np.array(sorted(set(pts))); pick = rng.choice(up, len(up), replace=True)
    g = {gid: d for gid, d in df.groupby('gen_idx')}; parts, k = [], 0
    for p in pick:
        for gid in np.where(pts == p)[0] + 1:
            k += 1; parts.append(g[gid].assign(gen_idx=k))
    return L2.curve(pd.concat(parts, ignore_index=True))
cr = []
for n, sub in (('ACEB', 'mlp'), ('ACEBU', 'attn'), ('ACEBU', 'mlp')):
    df, pts = frame(n, sub); obs = L2.curve(df)
    with Pool(8) as pool: bs = pool.map(pboot, [(df, pts, 3000 + k) for k in range(100)])
    c = pd.Series([len(b) for b in bs])
    cr.append(dict(cube=f'{n}_{sub}', point_n=len(obs), point_pattern=pat(obs), boot_mode=int(c.mode()[0]), boot_mode_share=float((c == c.mode()[0]).mean()), count_dist=c.value_counts().sort_index().to_dict()))
    print(cr[-1], flush=True)
pd.DataFrame(cr).to_csv(f'{OUTD}/posthoc_aceb_crossings.csv', index=False)

# (d) backbone (999 depth-shift nulls) and clock-axis three-segment topology (S4 fit3, 1000 prompt bootstraps): functions copied
# unchanged from review/peer_review_2026-10-17/r11/scripts/score_r11.py. Real Z-Image Base (Z1 cond) for reference.
def backbone(X, nnull=999, seed=0):
    Y = np.nanmean(X, 0).copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1])
    Z = (Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-12)
    def stats(Z):
        A = Z.reshape(Z.shape[0], -1); A = A - A.mean(0); ev = np.clip(np.linalg.svd(A, compute_uv=False) ** 2, 0, None); p = ev / ev.sum(); p = p[p > 0]
        return p[0], float(np.exp(-(p * np.log(p)).sum()))
    pc1, er = stats(Z); r = np.random.default_rng(seed); T, L, _ = Z.shape; npc, nrk = [], []
    for _ in range(nnull):
        Zs = np.stack([np.roll(Z[t], r.integers(L), axis=0) for t in range(T)]); a, b = stats(Zs); npc.append(a); nrk.append(b)
    npc, nrk = np.array(npc), np.array(nrk)
    return dict(pc1=pc1, eff_rank=er, null_pc1_max=npc.max(), p_pc1=(1 + (npc >= pc1).sum()) / (nnull + 1))
def fit3(y, s):
    u = -s; T = len(y); best = None
    for i in range(1, T - 2):
        for j in range(i + 1, T - 1):
            A = np.stack([np.ones(T), u, np.maximum(0, u - u[i]), np.maximum(0, u - u[j])], 1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None); sse = ((A @ coef - y) ** 2).sum()
            if best is None or sse < best[0]: best = (sse, i, j, coef)
    sse, i, j, c = best; sl = np.array([c[1], c[1] + c[2], c[1] + c[2] + c[3]])
    return ''.join('+' if v > 0 else '-' for v in sl), i, j, sl
def clock(meta, X, s, nb=1000):
    k = meta.pt.astype(str).str.strip().values; up = np.unique(k); PP = np.stack([X[k == p][..., 0].mean((0, 2)) for p in up])
    top = fit3(PP.mean(0), s)[0]; r = np.random.default_rng(64)
    boot = np.array([fit3(PP[r.integers(0, len(PP), len(PP))].mean(0), s)[0] for _ in range(nb)])
    return dict(topology=top, frac_pmp=float((boot == '+-+').mean()))
BK = []
for n in ('ACEB', 'ACET', 'Z1'):
    for sub in ('attn', 'mlp'):
        meta, X, s = cube(n, sub)
        BK.append(dict(cube=n, sub=sub, **backbone(X), **clock(meta, X, s))); print(BK[-1], flush=True)
pd.DataFrame(BK).to_csv(f'{OUTD}/posthoc_backbone_clock.csv', index=False)
