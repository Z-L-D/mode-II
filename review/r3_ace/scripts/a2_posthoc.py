"""A2 POST HOC (R3B_SCORING_PLAN.md, post hoc list): P1 backbone (u2_posthoc.backbone, copied), P5 ten-way domain decoding
(I-F5 decoder, copied from u3_content.py), P7 clock-axis topology (R6 S4 fit3/extrema, 1000 prompt bootstraps, rng 64),
P8 layer-axis crossings in MLP (analyzer plot_beta_layer, descriptive; B6 covered attention).
Output: out/P1_backbone.csv, P5_decoding.csv, P7_clock_topology.csv, P7_clock_curves.csv, P8_mlp_crossings.json"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import os, sys, glob, json, types, warnings, importlib.util
sys.modules['gradio'] = types.ModuleType('gradio'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE); OUT = f'{BASE}/out'
C = {s: dict(np.load(f'{OUT}/cache_{s}.npz', allow_pickle=True)) for s in ('attn', 'mlp')}
X3 = lambda s: np.stack([C[s]['g_gen'], C[s]['gap_gen'], C[s]['ent_gen']], -1).astype(float)

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
    return dict(pc1=pc1, eff_rank=er, null_pc1_mean=npc.mean(), null_rank_mean=nrk.mean(), p_rank=(1 + (nrk <= er).sum()) / (nnull + 1), p_pc1=(1 + (npc >= pc1).sum()) / (nnull + 1))
P1 = pd.DataFrame([dict(backbone(X3(s)), channel=f'ACE {s}') for s in C]); P1.to_csv(f'{OUT}/P1_backbone.csv', index=False); print(P1.round(3).to_string(index=False), flush=True)

def clf(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
def acc(F, y, grp): return float(cross_val_score(clf(), F, y, groups=grp, cv=GroupKFold(5)).mean())
def scr_one(F0, y, grp, seed):
    rng = np.random.default_rng(seed); Fs = np.stack([f[rng.permutation(f.shape[0])] for f in F0]).reshape(len(F0), -1); return acc(Fs, y, grp)
pt = C['attn']['prompts'].astype(str); y = C['attn']['dom'].astype(str); grp = pd.factorize(pt)[0]; pl = np.array([len(p) for p in pt], float); rows = []
for s in C:
    X = X3(s); T = X.shape[1]
    for t in range(T):
        Ft = np.log(np.abs(X[:, t]) + 1e-9)
        rows.append(dict(channel=f'ACE-{s}', measure='ordered_profile', step=t, acc=acc(Ft.reshape(len(X), -1), y, grp)))
        rows.append(dict(channel=f'ACE-{s}', measure='layer_mean', step=t, acc=acc(Ft.mean(1), y, grp)))
    rows.append(dict(channel=f'ACE-{s}', measure='prompt_length', step=0, acc=acc(pl[:, None], y, grp)))
    txt = TfidfVectorizer(min_df=1).fit_transform(list(pt))
    rows.append(dict(channel=f'ACE-{s}', measure='tfidf_text', step=0, acc=float(cross_val_score(LogisticRegression(C=10, max_iter=3000), txt, y, groups=grp, cv=GroupKFold(5)).mean())))
    F0 = np.log(np.abs(X[:, 0]) + 1e-9)
    for k, v in enumerate(Parallel(n_jobs=4)(delayed(scr_one)(F0, y, grp, k) for k in range(99))):
        rows.append(dict(channel=f'ACE-{s}', measure='depth_scramble', step=0, acc=v, realization=k))
P5 = pd.DataFrame(rows); P5.to_csv(f'{OUT}/P5_decoding.csv', index=False)
for ch, d in P5.groupby('channel'):
    real = d[(d.measure == 'ordered_profile') & (d.step == 0)].acc.iloc[0]; nul = d[d.measure == 'depth_scramble'].acc
    print(ch, 'ordered %.3f layer_mean %.3f prompt_len %.3f tfidf %.3f scr_max %.3f p %.3f' % (real, d[(d.measure == 'layer_mean') & (d.step == 0)].acc.iloc[0],
          d[d.measure == 'prompt_length'].acc.iloc[0], d[d.measure == 'tfidf_text'].acc.iloc[0], nul.max(), (1 + (nul >= real).sum()) / 100),
          'along steps', d[d.measure == 'ordered_profile'].acc.round(3).tolist(), flush=True)

def fit3(y, s):
    u = -s; T = len(y); best = None
    for i in range(1, T - 2):
        for j in range(i + 1, T - 1):
            A = np.stack([np.ones(T), u, np.maximum(0, u - u[i]), np.maximum(0, u - u[j])], 1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None); sse = ((A @ coef - y) ** 2).sum()
            if best is None or sse < best[0]: best = (sse, i, j, coef)
    sse, i, j, c = best; sl = np.array([c[1], c[1] + c[2], c[1] + c[2] + c[3]])
    A = np.stack([np.ones(T), u, np.maximum(0, u - u[i]), np.maximum(0, u - u[j])], 1)
    return ''.join('+' if v > 0 else '-' for v in sl), i, j, sl, A @ c
def extrema(y, s):
    tr = int(np.argmin(y)); pk = int(np.argmax(y[:tr + 1])) if tr > 0 else 0
    return s[pk], s[tr], (y[tr] - y[pk]) / y[pk], (y[-1] - y[tr]) / y[tr]
rows, curves = [], []; up = np.unique(pt); sig = C['attn']['sigma'].astype(float)
for s in C:
    G = C[s]['g_gen']; PP = np.stack([np.nanmean(G[pt == p], axis=(0, 2)) for p in up]); yv = PP.mean(0); rng = np.random.default_rng(64)
    top, i, j, sl, fit = fit3(yv, sig); pk, tr, amp, reb = extrema(yv, sig)
    boot = np.array([fit3(PP[rng.integers(0, len(PP), len(PP))].mean(0), sig)[0] for _ in range(1000)]); v, cn = np.unique(boot, return_counts=True)
    rows.append(dict(channel=f'ACE-{s}', clock='sigma', prompts=len(PP), steps=len(sig), topology=top, frac_pmp=float((boot == '+-+').mean()), frac_ppm=float((boot == '++-').mean()),
                     modal=v[cn.argmax()], modal_frac=cn.max() / 1000, break1=sig[i], break2=sig[j], contraction=amp, rebound=reb, late_drop=(yv[-1] - yv.max()) / yv.max(),
                     sigma_at_max=sig[yv.argmax()]))
    curves += [dict(channel=f'ACE-{s}', step=k, sigma=sig[k], g=yv[k], fit=fit[k]) for k in range(len(sig))]
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/P7_clock_topology.csv', index=False); pd.DataFrame(curves).to_csv(f'{OUT}/P7_clock_curves.csv', index=False)

sys.path.insert(0, f'{_COSPAFS_ROOT}/review/lenses'); import lens_L2_rg as L2
dm = L2.A.DataManager(); dm.load_paths(','.join(sorted(glob.glob(f'{BASE}/data/*.csv')))); DF = dm.get_df(); DF = DF[DF.g_micro.notna()]
o = L2.curve(DF[DF.submodule == 'mlp']); json.dump(dict(n=len(o), crossings=[(float(a), b, float(c)) for a, b, c in o]), open(f'{OUT}/P8_mlp_crossings.json', 'w'))
print('P8 MLP crossings', o)
