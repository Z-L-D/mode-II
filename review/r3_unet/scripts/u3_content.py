"""U3 POST HOC (R3C_SCORING_PLAN_ADDENDUM.md): P5 domain decoding (Mode-I I-F5 decoder unchanged) and P6 prompt RSA
(Round 6 S2 method unchanged) for SD1.5 cond. Output: out/P5_decoding.csv, out/P6_rsa.csv"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import os, sys, pickle, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
SCR = os.environ.get('SCR', f'{_COSPAFS_CACHE}')
c = np.load(f'{BASE}/out/u0_cache.npz', allow_pickle=True); pt = c['pt'].astype(str); dom = c['dom'].astype(str)
def X_sd(sub): return np.stack([c[f'{sub}_cond_{v}'].astype(float) for v in ('g_micro', 'spectral_gap', 'sv_entropy')], -1)

# ---- P5 (decoder verbatim from papers/mode-I/scripts/fig_I_F05_semantic.py) ----
def clf(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
def acc(F, y, grp): return float(cross_val_score(clf(), F, y, groups=grp, cv=GroupKFold(5)).mean())
def scr_one(F0, y, grp, seed):
    rng = np.random.default_rng(seed); Fs = np.stack([f[rng.permutation(f.shape[0])] for f in F0]).reshape(len(F0), -1); return acc(Fs, y, grp)
rows = []
y = dom; grp = pd.factorize(pt)[0]; pl = np.array([len(p) for p in pt], float)
for sub in ('attn', 'mlp'):
    X = X_sd(sub); X = np.where(np.isfinite(X), X, np.nanmean(X, 0, keepdims=True)); T = X.shape[1]
    for t in sorted({0, T // 4, T // 2, 3 * T // 4, T - 1}):
        Ft = np.log(np.abs(X[:, t]) + 1e-9)
        rows.append(dict(channel=f'SD1.5-{sub}', measure='ordered_profile', step=t, acc=acc(Ft.reshape(len(X), -1), y, grp)))
        rows.append(dict(channel=f'SD1.5-{sub}', measure='layer_mean', step=t, acc=acc(Ft.mean(1), y, grp)))
    rows.append(dict(channel=f'SD1.5-{sub}', measure='prompt_length', step=0, acc=acc(pl[:, None], y, grp)))
    txt = TfidfVectorizer(min_df=1).fit_transform(list(pt))
    rows.append(dict(channel=f'SD1.5-{sub}', measure='tfidf_text', step=0, acc=float(cross_val_score(LogisticRegression(C=10, max_iter=3000), txt, y, groups=grp, cv=GroupKFold(5)).mean())))
    F0 = np.log(np.abs(X[:, 0]) + 1e-9)
    for k, v in enumerate(Parallel(n_jobs=4)(delayed(scr_one)(F0, y, grp, k) for k in range(99))):
        rows.append(dict(channel=f'SD1.5-{sub}', measure='depth_scramble', step=0, acc=v, realization=k))
    print(sub, 'P5 done', flush=True)
P5 = pd.DataFrame(rows); P5.to_csv(f'{BASE}/out/P5_decoding.csv', index=False)
for ch, d in P5.groupby('channel'):
    real = d[(d.measure == 'ordered_profile') & (d.step == 0)].acc.iloc[0]; nul = d[d.measure == 'depth_scramble'].acc
    print(ch, 'ordered %.3f layer_mean %.3f prompt_len %.3f tfidf %.3f scr_max %.3f p %.3f' % (real, d[(d.measure == 'layer_mean') & (d.step == 0)].acc.iloc[0],
          d[d.measure == 'prompt_length'].acc.iloc[0], d[d.measure == 'tfidf_text'].acc.iloc[0], nul.max(), (1 + (nul >= real).sum()) / 100),
          'along steps', d[d.measure == 'ordered_profile'].acc.round(3).tolist(), flush=True)

# ---- P6 (Round 6 S2 method) ----
rng = np.random.default_rng(62); NPERM = 2000; gr24 = np.linspace(0, 1, 24); MODELS = ['ZIT', 'ZIB', 'MOCHI', 'SD35M', 'SD35L']
A = {m: pickle.load(open(f'{SCR}/F_{m}_attn.pkl', 'rb')) for m in MODELS}
keys = [set(A[m][1].pt.astype(str).str.strip()) for m in MODELS] + [set(np.char.strip(pt))]
P = sorted(set.intersection(*keys)); NP = len(P); print('shared prompts', NP, flush=True)
def feats_arr(X, k, t):
    Y = X[:, t].copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1])
    F = np.stack([np.concatenate([np.interp(gr24, np.linspace(0, 1, Y.shape[1]), Y[i, :, v]) for v in range(3)]) for i in range(len(Y))])
    M = np.stack([np.nanmean(F[k == p], 0) for p in P]); return (M - M.mean(0)) / (M.std(0) + 1e-12)
FE = {}
for m in MODELS:
    g, meta, X, s = A[m]; k = meta.pt.astype(str).str.strip().values; T = X.shape[1]
    for tau, t in (('first', 0), ('middle', T // 2), ('last', T - 1)): FE[(m, tau)] = feats_arr(X, k, t)
for sub in ('attn', 'mlp'):
    X = X_sd(sub); k = np.char.strip(pt); T = X.shape[1]
    for tau, t in (('first', 0), ('middle', T // 2), ('last', T - 1)): FE[(f'SD15{sub}', tau)] = feats_arr(X, k, t)
iu = np.triu_indices(NP, 1)
def rdm(F): return 1 - np.corrcoef(F)
def mantel(Am, Bm):
    a = rankdata(Am[iu]); b = rankdata(Bm[iu]); r = np.corrcoef(a, b)[0, 1]; null = np.empty(NPERM)
    for i in range(NPERM):
        p = rng.permutation(NP); null[i] = np.corrcoef(a, rankdata(Bm[np.ix_(p, p)][iu]))[0, 1]
    return float(r), float((1 + (null >= r).sum()) / (NPERM + 1)), float(np.percentile(null, 95))
RD = {kk: rdm(v) for kk, v in FE.items()}; rows = []
pairs = [(f'SD15{s}', m) for s in ('attn', 'mlp') for m in MODELS] + [(a, b) for i, a in enumerate(MODELS) for b in MODELS[i + 1:]] + [('SD15attn', 'SD15mlp')]
for a, b in pairs:
    for tau in ('first', 'middle', 'last'):
        r, p, n95 = mantel(RD[(a, tau)], RD[(b, tau)]); rows.append(dict(a=a, b=b, tau=tau, rsa=r, p=p, null_p95=n95, n_prompts=NP))
P6 = pd.DataFrame(rows); P6.to_csv(f'{BASE}/out/P6_rsa.csv', index=False)
pd.set_option('display.width', 250); print(P6.pivot_table(index=['a', 'b'], columns='tau', values='rsa').round(3).to_string()); print('max p', P6.p.max())
