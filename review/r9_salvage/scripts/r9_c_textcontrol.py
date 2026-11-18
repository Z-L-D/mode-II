"""R9-B6b text control (R9_ADDENDUM_B6b.md, SHA cdb6f994, frozen 2026-10-09T09:30:44Z). Features, residualisation and pairs as R9-B6
(r9_b_llm.py, copied unchanged); TF-IDF vectoriser as Round 6 S2f. Output: out/R9_B6b_text_control.csv, out/R9_B6b_results.json"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import sys, os, glob, json, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
SCR = os.environ.get('SCR', f'{_COSPAFS_CACHE}'); PQ = f'{SCR}/pq'
OUT = f'{_COSPAFS_ROOT}/review/r9_salvage/out'; PSEED = 2026100908; NPERM = 10000; grid = np.linspace(0, 1, 24)
def load(name):
    d = pd.read_parquet(glob.glob(f'{PQ}/LLM_{name}_*.parquet')[0])
    if 'submodule' in d: d = d[d.submodule == 'attn']
    d['prompt_text'] = d.groupby('gen_idx').prompt_text.transform('first'); return d
def prof_at(d, k):
    r = d[d.token_idx == k]; out = {}; nn = {}
    for p, dd in r.groupby('prompt_text'):
        F = []
        for g, gg in dd.groupby('gen_idx'):
            Y = gg.sort_values('layer')[['g_micro', 'spectral_gap', 'sv_entropy']].values.astype(float); Y[:, 0] = np.log(Y[:, 0]); Y[:, 1] = np.log(Y[:, 1])
            u = np.linspace(0, 1, len(Y)); F.append(np.concatenate([np.interp(grid, u, Y[:, v]) for v in range(3)]))
        out[p] = np.mean(F, 0); nn[p] = float(dd.seq_len.iloc[0])
    return out, nn
def resid_feats(F, n, P):
    M = np.stack([F[p] for p in P]); X = np.c_[np.ones(len(P)), np.log([n[p] for p in P])]
    R = M - X @ np.linalg.lstsq(X, M, rcond=None)[0]; return (R - R.mean(0)) / (R.std(0) + 1e-12)
rdm = lambda F: 1 - np.corrcoef(F)
def textdist(P):
    V = TfidfVectorizer().fit_transform(P); return 1 - (V @ V.T).toarray()
def partial_multi(RA, RB, Cs, rng_):
    iu = np.triu_indices(RA.shape[0], 1); Z = np.c_[np.ones(len(iu[0])), np.stack([rankdata(C[iu]) for C in Cs], 1)]
    H = Z @ np.linalg.pinv(Z); res = lambda v: v - H @ v
    a = res(rankdata(RA[iu])); r0 = lambda b: float(np.corrcoef(a, res(b))[0, 1])
    obs = r0(rankdata(RB[iu])); NP = RA.shape[0]; nul = np.empty(NPERM)
    for i in range(NPERM):
        pm = rng_.permutation(NP); nul[i] = r0(rankdata(RB[np.ix_(pm, pm)][iu]))
    return obs, float((1 + (nul >= obs).sum()) / (NPERM + 1))
def mantel1(RA, C):
    iu = np.triu_indices(RA.shape[0], 1); return float(np.corrcoef(rankdata(RA[iu]), rankdata(C[iu]))[0, 1])
QN = ['qwen25_1_5b', 'qwen25_3b', 'qwen25_7b']
models = {k: load(k) for k in QN}; models['mistral_instruct'] = load('mistral-03-7B-instruct'); models['llama31_instruct'] = load('llama-31-8B-instruct')
PRO = {(m, k): prof_at(d, k) for m, d in models.items() for k in (0, 128, 255)}
B6 = pd.read_csv(f'{OUT}/R9_B6_llm_rsa.csv').set_index(['a', 'b', 'k']).resid_partial
rr = np.random.default_rng(PSEED); rows = []
for a, b in [('qwen25_1_5b', 'qwen25_3b'), ('qwen25_1_5b', 'qwen25_7b'), ('qwen25_3b', 'qwen25_7b'), ('mistral_instruct', 'llama31_instruct')]:
    for k in (0, 128, 255):
        (Fa, na), (Fb, nb) = PRO[(a, k)], PRO[(b, k)]; P = sorted(set(Fa) & set(Fb))
        RA = rdm(resid_feats(Fa, na, P)); RB = rdm(resid_feats(Fb, nb, P))
        ln = np.array([(np.log(na[p]) + np.log(nb[p])) / 2 for p in P]); CL = np.abs(ln[:, None] - ln[None, :]); CT = textdist(P)
        part, p = partial_multi(RA, RB, [CL, CT], rr); prev = float(B6[(a, b, k)])
        rows.append(dict(substrate='LLM', a=a, b=b, k=k, prompts=len(P), partial_len_only=prev, partial_len_text=part, p=p, ratio=part / prev,
                         text_rsa_a=mantel1(RA, CT), text_rsa_b=mantel1(RB, CT)))
        print(rows[-1], flush=True)
# like-for-like diffusion comparison: S2 first-state features, unrelated pairs, controls = |len chars| + TF-IDF
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round6/scripts')
from r6c_common import arch, MODELS, E10_UNREL, gr24
D = {m: arch(m) for m in MODELS}
P = sorted(set.intersection(*[set(D[m][1].pt.astype(str).str.strip()) for m in MODELS]))
def feats(m, t):   # r6c_S2_content.py feats, unchanged
    g, meta, X, s = D[m]; k = meta.pt.astype(str).str.strip().values
    Y = X[:, t].copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1])
    F = np.stack([np.concatenate([np.interp(gr24, np.linspace(0, 1, Y.shape[1]), Y[i, :, v]) for v in range(3)]) for i in range(len(Y))])
    M = np.stack([F[k == p].mean(0) for p in P]); return (M - M.mean(0)) / (M.std(0) + 1e-12)
lc = np.array([len(p) for p in P], float); CLd = np.abs(lc[:, None] - lc[None, :]); CTd = textdist(P)
RDd = {m: rdm(feats(m, 0)) for m in MODELS}
for a, b in E10_UNREL:
    part, p = partial_multi(RDd[a], RDd[b], [CLd, CTd], rr)
    iu = np.triu_indices(len(P), 1); full = float(np.corrcoef(rankdata(RDd[a][iu]), rankdata(RDd[b][iu]))[0, 1])
    rows.append(dict(substrate='diffusion', a=a, b=b, k='first', prompts=len(P), partial_len_only=full, partial_len_text=part, p=p, ratio=part / full,
                     text_rsa_a=mantel1(RDd[a], CTd), text_rsa_b=mantel1(RDd[b], CTd)))
    print(rows[-1], flush=True)
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/R9_B6b_text_control.csv', index=False)
res = {}
for (a, b), d in T[T.substrate == 'LLM'].groupby(['a', 'b'], sort=False):
    ok = int(((d.p <= 0.01) & (d.ratio >= 0.5)).sum()); low = int((d.ratio < 0.5).sum())
    res[f'{a}~{b}'] = dict(verdict='survives the text control' if ok >= 2 else ('mostly text' if low >= 2 else 'tested, not supported'),
                          partial=d.partial_len_text.round(3).tolist(), ratio=d.ratio.round(3).tolist(), p=d.p.round(4).tolist())
dd = T[T.substrate == 'diffusion']; res['diffusion_unrelated_first'] = dict(partial_range=[float(dd.partial_len_text.min()), float(dd.partial_len_text.max())],
                                                                             ratio_range=[float(dd.ratio.min()), float(dd.ratio.max())], max_p=float(dd.p.max()))
json.dump(res, open(f'{OUT}/R9_B6b_results.json', 'w'), indent=1); print(json.dumps(res, indent=1))
