# RC16 (post hoc, round-2 review Qwen A / DeepSeek 6): prompt RSA (Mantel on prompt-dissimilarity matrices of
# standardised per-prompt mean depth profiles, as in R15-7) between the Round 11 captures of Z-Image Base (CFG 1),
# Z-Image Turbo and SD3.5 Medium, on every row set: all rows, image rows, caption rows, centred, top-8 removed.
# Question: does the shared prompt geometry survive when caption rows are excluded or the mean direction is removed?
import os, sys, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R15 = os.path.dirname(HERE); ROOT = os.path.abspath(os.path.join(R15, '..', '..'))
SCR = os.environ.get('SCR', os.path.join(ROOT, 'cache')); OUT = os.path.join(HERE, 'out')
sys.path.insert(0, os.path.join(R15, 'scripts'))
from scipy.stats import rankdata
def feat(X, t): return np.concatenate([np.log(np.abs(X[:, t, :, 0]) + 1e-12), np.log(np.abs(X[:, t, :, 1]) + 1e-12), X[:, t, :, 2]], 1)
def pmeans(k, Fm, P):
    M = np.stack([Fm[k == p].mean(0) for p in P]); sd = M.std(0); return (M - M.mean(0)) / (sd + 1e-12)
def rsa(A, B, nperm=1999, seed=15):
    iu = np.triu_indices(len(A), 1); RA = 1 - np.corrcoef(A); RB = 1 - np.corrcoef(B)
    if not (np.isfinite(RA).all() and np.isfinite(RB).all()): return np.nan, np.nan
    a = rankdata(RA[iu]); b = rankdata(RB[iu]); r = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(seed); n = len(A); nul = np.array([np.corrcoef(a, rankdata(RB[np.ix_(p, p)][iu]))[0, 1] for p in (rng.permutation(n) for _ in range(nperm))])
    return r, float((1 + (nul >= r).sum()) / (nperm + 1))
SETS = {'image rows': ['g_img', 'gap_img', 'H_img'], 'caption rows': ['g_txt', 'gap_txt', 'H_txt'], 'centred': ['g_ctr', 'gap_ctr', 'H_ctr'], 'top-8 removed': ['g_topk', 'gap_topk', 'H_topk']}
def load(n, rs):
    if rs == 'all rows':
        g, m, X, s = pickle.load(open(f'{SCR}/F_{n}.pkl', 'rb')); return np.asarray(m.pt.astype(str).str.strip().tolist()), X.astype(float), np.asarray(s, float)
    o = pickle.load(open(f'{SCR}/V_{n}.pkl', 'rb')); idx = [o['cols'].index(c) for c in SETS[rs]]
    return np.asarray(o['meta'].pt.astype(str).str.strip().tolist()), o['V'][..., idx].astype(float), np.asarray(o['s'], float)
PAIRS = [('lineage', 'Z2_attn', 'ZITSV_attn'), ('unrelated', 'Z2_attn', 'SD3SV_attn'), ('unrelated', 'ZITSV_attn', 'SD3SV_attn'), ('lineage', 'Z2_mlp', 'ZITSV_mlp')]
rows = []
for rs in ['all rows'] + list(SETS):
    for kind, a, b in PAIRS:
        try: ka, XA, sa = load(a, rs); kb, XB, sb = load(b, rs)
        except (FileNotFoundError, ValueError): continue
        if rs == 'caption rows' and 'SD3' in b: continue  # SD3.5 M caption rows are a separate stream
        P = sorted(set(ka) & set(kb))
        for lab, t in [('first', 0), ('middle', len(sa) // 2), ('last', len(sa) - 1)]:
            j = int(np.argmin(np.abs(sb - sa[t])))
            A = pmeans(ka, feat(XA, t), P); B = pmeans(kb, feat(XB, j), P); r, p = rsa(A, B)
            rows.append(dict(rows=rs, kind=kind, a=a, b=b, step=lab, sigma_a=round(float(sa[t]), 3), sigma_b=round(float(sb[j]), 3), prompts=len(P), rsa=r, p=p))
        print(rows[-1], flush=True)
D = pd.DataFrame(rows); D.to_csv(f'{OUT}/rc16_rowset_rsa.csv', index=False)
print(D.pivot_table(index=['kind', 'a', 'b'], columns=['rows', 'step'], values='rsa').round(2).to_string())
