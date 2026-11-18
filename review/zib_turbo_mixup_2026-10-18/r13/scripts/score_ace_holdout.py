import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Round 13 HOLDOUT SCORING for Zac's ACE question (EXPLORE prompts only; descriptive).
import os, sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round6/scripts')
from r6c_common import tangent
SCR = os.environ.get('SCR', _COSPAFS_CACHE); L = lambda n: pickle.load(open(f'{SCR}/F_{n}.pkl', 'rb'))
def pp(n):
    """per-prompt mean over gens and steps of layer-mean log g (scalar per prompt) and per-prompt depth profile."""
    _, m, X, s = L(n); k = m.pt.astype(str).str.strip().values
    df = pd.DataFrame(np.log(X[..., 0]).mean(1), index=k)  # [G, L] step-averaged log g by layer
    P = df.groupby(level=0).mean(); return P.mean(1), P, m.groupby(m.pt.astype(str).str.strip()).dom.first()
rng = np.random.default_rng(0)
for sub in ('attn', 'mlp'):
    a_s, a_P, dom = pp(f'ACEMH_{sub}')
    print(f'== {sub}  ACE turbo on image prompts: {len(a_s)} prompts')
    others = ['Z1H', 'Z3H', 'ZITH'] + (['SD35MH', 'SD35LH', 'MOCHIH'] if sub == 'attn' else [])
    for o in others:
        o_s, o_P, _ = pp(f'{o}_{sub}'); c = a_s.index.intersection(o_s.index)
        r = np.corrcoef(a_s[c], o_s[c])[0, 1]; rs = pd.Series(a_s[c]).rank().corr(pd.Series(o_s[c]).rank())
        _, _, XA, sa = L(f'ACEMH_{sub}'); _, _, XB, sb = L(f'{o}_{sub}'); tc = tangent(XA, sa, XB, sb)[0]
        print(f'  vs {o:7s} prompts {len(c)}  per-prompt level r {r:+.3f} (spearman {rs:+.3f})   depth tangent {tc:+.3f}')
    # domain structure inside ACE: share of per-prompt level variance explained by domain
    d = pd.DataFrame(dict(y=a_s, dom=dom.reindex(a_s.index))); eta = ((d.groupby('dom').y.transform('mean') - d.y.mean()) ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum()
    print(f'  ACE domain share of per-prompt level: {eta:.3f}')
    _, _, XA, sa = L(f'ACEMH_{sub}'); _, _, XB, sb = L(f'ACEAUD_{sub}')
    print(f'  ACE image prompts vs ACE audio prompts (same weights): tangent {tangent(XA, sa, XB, sb)[0]:+.3f}; mean log g image {np.log(XA[...,0]).mean():.3f} audio {np.log(XB[...,0]).mean():.3f}')
    # prompt length effect
    _, m, X, s = L(f'ACEMH_{sub}'); y = np.log(X[..., 0]).mean((1, 2)); print(f'  ACE level vs prompt_len r {np.corrcoef(y, m.pl.values)[0,1]:+.3f}')
    for o in ('Z1H',):
        _, m2, X2, _ = L(f'{o}_{sub}'); y2 = np.log(X2[..., 0]).mean((1, 2)); print(f'  {o} level vs prompt_len r {np.corrcoef(y2, m2.pl.values)[0,1]:+.3f}')
