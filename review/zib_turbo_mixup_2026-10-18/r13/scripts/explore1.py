import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Round 13 EXPLORATION (EXPLORE prompts only; descriptive, not evidence).
import os, pickle, numpy as np, pandas as pd
SCR = os.environ.get('SCR', _COSPAFS_CACHE); np.set_printoptions(precision=3, suppress=True, linewidth=220)
L = lambda n: pickle.load(open(f'{SCR}/F_{n}.pkl', 'rb'))
def pshare(X, meta):
    """per step: share of between-gen variance of layer-mean log g explained by prompt (one-way ANOVA eta^2)."""
    y = np.log(X[..., 0]).mean(2); k = meta.pt.astype(str).str.strip().values; out = []
    for t in range(y.shape[1]):
        d = pd.DataFrame(dict(y=y[:, t], k=k)); tot = ((d.y - d.y.mean()) ** 2).sum(); b = d.groupby('k').y.transform('mean'); out.append(((b - d.y.mean()) ** 2).sum() / tot)
    return np.array(out)
print('== A. prompt share of layer-mean log g by step (4 seeds/prompt)')
for n in ('Z1E_attn', 'Z1E_mlp', 'ZITE_attn', 'ZITE_mlp', 'ZIBE_attn', 'ZIBE_mlp'):
    g, meta, X, s = L(n); print(n, 'sigma', np.round(s[[0, len(s)//4, len(s)//2, 3*len(s)//4, -1]], 3), 'share', pshare(X, meta)[[0, len(s)//4, len(s)//2, 3*len(s)//4, -1]])
print('== B. CFG signature: Z1E cond - uncond mean log g, by layer (avg over steps) and by step (avg over layers)')
for sub in ('attn', 'mlp'):
    _, mc, Xc, s = L(f'Z1E_{sub}'); _, mu, Xu, _ = L(f'Z1UE_{sub}')
    kc = mc.index.str.rsplit(':', n=1).str[0]; ku = mu.index.str.rsplit(':', n=1).str[0]  # file:gen
    common = sorted(set(kc) & set(ku)); ic = [list(kc).index(c) for c in common]; iu = [list(ku).index(c) for c in common]
    D = np.log(Xc[ic, ..., 0]) - np.log(Xu[iu, ..., 0])
    print(sub, 'n', len(common), '\n by layer', D.mean((0, 1)), '\n by step ', D.mean((0, 2)))
print('== C. Base (Z1E cond) - Turbo matched (Z3E cond), mean log g, identical settings')
for sub in ('attn', 'mlp'):
    _, _, B, s = L(f'Z1E_{sub}'); _, _, T, _ = L(f'Z3E_{sub}')
    D = np.log(B[..., 0]).mean(0) - np.log(T[..., 0]).mean(0)
    print(sub, '\n by layer', D.mean(0), '\n by step ', D.mean(1))
    for k, nm in ((1, 'gap'), (2, 'H')):
        Dk = (np.log(B[..., k]) if k == 1 else B[..., k]).mean(0) - (np.log(T[..., k]) if k == 1 else T[..., k]).mean(0)
        print(f'  {nm} by layer', Dk.mean(0))
