# RC24 (post hoc, round-5 review DeepSeek Q10): does residual vector-field transfer fail between unrelated models when depth is
# matched? A RE-IMPLEMENTATION of the D15 design (the ledger's script is not in this repository): state z = (log g, log gap, H) per
# generation, layer and sampling state, standardised per model; the depth step dz = z(l+1) - z(l); scaffold F0(u, sigma) = the
# population mean of dz at each (layer, state); residual R = dz - F0. A gradient-boosted regressor predicts R from
# (z - mean z at (l, state), u, sigma). Self: prompt-grouped 5-fold CV R^2 on the target. Cross: fitted on all of the source,
# applied to the target (target's own scaffold subtracted). Pairs are depth-matched (same number of hooked layers).
import os, sys, pickle, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
CA, R16, R17 = sys.argv[1:4]; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
SRC = {'SD35M17': (R17, 'SD35M'), 'SD3M': (R17, 'SD3M'), 'PIXART': (R17, 'PIXART'), 'COSMOS2B': (R17, 'COSMOS2B'), 'T8': (R16, 'T8')}
TCLK = {'PIXART', 'COSMOS2B'}
def load(n, sub):
    d, f = SRC.get(n, (CA, n)); _, m, X, s = pickle.load(open(f'{d}/F_{f}_{sub}.pkl', 'rb')); X = np.asarray(X, float); s = np.asarray(s, float)
    if n in TCLK: s = s / (1 + s)
    Z = np.stack([np.log(X[..., 0]), np.log(X[..., 1]), X[..., 2]], -1); Z = (Z - Z.mean((0, 1, 2))) / Z.std((0, 1, 2))
    return Z, s / s.max(), m.pt.astype(str).values
def design(Z, s):
    G, T, L, _ = Z.shape; dz = np.diff(Z, axis=2); F0 = dz.mean(0, keepdims=True); R = dz - F0; zc = Z[:, :, :-1] - Z[:, :, :-1].mean(0, keepdims=True)
    u = np.broadcast_to((np.arange(L - 1) / (L - 2))[None, None, :, None], (G, T, L - 1, 1)); tau = np.broadcast_to(s[None, :, None, None], (G, T, L - 1, 1))
    Xd = np.concatenate([zc, u, tau], -1).reshape(G, -1, 5); return Xd, R.reshape(G, -1, 3)
def r2(y, p): return float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
def fit(X, Y): return [HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X, Y[:, k]) for k in range(3)]
def pred(M, X): return np.stack([m.predict(X) for m in M], 1)
_C = {}
def prep(n, sub):
    if (n, sub) not in _C:
        Z, s, pt = load(n, sub); Xd, Y = design(Z, s); _C[(n, sub)] = (Xd, Y, pt)
    return _C[(n, sub)]
def self_r2(n, sub):
    Xd, Y, pt = prep(n, sub); grp = pd.factorize(pt)[0]; P = np.zeros_like(Y)
    for tr, te in GroupKFold(5).split(Xd, groups=grp):
        M = fit(Xd[tr].reshape(-1, 5), Y[tr].reshape(-1, 3)); P[te] = pred(M, Xd[te].reshape(-1, 5)).reshape(len(te), -1, 3)
    return r2(Y.reshape(-1, 3), P.reshape(-1, 3))
def cross_r2(a, b, sub):
    Xa, Ya, _ = prep(a, sub); Xb, Yb, _ = prep(b, sub); M = fit(Xa.reshape(-1, 5), Ya.reshape(-1, 3)); return r2(Yb.reshape(-1, 3), pred(M, Xb.reshape(-1, 5)))
PAIRS = [('same weights', 'Z3', 'ZIT', 'attn'), ('same weights', 'Z3', 'ZIT', 'mlp'), ('same weights', 'T8', 'ACET', 'attn'), ('same weights', 'T8', 'ACET', 'mlp'),
         ('lineage', 'Z1', 'ZIT', 'attn'), ('lineage', 'Z1', 'ZIT', 'mlp'), ('lineage', 'ACEB', 'ACET', 'attn'), ('lineage', 'ACEB', 'ACET', 'mlp'),
         ('successor', 'SD3M', 'SD35M17', 'attn'),
         ('unrelated, depth-matched', 'PIXART', 'COSMOS2B', 'attn'), ('unrelated, depth-matched', 'PIXART', 'COSMOS2B', 'mlp'),
         ('unrelated, depth-matched', 'ACET', 'SD35M17', 'attn'), ('unrelated, depth-matched', 'ACEB', 'SD3M', 'attn')]
rows = []; S = {}
for kind, a, b, sub in PAIRS:
    for src, tgt in [(a, b), (b, a)]:
        if (tgt, sub) not in S: S[(tgt, sub)] = self_r2(tgt, sub)
        c = cross_r2(src, tgt, sub); rows.append(dict(relation=kind, source=src, target=tgt, sub=sub, self=S[(tgt, sub)], cross=c, shortfall=S[(tgt, sub)] - c)); print(rows[-1], flush=True)
D = pd.DataFrame(rows); D.to_csv(f'{OUT}/rc24_depth_matched_transfer.csv', index=False)
print(D.round(3).to_string(index=False)); print(D.groupby('relation', sort=False)[['self', 'cross', 'shortfall']].mean().round(3))
