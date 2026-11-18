# RC5b (post hoc): does a depth low-pass in normalised-depth units make the tangent depth-invariant, and what are the tiers then?
# Each standardised profile is smoothed across depth with a Gaussian kernel of bandwidth h (fraction of depth) before the usual
# 24-point tangent. Self-controls: split-half copy of a model against a decimated or resampled copy of itself.
import os, sys, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R15 = os.path.dirname(HERE); ROOT = os.path.abspath(os.path.join(R15, '..', '..'))
SCR = os.environ.get('SCR', os.path.join(ROOT, 'cache')); OUT = os.path.join(HERE, 'out')
sys.path.insert(0, os.path.join(ROOT, 'review', 'round5', 'scripts')); os.environ.setdefault('SCR', SCR)
from r5_common import depthnorm, cos
GR = np.linspace(0, 1, 24); C = {}
def F(n):
    if n not in C:
        g, m, X, s = pickle.load(open(f'{SCR}/F_{n}.pkl', 'rb')); C[n] = (np.asarray(m.pt.astype(str).str.strip().tolist()), X.astype(float), np.asarray(s, float))
    return C[n]
def prof(X, s, h):
    D = depthnorm(X.mean(0)); L = D.shape[1]; u = np.linspace(0, 1, L)
    if h > 0:
        W = np.exp(-0.5 * ((GR[:, None] - u[None, :]) / h) ** 2); W /= W.sum(1, keepdims=True)
        return np.einsum('gl,tlk->tgk', W, D)
    return np.stack([[np.interp(GR, u, D[t, :, j]) for j in range(3)] for t in range(len(s))]).transpose(0, 2, 1)
def tangent(XA, sa, XB, sb, h):
    A = prof(XA, sa, h); B = prof(XB, sb, h)
    return float(np.mean([cos(np.diff(A[t], axis=0), np.diff(B[int(np.argmin(np.abs(sb - x)))], axis=0)) for t, x in enumerate(sa)]))
def decimate(X, Lp): return X[:, :, np.unique(np.round(np.linspace(0, X.shape[2] - 1, Lp)).astype(int)), :]
rng = np.random.default_rng(56); H = (0, 0.04, 0.06, 0.08, 0.12)
rows = []
for name, Lp in [('MOCHI_attn', 38), ('MOCHI_attn', 30), ('MOCHI_attn', 24), ('SD35L_attn', 30), ('SD35L_attn', 24), ('ZIT_attn', 24), ('ZIT_mlp', 24), ('Z1_attn', 24), ('SD35M_attn', 16), ('ACET_attn', 16)]:
    P, X, s = F(name); u = np.unique(P); rng.shuffle(u); m = np.isin(P, u[: len(u) // 2])
    for h in H: rows.append(dict(kind='self-control', a=name, b=f'{name} decimated to {Lp}', h=h, tangent=tangent(X[m], s, decimate(X[~m], Lp), s, h)))
PAIRS = [('same', 'ZIT_attn', 'ZIB_attn'), ('same', 'ZIT_mlp', 'ZIB_mlp'), ('same', 'ACEB0_attn', 'ACEB1_attn'), ('same', 'ACEB0_mlp', 'ACEB1_mlp'),
         ('lineage', 'Z1_attn', 'ZIT_attn'), ('lineage', 'Z1_mlp', 'ZIT_mlp'), ('lineage', 'ACEB_attn', 'ACET_attn'), ('lineage', 'ACEB_mlp', 'ACET_mlp'),
         ('unrelated', 'ZIT_attn', 'MOCHI_attn'), ('unrelated', 'ZIT_attn', 'SD35M_attn'), ('unrelated', 'ZIT_attn', 'SD35L_attn'), ('unrelated', 'Z1_attn', 'MOCHI_attn'),
         ('unrelated', 'Z1_attn', 'SD35M_attn'), ('unrelated', 'Z1_attn', 'SD35L_attn'), ('unrelated', 'MOCHI_attn', 'SD35M_attn'), ('unrelated', 'MOCHI_attn', 'SD35L_attn'),
         ('unrelated', 'SD35M_attn', 'SD35L_attn'), ('unrelated', 'ACET_attn', 'ZIT_attn'), ('unrelated', 'ACEB_attn', 'Z1_attn'), ('unrelated', 'ACEB_mlp', 'Z1_mlp'),
         ('unrelated', 'ACET_mlp', 'ZIT_mlp'), ('unrelated', 'ACET_attn', 'SD35M_attn'), ('unrelated', 'ACEB_attn', 'SD35M_attn')]
for k, a, b in PAIRS:
    _, XA, sa = F(a); _, XB, sb = F(b)
    for h in H:
        t = tangent(XA, sa, XB, sb, h)
        nb = [tangent(XA, sa, XB[:, :, rng.permutation(XB.shape[2])], sb, h) for _ in range(200)] if k != 'same' else [np.nan]
        rows.append(dict(kind=k, a=a, b=b, h=h, tangent=t, perm_p95=float(np.nanpercentile(nb, 95)) if k != 'same' else np.nan))
D = pd.DataFrame(rows); D.to_csv(f'{OUT}/rc5b_lowpass.csv', index=False)
print(D.pivot_table(index=['kind', 'a', 'b'], columns='h', values='tangent', sort=False).round(3).to_string(), flush=True)
print(D.pivot_table(index=['kind', 'a', 'b'], columns='h', values='perm_p95', sort=False).round(3).to_string())
S = D[D.kind != 'self-control'].groupby(['h', 'kind']).tangent.agg(['min', 'max']).unstack(); print(S.round(3).to_string())
