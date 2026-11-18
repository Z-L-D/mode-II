# Review round 3 checks (post hoc, existing data, 2026-11-06).
# RC18 per-coordinate tangents (GLM A6, DeepSeek): the tangent on log g, log gap and H separately, with a layer-permutation p95 each.
# RC19 identity gate on g instead of log g, and per coordinate (DeepSeek Q7).
# RC20 relative-onset threshold sensitivity, 0.3-0.7 instead of 1/2 (DeepSeek Q8), 7 archive channels, native n.
# RC21 pooling increment by resolution: generation-bootstrap intervals and the 512-1024 difference (DeepSeek Q5).
import os, sys, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R15 = os.path.dirname(HERE); ROOT = os.path.abspath(os.path.join(R15, '..', '..'))
SCR = os.environ.get('SCR', os.path.join(ROOT, 'cache')); OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
os.environ.setdefault('SCR', SCR); sys.path.insert(0, os.path.join(ROOT, 'review', 'round5', 'scripts'))
from r5_common import depthnorm, cos, R3, R5, BM
rng = np.random.default_rng(18)
def F(n): g, m, X, s = pickle.load(open(f'{SCR}/F_{n}.pkl', 'rb')); return X.astype(float), np.asarray(s, float)
gr = np.linspace(0, 1, 24)
def prof(X, s):
    D = depthnorm(X.mean(0)); u = np.linspace(0, 1, D.shape[1])
    return np.stack([[np.interp(gr, u, D[t, :, k]) for k in range(3)] for t in range(len(s))]).transpose(0, 2, 1)
def tan_k(A, sa, B, sb, ks):
    return float(np.mean([cos(np.diff(A[t][:, ks], axis=0), np.diff(B[int(np.argmin(np.abs(sb - x)))][:, ks], axis=0)) for t, x in enumerate(sa)]))
PAIRS = [('same', 'ZIT_attn', 'ZIB_attn'), ('same', 'ZIT_mlp', 'ZIB_mlp'), ('same', 'ACEB0_attn', 'ACEB1_attn'), ('same', 'ACEB0_mlp', 'ACEB1_mlp'),
         ('lineage', 'Z1_attn', 'ZIT_attn'), ('lineage', 'Z1_mlp', 'ZIT_mlp'), ('lineage', 'ACEB_attn', 'ACET_attn'), ('lineage', 'ACEB_mlp', 'ACET_mlp'),
         ('unrelated (depth-matched)', 'ACET_attn', 'SD35M_attn'), ('unrelated (depth-matched)', 'ACEB_attn', 'SD35M_attn')]
COORD = {'all three': [0, 1, 2], 'log g': [0], 'log gap': [1], 'H': [2]}
C = {}
def cube(n):
    if n not in C: C[n] = F(n)
    return C[n]
rows = []
for kind, a, b in PAIRS:
    XA, sa = cube(a); XB, sb = cube(b); A = prof(XA, sa); B = prof(XB, sb); L = XB.shape[2]
    perms = [rng.permutation(L) for _ in range(300)]; Bp = [prof(XB[:, :, p], sb) for p in perms]
    for lab, ks in COORD.items():
        nul = [tan_k(A, sa, P, sb, ks) for P in Bp]
        rows.append(dict(kind=kind, a=a, b=b, coord=lab, tangent=tan_k(A, sa, B, sb, ks), perm_p95=float(np.percentile(nul, 95))))
R18 = pd.DataFrame(rows); R18.to_csv(f'{OUT}/rc18_per_coordinate.csv', index=False)
print(R18.pivot_table(index=['kind', 'a', 'b'], columns='coord', values='tangent', sort=False).round(3).to_string(), flush=True)
# ---------------- RC19: gate statistic ----------------
G = pd.read_csv(f'{OUT}/rc10_identity_gate.csv'); G = G[G.n_gen.isna() & G.a.isin(['ZIB', 'Z3', 'ZITSV', 'Z1', 'Z2', 'ACEB0', 'ACEB', 'MOCHI', 'SD35M', 'ACET'])]
rows = []
for r in G.itertuples():
    try: XA, _ = cube(f'{r.a}_{r.sub}'); XB, _ = cube(f'{r.b}_{r.sub}')
    except FileNotFoundError: continue
    if XA.shape[2] != XB.shape[2]: continue
    d = dict(kind=r.kind, pair=r.pair, sub=r.sub, r_log_g_published=r.r)
    for lab, f in [('log_g', lambda X: np.log(X[:, 0, :, 0]).mean(0)), ('g', lambda X: X[:, 0, :, 0].mean(0)),
                   ('log_gap', lambda X: np.log(X[:, 0, :, 1]).mean(0)), ('H', lambda X: X[:, 0, :, 2].mean(0))]:
        d[f'r_{lab}'] = float(np.corrcoef(f(XA), f(XB))[0, 1])
    rows.append(d)
R19 = pd.DataFrame(rows); R19.to_csv(f'{OUT}/rc19_gate_statistic.csv', index=False); print(R19.round(3).to_string(index=False), flush=True)
for lab in ('log_g', 'g', 'log_gap', 'H'):
    s_ = R19[R19.kind == 'same'][f'r_{lab}']; d_ = R19[R19.kind == 'different'][f'r_{lab}']
    print(f'RC19 {lab}: same min {s_.min():.3f}, different max {d_.max():.3f}, separated {s_.min() > d_.max()}')
