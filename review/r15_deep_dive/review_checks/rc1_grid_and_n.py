# Review checks (post hoc, 2026-11-05; requested by the DeepSeek and Qwen reviews).
# RC1 tangent sensitivity to the depth grid size (12/24/48) and interpolation (linear vs cubic spline).
# RC2 generations needed: tangent from n-generation subsets (n = 3, 5, 10, 20), 100 draws, per tier.
# RC3 transfer of a tier threshold fitted on Z-Image pairs to ACE-Step and LLM pairs.
import os, sys, pickle, numpy as np, pandas as pd
from scipy.interpolate import CubicSpline
HERE = os.path.dirname(os.path.abspath(__file__)); R15 = os.path.dirname(HERE); ROOT = os.path.abspath(os.path.join(R15, '..', '..'))
SCR = os.environ.get('SCR', os.path.join(ROOT, 'cache')); OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, 'review', 'round5', 'scripts')); os.environ.setdefault('SCR', SCR)
from r5_common import depthnorm, cos
def F(n): g, m, X, s = pickle.load(open(f'{SCR}/F_{n}.pkl', 'rb')); return X.astype(float), np.asarray(s, float)
def L(n): g, m, X, s = pickle.load(open(os.path.join(R15, 'out', 'llm_cubes', f'L_{n}.pkl'), 'rb')); return X.astype(float), np.asarray(s, float)
def prof(X, s, ng, kind):
    D = depthnorm(X.mean(0)); u = np.linspace(0, 1, D.shape[1]); gr = np.linspace(0, 1, ng)
    f = (lambda y: np.interp(gr, u, y)) if kind == 'linear' else (lambda y: CubicSpline(u, y)(gr))
    return np.stack([[f(D[t, :, k]) for k in range(3)] for t in range(len(s))]).transpose(0, 2, 1)
def tangent(XA, sa, XB, sb, ng=24, kind='linear'):
    A = prof(XA, sa, ng, kind); B = prof(XB, sb, ng, kind)
    return float(np.mean([cos(np.diff(A[t], axis=0), np.diff(B[int(np.argmin(np.abs(sb - x)))], axis=0)) for t, x in enumerate(sa)]))
PAIRS = [('same', 'ZIT_attn', 'ZIB_attn'), ('same', 'ZIT_mlp', 'ZIB_mlp'), ('lineage', 'Z1_attn', 'ZIT_attn'), ('lineage', 'Z1_mlp', 'ZIT_mlp'),
         ('lineage', 'ACEB_attn', 'ACET_attn'), ('lineage', 'ACEB_mlp', 'ACET_mlp'), ('unrelated', 'ZIT_attn', 'SD35M_attn'), ('unrelated', 'Z1_attn', 'MOCHI_attn'),
         ('unrelated', 'ZIT_attn', 'MOCHI_attn'), ('unrelated', 'ACET_attn', 'ZIT_attn'), ('unrelated', 'ACEB_mlp', 'Z1_mlp')]
C = {}
def cube(n):
    if n not in C: C[n] = F(n)
    return C[n]
rows = []
for k, a, b in PAIRS:
    XA, sa = cube(a); XB, sb = cube(b)
    for ng in (12, 24, 48):
        for kind in ('linear', 'cubic'):
            rows.append(dict(kind=k, a=a, b=b, grid=ng, interp=kind, tangent=tangent(XA, sa, XB, sb, ng, kind)))
R1 = pd.DataFrame(rows); R1.to_csv(f'{OUT}/rc1_grid_interp.csv', index=False)
print(R1.pivot_table(index=['kind', 'a', 'b'], columns=['grid', 'interp'], values='tangent').round(3).to_string(), flush=True)
rng = np.random.default_rng(16); rows = []
for k, a, b in PAIRS:
    XA, sa = cube(a); XB, sb = cube(b)
    for n in (3, 5, 10, 20):
        v = [tangent(XA[rng.choice(len(XA), n, replace=False)], sa, XB[rng.choice(len(XB), n, replace=False)], sb) for _ in range(100)]
        rows.append(dict(kind=k, a=a, b=b, n=n, mean=np.mean(v), p05=np.percentile(v, 5), p95=np.percentile(v, 95)))
R2 = pd.DataFrame(rows); R2.to_csv(f'{OUT}/rc2_generations_needed.csv', index=False); print(R2.round(3).to_string(index=False), flush=True)
# RC3: threshold = midpoint between the lowest Z-Image lineage value and the highest Z-Image-involving unrelated value (R15-4 table), applied elsewhere
B4 = pd.read_csv(os.path.join(R15, 'out', 'R15_4_block_null.csv'))
zl = B4[(B4.kind == 'lineage') & B4.a.str.startswith('Z1')].tangent.min()
zu = B4[(B4.kind == 'unrelated') & (B4.a.str.startswith('Z') & ~B4.b.str.startswith('ACE'))].tangent.max()
thr = (zl + zu) / 2
L6 = pd.read_csv(os.path.join(R15, 'out', 'R15_6_llm_tangents.csv')); L6 = L6[L6.drop_last == 0]
test = [('ACE lineage attn', 'lineage', B4.query("a=='ACEB_attn' and b=='ACET_attn'").tangent.iloc[0]), ('ACE lineage mlp', 'lineage', B4.query("a=='ACEB_mlp' and b=='ACET_mlp'").tangent.iloc[0])]
test += [(f'{r.a}-{r.b}', 'unrelated', r.tangent) for r in B4[(B4.kind == 'unrelated') & (B4.a.str.startswith('ACE') | B4.b.str.startswith('ACE'))].itertuples()]
test += [(f'LLM {r.a}-{r.b} {r.sub}', 'unrelated', r.tangent) for r in L6.itertuples()]
R3 = pd.DataFrame([dict(pair=p, truth=t, tangent=v, called='lineage-or-closer' if v >= thr else 'unrelated', correct=(v >= thr) == (t != 'unrelated')) for p, t, v in test])
R3.to_csv(f'{OUT}/rc3_threshold_transfer.csv', index=False); print(f'threshold {thr:.3f} (Z-Image lineage min {zl:.3f}, unrelated max {zu:.3f})'); print(R3.round(3).to_string(index=False))
