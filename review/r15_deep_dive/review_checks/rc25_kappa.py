# RC25 (post hoc, round-5 review GLM Q4): empirical kappa of Mode-II Prop. 7 per archive channel. Envelope a(sigma) = layer-mean
# linear g (population mean); trend T = sigma (linear) and, as a check, T = the cross-model mean envelope interpolated to each
# model's grid. kappa = Var(least-squares residual) / Var(fitted trend). Bound and observed correlation for every unrelated pair.
import sys, pickle, itertools, numpy as np, pandas as pd, os
CA = sys.argv[1]; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
M = ['ZIT', 'ZIB', 'MOCHI', 'SD35M', 'SD35L']; E = {}
for m in M:
    _, _, X, s = pickle.load(open(f'{CA}/F_{m}_attn.pkl', 'rb')); s = np.asarray(s, float); E[m] = (s, X[..., 0].mean((0, 2)))
grid = np.linspace(max(E[m][0].min() for m in M), min(E[m][0].max() for m in M), 40)
def on(m): s, a = E[m]; o = np.argsort(s); return np.interp(grid, s[o], a[o])
A = {m: on(m) for m in M}; Tm = np.mean([(A[m] - A[m].mean()) / A[m].std() for m in M], 0)
def kappa(a, T):
    c = np.polyfit(T, a, 1); f = np.polyval(c, T); return float(np.var(a - f) / np.var(f))
rows = []
for m in M: rows.append(dict(model=m, kappa_sigma=kappa(A[m], grid), kappa_meanenv=kappa(A[m], Tm)))
K = pd.DataFrame(rows); print(K.round(3).to_string(index=False))
pr = []
for a, b in itertools.combinations(M, 2):
    if {a, b} == {'ZIT', 'ZIB'}: continue
    for nm, T in (('sigma', grid), ('mean envelope', Tm)):
        ka, kb = kappa(A[a], T), kappa(A[b], T); bound = (1 - np.sqrt(ka * kb)) / np.sqrt((1 + ka) * (1 + kb))
        pr.append(dict(a=a, b=b, trend=nm, kappa_a=ka, kappa_b=kb, bound=bound, corr=float(np.corrcoef(A[a], A[b])[0, 1])))
P = pd.DataFrame(pr); print(P.round(3).to_string(index=False)); K.to_csv(f'{OUT}/rc25_kappa.csv', index=False); P.to_csv(f'{OUT}/rc25_kappa_pairs.csv', index=False)
