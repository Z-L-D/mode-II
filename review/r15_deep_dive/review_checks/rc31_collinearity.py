# RC31 (post hoc, round-7 reviews: GLM mi13, DeepSeek 4.10). Per cube: Spearman correlation over all
# (generation, state, layer) cells of log g with H, log g with log gap, and H with log gap.
# Usage: python rc31_collinearity.py <cache> <r16cubes> <r17cubes>
import os, sys, pickle, numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'out')
CA, R16, R17 = sys.argv[1:4]
CUBES = [(CA, m) for m in ('ZIT', 'ZIB', 'MOCHI', 'SD35M', 'SD35L', 'Z1', 'Z3', 'ACEB', 'ACET')] + [(R17, m) for m in ('SD3M', 'PIXART', 'COSMOS2B')]
rows = []
for d, m in CUBES:
    for sub in ('attn', 'mlp'):
        f = f'{d}/F_{m}_{sub}.pkl'
        if not os.path.exists(f): continue
        X = np.asarray(pickle.load(open(f, 'rb'))[2], float).reshape(-1, 3); X = X[np.all(np.isfinite(X), 1) & (X[:, 0] > 0) & (X[:, 1] > 0)]
        lg, lgap, H = np.log(X[:, 0]), np.log(X[:, 1]), X[:, 2]
        rows.append(dict(model=m, sub=sub, cells=len(X), r_logg_H=spearmanr(lg, H)[0], r_logg_loggap=spearmanr(lg, lgap)[0], r_H_loggap=spearmanr(H, lgap)[0]))
D = pd.DataFrame(rows); D.to_csv(f'{OUT}/rc31_collinearity.csv', index=False); print(D.round(3).to_string(index=False))
