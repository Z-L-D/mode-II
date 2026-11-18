# Post hoc P3: temporal-order permutation test for the three-segment fit (09-15 summary claim 3, "0/1000 temporal permutations
# fit comparably"), on archive and clean MLP channels. Statistic: R^2 of the best continuous 3-segment fit. Output: out/P3_order_perm.csv
from r6c_common import *
rng = np.random.default_rng(67)
def best_r2(y, s):
    u = -s; T = len(y); best = np.inf
    for i in range(1, T - 2):
        for j in range(i + 1, T - 1):
            A = np.stack([np.ones(T), u, np.maximum(0, u - u[i]), np.maximum(0, u - u[j])], 1)
            c, *_ = np.linalg.lstsq(A, y, rcond=None); best = min(best, ((A @ c - y) ** 2).sum())
    return 1 - best / ((y - y.mean()) ** 2).sum()
rows = []
sets = [('archive ZIT-mlp',) + arch('ZIT', 'mlp')[1:], ('archive ZIB-mlp',) + arch('ZIB', 'mlp')[1:]]
z4 = R5('Z40_1024_s1001', 'mlp'); sets.append(('R4 ZIB CFG4 cond-mlp', z4['meta'], z4['X']['cond'], np.asarray(z4['sigma'], float)))
_, m3, X3, s3 = R3('Z10_1024_s1001', 'mlp'); sets.append(('R3 ZIB CFG1 cond-only-mlp', m3, X3, np.asarray(s3, float)))
for lab, meta, X, s in sets:
    y = X[..., 0].mean((0, 2)); r2 = best_r2(y, s); nul = np.array([best_r2(y[rng.permutation(len(y))], s) for _ in range(1000)])
    rows.append(dict(set=lab, steps=len(s), R2_observed=r2, null_median=float(np.median(nul)), null_max=float(nul.max()), n_null_ge_obs=int((nul >= r2).sum())))
P3 = pd.DataFrame(rows); P3.to_csv(f'{OUT6}/P3_order_perm.csv', index=False); print(P3.round(4).to_string())
