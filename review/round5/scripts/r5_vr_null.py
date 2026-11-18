# Post hoc: calibrate VR(k) with Claude's estimator (r5_common.resid + VR) on synthetic residual fields of the
# same shape as the data (30 prompts x 2 seeds, 40 steps, 72 features; and 1 prompt x 15 seeds). Output: out/VR_null.json
import json, numpy as np
from r5_common import *
rng = np.random.default_rng(57); out = {}
def sim(kind, G, P, T=40, K=72, reps=50, k=4):
    v = []
    for _ in range(reps):
        if kind == 'walk': X = np.cumsum(rng.normal(size=(G, T, K)), 1)
        elif kind == 'iid': X = rng.normal(size=(G, T, K))
        elif kind == 'ar1_0.8':
            X = np.zeros((G, T, K)); X[:, 0] = rng.normal(size=(G, K))
            for t in range(1, T): X[:, t] = 0.8 * X[:, t - 1] + rng.normal(size=(G, K))
        Xp = np.exp(X[..., None] * 0.01).repeat(3, -1)[..., :K // 3, :] if False else None
        ph = np.repeat(np.arange(P), G // P)
        Y = X.reshape(G, T, K // 3, 3); Y = np.concatenate([np.exp(0.05 * Y[..., :2]), Y[..., 2:]], -1)   # g, gap positive; H raw
        r, _, _ = resid(Y, ph); v.append(VR(r, (k,))[k])
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
for kind in ['walk', 'ar1_0.8', 'iid']:
    out[f'{kind}_30x2_VR4'] = sim(kind, 60, 30); out[f'{kind}_1x15_VR4'] = sim(kind, 15, 1)
json.dump(out, open(f'{OUT}/VR_null.json', 'w'), indent=1); print(json.dumps(out, indent=1))
