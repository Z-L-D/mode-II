# RC28 (post hoc, round-6 reviews: Kimi M2, Qwen M9, GLM 5/Q3): a null distribution for the Finding 12 shortfall on the RC24
# depth-matched design (same learner, features, scaffold and prompt-grouped CV as rc24_depth_matched_transfer.py).
#   null       same-weights transfer: for each model, 20 random prompt halves; the regressor is fitted on half A and applied to
#              half B; null shortfall = the model's self CV R^2 (rc24) - R^2 on half B. Training on half the prompts makes this
#              null conservative (it overstates same-weights shortfall).
#   cross CI   for every RC24 pair, the source model (fitted on all its data) is applied to prompt-bootstrap resamples of the
#              target (200 draws, prompts with replacement); 5th/95th percentiles of the shortfall.
#   test       a pair's shortfall counts as beyond same weights if its 5th percentile exceeds the pooled null p95 of its submodule.
# Usage: python rc28_finding12_null.py <cache> <r16cubes> <r17cubes>
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); src = open(os.path.join(HERE, 'rc24_depth_matched_transfer.py')).read().split('\nPAIRS =')[0]
exec(compile(src, 'rc24', 'exec'))
T24 = pd.read_csv(f'{OUT}/rc24_depth_matched_transfer.csv'); rng = np.random.default_rng(28); rows = []
for (tgt, sub), g in T24.groupby(['target', 'sub']):
    Xd, Y, pt = prep(tgt, sub); u = np.unique(pt); selfr2 = float(g['self'].iloc[0])
    for k in range(20):
        h = rng.permutation(u); A = np.isin(pt, h[:len(u) // 2]); B = ~A
        M = fit(Xd[A].reshape(-1, 5), Y[A].reshape(-1, 3)); c = r2(Y[B].reshape(-1, 3), pred(M, Xd[B].reshape(-1, 5)))
        rows.append(dict(kind='null', target=tgt, sub=sub, draw=k, shortfall=selfr2 - c))
    print('null', tgt, sub, np.round(np.percentile([r['shortfall'] for r in rows if r['target'] == tgt and r['sub'] == sub and r['kind'] == 'null'], [50, 95]), 3), flush=True)
N = pd.DataFrame(rows); CI = []
for _, r in T24.iterrows():
    Xa, Ya, _ = prep(r.source, r['sub']); Xb, Yb, pb = prep(r.target, r['sub']); M = fit(Xa.reshape(-1, 5), Ya.reshape(-1, 3))
    P = pred(M, Xb.reshape(-1, 5)).reshape(Yb.shape); u = np.unique(pb); idx = {q: np.where(pb == q)[0] for q in u}; s = []
    for _ in range(200):
        gi = np.concatenate([idx[q] for q in rng.choice(u, len(u))]); s.append(r['self'] - r2(Yb[gi].reshape(-1, 3), P[gi].reshape(-1, 3)))
    CI.append(dict(relation=r.relation, source=r.source, target=r.target, sub=r['sub'], shortfall=r.shortfall, sf05=np.percentile(s, 5), sf95=np.percentile(s, 95)))
C = pd.DataFrame(CI); p95 = N.groupby('sub').shortfall.quantile(0.95); C['null_p95_sub'] = C['sub'].map(p95); C['beyond_null'] = C.sf05 > C.null_p95_sub
N.to_csv(f'{OUT}/rc28_null_draws.csv', index=False); C.to_csv(f'{OUT}/rc28_finding12_null.csv', index=False)
pd.set_option('display.width', 250); print('\nnull shortfall by target:'); print(N.groupby(['target', 'sub']).shortfall.describe(percentiles=[.5, .95])[['mean', '50%', '95%', 'max']].round(3))
print('\npooled null p95:', p95.round(3).to_dict()); print(C.round(3).to_string(index=False))
print(C.groupby('relation', sort=False).beyond_null.agg(['sum', 'count']))
