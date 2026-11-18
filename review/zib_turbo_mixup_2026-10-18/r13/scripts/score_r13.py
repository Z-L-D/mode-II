import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Round 13: score PREDICTIONS_R13.md (sha 47cf1e67) on HOLDOUT prompts only.
import os, pickle, json, numpy as np, pandas as pd
SCR = os.environ.get('SCR', _COSPAFS_CACHE); OUTD = f'{_COSPAFS_ROOT}/review/zib_turbo_mixup_2026-10-18/r13/out'
L = lambda n: pickle.load(open(f'{SCR}/F_{n}.pkl', 'rb')); LV = lambda n: pickle.load(open(f'{SCR}/V_{n}.pkl', 'rb'))
R = []; det = {}
def add(i, ok, d): R.append(dict(id=i, verdict='SUPPORTED' if ok else 'REFUTED', detail=d)); print(R[-1], flush=True)
V = {sub: (LV(f'Z2H_{sub}'), LV(f'ZITSVH_{sub}')) for sub in ('attn', 'mlp')}
def dl(sub, c):
    B, T = V[sub]; j = [int(np.argmin(np.abs(B['s'] - x))) for x in T['s']]
    return np.log(B['V'][:, j, :, c]).mean((0, 1)) - np.log(T['V'][..., c]).mean((0, 1))
# H1
h1 = {sub: int((np.median(V[sub][0]['V'][..., 15], (0, 1)) > np.median(V[sub][1]['V'][..., 15], (0, 1))).sum()) for sub in V}
add('H1', all(x >= 27 for x in h1.values()), f"layers with Base > Turbo mean share: attn {h1['attn']}/30, mlp {h1['mlp']}/30")
# H2
h2 = {sub: (np.abs(dl(sub, 0)).mean(), np.abs(dl(sub, 6)).mean()) for sub in V}
add('H2', all(a >= 2 * b for a, b in h2.values()), '; '.join(f'{s} V1 {a:.3f} vs V3 {b:.3f} (ratio {a/b:.2f})' for s, (a, b) in h2.items()))
# H3
h3a = {sub: int((dl(sub, 0) < 0).sum()) for sub in V}
h3b = {}
for sub in ('attn', 'mlp'):
    _, _, B, s = L(f'Z1H_{sub}'); _, _, T, _ = L(f'Z3H_{sub}'); d = np.log(B[..., 0]).mean((0, 2)) - np.log(T[..., 0]).mean((0, 2)); h3b[sub] = int((d < 0).sum())
add('H3', all(x >= 27 for x in h3a.values()) and all(x == 25 for x in h3b.values()), f"(a) V1 layers Base<Turbo attn {h3a['attn']}/30 mlp {h3a['mlp']}/30; (b) Z1<Z3 steps attn {h3b['attn']}/25 mlp {h3b['mlp']}/25")
# H4
h4 = {}
for sub in V:
    a = np.abs(dl(sub, 0)); h4[sub] = (a[17:23].mean(), a[0:6].mean(), a[24:30].mean())
add('H4', all(m < 0.5 * e and m < 0.5 * l for m, e, l in h4.values()), '; '.join(f'{s} mid {m:.3f} early {e:.3f} late {l:.3f}' for s, (m, e, l) in h4.items()))
# H5
_, mc, Xc, s = L('Z1H_attn'); _, mu, Xu, _ = L('Z1UH_attn')
kc = list(mc.index.str.rsplit(':', n=1).str[0]); ku = list(mu.index.str.rsplit(':', n=1).str[0]); com = sorted(set(kc) & set(ku))
D = np.log(Xc[[kc.index(c) for c in com], ..., 0]) - np.log(Xu[[ku.index(c) for c in com], ..., 0])
byl = D.mean((0, 1)); bys = D.mean((0, 2))
ok5 = byl[0:3].mean() > 0 and byl[25:30].mean() > 0 and byl[6:22].mean() < 0 and (bys[2:16] < 0).all()
add('H5', ok5, f'L0-2 {byl[0:3].mean():.3f}, L25-29 {byl[25:30].mean():.3f}, L6-21 {byl[6:22].mean():.3f}; steps 2-15 negative: {int((bys[2:16] < 0).sum())}/14')
# H6
_, meta, X, s = L('Z1H_attn'); y = np.log(X[:, 0, :, 0]).mean(1); d = pd.DataFrame(dict(y=y, k=meta.pt.astype(str).str.strip().values))
eta = ((d.groupby('k').y.transform('mean') - d.y.mean()) ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum()
add('H6', eta >= 0.90, f'step-0 prompt share {eta:.3f} over {d.k.nunique()} prompts, {len(d)} gens')
pd.DataFrame(R).to_csv(f'{OUTD}/r13_verdicts.csv', index=False)
