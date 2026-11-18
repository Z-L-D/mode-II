import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Round 11 scoring, as frozen in capture/PREREG_R11.md (sha 3834abb0, 2026-10-17 13:22Z) + ADDENDUM_1 (primary = the 60 prereg prompts).
# Arms as run after the Base/Turbo mix-up: Base = Z2 (Z-Image Base, 25 steps, CFG 1, seed 1001), Turbo = ZITSV (8 steps, CFG 1, seed 1001),
# SD3.5 M = SD3SV (40 steps, CFG 1, seed 1001, attn only). Cond branch, complete generations (build_cubes.py).
# Methods unchanged: Round 6 S1 tangent() + layer-permutation null (1000 draws, p95); backbone() copied from r3_ace a2_posthoc (999 depth-shift nulls);
# S4 fit3 on the per-prompt layer-mean g (1000 prompt bootstraps). Variants: V0 all rows, V1 image rows, V3 centred, V4 top-8 channels removed.
import os, sys, json, pickle
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round6/scripts')
from r6c_common import *
SCR = os.environ.get('SCR', _COSPAFS_CACHE); OUTD = f'{_COSPAFS_ROOT}/review/peer_review_2026-10-17/r11/out'
P60 = [l.strip() for l in open(f'{_COSPAFS_ROOT}/review/peer_review_2026-10-17/capture/prompts_r3_cond60.txt') if l.strip()]
VI = {'V1': (0, 1, 2), 'V3': (6, 7, 8), 'V4': (9, 10, 11)}
ARMS = {'Base': 'Z2', 'Turbo': 'ZITSV', 'SD35M': 'SD3SV'}
def load(arm, sub, var):
    g, meta, X, s = pickle.load(open(f'{SCR}/F_{ARMS[arm]}_{sub}.pkl', 'rb'))
    keep = meta.pt.astype(str).str.strip().isin(P60).values
    if var != 'V0':
        V = pickle.load(open(f'{SCR}/V_{ARMS[arm]}_{sub}.pkl', 'rb'))['V']; X = V[..., list(VI[var])]
    return meta[keep], X[keep].astype(float), np.asarray(s, float)
rng = np.random.default_rng(11)
def tan(a, b, sub, var, null=False):
    _, XA, sa = load(a, sub, var); _, XB, sb = load(b, sub, var)
    tc, sr = tangent(XA, sa, XB, sb); p95 = float('nan')
    if null: p95 = float(np.percentile([tangent(XA, sa, XB[:, :, rng.permutation(XB.shape[2])], sb)[0] for _ in range(1000)], 95))
    return dict(a=a, b=b, sub=sub, var=var, n_a=len(XA), n_b=len(XB), tangent=tc, shape=sr, null_p95=p95)
def backbone(X, nnull=999, seed=0):
    Y = np.nanmean(X, 0).copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1])
    Z = (Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-12)
    def stats(Z):
        A = Z.reshape(Z.shape[0], -1); A = A - A.mean(0); ev = np.clip(np.linalg.svd(A, compute_uv=False) ** 2, 0, None); p = ev / ev.sum(); p = p[p > 0]
        return p[0], float(np.exp(-(p * np.log(p)).sum()))
    pc1, er = stats(Z); r = np.random.default_rng(seed); T, L, _ = Z.shape; npc, nrk = [], []
    for _ in range(nnull):
        Zs = np.stack([np.roll(Z[t], r.integers(L), axis=0) for t in range(T)]); a, b = stats(Zs); npc.append(a); nrk.append(b)
    npc, nrk = np.array(npc), np.array(nrk)
    return dict(pc1=pc1, eff_rank=er, null_pc1_max=npc.max(), p_pc1=(1 + (npc >= pc1).sum()) / (nnull + 1))
def fit3(y, s):
    u = -s; T = len(y); best = None
    for i in range(1, T - 2):
        for j in range(i + 1, T - 1):
            A = np.stack([np.ones(T), u, np.maximum(0, u - u[i]), np.maximum(0, u - u[j])], 1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None); sse = ((A @ coef - y) ** 2).sum()
            if best is None or sse < best[0]: best = (sse, i, j, coef)
    sse, i, j, c = best; sl = np.array([c[1], c[1] + c[2], c[1] + c[2] + c[3]])
    return ''.join('+' if v > 0 else '-' for v in sl), i, j, sl
def clock(meta, X, s, nb=1000):
    k = meta.pt.astype(str).str.strip().values; up = np.unique(k); PP = np.stack([X[k == p][..., 0].mean((0, 2)) for p in up])
    top = fit3(PP.mean(0), s)[0]; r = np.random.default_rng(64)
    boot = np.array([fit3(PP[r.integers(0, len(PP), len(PP))].mean(0), s)[0] for _ in range(nb)])
    return dict(topology=top, frac_pmp=float((boot == '+-+').mean()))
VARS = ['V0', 'V1', 'V3', 'V4']; T = []
for var in VARS:
    for sub in ('attn', 'mlp'):
        T.append(tan('Base', 'Turbo', sub, var, null=True)); print(T[-1], flush=True)
    for a in ('Base', 'Turbo'):
        T.append(tan(a, 'SD35M', 'attn', var, null=True)); print(T[-1], flush=True)
TT = pd.DataFrame(T); TT.to_csv(f'{OUTD}/r11_tangents.csv', index=False)
t = lambda a, b, sub, var: TT[(TT.a == a) & (TT.b == b) & (TT['sub'] == sub) & (TT['var'] == var)].iloc[0]
B = []
for arm in ARMS:
    for sub in (('attn',) if arm == 'SD35M' else ('attn', 'mlp')):
        for var in VARS:
            meta, X, s = load(arm, sub, var); B.append(dict(arm=arm, sub=sub, var=var, **backbone(X))); print(B[-1], flush=True)
BB = pd.DataFrame(B); BB.to_csv(f'{OUTD}/r11_backbone.csv', index=False)
K = []
for var in VARS:
    for arm in ('Base', 'Turbo'):
        meta, X, s = load(arm, 'mlp', var); K.append(dict(arm=arm, var=var, **clock(meta, X, s))); print(K[-1], flush=True)
KK = pd.DataFrame(K); KK.to_csv(f'{OUTD}/r11_clock.csv', index=False)
# R11-7 diagnostics on Z-Image cells (txt_energy_share = column 14, mean_energy_share = 15, n_massive_100x = 17)
D7 = {}
for arm in ('Base', 'Turbo'):
    for sub in ('attn', 'mlp'):
        meta, _, _ = load(arm, sub, 'V0'); V = pickle.load(open(f'{SCR}/V_{ARMS[arm]}_{sub}.pkl', 'rb'))
        keep = V['meta'].pt.astype(str).str.strip().isin(P60).values; W = V['V'][keep]
        D7[f'{arm}_{sub}'] = dict(txt_share_ge_0p10=float((W[..., 14] >= 0.10).mean()), txt_share_median=float(np.median(W[..., 14])),
                                  mean_energy_share_median=float(np.median(W[..., 15])), mean_energy_by_layer=np.median(W[..., 15], (0, 1)).round(4).tolist(),
                                  n_massive_median=float(np.median(W[..., 17])), n_massive_gt8=float((W[..., 17] > 8).mean()))
json.dump(D7, open(f'{OUTD}/r11_diagnostics.json', 'w'), indent=1)
V = []
def add(i, test, ok, det): V.append(dict(id=i, test=test, verdict='SUPPORTED' if ok else 'REFUTED', detail=det))
a1, m1, a0, m0 = t('Base', 'Turbo', 'attn', 'V1'), t('Base', 'Turbo', 'mlp', 'V1'), t('Base', 'Turbo', 'attn', 'V0'), t('Base', 'Turbo', 'mlp', 'V0')
add('R11-1', 'Base<->Turbo V1 >= 0.95 and within 0.03 of V0', a1.tangent >= 0.95 and m1.tangent >= 0.95 and abs(a1.tangent - a0.tangent) <= 0.03 and abs(m1.tangent - m0.tangent) <= 0.03,
    f'attn V1 {a1.tangent:.3f} (V0 {a0.tangent:.3f}); mlp V1 {m1.tangent:.3f} (V0 {m0.tangent:.3f})')
r2 = {v: (t('Base', 'Turbo', 'attn', v).tangent, t('Base', 'Turbo', 'mlp', v).tangent) for v in ('V3', 'V4')}
add('R11-2', 'Base<->Turbo V3 and V4 >= 0.90', all(min(x) >= 0.90 for x in r2.values()), '; '.join(f'{v} attn {x[0]:.3f} mlp {x[1]:.3f}' for v, x in r2.items()))
r3 = [(a, v, t(a, 'SD35M', 'attn', v)) for v in ('V1', 'V3', 'V4') for a in ('Base', 'Turbo')]
add('R11-3', 'unrelated (vs SD3.5 M, attn) below their permutation p95, V1/V3/V4', all(r.tangent < r.null_p95 for _, _, r in r3),
    '; '.join(f'{a}-{v} {r.tangent:.3f} (p95 {r.null_p95:.3f})' for a, v, r in r3))
mg = {v: t('Base', 'Turbo', 'attn', v).tangent - max(t('Base', 'SD35M', 'attn', v).tangent, t('Turbo', 'SD35M', 'attn', v).tangent) for v in ('V3', 'V4')}
add('R11-4', 'lineage margin (attn) >= 0.50, V3 and V4', all(x >= 0.50 for x in mg.values()), '; '.join(f'{v} {x:.3f}' for v, x in mg.items()))
ok5 = all((r.pc1 >= 0.70) and (r.p_pc1 <= 0.001) for _, r in BB.iterrows())
fails = BB[(BB.pc1 < 0.70) | (BB.p_pc1 > 0.001)]
add('R11-5', 'backbone PC1 >= 0.70 and beats all 999 depth-shift nulls, every arm x sub x variant', ok5,
    'fails: ' + ('; '.join(f'{r.arm} {r["sub"]} {r["var"]} pc1 {r.pc1:.3f} p {r.p_pc1:.3f}' for _, r in fails.iterrows()) or 'none') +
    f' | pc1 range {BB.pc1.min():.3f}-{BB.pc1.max():.3f}')
kb = KK[(KK.arm == 'Base') & KK['var'].isin(['V1', 'V3', 'V4'])]
add('R11-6', 'Base MLP clock-axis +-+ in >= 90% of bootstraps, V1/V3/V4', bool((kb.frac_pmp >= 0.90).all()),
    '; '.join(f'{r["var"]} {r.topology} {r.frac_pmp:.3f}' for _, r in KK[KK.arm == 'Base'].iterrows()))
sh = {k: v['txt_share_ge_0p10'] for k, v in D7.items()}
add('R11-7', 'txt_energy_share < 0.10 in >= 90% of Z-Image cells', all(x <= 0.10 for x in sh.values()), '; '.join(f'{k} {x:.3f} of cells >= 0.10' for k, x in sh.items()))
VV = pd.DataFrame(V); VV.to_csv(f'{OUTD}/r11_verdicts.csv', index=False); print(VV.to_string())
