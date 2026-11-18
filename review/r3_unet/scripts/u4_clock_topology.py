"""U4 (post hoc, R3C_ADDENDUM_P7.md): R6 S4 clock-axis three-segment topology of the population-mean layer-mean g, SD1.5 cond.
fit3 / extrema / bootstrap copied unchanged from review/round6/scripts/r6c_S4_topology.py (nanmean for the one missing cell).
Output: out/P7_clock_topology.csv"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
Z = np.load(f'{BASE}/out/u0_cache.npz', allow_pickle=True)
NB = 1000


def fit3(y, s):
    u = -s; T = len(y); best = None
    for i in range(1, T - 2):
        for j in range(i + 1, T - 1):
            A = np.stack([np.ones(T), u, np.maximum(0, u - u[i]), np.maximum(0, u - u[j])], 1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None); sse = ((A @ coef - y) ** 2).sum()
            if best is None or sse < best[0]: best = (sse, i, j, coef)
    sse, i, j, c = best; sl = np.array([c[1], c[1] + c[2], c[1] + c[2] + c[3]])
    A = np.stack([np.ones(T), u, np.maximum(0, u - u[i]), np.maximum(0, u - u[j])], 1)
    return ''.join('+' if v > 0 else '-' for v in sl), i, j, sl, A @ c


def extrema(y, s):
    tr = int(np.argmin(y)); pk = int(np.argmax(y[:tr + 1])) if tr > 0 else 0
    return s[pk], s[tr], (y[tr] - y[pk]) / y[pk], (y[-1] - y[tr]) / y[tr]


sig = Z['sigma'].astype(float); t = sig / (1 + sig)
CLK = {'t=sigma/(1+sigma)': t, 'sigma_VP': sig, "t'=2t/(1+t)": 2 * t / (1 + t)}
pt = np.char.strip(Z['pt'].astype(str)); up = np.unique(pt)
rows, curves = [], []
for sub in ('attn', 'mlp'):
    G = Z[f'{sub}_cond_g_micro']
    PP = np.stack([np.nanmean(G[pt == p], axis=(0, 2)) for p in up]); y = PP.mean(0)
    for ck, s in CLK.items():
        rng = np.random.default_rng(64)
        top, i, j, sl, fit = fit3(y, s); pk, tr, amp, reb = extrema(y, s)
        boot = np.array([fit3(PP[rng.integers(0, len(PP), len(PP))].mean(0), s)[0] for _ in range(NB)])
        v, c = np.unique(boot, return_counts=True)
        rows.append(dict(channel=f'SD1.5-{sub}', clock=ck, prompts=len(PP), steps=len(s), topology=top, frac_pmp=float((boot == '+-+').mean()),
                         frac_ppm=float((boot == '++-').mean()), modal=v[c.argmax()], modal_frac=c.max() / NB, break1=s[i], break2=s[j],
                         slopes=sl.round(6).tolist(), peak=pk, trough=tr, contraction=amp, rebound=reb,
                         late_drop=(y[-1] - y.max()) / y.max(), clock_at_max=s[y.argmax()]))
        print(rows[-1], flush=True)
        if ck == 't=sigma/(1+sigma)':
            curves += [dict(channel=f'SD1.5-{sub}', step=k, sigma_VP=sig[k], t=t[k], g=y[k], fit=fit[k]) for k in range(len(s))]
pd.DataFrame(rows).to_csv(f'{BASE}/out/P7_clock_topology.csv', index=False)
pd.DataFrame(curves).to_csv(f'{BASE}/out/P7_clock_curves.csv', index=False)
