"""U5 (post hoc, P3 companion): the E6 naive between-prompt share alongside the ANOVA share, SD1.5 cond, for Mode-II Fig. II-F5c
(which plots the naive share). Same code as u2_posthoc.py P3 / round2 e6_varcomp.py. Output: out/P3_prompt_share_by_step_naive.csv"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
c = np.load(f'{BASE}/out/u0_cache.npz', allow_pickle=True); pt = c['pt']
sig_ve = c['sigma'].astype(float); t_rf = sig_ve / (1 + sig_ve)
rows = []
up, inv = np.unique(pt, return_inverse=True); k = len(up); cnt = np.bincount(inv)
for sub in ('attn', 'mlp'):
    Y = np.stack([c[f'{sub}_cond_{v}'].astype(float) for v in ('g_micro', 'spectral_gap', 'sv_entropy')], -1).copy()
    Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1])
    G, T, L, _ = Y.shape; Yf = Y.reshape(G, T, -1)
    Yf = np.where(np.isfinite(Yf), Yf, np.nanmean(Yf, 0, keepdims=True))
    Yf = (Yf - Yf.mean(0)) / (Yf.std(0) + 1e-12); n0 = (G - (cnt ** 2).sum() / G) / (k - 1)
    for t in range(T):
        Z = Yf[:, t]; mu = np.stack([Z[inv == i].mean(0) for i in range(k)])
        ssw = ((Z - mu[inv]) ** 2).sum(0); ssb = (cnt[:, None] * (mu - Z.mean(0)) ** 2).sum(0)
        msw = ssw / (G - k); msb = ssb / (k - 1); vp = np.maximum((msb - msw) / n0, 0)
        rows.append(dict(ch=f'SD1.5-{sub}', step=t, t=t_rf[t], sigma_VE=sig_ve[t], prompt_share_naive=ssb.sum() / (ssb.sum() + ssw.sum()),
                         prompt_share_anova=vp.sum() / (vp.sum() + msw.sum())))
D = pd.DataFrame(rows)
old = pd.read_csv(f'{BASE}/out/P3_prompt_share_by_step.csv'); assert np.allclose(old.prompt_share_anova, D.prompt_share_anova)
D.to_csv(f'{BASE}/out/P3_prompt_share_by_step_naive.csv', index=False)
print(D.groupby('ch')[['prompt_share_naive', 'prompt_share_anova']].agg(['min', 'max']).round(3))
