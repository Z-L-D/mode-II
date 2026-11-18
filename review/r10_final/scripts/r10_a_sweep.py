"""R10-1..R10-3: final correlation sweep (preregistered, R10_PREREGISTRATION.md).
R10-1 [desc] Spearman between log g, H, log gap over all cells per archive cube; mean depth-profile corr(z~_g, z~_H).
R10-2 [rule] plateau onset (E3) vs matched-60 relative seed-predictability onset (S3), 7 channels, LOO.
R10-3 [rule] VR4/walk (S3) vs matched-60 relative onset, with and without ZIT-attn, LOO.
Inputs: <SCR>/F_*.pkl, review/round2/out/E3_plateau_onset.csv, review/round6/out/S3_estimators.csv. Output: out/R10_*."""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import os, pickle, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', 'out'); RV = os.path.join(HERE, '..', '..')
SCR = os.environ.get('SCR', f'{_COSPAFS_CACHE}')
RES = {}

# ---------------- R10-1
rows = []
for name in ['ZIT_attn', 'ZIT_mlp', 'ZIB_attn', 'ZIB_mlp', 'MOCHI_attn', 'MOCHI1_attn', 'SD35M_attn', 'SD35L_attn']:
    gids, meta, X, sig = pickle.load(open(f'{SCR}/F_{name}.pkl', 'rb'))
    X = np.asarray(X, float)                               # [G, T, L, 3] = g, gap, H
    ok = np.isfinite(X).all(-1) & (X[..., 0] > 0) & (X[..., 1] > 0)
    lg, gp, H = np.log(X[..., 0][ok]), np.log(X[..., 1][ok]), X[..., 2][ok]
    r_gH, r_ggap, r_Hgap = spearmanr(lg, H).statistic, spearmanr(lg, gp).statistic, spearmanr(H, gp).statistic
    # depth-profile correlation of z-scored log g and H at each (gen, state), averaged
    Z = np.stack([np.log(X[..., 0]), X[..., 2]], -1)       # [G,T,L,2]
    Zs = (Z - Z.mean(2, keepdims=True)) / Z.std(2, keepdims=True)
    cu = (Zs[..., 0] * Zs[..., 1]).mean(2)                 # corr over depth per (gen, state)
    rows.append(dict(cube=name, G=X.shape[0], T=X.shape[1], L=X.shape[2], cells=int(ok.sum()), rho_logg_H=r_gH, rho_logg_loggap=r_ggap, rho_H_loggap=r_Hgap,
                     depth_corr_g_H_mean=float(np.nanmean(cu)), depth_corr_g_H_p5=float(np.nanpercentile(cu, 5)), depth_corr_g_H_p95=float(np.nanpercentile(cu, 95))))
D1 = pd.DataFrame(rows); D1.to_csv(f'{OUT}/R10_1_triple_redundancy.csv', index=False)
RES['R10-1'] = dict(rho_logg_H_range=[float(D1.rho_logg_H.min()), float(D1.rho_logg_H.max())], rho_logg_loggap_range=[float(D1.rho_logg_loggap.min()), float(D1.rho_logg_loggap.max())],
                    rho_H_loggap_range=[float(D1.rho_H_loggap.min()), float(D1.rho_H_loggap.max())], depth_corr_g_H_range=[float(D1.depth_corr_g_H_mean.min()), float(D1.depth_corr_g_H_mean.max())])
print(D1.round(3).to_string(index=False))

# ---------------- R10-2 / R10-3
E3 = pd.read_csv(f'{RV}/round2/out/E3_plateau_onset.csv').set_index('ch')
S3 = pd.read_csv(f'{RV}/round6/out/S3_estimators.csv').set_index('channel')
ch = [c for c in S3.index if c in E3.index]
T = pd.DataFrame(dict(channel=ch, plateau_onset=E3.loc[ch, 'onset_sigma'].values, rel_onset=S3.loc[ch, 'matched60_rel_median'].values, VR4_over_walk=S3.loc[ch, 'VR4_over_walk'].values))


def loo(x, y, thr=0.7):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearmanr(x, y).statistic
    l = [spearmanr(np.delete(x, i), np.delete(y, i)).statistic for i in range(len(x))]
    held = abs(rho) >= thr and all(np.sign(v) == np.sign(rho) and v != 0 for v in l)
    return dict(n=int(len(x)), rho=float(rho), loo=[float(v) for v in l], loo_min=float(min(l)), loo_max=float(max(l)), verdict='associated' if held else 'tested, not supported')


RES['R10-2'] = loo(T.plateau_onset, T.rel_onset)
r3_all = loo(T.VR4_over_walk, T.rel_onset)
m = T.channel != 'ZIT-attn'
r3 = loo(T.VR4_over_walk[m], T.rel_onset[m])
RES['R10-3'] = dict(without_ZIT_attn=r3, all_channels=r3_all, verdict=r3['verdict'])
T.to_csv(f'{OUT}/R10_2_3_channels.csv', index=False)
print(T.round(3).to_string(index=False))
print('R10-2', json.dumps(RES['R10-2']))
print('R10-3', json.dumps(RES['R10-3']))
json.dump(RES, open(f'{OUT}/R10_a_results.json', 'w'), indent=1)
