"""POST HOC (not preregistered): is the U1 bimodality coefficient driven by skew rather than two modes?
Counts KDE modes (Scott bandwidth) of within-prompt overlaps and reports skew/kurtosis. Output: out/L5_posthoc_modes.csv"""
import numpy as np, pandas as pd
from scipy.stats import skew, kurtosis, gaussian_kde
from lens_common import CH, OUT, load
def dist(F):  # same as lens_L5_spin.dist
    F = F - F.mean(0); sd = F.std(0); sd[sd == 0] = 1; Z = F / sd
    Z = (Z - Z.mean(1, keepdims=True)) / Z.std(1, keepdims=True); q = Z @ Z.T / Z.shape[1]; return q, None
rows = []
for lab, key, D, fam in CH:
    d = load(key); F = d['g_gen'].reshape(len(d['g_gen']), -1).astype(float); F = F[:, np.isfinite(F).all(0)]; pr = d['prompts']
    q, _ = dist(F); iu = np.triu_indices(len(F), 1); x = q[iu][pr[iu[0]] == pr[iu[1]]]
    gx = np.linspace(x.min(), x.max(), 512); k = gaussian_kde(x)(gx); m = np.sum((k[1:-1] > k[:-2]) & (k[1:-1] > k[2:]) & (k[1:-1] > 0.05 * k.max()))
    rows.append(dict(channel=lab, n_pairs=len(x), skew=skew(x), excess_kurtosis=kurtosis(x), kde_modes=int(m), q_median=np.median(x)))
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/L5_posthoc_modes.csv', index=False); print(df.round(3).to_string(index=False))
