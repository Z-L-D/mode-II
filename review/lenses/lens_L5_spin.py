"""L5 spin-glass lens: replica overlaps and ultrametricity among generations (LENS_PREREGISTRATION.md).
Field per generation = g over steps x layers (analyzer-cached), minus the channel-mean field, each feature z-scored.
q_ab = Pearson correlation; d = sqrt(2(1-q)).
U1: bimodality coefficient of P(q) over within-prompt pairs.
U2: ultrametric index K = (d3 - d2) / sd(all distances) for (a) within-prompt triples, (b) cross-prompt triples.
Null ("same covariance, same triple design"): a two-level Gaussian with the observed between-prompt and within-prompt
covariances (top 50 PCs each, leftover variance added isotropically so total variance matches), same prompt membership,
same preprocessing, same triples; 1000 replicates. Excess ultrametricity = observed median K < 5th percentile of null medians.
Output: out/L5_spin.csv"""
import os, itertools, numpy as np, pandas as pd
from lens_common import CH, OUT, load
rng = np.random.default_rng(20261008); NR = int(os.environ.get('NR', 1000))


def prep(F):
    F = F - F.mean(0); sd = F.std(0); sd[sd == 0] = 1; return F / sd


def dist(F):
    Z = prep(F); Z = (Z - Z.mean(1, keepdims=True)) / Z.std(1, keepdims=True); q = Z @ Z.T / Z.shape[1]
    return q, np.sqrt(np.clip(2 * (1 - q), 0, None))


def Kstat(Dm, T, sd):
    d = np.sort(np.stack([Dm[T[:, 0], T[:, 1]], Dm[T[:, 0], T[:, 2]], Dm[T[:, 1], T[:, 2]]], 1), 1)
    return np.median((d[:, 2] - d[:, 1]) / sd), np.median((d[:, 2] - d[:, 1]) / d[:, 2])


def bc(x):
    from scipy.stats import skew, kurtosis
    n = len(x); return (skew(x) ** 2 + 1) / (kurtosis(x) + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def gauss_level(M, k=50):
    """returns a sampler for N(0, Cov(M)) using the top-k PCs plus isotropic leftover variance."""
    M = M - M.mean(0); U, s, Vt = np.linalg.svd(M, full_matrices=False); k = min(k, len(s))
    sd = s[:k] / np.sqrt(len(M) - 1); left = max((s[k:] ** 2).sum() / (len(M) - 1) / M.shape[1], 0)
    return lambda n: (rng.standard_normal((n, k)) * sd) @ Vt[:k] + rng.standard_normal((n, M.shape[1])) * np.sqrt(left)


rows = []
for lab, key, D, fam in CH:
    d = load(key); F = d['g_gen'].reshape(len(d['g_gen']), -1).astype(float); pr = d['prompts']
    F = F[:, np.isfinite(F).all(0)]
    q, Dm = dist(F); iu = np.triu_indices(len(F), 1); sd = Dm[iu].std()
    same = pr[iu[0]] == pr[iu[1]]
    up, inv = np.unique(pr, return_inverse=True)
    Tw = np.array([c for p in range(len(up)) for c in itertools.combinations(np.where(inv == p)[0], 3)])
    first = {}; [first.setdefault(p, []).append(i) for i, p in enumerate(inv)]
    Tc = []
    while len(Tc) < 5000:
        ps = rng.choice(len(up), 3, replace=False); Tc.append([rng.choice(first[p]) for p in ps])
    Tc = np.array(Tc)
    Kw, Kw_rel = Kstat(Dm, Tw, sd); Kc, Kc_rel = Kstat(Dm, Tc, sd)
    # null
    Mp = np.array([F[inv == p].mean(0) for p in range(len(up))]); E = F - Mp[inv]
    samp_b, samp_w = gauss_level(Mp), gauss_level(E)
    nw, nc = [], []
    for _ in range(NR):
        Fn = samp_b(len(up))[inv] + samp_w(len(F)); _, Dn = dist(Fn); sdn = Dn[iu].std()
        nw.append(Kstat(Dn, Tw, sdn)[0]); nc.append(Kstat(Dn, Tc, sdn)[0])
    nw, nc = np.array(nw), np.array(nc)
    rows.append(dict(channel=lab, family=fam, gens=len(F), features=F.shape[1], q_same_mean=q[iu][same].mean(), q_diff_mean=q[iu][~same].mean(),
                     BC_within=bc(q[iu][same]), BC_bimodal=bc(q[iu][same]) > 0.555,
                     n_triples_within=len(Tw), K_within=Kw, K_within_null_p05=np.quantile(nw, 0.05), K_within_null_med=np.median(nw),
                     p_within=(np.sum(nw <= Kw) + 1) / (NR + 1), excess_within=Kw < np.quantile(nw, 0.05), Krel_within=Kw_rel,
                     K_cross=Kc, K_cross_null_p05=np.quantile(nc, 0.05), K_cross_null_med=np.median(nc),
                     p_cross=(np.sum(nc <= Kc) + 1) / (NR + 1), excess_cross=Kc < np.quantile(nc, 0.05), Krel_cross=Kc_rel))
    print(lab, 'done', flush=True)
pd.set_option('display.width', 300); df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'L5_spin.csv'), index=False)
print(df.round(4).to_string(index=False))
