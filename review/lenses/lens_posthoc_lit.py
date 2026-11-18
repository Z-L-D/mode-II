"""POST HOC follow-ups suggested by the literature survey (LIT_LENSES.md T2, T4, T7). Not preregistered; labelled as such.
PH2 (T4): R1 autonomy repeated within early / mid / late sigma terciles.
PH3 (T7): within-prompt variance of logit(g) (bounded-statistic normalisation); interior-peak criterion as in L4.
PH5 (T2): across layers at the final step, share of the variance of H2 explained by the SPL spike strength (ln s) alone vs the
          bulk exponent a alone.
Output: out/PH2_autonomy_by_sigma.csv, out/PH3_chi_logit.csv, out/PH5_spike_vs_bulk.csv"""
import numpy as np, pandas as pd
from lens_common import CH, OUT, load, SPL, prompt_boot_idx
from lens_L2_rg import spline_basis
rng = np.random.default_rng(20261008)


def r1_subset(g, pr, steps):
    g = g[:, steps]; n, T, L = g.shape
    x = g[:, :, :-1].ravel(); y = (g[:, :, 1:] - g[:, :, :-1]).ravel()
    lay = np.broadcast_to(np.arange(L - 1), (n, T, L - 1)).ravel(); grp = np.repeat(pr, T * (L - 1))
    ok = np.isfinite(x) & np.isfinite(y); x, y, lay, grp = x[ok], y[ok], lay[ok], grp[ok]
    up = np.unique(pr); fold_of = dict(zip(up, np.random.default_rng(1).permutation(len(up)) % 5)); fold = np.array([fold_of[p] for p in grp])
    pg, pgl = np.empty_like(y), np.empty_like(y)
    for f in range(5):
        tr, te = fold != f, fold == f; lo, hi = x[tr].min(), x[tr].max(); kn = np.quantile(x[tr], np.linspace(0, 1, 14)[1:-1])
        Btr, Bte = spline_basis(x[tr], lo, hi, kn), spline_basis(x[te], lo, hi, kn)
        pg[te] = Bte @ np.linalg.lstsq(Btr, y[tr], rcond=None)[0]
        Etr, Ete = np.eye(L - 1)[lay[tr]][:, 1:], np.eye(L - 1)[lay[te]][:, 1:]
        pgl[te] = np.hstack([Bte, Ete]) @ np.linalg.lstsq(np.hstack([Btr, Etr]), y[tr], rcond=None)[0]
    ss = np.sum((y - y.mean()) ** 2); return 1 - np.sum((y - pg) ** 2) / ss, 1 - np.sum((y - pgl) ** 2) / ss


def chi_logit(d, ix):
    g = d['g_gen'][ix]; lg = np.nanmean(np.log(g / (1 - g)), 2); df = pd.DataFrame(lg); df['p'] = d['prompts'][ix]
    return df.groupby('p').var(ddof=1).dropna().mean(0).values


ph2, ph3, ph5 = [], [], []
for lab, key, D, fam in CH:
    d = load(key); g = d['g_gen'].astype(float); T = g.shape[1]
    for nm, st in zip(('early', 'mid', 'late'), np.array_split(np.arange(T), 3)):
        a, b = r1_subset(g, d['prompts'], st); ph2.append(dict(channel=lab, sigma_bin=nm, sigma_range=f'{d["sigma"][st[0]]:.2f}-{d["sigma"][st[-1]]:.2f}', R2_g=a, R2_g_layer=b, dR2=b - a))
    edge, tol = (1, 1) if T <= 8 else (2, 2)
    c0 = chi_logit(d, np.arange(len(d['prompts']))); j0 = int(np.argmax(c0)); B = np.array([chi_logit(d, prompt_boot_idx(d['prompts'], rng)) for _ in range(200)])
    stab = np.mean(np.abs(B.argmax(1) - j0) <= tol); m = (c0[j0] - c0[0] > 2 * np.std(B[:, j0] - B[:, 0], ddof=1)) and (c0[j0] - c0[-1] > 2 * np.std(B[:, j0] - B[:, -1], ddof=1))
    ph3.append(dict(channel=lab, argmax=j0, sigma=d['sigma'][j0], stab=stab, interior=edge <= j0 <= T - 1 - edge, margin=m, peak=(edge <= j0 <= T - 1 - edge) and stab >= 0.8 and m))
    s = SPL(D); H2 = np.log(np.nanmean(g[:, -1], 0) * D); Lm = np.log(np.nanmean(d['gap_gen'][:, -1], 0)); _, a, sp = s.fit(H2, Lm); ok = np.isfinite(a)
    r2 = lambda x, y: np.corrcoef(x, y)[0, 1] ** 2
    ph5.append(dict(channel=lab, layers_fitted=int(ok.sum()), layers=len(H2), R2_H2_on_ln_spike=r2(np.log(sp[ok]), H2[ok]) if ok.sum() > 2 else np.nan,
                    R2_H2_on_bulk_a=r2(a[ok], H2[ok]) if ok.sum() > 2 else np.nan))
    print(lab, 'done', flush=True)
pd.set_option('display.width', 250)
for nm, rows in (('PH2_autonomy_by_sigma', ph2), ('PH3_chi_logit', ph3), ('PH5_spike_vs_bulk', ph5)):
    df = pd.DataFrame(rows); df.to_csv(f'{OUT}/{nm}.csv', index=False); print(nm); print(df.round(4).to_string(index=False))
