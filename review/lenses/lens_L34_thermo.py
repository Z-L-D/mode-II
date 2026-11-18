"""L3 heat-kernel vs thermal deformation, L4 spectral heat capacity and Z2 susceptibility, and the sigma-relaxation
exponent z for L6 (LENS_PREREGISTRATION.md). Data: analyzer-cached fields (sw_cache). 200 prompt bootstraps.
Output: out/L3_deformation.csv, out/L4_peaks.csv, out/L4_curves.csv, out/L6_z_exponent.csv"""
import os, numpy as np, pandas as pd
from lens_common import CH, OUT, load, SPL, entropies_from_sigma, prompt_boot_idx
rng = np.random.default_rng(20261008); NB = 200


# ---------- L3
def deform_curves(lam, kind):
    if kind == 'HK':
        par = np.r_[0, np.geomspace(1e-4, 1e3, 400)]; S = lam[None] + par[:, None] * lam.mean()
    else:
        par = np.geomspace(0.05, 20, 400); S = lam[None] ** (1 / par[:, None])
    H1, H2, L = entropies_from_sigma(np.sqrt(S)); o = np.argsort(H2)
    return H2[o], H1[o], L[o], par[o]


def l3(lab, key, D):
    d = load(key); G, GAP, ENT = (np.nanmean(d[k], 0) for k in ('g_gen', 'gap_gen', 'ent_gen'))   # [t, L]
    H2, L, H1 = np.log(G * D), np.log(GAP), ENT; T, nL = G.shape; s = SPL(D); lnD = np.log(D)
    h1f, af, sf = s.fit(H2[-1], L[-1]); rows = []
    for l in range(nL):
        if not np.isfinite(af[l]): continue
        lam = s.spectrum(af[l], sf[l], D) ** 2
        for kind in ('HK', 'TH'):
            h2s, h1s, ls, ps = deform_curves(lam, kind)
            for t in range(T - 1):
                feas = h2s[0] <= H2[t, l] <= h2s[-1]
                sc = abs(np.interp(H2[t, l], h2s, h1s) - H1[t, l]) / lnD + abs(np.interp(H2[t, l], h2s, ls) - L[t, l]) if feas else np.inf
                rows.append(dict(channel=lab, layer=l, step=t, kind=kind, feasible=feas, score=sc, par=np.interp(H2[t, l], h2s, ps) if feas else np.nan,
                                 h2_norm=H2[t, l] / lnD, h2_final=H2[-1, l] / lnD))
    return rows


# ---------- L4
def interior_peak(curve_fn, d, edge, tol):
    c0 = curve_fn(np.arange(len(d['prompts']))); T = len(c0); j0 = int(np.argmax(c0))
    B = np.array([curve_fn(prompt_boot_idx(d['prompts'], rng)) for _ in range(NB)])
    stab = np.mean(np.abs(B.argmax(1) - j0) <= tol)
    se_first = np.std(B[:, j0] - B[:, 0], ddof=1); se_last = np.std(B[:, j0] - B[:, -1], ddof=1)
    interior = edge <= j0 <= T - 1 - edge
    margin_ok = (c0[j0] - c0[0] > 2 * se_first) and (c0[j0] - c0[-1] > 2 * se_last)
    return c0, j0, stab, interior, margin_ok, interior and stab >= 0.8 and margin_ok


def chi_w(d, ix):
    g = np.nanmean(d['g_gen'][ix], 2); pr = d['prompts'][ix]           # [n, t] layer-mean g
    df = pd.DataFrame(g); df['p'] = pr
    v = df.groupby('p').var(ddof=1).dropna(); return v.mean(0).values


def chi_b(d, ix):
    g = np.nanmean(d['g_gen'][ix], 2); df = pd.DataFrame(g); df['p'] = d['prompts'][ix]
    return df.groupby('p').mean().var(ddof=1).values


# ---------- L6 z
def z_exp(G, sig):
    dif = np.abs(G[:-1] - G[-1]); ok = dif > 0
    return np.polyfit(np.log(sig[:-1][ok]), np.log(dif[ok]), 1)[0] if ok.sum() >= 3 else np.nan


if __name__ == '__main__':
    pd.set_option('display.width', 250); L3, L4, L4c, L6 = [], [], [], []
    for lab, key, D, fam in CH:
        L3 += l3(lab, key, D)
        d = load(key); edge, tol = (1, 1) if d['g_gen'].shape[1] <= 8 else (2, 2)
        Cfn = lambda ix: np.nanmean(2 * (d['ent_gen'][ix] - np.log(d['g_gen'][ix] * D)), (0, 2))
        cC, jC, stC, inC, mC, pkC = interior_peak(Cfn, d, edge, tol)
        cX, jX, stX, inX, mX, pkX = interior_peak(lambda ix: chi_w(d, ix), d, edge, tol)
        cb = chi_b(d, np.arange(len(d['prompts'])))
        L4.append(dict(channel=lab, family=fam, steps=len(cC), C_argmax=jC, C_sigma=d['sigma'][jC], C_stab=stC, C_interior=inC, C_margin=mC, C_peak=pkC,
                       chi_argmax=jX, chi_sigma=d['sigma'][jX], chi_stab=stX, chi_interior=inX, chi_margin=mX, chi_peak=pkX,
                       coincide=(abs(jC - jX) <= 2) if (pkC and pkX) else np.nan))
        for t in range(len(cC)):
            L4c.append(dict(channel=lab, step=t, sigma=d['sigma'][t], C_bar=cC[t], chi_w=cX[t], chi_b=cb[t], seed_share=cX[t] / (cX[t] + cb[t])))
        G = np.nanmean(d['g_gen'], (0, 2)); sig = d['sigma']
        for cad, sl in (('native', slice(None)), ('every 2nd step', slice(None, None, -2))):
            Gs, ss = (G[sl], sig[sl]) if cad == 'native' else (G[sl][::-1], sig[sl][::-1])
            z0 = z_exp(Gs, ss); B = []
            for _ in range(NB):
                Gb = np.nanmean(d['g_gen'][prompt_boot_idx(d['prompts'], rng)], (0, 2)); B.append(z_exp(Gb[sl] if cad == 'native' else Gb[sl][::-1], ss))
            L6.append(dict(channel=lab, family=fam, cadence=cad, steps=len(Gs), z=z0, se=np.nanstd(B, ddof=1)))
        print(lab, 'done', flush=True)
    df3 = pd.DataFrame(L3); df3.to_csv(os.path.join(OUT, 'L3_deformation.csv'), index=False)
    s3 = df3.groupby(['channel', 'kind']).agg(median_score=('score', 'median'), feasible=('feasible', 'mean'), h2_norm_first=('h2_norm', 'first')).unstack()
    print(s3.round(4).to_string())
    for name, rows in (('L4_peaks', L4), ('L4_curves', L4c), ('L6_z_exponent', L6)):
        df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, f'{name}.csv'), index=False)
        if name != 'L4_curves': print(f'\n{name}'); print(df.round(4).to_string(index=False))
