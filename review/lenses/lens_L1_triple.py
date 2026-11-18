"""L0 sanity and L1 geometry of the spectral triple (LENS_PREREGISTRATION.md).
Diffusion: 7 archive channels from sw_cache (analyzer DataManager). LLM: 7 archive captures (generated tokens; rank = seq_len).
Output: out/L0_sanity.csv, out/L1_dimension.csv, out/L1_families.csv, out/L1_bulk_exponent.csv, out/L1_llm.csv"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import os, glob, numpy as np, pandas as pd
from lens_common import CH, OUT, load, triple, Family, SPL
rng = np.random.default_rng(20261008)
SCR = os.environ.get('SCR', f'{_COSPAFS_CACHE}')


def pca_shares(H1, H2, L):
    M = np.column_stack([H1, H2, L]); M = (M - M.mean(0)) / M.std(0)
    ev = np.sort(np.linalg.eigvalsh(np.cov(M.T)))[::-1]; return ev / ev.sum()


def families(H1, H2, L, R, lnD):
    out = {}
    for k in ('PL', 'EXP', 'SB'):
        f = Family(R, k); h1, lam, par, inr = f.predict(H2)
        out[k] = dict(H1_err=np.median(np.abs(h1 - H1)) / lnD, Lam_err=np.median(np.abs(lam - L)), in_range=inr.mean(), par_median=np.median(par))
    s = SPL(R); h1, a, sp = s.fit(H2, L); ok = np.isfinite(h1)
    out['SPL'] = dict(H1_err=np.median(np.abs(h1[ok] - H1[ok])) / lnD, Lam_err=0.0, in_range=ok.mean(), par_median=np.nanmedian(a))
    return out, h1, a, sp


san, dim, fam, bulk = [], [], [], []
for lab, key, D, fam_name in CH:
    d = load(key); H1, H2, L = triple(d, D); lnD = np.log(D)
    san.append(dict(channel=lab, rows=H1.size, viol_H1_lt_H2=np.mean(H1 < H2 - 1e-3), viol_Lam_neg=np.mean(L < 0),
                    h1_med=np.median(H1) / lnD, h2_med=np.median(H2) / lnD, Lam_med=np.median(L)))
    sh = pca_shares(H1.ravel(), H2.ravel(), L.ravel())
    dim.append(dict(channel=lab, family=fam_name, PC1=sh[0], PC2=sh[1], PC3=sh[2], PC12=sh[0] + sh[1]))
    sub = rng.choice(H1.size, min(H1.size, 120000), replace=False)            # row subsample for the family scores
    o, _, _, _ = families(H1.ravel()[sub], H2.ravel()[sub], L.ravel()[sub], D, lnD)
    for k, v in o.items(): fam.append(dict(channel=lab, family_arch=fam_name, spectrum=k, **v))
    # P1c: bulk exponent at the final step, channel-mean per layer, median over layers; SE from 200 prompt bootstraps
    s = SPL(D); pr = d['prompts']
    def a_final(ix):
        h2 = np.log(np.nanmean(d['g_gen'][ix, -1], 0) * D); lm = np.log(np.nanmean(d['gap_gen'][ix, -1], 0))
        return np.nanmedian(s.fit(h2, lm)[1])
    a0 = a_final(np.arange(len(pr)))
    up = np.unique(pr); idx = {p: np.where(pr == p)[0] for p in up}
    bs = [a_final(np.concatenate([idx[p] for p in rng.choice(up, len(up))])) for _ in range(200)]
    bulk.append(dict(channel=lab, family=fam_name, a_bulk_final=a0, se=np.nanstd(bs, ddof=1)))
    print(lab, 'done', flush=True)

pd.set_option('display.width', 250)
for name, rows in (('L0_sanity', san), ('L1_dimension', dim), ('L1_families', fam), ('L1_bulk_exponent', bulk)):
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, f'{name}.csv'), index=False); print(f'\n{name}'); print(df.round(4).to_string(index=False))

# LLM, descriptive: rank = seq_len per row; families on R = seq_len (grouped to unique R)
llm = []
for path in sorted(glob.glob(f'{SCR}/tel/LLM/*.csv')):
    ch = os.path.basename(path).rsplit('_2026', 1)[0]
    x = pd.read_csv(path, usecols=lambda c: c in ('token_type', 'g_micro', 'spectral_gap', 'sv_entropy', 'seq_len', 'hidden_dim'))
    x = x[(x.token_type == 'generated') & x.g_micro.notna() & np.isfinite(x.spectral_gap)]
    x = x.sample(min(len(x), 40000), random_state=1)
    H1, H2, L = x.sv_entropy.values, np.log(x.g_micro.values * x.hidden_dim.values), np.log(x.spectral_gap.values)
    sh = pca_shares(H1, H2, L); errs = {k: [] for k in ('PL', 'EXP', 'SB', 'SPL')}
    for R, g in x.assign(H1=H1, H2=H2, L=L).groupby('seq_len'):
        if R < 8: continue
        o, *_ = families(g.H1.values, g.H2.values, g.L.values, int(R), np.log(R))
        for k in errs: errs[k].append((o[k]['H1_err'], len(g)))
    rec = dict(channel=ch, rows=len(x), viol_H1_lt_H2=np.mean(H1 < H2 - 1e-3), PC1=sh[0], PC12=sh[0] + sh[1])
    for k, v in errs.items():
        v = np.array(v); rec[f'H1err_{k}'] = np.average(v[:, 0], weights=v[:, 1])
    llm.append(rec); print(ch, 'llm done', flush=True)
df = pd.DataFrame(llm); df.to_csv(os.path.join(OUT, 'L1_llm.csv'), index=False); print(df.round(4).to_string(index=False))
