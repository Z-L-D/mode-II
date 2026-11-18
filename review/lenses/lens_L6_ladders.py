"""L6 / Part B: Fibonacci-Kepler (Popkov-Schutz) and Coxeter 2cos(pi/m) ladders (LENS_PREREGISTRATION.md Part B).
Hit = target within +/-1 SE; chance coverage = P(target within +/-1 SE of a log-uniform value in [x/1.25, 1.25x]);
p = P(Poisson-binomial >= hits). Output: out/L6_ladders.csv, out/L6_ladder_summary.csv"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import os, numpy as np, pandas as pd
from lens_common import OUT
rng = np.random.default_rng(20261008)
F = [1, 1]
while len(F) < 18: F.append(F[-1] + F[-2])
K = np.sort(np.unique([F[n + 1] / F[n] for n in range(1, 16)]))
C = np.sort(np.unique(np.r_[[2 * np.cos(np.pi / m) for m in range(3, 31)], 2.0]))
SA = f'{_COSPAFS_ROOT}/review/salem_avalanche/out'


def coverage(x, se, T, n=20000):
    u = np.exp(rng.uniform(np.log(x / 1.25), np.log(x * 1.25), n)); i = np.searchsorted(T, u)
    lo, hi = T[np.clip(i - 1, 0, len(T) - 1)], T[np.clip(i, 0, len(T) - 1)]
    return np.mean(np.minimum(abs(u - lo), abs(u - hi)) <= se)


def pb_sf(c, k, n=200000): return np.mean((rng.random((n, len(c))) < c).sum(1) >= k) if len(c) else np.nan


z = pd.read_csv(f'{OUT}/L6_z_exponent.csv'); zn = z[z.cadence == 'native']
obs = {'z': zn.assign(observable='z (sigma relaxation)')[['channel', 'family', 'observable', 'z', 'se']].rename(columns={'z': 'value'})}
O1 = pd.read_csv(f'{SA}/SA_O1_crossing_ratios.csv'); fam = lambda c: 'Z-Image' if c.startswith('ZI') else 'SD3.5' if c.startswith('SD') else 'Mochi'
obs['crossing ratio'] = O1.assign(family=O1.channel.map(fam))[['channel', 'family', 'observable', 'value', 'se']]
a = pd.read_csv(f'{OUT}/L1_bulk_exponent.csv'); obs['a_bulk'] = a.assign(observable='a_bulk')[['channel', 'family', 'observable', 'a_bulk_final', 'se']].rename(columns={'a_bulk_final': 'value'})
y = pd.read_csv(f'{OUT}/L2_R2_slopes.csv'); obs['|y|'] = y.assign(observable='|y| crossing ' + y.k.astype(str), value=y.slope_y.abs())[['channel', 'family', 'observable', 'value', 'slope_se']].rename(columns={'slope_se': 'se'})
cr = pd.read_csv(f'{OUT}/L2_R3_cross_ratios.csv'); cr = cr[cr.quantity.str.startswith('CR')]
obs['cross-ratio'] = cr.rename(columns={'quantity': 'observable'})[['channel', 'family', 'observable', 'value', 'se']]
dm = pd.read_csv(f'{SA}/SA_O2_dmd.csv'); dm = dm[(dm.cadence == 'native') & (dm.observable == 'omega_over_gamma')]
obs['omega/gamma'] = dm.assign(family=dm.channel.map(lambda c: 'Z-Image' if c.startswith('ZI') else 'SD3.5' if c.startswith('SD') else 'Mochi'))[['channel', 'family', 'observable', 'value', 'se']]

rows = []
for src, tab in obs.items():
    for _, r in tab.iterrows():
        if not (np.isfinite(r.value) and np.isfinite(r.se) and r.value > 0): continue
        for nm, T in (('K Fibonacci-Kepler', K), ('C Coxeter 2cos(pi/m)', C)):
            t = T[np.argmin(abs(T - r.value))]
            rows.append(dict(ladder=nm, source=src, channel=r.channel, family=r.family, observable=r.observable, value=r.value, se=r.se,
                             nearest=t, dist_over_se=abs(r.value - t) / r.se, hit=abs(r.value - t) <= r.se,
                             targets_within_2se=int(np.sum(abs(T - r.value) <= 2 * r.se)), chance=coverage(r.value, r.se, T)))
D = pd.DataFrame(rows); D.to_csv(f'{OUT}/L6_ladders.csv', index=False)
PRIMARY = {'K Fibonacci-Kepler': ['z'], 'C Coxeter 2cos(pi/m)': ['crossing ratio', 'z']}
summ = []
for nm, g in D.groupby('ladder'):
    for scope, sub in (('primary', g[g.source.isin(PRIMARY[nm])]), ('all', g)):
        h = int(sub.hit.sum()); summ.append(dict(ladder=nm, scope=scope, n=len(sub), hits=h, expected=sub.chance.sum(), p=pb_sf(sub.chance.values, h),
                                               median_targets_2se=sub.targets_within_2se.median(), hit_families=sub[sub.hit].family.nunique(),
                                               hits_list='; '.join(f'{r.channel} {r.observable} {r.value:.3f}~{r.nearest:.3f}' for r in sub[sub.hit].itertuples())))
S = pd.DataFrame(summ); S.to_csv(f'{OUT}/L6_ladder_summary.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 200)
print('K =', np.round(K, 4)); print('C =', np.round(C, 4)); print(S.round(4).to_string(index=False))
z2 = z.pivot_table(index='channel', columns='cadence', values=['z', 'se'])
print('\nz cadence check:'); print(z2.round(4).to_string())
