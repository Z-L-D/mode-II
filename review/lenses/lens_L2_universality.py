"""Applies the preregistered R2 and R3 decision rules to out/L2_R2_slopes.csv and out/L2_R3_cross_ratios.csv.
R2: for each crossing index k, cross-family pairs whose slopes agree within 2 combined SE.
R3: cross-family pairs where a cross-ratio agrees within 2 SE while BOTH raw ratios r21 and r31 disagree (> 2 SE).
Also (T5 of LIT_LENSES, post hoc): the same R3 check with crossings expressed in H2 = ln(g D) (delta-method SEs, covariances ignored).
Output: out/L2_universality.csv"""
import itertools, numpy as np, pandas as pd
from lens_common import OUT, CH
S = pd.read_csv(f'{OUT}/L2_R2_slopes.csv'); C = pd.read_csv(f'{OUT}/L2_R3_cross_ratios.csv'); Dw = {c[0]: c[2] for c in CH}
rows = []
for k, g in S.groupby('k'):
    for a, b in itertools.combinations(g.itertuples(), 2):
        if a.family == b.family: continue
        z = abs(a.slope_y - b.slope_y) / np.hypot(a.slope_se, b.slope_se)
        rows.append(dict(test='R2 slope', item=f'crossing {k}', a=a.channel, b=b.channel, va=a.slope_y, vb=b.slope_y, z=z, agree=z < 2))
P = C.pivot_table(index='channel', columns='quantity', values=['value', 'se']); fam = C.groupby('channel').family.first()
def cr(a, b, c, d): return (c - a) * (d - b) / ((c - b) * (d - a))
for a, b in itertools.combinations(P.index, 2):
    if fam[a] == fam[b]: continue
    zr = {q: abs(P.loc[a, ('value', q)] - P.loc[b, ('value', q)]) / np.hypot(P.loc[a, ('se', q)], P.loc[b, ('se', q)]) for q in ('r21', 'r31')}
    for q in ('CR1(0,g1;g2,g3)', 'CR2(g1,g2;g3,1)'):
        z = abs(P.loc[a, ('value', q)] - P.loc[b, ('value', q)]) / np.hypot(P.loc[a, ('se', q)], P.loc[b, ('se', q)])
        rows.append(dict(test='R3 cross-ratio (g)', item=q, a=a, b=b, va=P.loc[a, ('value', q)], vb=P.loc[b, ('value', q)], z=z, agree=z < 2,
                         z_r21=zr['r21'], z_r31=zr['r31'], rule_met=(z < 2) and zr['r21'] > 2 and zr['r31'] > 2))
# post hoc: cross-ratios in H2 coordinates (0 -> ln D... use the H2 images of g*, with endpoints H2(g->0) undefined, so use CR(g1,g2;g3,H2(1)=lnD) only)
for a, b in itertools.combinations(sorted(S[S.k == 3].channel.unique()), 2):
    if fam[a] == fam[b]: continue
    vals = []
    for ch in (a, b):
        s = S[S.channel == ch].sort_values('k'); h = np.log(s.g_star.values * Dw[ch]); se = s.g_star_se.values / s.g_star.values
        v = cr(h[0], h[1], h[2], np.log(Dw[ch])); eps = 1e-6
        grad = [(cr(*(h + eps * (np.arange(3) == i)), np.log(Dw[ch])) - v) / eps for i in range(3)]
        vals.append((v, np.sqrt(np.sum((np.array(grad) * se) ** 2))))
    z = abs(vals[0][0] - vals[1][0]) / np.hypot(vals[0][1], vals[1][1])
    rows.append(dict(test='R3 post hoc, H2 coords', item='CR(h1,h2;h3,lnD)', a=a, b=b, va=vals[0][0], vb=vals[1][0], z=z, agree=z < 2))
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/L2_universality.csv', index=False)
pd.set_option('display.width', 250); print(df.round(3).to_string(index=False))
