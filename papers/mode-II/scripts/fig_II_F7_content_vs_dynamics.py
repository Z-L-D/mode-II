"""II-F7 Content geometry and dynamics are distinct quotients.
(a) DATA: per model pair (10 pairs), prompt RSA at the first step (Round 6 S2, review/round6/out/S2_rsa.csv; Mantel rho over
    117x117 prompt-distance matrices, permutation p 0.0005 for every pair) against the depth-derivative tangent cosine
    (Round 6 S1, review/round6/out/S1_hierarchy.csv, column L3_tangent). Spearman over the 10 pairs (preregistered S2g).
(b) DATA: RSA at the first, middle and last step for each pair, with the 95th percentile of each pair's permutation null.
"""
from m2common import *
from scipy.stats import spearmanr

FIG = 'II-F7_content_vs_dynamics'
S2 = pd.read_csv(f'{R6}/S2_rsa.csv'); S1 = pd.read_csv(f'{R6}/S1_hierarchy.csv')
first = S2[S2.tau == 'first'].merge(S1[['a', 'b', 'L3_tangent']], on=['a', 'b'])
rho = float(spearmanr(first.L3_tangent, first.rsa).statistic)

fig = plt.figure(figsize=(TEXTWIDTH, 2.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.35)
ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
kc = lambda k: PAIR_COLOR[k]
for r in first.itertuples():
    ax1.plot(r.L3_tangent, r.rsa, 'o', ms=5 if r.kind == 'lineage' else 4.2, color=kc(r.kind), mec='white', mew=0.5, zorder=3)
off = {('ZIT', 'ZIB'): (-0.04, 0, 'right'), ('SD35M', 'SD35L'): (-0.03, 0.035, 'left'), ('ZIT', 'MOCHI'): (0.04, 0.0, 'left')}
for r in first.itertuples():
    if (r.a, r.b) in off:
        dx, dy, ha = off[(r.a, r.b)]
        ax1.text(r.L3_tangent + dx, r.rsa + dy, f'{SHORT[r.a]}↔{SHORT[r.b]}', fontsize=6.5, color=INK, ha=ha, va='center')
ax1.set_xlabel('tangent cosine (S1, depth derivative)'); ax1.set_ylabel('prompt RSA, first step (Mantel ρ)')
ax1.set_xlim(-0.35, 1.05); ax1.set_ylim(0, 0.7)
ax1.text(0.03, 0.97, f'Spearman ρ = {rho:.3f} (10 pairs)\nall RSA p = 0.0005', transform=ax1.transAxes, ha='left', va='top', fontsize=7, color=INK)
panel_label(ax1, 'a')

xt = {'first': 0, 'middle': 1, 'last': 2}
for (a, b), d in S2.groupby(['a', 'b'], sort=False):
    k = d.kind.iloc[0]; x = d.tau.map(xt)
    ax2.plot(x, d.rsa, color=kc(k), lw=1.6 if k == 'lineage' else 0.9, marker='o', ms=2.8, zorder=3 if k != 'unrelated' else 2)
ax2.fill_between([0, 1, 2], 0, S2.groupby('tau').null_p95.max().reindex(['first', 'middle', 'last']).values, color=LIGHT, lw=0, zorder=1)
ax2.text(2.05, 0.03, 'null p95', fontsize=6.5, color=INK2, va='center')
lin = S2[S2.kind == 'lineage'].set_index('tau').rsa; fam = S2[S2.kind == 'same family'].set_index('tau').rsa
ax2.text(2.05, lin['last'], 'ZIT-25↔ZIT', fontsize=6.5, color=INK, va='center')
ax2.text(2.05, fam['last'], 'SD3.5 M↔L', fontsize=6.5, color=INK, va='center')
ax2.text(2.05, 0.19, '8 unrelated', fontsize=6.5, color=INK2, va='center')
ax2.set_xticks([0, 1, 2]); ax2.set_xticklabels(['first', 'middle', 'last']); ax2.set_xlim(-0.15, 2.75); ax2.set_ylim(0, 0.7)
ax2.set_xlabel('sampling step'); ax2.set_ylabel('prompt RSA (Mantel ρ)'); ax2.grid(axis='x', visible=False)
panel_label(ax2, 'b')

rows = [pd.DataFrame(dict(panel='a', series=first.kind, pair=first.a + '-' + first.b, x=first.L3_tangent, y=first.rsa, p=first.p,
                          source='review/round6/out/S2_rsa.csv (tau=first) + review/round6/out/S1_hierarchy.csv (L3_tangent)')),
        pd.DataFrame(dict(panel='a', series=['spearman rho'], y=[rho], source=['computed here; preregistered S2g reports 0.297 (S2_results.json)'])),
        pd.DataFrame(dict(panel='b', series=S2.kind, pair=S2.a + '-' + S2.b, x=S2.tau, y=S2.rsa, null_p95=S2.null_p95, p=S2.p,
                          source='review/round6/out/S2_rsa.csv'))]
write_data(FIG, rows)
save(fig, FIG)
print(FIG, 'spearman', round(rho, 3))
