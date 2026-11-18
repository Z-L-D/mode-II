"""II-F2 Pooling theorem and capture functional.
Theorem (Mode-II §3): Phi(M) = (tr M^{1/2})^2 = D * g(M) * tr M is concave and 1-homogeneous on the PSD cone.
For a pooled matrix M = sum_i pi_i M_i:  g_pool >= sum_i w_i g_i  with  w_i = pi_i tr M_i / tr M  (energy-weighted mean).
With equal branch energies this is g_pool >= mean_i g_i; the weaker corollary sqrt(g_pool) >= mean_i sqrt(g_i)
is what Round 5 Q1e tested. g_pool >= the UNWEIGHTED mean of g_i is NOT implied when energies differ.
(a) ILLUSTRATION: Monte Carlo over random PSD branch pairs with unequal energies.
(b) DATA: identical-trajectory pooled-over-branch-mean increments, SD3.5 M CFG 4.5 (Round 5 Q1c/Q1d).
(c) DATA: the same increment by step at 1024 px (Round 5 Q1c_median_log_g_by_step).
"""
import json
from m2common import *

FIG = 'II-F2_pooling_theorem'
rng = np.random.default_rng(2026)
NSIM = 20000
rec = []
for k in range(NSIM):
    D = int(rng.integers(2, 33))
    Ms = []
    for b in range(2):
        r = int(rng.integers(1, D + 1))
        W = rng.normal(size=(r, D)) * rng.gamma(0.7, 1.0, size=D)          # anisotropic, possibly low rank
        mu = rng.normal(size=D) * rng.uniform(0, 2)                         # uncentred: shared mean component
        M = (W.T @ W) / r + np.outer(mu, mu)
        Ms.append(M * np.exp(rng.normal(0, 1.0)))                           # unequal branch energies
    gi = np.array([amp_stats(M)[0] for M in Ms]); E = np.array([np.trace(M) for M in Ms])
    gp = amp_stats((Ms[0] + Ms[1]) / 2)[0]
    gw = (E * gi).sum() / E.sum()
    rec.append((D, gp, gw, gi.mean(), np.sqrt(gi).mean(), E.max() / E.min()))
S = pd.DataFrame(rec, columns=['D', 'g_pool', 'g_energy_weighted', 'g_unweighted', 'mean_sqrt_g', 'energy_ratio'])
lw_ = np.log(S.g_pool / S.g_energy_weighted); lu = np.log(S.g_pool / S.g_unweighted)
n_w = int((lw_ < -1e-12).sum()); n_u = int((lu < -1e-12).sum())
n_sqrt = int((np.sqrt(S.g_pool) < S.mean_sqrt_g - 1e-12).sum())

Q = json.load(open(f'{R5}/Q_results.json'))
inc = pd.DataFrame(dict(res=['1024', '768', '512'],
                        y=[Q['Q1c_median_log_g_pooled_over_BM_steps>=2'], Q['Q1d_768_median_log_g_pooled_over_BM_steps>=2'],
                           Q['Q1d_512_median_log_g_pooled_over_BM_steps>=2']],
                        pairs=[Q['Q1c_pairs'], Q['Q1d_768_pairs'], Q['Q1d_512_pairs']],
                        gap=[Q['Q1c_median_log_gap_steps>=2'], Q['Q1d_768_median_log_gap_steps>=2'], Q['Q1d_512_median_log_gap_steps>=2']]))
bystep = np.array(Q['Q1c_median_log_g_by_step'])

fig = plt.figure(figsize=(TEXTWIDTH, 2.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.75, 1.2], wspace=0.5)
ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]

bins = np.linspace(-1.0, 0.8, 91)
ax1.hist(lu, bins=bins, histtype='step', color=C[1], lw=1.0)
ax1.hist(lw_, bins=bins, histtype='step', color=C[0], lw=1.0)
ax1.axvline(0, color=INK2, lw=0.7, ls=(0, (3, 2)))
ax1.set_yscale('log'); ax1.set_ylim(0.7, 4e4); ax1.set_xlabel(r'$\log(g_{\rm pool}/\bar g)$'); ax1.set_ylabel('random branch pairs')
from matplotlib.lines import Line2D
hd = [Line2D([], [], color=C[0], lw=1.0), Line2D([], [], color=C[1], lw=1.0)]
ax1.legend(hd, [f'energy-weighted $\\bar g$\n{n_w} of {NSIM:,} below 0', f'unweighted $\\bar g$\n{n_u:,} of {NSIM:,} below 0'],
           loc='upper left', fontsize=6.5, handlelength=1.2, borderaxespad=0.2)
panel_label(ax1, 'a')

x = np.arange(3)
ax2.vlines(x, 0, inc.y, color=INK2, lw=1.0)
ax2.plot(x, inc.y, 'o', color=INK2, ms=4.5)
for xi, yi in zip(x, inc.y):
    ax2.text(xi, yi + 0.004, f'+{yi:.3f}', ha='center', va='bottom', fontsize=7, color=INK)
ax2.set_xticks(x); ax2.set_xticklabels([f'{r}' for r in inc.res]); ax2.set_xlim(-0.6, 2.6)
ax2.set_ylim(0, 0.11); ax2.set_xlabel('resolution (px)'); ax2.set_ylabel(r'median $\log(g_{\rm pool}/g_{\rm BM})$')
ax2.grid(axis='x', visible=False)
panel_label(ax2, 'b')

st = np.arange(len(bystep))
ax3.plot(st, bystep, color=MODEL_COLOR['SD35M'], lw=1.4)
ax3.axhline(0, color=INK2, lw=0.6)
ax3.axvspan(-0.5, 1.5, color=LIGHT, alpha=0.6, lw=0)
ax3.text(2.2, bystep[0] * 0.97, 'steps 0–1 (excluded\nfrom the medians in b)', fontsize=6.5, color=INK2, va='top')
ax3.set_xlabel('sampling step (40-step schedule)'); ax3.set_ylabel(r'median $\log(g_{\rm pool}/g_{\rm BM})$')
ax3.set_xlim(-0.5, 39.5)
ax3.text(13, 0.105, 'SD3.5 M, 1024 px', fontsize=7, color=INK, va='bottom')
panel_label(ax3, 'c')

src_sim = f'illustration: Monte Carlo, {NSIM} random PSD branch pairs (D 2-32, log-normal energy ratio), seed 2026'
rows = [pd.DataFrame(dict(panel='a', series='summary', x=['violations_energy_weighted', 'violations_unweighted', 'violations_sqrt_corollary', 'n_sim'],
                          y=[n_w, n_u, n_sqrt, NSIM], source=src_sim)),
        pd.DataFrame(dict(panel='a', series='histogram of log(g_pool/gbar)', x=(bins[:-1] + bins[1:]) / 2,
                          count_energy_weighted=np.histogram(lw_, bins)[0], count_unweighted=np.histogram(lu, bins)[0], source=src_sim)),
        pd.DataFrame(dict(panel='b', series='median log(g_pooled/g_branch-mean), steps>=2', x=inc.res, y=inc.y, pairs=inc.pairs,
                          median_log_gap=inc.gap,
                          source=['review/round5/out/Q_results.json: Q1c_median_log_g_pooled_over_BM_steps>=2',
                                  'review/round5/out/Q_results.json: Q1d_768_median_log_g_pooled_over_BM_steps>=2',
                                  'review/round5/out/Q_results.json: Q1d_512_median_log_g_pooled_over_BM_steps>=2'])),
        pd.DataFrame(dict(panel='b', series='Q1e sqrt corollary', x=['frac_cells_bound_holds', 'median_slack_log', 'spearman_excess_vs_1-PI'],
                          y=[Q['Q1e_frac_cells_bound_holds'], Q['Q1e_median_slack_log'], Q['Q1f_spearman_excess_vs_1-PI']],
                          source='review/round5/out/Q_results.json: Q1e_frac_cells_bound_holds, Q1e_median_slack_log, Q1f_spearman_excess_vs_1-PI')),
        pd.DataFrame(dict(panel='c', series='1024 by step', x=st, y=bystep, source='review/round5/out/Q_results.json: Q1c_median_log_g_by_step'))]
write_data(FIG, rows)
save(fig, FIG)
print(FIG, 'violations weighted', n_w, 'unweighted', n_u, 'sqrt', n_sqrt, inc.y.round(3).tolist())
