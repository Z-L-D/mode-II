"""II-F11 Estimator mathematics for seed predictability.
(a) DATA: sigma50 (first sigma where cross-validated R^2(state -> final state) passes 0.5) and the relative onset on the same
    curves, for nested subsets of the SD3.5 M archive (branch-mean capture), by number of generations
    (review/round5/out/H1_seedfate_vs_sample_size.csv). sigma50 is a threshold crossing, so it moves with the R^2 ceiling and n.
(b) DATA: Round 6 S3 (review/round6/out/S3_estimators.csv), all seven archive channels at matched n = 60 generations:
    sigma50 and relative onset (median and p5-p95 over fold partitions).
(c) DATA: VR(4) of the seed-deviation residuals against finite-length references simulated at each channel's own length
    (i.i.d., AR(1) 0.8, random walk; S3_estimators.csv). Estimator calibration only; no commitment or separatrix reading.
"""
from m2common import *

FIG = 'II-F11_estimators'
H1 = pd.read_csv(f'{R5}/H1_seedfate_vs_sample_size.csv')
H1 = H1[H1.set.str.startswith('archive SD35M')].copy()
S3 = pd.read_csv(f'{R6}/S3_estimators.csv')
lab = {'ZIT-attn': 'ZIT attn', 'ZIT-mlp': 'ZIT MLP', 'ZIB-attn': 'ZIT-25 attn', 'ZIB-mlp': 'ZIT-25 MLP', 'MOCHI-attn': 'Mochi attn',
       'SD35M-attn': 'SD3.5 M attn', 'SD35L-attn': 'SD3.5 L attn'}
mdl = lambda ch: ch.split('-')[0]

fig = plt.figure(figsize=(TEXTWIDTH - 0.2, 2.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 0.8], wspace=0.5, left=0.08, right=0.99)
ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[0, 2], sharey=ax2)

H1['tag'] = H1.prompts.astype(str) + '×' + H1.runs_per_prompt.astype(str)
xs = np.arange(len(H1))
o = np.argsort(H1.n.values, kind='stable'); H1 = H1.iloc[o].reset_index(drop=True)
for col, lo, hi, mk, mfc, name in [('sigma50', 'sigma50_p5', 'sigma50_p95', 'o', INK2, r'$\sigma_{50}$'),
                                   ('rel_onset', 'rel_p5', 'rel_p95', 'D', 'white', 'relative onset')]:
    dx = -0.12 if col == 'sigma50' else 0.12
    ax1.vlines(xs + dx, H1[lo], H1[hi], color=INK2, lw=0.8)
    ax1.plot(xs + dx, H1[col], mk, ms=4, mfc=mfc, mec=INK2, mew=1.0, ls='none', label=name)
u = H1.sigma50_undefined_frac.iloc[0]
ax1.annotate(f'undefined in {u:.0%}\nof partitions', xy=(0 - 0.12, H1.sigma50.iloc[0]), xytext=(0.6, 0.25), fontsize=6.3, color=INK,
             arrowprops=dict(arrowstyle='-', color=INK2, lw=0.5))
ax1.set_xticks(xs); ax1.set_xticklabels([f'{t}\n{n}' for t, n in zip(H1.tag, H1.n)], fontsize=5.8)
ax1.set_xlabel('prompts × runs, $n$ generations'); ax1.set_ylabel(r'$\sigma$'); ax1.set_ylim(0, 1.05); ax1.grid(axis='x', visible=False)
ax1.legend(loc='lower right', fontsize=6.5, handletextpad=0.2, borderaxespad=0.1)
panel_label(ax1, 'a')

y = np.arange(len(S3))[::-1]
for yi, r in zip(y, S3.itertuples()):
    ax2.plot([r.matched60_s50_p5, r.matched60_s50_p95], [yi + 0.15] * 2, color=INK2, lw=0.8)
    ax2.plot(r.matched60_s50_median, yi + 0.15, 'o', ms=4, color=INK2)
    ax2.plot([r.matched60_rel_p5, r.matched60_rel_p95], [yi - 0.15] * 2, color=MODEL_COLOR[mdl(r.channel)], lw=0.8)
    ax2.plot(r.matched60_rel_median, yi - 0.15, 'D', ms=3.8, mfc='white', mec=MODEL_COLOR[mdl(r.channel)], mew=1.0)
ax2.set_yticks(y); ax2.set_yticklabels(S3.channel.map(lab), fontsize=7); ax2.set_xlim(0.3, 1.02); ax2.set_ylim(-0.6, len(S3) + 0.55)
ax2.set_xlabel(r'$\sigma$ at matched $n$ = 60'); ax2.grid(axis='y', visible=False)
rng_rel = S3.matched60_rel_median.max() - S3.matched60_rel_median.min(); rng_s50 = S3.matched60_s50_median.max() - S3.matched60_s50_median.min()
ax2.plot([], [], 'o', ms=4, color=INK2, ls='none', label=r'$\sigma_{50}$')
ax2.plot([], [], 'D', ms=3.8, mfc='white', mec=INK2, ls='none', label='relative onset')
ax2.legend(loc='lower left', fontsize=6.3, handletextpad=0.2, borderaxespad=0.1)
panel_label(ax2, 'b')

for yi, r in zip(y, S3.itertuples()):
    ax3.plot([r.VR4_iid, r.VR4_walk], [yi, yi], color=LIGHT, lw=2.4, solid_capstyle='butt', zorder=1)
    for v, mk in [(r.VR4_iid, '|'), (r.VR4_ar08, '|'), (r.VR4_walk, '|')]:
        ax3.plot(v, yi, mk, ms=7, mew=1.1, color=INK2, zorder=2)
    ax3.plot(r.VR4, yi, 'o', ms=4.6, color=MODEL_COLOR[mdl(r.channel)], zorder=3)
    ax3.text(1.38, yi, f'{r.VR4_over_walk:.2f}', va='center', ha='right', fontsize=6.5, color=INK)
r0 = S3.iloc[0]
for v, t_, dy in [(r0.VR4_iid, 'iid', 0.5), (r0.VR4_ar08, 'AR(0.8)', 0.5), (r0.VR4_walk, 'walk', 1.05)]:
    ax3.text(v, y[0] + dy, t_, ha='center', va='bottom', fontsize=6.3, color=INK2)
ax3.text(1.38, y[0] + 0.5, '/walk', ha='right', va='bottom', fontsize=6.3, color=INK2)
plt.setp(ax3.get_yticklabels(), visible=False); ax3.tick_params(axis='y', length=0); ax3.set_xlim(0.15, 1.42)
ax3.set_xlabel('variance ratio VR(4)'); ax3.grid(axis='y', visible=False)
panel_label(ax3, 'c')

rows = [pd.DataFrame(dict(panel='a', series=H1.set, x=H1.n, sigma50=H1.sigma50, sigma50_p5=H1.sigma50_p5, sigma50_p95=H1.sigma50_p95,
                          sigma50_undefined_frac=H1.sigma50_undefined_frac, rel_onset=H1.rel_onset, rel_p5=H1.rel_p5, rel_p95=H1.rel_p95,
                          final_R2=H1.final_R2, source='review/round5/out/H1_seedfate_vs_sample_size.csv')),
        S3[['channel', 'matched60_s50_median', 'matched60_s50_p5', 'matched60_s50_p95', 'matched60_rel_median', 'matched60_rel_p5',
            'matched60_rel_p95', 'native_s50_median', 'native_rel_median', 'VR4', 'VR4_iid', 'VR4_ar08', 'VR4_walk', 'VR4_over_walk']]
        .rename(columns={'channel': 'series'}).assign(panel='b,c', source='review/round6/out/S3_estimators.csv')]
write_data(FIG, rows)
save(fig, FIG)
print(FIG, 'range rel', round(rng_rel, 3), 's50', round(rng_s50, 3))
