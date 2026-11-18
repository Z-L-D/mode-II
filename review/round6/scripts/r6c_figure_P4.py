# Fig. 8 (post hoc P4): time-averaged depth profile of layer g. Left: Z-Image MLP (archive ZIT, ZIB; clean ZIB CFG4 cond). Right: attention, five models.
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from r6c_common import *
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']; INK2 = '#52514e'; SURF = '#fcfcfb'; MC = dict(zip(MODELS, C))
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#c9c8c3', 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.grid': True,
                     'grid.color': '#ecebe7', 'grid.linewidth': 0.8, 'axes.titlesize': 11, 'axes.titleweight': 'bold'})
D = pd.read_csv(f'{OUT6}/P4_depth_topology.csv').set_index('set')
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
z4 = R5('Z40_1024_s1001', 'mlp')
for lab, X, c in [('ZIT (archive)', arch('ZIT', 'mlp')[2], C[0]), ('ZIB (archive)', arch('ZIB', 'mlp')[2], C[1]), ('ZIB CFG4 cond (clean)', z4['X']['cond'], C[2])]:
    y = X[..., 0].mean((0, 1)); L = len(y); ax[0].plot(np.arange(L) / (L - 1), np.log10(y), lw=2, color=c, label=lab, marker='o', ms=3)
r = D.loc['archive ZIB-mlp']
for b in [r.break1_depth, r.break2_depth]: ax[0].axvline(b, color=INK2, lw=1, ls='--')
ax[0].text(0.06, 0.95, 'expand', transform=ax[0].transAxes, color=INK2); ax[0].text(0.42, 0.95, 'contract', transform=ax[0].transAxes, color=INK2); ax[0].text(0.83, 0.95, 're-expand', transform=ax[0].transAxes, color=INK2)
ax[0].set(title='Z-Image MLP: three phases along depth', xlabel='normalized depth (layer / last layer)', ylabel='log10 g, mean over σ and generations'); ax[0].legend(frameon=False, fontsize=8, loc='lower center')
for m in MODELS:
    y = arch(m, 'attn')[2][..., 0].mean((0, 1)); L = len(y); ax[1].plot(np.arange(L) / (L - 1), y / y.max(), lw=2, color=MC[m], label=f"{m} ({D.loc[f'archive {m}-attn','topology_time_avg']})")
ax[1].set(title='Attention: no shared three-phase depth profile', xlabel='normalized depth (layer / last layer)', ylabel='g / channel max'); ax[1].legend(frameon=False, fontsize=8, loc='upper left')
fig.tight_layout(); fig.savefig(f'{FIG6}/R6C_fig8_depth_profiles.png', dpi=160); plt.close(fig)
