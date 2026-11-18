"""Summary figure for the lens round: out/LENS_summary.pdf/.png
(a) the triple in (H2/lnD, H1/lnD) for ZIB attn and MLP rows, with the PL / EXP / SB family curves at D = 3840
(b) R1 autonomy: out-of-sample R2 of beta_layer on g alone vs g + layer
(c) spectral heat capacity C = 2(H1 - H2) along sigma (layer-mean, channel-mean)
(d) within-prompt variance of logit g along sigma, scaled to its maximum (post hoc PH3 normalisation)"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import sys, numpy as np, pandas as pd
sys.path.insert(0, f'{_COSPAFS_ROOT}/papers/common')
from figstyle import *
from lens_common import CH, OUT, load, triple, Family
COL = {'ZIT': MODEL_COLOR['ZIT'], 'ZIB': MODEL_COLOR['ZIB'], 'Mochi': MODEL_COLOR['MOCHI'], 'SD3.5 M': MODEL_COLOR['SD35M'], 'SD3.5 L': MODEL_COLOR['SD35L']}
col = lambda ch: COL[ch.rsplit(' ', 1)[0]]; ls = lambda ch: '--' if ch.endswith('MLP') else '-'
rng = np.random.default_rng(0)
fig, ax = plt.subplots(2, 2, figsize=(TEXTWIDTH, 5.0)); ax = ax.ravel()

for lab, key, mk, cc in (('ZIB attn', 'ZIB_attn', 'o', MODEL_COLOR['ZIB']), ('ZIB MLP', 'ZIB_mlp', 's', '#8a3c12')):
    H1, H2, L = triple(load(key), 3840); i = rng.choice(H1.size, 3000, replace=False)
    ax[0].scatter(H2.ravel()[i] / np.log(3840), H1.ravel()[i] / np.log(3840), s=2, alpha=0.25, color=cc, marker=mk, lw=0, label=lab)
for k, c, s_ in (('PL', INK, '-'), ('EXP', INK2, ':'), ('SB', GRAY, '--')):
    f = Family(3840, k); ax[0].plot(f.H2s / np.log(3840), f.H1s / np.log(3840), s_, color=c, lw=1.1, label=f'{k} family')
ax[0].plot([0, 1], [0, 1], color=LIGHT, lw=0.8)
ax[0].set(xlim=(0, 0.9), ylim=(0, 1), xlabel=r'$H_2/\ln D$  ($=\ln(gD)/\ln D$)', ylabel=r'$H_1/\ln D$  (sv entropy)')
ax[0].legend(loc='lower right', markerscale=4, fontsize=6.5); panel_label(ax[0], 'a')

R = pd.read_csv(f'{OUT}/L2_R1_autonomy.csv'); x = np.arange(len(R))
ax[1].bar(x - 0.2, R.R2_g, 0.38, color=GRAY, label=r'$\beta = f(g)$')
ax[1].bar(x + 0.2, R.R2_g_layer, 0.38, color=[col(c) for c in R.channel], label=r'$\beta = f(g)$ + layer')
ax[1].set_xticks(x, [c.replace('SD3.5 ', 'SD3.5\n').replace(' attn', '\nattn').replace(' MLP', '\nMLP') for c in R.channel], fontsize=6.3)
ax[1].set(ylabel='out-of-sample $R^2$ (prompt-grouped CV)', ylim=(0, 1)); ax[1].legend(loc='upper left', fontsize=6.5); panel_label(ax[1], 'b')

C4 = pd.read_csv(f'{OUT}/L4_curves.csv')
for ch, g in C4.groupby('channel', sort=False):
    ax[2].plot(g.sigma, g.C_bar, ls(ch), color=col(ch), lw=1.2, label=ch)
ax[2].invert_xaxis(); ax[2].set(xlabel=r'$\sigma$ (sampling runs right)', ylabel=r'$\bar C = 2(H_1 - H_2)$'); ax[2].legend(fontsize=6, ncol=2); panel_label(ax[2], 'c')

for lab, key, D, fam in CH:
    d = load(key); g = d['g_gen']; lg = np.nanmean(np.log(g / (1 - g)), 2); df = pd.DataFrame(lg); df['p'] = d['prompts']
    v = df.groupby('p').var(ddof=1).dropna().mean(0).values
    ax[3].plot(d['sigma'], v / v.max(), ls(lab), color=col(lab), lw=1.2, label=lab)
ax[3].text(0.02, 0.03, 'colours and line styles as in (c)', transform=ax[3].transAxes, fontsize=6.3, color=INK2); ax[3].invert_xaxis(); ax[3].set(xlabel=r'$\sigma$', ylabel=r'within-prompt var of logit $g$ (/max)', ylim=(0, 1.05)); panel_label(ax[3], 'd')
fig.tight_layout(); fig.savefig(f'{OUT}/LENS_summary.pdf'); fig.savefig(f'{OUT}/LENS_summary.png', dpi=200); print('saved')
