import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Fig. 9 (Round 6 A4): crossing count of the analyzer's beta_layer(g) at each denoising step, per channel (out/A4_beta_layer_per_step.csv).
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, pandas as pd, numpy as np
R6 = f'{_COSPAFS_ROOT}/review/round6'
INK2 = '#52514e'; SURF = '#fcfcfb'; BLUE = '#2a78d6'; ORANGE = '#eb6834'; GRAY = '#b9b7b1'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#c9c8c3', 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'axes.titlesize': 11, 'axes.titleweight': 'bold'})
d = pd.read_csv(f'{R6}/out/A4_beta_layer_per_step.csv'); s = pd.read_csv(f'{R6}/out/A4_beta_layer_surrogates.csv').set_index('channel')
order = ['SD35M archive attn', 'SD35L archive attn', 'MOCHI archive CFG4.5 attn', 'ZIB archive (pooled CFG4) attn', 'ZIB archive (pooled CFG4) mlp', 'ZIT archive attn', 'ZIT archive mlp']
lab = {'SD35M archive attn': 'SD3.5 M attn', 'SD35L archive attn': 'SD3.5 L attn', 'MOCHI archive CFG4.5 attn': 'Mochi attn',
       'ZIB archive (pooled CFG4) attn': 'ZIB attn', 'ZIB archive (pooled CFG4) mlp': 'ZIB MLP', 'ZIT archive attn': 'ZIT attn', 'ZIT archive mlp': 'ZIT MLP'}
fig, ax = plt.subplots(figsize=(10, 4.4))
for y, ch in enumerate(order):
    g = d[d.channel == ch]
    for r in g.itertuples():
        c = BLUE if r.pattern == '+- -+ +-' else (ORANGE if r.n >= 5 else GRAY)
        ax.scatter(r.sigma, y, s=60, color=c, edgecolors=SURF, linewidths=1.5, zorder=3)
    obs = s.loc[ch, 'observed']; tag = 'all steps: 3 crossings' if obs == '+- -+ +-' else f'all steps: {len(obs.split())} crossing'
    ax.text(1.01, y, tag, va='center', fontsize=8, color=INK2, transform=ax.get_yaxis_transform())
ax.set_yticks(range(len(order))); ax.set_yticklabels([lab[c] for c in order]); ax.invert_yaxis(); ax.set_xlim(1.02, 0.0)
ax.set_xlabel('σ (denoising runs right)'); ax.grid(axis='x', color='#ecebe7')
ax.set_title("Analyzer β_layer(g) at each step: where the + − + − three-crossing pattern appears")
for c, t in [(BLUE, '+ − + − (3 crossings)'), (GRAY, '1 crossing (or 2/4)'), (ORANGE, '5 or more')]:
    ax.scatter([], [], color=c, s=60, label=t)
ax.legend(frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(0, -0.32), ncol=3)
fig.tight_layout(); fig.savefig(f'{R6}/fig/R6C_fig9_analyzer_beta_per_step.png', dpi=160, bbox_inches='tight')
