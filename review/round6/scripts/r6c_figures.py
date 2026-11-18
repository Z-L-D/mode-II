# Round 6 figures (static PNGs for the ledger). Palette: dataviz reference categorical slots 1-5 (validated; contrast WARN
# is handled with direct labels and the ledger tables).
import ast, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from r6c_common import *
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']; GRAY = '#9a9893'; INK = '#0b0b0b'; INK2 = '#52514e'; SURF = '#fcfcfb'
MC = dict(zip(MODELS, C))
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#c9c8c3', 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.grid': True,
                     'grid.color': '#ecebe7', 'grid.linewidth': 0.8, 'axes.titlesize': 11, 'axes.titleweight': 'bold'})
def save(fig, name): fig.tight_layout(); fig.savefig(f'{FIG6}/{name}', dpi=160); plt.close(fig)
# ---- Fig 1: observable hierarchy ----
H = pd.read_csv(f'{OUT6}/S1_hierarchy.csv'); lv = ['L0_D1', 'L0_D2', 'L1_field', 'L2_shape', 'L3_tangent']
lab = ['Scalar envelope\n(g)', 'Scalar envelope\n(log g)', 'Depth × σ field\n(log g)', 'Registered shape\n(g, gap, H)', 'Tangent\n(direction of motion)']
fig, ax = plt.subplots(figsize=(9, 4.6))
for i, k in enumerate(lv):
    u = H[H.kind == 'unrelated'][k]; ax.scatter(np.full(len(u), i) + np.linspace(-0.12, 0.12, len(u)), u, s=36, color=GRAY, zorder=3, label='Unrelated pairs (8)' if i == 0 else None)
    sf = H[H.kind == 'same family'][k]; ax.scatter([i + 0.2], sf, s=50, marker='s', color=C[3], zorder=3, label='SD3.5 Medium ↔ Large' if i == 0 else None)
    l = H[H.kind == 'lineage'][k].iloc[0]; ax.scatter([i], [l], s=90, color=C[0], zorder=4, edgecolor=SURF, linewidth=2, label='ZIB ↔ ZIT (lineage)' if i == 0 else None)
    ax.text(i, 1.08, f'margin {l - u.max():+.2f}', ha='center', color=INK2, fontsize=9)
ax.set_xticks(range(len(lv)), lab); ax.set_ylim(-0.35, 1.15); ax.set_ylabel('Pearson r / cosine'); ax.axhline(0, color='#c9c8c3', lw=1)
ax.set_title('Lineage is invisible in the scalar envelope and decisive in the tangent'); ax.legend(loc='lower left', frameon=False, fontsize=9)
save(fig, 'R6C_fig1_observable_hierarchy.png')
# ---- Fig 2: scalar envelopes all rise ----
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for m in MODELS:
    g, meta, X, s = arch(m); E = X[..., 0].mean((0, 2)); ax.plot(s, E / E.max(), color=MC[m], lw=2); ax.text(s[-1] - 0.01, E[-1] / E.max(), m, color=INK, fontsize=9, ha='right', va='bottom')
ax.invert_xaxis(); ax.set_xlabel('σ (denoising runs right)'); ax.set_ylabel('Layer-mean g / its maximum'); ax.set_title('Attention g rises through denoising in all five models')
save(fig, 'R6C_fig2_scalar_envelopes.png')
# ---- Fig 3: capture regime decides the old ordering ----
S1 = json.load(open(f'{OUT6}/S1_results.json'))['S1d']; keys = [k for k in S1 if k.endswith('| D1')]
fig, ax = plt.subplots(figsize=(9, 4.2)); y = np.arange(len(keys))
for i, k in enumerate(keys):
    v = S1[k]; ax.plot([v['r_MOCHI'], v['r_ZIT']], [i, i], color='#c9c8c3', lw=2, zorder=2)
    ax.scatter(v['r_ZIT'], i, color=C[0], s=60, zorder=3, label='ZIB ↔ ZIT' if i == 0 else None); ax.scatter(v['r_MOCHI'], i, color=C[1], s=60, zorder=3, label='ZIB ↔ Mochi' if i == 0 else None)
ax.set_yticks(y, [k.replace(' | D1', '') for k in keys], fontsize=8.5); ax.invert_yaxis(); ax.set_xlabel('Scalar-envelope r (layer-mean g, as in the 09-15 analysis)')
ax.set_title('The 0.9405 > 0.9325 ordering appears only with pooled ZIB'); ax.legend(frameon=False, loc='lower left', fontsize=9)
save(fig, 'R6C_fig3_capture_regime_ordering.png')
# ---- Fig 4: content RSA ----
RS = pd.read_csv(f'{OUT6}/S2_rsa.csv'); fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
for ax, tau in zip(axs, ['first', 'middle', 'last']):
    M = np.full((5, 5), np.nan)
    for r in RS[RS.tau == tau].itertuples(): i, j = MODELS.index(r.a), MODELS.index(r.b); M[i, j] = M[j, i] = r.rsa
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=0.7); ax.set_xticks(range(5), MODELS, rotation=45, fontsize=8); ax.set_yticks(range(5), MODELS, fontsize=8); ax.grid(False)
    for i in range(5):
        for j in range(5):
            if i != j: ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=8, color='white' if M[i, j] > 0.45 else INK)
    ax.set_title(f'{tau} step')
fig.colorbar(im, ax=axs, shrink=0.8, label='Mantel ρ (117 shared prompts)'); fig.suptitle('Prompt-similarity structure is shared by every model pair; the lineage pair shares most', fontweight='bold', fontsize=11)
fig.savefig(f'{FIG6}/R6C_fig4_content_rsa.png', dpi=160, bbox_inches='tight'); plt.close(fig)
# ---- Fig 5: MLP three-regime on clean captures + attention shared shape ----
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
z4 = R5('Z40_1024_s1001', 'mlp'); _, m3, X3, s3 = R3('Z10_1024_s1001', 'mlp')
for (lb, X, s), col in zip([('ZIB archive (pooled, CFG 4)',) + tuple(arch('ZIB', 'mlp')[2:]), ('ZIB CFG 4 cond branch (R4)', z4['X']['cond'], np.asarray(z4['sigma'], float)),
                            ('ZIB CFG 1 cond-only (R3)', X3, np.asarray(s3, float)), ('ZIT archive (CFG 1)',) + tuple(arch('ZIT', 'mlp')[2:])], C[:4]):
    E = X[..., 0].mean((0, 2)); axs[0].plot(s, E / E[0], color=col, lw=2, marker='o', ms=3, label=lb)
axs[0].invert_xaxis(); axs[0].set_xlabel('σ'); axs[0].set_ylabel('MLP layer-mean g / value at σ = 1'); axs[0].set_title('Z-Image MLP: expand → contract → re-expand'); axs[0].legend(frameon=False, fontsize=8)
for m in MODELS:
    g, meta, X, s = arch(m); E = X[..., 0].mean((0, 2)); axs[1].plot(s, (E - E.min()) / (E.max() - E.min()), color=MC[m], lw=2, label=m)
axs[1].invert_xaxis(); axs[1].set_xlabel('σ'); axs[1].set_ylabel('Attention layer-mean g (min–max scaled)'); axs[1].set_title('Attention, all five models: rise, then a late fall'); axs[1].legend(frameon=False, fontsize=8)
save(fig, 'R6C_fig5_topology.png')
# ---- Fig 6: synchronization vs nulls ----
S5 = pd.read_csv(f'{OUT6}/S5_sync.csv'); fig, ax = plt.subplots(figsize=(8.5, 4)); x = np.arange(len(S5))
ax.vlines(x, S5.warp_null_median, S5.warp_null_p95, color=GRAY, lw=6, label='Warp null, median to p95')
ax.scatter(x, S5.R_peak, color=C[0], s=60, zorder=3, label='Observed R peak'); ax.scatter(x, S5.trend_R_peak, color=C[1], marker='D', s=40, zorder=3, label='Straight-line surrogate')
ax.set_xticks(x, S5.channel, rotation=20); ax.set_ylabel('Peak of R(σ)'); ax.set_title('R peaks are matched by straight lines and mostly by desynchronized drift'); ax.legend(frameon=False, fontsize=8, loc='lower right')
save(fig, 'R6C_fig6_sync_nulls.png')
# ---- Fig 7: estimators at matched n ----
S3 = pd.read_csv(f'{OUT6}/S3_estimators.csv'); fig, axs = plt.subplots(1, 2, figsize=(11, 4)); x = np.arange(len(S3))
for off, (k, col, lb) in zip([-0.12, 0.12], [('s50', C[1], 'σ50 (R² = 0.5)'), ('rel', C[0], 'Relative onset (R²/R²final = 0.5)')]):
    axs[0].errorbar(x + off, S3[f'matched60_{k}_median'], yerr=[S3[f'matched60_{k}_median'] - S3[f'matched60_{k}_p5'], S3[f'matched60_{k}_p95'] - S3[f'matched60_{k}_median']], fmt='o', color=col, capsize=3, label=lb)
axs[0].set_xticks(x, S3.channel, rotation=30, fontsize=8); axs[0].set_ylabel('σ at onset (60 prompts, 50 subsets)'); axs[0].set_title('Seed fate at matched n'); axs[0].legend(frameon=False, fontsize=8)
axs[1].bar(x, S3.VR4, color=C[0], width=0.6); [axs[1].axhline(S3[k].mean(), color=c, lw=1.5, ls=ls) for k, c, ls in [('VR4_walk', INK2, '-'), ('VR4_ar08', INK2, '--'), ('VR4_iid', INK2, ':')]]
for k, t in [('VR4_walk', 'walk'), ('VR4_ar08', 'AR(0.8)'), ('VR4_iid', 'jitter')]: axs[1].text(len(S3) - 0.5, S3[k].mean(), t, color=INK2, fontsize=8, va='bottom', ha='right')
axs[1].set_xticks(x, S3.channel, rotation=30, fontsize=8); axs[1].set_ylabel('VR(4)'); axs[1].set_title('Persistence against simulated references')
save(fig, 'R6C_fig7_estimators.png')
print('figures written')
