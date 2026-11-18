"""II-F1 Mathematical observable stack.
(a) ILLUSTRATION: a simulated token matrix X (N x D) with a non-zero token mean, as captured uncentred.
(b) ILLUSTRATION: the amplitude spectrum s_i = sqrt(eig(X^T X / N)) of that matrix, with g, gap, H computed from it.
(c) DATA: the observed field g(u, sigma) for Z-Image Base MLP (archive, pooled CFG 4 capture), mean over 476 generations,
    on normalized depth u = layer/(L-1) and the sampling clock sigma.
Definitions: g = (sum s)^2 / (sum s^2) / D; gap = s1/s2; H = entropy of s / sum s.
"""
from m2common import *
from matplotlib.colors import LinearSegmentedColormap

FIG = 'II-F1_observable_stack'
rng = np.random.default_rng(11)

# ---- simulated token matrix (illustration only) ----
N, D = 256, 64
mu = rng.normal(0, 1, D) * 0.9                       # shared token-mean direction (uncentred capture keeps it)
lam = 1.0 / (1 + np.arange(D)) ** 1.1                # decaying fluctuation spectrum
Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
Z = rng.normal(size=(N, D)) * np.sqrt(lam) @ Q.T
X = mu[None, :] + Z
M = X.T @ X / N
g, gap, H = amp_stats(M)
s = np.sqrt(np.clip(np.linalg.eigvalsh(M), 0, None))[::-1]
Mc = np.cov(X.T, bias=True)
g_c, gap_c, H_c = amp_stats(Mc)

# ---- observed field (data) ----
_, meta, Xz, sig = cube('ZIB_mlp')
F = np.log10(Xz[..., 0]).mean(0)                     # [T, L]
u = np.arange(F.shape[1]) / (F.shape[1] - 1)

fig = plt.figure(figsize=(TEXTWIDTH, 2.35))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.35], wspace=0.55)
ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]

div = LinearSegmentedColormap.from_list('div', [C[0], '#f4f3f0', C[1]])
v = np.abs(X[:64]).max()
ax1.imshow(X[:64], cmap=div, vmin=-v, vmax=v, aspect='auto', interpolation='nearest')
ax1.set_xlabel('channel $j$ (of $D$)'); ax1.set_ylabel('token $i$ (of $N$)')
ax1.set_xticks([0, 63]); ax1.set_xticklabels(['1', '$D$']); ax1.set_yticks([0, 63]); ax1.set_yticklabels(['1', '64'])
ax1.grid(False)
panel_label(ax1, 'a')

i = np.arange(1, D + 1)
ax2.plot(i, s / s.sum(), color=INK2, lw=1.0, marker='o', ms=2.2, mfc=INK2, mec='none')
ax2.set_yscale('log'); ax2.set_xlabel(r'$i$, with $s_i^2$ = eig$(X^{\top}X/N)$'); ax2.set_ylabel(r'$s_i/\Sigma s$')
ax2.annotate('$s_1$ (token-mean\ndirection)', xy=(1, s[0] / s.sum()), xytext=(14, s[0] / s.sum() * 0.75),
             fontsize=7, color=INK2, va='center', arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
ax2.text(0.97, 0.72, f'$g$ = {g:.3f}\ngap = $s_1/s_2$ = {gap:.1f}\n$H$ = {H:.2f} nats', transform=ax2.transAxes, ha='right', va='top', fontsize=7, color=INK)
ax2.text(0.97, 0.47, f'centred, for contrast:\n$g$ = {g_c:.3f}, gap = {gap_c:.1f}', transform=ax2.transAxes,
         ha='right', va='top', fontsize=6.5, color=INK2)
panel_label(ax2, 'b')

seq = LinearSegmentedColormap.from_list('seq', ['#eef4fb', C[0], '#123a6b'])
pc = ax3.pcolormesh(u, sig, F, cmap=seq, shading='nearest', rasterized=True)
ax3.set_ylim(sig.min() - 0.02, sig.max() + 0.02)
ax3.set_xlabel('normalized depth $u$'); ax3.set_ylabel(r'clock $\tau$: $\sigma$ (runs downward)')
ax3.grid(False)
cb = fig.colorbar(pc, ax=ax3, fraction=0.06, pad=0.03); cb.set_label(r'$\log_{10} g$', color=INK); cb.outline.set_linewidth(0.5)
cb.ax.tick_params(labelsize=7, colors=INK2)
panel_label(ax3, 'c')

rows = [pd.DataFrame(dict(panel='b', series='simulated spectrum', x=i, y=s / s.sum(),
                          source='illustration: simulated X (seed 11), N=256, D=64'))]
rows.append(pd.DataFrame(dict(panel='b', series=['g', 'gap', 'H', 'g_centred', 'gap_centred'], x=np.nan,
                              y=[g, gap, H, g_c, gap_c], source='illustration: computed from the simulated X')))
T, L = F.shape
rows.append(pd.DataFrame(dict(panel='c', series='ZIT-25 MLP mean log10 g', x=np.tile(u, T), sigma=np.repeat(sig, L),
                              y=F.ravel(), source=f'raw cube F_ZIB_mlp.pkl (archive, pooled CFG 4), {Xz.shape[0]} generations, mean over generations')))
write_data(FIG, rows)
save(fig, FIG)
print(FIG, 'g', round(g, 4), 'gap', round(gap, 2), 'H', round(H, 3))
