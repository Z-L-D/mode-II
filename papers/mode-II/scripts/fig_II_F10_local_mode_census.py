"""II-F10 Local linear mode census.
(a,b) DATA, RE-IMPLEMENTED ESTIMATOR (not the ledger's script 14): for each channel x sampling state (first, middle, last
      step) x semantic domain, fit the affine layer map X_{L+1} = A X_L + b by least squares over all generations of that
      domain, with X = [g, -gap, H] (the ledger's rebound coordinates). Eigenvalues of A are invariant to per-coordinate
      affine rescaling. (a) fraction of 100 generation bootstraps in which A has a complex-conjugate pair, averaged over
      domains; (b) dominant eigenvalue modulus |lambda|_max, one dot per domain. Descriptive; no claim of a physical oscillator.
(c)   DATA (ledger, transcribed): geometry-derived Base->Turbo warp, exact permutation test of the local complex-mode
      sequence (script 12): MLP p = 0.00196, attention p = 0.259.
"""
from m2common import *
from matplotlib.colors import LinearSegmentedColormap

FIG = 'II-F10_local_mode_census'
Q12 = 'Using the time warp chosen only from normalized depth geometry, the MLP local complex-mode sequence is much closer than all-time permutation controls (p=0.00196 over all 8! mappings). Attention does not pass the same independent closure (p=0.259).'
Q14 = 'Across start/mid/end × semantic domains, many states have no complex local eigenpair at all; complex fractions range from 0 to 1 depending on architecture and state. The dominant local eigenvalue modulus is much more stable across domains (median across-domain CV ≈2.2%).'
ledger_has(Q12); ledger_has(Q14)
rng = np.random.default_rng(14)
NB = 100
CH = [('ZIT', 'attn'), ('ZIT', 'mlp'), ('ZIB', 'attn'), ('ZIB', 'mlp'), ('MOCHI', 'attn'), ('SD35M', 'attn'), ('SD35L', 'attn')]


def fitA(Xs):
    A0 = Xs[:, :-1].reshape(-1, 3); A1 = Xs[:, 1:].reshape(-1, 3)
    coef, *_ = np.linalg.lstsq(np.c_[A0, np.ones(len(A0))], A1, rcond=None)
    ev = np.linalg.eigvals(coef[:3].T)
    return float(np.abs(ev).max()), bool((np.abs(ev.imag) > 1e-9).any())


rows = []
for m, sub in CH:
    _, meta, X, s = cube(f'{m}_{sub}')
    Y = X.copy(); Y[..., 1] = -Y[..., 1]
    T = X.shape[1]
    for st, t in [('first', 0), ('middle', T // 2), ('last', T - 1)]:
        for dom in sorted(meta.dom.unique()):
            idx = np.where(meta.dom.values == dom)[0]
            mod, cx = fitA(Y[idx, t])
            bc = [fitA(Y[rng.choice(idx, len(idx)), t])[1] for _ in range(NB)]
            rows.append(dict(channel=f'{m}-{sub}', model=m, sub=sub, state=st, sigma=float(s[t]), domain=dom, gens=len(idx),
                             modulus=mod, complex_point=cx, complex_boot_frac=float(np.mean(bc))))
D = pd.DataFrame(rows)
cv = D.groupby(['channel', 'state'], sort=False).modulus.agg(lambda v: v.std(ddof=0) / v.mean())
cx = D.groupby(['channel', 'state'], sort=False).complex_boot_frac.mean()

chs = [f'{m}-{s_}' for m, s_ in CH]
chlab = [f'{SHORT[m]} {"MLP" if s_ == "mlp" else "attn"}' for m, s_ in CH]
states = ['first', 'middle', 'last']

fig = plt.figure(figsize=(TEXTWIDTH, 2.65))
gs = fig.add_gridspec(1, 3, width_ratios=[0.75, 1.55, 0.6], wspace=0.5, left=0.14, right=0.99)
ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]

Mx = np.array([[cx[(c, s_)] for s_ in states] for c in chs])
seq = LinearSegmentedColormap.from_list('seq', ['#f6f5f2', C[0], '#123a6b'])
ax1.imshow(Mx, cmap=seq, vmin=0, vmax=1, aspect='auto')
for i in range(Mx.shape[0]):
    for j in range(3):
        ax1.text(j, i, f'{Mx[i, j]:.2f}', ha='center', va='center', fontsize=6.3, color='white' if Mx[i, j] > 0.55 else INK)
ax1.set_xticks(range(3)); ax1.set_xticklabels(['first', 'mid', 'last'], fontsize=7); ax1.set_yticks(range(len(chs))); ax1.set_yticklabels(chlab, fontsize=7)
ax1.set_xlabel('sampling state'); ax1.grid(False)
for sp in ax1.spines.values(): sp.set_visible(False)
ax1.tick_params(length=0)
panel_label(ax1, 'a')

off = {'first': -0.25, 'middle': 0.0, 'last': 0.25}
mk = {'first': 'o', 'middle': 's', 'last': '^'}
for i, (c, (m, s_)) in enumerate(zip(chs, CH)):
    for st in states:
        v = D[(D.channel == c) & (D.state == st)].modulus
        jit = (np.arange(len(v)) - len(v) / 2) * 0.012
        ax2.plot(i + off[st] + jit, v, mk[st], ms=2.6, color=MODEL_COLOR[m], mec='none', alpha=0.9, ls='none')
ax2.axhline(1.0, color=INK2, lw=0.6, ls=(0, (3, 2)))
ax2.set_xticks(range(len(chs))); ax2.set_xticklabels(chlab, rotation=40, ha='right', fontsize=6.8)
ax2.set_ylabel(r'dominant modulus $|\lambda|_{\max}$'); ax2.grid(axis='x', visible=False); ax2.set_xlim(-0.6, len(chs) - 0.4)
for st in states:
    ax2.plot([], [], mk[st], ms=3.2, color=INK2, ls='none', label=st)
ax2.set_ylim(0.3, 1.34)
ax2.legend(loc='upper right', fontsize=6.3, ncol=3, handletextpad=0.1, columnspacing=0.6, borderaxespad=0.1)
ax2.text(0.02, 0.03, f'dots: 10 domains\nmedian CV {100 * cv.median():.1f}%\n(ledger: ≈2.2%)',
         transform=ax2.transAxes, ha='left', va='bottom', fontsize=6.5, color=INK)
panel_label(ax2, 'b')

pv = [('MLP', 0.00196), ('attention', 0.259)]  # ZIT-25 -> ZIT (same weights; v1 'Base->Turbo')
for k, (lab, p) in enumerate(pv):
    ax3.plot(k, p, 'o', ms=5.5, color=INK2)
    ax3.text(k, p * 1.6, f'{p:g}', va='bottom', ha='center', fontsize=6.8, color=INK)
ax3.axhline(0.05, color=INK2, lw=0.6, ls=(0, (3, 2))); ax3.text(1.55, 0.05, '0.05', va='bottom', ha='right', fontsize=6.3, color=INK2)
ax3.set_yscale('log'); ax3.set_ylim(5e-4, 1); ax3.set_xlim(-0.5, 1.6); ax3.set_xticks([0, 1]); ax3.set_xticklabels([p[0] for p in pv], fontsize=7)
ax3.set_ylabel('exact permutation $p$ (8! maps)'); ax3.grid(axis='x', visible=False)
panel_label(ax3, 'c')

out = [D.assign(panel='a,b', series=D.channel + ' | ' + D.state, source='recomputed from raw cubes F_<model>_<sub>.pkl; re-implemented local-linear census (not ledger script 14)')
       .drop(columns=['channel']),
       pd.DataFrame(dict(panel='b', series=['median across-domain CV (this figure)'], y=[float(cv.median())], source=['computed here'])),
       pd.DataFrame(dict(panel='b', series=['median across-domain CV (ledger)'], y=[0.022], source=[f'ledger card "Complex local modes are regime-specific" (script 14): "{Q14}"'])),
       pd.DataFrame(dict(panel='c', series=['MLP', 'attention'], y=[0.00196, 0.259],
                         source=f'ledger card "Geometry-derived Base→Turbo warp independently closes MLP local eigenmodes but not attention" (script 12): "{Q12}"'))]
write_data(FIG, out)
save(fig, FIG)
print(FIG, 'median CV', round(float(cv.median()), 4)); print(cx.unstack().round(2))
