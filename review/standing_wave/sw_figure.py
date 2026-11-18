"""Figure for the standing-wave re-test: (a) dominant depth mode |Psi(L)| with its minima, (b) the mode's amplitude through
denoising and the drift of the time-varying part, (c) node-regularity statistic against the phase-randomised null."""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('FIGSTYLE', f'{_COSPAFS_ROOT}/papers/common'))
from figstyle import *  # noqa
src = open(os.path.join(HERE, 'sw_tests.py')).read().split("CHANNELS = [")[0]
ns = {'__file__': os.path.join(HERE, 'sw_tests.py')}; exec(compile(src, 'sw_tests_lib', 'exec'), ns)
CH = [('ZIT_mlp', 'Z-Image Turbo MLP', C[0]), ('ZIB_mlp', 'Z-Image Base MLP', C[1])]
fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.5), gridspec_kw=dict(wspace=0.42, width_ratios=[1.25, 1, 1]))
P3 = pd.read_csv(os.path.join(HERE, 'out', 'SW_P3_nulls.csv')).set_index('channel')
for ch, lab, col in CH:
    d = ns['load'](ch); L = d['layers']; s = d['sigma']
    r = ns['summarise'](d['G'], d['GAP'], L, s, 'z'); a = np.abs(r['psi']); a = a / np.median(a)
    ax = axs[0]; ax.plot(L, a, color=col, lw=1.4, label=lab); ax.scatter(L[r['mins']], a[r['mins']], s=14, color=col, zorder=3, lw=0)
    A = r['A']; ax = axs[1]
    ax.plot(s, np.abs(A) / np.abs(A).max(), color=col, lw=1.4)
    Z = r['Z']; Cz = Z - Z.mean(0); U, sv, Vh = np.linalg.svd(Cz.T, full_matrices=False); c = sv[0] * Vh[0].conj()
    W = np.array([c.real, c.imag]); ev, evec = np.linalg.eigh(W @ W.T); p = evec[:, -1] @ W; p = p / np.abs(p).max()
    if p[0] < 0: p = -p
    ax.plot(s, p, color=col, lw=1.0, ls='--')
    n1 = [ns['run3'](ns['interior_minima'](np.abs(ns['phase_rand_complex'](r['psi'])))) for _ in range(10000)]
    ax = axs[2]; k = np.arange(0, 8); h = np.array([np.mean(np.array(n1) == i) for i in k])
    off = -0.2 if ch == 'ZIT_mlp' else 0.2
    ax.bar(k + off, h, width=0.38, color=col, alpha=0.55, lw=0)
    ax.annotate('', xy=(r['r3'] + off, 0.02), xytext=(r['r3'] + off, 0.32), arrowprops=dict(arrowstyle='->', color=col, lw=1.3))
    ax.text(r['r3'] + off, 0.34, f"p={P3.loc[ch, 'p_r3_null1']:.3f}", color=INK, fontsize=6.5, ha='center')
for c in ns['CENTRAL']: axs[0].axvline(c, color=GRAY, lw=0.6, ls=':', zorder=1)
axs[0].set_xlabel('layer L'); axs[0].set_ylabel('|Ψ(L)| / median'); axs[0].legend(fontsize=6.5, frameon=False, loc='upper left')
axs[1].invert_xaxis(); axs[1].axhline(0, color=GRAY, lw=0.5); axs[1].set_xlabel('σ'); axs[1].set_ylabel('mode amplitude (norm.)')
axs[1].text(0.97, 0.72, 'solid: |A(σ)|\ndashed: time-varying part', transform=axs[1].transAxes, fontsize=6.3, ha='right', color=INK2)
axs[2].set_xlabel('longest run of ΔL = 3 spacings'); axs[2].set_ylabel('null probability'); axs[2].set_ylim(0, 0.6)
for i, ax in enumerate(axs): panel_label(ax, 'abc'[i])
save(fig, os.path.join(HERE, 'out', 'SW_standing_wave_retest'))
