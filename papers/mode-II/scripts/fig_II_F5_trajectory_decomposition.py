"""II-F5 Trajectory decomposition M_{m,p,s}(tau) ~ Gamma_m(tau) + delta_{m,p}(tau) + eps_{m,p,s}(tau).
(a) SCHEMATIC (simulated curves, not data): a backbone Gamma, prompt modulations delta, seed neighbourhoods eps.
(b) DATA (ledger D12, transcribed): nested variance split, system x time scaffold vs theme+prompt+seed.
(c) DATA (Round 2 E6, review/round2/out/E6_varcomp_by_step.csv): prompt share of variance within each
    (model, sampling-state) cell, naive between-prompt share, by sigma. Mochi CFG 1 (one seed per prompt) is omitted.
The components are descriptive and not orthogonal; (b) and (c) answer different questions (global vs within-cell).
(c) also shows the SD1.5 U-Net (post hoc, cond branch, black; review/r3_unet/out/P3_prompt_share_by_step_naive.csv, same E6 code),
    drawn on the rectified-flow clock at equal signal-to-noise ratio, t = sigma_VP / (1 + sigma_VP).
(c) also shows the audio flow transformer ACE-Step 1.5 turbo (post hoc, cond-only, 6 steps, diamonds, dash-dot;
    review/r3_ace/out/P3_prompt_share_by_step.csv, same E6 code) on its own flow-matching sigma.
"""
from m2common import *

FIG = 'II-F5_trajectory_decomposition'
Q12 = ('The order-robust coarse summary assigns 96.16% of normalized depth-geometry variance to the system×time scaffold '
       'and 3.84% to theme+prompt+seed. For absolute operating point the split is 95.55% versus 4.45%.')
ledger_has(Q12)
rng = np.random.default_rng(5)

tau = np.linspace(0, 1, 120)
Gam = 0.35 + 0.5 * (1 - np.exp(-4 * tau)) - 0.15 * tau ** 2
prompts = [0.10 * np.sin(np.pi * tau) + 0.02, -0.08 * np.sin(np.pi * tau * 0.9), 0.05 * tau ** 1.5 - 0.11 * np.sin(np.pi * tau)]

E6 = pd.read_csv(f'{R2}/E6_varcomp_by_step.csv')
E6 = E6[E6.ch != 'MOCHI1-attn']

fig = plt.figure(figsize=(TEXTWIDTH, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.3], wspace=0.55)
ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]

ax1.plot(tau, Gam, color=INK, lw=2.0, zorder=4)
for k, d in enumerate(prompts):
    ax1.plot(tau, Gam + d, color=INK2, lw=0.9, zorder=3)
    for s in range(4):
        w = np.cumsum(rng.normal(0, 0.006, len(tau))) * np.sqrt(tau + 0.05)
        ax1.plot(tau, Gam + d + w, color=GRAY, lw=0.4, alpha=0.6, zorder=2)
ax1.text(1.03, Gam[-1], r'$\Gamma_m(\tau)$', ha='left', va='center', fontsize=8, color=INK)
ax1.annotate(r'$\Gamma_m+\delta_{m,p}$', xy=(0.5, Gam[60] + prompts[0][60]), xytext=(0.18, 0.98), fontsize=7.5, color=INK,
             arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
ax1.annotate(r'$+\,\varepsilon_{m,p,s}$ (seeds)', xy=(0.86, Gam[103] + prompts[2][103] - 0.01), xytext=(0.40, 0.36), fontsize=7.5, color=INK,
             arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
ax1.set_xlabel(r'clock $\tau$'); ax1.set_ylabel('observable (schematic)'); ax1.set_yticklabels([]); ax1.set_ylim(0.3, 1.05); ax1.set_xlim(-0.03, 1.3); ax1.set_xticks([0, 0.5, 1])
ax1.text(0.02, 0.02, 'schematic', transform=ax1.transAxes, fontsize=6.5, color=INK2, style='italic')
panel_label(ax1, 'a')

bars = [('depth\ngeometry', 96.16, 3.84), ('operating\npoint', 95.55, 4.45)]
for k, (lab, sc, ct) in enumerate(bars):
    yk = 1 - k
    ax2.barh(yk, sc, height=0.55, color=LIGHT, edgecolor='white', lw=1.0)
    ax2.barh(yk, ct, left=sc, height=0.55, color=C[0], edgecolor='white', lw=1.0)
    ax2.text(sc / 2, yk, f'{sc:.2f}%', ha='center', va='center', fontsize=7, color=INK)
    ax2.text(101.5, yk, f'{ct:.2f}%', ha='left', va='center', fontsize=7, color=INK)
ax2.text(2, 1.36, 'gray: system×time scaffold', ha='left', va='bottom', fontsize=6.5, color=INK2)
ax2.text(118, -0.36, 'blue: theme+prompt+seed', ha='right', va='top', fontsize=6.5, color=INK2)
ax2.set_yticks([1, 0]); ax2.set_yticklabels([b[0] for b in bars], fontsize=7); ax2.set_xlim(0, 118); ax2.set_ylim(-0.7, 1.65)
ax2.set_xticks([0, 50, 100]); ax2.set_xlabel('share of variance (%)'); ax2.grid(axis='y', visible=False)
panel_label(ax2, 'b')

order = ['ZIT-attn', 'ZIT-mlp', 'ZIB-attn', 'ZIB-mlp', 'MOCHI-attn', 'SD35M-attn', 'SD35L-attn']
for ch in order:
    d = E6[E6.ch == ch]; m, sub = ch.split('-')
    ax3.plot(d.sigma, d.prompt_share_naive, color=MODEL_COLOR[m], lw=1.1, ls='-' if sub == 'attn' else (0, (3, 1.5)),
             label=f'{SHORT[m]} {"MLP" if sub == "mlp" else "attn"}')
U3 = pd.read_csv(f'{REV}/r3_unet/out/P3_prompt_share_by_step_naive.csv')
for sub in ('attn', 'mlp'):
    d = U3[U3.ch == f'SD1.5-{sub}']
    ax3.plot(d.t, d.prompt_share_naive, color=INK, lw=1.1, ls='-' if sub == 'attn' else (0, (3, 1.5)), label=f'SD1.5 {"MLP" if sub == "mlp" else "attn"}')
A3 = pd.read_csv(f'{REV}/r3_ace/out/P3_prompt_share_by_step.csv')
for sub in ('attn', 'mlp'):
    d = A3[A3.ch == f'ACE-{sub}']
    ax3.plot(d.sigma, d.prompt_share_naive, color=INK2, lw=1.0, ls='-.', marker='D', ms=3.2,
             mfc=INK2 if sub == 'attn' else 'white', mec=INK2, mew=0.8, label=f'ACE-Step {"MLP" if sub == "mlp" else "attn"}')
ax3.set_xlim(1.02, 0.0); ax3.set_ylim(0.2, 1.0)
ax3.set_xlabel(r'$\sigma$; SD1.5: $t=\sigma/(1+\sigma)$'); ax3.set_ylabel('prompt share within cell')
ax3.legend(ncol=2, fontsize=5.6, loc='lower right', handlelength=1.3, columnspacing=0.6, handletextpad=0.4, borderaxespad=0.1)
panel_label(ax3, 'c')

rng2 = (E6.prompt_share_naive.min(), E6.prompt_share_naive.max())
rows = [pd.DataFrame(dict(panel='b', series=['depth geometry: scaffold', 'depth geometry: theme+prompt+seed',
                                             'operating point: scaffold', 'operating point: theme+prompt+seed'],
                          y=[96.16, 3.84, 95.55, 4.45], source=f'ledger D12: "{Q12}"')),
        pd.DataFrame(dict(panel='c', series=E6.ch, x=E6.sigma, y=E6.prompt_share_naive, prompt_share_anova=E6.prompt_share_anova,
                          source='review/round2/out/E6_varcomp_by_step.csv (prompt_share_naive; ANOVA seed-corrected share alongside)')),
        pd.DataFrame(dict(panel='c', series=['range naive', 'range anova'], lo=[rng2[0], E6.prompt_share_anova.min()],
                          hi=[rng2[1], E6.prompt_share_anova.max()], source='derived from review/round2/out/E6_varcomp_by_step.csv')),
        pd.DataFrame(dict(panel='c', series=U3.ch, x=U3.t, y=U3.prompt_share_naive, prompt_share_anova=U3.prompt_share_anova,
                          source='review/r3_unet/out/P3_prompt_share_by_step_naive.csv (post hoc, cond; x = t = sigma/(1+sigma))')),
        pd.DataFrame(dict(panel='c', series=A3.ch, x=A3.sigma, y=A3.prompt_share_naive, prompt_share_anova=A3.prompt_share_anova,
                          source='review/r3_ace/out/P3_prompt_share_by_step.csv (post hoc, cond-only; x = flow-matching sigma)'))]
write_data(FIG, rows)
save(fig, FIG)
print(FIG, 'naive range', np.round(rng2, 3), 'anova range', round(E6.prompt_share_anova.min(), 3), round(E6.prompt_share_anova.max(), 3))
