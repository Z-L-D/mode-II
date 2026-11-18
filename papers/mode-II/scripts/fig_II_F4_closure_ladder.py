"""II-F4 Nonautonomous closure ladder.
All values are transcribed from the ledger text (papers/_sources/ledger_v020_text.txt); every quote is checked
verbatim against the ledger before plotting (ledger_has). The ledger gives ranges and medians, not per-channel
tables, so ranges are drawn as intervals and medians as points.
(a) held-out R^2 by predictor set; (b) incremental R^2 of the spectral state over g alone versus over (u, tau);
(c) vector-field transfer: within target, Base<->Turbo, unrelated (D03, D03b, D15).
"""
from m2common import *

FIG = 'II-F4_closure_ladder'
Q = dict(
    layer='A held-out control using layer position alone reaches R²≈0.70–0.95 in many attention/AR channels.',
    clock='In diffusion, adding sampling clock to layer position raises the score to ≈0.89–0.97.',
    zmlp='Z-Image MLP is less reducible to layer anatomy (layer-only ≈0.43–0.46), and its measured state contributes substantially.',
    d01='With entire prompt identities held out, the rich local state/coordinate predictor retains R²=0.894–0.965 across the tested diffusion and fixed-context AR channels.',
    d08='the median incremental gain from the full state is only about 0.014 R² overall (about 0.032 in diffusion and 0.010 in fixed-context AR)',
    d09='Under the frozen minimal-sufficiency rule, 7/12 channels require only (u,tau), while the other 5/12 require only g+u+tau.',
    cs='Median gain is ≈0.314 R² across native diffusion channels and ≈0.146 across the fixed-context AR set; individual gains range from ≈0.022 to 0.658.',
    d03='Within-target prompt-held state+coordinate R² remains 0.941–0.987, but transfer into Mochi and SD3.5 M/L is approximately zero or negative after dimensionless normalization.',
    d03b='Direct ZIB↔ZIT transfer remains substantial: attention R²=0.700–0.746, MLP R²=0.722–0.892.',
    d15='Base↔Turbo cross-residual transfer averages R²=0.437 against a target-self residual ceiling of 0.579. Representative unrelated transfers average R²=-0.256 despite target-self ceilings around 0.459.')
for v in Q.values():
    ledger_has(v)
CARD = dict(layer='ledger card "Depth coordinate is a major explanatory variable" (script 15)', clock='same card (script 15)',
            zmlp='same card (script 15)', d01='ledger D01', d08='ledger D08/D09', d09='ledger D08/D09',
            cs='ledger card "Cross-substrate CoS state predicts next-layer flow beyond scalar g" (script 11)',
            d03='ledger D03', d03b='ledger D03 lineage control', d15='ledger D15')
fmt = lambda v: (lambda s: s[:-1] if s.endswith('0') else s)(f'{v:.3f}')
src = lambda k: f'{CARD[k]}: "{Q[k]}"'

A = pd.DataFrame([
    ('depth $u$ only\n(many attn/AR channels)', 0.70, 0.95, 'layer'),
    ('depth $u$ only\n(Z-Image MLP)', 0.43, 0.46, 'zmlp'),
    ('$u$ + clock $\\tau$\n(diffusion)', 0.89, 0.97, 'clock'),
    ('state + $(u,\\tau)$\n(prompt held out)', 0.894, 0.965, 'd01')], columns=['label', 'lo', 'hi', 'key'])
B = pd.DataFrame([
    ('$[g,\\mathrm{gap},H]$ over $g$', 'diffusion', 0.314, 'cs'), ('$[g,\\mathrm{gap},H]$ over $g$', 'fixed-context AR', 0.146, 'cs'),
    ('full state over $(u,\\tau)$', 'diffusion', 0.032, 'd08'), ('full state over $(u,\\tau)$', 'fixed-context AR', 0.010, 'd08')],
    columns=['label', 'family', 'y', 'key'])
Cc = pd.DataFrame([
    ('within target', 0.941, 0.987, 'd03', INK2),
    ('ZIT-25↔ZIT attn', 0.700, 0.746, 'd03b', PAIR_COLOR['lineage']),
    ('ZIT-25↔ZIT MLP', 0.722, 0.892, 'd03b', PAIR_COLOR['lineage'])], columns=['label', 'lo', 'hi', 'key', 'col'])
D15 = pd.DataFrame([('ZIT-25↔ZIT', 0.437, 0.579, PAIR_COLOR['lineage']), ('unrelated', -0.256, 0.459, GRAY)],
                   columns=['label', 'transfer', 'ceiling', 'col'])

fig = plt.figure(figsize=(TEXTWIDTH - 0.5, 2.55))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.95, 1.05], wspace=0.85)
ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]

y = np.arange(len(A))[::-1]
for yi, r in zip(y, A.itertuples()):
    ax1.plot([r.lo, r.hi], [yi, yi], color=INK2, lw=2.2, solid_capstyle='round')
    ax1.text(r.hi + 0.02, yi, f'{fmt(r.lo)}–{fmt(r.hi)}', va='center', fontsize=6.5, color=INK)
ax1.set_yticks(y); ax1.set_yticklabels(A.label, fontsize=6.8); ax1.set_xlim(0.35, 1.12); ax1.set_ylim(-0.6, len(A) - 0.4)
ax1.set_xlabel(r'held-out $R^2$ for $\beta_g$'); ax1.grid(axis='y', visible=False)
panel_label(ax1, 'a')

for k, (lab, grp) in enumerate(B.groupby('label', sort=False)):
    for r in grp.itertuples():
        filled = r.family == 'diffusion'
        ax2.plot(k, r.y, 'o', ms=5, color=INK2, mfc=INK2 if filled else 'white', mew=1.0)
        ax2.text(k + 0.1, r.y, f'{r.y:.3f}', fontsize=6.5, color=INK, va='center')
ax2.set_xticks([0, 1]); ax2.set_xticklabels(['state\nover $g$', 'state over\n$(u,\\tau)$'], fontsize=7)
ax2.set_xlim(-0.45, 1.6); ax2.set_ylim(-0.01, 0.36); ax2.set_ylabel(r'median $\Delta R^2$'); ax2.grid(axis='x', visible=False)
ax2.plot([], [], 'o', color=INK2, ms=4, label='diffusion'); ax2.plot([], [], 'o', color=INK2, mfc='white', ms=4, label='fixed-context\nAR')
ax2.legend(loc='upper right', bbox_to_anchor=(1.04, 0.72), fontsize=6.5, handletextpad=0.2, borderaxespad=0.1)
panel_label(ax2, 'b')

rows_c = [(r.label, r.lo, r.hi, r.col) for r in Cc.itertuples()]
y3 = np.arange(5)[::-1]
for yi, (lab, lo, hi, col) in zip(y3[:3], rows_c):
    ax3.plot([lo, hi], [yi, yi], color=col, lw=2.2, solid_capstyle='round')
    ax3.text(hi + 0.03, yi, f'{fmt(lo)}–{fmt(hi)}', va='center', fontsize=6.5, color=INK)
for yi, r in zip(y3[3:], D15.itertuples()):
    ax3.plot([r.transfer], [yi], 'o', ms=4.6, color=r.col)
    ax3.plot([r.ceiling], [yi], '|', ms=8, mew=1.2, color=INK2)
    ax3.text(max(r.transfer, r.ceiling) + 0.03, yi, f'{r.transfer:.3f} / {r.ceiling:.3f}'.replace('-', '−'), va='center', fontsize=6.5, color=INK)
ax3.axvline(0, color=INK2, lw=0.5)
ax3.set_yticks(y3); ax3.set_yticklabels(['within target', 'ZIT-25↔ZIT attn', 'ZIT-25↔ZIT MLP', 'residual,\nZIT-25↔ZIT', 'residual,\nunrelated'], fontsize=6.8)
ax3.set_xlim(-0.4, 1.45); ax3.set_ylim(-0.6, 4.6); ax3.set_xlabel(r'transfer $R^2$ (| = self ceiling)'); ax3.grid(axis='y', visible=False)
panel_label(ax3, 'c')

rows = [pd.DataFrame(dict(panel='a', series=A.label.str.replace('\n', ' '), lo=A.lo, hi=A.hi, source=[src(k) for k in A.key])),
        pd.DataFrame(dict(panel='b', series=B.label + ' | ' + B.family, y=B.y, source=[src(k) for k in B.key])),
        pd.DataFrame(dict(panel='b', series=['channels needing only (u,tau)', 'channels needing g+u+tau', 'n channels'], y=[7, 5, 12], source=src('d09'))),
        pd.DataFrame(dict(panel='c', series=Cc.label, lo=Cc.lo, hi=Cc.hi, source=[src(k) for k in Cc.key])),
        pd.DataFrame(dict(panel='c', series='residual transfer ' + D15.label, y=D15.transfer, ceiling=D15.ceiling, source=src('d15'))),
        pd.DataFrame(dict(panel='c', series=['unrelated transfer (D03)'], y=[np.nan], note=['approximately zero or negative; no value tabulated, not plotted'], source=[src('d03')]))]
write_data(FIG, rows)
save(fig, FIG)
print(FIG, 'ok')
