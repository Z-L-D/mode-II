"""II-F6 Tangent-field inheritance: state (shape) correspondence versus derivative (tangent) correspondence.
(a) DATA: Round 2 E10 (review/round2/out/E10_tangent.csv): registered depth-shape r vs depth-derivative (tangent) cosine at
    matched sigma, 13 channel pairs, with each pair's permutation-null p95; plus Round 5 Q2 clean-branch checks
    (review/round5/out/Q_results.json: ZIB CFG 4 cond-branch vs ZIT, and ZIB CFG 1 vs CFG 4 cond).
(b) DATA (ledger D10, transcribed): derivative correlation after state-only alignment maps (Procrustes, linear, CCA).
"""
import json
from m2common import *

FIG = 'II-F6_tangent_inheritance'
Q10a = 'State-only Procrustes, linear and CCA maps can visibly improve alignment between unrelated population fields, but their matched depth derivatives remain near zero or anti-aligned.'
Q10b = 'Base↔Turbo is exceptional: matched derivative correlation averages about 0.937 across mappings/channels, while the best mapping for each unrelated pair averages only about 0.007.'
ledger_has(Q10a); ledger_has(Q10b)

E = pd.read_csv(f'{R2}/E10_tangent.csv')
Q = json.load(open(f'{R5}/Q_results.json'))
Q2 = pd.DataFrame([
    ('ZIT-25 CFG4 cond↔ZIT attn', 'lineage', Q['Q2a_shape_ZIB4cond_ZIT_attn'], Q['Q2a_tangent_ZIB4cond_ZIT_attn'], 'Q2a_shape/tangent_ZIB4cond_ZIT_attn'),
    ('ZIT-25 CFG4 cond↔ZIT mlp', 'lineage', Q['Q2a_shape_ZIB4cond_ZIT_mlp'], Q['Q2a_tangent_ZIB4cond_ZIT_mlp'], 'Q2a_shape/tangent_ZIB4cond_ZIT_mlp'),
    ('ZIT-25 CFG1↔CFG4 cond attn', 'same-model CFG', Q['Q2c_shape_ZIB1_ZIB4cond_attn'], Q['Q2c_tangent_ZIB1_ZIB4cond_attn'], 'Q2c_shape/tangent_ZIB1_ZIB4cond_attn'),
    ('ZIT-25 CFG1↔CFG4 cond mlp', 'same-model CFG', Q['Q2c_shape_ZIB1_ZIB4cond_mlp'], Q['Q2c_tangent_ZIB1_ZIB4cond_mlp'], 'Q2c_shape/tangent_ZIB1_ZIB4cond_mlp')],
    columns=['pair', 'kind', 'shape_r', 'tangent_cos', 'key'])
null_med = float(E.perm_null_p95.median())

fig = plt.figure(figsize=(TEXTWIDTH, 2.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.55)
ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

ax1.plot([-0.3, 1.02], [-0.3, 1.02], color=LIGHT, lw=0.8, ls=(0, (3, 2)), zorder=1)
ax1.axhspan(-0.3, null_med, color='#f1f0ec', lw=0, zorder=0)
ax1.text(-0.27, null_med + 0.015, f'below median permutation-null p95 ({null_med:.2f})', fontsize=6.3, color=INK2, va='bottom')
for kind in ['unrelated', 'attn-vs-mlp', 'same family', 'same-model CFG', 'lineage']:
    d = E[E.kind == ('family size' if kind == 'same family' else kind)]
    ax1.plot(d.shape_r, d.tangent_cos, 'o', ms=4.8 if kind == 'lineage' else 4, color=PAIR_COLOR[kind], mec='white', mew=0.5,
             ls='none', zorder=3, label=PAIR_LABEL[kind])
for r in Q2.itertuples():
    ax1.plot(r.shape_r, r.tangent_cos, 'o', ms=4.5, mfc='white', mec=PAIR_COLOR[r.kind], mew=1.0, zorder=4, ls='none')
ax1.plot([], [], 'o', ms=4, mfc='white', mec=INK2, mew=1.0, ls='none', label='clean-branch checks (Round 5)')
zm = E[(E.a == 'ZIT-attn') & (E.b == 'MOCHI-attn')].iloc[0]
ax1.annotate('Z-Image↔Mochi', xy=(zm.shape_r, zm.tangent_cos), xytext=(0.47, 0.47), fontsize=6.5, color=INK,
             arrowprops=dict(arrowstyle='-', color=INK2, lw=0.5))
ax1.set_xlim(-0.3, 1.04); ax1.set_ylim(-0.3, 1.06)
ax1.set_xlabel('registered depth-shape $r$ (state correspondence)'); ax1.set_ylabel('tangent cosine (derivative)')
ax1.legend(loc='upper left', fontsize=6.0, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.3)
panel_label(ax1, 'a')

vals = [('ZIT-25↔ZIT', 0.937, PAIR_COLOR['lineage']), ('best map,\nunrelated pairs', 0.007, GRAY)]
for k, (lab, v, col) in enumerate(vals):
    yk = 1 - k
    ax2.plot([0, v], [yk, yk], color=col, lw=1.2); ax2.plot(v, yk, 'o', ms=5.5, color=col)
    ax2.text(v + 0.05, yk, f'≈{v:.3f}', va='center', fontsize=7, color=INK)
ax2.axvline(0, color=INK2, lw=0.5)
ax2.set_yticks([1, 0]); ax2.set_yticklabels([v[0] for v in vals], fontsize=7); ax2.set_ylim(-0.6, 1.6); ax2.set_xlim(-0.15, 1.25)
ax2.set_xlabel('derivative correlation after\nstate-only alignment (mean)'); ax2.grid(axis='y', visible=False)
panel_label(ax2, 'b')

rows = [pd.DataFrame(dict(panel='a', series=E.kind, pair=E.a + '|' + E.b, x=E.shape_r, y=E.tangent_cos, perm_null_p95=E.perm_null_p95,
                          source='review/round2/out/E10_tangent.csv')),
        pd.DataFrame(dict(panel='a', series=Q2.kind + ' (clean branch)', pair=Q2.pair, x=Q2.shape_r, y=Q2.tangent_cos,
                          source='review/round5/out/Q_results.json: ' + Q2.key)),
        pd.DataFrame(dict(panel='b', series=['Base<->Turbo mean over mappings/channels', 'best mapping per unrelated pair, mean'],
                          y=[0.937, 0.007], source=f'ledger D10: "{Q10a} {Q10b}"'))]
write_data(FIG, rows)
save(fig, FIG)
print(FIG, 'null median', round(null_med, 3))
