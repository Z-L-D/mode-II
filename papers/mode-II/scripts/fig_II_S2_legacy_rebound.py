"""II-S2 Legacy rebound / local complex mode (supplement; historical, instrument-labelled).
Everything marked "legacy" comes from EMAG-schema files written by an older EMAG-guider build whose measuring hook swaps in
the stale EMA attention; it is not release evidence (ledger v0.12+ annotations; R6C register: not used).
(a) DATA: 24-bin beta_layer(g) curves for the matched 100-step SD3.5 M study, same prompt and schedule: 30 legacy-schema runs
    vs 15 current passive runs (reviewer v0.16, review/round5/reviewer_v016/results/B7_MATCHED100_*).
(b) DATA (ledger, transcribed): crossing locations of the legacy four-crossing card (512/768/1024 px, single prompt) and of the
    passive 120-prompt SD3.5 M campaign (three crossings).
(c) DATA (ledger, transcribed): complex-pair modulus of the local affine map on the legacy rebound branch, and in the legacy
    CFG 1.0 -> 5.5 study (steps 0-9; bootstrap 95% intervals as reproduced in Round 2 P13).
"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import ast
from m2common import *

FIG = 'II-S2_legacy_rebound'
B7 = f'{_COSPAFS_ROOT}/review/round5/reviewer_v016/results'
Qc = '512/768/1024 each show the same sign sequence −→+→−→+→− . Crossings are 512: 0.00223, 0.00663, 0.01570, 0.02987; 768: 0.00238, 0.00802, 0.02198, 0.03585; 1024: 0.00519, 0.01057, 0.02084, 0.03403.'
Qp = 'On the passive 120-prompt SD35M campaign the topology is +−+−, with crossings at g 0.018/0.020/0.077 and TRI≈0.01–0.02'
Qm = 'Median pair moduli are 0.526 (512), 0.501 (768), 0.400 (1024).'
Qf = 'modulus 0.878 [0.862, 0.891] → 0.849 [0.836, 0.861]'
Qs = 'It appears for 0/60 single passive prompts, against 30/30 single legacy generations.'
for q in (Qc, Qp, Qm, Qf, Qs):
    ledger_has(q)

L = pd.read_csv(f'{B7}/B7_MATCHED100_LEGACY_BETA.csv'); P = pd.read_csv(f'{B7}/B7_MATCHED100_PASSIVE_BETA.csv')
SM = pd.read_csv(f'{B7}/B7_MATCHED100_SUMMARY.csv').iloc[0]
lc = ast.literal_eval(SM.legacy_population_crossings); pc = ast.literal_eval(SM.passive_population_crossings)

fig = plt.figure(figsize=(TEXTWIDTH, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.95, 0.8], wspace=0.85)
ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]

ax1.axhline(0, color=INK2, lw=0.6)
ax1.plot(L.g, L.beta, color=GRAY, lw=1.2, marker='o', ms=2.2)
ax1.plot(P.g, P.beta, color=MODEL_COLOR['SD35M'], lw=1.3, marker='o', ms=2.2)
for x_ in lc: ax1.plot(x_, 0, '|', ms=6, mew=1.0, color=INK2)
for x_ in pc: ax1.plot(x_, 0, '|', ms=6, mew=1.0, color=INK)
ax1.text(0.003, -0.046, f'legacy: {SM.legacy_population_sequence.replace("-", "−")}\n({SM.legacy_runs} runs)', fontsize=6.5, color=INK2, va='bottom')
ax1.text(0.135, 0.032, f'passive: {SM.passive_population_sequence.replace("-", "−")}\n({SM.new_runs} runs)', fontsize=6.5, color=INK, ha='right', va='bottom')
ax1.set_xlabel('$g$ (24 bins)'); ax1.set_ylabel(r'$\beta_{\rm layer}=g(L{+}1)-g(L)$'); ax1.set_ylim(-0.05, 0.05)
panel_label(ax1, 'a')

sets = [('legacy 512', [0.00223, 0.00663, 0.01570, 0.02987], GRAY), ('legacy 768', [0.00238, 0.00802, 0.02198, 0.03585], GRAY),
        ('legacy 1024', [0.00519, 0.01057, 0.02084, 0.03403], GRAY), ('passive,\n120 prompts', [0.018, 0.020, 0.077], MODEL_COLOR['SD35M'])]
for k, (lab, xs, col) in enumerate(sets):
    yk = len(sets) - 1 - k
    ax2.plot(xs, [yk] * len(xs), 'o', ms=4, color=col, mec='white', mew=0.4)
    ax2.plot([min(xs), max(xs)], [yk, yk], color=col, lw=0.6, zorder=0)
ax2.set_yticks(range(len(sets))[::-1]); ax2.set_yticklabels([s_[0] for s_ in sets], fontsize=6.8); ax2.set_xscale('log')
ax2.set_xlabel(r'crossing location $g^*$'); ax2.set_ylim(-0.6, len(sets) - 0.4); ax2.grid(axis='y', visible=False)
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
ax2.xaxis.set_major_locator(FixedLocator([0.003, 0.01, 0.03, 0.1])); ax2.xaxis.set_minor_formatter(NullFormatter())
ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); ax2.set_xlim(0.0018, 0.12)
panel_label(ax2, 'b')

mods = [('rebound, 512', 0.526, None, None), ('rebound, 768', 0.501, None, None), ('rebound, 1024', 0.400, None, None),
        ('CFG 1.0 (legacy)', 0.878, 0.862, 0.891), ('CFG 5.5 (legacy)', 0.849, 0.836, 0.861)]
for k, (lab, v, lo, hi) in enumerate(mods):
    yk = len(mods) - 1 - k
    if lo is not None: ax3.plot([lo, hi], [yk, yk], color=INK2, lw=1.0)
    ax3.plot(v, yk, 'o', ms=4, color=GRAY if lo is None else INK2)
    ax3.text(v + 0.04, yk, f'{v:.3f}', va='center', fontsize=6.5, color=INK)
ax3.set_yticks(range(len(mods))[::-1]); ax3.set_yticklabels([m[0] for m in mods], fontsize=6.8); ax3.set_xlim(0.3, 1.08)
ax3.set_xlabel('complex-pair modulus'); ax3.set_ylim(-0.6, len(mods) - 0.4); ax3.grid(axis='y', visible=False)
panel_label(ax3, 'c')

srcB7 = 'review/round5/reviewer_v016/results/B7_MATCHED100_{}_BETA.csv (reviewer v0.16; legacy-schema vs passive, matched 100-step SD3.5 M)'
rows = [pd.DataFrame(dict(panel='a', series='legacy (EMAG-schema)', x=L.g, y=L.beta, n=L.n, source=srcB7.format('LEGACY'))),
        pd.DataFrame(dict(panel='a', series='passive', x=P.g, y=P.beta, n=P.n, source=srcB7.format('PASSIVE'))),
        pd.DataFrame(dict(panel='a', series=['legacy crossings', 'passive crossings'], x=[str(lc), str(pc)],
                          pattern=[SM.legacy_population_sequence, SM.passive_population_sequence],
                          source='review/round5/reviewer_v016/results/B7_MATCHED100_SUMMARY.csv')),
        pd.DataFrame(dict(panel='b', series=[s_[0] for s_ in sets], x=[str(s_[1]) for s_ in sets],
                          source=[f'ledger legacy four-crossing card: "{Qc}"'] * 3 + [f'ledger v0.12 annotation: "{Qp}"'])),
        pd.DataFrame(dict(panel='c', series=[m[0] for m in mods], y=[m[1] for m in mods], lo=[m[2] for m in mods], hi=[m[3] for m in mods],
                          source=[f'ledger legacy complex-mode card: "{Qm}"'] * 3 + [f'ledger CFG card v0.12 annotation (Round 2 P13): "{Qf}"'] * 2)),
        pd.DataFrame(dict(panel='note', series=['single-prompt reproduction'], source=[f'ledger v0.15 annotation: "{Qs}"']))]
write_data(FIG, rows)
save(fig, FIG)
print(FIG, lc, pc)
