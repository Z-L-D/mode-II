"""II-S1 Intrinsic-progress negative result (supplement).
(a,b) ILLUSTRATION ON DATA: layer-mean attention log g envelopes of the five models (raw cubes, archive) drawn against the
      sampling clock sigma (a) and against normalized cumulative spectral arc length s (b). Arc length is re-implemented here as
      the cumulative Euclidean path length of the layer-mean state (log g, log gap, H), each coordinate z-scored over steps;
      it shows what the reparameterization does and is NOT the ledger D17 computation.
(c)   DATA (ledger D17, transcribed): mean change in envelope correlation after arc-length reparameterization.
"""
from m2common import *

FIG = 'II-S1_intrinsic_progress'
Q17 = 'Reparameterizing coarse curves by cumulative spectral arc length raises unrelated/cross-channel correlation by +0.057 on average, but the preregistered paired test is not significant (p≈0.105); Base/Turbo changes by -0.004 on average.'
ledger_has(Q17)

env = []
for m in ['ZIT', 'ZIB', 'MOCHI', 'SD35M', 'SD35L']:
    _, meta, X, s = cube(f'{m}_attn')
    Z = np.stack([np.log(X[..., 0]).mean((0, 2)), np.log(X[..., 1]).mean((0, 2)), X[..., 2].mean((0, 2))], 1)
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Zs, axis=0), axis=1))]; arc /= arc[-1]
    env.append(pd.DataFrame(dict(model=m, sigma=s, arc=arc, log_g=Z[:, 0])))
E = pd.concat(env)

fig = plt.figure(figsize=(TEXTWIDTH, 2.3))
gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 0.5, 0.75], wspace=0.16)
ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1], sharey=ax1); ax3 = fig.add_subplot(gs[0, 3])
for m, d in E.groupby('model', sort=False):
    ax1.plot(d.sigma, d.log_g, color=MODEL_COLOR[m], lw=1.2, label=SHORT[m])
    ax2.plot(d.arc, d.log_g, color=MODEL_COLOR[m], lw=1.2)
ax1.set_xlim(1.03, 0.05); ax1.set_xticks([1.0, 0.5]); ax1.set_xlabel(r'$\sigma$ (runs right)'); ax1.set_ylabel(r'layer-mean attention $\log g$')
ax1.legend(fontsize=6.3, loc='lower right', handlelength=1.3, borderaxespad=0.1)
ax2.set_xlabel('spectral arc length $s$ (normalized)'); plt.setp(ax2.get_yticklabels(), visible=False)
ax2.text(0.97, 0.04, 'illustration', transform=ax2.transAxes, ha='right', va='bottom', fontsize=6.5, color=INK2, style='italic')
panel_label(ax1, 'a'); panel_label(ax2, 'b')

vals = [('unrelated /\ncross-channel', 0.057, 'paired test\np ≈ 0.105 (n.s.)'), ('ZIT-25/ZIT', -0.004, '')]
for k, (lab, v, note) in enumerate(vals):
    yk = 1 - k
    ax3.plot([0, v], [yk, yk], color=INK2, lw=1.0); ax3.plot(v, yk, 'o', ms=5, color=INK2)
    ax3.text(v + (0.006 if v >= 0 else -0.006), yk + 0.18, f'{v:+.3f}'.replace('-', '−'), ha='center', va='bottom', fontsize=7, color=INK)
    if note: ax3.text(0.003, yk - 0.22, note, ha='left', va='top', fontsize=6.3, color=INK2)
ax3.axvline(0, color=INK2, lw=0.6)
ax3.set_yticks([1, 0]); ax3.set_yticklabels([v[0] for v in vals], fontsize=6.8); ax3.set_ylim(-0.6, 1.6); ax3.set_xlim(-0.03, 0.09)
ax3.set_xticks([0, 0.04, 0.08])
ax3.set_xlabel(r'mean change $\Delta r$'); ax3.grid(axis='y', visible=False)
panel_label(ax3, 'c')

rows = [pd.DataFrame(dict(panel='a,b', series=E.model, x=E.sigma, arc_length=E.arc, y=E.log_g,
                          source='raw cubes F_<model>_attn.pkl (archive); arc length re-implemented here (illustration, not D17)')),
        pd.DataFrame(dict(panel='c', series=['unrelated/cross-channel mean change', 'Base/Turbo mean change', 'paired test p'],
                          y=[0.057, -0.004, 0.105], source=f'ledger D17: "{Q17}"'))]
write_data(FIG, rows)
save(fig, FIG)
print(FIG, 'ok')
