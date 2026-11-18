"""II-F3 Projection degeneracy and the equivalence hierarchy.
(a,b) ILLUSTRATION: two different planar vector fields F_a (drift + relaxation) and F_b (drift + rotation).
      Their trajectories from the same start have almost identical scalar projections P(z) = z1, while the
      centred fields themselves are nearly orthogonal over the plotted domain.
(c)   DATA: Round 6 S1 hierarchy (review/round6/out/S1_hierarchy.csv). For each observable level, the
      Base<->Turbo (lineage) score, the 8 unrelated pairs and the SD3.5 M<->L same-family pair.
      Margin = lineage minus best unrelated pair (preregistered S1a-c).
"""
from m2common import *
from scipy.integrate import solve_ivp

FIG = 'II-F3_projection_degeneracy'
Fa = lambda t, z: [1.0, -3.0 * z[1]]
Fb = lambda t, z: [1.0 + 0.5 * z[1], -6.0 * (z[0] - 0.5) - 0.4 * z[1]]
t = np.linspace(0, 1, 101)
za = solve_ivp(Fa, (0, 1), [0, 0.5], t_eval=t, rtol=1e-8).y
zb = solve_ivp(Fb, (0, 1), [0, 0.5], t_eval=t, rtol=1e-8).y
r_proj = float(np.corrcoef(za[0], zb[0])[0, 1])
Z1, Z2 = np.meshgrid(np.linspace(-0.05, 1.3, 23), np.linspace(-0.85, 1.0, 23))
FA = np.stack([np.ones_like(Z1), -3.0 * Z2]); FB = np.stack([1 + 0.5 * Z2, -6.0 * (Z1 - 0.5) - 0.4 * Z2])
fa = (FA - FA.mean((1, 2), keepdims=True)).ravel(); fb = (FB - FB.mean((1, 2), keepdims=True)).ravel()
cos_field = float(fa @ fb / np.linalg.norm(fa) / np.linalg.norm(fb))

H = pd.read_csv(f'{R6}/S1_hierarchy.csv')
LEV = ['L0_D1', 'L0_D2', 'L1_field', 'L2_shape', 'L3_tangent']
LEVLAB = ['envelope\n$g$', 'envelope\n$\\log g$', 'depth×σ\nfield', 'registered\nshape', 'tangent']
lin = H[H.kind == 'lineage'].iloc[0]; unr = H[H.kind == 'unrelated']; fam = H[H.kind == 'same family'].iloc[0]
margin = [float(lin[l] - unr[l].max()) for l in LEV]

fig = plt.figure(figsize=(TEXTWIDTH, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 0.75, 1.45], wspace=0.72)
ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]
ca, cb = INK2, C[1]
sk = (slice(None, None, 2), slice(None, None, 2))
ax1.plot(za[0], za[1], color=ca, lw=1.4, zorder=3); ax1.plot(zb[0], zb[1], color=cb, lw=1.4, zorder=3)
ax1.plot([0], [0.5], 'o', color=INK, ms=3, zorder=4)
ax1.legend(['$F_a$: relaxation', '$F_b$: rotation'], loc='lower left', fontsize=6.5, handlelength=1.2, borderaxespad=0.2)
ax1.set_xlabel('$z_1$'); ax1.set_ylabel('$z_2$'); ax1.set_xlim(-0.05, 1.3); ax1.set_ylim(-0.9, 1.0)
panel_label(ax1, 'a')

ax2.plot(t, za[0], color=ca, lw=1.4); ax2.plot(t, zb[0], color=cb, lw=1.4)
ax2.text(0.55, 0.32, '$P z_a$', color=INK, fontsize=7); ax2.text(0.06, 0.62, '$P z_b$', color=INK, fontsize=7)
ax2.set_xlabel('time $t$'); ax2.set_ylabel('projection $P(z)=z_1$')
ax2.text(0.03, 0.97, f'corr = {r_proj:.3f}\nfield cos = {cos_field:.2f}',
         transform=ax2.transAxes, ha='left', va='top', fontsize=7, color=INK)
panel_label(ax2, 'b')

y = np.arange(len(LEV))[::-1]                       # top row = coarsest observable
for k, l in enumerate(LEV):
    ax3.plot(unr[l], np.full(len(unr), y[k]), 'o', ms=3.2, color=GRAY, mec='white', mew=0.4, zorder=2, ls='none')
ax3.plot([fam[l] for l in LEV], y, 'D', ms=3.4, mfc='white', mec=PAIR_COLOR['same family'], mew=1.0, zorder=3, ls='none')
ax3.plot([lin[l] for l in LEV], y, 'o', ms=4.8, color=PAIR_COLOR['lineage'], zorder=4, ls='none')
for k, m in enumerate(margin):
    ax3.text(1.13, y[k], f'{m:+.3f}'.replace('-', '−'), ha='left', va='center', fontsize=7, color=INK)
ax3.text(1.13, y[0] + 0.62, 'margin', ha='left', va='center', fontsize=7, color=INK2)
ax3.set_yticks(y); ax3.set_yticklabels([s.replace('\n', ' ') for s in LEVLAB], fontsize=7)
ax3.set_ylim(-0.6, len(LEV) - 0.2); ax3.set_xlim(-0.35, 1.08)
ax3.set_xlabel('cross-model similarity ($r$ or cosine)'); ax3.grid(axis='y', visible=False)
ax3.axvline(0, color=INK2, lw=0.5)
ax3.text(lin['L3_tangent'], y[-1] + 0.3, 'ZIT-25↔ZIT', fontsize=6.5, color=INK, ha='center', va='bottom')
ax3.text(unr['L3_tangent'].mean(), y[-1] + 0.3, '8 unrelated', fontsize=6.5, color=INK2, ha='center', va='bottom')
ax3.text(fam['L3_tangent'] - 0.04, y[-1] - 0.32, 'SD3.5 M↔L', fontsize=6.5, color=INK2, ha='left', va='top')
panel_label(ax3, 'c')

src_ill = 'illustration: F_a=(1,-3 z2), F_b=(1+0.5 z2, -6(z1-0.5)-0.4 z2), z(0)=(0,0.5)'
rows = [pd.DataFrame(dict(panel='a', series=np.repeat(['F_a', 'F_b'], len(t)), x=np.r_[za[0], zb[0]], y=np.r_[za[1], zb[1]], t=np.r_[t, t], source=src_ill)),
        pd.DataFrame(dict(panel='b', series=['corr_projection', 'field_cosine_centred'], y=[r_proj, cos_field], source=src_ill))]
for l in LEV:
    for _, r in H.iterrows():
        rows.append(pd.DataFrame(dict(panel='c', series=[r.kind], x=[l], pair=[f'{r.a}-{r.b}'], y=[r[l]], source='review/round6/out/S1_hierarchy.csv')))
rows.append(pd.DataFrame(dict(panel='c', series='margin lineage minus best unrelated', x=LEV, y=margin,
                              source='derived from review/round6/out/S1_hierarchy.csv (S1a-c)')))
write_data(FIG, rows)
save(fig, FIG)
print(FIG, round(r_proj, 4), round(cos_field, 3), np.round(margin, 3))
