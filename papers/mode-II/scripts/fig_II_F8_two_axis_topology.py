"""II-F8 Two distinct phase/topology constructions.
(a) sigma axis. DATA: population-mean, layer-mean MLP g(sigma) for Z-Image (prompt means, then mean over prompts, as in
    Round 6 S4), recomputed from the raw cubes: ZIT archive (8 steps), ZIB archive (25 steps, pooled CFG 4) and the clean
    ZIB CFG 4 cond branch (Round 4/5 capture R5_Z40_1024_s1001). Breakpoints, bootstrap rates and contraction are read
    from review/round6/out/S4_topology.csv. Scope: Z-Image MLP only; attention is "++-" in all five models (S4e, P1).
(b) layer axis. DATA (post hoc, analyzer replay A2-A4): the sign pattern of the analyzer's beta_layer = g(L+1) - g(L)
    binned against g, at each sampling step (review/round6/out/A4_beta_layer_per_step.csv). Right: whole-run generation
    bootstrap rate of + - + - (A2_beta_layer_bootstrap.csv) and the surrogate class (A4_beta_layer_surrogates.csv;
    09-15 Surrogate A = per-layer marginals, Surrogate B = mean field g(L, sigma)).
"""
from m2common import *

FIG = 'II-F8_two_axis_topology'
S4 = pd.read_csv(f'{R6}/S4_topology.csv').set_index('set')
A2 = pd.read_csv(f'{R6}/A2_beta_layer_bootstrap.csv'); A4s = pd.read_csv(f'{R6}/A4_beta_layer_surrogates.csv').set_index('channel')
A4 = pd.read_csv(f'{R6}/A4_beta_layer_per_step.csv')


def layer_mean_curve(X, meta):
    k = meta.pt.astype(str).str.strip().values; up = np.unique(k)
    return np.stack([X[k == p][..., 0].mean((0, 2)) for p in up]).mean(0), len(up)


curves = []
for lab, key, model, ls in [('ZIT archive', 'archive ZIT-mlp', 'ZIT', '-'), ('ZIT-25 archive', 'archive ZIB-mlp', 'ZIB', '-')]:
    _, meta, X, s = cube(f'{model}_mlp'); y, n = layer_mean_curve(X, meta)
    curves.append((lab, key, model, ls, s, y, n, f'raw cube F_{model}_mlp.pkl'))
d = r5cube('Z40_1024_s1001', 'mlp'); y, n = layer_mean_curve(d['X']['cond'], d['meta'])
curves.append(('ZIT-25 clean CFG 4 cond', 'R4 ZIB CFG4 cond-mlp', 'ZIB', (0, (3, 1.5)), np.asarray(d['sigma'], float), y, n, 'raw capture R5_Z40_1024_s1001_mlp.pkl, X[cond]'))

fig = plt.figure(figsize=(TEXTWIDTH, 2.75))
gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.3], wspace=0.42, left=0.08, right=0.70)
ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

for lab, key, model, ls, s, y, n, src in curves:
    ax1.plot(s, y * 1e3, color=MODEL_COLOR[model], ls=ls, lw=1.3, marker='o' if model == 'ZIT' else None, ms=2.5, label=lab)
ax1.set_xlim(1.02, 0.0); ax1.set_xlabel(r'$\sigma$ (sampling runs right)'); ax1.set_ylabel(r'layer-mean MLP $g$ ($\times10^{-3}$)')
ax1.legend(loc='lower right', fontsize=6.3, handlelength=1.6, borderaxespad=0.1)
r = S4.loc['archive ZIB-mlp']; rc = S4.loc['R4 ZIB CFG4 cond-mlp']
ax1.text(0.56, 0.98, f'+ − + in {r.frac_pmp:.0%} of\nprompt bootstraps\ncontraction {100 * r.contraction:.1f}%\n(clean {100 * rc.contraction:.1f}%)'.replace('-', '−'),
         transform=ax1.transAxes, ha='center', va='top', fontsize=6.5, color=INK)
for x_ in (r.break1_sigma, r.break2_sigma):
    ax1.axvline(x_, color=LIGHT, lw=0.8, zorder=0)
ax1.set_ylim(0, 10.5)
panel_label(ax1, 'a')

order = [('SD35M archive attn', 'SD35M archive', 'as loaded (pooled)', 'SD35M', 'SD3.5 M attn'),
         ('SD35L archive attn', 'SD35L archive', 'as loaded (pooled)', 'SD35L', 'SD3.5 L attn'),
         ('ZIB archive (pooled CFG4) attn', 'ZIB archive (pooled CFG4)', 'attn', 'ZIB', 'ZIT-25 attn'),
         ('ZIB archive (pooled CFG4) mlp', 'ZIB archive (pooled CFG4)', 'mlp', 'ZIB', 'ZIT-25 MLP'),
         ('ZIT archive mlp', 'ZIT archive', 'mlp', 'ZIT', 'ZIT MLP'),
         ('MOCHI archive CFG4.5 attn', 'MOCHI archive CFG4.5', 'as loaded (pooled)', 'MOCHI', 'Mochi attn'),
         ('ZIT archive attn', 'ZIT archive', 'attn', 'ZIT', 'ZIT attn')]
PMPM = '+- -+ +-'


def surr_class(ch):
    r_ = A4s.loc[ch]
    if r_.observed != PMPM:
        return '1 crossing;\nnull-generic'
    if r_.surrA_frac_obs == 0 and r_.surrB_frac_obs == 0:
        return 'beats Surr. A and B'
    if r_.surrA_frac_obs == 0 and r_.surrB_frac_obs > 0.5:
        return 'beats A; B (mean field)\nreproduces'
    if r_.surrA_frac_obs > 0.5 and r_.surrB_frac_obs > 0.5:
        return 'null-generic\n(A and B reproduce)'
    return 'mixed'


rows_b = []
for yk, (ch, aset, asub, model, lab) in enumerate(order):
    d = A4[A4.channel == ch]
    boot = A2[(A2.set == aset) & (A2.submodule == asub)].iloc[0]
    for rr in d.itertuples():
        if rr.pattern == PMPM:
            ax2.plot(rr.sigma, yk, 'o', ms=4.2, color=MODEL_COLOR[model], mec='white', mew=0.4, zorder=3)
        elif rr.n >= 5:
            ax2.plot(rr.sigma, yk, 'x', ms=3.5, mew=0.9, color=INK2, zorder=3)
        else:
            ax2.plot(rr.sigma, yk, 'o', ms=3.4, mfc='white', mec=GRAY, mew=0.8, zorder=2)
    cls = surr_class(ch)
    ax2.text(1.13, yk, f'{boot.frac_3_plus_minus_plus:.0%}', transform=ax2.get_yaxis_transform(), va='center', ha='right', fontsize=6.8, color=INK)
    ax2.text(1.17, yk, cls, transform=ax2.get_yaxis_transform(), va='center', ha='left', fontsize=6.5, color=INK if 'null' not in cls else INK2)
    rows_b.append(pd.DataFrame(dict(panel='b', series=lab, x=d.sigma, y=d.n, pattern=d.pattern,
                                    source='review/round6/out/A4_beta_layer_per_step.csv (post hoc, analyzer replay)')))
    rows_b.append(pd.DataFrame(dict(panel='b', series=[lab], x=['whole run'], y=[boot.frac_3_plus_minus_plus], pattern=[boot.observed_pattern],
                                    surrogate_class=[cls], surrA_frac_obs=[A4s.loc[ch, 'surrA_frac_obs']], surrB_frac_obs=[A4s.loc[ch, 'surrB_frac_obs']],
                                    gens=[boot.gens], source=['review/round6/out/A2_beta_layer_bootstrap.csv (frac_3_plus_minus_plus, 100 generation bootstraps) + A4_beta_layer_surrogates.csv (30 draws each)'])))
ax2.text(1.13, -0.95, 'boot.', transform=ax2.get_yaxis_transform(), ha='right', va='center', fontsize=6.5, color=INK2)
ax2.text(1.17, -0.95, 'surrogates', transform=ax2.get_yaxis_transform(), ha='left', va='center', fontsize=6.5, color=INK2)
ax2.set_yticks(range(len(order))); ax2.set_yticklabels([o[4] for o in order], fontsize=7); ax2.set_ylim(len(order) - 0.4, -1.3)
ax2.set_xlim(1.03, 0.06); ax2.set_xlabel(r'$\sigma$ of the step whose layer flow is read'); ax2.grid(axis='y', visible=False)
ax2.plot([], [], 'o', ms=4, color=INK2, ls='none', label='+ − + − (3 crossings)')
ax2.plot([], [], 'o', ms=3.4, mfc='white', mec=GRAY, ls='none', label='1 crossing (or 2/4)')
ax2.plot([], [], 'x', ms=3.5, color=INK2, ls='none', label='5 or more')
ax2.legend(loc='upper center', bbox_to_anchor=(0.75, -0.22), ncol=3, fontsize=6.5, handletextpad=0.2, columnspacing=1.0)
panel_label(ax2, 'b')

rows = []
for lab, key, model, ls, s, y, n, src in curves:
    rr = S4.loc[key]
    rows.append(pd.DataFrame(dict(panel='a', series=lab, x=s, y=y, prompts=n, source=src + ' (layer-mean g, mean over prompts)')))
    rows.append(pd.DataFrame(dict(panel='a', series=[lab + ' S4 fit'], topology=[rr.topology], frac_pmp=[rr.frac_pmp], break1_sigma=[rr.break1_sigma],
                                  break2_sigma=[rr.break2_sigma], contraction=[rr.contraction], source=[f'review/round6/out/S4_topology.csv row "{key}"'])))
write_data(FIG, rows + rows_b)
save(fig, FIG)
print(FIG, 'ok')
