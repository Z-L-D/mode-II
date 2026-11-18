"""II-F9 Differencing and surrogate nulls.
(a,b) ILLUSTRATION (Monte Carlo): v_1, v_2, ... i.i.d. with a common mean and common covariance Sigma (power-law spectrum).
      With d_k = v_{k+1} - v_k:  E[d_k . d_{k+1}] = -tr Sigma and E|d_k|^2 = 2 tr Sigma, so the adjacent-difference cosine
      phi = cos(d_k, d_{k+1}) concentrates at -1/2 as the effective dimension (tr Sigma)^2 / tr Sigma^2 grows, and the
      stride-2 cosine cos(d_k, d_{k+2}) concentrates at 0. Raw phi near -1/2 is therefore the null value, not a signal.
(c)   DATA (post hoc, analyzer replay A4; review/round6/out/A4_beta_layer_surrogates.csv): fraction of 30 draws of the 09-15
      Surrogate A (per-layer marginals, within-generation layer coupling broken) and Surrogate B (keeps the mean field
      g(L, sigma)) that reproduce each channel's observed whole-run beta_layer sign pattern.
"""
from m2common import *

FIG = 'II-F9_differencing_nulls'
rng = np.random.default_rng(915)
DIMS = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 3072]
NTR = 4000


def sim(D, n=NTR):
    lam = 1.0 / np.arange(1, D + 1) ** 0.5                  # common covariance spectrum (power law)
    mu = rng.normal(0, 1, D) * 2.0                           # common mean (cancels in differences)
    V = mu + rng.normal(size=(n, 4, D)) * np.sqrt(lam)
    d = np.diff(V, axis=1)                                   # d1, d2, d3
    cs = lambda a, b: (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
    return cs(d[:, 0], d[:, 1]), cs(d[:, 0], d[:, 2]), lam.sum() ** 2 / (lam ** 2).sum()


res = []; dist = {}
for D in DIMS:
    a, s2, deff = sim(D)
    res.append((D, deff, a.mean(), a.std(), s2.mean(), s2.std()))
    if D in (16, 3072): dist[D] = (a, s2)
R = pd.DataFrame(res, columns=['D', 'd_eff', 'adj_mean', 'adj_sd', 'stride2_mean', 'stride2_sd'])
A4 = pd.read_csv(f'{R6}/A4_beta_layer_surrogates.csv')

fig = plt.figure(figsize=(TEXTWIDTH, 2.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.15], wspace=0.75, left=0.07, right=0.86)
ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]

bins = np.linspace(-1, 1, 81)
a, s2 = dist[3072]
ax1.hist(a, bins=bins, histtype='stepfilled', color=C[0], alpha=0.85, lw=0)
ax1.hist(s2, bins=bins, histtype='stepfilled', color=GRAY, alpha=0.85, lw=0)
a16, s16 = dist[16]
ax1.hist(a16, bins=bins, histtype='step', color=C[0], lw=0.8)
ax1.hist(s16, bins=bins, histtype='step', color=INK2, lw=0.8)
for xv in (-0.5, 0):
    ax1.axvline(xv, color=INK2, lw=0.6, ls=(0, (3, 2)))
ax1.set_xlabel('cosine'); ax1.set_ylabel(f'draws (of {NTR})'); ax1.set_xlim(-1, 1)
ax1.set_ylim(0, ax1.get_ylim()[1] * 1.16)
ax1.text(-0.5, ax1.get_ylim()[1] * 0.99, 'adjacent', ha='center', va='top', fontsize=6.6, color=INK)
ax1.text(0.06, ax1.get_ylim()[1] * 0.99, 'stride 2', ha='left', va='top', fontsize=6.6, color=INK)
ax1.text(0.99, 0.42, 'filled:\n$D$ = 3072\noutline:\n$D$ = 16', transform=ax1.transAxes, ha='right', va='top', fontsize=6.0, color=INK2)
panel_label(ax1, 'a')

ax2.fill_between(R.d_eff, R.adj_mean - R.adj_sd, R.adj_mean + R.adj_sd, color=C[0], alpha=0.18, lw=0)
ax2.plot(R.d_eff, R.adj_mean, color=C[0], lw=1.3, marker='o', ms=2.5)
ax2.fill_between(R.d_eff, R.stride2_mean - R.stride2_sd, R.stride2_mean + R.stride2_sd, color=GRAY, alpha=0.25, lw=0)
ax2.plot(R.d_eff, R.stride2_mean, color=INK2, lw=1.3, marker='o', ms=2.5)
ax2.axhline(-0.5, color=INK2, lw=0.6, ls=(0, (3, 2))); ax2.axhline(0, color=INK2, lw=0.6, ls=(0, (3, 2)))
ax2.set_xscale('log'); ax2.set_ylim(-1.0, 0.6)
ax2.set_xlabel(r'effective dimension $(\mathrm{tr}\,\Sigma)^2/\mathrm{tr}\,\Sigma^2$'); ax2.set_ylabel('cosine, mean ± sd')
ax2.text(R.d_eff.iloc[-1], -0.92, r'$\varphi$ → −1/2', ha='right', va='bottom', fontsize=7, color=INK)
ax2.text(R.d_eff.iloc[-1], 0.10, 'stride 2 → 0', ha='right', va='bottom', fontsize=7, color=INK)
panel_label(ax2, 'b')

lab = {'SD35M archive attn': 'SD3.5 M attn', 'SD35L archive attn': 'SD3.5 L attn', 'ZIB archive (pooled CFG4) attn': 'ZIT-25 attn',
       'ZIB archive (pooled CFG4) mlp': 'ZIT-25 MLP', 'ZIT archive mlp': 'ZIT MLP', 'MOCHI archive CFG4.5 attn': 'Mochi attn', 'ZIT archive attn': 'ZIT attn'}
order = list(lab)
A4 = A4.set_index('channel').loc[order].reset_index()
yy = np.arange(len(A4))
ax3.plot(A4.surrA_frac_obs, yy - 0.12, 's', ms=4.2, color=C[1], ls='none', label='Surrogate A (marginals)')
ax3.plot(A4.surrB_frac_obs, yy + 0.12, '^', ms=4.6, mfc='white', mec=C[0], mew=1.0, ls='none', label='Surrogate B (mean field)')
for k, r in A4.iterrows():
    obs = '+ − + −' if r.observed == '+- -+ +-' else '+ −'
    ax3.text(1.1, k, obs, va='center', ha='left', fontsize=6.8, color=INK, transform=ax3.get_yaxis_transform())
ax3.text(1.1, -1.0, 'observed', va='center', ha='left', fontsize=6.5, color=INK2, transform=ax3.get_yaxis_transform())
ax3.set_yticks(yy); ax3.set_yticklabels(A4.channel.map(lab), fontsize=7); ax3.set_ylim(len(A4) - 0.5, -1.5)
ax3.set_xlim(-0.06, 1.06); ax3.set_xticks([0, 0.5, 1]); ax3.set_xlabel('fraction of 30 draws\nreproducing the observed pattern')
ax3.grid(axis='y', visible=False)
ax3.legend(loc='upper center', bbox_to_anchor=(0.45, -0.32), ncol=1, fontsize=6.3, handletextpad=0.3, borderaxespad=0)
panel_label(ax3, 'c')

src = f'illustration: Monte Carlo, i.i.d. Gaussian vectors with common mean and power-law covariance (lambda_i ~ i^-0.5), {NTR} draws per D, seed 915'
rows = [pd.DataFrame(dict(panel='b', series='summary by D', x=R.D, d_eff=R.d_eff, adj_mean=R.adj_mean, adj_sd=R.adj_sd,
                          stride2_mean=R.stride2_mean, stride2_sd=R.stride2_sd, source=src))]
for D, (a_, s_) in dist.items():
    h1, _ = np.histogram(a_, bins); h2, _ = np.histogram(s_, bins)
    rows.append(pd.DataFrame(dict(panel='a', series=f'D={D} histogram', x=(bins[:-1] + bins[1:]) / 2, adjacent=h1, stride2=h2, source=src)))
rows.append(pd.DataFrame(dict(panel='c', series=A4.channel.map(lab), observed=A4.observed, surrA_frac_obs=A4.surrA_frac_obs,
                              surrB_frac_obs=A4.surrB_frac_obs, surrA_modal=A4.surrA_modal, surrB_modal=A4.surrB_modal,
                              source='review/round6/out/A4_beta_layer_surrogates.csv (post hoc analyzer replay A4; 30 draws per surrogate)')))
write_data(FIG, rows)
save(fig, FIG)
print(FIG); print(R.round(3).to_string())
