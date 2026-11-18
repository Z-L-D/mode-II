import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Round 5 figures (from out/*.json and the R5/F3/F pickles). Output: review/round5/fig/R5_fig*.png
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from r5_common import *
FIG = f'{_COSPAFS_ROOT}/review/round5/fig'; os.makedirs(FIG, exist_ok=True)
Q = json.load(open(f'{OUT}/Q_results.json')); H = json.load(open(f'{OUT}/posthoc_R5.json')); VN = json.load(open(f'{OUT}/VR_null.json'))
R = json.load(open(f'{OUT}/R_section_results.json'))
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
# ---- Fig 1: pooling on one trajectory ----
def excess(arm):
    _, mp, Xp, _ = R3(arm); dn = R5(arm)
    keep = np.array([len(str(x).split(',')) == 1 for x in mp.seqs]); mp = mp[keep]; Xp = Xp[keep]
    ka = list(zip(mp.pt.astype(str), mp.seed.astype('int64'))); kb = list(zip(dn['meta'].pt, dn['meta'].seed.astype('int64')))
    c = [k for k in ka if k in kb]; ia = [ka.index(k) for k in c]; ib = [kb.index(k) for k in c]
    return np.log(Xp[ia][..., 0] / BM(dn)[ib][..., 0]), dn['PI'][ib], dn['sigma']
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for arm, lab, col in [('S45_1024_s1001+S45_1024_s1002', '1024', '#1f4e79'), ('S45_768_s1001', '768', '#4a90c2'), ('S45_512_s1001', '512', '#9cc3e4')]:
    lg, PI, s = excess(arm); m = np.median(lg, (0, 2)); ax[0].plot(s, m, color=col, label=f'{lab} px');
    if lab == '1024': ex = np.median(lg, 0); dis = np.median(1 - PI, 0)
ax[0].invert_xaxis(); ax[0].axhline(0, color='k', lw=.6); ax[0].set_xlabel('σ'); ax[0].set_ylabel('median log(g pooled / g branch-mean)')
ax[0].set_title('Pooling raises g on the same trajectory'); ax[0].legend(frameon=False)
sc = ax[1].scatter(dis.ravel(), ex.ravel(), c=np.repeat(np.arange(ex.shape[0]), ex.shape[1]), s=6, cmap='viridis')
ax[1].set_xscale('log'); ax[1].set_xlabel('1 − prompt_influence (median, per step × layer)'); ax[1].set_ylabel('pooled excess (log)')
ax[1].set_title(f"Excess tracks branch dissimilarity (ρ = {Q['Q1f_spearman_excess_vs_1-PI']:.2f})"); fig.colorbar(sc, ax=ax[1], label='step')
fig.tight_layout(); fig.savefig(f'{FIG}/R5_fig1_pooling_same_trajectory.png', dpi=150); plt.close(fig)
# ---- Fig 2: seed-fate estimator vs sample size ----
H1 = pd.DataFrame(H['H1']); a = H1[H1.set.str.startswith('archive')].copy()
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for runs, mk in [(2, 'o'), (4, 's')]:
    b = a[a.runs_per_prompt == runs].sort_values('n')
    ax[0].errorbar(b.n, b.sigma50, yerr=[b.sigma50 - b.sigma50_p5, b.sigma50_p95 - b.sigma50], marker=mk, color='#b5523b', capsize=2, label=f'archive σ50, {runs} runs/prompt')
    ax[0].errorbar(b.n, b.rel_onset, yerr=[b.rel_onset - b.rel_p5, b.rel_p95 - b.rel_onset], marker=mk, color='#1f4e79', capsize=2, label=f'archive relative onset, {runs} runs')
new = H1[(H1.set.str.startswith('new')) & (H1.centred)]
ax[0].scatter([60] * len(new), new.sigma50, marker='x', color='#b5523b', label='new arms σ50 (seed-centred)'); ax[0].scatter([60] * len(new), new.rel_onset, marker='x', color='#1f4e79', label='new arms relative onset')
ax[0].set_xlabel('generations in the regression (n)'); ax[0].set_ylabel('onset σ'); ax[0].set_title('σ50 tracks n; the relative onset does not'); ax[0].legend(frameon=False, fontsize=6.5)
ax[0].annotate('archive, 30 prompts × 2 runs:\nσ50 undefined in 98% of partitions', (60, 0.08), fontsize=6.5, color='#b5523b')
ax[1].scatter(H1.n, H1.final_R2, c=['#1f4e79' if s.startswith('archive') else '#b5523b' for s in H1.set])
ax[1].axhline(0.5, color='k', lw=.6, ls=':'); ax[1].text(300, 0.55, 'blue: archive subsamples\nred: R3 / new arms', fontsize=7); ax[1].set_xlabel('n'); ax[1].set_ylabel('R² at the final step (ridge ceiling)'); ax[1].set_title('The ridge ceiling rises with n')
fig.tight_layout(); fig.savefig(f'{FIG}/R5_fig2_seedfate_sample_size.png', dpi=150); plt.close(fig)
# ---- Fig 3: VR with calibration ----
lab = ['core CFG1\ncond', 'core CFG4.5\ncond', 'core\nbranch-mean', 'core\nuncond', 'pyramid CFG1\n(unique seeds)', 'pyramid 4.5\ncond', 'pyramid\nbranch-mean', 'pyramid\nuncond']
val = [R['R1e_post_VR_uncentred_CFG1_cond']['4'], R['R1e_post_VR_uncentred_CFG45_cond']['4'], R['R1e_post_VR_uncentred_CFG45_BM']['4'], R['R1e_post_VR_uncentred_CFG45_uncond']['4'],
       Q['Q4a_VR4_CFG1_40'], Q['Q4_VR4_CFG45_40_cond'], Q['Q4_VR4_CFG45_40_BM'], Q['Q4_VR4_CFG45_40_uncond']]
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.bar(range(8), val, color=['#1f4e79'] * 4 + ['#4a90c2'] * 4)
for k, ls, nm in [('walk_30x2_VR4', '-', 'random walk'), ('ar1_0.8_30x2_VR4', '--', 'AR(1) 0.8'), ('iid_30x2_VR4', ':', 'independent jitter')]:
    ax.axhline(VN[k][0], color='#b5523b', ls=ls, lw=1); ax.text(7.6, VN[k][0], nm, fontsize=7, color='#b5523b', va='bottom', ha='right')
ax.axhline(R['R1e_post_VR_uncentred_CFG45_BM']['4'] * 0 + 0.46, color='grey', lw=.6); ax.text(-0.4, 0.47, 'archive branch-mean 0.41–0.51', fontsize=7, color='grey')
ax.set_xticks(range(8)); ax.set_xticklabels(lab, fontsize=7); ax.set_ylabel('VR(4), 40 steps'); ax.set_title('Persistence against simulated references (same estimator, same shape)')
fig.tight_layout(); fig.savefig(f'{FIG}/R5_fig3_vr_calibrated.png', dpi=150); plt.close(fig)
# ---- Fig 4: lineage ladder ----
P4 = json.load(open(f'{_COSPAFS_ROOT}/review/round4/out/P_results.json'))
names = ['ZIB CFG 1', 'ZIB CFG 4\ncond', 'ZIB CFG 4\nbranch-mean', 'ZIB CFG 4\nuncond', 'ZIB CFG 4\npooled (archive)']
fig, ax = plt.subplots(figsize=(7, 3.2))
for k, (sub, off, col) in enumerate([('attn', -0.18, '#1f4e79'), ('mlp', 0.18, '#9cc3e4')]):
    v = [Q[f'Q2_ref_tangent_ZIB1new_ZIT_{sub}'], Q[f'Q2a_tangent_ZIB4cond_ZIT_{sub}'], Q[f'Q2_ref_tangent_ZIB4BMsynth_ZIT_{sub}'], Q[f'Q2b_tangent_ZIB4uncond_ZIT_{sub}'], P4[f'ref_tangent_ZIB4pooled_ZIT_{sub}']]
    ax.bar(np.arange(5) + off, v, width=0.36, color=col, label=sub)
ax.set_ylim(0.93, 1.0); ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=7.5); ax.set_ylabel('tangent cosine with ZIT'); ax.legend(frameon=False)
ax.set_title('Base→Turbo depth geometry by capture functional'); fig.tight_layout(); fig.savefig(f'{FIG}/R5_fig4_lineage_ladder.png', dpi=150); plt.close(fig)
# ---- Fig 5: contraction timing ----
fig, ax = plt.subplots(figsize=(7.5, 3.3)); s40 = R5('S10_1024_s1001')['sigma']
ax.plot(s40, H['H4_core_per_step_median_ratio'], color='#1f4e79', label='core: 30 prompts, paired seeds (median over prompts)')
for lab_, key, col in [('pyramid 40 steps', 'P10_40', '#b5523b'), ('pyramid 100 steps', 'P10_100', '#e0a458')]:
    A = R5(key)['X']['cond']; B = R5(key.replace('P10', 'P45'))['X']['cond']
    r = np.log(B[..., 0]).mean(2).std(0, ddof=1) / np.log(A[..., 0]).mean(2).std(0, ddof=1); ax.plot(R5(key)['sigma'], r, color=col, lw=1, label=f'{lab_}: 1 prompt, unpaired unique seeds')
ax.axhline(1, color='k', lw=.6); ax.axhline(0.7, color='k', lw=.6, ls=':'); ax.invert_xaxis(); ax.set_xlabel('σ'); ax.set_ylabel('SD(CFG 4.5 cond) / SD(CFG 1)')
ax.set_title('When guidance contracts across-seed spread'); ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(f'{FIG}/R5_fig5_contraction_timing.png', dpi=150); plt.close(fig)
# ---- Fig 6: onset vs resolution ----
fig, ax = plt.subplots(figsize=(5.5, 3.2)); res = [512, 768, 1024]
for key, col, nm in [('cond', '#1f4e79', 'cond branch (new)'), ('pooled_R3', '#9cc3e4', 'pooled (Round 3)')]:
    raw = [H['H3'][key][str(r)]['raw'] for r in res]; sh = [H['H3'][key][str(r)]['shifted_to_1024'] for r in res]
    ax.plot(res, raw, 'o-', color=col, label=f'{nm}, raw σ'); ax.plot(res, sh, 's--', color=col, label=f'{nm}, SD3 resolution shift')
ax.axhspan(0.75, 0.90, color='grey', alpha=.12); ax.set_xticks(res); ax.set_xlabel('resolution (px)'); ax.set_ylabel('plateau onset σ (E3)'); ax.legend(frameon=False, fontsize=7)
ax.set_title('Onset moves with resolution; the SD3 shift realigns cond'); fig.tight_layout(); fig.savefig(f'{FIG}/R5_fig6_onset_resolution.png', dpi=150); plt.close(fig)
print(sorted(os.listdir(FIG)))
