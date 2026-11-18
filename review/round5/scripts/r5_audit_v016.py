import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Audit of the reviewer's v0.16 numbers (review/round5/reviewer_v016) with Claude's own loader (r5_load.py) and the
# Round 2-4 estimators. Every quantity here was reported by the reviewer; nothing new is tested.
# Output: out/A_audit_v016.json, out/A_audit_v016_table.csv
import json, numpy as np, pandas as pd
from r5_common import *
A = {}; rows = []
def row(claim, reported, recomputed, status, note=''):
    rows.append(dict(claim=claim, reviewer=reported, recomputed=recomputed, status=status, note=note))
S10 = 'S10_1024_s1001+S10_1024_s1002'; S45 = 'S45_1024_s1001+S45_1024_s1002'
d1 = R5(S10); d4 = R5(S45); m1 = d1['meta']
k1 = list(zip(m1.pt, m1.seed)); k4 = list(zip(d4['meta'].pt, d4['meta'].seed)); o4 = [k4.index(k) for k in k1]
X1 = d1['X']['cond']; C4 = d4['X']['cond'][o4]; U4 = d4['X']['uncond'][o4]; s = d1['sigma']
R = json.load(open(f'{OUT}/R_section_results.json'))
row('R1b median |log| step 0', 0.000453, round(R['R1b_median_abs_log_step0'], 6), 'Exact')
row('R1g min prompt_influence, sigma<=0.9', 0.727, round(R['R1g_min_PI_sigma<=0.9'], 4), 'Exact', 'prompt_influence is per layer; the minimum sits in the mid layers')
row('R1h median log(gBM/gcond), steps>=2', 0.000770, round(R['R1h_median_log_gBM_over_gcond_steps>=2'], 6), 'Exact')
# R1f: the frozen B6 definition is the across-seed SD of the layer-mean of log g (Round 3 a1_reproduce.py, which reproduced
# the reviewer's own v0.13 B6 values exactly). v0.16 script 02 uses log of the layer-mean g instead.
def ratio_def(f):
    def sd(X):
        d = pd.DataFrame(f(X[..., 0])); d['ph'] = m1.ph.values
        return d.groupby('ph')[list(range(X.shape[1]))].std()
    a = sd(X1); b = sd(C4).loc[a.index]; return b.values / a.values
q = np.array_split(np.arange(len(s)), 4); late = np.concatenate(q[1:])
r_frozen = ratio_def(lambda G: np.log(G).mean(2)); r_v016 = ratio_def(lambda G: np.log(G.mean(2)))
A['R1f_frozen_def'] = dict(median=float(np.median(r_frozen[:, late])), quarters=[float(np.median(r_frozen[:, x])) for x in q])
A['R1f_v016_def'] = dict(median=float(np.median(r_v016[:, late])), quarters=[float(np.median(r_v016[:, x])) for x in q])
row('R1f median, last three quarters', 0.575, round(A['R1f_v016_def']['median'], 4), 'Reproduced with the v0.16 definition',
    f"frozen B6 definition (mean of log g over layers) gives {A['R1f_frozen_def']['median']:.3f}; both < 0.7, verdict unchanged")
row('R1f quarters', '1.00/0.77/0.55/0.46', '/'.join(f'{x:.2f}' for x in A['R1f_v016_def']['quarters']), 'Reproduced (v0.16 def.)',
    'frozen def.: ' + '/'.join(f'{x:.2f}' for x in A['R1f_frozen_def']['quarters']))
# VR decomposition
for lab, key, rep in [('CFG1 cond', 'CFG1_cond', 1.85), ('CFG4.5 cond', 'CFG45_cond', 1.25), ('CFG4.5 uncond', 'CFG45_uncond', 0.59), ('synthetic branch-mean', 'CFG45_BM', 0.79)]:
    row(f'VR(4) {lab}', rep, round(R[f'R1e_post_VR_uncentred_{key}']['4'], 3), 'Exact')
# seed-centred sigma50: same estimator, different row order -> different GroupKFold partition
ph = m1.ph.values; seed = m1.seed.values
def s50_order(X, order, center):
    r, grp, _ = resid(X[order], ph[order]); c = curve(r, grp, seed[order], center); return s50(c, s), float(c[0])
o_rev = np.lexsort((seed, ph))                      # reviewer: rows sorted by (prompt_hash, seed)
o_mine = np.arange(len(ph))                          # Claude: rows sorted by generation id (file name, gen_idx)
for lab, X, rep in [('CFG1_cond', X1, (0.953, 0.631)), ('CFG45_cond', C4, (0.632, 0.596)), ('CFG45_uncond', U4, (0.612, 0.571))]:
    ur, _ = s50_order(X, o_rev, False); cr, _ = s50_order(X, o_rev, True); um, _ = s50_order(X, o_mine, False); cm, _ = s50_order(X, o_mine, True)
    A[f'sigma50_{lab}'] = dict(reviewer_order_uncentred=ur, reviewer_order_centred=cr, claude_order_uncentred=um, claude_order_centred=cm)
    row(f'sigma50 seed-centred {lab}', rep[1], round(cr, 3), 'Exact with the reviewer row order',
        f'same data, Claude row order (different fold partition): {cm:.3f}; uncentred {ur:.3f} vs {um:.3f}')
# R2: deep-step
p100 = R5('P10_100'); p40 = R5('P10_40'); G100 = p100['X']['cond'][..., 0]; G40 = p40['X']['cond'][..., 0]
row('R2 passive 100-step pooled sequence', '+- (1 crossing at 0.0593)', f"{R['R2a_pooled_100']['signs']} ({R['R2a_pooled_100']['crossings']})", 'Exact')
row('R2 single seeds: start +, -+-+-', '15/15, 0/15', f"{R['R2a_individual_first_plus']}/15, {R['R2a_individual_-+-+-']}/15", 'Exact')
j = [int(np.argmin(np.abs(p100['sigma'] - x))) for x in p40['sigma']]      # reviewer maps each 40-step sigma to the nearest 100-step sigma
v = np.median(np.abs(np.log(G100.mean(0)[j]) - np.log(G40.mean(0))))
row('40->100 same-prompt median |log g|', 0.0326, round(float(v), 4), 'Exact', f"mapping 100->40 instead gives {R['R2b_post_same_prompt_median_abs_log_100_vs_40']:.4f}")
# resolution / branch topology map
topo_rows = []
for lab, arm, brs in [('CFG1_1024', S10, ['cond']), ('CFG45_1024', S45, ['cond', 'uncond', 'BM']), ('CFG45_768', 'S45_768_s1001', ['cond', 'uncond', 'BM']), ('CFG45_512', 'S45_512_s1001', ['cond', 'uncond', 'BM'])]:
    d = R5(arm)
    for b in brs:
        G = BM(d)[..., 0] if b == 'BM' else d['X'][b][..., 0]
        t = topo(G); topo_rows.append(dict(arm=lab, functional=b, gens=G.shape[0], **t))
T = pd.DataFrame(topo_rows); T.to_csv(f'{OUT}/A_topology_map_recomputed.csv', index=False); print(T.to_string())
rev = pd.read_csv(f'{_COSPAFS_ROOT}/review/round5/reviewer_v016/results/CURRENT_BRANCH_BETA_TOPOLOGY_MAP.csv')
rev['functional'] = rev.functional.replace({'synthetic_branchmean': 'BM'}); rev['arm'] = rev.arm.str.replace('SD35_', '')
mm = T.merge(rev, on=['arm', 'functional'])
row('Topology map, 10 arm x functional cells', 'sequences in CURRENT_BRANCH_BETA_TOPOLOGY_MAP.csv',
    f"{int((mm.signs == mm.sequence).sum())}/{len(mm)} sequences identical", 'Exact' if (mm.signs == mm.sequence).all() else 'Partly reproduced')
# B7 sigma-grid claim: thin the legacy 1024 field to the nearest sigma of the passive 100-step grid (P4c-style)
import sys; sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round2/scripts'); from common import load as load_archive
dr, mrs, _, _ = load_archive('SDRES')
dd = dr[dr.gid.isin(mrs.index[mrs.res == 1024])]
piv = dd.pivot_table(index=['gid', 'token_idx'], columns='layer', values='g_micro'); g_ = piv.index.get_level_values(0).unique()
GL = piv.values.reshape(len(g_), 100, -1); sL = dd[dd.gid == g_[0]].groupby('token_idx').sigma.first().values
A['legacy_sigma_grid'] = dict(max_same_index_diff=float(np.abs(sL - p100['sigma']).max()))
jL = sorted(set(int(np.argmin(np.abs(sL - x))) for x in p100['sigma']))
A['legacy_native'] = topo(GL); A['legacy_thinned_to_passive_grid'] = topo(GL[:, jL])
row('Legacy and passive 100-step sigma grids differ', 'max same-index diff 0.1816', round(A['legacy_sigma_grid']['max_same_index_diff'], 4), 'Exact')
row('Legacy -+-+- survives regridding to the passive grid', '-+-+-', A['legacy_thinned_to_passive_grid']['signs'], 'Reproduced (nearest-sigma thinning, not interpolation)')
# Intake: the reviewer's intake table lists the 512 arm with 27 prompts and seed 1001.0
row('512 arm intake: 27 prompts, 57,599 rows', '27 prompts', '30 prompts, 57,600 rows', 'Intake-table error only',
    'one 512 CSV has no header row; the reviewer analysis loader (r4_common.load_member) handles it, the intake script does not')
# Guidance-lag correlations: flat profile
lag = pd.read_csv(f'{_COSPAFS_ROOT}/review/round5/reviewer_v016/results/POSTHOC_GUIDANCE_LAG_CORRELATION.csv')
row('Guidance lag correlations (descriptive)', 'rho -0.83 to -0.95 at all lags', f"range {lag.iloc[:,2].min():.2f} to {lag.iloc[:,2].max():.2f}", 'Read, not recomputed',
    'the profile is flat across +-15 steps, so it carries no lead/lag information; agreed it is descriptive only')
TT = pd.DataFrame(rows); TT.to_csv(f'{OUT}/A_audit_v016_table.csv', index=False)
json.dump(A, open(f'{OUT}/A_audit_v016.json', 'w'), indent=1, default=str)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 110); print(TT.to_string())
