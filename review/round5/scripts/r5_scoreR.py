# Round 4 prereg section R (R4_PREREGISTRATION.md, SHA 82ef210a..., frozen 2026-10-06T23:01:29Z), scored by Claude
# on the corrected captures. Rules applied exactly as frozen; deviations of the executed design are reported per row.
# Rows whose estimand depends on a design element that was not executed (unique seed pair per prompt) are NOT SCORABLE;
# their values are computed and reported as post hoc. Output: out/R_section_results.json, out/R_section_scorecard.csv
import json, numpy as np, pandas as pd
from r5_common import *
R = {}; rng = np.random.default_rng(5)
S10 = 'S10_1024_s1001+S10_1024_s1002'; S45 = 'S45_1024_s1001+S45_1024_s1002'
d1 = R5(S10); d4 = R5(S45)
m1, m4 = d1['meta'], d4['meta']
k1 = list(zip(m1.pt, m1.seed)); k4 = list(zip(m4.pt, m4.seed))
assert sorted(k1) == sorted(k4), 'CFG arms are not paired by (prompt, seed)'
o4 = [k4.index(k) for k in k1]                      # reorder CFG4.5 to CFG1 order
C4 = d4['X']['cond'][o4]; U4 = d4['X']['uncond'][o4]; PI4 = d4['PI'][o4]; X1 = d1['X']['cond']; s = d1['sigma']
assert np.allclose(s, d4['sigma'])
# ---- R1a preflight ----
R['R1a'] = dict(hook_mode='passive-branch (all rows)', branches_CFG45=sorted(d4['X']), branches_CFG1=sorted(d1['X']),
                layers=int(X1.shape[2]), steps=int(X1.shape[1]), finite_PI_CFG45=bool(np.isfinite(PI4).all()),
                gens_CFG1=int(X1.shape[0]), gens_CFG45=int(C4.shape[0]))
# ---- R1b step-0 identity ----
lr0 = np.log(C4[:, 0, :, 0] / X1[:, 0, :, 0])
R['R1b_median_abs_log_step0'] = float(np.median(np.abs(lr0))); R['R1b_max_abs_log_step0'] = float(np.abs(lr0).max())
R['R1b_median_abs_log_step1'] = float(np.median(np.abs(np.log(C4[:, 1, :, 0] / X1[:, 1, :, 0]))))
R['R1b_uncond_vs_CFG1_step0'] = float(np.median(np.abs(np.log(U4[:, 0, :, 0] / X1[:, 0, :, 0]))))
# ---- R1c-e (post hoc: shared seeds) ----
ph1 = m1.ph.values; seed = m1.seed.values
for lab, X in [('CFG1_cond', X1), ('CFG45_cond', C4), ('CFG45_uncond', U4), ('CFG45_BM', (C4 + U4) / 2)]:
    r, grp, Fz = resid(X, ph1)
    R[f'R1c_post_seed_identity_acc_{lab}'] = seed_identity(Fz, seed, grp)
    cu = curve(r, grp, seed, False); cc = curve(r, grp, seed, True)
    R[f'R1d_post_sigma50_uncentred_{lab}'] = s50(cu, s); R[f'R1d_post_sigma50_centred_{lab}'] = s50(cc, s)
    R[f'R1d_post_R2step0_uncentred_{lab}'] = float(cu[0]); R[f'R1d_post_R2step0_centred_{lab}'] = float(cc[0])
    R[f'curve_uncentred_{lab}'] = cu.round(4).tolist(); R[f'curve_centred_{lab}'] = cc.round(4).tolist()
    R[f'R1e_post_VR_uncentred_{lab}'] = VR(r); R[f'R1e_post_VR_centred_{lab}'] = VR(seed_center(r, seed))
# ---- R1f spread ratio (B6 definition: across-seed SD of layer-mean log g, per prompt and step) ----
def sd_by_prompt(X, m):
    d = pd.DataFrame(np.log(X[..., 0]).mean(2)); d['ph'] = m.ph.values
    return d.groupby('ph')[list(range(X.shape[1]))].std()
A = sd_by_prompt(X1, m1); B = sd_by_prompt(C4, m1).loc[A.index]
ratio = B.values / A.values; q = np.array_split(np.arange(len(s)), 4)
R['R1f_median_last3q'] = float(np.median(ratio[:, np.concatenate(q[1:])]))
R['R1f_quarters'] = [float(np.median(ratio[:, qq])) for qq in q]
bt = []
for _ in range(2000):
    i = rng.integers(0, ratio.shape[0], ratio.shape[0]); bt.append(np.median(ratio[i][:, np.concatenate(q[1:])]))
R['R1f_prompt_bootstrap_CI95'] = [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]
R['R1f_frac_prompts_median_below_0.7'] = float(np.mean(np.median(ratio[:, np.concatenate(q[1:])], 1) < 0.7))
# ---- R1g prompt influence ----
R['R1g_min_PI_sigma<=0.9'] = float(np.nanmin(PI4[:, s <= 0.9])); R['R1g_median_PI_by_step'] = np.nanmedian(PI4, (0, 2)).round(4).tolist()
R['R1g_min_PI_by_step'] = np.nanmin(PI4, (0, 2)).round(4).tolist()
R['R1g_median_PI_by_layer'] = np.nanmedian(PI4, (0, 1)).round(4).tolist()
R['R1g_frac_gens_PI<0.999_somewhere'] = float(np.mean((PI4[:, s <= 0.9] < 0.999).any((1, 2))))
# ---- R1h branch-mean vs cond ----
R['R1h_median_log_gBM_over_gcond_steps>=2'] = float(np.median(np.log(((C4 + U4) / 2)[:, 2:, :, 0] / C4[:, 2:, :, 0])))
R['R1h_median_log_guncond_over_gcond_steps>=2'] = float(np.median(np.log(U4[:, 2:, :, 0] / C4[:, 2:, :, 0])))
# ---- R2 deep-step passive pyramid ----
p100 = R5('P10_100'); p40 = R5('P10_40')
G100 = p100['X']['cond'][..., 0]; G40 = p40['X']['cond'][..., 0]
R['R2_n_seeds_100'] = int(G100.shape[0]); R['R2_n_seeds_40'] = int(G40.shape[0])
R['R2a_pooled_100'] = topo(G100); R['R2a_pooled_40_same_prompt'] = topo(G40)
ind = [topo(G100[i:i + 1]) for i in range(G100.shape[0])]
R['R2a_individual_first_plus'] = int(sum(x['signs'][0] == '+' for x in ind)); R['R2a_individual_-+-+-'] = int(sum(x['signs'] == '-+-+-' for x in ind))
R['R2a_individual_seq_freq'] = pd.Series([x['signs'] for x in ind]).value_counts().to_dict()
bs = pd.Series([topo(G100[rng.integers(0, len(G100), len(G100))])['signs'] for _ in range(1000)]).value_counts()
R['R2a_seed_bootstrap_seq_freq'] = (bs / bs.sum()).round(4).to_dict()
# R2b as frozen: 100-step passive pyramid vs R3 CFG1 40-step passive (60 prompts, prompt differs), nearest sigma, same layer
_, m3, X3, s3 = R3(S10)
j = [int(np.argmin(np.abs(s3 - x))) for x in p100['sigma']]
lr = np.log(G100.mean(0)) - np.log(X3[..., 0].mean(0))[j]
R['R2b_frozen_median_abs_log_100_vs_R3CFG1'] = float(np.median(np.abs(lr)))
R['R2b_frozen_median_log_100_vs_R3CFG1'] = float(np.median(lr))
# yardstick: prompt-level spread in R3 CFG1 (each prompt's mean field vs the population mean)
pt = np.asarray(m3.pt.astype(str)); pm = np.log(X3[..., 0].mean(0))
R['R2b_yardstick_R3_prompt_median_abs_log'] = float(np.median([np.median(np.abs(np.log(X3[pt == p][..., 0].mean(0)) - pm)) for p in np.unique(pt)]))
# same-prompt control (post hoc design upgrade): 100-step vs 40-step pyramid at nearest sigma
j40 = [int(np.argmin(np.abs(p40['sigma'] - x))) for x in p100['sigma']]
lr2 = np.log(G100.mean(0)) - np.log(G40.mean(0))[j40]
R['R2b_post_same_prompt_median_abs_log_100_vs_40'] = float(np.median(np.abs(lr2)))
R['R2b_post_same_prompt_median_log_100_vs_40'] = float(np.median(lr2))
bt = []
for _ in range(1000):
    a = G100[rng.integers(0, len(G100), len(G100))].mean(0); b = G40[rng.integers(0, len(G40), len(G40))].mean(0)
    bt.append(np.median(np.abs(np.log(a) - np.log(b)[j40])))
R['R2b_post_same_prompt_CI95'] = [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]
json.dump(R, open(f'{OUT}/R_section_results.json', 'w'), indent=1, default=str)
# ---- scorecard ----
dev_p = '26/30 frozen prompts (4 replaced); global seeds 1001/1002 reused across prompts'
rows = [
 ('R1a', 'Preflight passes', 'CONFIRMED', 'passive-branch; {cond,uncond} at 4.5 and {cond} at 1; 24 layers x 40 steps; finite prompt_influence in all 60 core CFG 4.5 gens', dev_p),
 ('R1b', 'cond@4.5 = CFG1 at step 0', 'CONFIRMED' if R['R1b_median_abs_log_step0'] <= 1e-3 else 'REFUTED', f"median |log| {R['R1b_median_abs_log_step0']:.6f} (max {R['R1b_max_abs_log_step0']:.4f})", dev_p),
 ('R1c', 'Seed identity not decodable (unique seeds)', 'NOT SCORABLE', f"unique-seed design not run; post hoc accuracy {R['R1c_post_seed_identity_acc_CFG1_cond']:.3f} (CFG1)", 'shared seeds make seed identity trivially decodable; a different estimand'),
 ('R1d', 'sigma50 in [0.85,0.96] (unique seeds)', 'NOT SCORABLE', f"post hoc seed-centred: CFG1 {R['R1d_post_sigma50_centred_CFG1_cond']:.3f}, CFG4.5 cond {R['R1d_post_sigma50_centred_CFG45_cond']:.3f}", 'unique-seed design not run'),
 ('R1e', 'VR(4) CFG1 cond <= 1.0 (unique seeds)', 'NOT SCORABLE', f"post hoc seed-centred VR4 {R['R1e_post_VR_centred_CFG1_cond'][4]:.2f} (> 1.2 band)", 'unique-seed design not run'),
 ('R1f', 'Guidance contracts cond spread', 'CONFIRMED' if R['R1f_median_last3q'] < 0.7 else 'REFUTED', f"median {R['R1f_median_last3q']:.3f}; prompt bootstrap {R['R1f_prompt_bootstrap_CI95'][0]:.3f}-{R['R1f_prompt_bootstrap_CI95'][1]:.3f}", dev_p),
 ('R1g', 'cond/uncond distinguishable', 'CONFIRMED' if R['R1g_min_PI_sigma<=0.9'] < 0.999 else 'REFUTED', f"min prompt_influence at sigma<=0.9: {R['R1g_min_PI_sigma<=0.9']:.3f}", dev_p),
 ('R1h', 'BM vs cond (report only)', 'REPORTED', f"median log(gBM/gcond), steps>=2: {R['R1h_median_log_gBM_over_gcond_steps>=2']:+.4f}", dev_p),
 ('R2a', '100-step passive: first +, not -+-+-, <=3 crossings', 'CONFIRMED' if (R['R2a_pooled_100']['signs'][0] == '+' and R['R2a_pooled_100']['signs'] != '-+-+-' and R['R2a_pooled_100']['n_cross'] <= 3) else 'REFUTED',
  f"{R['R2a_pooled_100']['signs']}, {R['R2a_pooled_100']['n_cross']} crossing(s); {R['R2a_individual_first_plus']}/15 single seeds start +, {R['R2a_individual_-+-+-']}/15 are -+-+-", '15 of >=30 seeds (precision, not estimand)'),
 ('R2b', '|log(g100/g40)| <= 0.05 vs R3 CFG1', 'CONFIRMED' if R['R2b_frozen_median_abs_log_100_vs_R3CFG1'] <= 0.05 else 'REFUTED',
  f"frozen comparison {R['R2b_frozen_median_abs_log_100_vs_R3CFG1']:.4f} (yardstick {R['R2b_yardstick_R3_prompt_median_abs_log']:.4f}); same-prompt post hoc {R['R2b_post_same_prompt_median_abs_log_100_vs_40']:.4f}", '15 of >=30 seeds (precision, not estimand)'),
 ('R3', 'Instrument A/B', 'NOT SCORABLE', 'no A/B data', 'conditional on R2; not run'),
 ('R4', 'G4 LLM / PSTF (C0-C4, D0-D6)', 'NOT SCORABLE', 'no new data', ''),
]
SC = pd.DataFrame(rows, columns=['id', 'prediction', 'verdict', 'result', 'design_deviation'])
SC.to_csv(f'{OUT}/R_section_scorecard.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
print(SC.to_string())
for k, v in R.items():
    if 'curve' in k or 'by_step' in k: continue
    print(k, ':', v)
