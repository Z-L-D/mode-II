# Apply the frozen Round 5 decision rules (R5_PREREGISTRATION.md) to out/Q_results.json. Output: out/Q_scorecard.csv
import json, pandas as pd
from r5_common import OUT
Q = json.load(open(f'{OUT}/Q_results.json')); rows = []
def add(i, pred, ok, res): rows.append(dict(id=i, prediction=pred, verdict='CONFIRMED' if ok else 'REFUTED', result=res))
add('Q1a', 'New CFG1 run reproduces Round 3 CFG1', Q['Q1a_median_abs_log_g'] <= 0.01, f"median |log| {Q['Q1a_median_abs_log_g']:.1e}; {Q['Q1a_exact_equal_frac']:.0%} of cells identical to 6 decimals (60 pairs)")
add('Q1b', 'Round 3 ZIB "branch-mean" = mean of the two branches', max(Q['Q1b_ZIB4_median_abs_log_g_BMnew_vs_R3_attn'], Q['Q1b_ZIB4_median_abs_log_g_BMnew_vs_R3_mlp']) <= 0.01,
    f"attn {Q['Q1b_ZIB4_median_abs_log_g_BMnew_vs_R3_attn']:.1e}, mlp {Q['Q1b_ZIB4_median_abs_log_g_BMnew_vs_R3_mlp']:.1e} (19 pairs); ZIB CFG1 retest 0")
add('Q1c', 'Pooling raises g on one trajectory, gap unchanged', 0.04 <= Q['Q1c_median_log_g_pooled_over_BM_steps>=2'] <= 0.10 and Q['Q1c_frac_positive_steps>=2'] >= 0.9 and abs(Q['Q1c_median_log_gap_steps>=2']) <= 0.03,
    f"median +{Q['Q1c_median_log_g_pooled_over_BM_steps>=2']:.4f} (band 0.04-0.10, met by 0.0003); {Q['Q1c_frac_positive_steps>=2']:.0%} of cells > 0; gap {Q['Q1c_median_log_gap_steps>=2']:.1e}")
add('Q1d', 'Same direction at 768 and 512', min(Q['Q1d_768_median_log_g_pooled_over_BM_steps>=2'], Q['Q1d_512_median_log_g_pooled_over_BM_steps>=2']) > 0.03,
    f"768 +{Q['Q1d_768_median_log_g_pooled_over_BM_steps>=2']:.3f}; 512 +{Q['Q1d_512_median_log_g_pooled_over_BM_steps>=2']:.3f}")
add('Q1e', 'Concavity bound sqrt(g_pool) >= mean sqrt(g_branch)', Q['Q1e_frac_cells_bound_holds'] >= 0.9, f"holds in {Q['Q1e_frac_cells_bound_holds']:.1%} of cells; median slack {Q['Q1e_median_slack_log']:.3f} log")
add('Q1f', 'Pooled excess tracks 1 - prompt_influence', Q['Q1f_spearman_excess_vs_1-PI'] >= 0.4, f"rho {Q['Q1f_spearman_excess_vs_1-PI']:.2f}")
add('Q2a', 'ZIB CFG4 cond inherits ZIT geometry', min(Q['Q2a_tangent_ZIB4cond_ZIT_attn'], Q['Q2a_tangent_ZIB4cond_ZIT_mlp']) >= 0.985, f"{Q['Q2a_tangent_ZIB4cond_ZIT_attn']:.3f} attn / {Q['Q2a_tangent_ZIB4cond_ZIT_mlp']:.3f} mlp")
add('Q2b', 'cond closer to ZIT than uncond', Q['Q2a_tangent_ZIB4cond_ZIT_attn'] > Q['Q2b_tangent_ZIB4uncond_ZIT_attn'] and Q['Q2a_tangent_ZIB4cond_ZIT_mlp'] > Q['Q2b_tangent_ZIB4uncond_ZIT_mlp'],
    f"uncond {Q['Q2b_tangent_ZIB4uncond_ZIT_attn']:.3f} / {Q['Q2b_tangent_ZIB4uncond_ZIT_mlp']:.3f}")
add('Q2c', 'ZIB CFG1 vs CFG4 cond geometry', min(Q['Q2c_tangent_ZIB1_ZIB4cond_attn'], Q['Q2c_tangent_ZIB1_ZIB4cond_mlp']) >= 0.98, f"{Q['Q2c_tangent_ZIB1_ZIB4cond_attn']:.3f} / {Q['Q2c_tangent_ZIB1_ZIB4cond_mlp']:.3f}")
w60 = Q['Q3a_R3_60prompt_CFG1_centred_sigma50']['width']; w30 = Q['Q3_CFG1_cond_centred_sigma50']['width']
add('Q3a', 'More prompts stabilise sigma50', w60 < w30, f"5-95% width 60 prompts {w60:.3f} vs 30 prompts {w30:.3f}; medians {Q['Q3a_R3_60prompt_CFG1_centred_sigma50']['median']:.3f} vs {Q['Q3_CFG1_cond_centred_sigma50']['median']:.3f}")
add('Q3b', 'Guidance makes fate onset earlier (cond)', Q['Q3b_median_paired_diff'] > 0 and Q['Q3b_frac_positive'] >= 0.6, f"median paired diff +{Q['Q3b_median_paired_diff']:.3f}; positive in {Q['Q3b_frac_positive']:.1%} of partitions")
a = Q['Q3c_archive30_uncentred_sigma50']
add('Q3c', 'Archive early onset holds on the 30 core prompts (2 runs)', (a['median'] or 0) >= 0.85 and a['nan_frac'] < 0.5, f"sigma50 undefined (R2 never > 0.5) in {a['nan_frac']:.1%} of partitions; median of the rest {a['median']:.3f}")
add('Q4a', 'Walk-like CFG1 persistence with unique seeds', Q['Q4a_VR4_CFG1_40'] > 1.2, f"VR(4) {Q['Q4a_VR4_CFG1_40']:.2f}")
add('Q4b', 'cond > branch-mean > uncond with unique seeds', Q['Q4_VR4_CFG45_40_cond'] > Q['Q4_VR4_CFG45_40_BM'] > Q['Q4_VR4_CFG45_40_uncond'],
    f"cond {Q['Q4_VR4_CFG45_40_cond']:.2f}, BM {Q['Q4_VR4_CFG45_40_BM']:.2f}, uncond {Q['Q4_VR4_CFG45_40_uncond']:.2f}")
add('Q4c', 'VR(10)@100 within 25% of VR(4)@40', all(0.75 <= Q[k] <= 1.25 for k in ['Q4c_ratio_CFG1', 'Q4c_ratio_CFG45_cond']), f"ratios {Q['Q4c_ratio_CFG1']:.2f} (CFG1), {Q['Q4c_ratio_CFG45_cond']:.2f} (CFG4.5 cond)")
add('Q5a', 'CFG1 40->100 shift within seed noise', Q['Q5_CFG1_observed'] <= Q['Q5_CFG1_null_p95'], f"observed {Q['Q5_CFG1_observed']:.4f} vs null p95 {Q['Q5_CFG1_null_p95']:.4f} (p = {Q['Q5_CFG1_perm_p']:.3f}); 100-step g higher in {Q['Q5_CFG1_frac_cells_positive']:.0%} of cells")
add('Q5b', 'CFG4.5 cond 40->100 shift within seed noise', Q['Q5_CFG45_cond_observed'] <= Q['Q5_CFG45_cond_null_p95'], f"observed {Q['Q5_CFG45_cond_observed']:.4f} vs null p95 {Q['Q5_CFG45_cond_null_p95']:.4f}")
z = {k: v['lobe']['z'] for k, v in Q['Q6a'].items() if v['lobe']}
add('Q6a', 'Low-g lobe of +-+- below 2 bootstrap SE in every arm', max(z.values()) < 2, '; '.join(f"{k} z={v:.2f}" for k, v in z.items()))
add('Q6b', 'R3 pooled 512 on the 29 shared prompts is +-', Q['Q6b_R3pooled512_shared']['signs'] == '+-', f"{Q['Q6b_R3pooled512_shared']['signs']} ({Q['Q6b_R3pooled512_shared']['crossings']})")
add('Q7a', 'Contraction < 0.7 in last 3 quarters (pyramid, unique seeds)', Q['Q7a_40_last3q'] < 0.7 and Q['Q7a_100_last3q'] < 0.7, f"40 steps {Q['Q7a_40_last3q']:.3f}; 100 steps {Q['Q7a_100_last3q']:.3f}")
add('Q7b', 'No contraction in the first quarter (>= 0.85)', Q['Q7b_40_first_q'] >= 0.85 and Q['Q7b_100_first_q'] >= 0.85, f"40 steps {Q['Q7b_40_first_q']:.3f}; 100 steps {Q['Q7b_100_first_q']:.3f}")
on = Q['Q8_onsets']
add('Q8', 'Plateau onset in [0.75, 0.90] on every new arm', all(0.75 <= v <= 0.90 for v in on.values()), '; '.join(f"{k} {v:.3f}" for k, v in on.items()))
S = pd.DataFrame(rows); S.to_csv(f'{OUT}/Q_scorecard.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 150); print(S.to_string()); print(S.verdict.value_counts().to_dict())
