"""Check the Round 11-18 numbers of the Mode-I and Mode-II papers against the outputs that produce them, and write NUMBERS.md.

Usage:  python check_numbers.py            (from the repository root; exit status 1 if any check fails)

Each entry is (paper(s), printed value, source file, how the value is read, script that writes the source).
A check passes when (1) the value read from the source, rounded as printed, equals the printed value and (2) the printed
value occurs in each named paper's main.tex or supplement.tex (as it is typeset there, e.g. inside $...$).
Scope: the numbers from Rounds 11-18 and the post hoc analyses (real Base, review checks, RC33 native grid, Round 18 addendum B). All other numbers are
checked where they were produced: every figure script asserts its transcribed ledger values (ledger_has) and writes the
plotted values to papers/<paper>/data/, and each review round's RESULTS file lists its outputs.
"""
import json, os, sys, re
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.join(ROOT, 'review', *p)
R11 = R('peer_review_2026-10-17', 'r11', 'out'); R12 = R('zib_turbo_mixup_2026-10-18', 'r12', 'out'); R13 = R('zib_turbo_mixup_2026-10-18', 'r13', 'out')
R14 = R('ace_base_r14', 'out'); R6 = R('round6', 'out')
S_R11 = 'review/peer_review_2026-10-17/r11/scripts/score_r11.py'; S_R12 = 'review/zib_turbo_mixup_2026-10-18/r12/scripts/score_r12.py'
S_R128 = 'review/zib_turbo_mixup_2026-10-18/r12/scripts/r12_8_crossings.py'; S_PH = 'review/zib_turbo_mixup_2026-10-18/r12/scripts/posthoc_hierarchy_realbase.py'
S_R14 = 'review/ace_base_r14/scripts/score_r14.py'; S_R14B = 'review/ace_base_r14/scripts/r14_B_replay.py'; S_R14P = 'review/ace_base_r14/scripts/posthoc_r14.py'
S_R13 = 'review/zib_turbo_mixup_2026-10-18/r13/scripts/'

_cache = {}
def csv(path):
    if path not in _cache: _cache[path] = pd.read_csv(path)
    return _cache[path]
def cell(path, where, col):
    d = csv(path)
    for k, v in where.items(): d = d[d[k] == v]
    assert len(d) == 1, (path, where, len(d)); return float(d.iloc[0][col])
def jget(path, *keys):
    o = json.load(open(path))
    for k in keys: o = o[k]
    return o
def detail(path, idx, pattern, group=1):
    d = csv(path).set_index('id').loc[idx, 'detail']; m = re.search(pattern, d); assert m, (idx, pattern, d); return float(m.group(group))

T12 = os.path.join(R12, 'r12_tangents.csv'); T14 = os.path.join(R14, 'r14_tangents.csv'); T11 = os.path.join(R11, 'r11_tangents.csv')
PH = os.path.join(R12, 'posthoc_hierarchy_margins.csv'); C8 = os.path.join(R12, 'r12_8_crossings.csv')
BB = os.path.join(R14, 'posthoc_backbone_clock.csv'); VA = os.path.join(R14, 'r14_verdicts_A.csv'); R13V = os.path.join(R13, 'r13_verdicts.csv')
t12 = lambda a, b, s, c='tangent': cell(T12, dict(a=a, b=b, sub=s), c)
t14 = lambda a, b, s, c='tangent': cell(T14, dict(a=a, b=b, sub=s), c)
t11 = lambda a, b, s, v, c='tangent': cell(T11, {'a': a, 'b': b, 'sub': s, 'var': v}, c)
I, II, BOTH = ('I',), ('II',), ('I', 'II')
R15D = R('r15_deep_dive'); S_R15 = 'review/r15_deep_dive/scripts/score_r15.py'; S_X4 = 'review/r15_deep_dive/exploratory/x4_tangent_trim.py'; S_X1 = 'review/r15_deep_dive/exploratory/x1_content.py'
X4 = os.path.join(R15D, 'exploratory', 'x4_tangent_trim.csv'); X1B = os.path.join(R15D, 'exploratory', 'x1b_rsa.csv'); B4 = os.path.join(R15D, 'out', 'R15_4_block_null.csv'); C2 = os.path.join(R15D, 'out', 'R15_2_cfg_seed_share.csv')
RCD = os.path.join(R15D, 'review_checks', 'out'); S_RC5 = 'review/r15_deep_dive/review_checks/rc5_glm_kimi.py'; S_RC5B = 'review/r15_deep_dive/review_checks/rc5b_lowpass.py'
RC18, RC19, RC20, RC21 = [os.path.join(RCD, f) for f in ('rc18_per_coordinate.csv', 'rc19_gate_statistic.csv', 'rc20_onset_threshold.csv', 'rc21_pooling_by_resolution.csv')]
S_RC18 = 'review/r15_deep_dive/review_checks/rc18_round3_checks.py'; S_RC20 = 'review/r15_deep_dive/review_checks/rc20_onset_pooling.py'
L6 = R('r15_deep_dive', 'out', 'R15_6_llm_tangents.csv'); D15 = os.path.join(RCD, 'd15_residual_transfer.csv')
RC9, RC15, RC16, RC17 = [os.path.join(RCD, f) for f in ('rc9_caption_energy.csv', 'rc15_rowset_criteria.csv', 'rc16_rowset_rsa.csv', 'rc17_beta_curves_recompute.csv')]
RC13 = os.path.join(RCD, 'rc13_block_null.csv'); S_RC13 = 'review/r15_deep_dive/review_checks/rc13_block_null.py'
RC14 = os.path.join(RCD, 'rc14_surrogates_1000.csv'); S_RC14 = 'review/r15_deep_dive/review_checks/rc14_surrogates_1000.py'
R15P = os.path.join(R15D, 'out', 'R15_5_per_pair.csv'); S_R15 = 'review/r15_deep_dive/scripts/score_r15.py'
S_RC15 = 'review/r15_deep_dive/review_checks/rc15_rowset_criteria.py'; S_RC16 = 'review/r15_deep_dive/review_checks/rc16_rowset_rsa.py'
RC5D, RC5B, RC5M, RC6, RC7, RC8, RC10, RC11 = [os.path.join(RCD, f) for f in ('rc5_depth_selfcontrol.csv', 'rc5b_lowpass.csv', 'rc5_matched_depth_unrelated.csv', 'rc5_rc6_smoothing_asymmetry.csv', 'rc7_sd35_phase.csv', 'rc8_rowset_tiers.csv', 'rc10_identity_gate.csv', 'rc11_family_separation.csv')]
L7 = os.path.join(R15D, 'out', 'R15_7_llm_dit_rsa.csv'); L6 = os.path.join(R15D, 'out', 'R15_6_llm_tangents.csv'); H1 = os.path.join(R15D, 'out', 'R15_1_handoff.csv')

E = [
    # --- Round 12: real Z-Image Base lineage (preregistered) ---
    (BOTH, '0.707', 3, lambda: t12('Z1', 'ZIT', 'attn'), T12, S_R12, 'R12-1 lineage tangent, attention'),
    (BOTH, '0.719', 3, lambda: t12('Z1', 'ZIT', 'mlp'), T12, S_R12, 'R12-1 lineage tangent, MLP'),
    (I, '0.292', 3, lambda: t12('Z1', 'ZIT', 'attn', 'null_p95'), T12, S_R12, 'R12-1 null p95, attention'),
    (I, '0.198', 3, lambda: t12('Z1', 'ZIT', 'mlp', 'null_p95'), T12, S_R12, 'R12-1 null p95, MLP'),
    (I, '0.713', 3, lambda: t12('Z1', 'Z3', 'attn'), T12, S_R12, 'R12-4 identical settings, attention'),
    (I, '0.697', 3, lambda: t12('Z1', 'Z3', 'mlp'), T12, S_R12, 'R12-4 identical settings, MLP'),
    (I, '0.714', 3, lambda: t12('Z2', 'ZIT', 'attn'), T12, S_R12, 'R12-5 Base CFG 1, attention'),
    (I, '0.718', 3, lambda: t12('Z2', 'ZIT', 'mlp'), T12, S_R12, 'R12-5 Base CFG 1, MLP'),
    (I, '0.990', 3, lambda: t12('Z3', 'ZIT', 'attn'), T12, S_R12, 'same weights: matched Turbo rerun vs ZIT, attention'),
    (BOTH, '0.992', 3, lambda: t12('Z3', 'ZIT', 'mlp'), T12, S_R12, 'same weights: matched Turbo rerun vs ZIT, MLP'),
    (I, '0.996', 3, lambda: t12('Z1', 'Z1U', 'attn'), T12, S_R12, 'same weights: Base cond vs uncond, attention'),
    (I, '0.963', 3, lambda: t12('Z1', 'Z1U', 'mlp'), T12, S_R12, 'same weights: Base cond vs uncond, MLP'),
    (I, '0.212', 3, lambda: max(json.load(open(os.path.join(R12, 'r12_unrelated_attn.json'))).values()), os.path.join(R12, 'r12_unrelated_attn.json'), S_R12, 'largest unrelated tangent (Base/Turbo vs Mochi, SD3.5)'),
    # --- Round 12-8: layer-axis crossings (descriptive) ---
    (BOTH, '50', 0, lambda: 100 * cell(C8, dict(cube='Z1_attn'), 'boot_mode_share'), C8, S_R128, 'Base attention: 1 crossing modal share (%)'),
    (BOTH, '11', 0, lambda: 100 * cell(C8, dict(cube='Z1_attn'), 'share3'), C8, S_R128, 'Base attention: 3-crossing share (%)'),
    (I, '59', 0, lambda: 100 * cell(C8, dict(cube='ZIB_attn'), 'boot_mode_share'), C8, S_R128, 'ZIT-25 attention: 3 crossings modal (%)'),
    (I, '69', 0, lambda: 100 * cell(C8, dict(cube='ZIB_mlp'), 'boot_mode_share'), C8, S_R128, 'ZIT-25 MLP: 3 crossings modal (%)'),
    (BOTH, '74', 0, lambda: 100 * cell(os.path.join(R11, 'r11_beta_layer_by_variant.csv'), dict(cube='Z2V0_mlp'), 'share3'), os.path.join(R11, 'r11_beta_layer_by_variant.csv'), S_R11, 'Base CFG 1 MLP all rows: 3 crossings (%)'),
    # --- Round 11: row-split controls (preregistered; caption rows post hoc) ---
    (I, '0.928', 3, lambda: t11('Base', 'Turbo', 'attn', 'V1'), T11, S_R11, 'image rows, attention'),
    (II, '0.93', 2, lambda: t11('Base', 'Turbo', 'attn', 'V1'), T11, S_R11, 'image rows, attention'),
    (I, '0.754', 3, lambda: t11('Base', 'Turbo', 'mlp', 'V1'), T11, S_R11, 'image rows, MLP'),
    (II, '0.75', 2, lambda: t11('Base', 'Turbo', 'mlp', 'V1'), T11, S_R11, 'image rows, MLP'),
    (I, '0.720', 3, lambda: t11('Base', 'Turbo', 'attn', 'V0'), T11, S_R11, 'all rows, attention'),
    (I, '0.714', 3, lambda: t11('Base', 'Turbo', 'mlp', 'V0'), T11, S_R11, 'all rows, MLP'),
    (I, '0.762', 3, lambda: t11('Base', 'Turbo', 'attn', 'V3'), T11, S_R11, 'centred, attention'),
    (II, '0.76', 2, lambda: t11('Base', 'Turbo', 'attn', 'V3'), T11, S_R11, 'centred, attention'),
    (I, '0.615', 3, lambda: t11('Base', 'Turbo', 'mlp', 'V3'), T11, S_R11, 'centred, MLP'),
    (II, '0.61', 2, lambda: t11('Base', 'Turbo', 'mlp', 'V3'), T11, S_R11, 'centred, MLP'),
    (I, '0.679', 3, lambda: t11('Base', 'Turbo', 'attn', 'V4'), T11, S_R11, 'top-8 removed, attention'),
    (II, '0.68', 2, lambda: t11('Base', 'Turbo', 'attn', 'V4'), T11, S_R11, 'top-8 removed, attention'),
    (I, '0.435', 3, lambda: t11('Base', 'Turbo', 'mlp', 'V4'), T11, S_R11, 'top-8 removed, MLP'),
    (II, '0.43', 2, lambda: t11('Base', 'Turbo', 'mlp', 'V4'), T11, S_R11, 'top-8 removed, MLP'),
    (II, '0.70', 2, lambda: cell(os.path.join(R11, 'posthoc_text_vs_image_rows.csv'), dict(var='V2_caption', sub='attn'), 'tangent'), os.path.join(R11, 'posthoc_text_vs_image_rows.csv'), S_R11, 'caption rows, attention (post hoc)'),
    (BOTH, '-0.13', 2, lambda: min(t11(a, 'SD35M', 'attn', v) for a in ('Base', 'Turbo') for v in ('V1', 'V3', 'V4')), T11, S_R11, 'R11-3 unrelated minimum'),
    (BOTH, '0.02', 2, lambda: max(t11(a, 'SD35M', 'attn', v) for a in ('Base', 'Turbo') for v in ('V1', 'V3', 'V4')), T11, S_R11, 'R11-3 unrelated maximum'),
    # --- Round 13 (split-half holdout) ---
    (I, '4.3', 1, lambda: detail(R13V, 'H2', r'ratio ([0-9.]+)'), R13V, S_R13, 'H2 attention ratio'),
    (I, '2.8', 1, lambda: detail(R13V, 'H2', r'mlp .*ratio ([0-9.]+)'), R13V, S_R13, 'H2 MLP ratio'),
    (I, '0.16', 2, lambda: detail(R13V, 'H4', r'attn mid ([0-9.]+)'), R13V, S_R13, 'H4 attention, layers 17-22'),
    (I, '1.09', 2, lambda: detail(R13V, 'H4', r'mlp mid [0-9.]+ early ([0-9.]+)'), R13V, S_R13, 'H4 MLP, early layers'),
    # --- Round 14: ACE-Step Base (preregistered) ---
    (I, '0.850', 3, lambda: t14('ACEB', 'ACET', 'attn'), T14, S_R14, 'R12-6 ACE lineage tangent, attention'),
    (I, '0.572', 3, lambda: t14('ACEB', 'ACET', 'mlp'), T14, S_R14, 'R12-6 ACE lineage tangent, MLP'),
    (I, '0.316', 3, lambda: t14('ACEB', 'ACET', 'attn', 'null_p95'), T14, S_R14, 'R12-6 null p95, attention'),
    (I, '0.242', 3, lambda: t14('ACEB', 'ACET', 'mlp', 'null_p95'), T14, S_R14, 'R12-6 null p95, MLP'),
    (I, '0.542', 3, lambda: t14('ACEB', 'ACET', 'mlp', 'null_p95') + 0.30, T14, S_R14, 'R12-6 MLP bound'),
    (BOTH, '0.997', 3, lambda: t14('ACEB', 'ACEBU', 'attn'), T14, S_R14, 'R14-3 cond vs uncond, attention'),
    (BOTH, '0.964', 3, lambda: t14('ACEB', 'ACEBU', 'mlp'), T14, S_R14, 'R14-3 cond vs uncond, MLP'),
    (I, '0.992', 3, lambda: t14('ACEB0', 'ACEB1', 'attn'), T14, S_R14, 'R14-1 GPU 0 vs 1, attention'),
    (I, '0.972', 3, lambda: t14('ACEB0', 'ACEB1', 'mlp'), T14, S_R14, 'R14-1 GPU 0 vs 1, MLP'),
    (I, '2.30', 2, lambda: jget(os.path.join(R14, 'r14_details_A.json'), 'R14_4', 'ACEB', 'drop'), os.path.join(R14, 'r14_details_A.json'), S_R14, 'R14-4 Base MLP last-layer drop'),
    (BOTH, '14', 0, lambda: detail(VA, 'R14-2', r'attn (\d+)/24'), VA, S_R14, 'R14-2 layers with Base < Turbo, attention'),
    (BOTH, '0.254', 3, lambda: float(re.search(r'dR2 attn ([0-9.]+)', csv(os.path.join(R14, 'r14_B_scores.csv')).set_index('id').loc['B4', 'result']).group(1)), os.path.join(R14, 'r14_B_scores.csv'), S_R14B, 'B4 replay, attention dR2'),
    (II, '0.671', 3, lambda: float(re.search(r'mlp ([0-9.]+) \(R2', csv(os.path.join(R14, 'r14_B_scores.csv')).set_index('id').loc['B4', 'result']).group(1)), os.path.join(R14, 'r14_B_scores.csv'), S_R14B, 'B4 replay, MLP dR2'),
    (I, '0.957', 3, lambda: float(__import__('pickle').load(open(os.path.join(os.environ.get('SCR', os.path.join(ROOT, 'cache')), 'F_ACEB_attn.pkl'), 'rb'))[3][jget(os.path.join(R14, 'r14_B_details.json'), 'B9', 'attn', 'argmax')]), os.path.join(R14, 'r14_B_details.json') + ' + cache/F_ACEB_attn.pkl sigma', S_R14B, 'B9 attention peak sigma'),
    (II, '0.46', 2, lambda: jget(os.path.join(R14, 'r14_B_details.json'), 'B10', 'attn', 'anova')[1], os.path.join(R14, 'r14_B_details.json'), S_R14B, 'B10 attention share after first interval'),
    (II, '0.41', 2, lambda: jget(os.path.join(R14, 'r14_B_details.json'), 'B10', 'mlp', 'anova')[1], os.path.join(R14, 'r14_B_details.json'), S_R14B, 'B10 MLP share after first interval'),
    (BOTH, '0.27', 2, lambda: csv(os.path.join(R14, 'posthoc_ace_vs_image_tangents.csv')).tangent.max(), os.path.join(R14, 'posthoc_ace_vs_image_tangents.csv'), S_R14P, 'ACE vs image models, maximum (post hoc)'),
    (II, '-0.11', 2, lambda: csv(os.path.join(R14, 'posthoc_ace_vs_image_tangents.csv')).tangent.min(), os.path.join(R14, 'posthoc_ace_vs_image_tangents.csv'), S_R14P, 'ACE vs image models, minimum (post hoc)'),
    (I, '0.81', 2, lambda: cell(BB, dict(cube='ACEB', sub='attn'), 'pc1'), BB, S_R14P, 'ACE Base backbone PC1, attention (post hoc)'),
    (I, '1.87', 2, lambda: cell(BB, dict(cube='ACEB', sub='attn'), 'eff_rank'), BB, S_R14P, 'ACE Base backbone rank, attention (post hoc)'),
    # --- Round 15 (preregistered, 3acbb027) and the 2026-11-05 exploratory checks it replicates ---
    (BOTH, '0.38', 2, lambda: cell(X4, dict(a='ACEB_mlp', b='ACET_mlp', trim='drop last 1'), 'tangent'), X4, S_X4, 'ACE MLP lineage tangent, last depth step removed (post hoc)'),
    (I, '0.16', 2, lambda: cell(X4, dict(a='ACEB_mlp', b='ACET_mlp', trim='drop first+last 2'), 'tangent'), X4, S_X4, 'ACE MLP lineage tangent, first and last two removed (post hoc)'),
    (I, '0.44', 2, lambda: csv(B4).query("kind != 'unrelated' and a != 'ACEB_mlp'").margin.min(), B4, S_R15, 'R15-4 smallest same/lineage margin over block null'),
    (II, '0.70', 2, lambda: csv(B4).margin.max(), B4, S_R15, 'R15-4 largest margin over block null'),
    (BOTH, '0.19', 2, lambda: cell(B4, dict(a='ACEB_mlp', b='ACET_mlp'), 'margin'), B4, S_R15, 'R15-4 ACE MLP margin over block null'),
    (I, '0.23', 2, lambda: csv(C2).share_cfg45.iloc[1:].min(), C2, S_R15, 'R15-2 seed share at CFG 4.5, minimum after step 0'),
    (I, '0.66', 2, lambda: csv(C2).share_cfg1.iloc[1:].max(), C2, S_R15, 'R15-2 seed share at CFG 1, maximum'),
    (BOTH, '0.61', 2, lambda: csv(X1B).query("a == 'Z1_attn' and b == 'ZIT_attn'").rsa.min(), X1B, S_X1, 'Base-Turbo prompt RSA, attention, minimum (post hoc)'),
    (BOTH, '0.67', 2, lambda: csv(X1B).query("a == 'Z1_attn' and b == 'ZIT_attn'").rsa.max(), X1B, S_X1, 'Base-Turbo prompt RSA, attention, maximum (post hoc)'),
    (I, '0.14', 2, lambda: csv(L7).rsa.max(), L7, S_R15, 'R15-7 LLM vs Z-Image Base prompt RSA, maximum'),
    (II, '0.66', 2, lambda: cell(L6, dict(sub='mlp', a='Q35', b='Q25', drop_last=0), 'tangent'), L6, S_R15, 'R15-6 Qwen3.5 vs Qwen2.5 MLP tangent'),
    (BOTH, '0.68', 2, lambda: cell(H1, dict(channel='SD3.5M CFG4.5 attn', step=4), 'rsa'), H1, S_R15, 'R15-1 SD3.5 M branch RSA at 10% of steps'),
    # --- post hoc hierarchy with the real Base ---
    (II, '-0.018', 3, lambda: cell(PH, dict(pair='Z1-ZIT'), 'L0_D1'), PH, S_PH, 'lineage margin, envelope g (post hoc)'),
    (II, '+0.003', 3, lambda: cell(PH, dict(pair='Z1-ZIT'), 'L0_D2'), PH, S_PH, 'lineage margin, envelope log g (post hoc)'),
    (II, '+0.036', 3, lambda: cell(PH, dict(pair='Z1-ZIT'), 'L1_field'), PH, S_PH, 'lineage margin, field (post hoc)'),
    (II, '+0.060', 3, lambda: cell(PH, dict(pair='Z1-ZIT'), 'L2_shape'), PH, S_PH, 'lineage margin, shape (post hoc)'),
    (II, '+0.496', 3, lambda: cell(PH, dict(pair='Z1-ZIT'), 'L3_tangent'), PH, S_PH, 'lineage margin, tangent (post hoc)'),
    (II, '0.982', 3, lambda: cell(os.path.join(R12, 'posthoc_hierarchy_realbase.csv'), dict(sub='attn', a='Z1', b='SD35M'), 'L0_D1'), os.path.join(R12, 'posthoc_hierarchy_realbase.csv'), S_PH, 'Base vs SD3.5 M envelope (post hoc)'),
    (II, '0.964', 3, lambda: cell(os.path.join(R12, 'posthoc_hierarchy_realbase.csv'), dict(sub='attn', a='Z1', b='ZIT'), 'L0_D1'), os.path.join(R12, 'posthoc_hierarchy_realbase.csv'), S_PH, 'Base vs Turbo envelope (post hoc)'),
    (II, '+0.703', 3, lambda: jget(os.path.join(R6, 'S1_results.json'), 'margin', 'L3_tangent'), os.path.join(R6, 'S1_results.json'), 'review/round6/scripts/r6c_S1_hierarchy.py', 'same-weights margin, tangent (R6, preregistered)'),
    # --- post hoc review checks (Kimi, GLM, DeepSeek, Qwen; review/r15_deep_dive/review_checks) ---
    (BOTH, '0.15', 2, lambda: cell(RC5D, dict(model='MOCHI_attn', Lp=38, op='decimate'), 'tangent'), RC5D, S_RC5, 'RC5 Mochi vs 38-layer subsample of itself'),
    (BOTH, '0.08', 2, lambda: cell(RC5D, dict(model='SD35L_attn', Lp=30, op='decimate'), 'tangent'), RC5D, S_RC5, 'RC5 SD3.5 L vs 30-layer subsample of itself'),
    (BOTH, '-0.07', 2, lambda: csv(RC5M).grid24.max(), RC5M, S_RC5, 'RC5 ACE-Step vs SD3.5 M (depth-matched), upper end'),
    (BOTH, '-0.11', 2, lambda: csv(RC5M).grid24.min(), RC5M, S_RC5, 'RC5 ACE-Step vs SD3.5 M (depth-matched), lower end'),
    (BOTH, '0.022', 3, lambda: csv(RC6).asym.abs().max(), RC6, S_RC5, 'RC6 largest tangent asymmetry'),
    (BOTH, '0.84', 2, lambda: csv(RC5B).query("kind == 'unrelated' and h > 0").tangent.max(), RC5B, S_RC5B, 'RC5 low-pass: largest unrelated tangent'),
    (I, '0.37', 2, lambda: csv(RC7).tangent.max(), RC7, S_RC5, 'RC7 SD3.5 M-L tangent under depth shift, maximum'),
    (I, '0.928', 3, lambda: cell(RC8, dict(sub='attn', rows='image rows'), 'lineage_base_turbo'), RC8, S_RC5, 'RC8 image-row lineage, attention'),
    (I, '0.997', 3, lambda: cell(RC8, dict(sub='attn', rows='image rows'), 'same_weights_turbo_splithalf'), RC8, S_RC5, 'RC8 image-row same weights (Turbo split halves), attention'),
    (I, '0.699', 3, lambda: cell(RC8, dict(sub='attn', rows='caption rows'), 'lineage_base_turbo'), RC8, S_RC5, 'RC8 caption-row lineage, attention'),
    (I, '0.937', 3, lambda: csv(RC10).query("kind == 'same' and n_gen != n_gen").r.min(), RC10, S_RC5, 'RC10 identity gate, lowest same-weights r'),
    (I, '0.697', 3, lambda: csv(RC10).query("kind == 'different' and n_gen != n_gen").r.max(), RC10, S_RC5, 'RC10 identity gate, highest different-checkpoint r'),
    (I, '86', 0, lambda: 100 * csv(RC11)['ace_mlp_draws_above_0.542'].iloc[0], RC11, S_RC5, 'RC11 share of ACE MLP draws above 0.542 (%)'),
    (I, '0.069', 3, lambda: cell(RC15, dict(sub='attn', rows='image rows'), 'gap_same'), RC15, S_RC15, 'RC15 image-row attention: same weights minus lineage'),
    (I, '0.014', 3, lambda: 0.942 - cell(RC8, dict(sub='attn', rows='image rows'), 'lineage_base_turbo'), RC8, S_RC5, 'RC15 image-row lineage vs pooled-capture same weights 0.942'),
    (BOTH, '0.96', 2, lambda: csv(RC9).query("capture == 'ZITSV_mlp' and layer >= 8").txt_energy_share.mean(), RC9, S_RC5, 'RC9 Turbo MLP caption energy, mean over layers 8-29'),
    (BOTH, '0.99', 2, lambda: csv(RC9).query("capture == 'Z2_mlp' and layer >= 8").txt_energy_share.mean(), RC9, S_RC5, 'RC9 Base MLP caption energy, mean over layers 8-29'),
    (BOTH, '0.71', 2, lambda: csv(RC9).query("capture == 'ZITSV_mlp' and layer >= 27").txt_energy_share.min(), RC9, S_RC5, 'RC9 Turbo MLP caption energy, lowest of last three layers'),
    (I, '0.931', 3, lambda: csv(R15P).query("kind == 'same'")['5%'].min(), R15P, S_R15, 'R15-5 lowest same-weights 5th percentile'),
    (I, '0.862', 3, lambda: csv(R15P).query("kind == 'lineage'")['95%'].max(), R15P, S_R15, 'R15-5 highest lineage 95th percentile'),
    (I, '0.535', 3, lambda: csv(R15P).query("kind == 'lineage'")['5%'].min(), R15P, S_R15, 'R15-5 lowest lineage 5th percentile'),
    (I, '0.248', 3, lambda: csv(R15P).query("kind == 'unrelated'")['95%'].max(), R15P, S_R15, 'R15-5 highest unrelated 95th percentile'),
    (I, '0.11', 2, lambda: csv(RC16).query("rows == 'image rows' and kind == 'unrelated'").rsa.min(), RC16, S_RC16, 'RC16 image-row unrelated RSA, minimum'),
    (I, '0.62', 2, lambda: csv(RC16).query("rows == 'image rows' and kind == 'lineage' and a == 'Z2_attn'").rsa.max(), RC16, S_RC16, 'RC16 image-row lineage attention RSA, maximum'),
    (I, '0.84', 2, lambda: csv(RC16).query("rows == 'caption rows' and a == 'Z2_attn'").rsa.max(), RC16, S_RC16, 'RC16 caption-row lineage attention RSA, maximum'),
    (I, '0.006', 3, lambda: csv(RC16).query("rows in ['image rows', 'centred']").p.max(), RC16, S_RC16, 'RC16 largest p on image rows and centred'),
    (I, '13', 0, lambda: cell(RC13, dict(channel='SD35M archive attn'), 'observed_3cross_steps'), RC13, S_RC13, 'RC13 SD3.5 M three-crossing steps'),
    (I, '7', 0, lambda: cell(RC13, dict(channel='SD35M archive attn'), 'null_max'), RC13, S_RC13, 'RC13 SD3.5 M step-joint null maximum'),
    (I, '16', 0, lambda: cell(RC13, dict(channel='MOCHI archive CFG4.5 attn'), 'null_max'), RC13, S_RC13, 'RC13 Mochi step-joint null maximum'),
    (I, '15.8', 1, lambda: cell(RC13, dict(channel='ZIT-25 archive mlp'), 'null_mean'), RC13, S_RC13, 'RC13 25-step Turbo MLP null mean'),
    (I, '0.54', 2, lambda: cell(RC13, dict(channel='ZIT archive mlp'), 'p'), RC13, S_RC13, 'RC13 8-step Turbo MLP p'),
    (I, '999', 0, lambda: cell(RC14, dict(channel='ZIT archive mlp'), 'surrB_n_obs'), RC14, S_RC14, 'RC14 8-step Turbo MLP: Surrogate B reproductions of 1000'),
    (I, '140', 0, lambda: cell(RC14, dict(channel='MOCHI archive CFG4.5 attn'), 'surrB_n_obs'), RC14, S_RC14, 'RC14 Mochi: Surrogate B reproductions of 1000'),
    # round 3 review checks
    (I, '0.22', 2, lambda: cell(RC18, dict(a='ACEB_mlp', b='ACET_mlp', coord='log gap'), 'tangent'), RC18, S_RC18, 'RC18 ACE-Step MLP lineage, gap only'),
    (I, '0.34', 2, lambda: cell(RC18, dict(a='ACEB_mlp', b='ACET_mlp', coord='log gap'), 'perm_p95'), RC18, S_RC18, 'RC18 ACE-Step MLP lineage, gap-only null p95'),
    (I, '0.97', 2, lambda: cell(RC18, dict(a='ACEB_mlp', b='ACET_mlp', coord='H'), 'tangent'), RC18, S_RC18, 'RC18 ACE-Step MLP lineage, H only'),
    (I, '0.973', 3, lambda: csv(RC18).query("kind == 'same' and coord == 'log g'").tangent.min(), RC18, S_RC18, 'RC18 same weights, log g only, minimum'),
    (I, '0.975', 3, lambda: csv(RC18).query("kind == 'same' and coord == 'H'").tangent.min(), RC18, S_RC18, 'RC18 same weights, H only, minimum'),
    (I, '0.61', 2, lambda: csv(RC18).query("kind == 'lineage' and coord == 'log g'").tangent.min(), RC18, S_RC18, 'RC18 lineage, log g only, minimum'),
    (I, '0.91', 2, lambda: csv(RC18).query("kind == 'lineage' and coord == 'log g'").tangent.max(), RC18, S_RC18, 'RC18 lineage, log g only, maximum'),
    (I, '0.64', 2, lambda: csv(RC18).query("kind == 'lineage' and coord == 'H'").tangent.min(), RC18, S_RC18, 'RC18 lineage, H only, minimum'),
    (I, '0.968', 3, lambda: csv(RC19).query("kind == 'same'").r_H.min(), RC19, S_RC18, 'RC19 gate on H: same weights minimum'),
    (I, '0.788', 3, lambda: csv(RC19).query("kind == 'different'").r_H.max(), RC19, S_RC18, 'RC19 gate on H: different checkpoints maximum'),
    (I, '0.806', 3, lambda: csv(RC19).query("kind == 'same'").r_g.min(), RC19, S_RC18, 'RC19 gate on linear g: same weights minimum'),
    (I, '0.535', 3, lambda: csv(RC19).query("kind == 'different'").r_g.max(), RC19, S_RC18, 'RC19 gate on linear g: different maximum'),
    (I, '0.894', 3, lambda: csv(RC19).query("kind == 'different'").r_log_gap.max(), RC19, S_RC18, 'RC19 gate on gap: different maximum'),
    (II, '0.82', 2, lambda: csv(RC20)['rel_0.7'].corr(csv(RC20)['rel_0.5'], method='spearman'), RC20, S_RC20, 'RC20 onset order at threshold 0.7 vs 0.5'),
    (BOTH, '0.083', 3, lambda: cell(RC21, dict(res='512'), 'p05'), RC21, S_RC20, 'RC21 512 px pooling increment, bootstrap p5'),
    (BOTH, '0.093', 3, lambda: cell(RC21, dict(res='512'), 'p95'), RC21, S_RC20, 'RC21 512 px pooling increment, bootstrap p95'),
    (BOTH, '0.32', 2, lambda: cell(L6, dict(sub='mlp', a='Q35', b='Q25', drop_last=0), 'null_p95'), L6, 'review/r15_deep_dive/scripts/score_r15.py', 'R15-6 Qwen MLP null p95'),
    (II, '0.12', 2, lambda: -cell(D15, dict(source='ZIB', target='ZIT', submodule='attn', learner='histgb'), 'cross_minus_self'), D15, 'ledger D15 reproducer', 'D15 same-weights shortfall, minimum'),
    (II, '1.10', 2, lambda: -cell(D15, dict(source='Mochi45', target='ZIT', submodule='attn', learner='histgb'), 'cross_minus_self'), D15, 'ledger D15 reproducer', 'D15 largest unrelated shortfall'),
]


# --- Round 16 (a3b59c92) and Round 17 (16ccbe54), scored 2026-11-07/09 ---
T16 = R('ace_turbo_panel_r16', 'out', 'r16_tangents.csv'); V16 = R('ace_turbo_panel_r16', 'out', 'r16_verdicts.csv'); S_R16 = 'review/ace_turbo_panel_r16/scripts/score_r16.py'
T17 = R('tier_controls_r17', 'out', 'r17_tangents.csv'); V17 = R('tier_controls_r17', 'out', 'r17_verdicts.csv'); S_R17 = 'review/tier_controls_r17/scripts/score_r17.py'
P17 = R('tier_controls_r17', 'out', 'posthoc_r17_pixart_cosmos.csv')
def tt(path, a, b, s_, c='tangent'):
    d = csv(path).drop_duplicates(['a', 'b', 'sub']); d = d[(d.a == a) & (d.b == b) & (d['sub'] == s_)]; assert len(d) == 1, (a, b, s_); return float(d.iloc[0][c])
E += [
    (I, '0.795', 3, lambda: tt(T16, 'ACEB', 'T8', 'attn'), T16, S_R16, 'R16-2 ACE Base vs turbo 8 steps, attn'),
    (I, '0.594', 3, lambda: tt(T16, 'ACEB', 'T8', 'mlp'), T16, S_R16, 'R16-2 ACE Base vs turbo 8 steps, MLP'),
    (I, '0.337', 3, lambda: tt(T16, 'ACEB', 'T8', 'attn', 'null_p95'), T16, S_R16, 'R16-2 null p95 attn'),
    (I, '0.229', 3, lambda: tt(T16, 'ACEB', 'T8', 'mlp', 'null_p95'), T16, S_R16, 'R16-2 null p95 MLP'),
    (I, '0.918', 3, lambda: tt(T16, 'T8', 'ACET', 'attn'), T16, S_R16, 'R16-1 turbo 8 vs 6 steps, attn'),
    (I, '0.821', 3, lambda: tt(T16, 'T8', 'ACET', 'mlp'), T16, S_R16, 'R16-1 turbo 8 vs 6 steps, MLP'),
    (I, '0.906', 3, lambda: tt(T16, 'T3', 'T8', 'mlp'), T16, S_R16, 'R16-4 the failing cell, 3 vs 8 steps MLP'),
    (I, '0.692', 3, lambda: detail(V16, 'R16-8', r'attn ([0-9.]+)'), V16, S_R16, 'R16-8 prompt share attn'),
    (I, '0.593', 3, lambda: detail(V16, 'R16-8', r'mlp ([0-9.]+)'), V16, S_R16, 'R16-8 prompt share MLP'),
    (BOTH, '0.926', 3, lambda: tt(T17, 'SD3M', 'SD35M', 'attn'), T17, S_R17, 'R17-2 SD3 M vs SD3.5 M attn'),
    (I, '0.260', 3, lambda: tt(T17, 'SD3M', 'SD35M', 'attn', 'null_p95'), T17, S_R17, 'R17-2 null p95'),
    (BOTH, '0.980', 3, lambda: tt(T17, 'SD35M', 'SD3SV', 'attn'), T17, S_R17, 'R17-3 SD3.5 M recapture vs Round 11'),
    (BOTH, '0.303', 3, lambda: tt(T17, 'PIXART', 'COSMOS2B', 'attn'), T17, S_R17, 'R17-1 PixArt vs Cosmos attn'),
    (BOTH, '0.687', 3, lambda: tt(T17, 'PIXART', 'COSMOS2B', 'mlp'), T17, S_R17, 'R17-1 PixArt vs Cosmos MLP'),
    (I, '0.251', 3, lambda: tt(T17, 'PIXART', 'COSMOS2B', 'attn', 'null_p95'), T17, S_R17, 'R17-1 null p95 attn'),
    (I, '0.206', 3, lambda: tt(T17, 'PIXART', 'COSMOS2B', 'mlp', 'null_p95'), T17, S_R17, 'R17-1 null p95 MLP'),
    (I, '0.389', 3, lambda: cell(P17, dict(sub='mlp', test='drop last depth interval'), 'value'), P17, 'review/tier_controls_r17/scripts/posthoc_r17.py', 'R17 post hoc MLP without last interval'),
    (I, '0.940', 3, lambda: detail(V17, 'R17-0c', r'SD3 M vs SD3.5 M ([0-9.]+)'), V17, S_R17, 'R17-0c identity gate SD3 vs SD3.5 M'),
]


# --- review rounds 6-7 (2026-11-09): RC28 field-transfer null, RC30 10k permutations and native grid, RC31 collinearity, RC32 Mantel 10k ---
RC28 = os.path.join(RCD, 'rc28_finding12_null.csv'); RC28N = os.path.join(RCD, 'rc28_null_draws.csv'); S_RC28 = 'review/r15_deep_dive/review_checks/rc28_finding12_null.py'
RC30 = os.path.join(RCD, 'rc30_round7_tangent.csv'); S_RC30 = 'review/r15_deep_dive/review_checks/rc30_round7_tangent.py'
RC31 = os.path.join(RCD, 'rc31_collinearity.csv'); S_RC31 = 'review/r15_deep_dive/review_checks/rc31_collinearity.py'
RC32 = os.path.join(RCD, 'rc32_mantel_10k.csv'); S_RC32 = 'review/r15_deep_dive/review_checks/rc32_mantel_10k.py'
def r30(a, b, s_, c, rows='all'):
    d = csv(RC30); d = d[(d.a == a) & (d.b == b) & (d['sub'] == s_) & (d.rows == rows)]; assert len(d) == 1, (a, b, s_); return float(d.iloc[0][c])
E += [
    (II, '0.167', 3, lambda: csv(RC28N).groupby('sub').shortfall.quantile(0.95)['attn'], RC28N, S_RC28, 'RC28 pooled same-weights null p95, attention'),
    (II, '0.098', 3, lambda: csv(RC28N).groupby('sub').shortfall.quantile(0.95)['mlp'], RC28N, S_RC28, 'RC28 pooled same-weights null p95, MLP'),
    (II, '0.226', 3, lambda: cell(RC28, dict(source='ACET', target='ACEB', sub='attn'), 'sf05'), RC28, S_RC28, 'RC28 closest passing pair, 5th percentile'),
    (II, '0.255', 3, lambda: cell(RC28, dict(source='ACEB', target='ACET', sub='attn'), 'sf05'), RC28, S_RC28, 'RC28 ACE Base to turbo attn, 5th percentile'),
    (I, '0.0003', 4, lambda: r30('ACEB', 'ACET', 'mlp', 'p10k'), RC30, S_RC30, 'RC30 ACE-Step MLP lineage, 10k-permutation p'),
    (I, '0.0001', 4, lambda: r30('Z1', 'ZIT', 'attn', 'p10k'), RC30, S_RC30, 'RC30 Z-Image lineage attn, 10k-permutation p'),
    (I, '0.55', 2, lambda: r30('Z1', 'ZIT', 'attn', 'native'), RC30, S_RC30, 'RC30 Z-Image lineage attn on the native 30-point grid'),
    (I, '0.77', 2, lambda: r30('Z1', 'ZIT', 'mlp', 'native'), RC30, S_RC30, 'RC30 Z-Image lineage MLP on the native grid'),
    (I, '0.26', 2, lambda: r30('PIXART', 'COSMOS2B', 'attn', 'p95_native'), RC30, S_RC30, 'RC30 PixArt-Cosmos native-grid null p95'),
    (I, '0.20', 2, lambda: r30('PIXART', 'COSMOS2B', 'attn', 'native'), RC30, S_RC30, 'RC30 PixArt-Cosmos attn on the native 28-point grid'),
    (II, '0.926', 3, lambda: csv(RC31).query("model == 'SD35L' and sub == 'attn'").r_logg_H.iloc[0], RC31, S_RC31, 'RC31 lowest archive log g-H Spearman (SD3.5 L)'),
    (II, '0.087', 3, lambda: csv(RC31).query("model == 'ACET' and sub == 'mlp'").r_logg_loggap.iloc[0], RC31, S_RC31, 'RC31 ACE-Step turbo MLP log g-gap Spearman'),
    (I, '0.0002', 4, lambda: csv(RC32).query("kind == 'unrelated'").p10k.max(), RC32, S_RC32, 'RC32 largest unrelated Mantel p, 10k permutations'),
]

# --- 2026-11-09 after round 7: RC33 native grid with its own nulls (native grid leads in both papers) ---
RC33 = os.path.join(RCD, 'rc33_native_grid.csv'); S_RC33 = 'review/r15_deep_dive/review_checks/rc33_native_grid.py'
def r33(a, b, s_, c, rows='all'):
    d = csv(RC33); d = d[(d.a == a) & (d.b == b) & (d['sub'] == s_) & (d.rows == rows)]; assert len(d) == 1, (a, b, s_); return float(d.iloc[0][c])
E += [
    (BOTH, '0.547', 3, lambda: r33('Z1', 'ZIT', 'attn', 'native'), RC33, S_RC33, 'RC33 Z-Image sibling pair attn, native grid'),
    (BOTH, '0.771', 3, lambda: r33('Z1', 'ZIT', 'mlp', 'native'), RC33, S_RC33, 'RC33 Z-Image sibling pair MLP, native grid'),
    (I, '0.295', 3, lambda: r33('Z1', 'ZIT', 'attn', 'perm95'), RC33, S_RC33, 'RC33 Z-Image attn native permutation p95'),
    (I, '0.285', 3, lambda: r33('Z1', 'ZIT', 'attn', 'surr95'), RC33, S_RC33, 'RC33 Z-Image attn native surrogate p95'),
    (I, '0.0003', 4, lambda: r33('Z1', 'ZIT', 'attn', 'p10k'), RC33, S_RC33, 'RC33 Z-Image attn native 10k-permutation p'),
    (I, '0.559', 3, lambda: r33('Z2', 'ZIT', 'attn', 'native'), RC33, S_RC33, 'RC33 Z2-ZIT attn, native grid'),
    (I, '0.760', 3, lambda: r33('Z1', 'Z3', 'mlp', 'native'), RC33, S_RC33, 'RC33 Z1-Z3 MLP, native grid'),
    (I, '0.912', 3, lambda: r33('Z2', 'ZITSV', 'attn', 'native', 'image'), RC33, S_RC33, 'RC33 Z-Image image rows attn, native grid'),
    (I, '0.737', 3, lambda: r33('Z2', 'ZITSV', 'mlp', 'native', 'image'), RC33, S_RC33, 'RC33 Z-Image image rows MLP, native grid'),
    (I, '0.196', 3, lambda: r33('PIXART', 'COSMOS2B', 'attn', 'native'), RC33, S_RC33, 'RC33 PixArt-Cosmos attn, native grid'),
    (I, '0.255', 3, lambda: r33('PIXART', 'COSMOS2B', 'attn', 'perm95'), RC33, S_RC33, 'RC33 PixArt-Cosmos attn native permutation p95'),
    (BOTH, '0.57', 2, lambda: r33('PIXART', 'COSMOS2B', 'mlp', 'native'), RC33, S_RC33, 'RC33 PixArt-Cosmos MLP, native grid'),
    (I, '0.12', 2, lambda: r33('PIXART', 'COSMOS2B', 'attn', 'p10k'), RC33, S_RC33, 'RC33 PixArt-Cosmos attn native 10k-permutation p'),

]
# --- 2026-11-11: Round 18 (preregistered), addendum B (CKA) and the post hoc sweep RC34-RC49 ---
R18O = R('r18', 'out'); T18 = os.path.join(R18O, 'r18_tangents.csv'); N18 = os.path.join(R18O, 'r18_native_grid.csv')
S_R18 = 'review/r18/scripts/score_r18.py'; S_R18N = 'review/r18/scripts/r18_native.py'; S_R18X = 'review/r18/scripts/r18_extra.py'
CK = R('r18', 'cka_out', 'cka_r18b_pairs.csv'); S_CK = 'capture_code_r4/tier_controls_r18/cka_r18b.py (frozen copy review/r18/cka_r18b_frozen.py)'
def t18(a, b, s_, c, rows='all'):
    d = csv(T18); d = d[(d.a == a) & (d.b == b) & (d['sub'] == s_) & (d.rows == rows)]; assert len(d) == 1, (a, b, s_, rows); return float(d.iloc[0][c])
def n18(a, b, s_, c, rows='all'):
    d = csv(N18); d = d[(d.a == a) & (d.b == b) & (d['sub'] == s_) & (d.rows == rows)]; assert len(d) == 1, (a, b, s_, rows); return float(d.iloc[0][c])
def ck(a, b, s_, c='value'):
    d = csv(CK); d = d[(d.a == a) & (d.b == b) & (d['sub'] == s_) & d.primary]; assert len(d) == 1, (a, b, s_); return float(d.iloc[0][c])
E += [
    (BOTH, '0.818', 3, lambda: t18('SD35L18', 'SD35LT', 'attn', 'tangent'), T18, S_R18, 'R18-1 SD3.5 L <-> L Turbo attn, 24-point grid'),
    (I, '0.301', 3, lambda: t18('SD35L18', 'SD35LT', 'attn', 'null_p95'), T18, S_R18, 'R18-1 permutation p95, 24-point grid'),
    (BOTH, '0.904', 3, lambda: n18('SD35L18', 'SD35LT', 'attn', 'native'), N18, S_R18N, 'R18 SD3.5 L <-> L Turbo attn, native grid'),
    (I, '0.326', 3, lambda: n18('SD35L18', 'SD35LT', 'attn', 'perm95'), N18, S_R18N, 'R18 SD3.5 L <-> L Turbo native permutation p95'),
    (BOTH, '0.998', 3, lambda: n18('SD35L18', 'SD35L', 'attn', 'native'), N18, S_R18N, 'R18 SD3.5 L across sessions, native grid'),
    (I, '0.994', 3, lambda: t18('SD35L18', 'SD35L', 'attn', 'tangent'), T18, S_R18, 'R18-2 SD3.5 L across sessions, 24-point grid'),
    (BOTH, '0.520', 3, lambda: n18('ZB6', 'ZIT', 'attn', 'native'), N18, S_R18N, 'R18 Base shift 6 <-> Turbo attn, native grid'),
    (BOTH, '0.760', 3, lambda: n18('ZB6', 'ZIT', 'mlp', 'native'), N18, S_R18N, 'R18 Base shift 6 <-> Turbo MLP, native grid'),
    (I, '0.649', 3, lambda: t18('ZB6', 'ZIT', 'attn', 'tangent'), T18, S_R18, 'R18-4 Base shift 6 <-> Turbo attn, 24-point grid'),
    (I, '0.682', 3, lambda: t18('ZB6', 'ZIT', 'mlp', 'tangent'), T18, S_R18, 'R18-4 Base shift 6 <-> Turbo MLP, 24-point grid'),
    (I, '0.928', 3, lambda: t18('Z2', 'ZIT2', 'attn', 'tangent', 'image'), T18, S_R18, 'R18-3b Z2 <-> ZIT2 image rows attn, 24-point grid'),
    (I, '0.754', 3, lambda: t18('Z2', 'ZIT2', 'mlp', 'tangent', 'image'), T18, S_R18, 'R18-3b Z2 <-> ZIT2 image rows MLP, 24-point grid'),
    (I, '0.612', 3, lambda: ck('R18S_SD35M', 'R18S_SD35M1', 'attn'), CK, S_CK, 'Addendum B CKA, SD3.5 M CFG 4.5 vs 1'),
    (I, '0.625', 3, lambda: ck('R18_SD35L', 'R18_SD35LT', 'attn'), CK, S_CK, 'Addendum B CKA, SD3.5 L <-> L Turbo'),
    (I, '0.462', 3, lambda: ck('R18S_SD3M', 'R18S_SD35M', 'attn'), CK, S_CK, 'Addendum B CKA, SD3 M <-> SD3.5 M'),
    (BOTH, '0.344', 3, lambda: ck('R18_ZB6', 'R18_ZIT2', 'attn'), CK, S_CK, 'Addendum B CKA, Z-Image Base <-> Turbo attn'),
    (I, '0.338', 3, lambda: ck('R18S_PIXART', 'R18S_COSMOS2B', 'attn'), CK, S_CK, 'Addendum B CKA, PixArt <-> Cosmos attn'),
    (BOTH, '0.364', 3, lambda: max(ck(a, b, 'attn') for a, b in csv(CK)[(csv(CK).kind == 'unrelated') & csv(CK).primary & (csv(CK)['sub'] == 'attn')][['a', 'b']].values), CK, S_CK, 'Addendum B CKA, highest unrelated attn'),
    (BOTH, '1.004', 3, lambda: cell(os.path.join(R18O, 'r18_trace_ratio.csv'), dict(arm='R18_SD35L', sub='attn'), 'median_ratio'), os.path.join(R18O, 'r18_trace_ratio.csv'), S_R18X, 'Addendum A trace ratio median, SD3.5 L attn'),
    (BOTH, '1.021', 3, lambda: cell(os.path.join(R18O, 'r18_trace_ratio.csv'), dict(arm='R18_ZB6', sub='attn'), 'median_ratio'), os.path.join(R18O, 'r18_trace_ratio.csv'), S_R18X, 'Addendum A trace ratio median, Base shift 6 attn'),
    (BOTH, '1.013', 3, lambda: cell(os.path.join(R18O, 'r18_trace_ratio.csv'), dict(arm='R18_ZB6', sub='mlp'), 'median_ratio'), os.path.join(R18O, 'r18_trace_ratio.csv'), S_R18X, 'Addendum A trace ratio median, Base shift 6 MLP'),
]

# --- 2026-11-11: round-8 review checks RC53-RC55 (post hoc) ---
RC53F = os.path.join(RCD, 'rc53_fpr_bounds.csv'); RC54 = os.path.join(RCD, 'rc54_tangent_bands.csv'); RC55 = os.path.join(RCD, 'rc55_mlp_variants.csv')
S_RC53 = 'review/r15_deep_dive/review_checks/rc53_fpr_inventory.py'; S_RC54 = 'review/r15_deep_dive/review_checks/rc54_rc55_round8.py'
fb = lambda unit: cell(RC53F, dict(unit=unit, criterion='rule_native'), 'ub95_if_zero')
b54 = lambda a, b, s_, c: cell(RC54, dict(a=a, b=b, sub=s_), c)
m55 = lambda a, b, v, t, c='tangent': cell(RC55, dict(a=a, b=b, variant=v, end_trim=t), c)
E += [
    (I, '0.78', 2, lambda: fb('pair types (families)'), RC53F, S_RC53, 'RC53 FPR upper bound, two pair types'),
    (I, '0.45', 2, lambda: fb('distinct model pairs'), RC53F, S_RC53, 'RC53 FPR upper bound, five model pairs'),
    (I, '0.26', 2, lambda: fb('capture pair-directions'), RC53F, S_RC53, 'RC53 FPR upper bound, ten pair-directions'),
    (I, '0.655', 3, lambda: b54('Z1', 'ZIT', 'attn', 'low'), RC54, S_RC54, 'RC54 Z-Image siblings, low band'),
    (I, '0.651', 3, lambda: b54('Z1', 'ZIT', 'attn', 'low_p95'), RC54, S_RC54, 'RC54 Z-Image siblings, low-band null'),
    (I, '0.56', 2, lambda: b54('PIXART', 'COSMOS2B', 'attn', 'grid12'), RC54, S_RC54, 'RC54 PixArt-Cosmos, 12-point band'),
    (I, '0.523', 3, lambda: m55('Z2', 'ZITSV', 'centred', False), RC55, S_RC54, 'RC55 Z-Image siblings centred MLP'),
    (I, '0.166', 3, lambda: m55('Z2', 'ZITSV', 'centred', False, 'perm95'), RC55, S_RC54, 'RC55 siblings centred MLP null'),
    (I, '0.267', 3, lambda: m55('PIXART', 'COSMOS2B', 'centred', False), RC55, S_RC54, 'RC55 PixArt-Cosmos centred MLP'),
    (I, '0.757', 3, lambda: m55('PIXART', 'COSMOS2B', 'top8 removed', False), RC55, S_RC54, 'RC55 PixArt-Cosmos MLP, top 8 removed'),
]

# --- 2026-11-11: round-9 review check RC56 (post hoc): 12-point reconstruction ---
RC56 = os.path.join(RCD, 'rc56_grid12_reconstruction.csv'); S_RC56 = 'review/r15_deep_dive/review_checks/rc56_grid12_reconstruction.py'
_u56 = lambda: csv(RC56)[csv(RC56).kind.str.startswith('unrelated') & (csv(RC56)['sub'] == 'attn')]
E += [
    (BOTH, '0.96', 2, lambda: float(csv(RC56).t12.corr(csv(RC56).t12_lp)), RC56, S_RC56, 'RC56 corr 12-pt vs ideal low-pass tangent'),
    (BOTH, '0.17', 2, lambda: float((csv(RC56).t12 - csv(RC56).t12_lp).abs().max()), RC56, S_RC56, 'RC56 largest change from folding'),
    (BOTH, '0.49', 2, lambda: float(_u56().t12_lp.min()), RC56, S_RC56, 'RC56 unrelated attn, ideal low-pass, min'),
    (BOTH, '0.65', 2, lambda: float(_u56().t12_lp.max()), RC56, S_RC56, 'RC56 unrelated attn, ideal low-pass, max'),
    (BOTH, '35', 0, lambda: 100 * float(csv(RC56).alias_share.max()), RC56, S_RC56, 'RC56 largest folded energy share (%)'),
    (I, '0.67', 2, lambda: cell(RC56, dict(a='ACEB', b='T8', sub='attn'), 't12_lp'), RC56, S_RC56, 'RC56 ACE distillation attn, ideal low-pass'),
    (I, '0.51', 2, lambda: cell(RC56, dict(a='ACEB', b='T8', sub='attn'), 't12'), RC56, S_RC56, 'RC56 ACE distillation attn, 12-pt'),
]


def fmt(v, nd, printed):
    s = f'{v:.{nd}f}' if nd else f'{round(v):d}'
    if printed.startswith('+') and v >= 0: s = '+' + s
    return s


def main():
    def _paper(p):  # main text plus its supplementary material (input at the end of main.tex)
        d = os.path.join(ROOT, 'papers', f'mode-{p}')
        if not os.path.exists(os.path.join(d, 'main.tex')): return None
        return ''.join(open(os.path.join(d, f), encoding='utf-8').read() for f in ('main.tex', 'supplement.tex') if os.path.exists(os.path.join(d, f)))
    tex = {p: _paper(p) for p in ('I', 'II')}
    rows, bad = [], 0
    for papers, printed, nd, fn, src, script, what in E:
        try:
            v = fn(); got = fmt(v, nd, printed); ok_v = got == printed
        except Exception as e:  # noqa: BLE001
            v, got, ok_v = float('nan'), f'error: {e}', False
        tex_ok = []
        for p in papers:
            if tex[p] is None: continue
            needle = printed.replace('-', '$-') if printed.startswith('-') else printed
            tex_ok.append(printed in tex[p] or needle in tex[p] or printed.lstrip('+') in tex[p])
        ok = ok_v and all(tex_ok)
        bad += not ok
        rows.append((what, printed, '/'.join(papers), os.path.relpath(src, ROOT), script, 'ok' if ok else f'FAIL (source gives {got}; in tex: {tex_ok})'))
    with open(os.path.join(ROOT, 'NUMBERS.md'), 'w', encoding='utf-8') as f:
        f.write('# Round 11-18 numbers and where they come from\n\nGenerated by `check_numbers.py`; every row is re-read from its source and compared with the value printed in the paper(s).\n'
                'All other numbers are checked by the figure scripts (`ledger_has` assertions, `papers/<paper>/data/*.csv`) and by each review round\'s RESULTS file.\n\n'
                '| Number | Printed | Paper | Source | Script | Check |\n|---|---|---|---|---|---|\n')
        for r in rows: f.write('| ' + ' | '.join(r) + ' |\n')
    print(f'{len(rows) - bad}/{len(rows)} checks pass')
    for r in rows:
        if r[-1] != 'ok': print('  ', r)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
