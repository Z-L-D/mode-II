# RC53 (post hoc, round-8 reviews 2026-11-11: DeepSeek 1 and Q1, GLM M1 and Q1, ChatGPT 1.2). Inventory of every
# depth-matched unrelated pair with an attention tangent on file (RC22 24-point grid, RC30/RC33 native grid), how many
# of them are independent, and the exact one-sided 95% upper bound on the false-positive rate of the screen given zero
# false positives among k independent unrelated pairs (Clopper-Pearson: 1 - 0.05**(1/k)).
# A "false positive" is counted two ways: (a) above the pair's own layer-permutation p95 (the weak criterion),
# (b) at or above p95 + 0.30 (the lineage rule). Usage: python rc53_fpr_inventory.py <review_checks/out> <outdir>
import sys, pandas as pd
IN, OUT = sys.argv[1:3]
a = pd.read_csv(f'{IN}/rc30_round7_tangent.csv'); s = pd.read_csv(f'{IN}/rc22_round5_tangent_checks.csv')
u = a[a.kind.str.startswith('unrelated') & (a['sub'] == 'attn') & (a.rows == 'all')].copy()
u = u.merge(s[['a', 'b', 'sub', 'rows', 'perm95_full', 'surr95', 'ci05', 'ci95']], on=['a', 'b', 'sub', 'rows'], how='left')
model = {'ACEB': 'ACE-Step Base', 'ACET': 'ACE-Step turbo', 'T8': 'ACE-Step turbo', 'SD35M17': 'SD3.5 Medium', 'SD3SV': 'SD3.5 Medium',
         'SD3M': 'SD3 Medium', 'PIXART': 'PixArt-Sigma', 'COSMOS2B': 'Cosmos-Predict2 2B'}
u['model_pair'] = [f'{model[x]} / {model[y]}' for x, y in zip(u.a, u.b)]
u['above_perm95_24'] = u.full24 > u.perm95_full; u['rule_24'] = u.full24 >= u.perm95_full + 0.30
u['above_perm95_native'] = u.native > u.p95_native; u['rule_native'] = u.native >= u.p95_native + 0.30
u.to_csv(f'{OUT}/rc53_unrelated_inventory.csv', index=False)
fam = {'PixArt-Sigma / Cosmos-Predict2 2B': 'image-image', }
k_dirs = len(u); k_models = u.model_pair.nunique(); k_types = 2
rows = []
for label, k in (('pair types (families)', k_types), ('distinct model pairs', k_models), ('capture pair-directions', k_dirs)):
    for crit in ('above_perm95_24', 'rule_24', 'above_perm95_native', 'rule_native'):
        fp = int(u.groupby('model_pair')[crit].any().sum()) if label == 'distinct model pairs' else (int(u[crit].sum()) if label == 'capture pair-directions' else int(u.groupby(u.a.isin(['PIXART']))[crit].any().sum()))
        rows.append(dict(unit=label, k=k, criterion=crit, false_pos=fp, ub95_if_zero=1 - 0.05 ** (1 / k)))
F = pd.DataFrame(rows); F.to_csv(f'{OUT}/rc53_fpr_bounds.csv', index=False)
pd.set_option('display.width', 250); print(u[['a', 'b', 'model_pair', 'full24', 'perm95_full', 'surr95', 'native', 'p95_native', 'above_perm95_24', 'rule_24', 'above_perm95_native', 'rule_native']].round(3).to_string(index=False)); print(F.round(3).to_string(index=False))
