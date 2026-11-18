# RC15 (post hoc, round-2 review GLM M1): the Round 12 lineage criteria R12-1..R12-3 re-scored on every row set of
# the Round 11 row-split captures (values from out/rc8_rowset_tiers.csv). R12-1: lineage above the permutation p95.
# R12-2: lineage at least 0.10 below the lower same-weights split-half value. R12-3: lineage at least 0.30 above the
# larger unrelated value (Turbo or Base against SD3.5 M; attention only, since SD3.5 M has no MLP capture here).
import os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = pd.read_csv(f'{HERE}/out/rc8_rowset_tiers.csv')
R['same_min'] = R[['same_weights_turbo_splithalf', 'same_weights_base_splithalf']].min(1)
R['unrel_max'] = R[['unrelated_turbo_sd35m', 'unrelated_base_sd35m']].max(1)
R['R12_1'] = R.lineage_base_turbo > R.perm_p95
R['gap_same'] = R.same_min - R.lineage_base_turbo; R['R12_2'] = R.gap_same >= 0.10
R['gap_unrel'] = R.lineage_base_turbo - R.unrel_max; R['R12_3'] = (R.gap_unrel >= 0.30).where(R.unrel_max.notna())
O = R[['sub', 'rows', 'same_min', 'lineage_base_turbo', 'unrel_max', 'perm_p95', 'R12_1', 'gap_same', 'R12_2', 'gap_unrel', 'R12_3']]
O.to_csv(f'{HERE}/out/rc15_rowset_criteria.csv', index=False); print(O.round(3).to_string())
