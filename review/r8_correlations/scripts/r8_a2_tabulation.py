"""R8-2 tabulation: Spearman rho of step count against relative onset (round6 S3 estimators), with and without the ZIT
channels. Run after r8_a_grid.py; it adds the key 'R8-2_tabulation_steps_vs_rel_onset' to out/R8_2_8_results.json.
(This step was first run inline on 2026-10-09 at 08:44Z; the code is unchanged.)"""
import os
import pandas as pd, json
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', 'out')
S3 = pd.read_csv(os.path.join(HERE, '..', '..', 'round6', 'out', 'S3_estimators.csv'))
out = {}
for col in ['matched60_rel_median', 'native_rel_median']:
    a = spearmanr(S3.steps, S3[col]).statistic; k = ~S3.channel.str.startswith('ZIT'); b = spearmanr(S3.steps[k], S3[col][k]).statistic
    out[col] = dict(rho_7=float(a), rho_without_ZIT=float(b))
print(out)
r = json.load(open(f'{OUT}/R8_2_8_results.json')); r['R8-2_tabulation_steps_vs_rel_onset'] = out
json.dump(r, open(f'{OUT}/R8_2_8_results.json', 'w'), indent=1)
