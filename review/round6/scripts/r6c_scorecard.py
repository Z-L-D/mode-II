# Collects the frozen S1-S5 verdicts into out/R6C_scorecard.csv
from r6c_common import *
rows = []
for f in ['S1', 'S2', 'S3', 'S4', 'S5']:
    for k, (v, why) in json.load(open(f'{OUT6}/{f}_results.json'))['score'].items(): rows.append(dict(id=k, verdict=v, result=why))
S = pd.DataFrame(rows); S.to_csv(f'{OUT6}/R6C_scorecard.csv', index=False); print(S.verdict.value_counts().to_dict(), len(S))
