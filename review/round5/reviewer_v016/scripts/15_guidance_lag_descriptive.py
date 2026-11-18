#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
OUT=Path("/mnt/data/r4_r5_audit_v016/results")
d=pd.read_csv(OUT/"BRANCH_TRIANGULATION_BY_STEP.csv")
D=1-d.prompt_influence.to_numpy()
C=1-d.seed_spread_ratio.to_numpy()
rows=[]
# positive lag means branch divergence at t compared to contraction at t+lag
for lag in range(-15,16):
    if lag>=0:
        x=D[:len(D)-lag] if lag else D
        y=C[lag:]
    else:
        x=D[-lag:]
        y=C[:len(C)+lag]
    r,p=spearmanr(x,y)
    rows.append([lag,len(x),r,p])
q=pd.DataFrame(rows,columns=["lag_steps","n","spearman_divergence_vs_future_contraction","p_naive"])
q.to_csv(OUT/"POSTHOC_GUIDANCE_LAG_CORRELATION.csv",index=False)
best=q.loc[q.spearman_divergence_vs_future_contraction.idxmax()]
worst=q.loc[q.spearman_divergence_vs_future_contraction.idxmin()]
print("best positive association:",best.to_dict())
print("most negative:",worst.to_dict())
print(q.to_string(index=False))
