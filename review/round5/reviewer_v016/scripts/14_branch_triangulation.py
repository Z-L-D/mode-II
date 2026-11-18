#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr,pearsonr
sys.path.insert(0,str(Path(__file__).parent))
from r4_common import *
OUT=Path("/mnt/data/r4_r5_audit_v016/results");OUT.mkdir(parents=True,exist_ok=True)

c45,_=load_all(["SD3.5M-attn-TE3-CFG4.5-R1024-S1001-STUDY[CFG-RES-SEED-CORRECTION]",
                "SD3.5M-attn-TE3-CFG4.5-R1024-S1002-STUDY[CFG-RES-SEED-CORRECTION]"])
spread=pd.read_csv(OUT/"R1f_cfg_spread_ratio_cells.csv")
# step-level branch metrics
rows=[]
for t,g in c45.groupby("token_idx"):
    pis=g[g.branch=="cond"].prompt_influence.dropna().to_numpy(float)
    shape=[];dg=[];dgap=[];dh=[]
    for (p,sd),h in g.groupby(["prompt_hash","seed"]):
        c=h[h.branch=="cond"].sort_values("layer");u=h[h.branch=="uncond"].sort_values("layer")
        if len(c)!=24 or len(u)!=24:continue
        C=np.c_[c.g_micro,-c.spectral_gap,c.sv_entropy];U=np.c_[u.g_micro,-u.spectral_gap,u.sv_entropy]
        Cn=(C-C.mean(0))/(C.std(0)+1e-12);Un=(U-U.mean(0))/(U.std(0)+1e-12)
        shape.append(np.corrcoef(Cn.ravel(),Un.ravel())[0,1])
        dg.append(abs(np.log(c.g_micro.mean()/u.g_micro.mean())))
        dgap.append(abs(np.log(c.spectral_gap.mean()/u.spectral_gap.mean())))
        dh.append(abs(c.sv_entropy.mean()-u.sv_entropy.mean()))
    rr=spread[spread.step==t].sd_ratio_cfg45_over_cfg1
    rows.append([t,float(g.sigma.mean()),float(np.median(pis)),float(np.mean(shape)),float(np.median(dg)),
                 float(np.median(dgap)),float(np.median(dh)),float(np.nanmedian(rr))])
D=pd.DataFrame(rows,columns=["step","sigma","prompt_influence","shape_r_cond_uncond","abs_log_g_diff","abs_log_gap_diff","abs_H_diff","seed_spread_ratio"])
D.to_csv(OUT/"BRANCH_TRIANGULATION_BY_STEP.csv",index=False)
corr=[]
for y in ["shape_r_cond_uncond","abs_log_g_diff","abs_log_gap_diff","abs_H_diff","seed_spread_ratio"]:
    x=1-D.prompt_influence
    corr.append([y,float(spearmanr(x,D[y]).statistic),float(spearmanr(x,D[y]).pvalue),float(pearsonr(x,D[y]).statistic),float(pearsonr(x,D[y]).pvalue)])
C=pd.DataFrame(corr,columns=["observable","spearman_with_1_minus_prompt_influence","spearman_p","pearson","pearson_p"])
C.to_csv(OUT/"BRANCH_TRIANGULATION_CORRELATIONS.csv",index=False)
print(D.iloc[[0,5,10,20,30,39]].to_string(index=False))
print("\n",C.to_string(index=False))
