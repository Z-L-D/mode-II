#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from r4_common import *
OUT=Path("/mnt/data/r4_r5_audit_v016/results");OUT.mkdir(parents=True,exist_ok=True)

z1,_=load_member("ZIB-attn-TE1-CFG1.0-R1024-S1001-STUDY[CFG-RES-SEED-CORRECTION]")
z4,_=load_member("ZIB-attn-TE1-CFG4.0-R1024-S1001-STUDY[CFG-RES-SEED-CORRECTION]")
rows=[]
for sub in ["attn","mlp"]:
    A=z1[(z1.branch=="cond")&(z1.submodule==sub)]
    B=z4[(z4.branch=="cond")&(z4.submodule==sub)]
    shared=sorted(set(A.prompt_hash)&set(B.prompt_hash))
    for t in sorted(set(A.token_idx)&set(B.token_idx)):
        a=A[(A.token_idx==t)&A.prompt_hash.isin(shared)].groupby(["prompt_hash","layer"])[["g_micro","spectral_gap","sv_entropy"]].mean().reset_index()
        b=B[(B.token_idx==t)&B.prompt_hash.isin(shared)].groupby(["prompt_hash","layer"])[["g_micro","spectral_gap","sv_entropy"]].mean().reset_index()
        m=a.merge(b,on=["prompt_hash","layer"],suffixes=("_1","_4"))
        lg=np.median(np.abs(np.log(m.g_micro_4/m.g_micro_1)))
        # normalized population depth shape
        aa=a.groupby("layer")[["g_micro","spectral_gap","sv_entropy"]].mean().to_numpy(float);bb=b.groupby("layer")[["g_micro","spectral_gap","sv_entropy"]].mean().to_numpy(float)
        aa[:,1]*=-1;bb[:,1]*=-1;aa=(aa-aa.mean(0))/(aa.std(0)+1e-12);bb=(bb-bb.mean(0))/(bb.std(0)+1e-12)
        r=np.corrcoef(aa.ravel(),bb.ravel())[0,1]
        rows.append([sub,t,float(A[A.token_idx==t].sigma.mean()),lg,r,len(shared)])
df=pd.DataFrame(rows,columns=["submodule","step","sigma","median_abs_log_g_cfg4cond_vs_cfg1","shape_r_cfg4cond_vs_cfg1","n_prompts"])
df.to_csv(OUT/"POSTHOC_ZIB_cfg1_cfg4_cond.csv",index=False)
summ=df.groupby("submodule",as_index=False).agg(mean_shape_r=("shape_r_cfg4cond_vs_cfg1","mean"),min_shape_r=("shape_r_cfg4cond_vs_cfg1","min"),median_abs_log_g=("median_abs_log_g_cfg4cond_vs_cfg1","median"),max_abs_log_g=("median_abs_log_g_cfg4cond_vs_cfg1","max"),n_prompts=("n_prompts","max"))
summ.to_csv(OUT/"POSTHOC_ZIB_cfg1_cfg4_cond_summary.csv",index=False)

# cond/uncond on guided ZIB
bi=[]
for sub in ["attn","mlp"]:
    q=z4[z4.submodule==sub]
    for (p,t),g in q.groupby(["prompt_hash","token_idx"]):
        c=g[g.branch=="cond"].sort_values("layer");u=g[g.branch=="uncond"].sort_values("layer")
        if len(c)!=30 or len(u)!=30:continue
        C=np.c_[c.g_micro,-c.spectral_gap,c.sv_entropy];U=np.c_[u.g_micro,-u.spectral_gap,u.sv_entropy]
        C=(C-C.mean(0))/(C.std(0)+1e-12);U=(U-U.mean(0))/(U.std(0)+1e-12)
        bi.append([sub,p,t,float(c.sigma.mean()),np.corrcoef(C.ravel(),U.ravel())[0,1],float(np.log(c.g_micro.mean()/u.g_micro.mean()))])
B=pd.DataFrame(bi,columns=["submodule","prompt_hash","step","sigma","shape_r_cond_uncond","log_g_cond_over_uncond"])
B.to_csv(OUT/"POSTHOC_ZIB_cond_uncond.csv",index=False)
bs=B.groupby("submodule",as_index=False).agg(mean_shape_r=("shape_r_cond_uncond","mean"),early_shape_r=("shape_r_cond_uncond",lambda x: float(x.iloc[:max(1,len(x)//4)].mean())),median_abs_log_g=("log_g_cond_over_uncond",lambda x: float(np.median(np.abs(x)))))
bs.to_csv(OUT/"POSTHOC_ZIB_cond_uncond_summary.csv",index=False)
print("CFG1 vs CFG4 cond:\n",summ.to_string(index=False))
print("\ncond vs uncond:\n",bs.to_string(index=False))
