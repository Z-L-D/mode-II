#!/usr/bin/env python3
from __future__ import annotations
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from r4_common import *

OUT=Path("/mnt/data/r4_r5_audit_v016/results");OUT.mkdir(parents=True,exist_ok=True)

# Core files
c1,n1=load_all(["SD3.5M-attn-TE3-CFG1.0-R1024-S1001-STUDY[CFG-RES-SEED-CORRECTION]",
                "SD3.5M-attn-TE3-CFG1.0-R1024-S1002-STUDY[CFG-RES-SEED-CORRECTION]"])
c45,n45=load_all(["SD3.5M-attn-TE3-CFG4.5-R1024-S1001-STUDY[CFG-RES-SEED-CORRECTION]",
                  "SD3.5M-attn-TE3-CFG4.5-R1024-S1002-STUDY[CFG-RES-SEED-CORRECTION]"])

R={}

# R1a preflight
def preflight(d,cfg):
    expected_br={"cond"} if cfg==1.0 else {"cond","uncond"}
    branch=set(d.branch.dropna().astype(str))
    hook=set(d.hook_mode.dropna().astype(str))
    finite_pi = True if cfg==1.0 else bool(np.isfinite(pd.to_numeric(d.prompt_influence,errors="coerce")).all())
    key=["prompt_hash","seed","token_idx","branch"]
    nl=d.groupby(key).layer.nunique()
    return dict(cfg=cfg,hook_modes=sorted(hook),branches=sorted(branch),
                finite_prompt_influence=finite_pi,layer_min=int(nl.min()),layer_max=int(nl.max()),
                n_prompts=int(d.prompt_hash.nunique()),n_seeds=int(d.seed.nunique()),
                max_prompts_per_seed=int(d[["prompt_hash","seed"]].drop_duplicates().groupby("seed").prompt_hash.nunique().max()),
                pass_preflight=(hook=={"passive-branch"} and branch==expected_br and finite_pi and nl.min()==24 and nl.max()==24))
R["R1a_CFG1"]=preflight(c1,1.0);R["R1a_CFG45"]=preflight(c45,4.5)

# Seed design
ps=pd.concat([c1[["prompt_hash","seed"]],c45[["prompt_hash","seed"]]]).drop_duplicates()
prompts_per_seed=ps.groupby("seed").prompt_hash.nunique()
R["seed_design"]=dict(unique_seed_design=bool((prompts_per_seed<=1).all()),
                      n_unique_seeds=int(ps.seed.nunique()),
                      max_prompts_per_seed=int(prompts_per_seed.max()),
                      formal_R1c_R1d_R1e="NOT SCORABLE" if not (prompts_per_seed<=1).all() else "SCORABLE")

# R1b paired cond equality step 0. Pair by prompt/seed/layer.
keys=["prompt_hash","seed","token_idx","layer","submodule"]
a=c1[(c1.branch=="cond")&(c1.token_idx==0)][keys+["g_micro"]].rename(columns={"g_micro":"g1"})
b=c45[(c45.branch=="cond")&(c45.token_idx==0)][keys+["g_micro"]].rename(columns={"g_micro":"g45"})
m=a.merge(b,on=keys,how="inner")
m["abs_log_ratio"]=np.abs(np.log(m.g45/m.g1))
R["R1b"]=dict(n_cells=len(m),median_abs_log_ratio=float(m.abs_log_ratio.median()),
              q95_abs_log_ratio=float(m.abs_log_ratio.quantile(.95)),
              max_abs_log_ratio=float(m.abs_log_ratio.max()),
              verdict="CONFIRMED" if m.abs_log_ratio.median()<=1e-3 else "REFUTED")
m.to_csv(OUT/"R1b_cond_cfg45_vs_cfg1_step0.csv",index=False)

# Helper layermean logg per prompt/seed/step on cond branch.
def layermean(d,var="g_micro",log=True,branch="cond"):
    q=d[d.branch.eq(branch)].copy()
    x=q.groupby(["prompt_hash","seed","token_idx"])[var].mean().reset_index()
    if log:x[var]=np.log(np.clip(x[var].to_numpy(float),1e-12,None))
    return x
g1=layermean(c1);g45=layermean(c45)
wide=g1.rename(columns={"g_micro":"v1"}).merge(g45.rename(columns={"g_micro":"v45"}),on=["prompt_hash","seed","token_idx"])
sig=c1.groupby("token_idx").sigma.mean().sort_index()
ratio_rows=[];per_prompt={}
for (p,t),g in wide.groupby(["prompt_hash","token_idx"]):
    if g.seed.nunique()!=2:continue
    s1=g.v1.std(ddof=1);s45=g.v45.std(ddof=1)
    rr=s45/s1 if s1>0 else np.nan
    ratio_rows.append([p,t,float(sig.loc[t]),rr])
RR=pd.DataFrame(ratio_rows,columns=["prompt_hash","step","sigma","sd_ratio_cfg45_over_cfg1"])
RR.to_csv(OUT/"R1f_cfg_spread_ratio_cells.csv",index=False)
start=int(np.ceil(len(sig)/4))
sel=RR[RR.step>=start]
obs=float(np.nanmedian(sel.sd_ratio_cfg45_over_cfg1))
rng=np.random.default_rng(1601);prompts=RR.prompt_hash.unique();boots=[]
for _ in range(2000):
    pp=rng.choice(prompts,len(prompts),replace=True)
    arr=np.concatenate([RR[(RR.prompt_hash==p)&(RR.step>=start)].sd_ratio_cfg45_over_cfg1.to_numpy() for p in pp])
    boots.append(np.nanmedian(arr))
R["R1f"]=dict(last_three_quarters_median=obs,CI95=[float(np.nanquantile(boots,.025)),float(np.nanquantile(boots,.975))],
                n_prompts=int(len(prompts)),verdict="CONFIRMED" if obs<.7 else "REFUTED",
                caveat="global seeds 1001/1002 reused across prompts; ratio is paired within prompt and remains calculable")

# stage summary
edges=[(0,10),(10,20),(20,30),(30,40)]
st=[]
for i,(lo,hi) in enumerate(edges,1):
    x=RR[(RR.step>=lo)&(RR.step<hi)].sd_ratio_cfg45_over_cfg1
    st.append(dict(stage=i,steps=f"{lo}-{hi-1}",median=float(np.nanmedian(x)),q25=float(np.nanquantile(x,.25)),q75=float(np.nanquantile(x,.75))))
pd.DataFrame(st).to_csv(OUT/"R1f_cfg_spread_stage_summary.csv",index=False)
R["R1f_stages"]=st

# R1g prompt influence: cond rows only, one value/layer duplicated, summarize by step.
pi=c45[c45.branch.eq("cond")].groupby("token_idx").agg(sigma=("sigma","mean"),median_prompt_influence=("prompt_influence","median"),min_prompt_influence=("prompt_influence","min")).reset_index()
pi.to_csv(OUT/"R1g_prompt_influence_by_step.csv",index=False)
eligible=pi[pi.sigma<=.9]
minpi=float(eligible.min_prompt_influence.min())
R["R1g"]=dict(min_prompt_influence_sigma_le_09=minpi,
               first_sigma_below_0999=float(eligible.loc[eligible.min_prompt_influence.lt(.999),"sigma"].max()) if eligible.min_prompt_influence.lt(.999).any() else None,
               verdict="CONFIRMED" if minpi<.999 else "REFUTED")

# R1h branch-mean vs cond, steps >=2. Calculate g and other observables posthoc.
bh=[]
for var,log in [("g_micro",True),("spectral_gap",True),("sv_entropy",False)]:
    q=c45[c45.token_idx>=2].groupby(["prompt_hash","seed","token_idx","layer","branch"])[var].mean().unstack("branch").dropna()
    bm=(q["cond"]+q["uncond"])/2
    if log: delta=np.log(np.clip(bm,1e-12,None)/np.clip(q["cond"],1e-12,None))
    else: delta=bm-q["cond"]
    bh.append(dict(variable=var,median_BM_minus_cond=float(np.median(delta)),q25=float(np.quantile(delta,.25)),q75=float(np.quantile(delta,.75))))
pd.DataFrame(bh).to_csv(OUT/"R1h_branchmean_vs_cond.csv",index=False);R["R1h"]=bh

# Posthoc cond/uncond separation in normalized depth geometry.
# For each prompt/seed/step, standardize each 3-metric depth profile within branch and correlate.
rows=[]
for (p,sd,t),g in c45.groupby(["prompt_hash","seed","token_idx"]):
    a=g[g.branch=="cond"].sort_values("layer")
    b=g[g.branch=="uncond"].sort_values("layer")
    if len(a)!=24 or len(b)!=24: continue
    A=np.c_[a.g_micro,-a.spectral_gap,a.sv_entropy];B=np.c_[b.g_micro,-b.spectral_gap,b.sv_entropy]
    A=(A-A.mean(0))/(A.std(0)+1e-12);B=(B-B.mean(0))/(B.std(0)+1e-12)
    r=np.corrcoef(A.ravel(),B.ravel())[0,1]
    rows.append([p,sd,t,float(a.sigma.mean()),r,
                 float(np.log(a.g_micro.mean()/b.g_micro.mean())),
                 float(np.log(a.spectral_gap.mean()/b.spectral_gap.mean())),
                 float(a.sv_entropy.mean()-b.sv_entropy.mean())])
D=pd.DataFrame(rows,columns=["prompt_hash","seed","step","sigma","shape_r_cond_uncond","log_g_cond_over_uncond","log_gap_cond_over_uncond","H_cond_minus_uncond"])
D.to_csv(OUT/"POSTHOC_cond_uncond_geometry.csv",index=False)
R["posthoc_cond_uncond"]=dict(mean_shape_r=float(D.shape_r_cond_uncond.mean()),min_shape_r=float(D.shape_r_cond_uncond.min()),
    early_shape_r=float(D[D.step<10].shape_r_cond_uncond.mean()),late_shape_r=float(D[D.step>=30].shape_r_cond_uncond.mean()),
    median_abs_log_g_diff=float(np.median(np.abs(D.log_g_cond_over_uncond))),
    median_abs_log_gap_diff=float(np.median(np.abs(D.log_gap_cond_over_uncond))),
    median_abs_H_diff=float(np.median(np.abs(D.H_cond_minus_uncond))))

(OUT/"R1_SCORECARD_AND_POSTHOC.json").write_text(json.dumps(R,indent=2,default=lambda x: x.item() if hasattr(x,"item") else str(x)),encoding="utf-8")
print(json.dumps(R,indent=2,default=lambda x: x.item() if hasattr(x,"item") else str(x)))
