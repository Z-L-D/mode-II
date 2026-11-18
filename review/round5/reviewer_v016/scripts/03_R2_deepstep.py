#!/usr/bin/env python3
from __future__ import annotations
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from r4_common import *
OUT=Path("/mnt/data/r4_r5_audit_v016/results");OUT.mkdir(parents=True,exist_ok=True)
R={}

d100,n100=load_member("CFG1.0-R1024-S(RAND)-STUDY[CFG-PYRAMID-DEEPSTEP100]")
d40,n40=load_member("CFG1.0-R1024-S(RAND)-STUDY[CFG-PYRAMID-DEEPSTEP40]")
d100g,n100g=load_member("CFG4.5-R1024-S(RAND)-STUDY[CFG-PYRAMID-DEEPSTEP100]")
d40g,n40g=load_member("CFG4.5-R1024-S(RAND)-STUDY[CFG-PYRAMID-DEEPSTEP40]")

# Design/intake details
for lab,d in [("CFG1_100",d100),("CFG1_40",d40),("CFG45_100",d100g),("CFG45_40",d40g)]:
    ps=d[["prompt_hash","seed"]].drop_duplicates()
    R[lab+"_design"]=dict(n_prompts=int(d.prompt_hash.nunique()),n_seeds=int(d.seed.nunique()),n_steps=int(d.token_idx.nunique()),
        branches=sorted(d.branch.dropna().unique().tolist()),hook_modes=sorted(d.hook_mode.dropna().unique().tolist()),
        layer_min=int(d.groupby(["prompt_hash","seed","token_idx","branch"]).layer.nunique().min()),
        layer_max=int(d.groupby(["prompt_hash","seed","token_idx","branch"]).layer.nunique().max()))
R["R2_formal_status"]="NOT SCORABLE: CFG1 100-step capture has 15 seeds, below preregistered >=30"

def topology(d,branch="cond"):
    p=field_from_df(d,branch)
    curve=beta24_from_field(p.to_numpy(),24,10)
    return curve,sign_sequence(curve),crossings(curve)
for lab,d in [("CFG1_100",d100),("CFG1_40",d40),("CFG45_100_cond",d100g),("CFG45_40_cond",d40g),("CFG45_100_uncond",d100g),("CFG45_40_uncond",d40g)]:
    br="uncond" if lab.endswith("uncond") else "cond"
    c,sg,cr=topology(d,br)
    c.to_csv(OUT/f"R2_beta_curve_{lab}.csv",index=False)
    R[lab+"_topology"]=dict(sign_sequence=sg,n_crossings=len(cr),crossings=cr)

# R2a posthoc verdict from 15-seed pooled curve.
sg=R["CFG1_100_topology"]["sign_sequence"];nc=R["CFG1_100_topology"]["n_crossings"]
R["R2a_posthoc"]=dict(sign_sequence=sg,n_crossings=nc,
    prereg_condition_satisfied=(sg.startswith("+") and sg!="-+-+-" and nc<=3),
    formal_verdict="NOT SCORABLE (n=15 seeds < preregistered 30)")

# seed bootstrap stability for topology (post hoc)
piv=field_from_df(d100,"cond")
# Pivot index = prompt_hash, seed, token; only one prompt. Bootstrap seeds and preserve all steps.
seeds=np.array(sorted(d100.seed.unique()))
rng=np.random.default_rng(1602);seqs=[]
for _ in range(2000):
    sel=rng.choice(seeds,len(seeds),replace=True)
    mats=[]
    for j,sd in enumerate(sel):
        # all steps for seed; duplicate allowed
        x=piv.xs(sd,level="seed",drop_level=False).to_numpy()
        mats.append(x)
    G=np.vstack(mats)
    c=beta24_from_field(G,24,10);seqs.append(sign_sequence(c))
freq=pd.Series(seqs).value_counts(normalize=True).rename_axis("sequence").reset_index(name="fraction")
freq.to_csv(OUT/"R2a_seed_bootstrap_topology.csv",index=False)
R["R2a_bootstrap_topology"]=freq.head(10).to_dict("records")

# R2b: same-prompt 100 vs 40 matched sigma is a stronger posthoc comparison.
def mean_field(d,branch="cond"):
    q=d[d.branch.eq(branch)]
    return q.groupby(["token_idx","layer"]).g_micro.mean().unstack("layer").sort_index(),q.groupby("token_idx").sigma.mean().sort_index()
G100,s100=mean_field(d100);G40,s40=mean_field(d40)
diffs=[];jmap=[]
for t,s in s40.items():
    j=int((s100-s).abs().idxmin())
    v100=G100.loc[j].to_numpy(float);v40=G40.loc[t].to_numpy(float)
    diffs.extend(np.abs(np.log(v100/v40)))
    jmap.append([t,float(s),j,float(s100.loc[j]),float(np.median(np.abs(np.log(v100/v40))))])
J=pd.DataFrame(jmap,columns=["step40","sigma40","step100","sigma100","median_abs_log_ratio"])
J.to_csv(OUT/"R2b_same_prompt_sigma_match.csv",index=False)
R["R2b_same_prompt_posthoc"]=dict(median_abs_log_ratio=float(np.median(diffs)),q95=float(np.quantile(diffs,.95)),
    max=float(np.max(diffs)),condition_le_005=bool(np.median(diffs)<=.05),
    formal_verdict="NOT SCORABLE (n=15 seeds < preregistered 30), but same-prompt 40/100 comparison is stronger than frozen prompt-mismatched control")

# Compare path topology per individual seed.
seedrows=[]
for sd in seeds:
    q=d100[(d100.seed==sd)&(d100.branch=="cond")]
    p=q.pivot_table(index="token_idx",columns="layer",values="g_micro").sort_index()
    c=beta24_from_field(p.to_numpy(),24,5)
    seedrows.append([int(sd),sign_sequence(c),len(crossings(c))])
SD=pd.DataFrame(seedrows,columns=["seed","sign_sequence","n_crossings"])
SD.to_csv(OUT/"R2a_individual_seed_topology.csv",index=False)
R["R2a_individual_seeds"]=dict(first_plus_fraction=float(SD.sign_sequence.str.startswith("+").mean()),
                               legacy_four_cross_fraction=float((SD.sign_sequence=="-+-+-").mean()),
                               modal_sequence=str(SD.sign_sequence.mode().iloc[0]))

# Posthoc stage-count/guidance factorial: 40 vs 100 and CFG1 vs CFG4.5, cond branch.
# State fields compared at matched sigma after averaging seeds.
def shape_fields(d):
    q=d[d.branch=="cond"]
    rows={}
    for t,g in q.groupby("token_idx"):
        a=g.groupby("layer")[["g_micro","spectral_gap","sv_entropy"]].mean().sort_index().to_numpy(float)
        a[:,1]*=-1;a=(a-a.mean(0))/(a.std(0)+1e-12);rows[int(t)]=a
    s=q.groupby("token_idx").sigma.mean()
    return rows,s
factor=[]
datasets={"CFG1_40":d40,"CFG1_100":d100,"CFG45_40":d40g,"CFG45_100":d100g}
fields={k:shape_fields(v) for k,v in datasets.items()}
pairs=[("CFG1_40","CFG1_100"),("CFG45_40","CFG45_100"),("CFG1_40","CFG45_40"),("CFG1_100","CFG45_100")]
for A,B in pairs:
    fa,sa=fields[A];fb,sb=fields[B];rr=[];gd=[]
    for ta,s in sa.items():
        tb=int((sb-s).abs().idxmin())
        a=fa[int(ta)];b=fb[tb]
        rr.append(np.corrcoef(a.ravel(),b.ravel())[0,1])
        # absolute g population means
        ga=datasets[A][(datasets[A].branch=="cond")&(datasets[A].token_idx==ta)].g_micro.mean()
        gb=datasets[B][(datasets[B].branch=="cond")&(datasets[B].token_idx==tb)].g_micro.mean()
        gd.append(abs(np.log(ga/gb)))
    factor.append(dict(pair=f"{A} vs {B}",mean_shape_r=float(np.mean(rr)),min_shape_r=float(np.min(rr)),
                       median_abs_log_g=float(np.median(gd)),max_abs_log_g=float(np.max(gd))))
pd.DataFrame(factor).to_csv(OUT/"POSTHOC_deepstep_cfg_factorial.csv",index=False)
R["posthoc_factorial"]=factor

(OUT/"R2_DEEPSTEP_RESULTS.json").write_text(json.dumps(R,indent=2),encoding="utf-8")
print(json.dumps(R,indent=2))
