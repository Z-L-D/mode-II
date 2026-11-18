#!/usr/bin/env python3
from __future__ import annotations
import zipfile,io,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import fisher_exact,spearmanr
sys.path.insert(0,str(Path(__file__).parent))
from r4_common import *
OUT=Path("/mnt/data/r4_r5_audit_v016/results");OUT.mkdir(parents=True,exist_ok=True)

# New passive 100-step pyramid
new,_=load_member("CFG1.0-R1024-S(RAND)-STUDY[CFG-PYRAMID-DEEPSTEP100]")
new=new[new.branch=="cond"].copy()

# Legacy 100-step pyramid, 1024, two files = 30 runs. File name disambiguates repeated gen_idx.
with zipfile.ZipFile("/mnt/data/TELEMETRY.zip") as outer:
    raw=outer.read("DM/SD3.5MED-CFG1.0-attn-512.768.1024-30.zip")
frames=[]
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for n in z.namelist():
        if "1024" in n and n.endswith(".csv"):
            d=pd.read_csv(io.BytesIO(z.read(n)),low_memory=False)
            d["_file"]=n;d["_run"]=n+"::"+d.gen_idx.astype(str)
            frames.append(d)
legacy=pd.concat(frames,ignore_index=True)

# exact design checks
old_prompt=legacy.prompt_text.dropna().astype(str).unique().tolist() if "prompt_text" in legacy else []
new_prompt=new.prompt_text.dropna().astype(str).unique().tolist()
old_sig=legacy.groupby("token_idx").sigma.mean().sort_index().to_numpy(float)
new_sig=new.groupby("token_idx").sigma.mean().sort_index().to_numpy(float)
checks={
    "legacy_runs":int(legacy._run.nunique()),
    "new_runs":int(new.seed.nunique()),
    "legacy_steps":int(legacy.token_idx.nunique()),
    "new_steps":int(new.token_idx.nunique()),
    "sigma_grid_equal":bool(len(old_sig)==len(new_sig) and np.allclose(old_sig,new_sig,atol=1e-10)),
    "sigma_max_abs_diff":float(np.max(np.abs(old_sig-new_sig))) if len(old_sig)==len(new_sig) else None,
    "legacy_prompt_count":len(old_prompt),"new_prompt_count":len(new_prompt),
    "prompt_exact_match":bool(len(old_prompt)==1 and len(new_prompt)==1 and old_prompt[0].strip()==new_prompt[0].strip()),
    "legacy_prompt_len":len(old_prompt[0]) if old_prompt else None,
    "new_prompt_len":len(new_prompt[0]) if new_prompt else None,
}

def topo_from_df(d,runkey):
    rows=[]
    for rid,g in d.groupby(runkey):
        p=g.pivot_table(index="token_idx",columns="layer",values="g_micro").sort_index()
        c=beta24_from_field(p.to_numpy(),24,5);sg=sign_sequence(c);cr=crossings(c)
        rows.append([str(rid),sg,len(cr),sg[0] if sg else ""])
    return pd.DataFrame(rows,columns=["run","sequence","n_crossings","first_sign"])

oldT=topo_from_df(legacy,"_run")
newT=topo_from_df(new,"seed")
oldT["instrument"]="legacy_res_suite";newT["instrument"]="passive_branch_100"
allT=pd.concat([oldT,newT],ignore_index=True)
allT.to_csv(OUT/"B7_MATCHED100_PER_RUN_TOPOLOGY.csv",index=False)

# pooled
oldG=legacy.pivot_table(index=["_run","token_idx"],columns="layer",values="g_micro").to_numpy()
newG=new.pivot_table(index=["seed","token_idx"],columns="layer",values="g_micro").to_numpy()
co=beta24_from_field(oldG);cn=beta24_from_field(newG)
co.to_csv(OUT/"B7_MATCHED100_LEGACY_BETA.csv",index=False);cn.to_csv(OUT/"B7_MATCHED100_PASSIVE_BETA.csv",index=False)

# first sign 2x2
a=int((oldT.first_sign=="-").sum());b=int((oldT.first_sign=="+").sum())
c=int((newT.first_sign=="-").sum());d=int((newT.first_sign=="+").sum())
odds,p=fisher_exact([[a,b],[c,d]],alternative="two-sided")

# level and trend
old_step=legacy.groupby("token_idx").g_micro.mean();new_step=new.groupby("token_idx").g_micro.mean()
summary={
    **checks,
    "legacy_population_sequence":sign_sequence(co),
    "legacy_population_crossings":crossings(co),
    "passive_population_sequence":sign_sequence(cn),
    "passive_population_crossings":crossings(cn),
    "legacy_first_minus_fraction":float((oldT.first_sign=="-").mean()),
    "passive_first_plus_fraction":float((newT.first_sign=="+").mean()),
    "legacy_four_cross_fraction":float((oldT.sequence=="-+-+-").mean()),
    "passive_four_cross_fraction":float((newT.sequence=="-+-+-").mean()),
    "fisher_first_sign_odds_ratio":float(odds),
    "fisher_first_sign_p":float(p),
    "legacy_median_g":float(legacy.g_micro.median()),
    "passive_median_g":float(new.g_micro.median()),
    "median_log_g_legacy_over_passive":float(np.median(np.log(np.clip(legacy.g_micro.to_numpy(float),1e-12,None))))-float(np.median(np.log(np.clip(new.g_micro.to_numpy(float),1e-12,None)))),
    "legacy_rho_step_mean_g":float(spearmanr(old_step.index,old_step.values).statistic),
    "passive_rho_step_mean_g":float(spearmanr(new_step.index,new_step.values).statistic)
}
pd.DataFrame([summary]).to_csv(OUT/"B7_MATCHED100_SUMMARY.csv",index=False)
print(pd.Series(summary).to_string())
print("\nLegacy sequences:\n",oldT.sequence.value_counts().to_string())
print("\nPassive sequences:\n",newT.sequence.value_counts().to_string())
