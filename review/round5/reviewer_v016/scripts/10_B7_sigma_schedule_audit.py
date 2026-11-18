#!/usr/bin/env python3
from __future__ import annotations
import zipfile,io,sys,math
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from r4_common import *
OUT=Path("/mnt/data/r4_r5_audit_v016/results");OUT.mkdir(parents=True,exist_ok=True)

new,_=load_member("CFG1.0-R1024-S(RAND)-STUDY[CFG-PYRAMID-DEEPSTEP100]")
new=new[new.branch=="cond"]
with zipfile.ZipFile("/mnt/data/TELEMETRY.zip") as o:raw=o.read("DM/SD3.5MED-CFG1.0-attn-512.768.1024-30.zip")
frames=[]
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for n in z.namelist():
        if "1024" in n:
            d=pd.read_csv(io.BytesIO(z.read(n)));frames.append(d)
old=pd.concat(frames)
so=old.groupby("token_idx").sigma.mean().sort_index()
sn=new.groupby("token_idx").sigma.mean().sort_index()
rows=[]
for i in range(max(len(so),len(sn))):
    rows.append([i,float(so.iloc[i]) if i<len(so) else np.nan,float(sn.iloc[i]) if i<len(sn) else np.nan,
                 float(sn.iloc[i]-so.iloc[i]) if i<len(so) and i<len(sn) else np.nan])
D=pd.DataFrame(rows,columns=["step","sigma_legacy","sigma_passive","passive_minus_legacy"])
D.to_csv(OUT/"B7_100STEP_SIGMA_GRID_COMPARISON.csv",index=False)

# nearest matching residual each direction
oldvals=so.to_numpy();newvals=sn.to_numpy()
dn=[float(np.min(np.abs(oldvals-x))) for x in newvals]
do=[float(np.min(np.abs(newvals-x))) for x in oldvals]
summary=pd.DataFrame([{
    "max_same_index_diff":float(np.max(np.abs(newvals-oldvals))),
    "median_same_index_diff":float(np.median(np.abs(newvals-oldvals))),
    "new_to_old_nearest_median":float(np.median(dn)),
    "new_to_old_nearest_max":float(np.max(dn)),
    "old_to_new_nearest_median":float(np.median(do)),
    "old_to_new_nearest_max":float(np.max(do)),
    "legacy_sigma_min":float(oldvals.min()),"legacy_sigma_max":float(oldvals.max()),
    "passive_sigma_min":float(newvals.min()),"passive_sigma_max":float(newvals.max())
}])
summary.to_csv(OUT/"B7_100STEP_SIGMA_GRID_SUMMARY.csv",index=False)
print(summary.to_string(index=False))
print("\nSelected rows:")
print(D.iloc[[0,1,2,5,10,20,30,40,50,60,70,80,90,97,98,99]].to_string(index=False))
# exact hypergeometric probability for perfect separation of 30 and 15 signs, 2-sided Fisher = 2/C(45,15) for symmetric extremes if applicable
p_one=1/math.comb(45,15)
print("\nPerfect-separation one-sided hypergeometric:",format(p_one,'.6e'),"two-extreme:",format(2*p_one,'.6e'))
