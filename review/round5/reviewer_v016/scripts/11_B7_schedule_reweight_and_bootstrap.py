#!/usr/bin/env python3
from __future__ import annotations
import zipfile,io,sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from r4_common import *
OUT=Path("/mnt/data/r4_r5_audit_v016/results");OUT.mkdir(parents=True,exist_ok=True)

new,_=load_member("CFG1.0-R1024-S(RAND)-STUDY[CFG-PYRAMID-DEEPSTEP100]");new=new[new.branch=="cond"]
d40,_=load_member("CFG1.0-R1024-S(RAND)-STUDY[CFG-PYRAMID-DEEPSTEP40]");d40=d40[d40.branch=="cond"]

with zipfile.ZipFile("/mnt/data/TELEMETRY.zip") as o:raw=o.read("DM/SD3.5MED-CFG1.0-attn-512.768.1024-30.zip")
fr=[]
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for n in z.namelist():
        if "1024" in n:
            d=pd.read_csv(io.BytesIO(z.read(n)));d["_run"]=n+"::"+d.gen_idx.astype(str);fr.append(d)
old=pd.concat(fr,ignore_index=True)

def cube(d,runcol):
    s=d.groupby("token_idx").sigma.mean().sort_index()
    ids=sorted(d[runcol].unique())
    arr=[]
    for rid in ids:
        p=d[d[runcol]==rid].pivot_table(index="token_idx",columns="layer",values="g_micro").reindex(s.index)
        arr.append(p.to_numpy(float))
    return np.stack(arr),s.to_numpy(float)

Go,so=cube(old,"_run");Gn,sn=cube(new,"seed");G40,s40=cube(d40,"seed")

def interp_cube(G,s_from,s_to):
    order=np.argsort(s_from);sf=s_from[order]
    out=np.empty((G.shape[0],len(s_to),G.shape[2]))
    for i in range(G.shape[0]):
        for l in range(G.shape[2]):
            out[i,:,l]=np.interp(s_to,sf,G[i,order,l])
    return out
def top(G):
    c=beta24_from_field(G.reshape(-1,G.shape[-1]));return sign_sequence(c),crossings(c)

Oi=interp_cube(Go,so,sn);Ni=interp_cube(Gn,sn,so)
df=pd.DataFrame([
    ["legacy_native",*top(Go)],
    ["legacy_interpolated_to_passive_sigma_grid",*top(Oi)],
    ["passive_native",*top(Gn)],
    ["passive_interpolated_to_legacy_sigma_grid",*top(Ni)]
],columns=["condition","sign_sequence","crossings"])
df["n_crossings"]=df.crossings.map(len);df.to_csv(OUT/"B7_SIGMA_GRID_REWEIGHT_TOPOLOGY.csv",index=False)

# 40->100 nearest sigma map and vectorized bootstrap.
j=np.array([int(np.argmin(np.abs(sn-x))) for x in s40])
obs=float(np.median(np.abs(np.log(G40.mean(0)/Gn.mean(0)[j]))))
rng=np.random.default_rng(1611);B=5000;boots=np.empty(B)
for b in range(B):
    a=rng.integers(0,len(G40),len(G40));c=rng.integers(0,len(Gn),len(Gn))
    boots[b]=np.median(np.abs(np.log(G40[a].mean(0)/Gn[c].mean(0)[j])))
summ=pd.DataFrame([{"passive_40_vs_100_median_abs_log_g":obs,"boot_q025":float(np.quantile(boots,.025)),
                    "boot_q50":float(np.median(boots)),"boot_q975":float(np.quantile(boots,.975)),
                    "n_seed_40":len(G40),"n_seed_100":len(Gn)}])
summ.to_csv(OUT/"B7_PASSIVE_40_100_BOOTSTRAP.csv",index=False)

print(df.to_string(index=False))
print("\n",summ.to_string(index=False))
