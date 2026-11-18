#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
sys.path.insert(0,str(Path(__file__).parent))
from r4_common import *
OUT=Path("/mnt/data/r4_r5_audit_v016/results");OUT.mkdir(parents=True,exist_ok=True)

c45,_=load_all(["SD3.5M-attn-TE3-CFG4.5-R1024-S1001-STUDY[CFG-RES-SEED-CORRECTION]",
                "SD3.5M-attn-TE3-CFG4.5-R1024-S1002-STUDY[CFG-RES-SEED-CORRECTION]"])

def make_cube(d,kind):
    mets=["g_micro","spectral_gap","sv_entropy"]
    ids=d[["prompt_hash","seed"]].drop_duplicates().sort_values(["prompt_hash","seed"])
    toks=sorted(d.token_idx.unique());lays=sorted(d.layer.unique())
    X=[];meta=[]
    for _,r in ids.iterrows():
        z=d[(d.prompt_hash==r.prompt_hash)&(d.seed==r.seed)]
        arr=np.empty((len(toks),len(lays),3))
        good=True
        for it,t in enumerate(toks):
            q=z[z.token_idx==t]
            a=q[q.branch=="cond"].set_index("layer").reindex(lays)[mets]
            b=q[q.branch=="uncond"].set_index("layer").reindex(lays)[mets]
            if a.isna().any().any() or b.isna().any().any():good=False;break
            if kind=="cond":v=a.to_numpy(float)
            elif kind=="uncond":v=b.to_numpy(float)
            elif kind=="branchmean":v=(a.to_numpy(float)+b.to_numpy(float))/2
            arr[it]=v
        if good:X.append(arr);meta.append((int(r.prompt_hash),int(r.seed)))
    return pd.DataFrame(meta,columns=["prompt_hash","seed"]),np.stack(X),d.groupby("token_idx").sigma.mean().sort_index().to_numpy(float)

def residual(X,meta):
    Y=X.copy();Y[...,0]=np.log(np.clip(Y[...,0],1e-12,None));Y[...,1]=np.log(np.clip(Y[...,1],1e-12,None))
    F=Y.reshape(len(Y),Y.shape[1],-1);F=(F-F.mean(0))/(F.std(0)+1e-12)
    grp=pd.factorize(meta.prompt_hash)[0];r=np.empty_like(F)
    for t in range(F.shape[1]):
        for g in np.unique(grp):
            ix=grp==g;r[ix,t]=F[ix,t]-F[ix,t].mean(0)
    return r,grp

def center_seed(r,meta):
    rc=r.copy();s=meta.seed.to_numpy()
    for sd in np.unique(s):rc[s==sd]-=r[s==sd].mean(0)
    return rc

def VR(r,k=4):
    rr=r[:,1:];v1=((rr[:,1:]-rr[:,:-1])**2).mean()
    return float(((rr[:,k:]-rr[:,:-k])**2).mean()/(k*v1))

rows=[]
for kind in ["cond","uncond","branchmean"]:
    m,X,s=make_cube(c45,kind);r,_=residual(X,m)
    rows.append([kind,VR(r),VR(center_seed(r,m))])
pd.DataFrame(rows,columns=["representation","VR4_uncentred","VR4_seedcentred"]).to_csv(OUT/"POSTHOC_R1_branch_persistence_decomposition.csv",index=False)
print(pd.DataFrame(rows,columns=["representation","VR4_uncentred","VR4_seedcentred"]).to_string(index=False))
