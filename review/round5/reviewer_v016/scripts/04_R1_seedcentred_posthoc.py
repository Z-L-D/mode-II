#!/usr/bin/env python3
from __future__ import annotations
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
sys.path.insert(0,str(Path(__file__).parent))
from r4_common import *

OUT=Path("/mnt/data/r4_r5_audit_v016/results");OUT.mkdir(parents=True,exist_ok=True)

c1,_=load_all(["SD3.5M-attn-TE3-CFG1.0-R1024-S1001-STUDY[CFG-RES-SEED-CORRECTION]",
               "SD3.5M-attn-TE3-CFG1.0-R1024-S1002-STUDY[CFG-RES-SEED-CORRECTION]"])
c45,_=load_all(["SD3.5M-attn-TE3-CFG4.5-R1024-S1001-STUDY[CFG-RES-SEED-CORRECTION]",
                "SD3.5M-attn-TE3-CFG4.5-R1024-S1002-STUDY[CFG-RES-SEED-CORRECTION]"])

def cube(d,branch):
    q=d[d.branch.eq(branch)].copy()
    metrics=["g_micro","spectral_gap","sv_entropy"]
    ids=q[["prompt_hash","seed"]].drop_duplicates().sort_values(["prompt_hash","seed"])
    toks=sorted(q.token_idx.unique());layers=sorted(q.layer.unique())
    X=[];meta=[]
    for _,r in ids.iterrows():
        z=q[(q.prompt_hash==r.prompt_hash)&(q.seed==r.seed)]
        arr=np.empty((len(toks),len(layers),3))
        ok=True
        for it,t in enumerate(toks):
            a=z[z.token_idx==t].set_index("layer").reindex(layers)
            if a[metrics].isna().any().any():ok=False;break
            arr[it,:,0]=a.g_micro
            arr[it,:,1]=a.spectral_gap
            arr[it,:,2]=a.sv_entropy
        if ok:
            X.append(arr);meta.append((int(r.prompt_hash),int(r.seed)))
    return pd.DataFrame(meta,columns=["prompt_hash","seed"]),np.stack(X),q.groupby("token_idx").sigma.mean().sort_index().to_numpy(float)

def residual(X,meta):
    Y=X.copy().astype(float);Y[...,0]=np.log(np.clip(Y[...,0],1e-12,None));Y[...,1]=np.log(np.clip(Y[...,1],1e-12,None))
    G,T,L,K=Y.shape;F=Y.reshape(G,T,-1);F=(F-F.mean(0))/(F.std(0)+1e-12)
    grp=pd.factorize(meta.prompt_hash)[0]
    r=np.empty_like(F)
    for t in range(T):
        for g in np.unique(grp):
            idx=grp==g;r[idx,t]=F[idx,t]-F[idx,t].mean(0)
    return r,grp

def ridge(A,B,alpha=100):
    ma=A.mean(0);mb=B.mean(0);W=np.linalg.solve((A-ma).T@(A-ma)+alpha*np.eye(A.shape[1]),(A-ma).T@(B-mb))
    return lambda Z:(Z-ma)@W+mb

def fate_curve(r,meta,center):
    groups=pd.factorize(meta.prompt_hash)[0];seed=meta.seed.to_numpy();T=r.shape[1]
    pred=np.zeros((T,r.shape[0],r.shape[2]));target=np.zeros_like(r[:,-1])
    for tr,te in GroupKFold(5).split(r[:,0],groups=groups):
        rc=r.copy()
        if center:
            for sd in np.unique(seed):
                ii=tr[seed[tr]==sd]
                mu=r[ii].mean(0)
                rc[seed==sd]=r[seed==sd]-mu
        target[te]=rc[te,-1]
        for t in range(T):pred[t,te]=ridge(rc[tr,t],rc[tr,-1])(rc[te,t])
    den=((target-target.mean(0))**2).sum()
    return np.array([1-((target-pred[t])**2).sum()/den for t in range(T)])

def sigma50(curve,sigma):
    if not np.any(curve>.5):return np.nan
    i=int(np.argmax(curve>.5));tau=np.arange(len(curve))/(len(curve)-1)
    if i==0:t=.0
    else:t=tau[i-1]+(.5-curve[i-1])*(tau[i]-tau[i-1])/(curve[i]-curve[i-1])
    return float(np.interp(t,tau,sigma))

def vr(r,meta,center,k=4):
    rc=r.copy();seed=meta.seed.to_numpy()
    if center:
        for sd in np.unique(seed):rc[seed==sd]-=r[seed==sd].mean(0)
    rr=rc[:,1:]
    v1=((rr[:,1:]-rr[:,:-1])**2).mean()
    return float(((rr[:,k:]-rr[:,:-k])**2).mean()/(k*v1))

rows=[];curves=[]
for label,d,br in [("CFG1_cond",c1,"cond"),("CFG45_cond",c45,"cond"),("CFG45_uncond",c45,"uncond")]:
    m,X,s=cube(d,br);r,g=residual(X,m)
    for center in [False,True]:
        c=fate_curve(r,m,center);sf=sigma50(c,s);v=vr(r,m,center,4)
        rows.append([label,"seed-centred" if center else "uncentred",sf,float(c[0]),v])
        curves.extend([[label,"seed-centred" if center else "uncentred",i,s[i],c[i]] for i in range(len(s))])
pd.DataFrame(rows,columns=["arm","estimator","sigma50","R2_step0","VR4"]).to_csv(OUT/"POSTHOC_R1_seedcentred_summary.csv",index=False)
pd.DataFrame(curves,columns=["arm","estimator","step","sigma","R2"]).to_csv(OUT/"POSTHOC_R1_seedcentred_curves.csv",index=False)
print(pd.DataFrame(rows,columns=["arm","estimator","sigma50","R2_step0","VR4"]).to_string(index=False))
