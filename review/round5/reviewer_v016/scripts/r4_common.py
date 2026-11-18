#!/usr/bin/env python3
from __future__ import annotations
import zipfile,io,re
from pathlib import Path
import numpy as np,pandas as pd

OUTER="/mnt/data/uncond-cond+deepstep-telemetry.zip"

def members():
    with zipfile.ZipFile(OUTER) as z:
        return [n for n in z.namelist() if n.lower().endswith(".zip")]

def load_member(match):
    ms=[m for m in members() if match in m]
    if len(ms)!=1:
        raise ValueError(f"{match}: found {ms}")
    m=ms[0]
    with zipfile.ZipFile(OUTER) as outer:
        raw=outer.read(m)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        frames=[]
        for n in z.namelist():
            if n.lower().endswith(".csv"):
                raw_csv=z.read(n)
                d=pd.read_csv(io.BytesIO(raw_csv),low_memory=False)
                if "gen_idx" not in d.columns and len(d.columns)==21:
                    names=["gen_idx","token_idx","token_type","layer","submodule","g_micro",
                           "phi_micro","phi_micro_skip","spectral_gap","sv_entropy","sigma",
                           "seq_len","hidden_dim","hook_mode","prompt_hash","prompt_len",
                           "gen_len","prompt_text","branch","prompt_influence","seed"]
                    d=pd.read_csv(io.BytesIO(raw_csv),header=None,names=names,low_memory=False)
                d["_csv"]=n;frames.append(d)
    d=pd.concat(frames,ignore_index=True)
    # Forward-fill prompt_text within prompt_hash because uncond and subsequent rows may be blank.
    if "prompt_hash" in d and "prompt_text" in d:
        mapping=d.dropna(subset=["prompt_text"]).drop_duplicates("prompt_hash").set_index("prompt_hash").prompt_text
        d["prompt_text_full"]=d.prompt_hash.map(mapping)
    return d,m

def load_all(matches):
    ds=[];names=[]
    for x in matches:
        d,n=load_member(x);ds.append(d);names.append(n)
    return pd.concat(ds,ignore_index=True),names

def beta24_from_field(G,bins=24,min_per=5):
    """Exact frozen Round-2/4 beta24 estimator."""
    G=np.asarray(G,float)
    g=G[:,:-1].ravel()
    b=np.diff(G,axis=1).ravel()
    m=np.isfinite(g)&np.isfinite(b)
    g,b=g[m],b[m]
    e=np.linspace(np.percentile(g,2),np.percentile(g,98),25)
    c=0.5*(e[:-1]+e[1:])
    bm=np.full(24,np.nan)
    nn=np.zeros(24,int)
    for i in range(24):
        k=(g>=e[i])&(g<e[i+1])
        nn[i]=int(k.sum())
        if k.sum()>=min_per:
            bm[i]=b[k].mean()
    return pd.DataFrame({"g":c,"beta":bm,"n":nn})

def sign_sequence(curve):
    s=np.sign(curve.beta.to_numpy(float));s=s[s!=0]
    if not len(s):return ""
    out=["+" if s[0]>0 else "-"]
    for v in s[1:]:
        z="+" if v>0 else "-"
        if z!=out[-1]:out.append(z)
    return "".join(out)

def crossings(curve):
    out=[]
    for i in range(len(curve)-1):
        x0,x1=curve.g.iloc[i],curve.g.iloc[i+1]
        y0,y1=curve.beta.iloc[i],curve.beta.iloc[i+1]
        if y0*y1<0:
            out.append(float(x0-y0*(x1-x0)/(y1-y0)))
    return out

def field_from_df(d,branch="cond"):
    q=d[d.branch.eq(branch)].copy() if "branch" in d else d.copy()
    # one observation row per generation/prompt/seed/step x layer
    piv=q.pivot_table(index=["prompt_hash","seed","token_idx"],columns="layer",values="g_micro",aggfunc="mean").sort_index()
    return piv

def bootstrap_prompts(values_by_prompt,stat=np.nanmedian,B=1000,seed=1):
    keys=list(values_by_prompt)
    rng=np.random.default_rng(seed);vals=[]
    for _ in range(B):
        sel=rng.choice(keys,len(keys),replace=True)
        x=np.concatenate([np.ravel(values_by_prompt[k]) for k in sel])
        vals.append(stat(x))
    return np.asarray(vals,float)
