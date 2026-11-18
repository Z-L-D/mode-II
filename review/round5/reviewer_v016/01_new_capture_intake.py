#!/usr/bin/env python3
from __future__ import annotations
import argparse, zipfile, io, hashlib, json
from pathlib import Path
import pandas as pd
import numpy as np

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def read_nested_zip(outer_path, member):
    with zipfile.ZipFile(outer_path) as outer:
        raw=outer.read(member)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        csvs=[n for n in z.namelist() if n.lower().endswith(".csv")]
        frames=[]
        for n in csvs:
            try:
                d=pd.read_csv(io.BytesIO(z.read(n)),low_memory=False)
            except Exception:
                continue
            d["_csv"]=n
            frames.append(d)
        return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(), csvs

def summarize(outer_path, member):
    d,csvs=read_nested_zip(outer_path,member)
    out={"member":member,"n_csv":len(csvs),"n_rows":len(d)}
    if d.empty:return out
    for c in ["hook_mode","branch","submodule","cfg","resolution","seed","prompt_hash","prompt_influence","token_idx","layer","sigma","gen_idx","prompt_text"]:
        if c in d.columns:
            s=d[c]
            out[f"{c}_nunique"]=int(s.nunique(dropna=True))
            vals=s.dropna().astype(str).unique().tolist()
            out[f"{c}_examples"]=";".join(vals[:8])
            if c=="prompt_influence":
                out["prompt_influence_finite_frac"]=float(np.isfinite(pd.to_numeric(s,errors="coerce")).mean())
    if "token_idx" in d: out["n_steps"]=int(d.token_idx.nunique())
    if "sigma" in d:
        out["sigma_min"]=float(pd.to_numeric(d.sigma,errors="coerce").min())
        out["sigma_max"]=float(pd.to_numeric(d.sigma,errors="coerce").max())
    if "prompt_hash" in d and "seed" in d:
        ps=d[["prompt_hash","seed"]].drop_duplicates()
        seeds_per_prompt=ps.groupby("prompt_hash").seed.nunique()
        prompts_per_seed=ps.groupby("seed").prompt_hash.nunique()
        out["seed_pairs_per_prompt_median"]=float(seeds_per_prompt.median()) if len(seeds_per_prompt) else np.nan
        out["max_prompts_per_seed"]=int(prompts_per_seed.max()) if len(prompts_per_seed) else 0
        out["n_unique_seeds"]=int(ps.seed.nunique())
        out["unique_seed_design"]=bool((prompts_per_seed<=1).all())
    if {"prompt_hash","seed","token_idx","layer"}.issubset(d.columns):
        key=["prompt_hash","seed","token_idx"]
        if "branch" in d:key.append("branch")
        nlay=d.groupby(key).layer.nunique()
        out["layer_count_min"]=int(nlay.min());out["layer_count_max"]=int(nlay.max())
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive",default="/mnt/data/uncond-cond+deepstep-telemetry.zip")
    ap.add_argument("--outdir",default="/mnt/data/r4_r5_audit_v016")
    a=ap.parse_args()
    p=Path(a.archive);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(p) as z:
        members=[n for n in z.namelist() if n.lower().endswith(".zip")]
    rows=[summarize(p,m) for m in members]
    df=pd.DataFrame(rows)
    df.to_csv(out/"NEW_CAPTURE_INTAKE.csv",index=False)
    meta={"archive":str(p),"sha256":sha256(p),"n_members":len(members),"members":members}
    (out/"NEW_CAPTURE_INTAKE.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(df.to_string(index=False))
    print(json.dumps(meta,indent=2))
if __name__=="__main__":main()
