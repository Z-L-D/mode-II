#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from r4_common import *
OUT=Path("/mnt/data/r4_r5_audit_v016/results");OUT.mkdir(parents=True,exist_ok=True)

arms=[
("512","SD3.5M-attn-TE3-CFG4.5-R512-S1001-STUDY[CFG-RES-SEED-CORRECTION]"),
("768","SD3.5M-attn-TE3-CFG4.5-R768-S1001-STUDY[CFG-RES-SEED-CORRECTION]"),
("1024","SD3.5M-attn-TE3-CFG4.5-R1024-S1001-STUDY[CFG-RES-SEED-CORRECTION]")
]
rng=np.random.default_rng(1613);rows=[];B=500
for res,match in arms:
    d,_=load_member(match);d=d[d.branch=="cond"]
    cubes=[]
    for p,g in d.groupby("prompt_hash"):
        q=g.pivot_table(index="token_idx",columns="layer",values="g_micro").sort_index()
        if q.shape==(40,24) and not q.isna().any().any():cubes.append(q.to_numpy(float))
    X=np.stack(cubes) # P,T,L
    seqs=[]
    for _ in range(B):
        ix=rng.integers(0,len(X),len(X))
        c=beta24_from_field(X[ix].reshape(-1,24));seqs.append(sign_sequence(c))
    vc=pd.Series(seqs).value_counts()
    for seq,n in vc.items():rows.append([res,len(X),seq,n/B])
df=pd.DataFrame(rows,columns=["resolution","n_complete_prompts","sequence","bootstrap_fraction"])
df.to_csv(OUT/"CURRENT_RESOLUTION_TOPOLOGY_BOOTSTRAP.csv",index=False)
print(df.sort_values(["resolution","bootstrap_fraction"],ascending=[True,False]).to_string(index=False))
