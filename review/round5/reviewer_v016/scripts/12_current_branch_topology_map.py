#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from r4_common import *
OUT=Path("/mnt/data/r4_r5_audit_v016/results");OUT.mkdir(parents=True,exist_ok=True)

arms=[
("SD35_CFG1_1024",["SD3.5M-attn-TE3-CFG1.0-R1024-S1001-STUDY[CFG-RES-SEED-CORRECTION]","SD3.5M-attn-TE3-CFG1.0-R1024-S1002-STUDY[CFG-RES-SEED-CORRECTION]"]),
("SD35_CFG45_1024",["SD3.5M-attn-TE3-CFG4.5-R1024-S1001-STUDY[CFG-RES-SEED-CORRECTION]","SD3.5M-attn-TE3-CFG4.5-R1024-S1002-STUDY[CFG-RES-SEED-CORRECTION]"]),
("SD35_CFG45_768",["SD3.5M-attn-TE3-CFG4.5-R768-S1001-STUDY[CFG-RES-SEED-CORRECTION]"]),
("SD35_CFG45_512",["SD3.5M-attn-TE3-CFG4.5-R512-S1001-STUDY[CFG-RES-SEED-CORRECTION]"])
]
rows=[]
for lab,matches in arms:
    d,_=load_all(matches)
    for br in sorted(d.branch.dropna().unique()):
        p=field_from_df(d,br)
        c=beta24_from_field(p.to_numpy());sg=sign_sequence(c);cr=crossings(c)
        rows.append([lab,br,int(d.prompt_hash.nunique()),int(d.seed.nunique()),sg,len(cr),";".join(f"{x:.6g}" for x in cr)])
    if set(d.branch.dropna().unique())=={"cond","uncond"}:
        # synthetic branch mean per scalar, exactly the branch-mean functional used in separated-forward old capture
        q=d.pivot_table(index=["prompt_hash","seed","token_idx","layer"],columns="branch",values="g_micro").dropna()
        q["bm"]=(q["cond"]+q["uncond"])/2
        P=q["bm"].unstack("layer")
        c=beta24_from_field(P.to_numpy());sg=sign_sequence(c);cr=crossings(c)
        rows.append([lab,"synthetic_branchmean",int(d.prompt_hash.nunique()),int(d.seed.nunique()),sg,len(cr),";".join(f"{x:.6g}" for x in cr)])
df=pd.DataFrame(rows,columns=["arm","functional","n_prompts","n_seeds","sequence","n_crossings","crossings"])
df.to_csv(OUT/"CURRENT_BRANCH_BETA_TOPOLOGY_MAP.csv",index=False)
print(df.to_string(index=False))
