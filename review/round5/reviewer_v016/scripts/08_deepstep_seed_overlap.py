#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from r4_common import *
OUT=Path("/mnt/data/r4_r5_audit_v016/results");OUT.mkdir(parents=True,exist_ok=True)

sets={}
for lab,match in [
    ("CFG1_40","CFG1.0-R1024-S(RAND)-STUDY[CFG-PYRAMID-DEEPSTEP40]"),
    ("CFG1_100","CFG1.0-R1024-S(RAND)-STUDY[CFG-PYRAMID-DEEPSTEP100]"),
    ("CFG45_40","CFG4.5-R1024-S(RAND)-STUDY[CFG-PYRAMID-DEEPSTEP40]"),
    ("CFG45_100","CFG4.5-R1024-S(RAND)-STUDY[CFG-PYRAMID-DEEPSTEP100]")]:
    d,_=load_member(match);sets[lab]=set(map(int,d.seed.unique()))
rows=[]
for A,B in [("CFG1_40","CFG1_100"),("CFG45_40","CFG45_100"),("CFG1_40","CFG45_40"),("CFG1_100","CFG45_100")]:
    inter=sets[A]&sets[B]
    rows.append([A,B,len(sets[A]),len(sets[B]),len(inter),";".join(map(str,sorted(inter)))])
df=pd.DataFrame(rows,columns=["arm_a","arm_b","n_seed_a","n_seed_b","n_shared","shared_seeds"])
df.to_csv(OUT/"DEEPSTEP_SEED_OVERLAP.csv",index=False)
print(df.to_string(index=False))
