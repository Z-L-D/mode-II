#!/usr/bin/env python3
from __future__ import annotations
import sys,csv,json
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from r4_common import *
OUT=Path("/mnt/data/r4_r5_audit_v016/results");OUT.mkdir(parents=True,exist_ok=True)

# Parse frozen prompt TSV robustly.
frozen=pd.read_csv("/mnt/data/matched_prompts_30(2).tsv",sep="\t")
fset=set(frozen["prompt"].astype(str).str.strip().str.replace(r'^"|"$','',regex=True).str.strip())

core,_=load_all(["SD3.5M-attn-TE3-CFG1.0-R1024-S1001-STUDY[CFG-RES-SEED-CORRECTION]",
                 "SD3.5M-attn-TE3-CFG1.0-R1024-S1002-STUDY[CFG-RES-SEED-CORRECTION]",
                 "SD3.5M-attn-TE3-CFG4.5-R1024-S1001-STUDY[CFG-RES-SEED-CORRECTION]",
                 "SD3.5M-attn-TE3-CFG4.5-R1024-S1002-STUDY[CFG-RES-SEED-CORRECTION]"])
cset=set(core.prompt_text_full.dropna().astype(str).str.strip())
protocol={
    "frozen_prompt_count":len(fset),"core_prompt_count":len(cset),
    "exact_set_match":fset==cset,
    "missing_from_core":sorted(fset-cset),
    "extra_in_core":sorted(cset-fset)
}

# D14/output linkage schema audit across all new files.
output_like=["output","image","filename","file_name","path","reward","score","quality","correct","label"]
schema=set()
for m in members():
    d,_=load_member(m);schema.update(map(str,d.columns))
hits=sorted([c for c in schema if any(k in c.lower() for k in output_like)])
d14={
    "schema_columns":sorted(schema),
    "output_or_quality_like_fields":hits,
    "has_stable_output_identifier":any(c.lower() in {"output_filename","image_filename","output_id","image_id","file_name","filename"} for c in schema),
    "has_quality_or_correctness_label":any(any(k in c.lower() for k in ["reward","quality","correct","score","label"]) for c in schema),
    "status":"BLOCKED" 
}
(OUT/"PROTOCOL_AND_D14_AUDIT.json").write_text(json.dumps({"prompt_protocol":protocol,"D14":d14},indent=2),encoding="utf-8")
print(json.dumps({"prompt_protocol":protocol,"D14":d14},indent=2))
