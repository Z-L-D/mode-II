#!/usr/bin/env python3
"""Reproduce the Round-7 final release audit from the extracted Round-6 outputs.

Usage:
  python r7_final_release_audit.py --r6 /path/to/extracted/ModeSeries_Round6_outputs --out ./r7_repro

Requires: pandas, numpy, scipy, matplotlib.
"""
from pathlib import Path
import argparse, json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def pairkey(a,b): return "|".join(sorted([str(a),str(b)]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--r6", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    r6=Path(args.r6); out6=r6/"out"; out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    s1=pd.read_csv(out6/"S1_hierarchy.csv")
    s2=pd.read_csv(out6/"S2_rsa.csv")
    s3=pd.read_csv(out6/"S3_estimators.csv")
    s5=pd.read_csv(out6/"S5_sync.csv")
    a4=pd.read_csv(out6/"A4_beta_layer_surrogates.csv")
    score=pd.read_csv(out6/"R6C_scorecard.csv")

    line=s1[s1.kind=="lineage"].iloc[0]
    un=s1[s1.kind=="unrelated"]
    levels=["L0_D1","L0_D2","L1_field","L2_shape","L3_tangent"]
    margins={lv:float(line[lv]-un[lv].max()) for lv in levels}

    sx=s1.copy(); sx["pair"]=[pairkey(a,b) for a,b in zip(sx.a,sx.b)]
    rf=s2[s2.tau=="first"].copy(); rf["pair"]=[pairkey(a,b) for a,b in zip(rf.a,rf.b)]
    m=sx.merge(rf[["pair","rsa"]],on="pair")
    rho,p=spearmanr(m.L3_tangent,m.rsa)

    s3["s50_width"]=s3.matched60_s50_p95-s3.matched60_s50_p5
    s3["rel_width"]=s3.matched60_rel_p95-s3.matched60_rel_p5
    result={
      "margins":margins,
      "content_tangent_spearman":{"rho":float(rho),"p":float(p),"n":len(m)},
      "matched_n":{"sigma50_range":float(s3.matched60_s50_median.max()-s3.matched60_s50_median.min()),
                   "relative_onset_range":float(s3.matched60_rel_median.max()-s3.matched60_rel_median.min()),
                   "relative_interval_narrower":int((s3.rel_width<s3.s50_width).sum())},
      "sync":{"trend_ge_observed":int((s5.trend_R_peak>=s5.R_peak).sum()),
              "warp_p_lt_05":int((s5.warp_p<.05).sum())},
      "round6_scorecard":score.verdict.value_counts().to_dict()
    }
    (out/"R7_REPRO_RESULT.json").write_text(json.dumps(result,indent=2))
    pd.DataFrame({"level":levels,
                  "lineage":[line[x] for x in levels],
                  "best_unrelated":[un[x].max() for x in levels],
                  "margin":[margins[x] for x in levels]}).to_csv(out/"R7_REPRO_HIERARCHY.csv",index=False)
    m[["a","b","kind","L3_tangent","rsa"]].to_csv(out/"R7_REPRO_CONTENT_VS_DYNAMICS.csv",index=False)
    s3.to_csv(out/"R7_REPRO_ESTIMATORS.csv",index=False)
    a4.to_csv(out/"R7_REPRO_LAYER_NULLS.csv",index=False)
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
