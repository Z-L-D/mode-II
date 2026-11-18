#!/usr/bin/env python3
from __future__ import annotations
import argparse, zipfile, io
from pathlib import Path
import numpy as np
import pandas as pd

LEGACY_MEMBER='DM/SD3.5MED-CFG1.0-attn-512.768.1024-30.zip'

def load_legacy(telemetry,res):
    with zipfile.ZipFile(telemetry) as outer:
        raw=outer.read(LEGACY_MEMBER)
    rows=[]
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            if name.endswith('.csv') and f'-{res}-' in name:
                d=pd.read_csv(io.BytesIO(z.read(name)),low_memory=False)
                d['run_id']=name
                rows.append(d)
    return pd.concat(rows,ignore_index=True)

def load_new_schedule(new_zip):
    with zipfile.ZipFile(new_zip) as z:
        names=[n for n in z.namelist() if n.lower().endswith('.csv')]
        d=pd.read_csv(io.BytesIO(z.read(names[0])),low_memory=False)
    return d.groupby('token_idx').sigma.mean().sort_index().to_numpy(float)

def make_beta(df):
    d=df.sort_values(['run_id','gen_idx','token_idx','layer']).copy()
    d['beta']=d.groupby(['run_id','gen_idx','token_idx'])['g_micro'].shift(-1)-d['g_micro']
    return d[d.beta.notna()].copy()

def binned_beta(points,bins=24,min_per_bin=50):
    x=points.g_micro.to_numpy(float); y=points.beta.to_numpy(float)
    edges=np.linspace(x.min(),x.max(),bins+1)
    bi=np.clip(np.digitize(x,edges)-1,0,bins-1)
    rows=[]
    for i in range(bins):
        m=bi==i
        if m.sum()>=min_per_bin:
            rows.append([i,float(x[m].mean()),float(y[m].mean()),int(m.sum())])
    return pd.DataFrame(rows,columns=['bin','g','beta','n'])

def crossing_list(curve):
    out=[]
    for i in range(len(curve)-1):
        y0,y1=curve.beta.iloc[i],curve.beta.iloc[i+1]
        if y0*y1<0:
            x0,x1=curve.g.iloc[i],curve.g.iloc[i+1]
            out.append(float(x0-y0*(x1-x0)/(y1-y0)))
    return out

def sign_sequence(curve):
    s=np.sign(curve.beta.to_numpy(float)); s=s[s!=0]
    if not len(s): return ''
    seq=['+' if s[0]>0 else '-']
    for v in s[1:]:
        c='+' if v>0 else '-'
        if c!=seq[-1]: seq.append(c)
    return ''.join(seq)

def topology(df):
    c=binned_beta(make_beta(df)); cr=crossing_list(c)
    return sign_sequence(c),len(cr),cr

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--telemetry',default='/mnt/data/TELEMETRY.zip')
    ap.add_argument('--new-cfg1',default='/mnt/data/r3_work/additional/SD3.5M-attn-TE3-CFG1.0-R1024-S1001-STUDY[CFG-RES-SEED]-G60.zip')
    ap.add_argument('--r3-topology',default='/mnt/data/r3_round3_release_v013/results/R3_posthoc_beta_topology.csv')
    ap.add_argument('--outdir',default='/mnt/data/r3_b7_correction_v014')
    a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    newsig=load_new_schedule(a.new_cfg1); newmin=float(newsig.min())
    rows=[]
    for res in [512,768,1024]:
        d=load_legacy(a.telemetry,res)
        for label,dd in [('legacy_100_full',d),('legacy_100_truncated_to_new_sigma_range',d[d.sigma>=newmin-1e-12].copy())]:
            seq,n,cr=topology(dd)
            rows.append([res,label,dd.token_idx.nunique(),float(dd.sigma.min()),float(dd.sigma.max()),seq,n,';'.join(map(str,cr))])
        oldsig=d.groupby('token_idx').sigma.mean(); toks=[]
        for s in newsig:
            t=int((oldsig-s).abs().idxmin())
            if t not in toks:toks.append(t)
        dm=d[d.token_idx.isin(toks)].copy(); seq,n,cr=topology(dm)
        rows.append([res,'legacy_100_thinned_to_40_sigma_points',len(toks),float(dm.sigma.min()),float(dm.sigma.max()),seq,n,';'.join(map(str,cr))])
    pd.DataFrame(rows,columns=['resolution','condition','n_observed_steps','sigma_min','sigma_max','sign_sequence','n_crossings','crossing_g']).to_csv(out/'R3_B7_STEP_COUNT_REAUDIT.csv',index=False)
    r3=pd.read_csv(a.r3_topology)
    r3[r3.arm.str.contains('SD35',na=False)].to_csv(out/'R3_B7_NEW_40STEP_PASSIVE_REFERENCE.csv',index=False)
if __name__=='__main__': main()
