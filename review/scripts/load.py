import pandas as pd, glob, os, re, sys
W='.'  # work dir: unzip TELEMETRY.zip into W/tel and each nested DM zip into W/tel/DM/<name>/
os.makedirs(W+'/pq',exist_ok=True)
for d in sorted(glob.glob(W+'/tel/DM/*/')):
    name=os.path.basename(d.rstrip('/'))
    dfs=[]
    for f in sorted(glob.glob(d+'*.csv')):
        df=pd.read_csv(f,low_memory=False)
        base=os.path.basename(f)
        m=re.search(r'-([a-z_]+)\.(\d+)\.csv$',base)
        df['file']=base
        df['domain']=m.group(1) if m and 'sd-3.5' not in base else ''
        df['part']=int(m.group(2)) if m else -1
        if 'sd-3.5' in base:
            df['res']=int(re.search(r'attn-(\d+)-30',base).group(1)); df['part']=int(base.split('.')[-2])
        df['prompt_text']=df['prompt_text'].ffill() if 'prompt_text' in df else None
        dfs.append(df)
    df=pd.concat(dfs,ignore_index=True)
    df.to_parquet(W+f'/pq/{name}.parquet')
    print(name,df.shape,flush=True)
for f in sorted(glob.glob(W+'/tel/LLM/*.csv')):
    df=pd.read_csv(f,low_memory=False)
    df.to_parquet(W+'/pq/LLM_'+os.path.basename(f)[:-4]+'.parquet')
    print(os.path.basename(f),df.shape,list(df.columns),flush=True)
# NOTE: MOCHI-CFG-4.5 portraits_figures.0.csv has no header row; re-read it with header=None and names from another file
# (see fix_mochi() below) or that domain is lost.
def fix_mochi(W):
    d=W+'/tel/DM/MOCHI-CFG-4.5-attn-480-120/'
    cols=pd.read_csv(d+'MOCHI-CFG-4.5-attn-480-120-abstract_conceptual.0.csv',nrows=1).columns
    dfs=[]
    for f in sorted(glob.glob(d+'*.csv')):
        hdr=open(f).readline().startswith('gen_idx')
        df=pd.read_csv(f,low_memory=False) if hdr else pd.read_csv(f,header=None,names=cols,low_memory=False)
        base=os.path.basename(f); m=re.search(r'-([a-z_]+)\.(\d+)\.csv$',base)
        df['file']=base; df['domain']=m.group(1); df['part']=0; df['prompt_text']=df['prompt_text'].ffill(); dfs.append(df)
    pd.concat(dfs,ignore_index=True).to_parquet(W+'/pq/MOCHI-CFG-4.5-attn-480-120.parquet')
fix_mochi(W)
