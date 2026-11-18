# Context clock in all LLMs; PR vs n; Mistral instruct vs ablit divergence onset.
import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
for f in sorted(glob.glob('pq/LLM_*.parquet')):
    df=pd.read_parquet(f); df=df[df.token_type=='generated']
    if 'submodule' in df: df=df[df.submodule=='attn']
    m=df.groupby(['gen_idx','token_idx']).agg(seq=('seq_len','first'),g=('g_micro','mean'),H=('sv_entropy','mean'),D=('hidden_dim','first')).reset_index()
    r=[spearmanr(d.seq,d.g)[0] for _,d in m.groupby('gen_idx') if len(d)>=20]
    print(f, 'median rho(seq,g) %.4f'%np.nanmedian(r), 'median PR/n %.3f'%np.median(m.g*m.D/m.seq))
a=pd.read_parquet(glob.glob('pq/LLM_mistral-03-7B-instruct_*.parquet')[0]); b=pd.read_parquet(glob.glob('pq/LLM_mistral-03-7B-instruct-ablit_*.parquet')[0])
mm=a.merge(b,on=['gen_idx','token_idx','token_type','layer'],suffixes=('_a','_b')); mm['rd']=np.abs(np.log(mm.g_micro_a/mm.g_micro_b))
t=mm.groupby(['gen_idx','token_type','token_idx']).rd.median().reset_index(); thr=max(t[t.token_type=='prefill'].rd.quantile(.99)*5,1e-3)
fd=t[(t.token_type=='generated')&(t.rd>thr)].groupby('gen_idx').token_idx.min(); print('ablit divergence onset',fd.describe().round(1).to_dict())
