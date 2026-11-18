# Build per-generation LLM depth cubes for R15: median over generated tokens of (g_w64, gap_w64, H_w64) per (gen, layer, submodule).
import sys, zipfile, numpy as np, pandas as pd, pickle
z, name, out = sys.argv[1], sys.argv[2], sys.argv[3]
cols=['gen_idx','token_type','layer','submodule','g_w64','gap_w64','H_w64','prompt_text']
zf=zipfile.ZipFile(z); csvn=[n for n in zf.namelist() if n.endswith('.csv') and 'prompt_block' not in n][0]
parts=[]; pt={}
for ch in pd.read_csv(zf.open(csvn), usecols=cols, chunksize=2_000_000, low_memory=False):
    for g,t in ch.dropna(subset=['prompt_text'])[['gen_idx','prompt_text']].values: pt.setdefault(int(g),t)
    ch=ch[ch.token_type=='generated']
    parts.append(ch.groupby(['gen_idx','layer','submodule'])[['g_w64','gap_w64','H_w64']].agg(['sum','count']))
A=pd.concat(parts).groupby(level=[0,1,2]).sum()
M=pd.DataFrame({v:A[(v,'sum')]/A[(v,'count')] for v in ['g_w64','gap_w64','H_w64']})  # token mean (streaming); median not streamable
res={}
for sub in ['attn','mlp']:
    S=M.xs(sub,level='submodule'); gens=sorted(S.index.get_level_values(0).unique()); L=S.index.get_level_values(1).max()+1
    X=np.full((len(gens),1,L,3),np.nan)
    for i,g in enumerate(gens):
        r=S.loc[g]; X[i,0,r.index.values,:]=r.values
    ok=~np.isnan(X).any(axis=(1,2,3)); gens=[g for g,k in zip(gens,ok) if k]; X=X[ok]
    meta=pd.DataFrame({'pt':[pt.get(g,'') for g in gens]},index=gens)
    pickle.dump([gens,meta,X,np.array([0.5])],open(f'{out}/L_{name}_{sub}.pkl','wb')); res[sub]=X.shape
print(name,res)
