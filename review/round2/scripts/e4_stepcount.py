import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# E4 / P4a: ZIT 8-step vs 100-step at matched sigma (same prompts), CFG 1.0, cond-only.
import sys; sys.path.insert(0,'.'); from common import *
df=pd.read_parquet(P+'ZIT-CFG-1.0-attn-1024-120.parquet'); df['gid']=df.file+':'+df.gen_idx.astype(str)
n=df.groupby('gid').token_idx.nunique(); r=df.groupby('gid').size()
V=['g_micro','spectral_gap','sv_entropy']
def cubes(gids,ns):
    d=df[df.gid.isin(gids)]; out={}
    for sub in ['attn','mlp']:
        e=d[d.submodule==sub]
        piv=e.pivot_table(index=['gid','token_idx'],columns='layer',values=V)
        G=len(gids); L=e.layer.nunique()
        out[sub]=np.stack([piv[v].values.reshape(G,ns,L) for v in V],-1)
    sig=d[d.gid==gids[0]].groupby('token_idx').sigma.first().values
    return out,sig
g100=[g for g in n.index if n[g]==100 and r[g]==100*30*2] if False else [g for g in n.index if n[g]==100]
pts=df[df.gid.isin(g100)].groupby('gid').prompt_text.first()
L=df.layer.nunique()
g8=[g for g in n.index if n[g]==8 and r[g]==8*L*2]
pt8=df[df.gid.isin(g8)].groupby('gid').prompt_text.first()
g8m=[g for g in g8 if pt8[g] in set(pts.values)]
print('100-step gens',len(g100),'matched 8-step gens',len(g8m),'all 8-step',len(g8))
C100,s100=cubes(g100,100); C8m,s8=cubes(g8m,8); C8,_=cubes(g8,8)
print('sigma 8:',np.round(s8,4)); print('sigma100 head/tail',np.round(s100[:3],4),np.round(s100[-3:],4))
rows=[]
for sub in ['attn','mlp']:
    A=C100[sub].mean(0)  # T100,L,3
    # interpolate in sigma (100-step sigma is decreasing)
    Ai=np.stack([[np.interp(s8,s100[::-1],A[::-1,l,k]) for l in range(L)] for k in range(3)],-1).transpose(1,0,2) # T8,L,3
    for ref,B in [('matched',C8m[sub].mean(0)),('all',C8[sub].mean(0))]:
        rel=np.abs(Ai/B-1)  # T8,L,3
        for k,v in enumerate(V):
            rows.append(dict(sub=sub,ref=ref,var=v,median=np.median(rel[...,k]),p90=np.percentile(rel[...,k],90),
                             median_excl_sigma1=np.median(rel[1:,:,k]),max_step_median=np.median(rel[...,k],1).max()))
        if ref=='matched':
            per_step=np.median(rel,1)
            for t in range(8): print(sub,'step',t,'sigma',round(s8[t],3),'median rel dev g/gap/H',np.round(per_step[t],4))
R=pd.DataFrame(rows); print(R.round(4).to_string())
R.to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E4_stepcount.csv',index=False)
# late downturn below sigma 0.2 (prior note): 100-step layer-mean g at sigma<0.2
for sub in ['attn','mlp']:
    A=C100[sub].mean(0)[...,0].mean(1)
    m=s100<0.25; print(sub,'100-step layer-mean g for sigma<0.25:',np.round(s100[m][::4],3),np.round(A[m][::4],5))
    print(sub,'8-step layer-mean g:',np.round(C8m[sub].mean(0)[...,0].mean(1),5))
