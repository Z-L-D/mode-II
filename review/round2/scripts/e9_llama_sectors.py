import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# E9 / P9: Llama-3.1-8B-Base dual hook. Does cross-type phi (stride-1) beat same-type phi_skip for held-out next-layer beta_g prediction?
import glob, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
df=pd.read_parquet(glob.glob('pq/LLM_Llama-3.1-8B-base_*.parquet')[0]); df=df[df.token_type=='generated']
print(df.submodule.value_counts().to_dict(), 'phi nan frac',df.phi_micro.isna().mean().round(3),df.phi_micro_skip.isna().mean().round(3))
rng=np.random.default_rng(9); rows=[]
for sub in ['attn','mlp']:
    d=df[df.submodule==sub].sort_values(['gen_idx','token_idx','layer']).copy()
    d['g_next']=d.groupby(['gen_idx','token_idx']).g_micro.shift(-1); d['beta']=d.g_next-d.g_micro
    d=d.dropna(subset=['beta','phi_micro','phi_micro_skip'])
    keep=rng.random(len(d))<min(1,200000/len(d)); d=d[keep]
    d['logn']=np.log(d.seq_len); d['depth']=d.layer
    base=['g_micro','spectral_gap','sv_entropy','depth','logn']
    y=d.beta.values; grp=d.gen_idx.values
    for name,feats in [('base',base),('+phi(cross,stride1)',base+['phi_micro']),('+phi_skip(same,stride2)',base+['phi_micro_skip']),('+both',base+['phi_micro','phi_micro_skip'])]:
        X=d[feats].values; pred=np.zeros_like(y)
        for tr,te in GroupKFold(5).split(X,groups=grp):
            pred[te]=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.08,random_state=0).fit(X[tr],y[tr]).predict(X[te])
        r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum(); rows.append(dict(sub=sub,model=name,R2=r2,n=len(y)))
        print(sub,name,round(r2,4),flush=True)
R=pd.DataFrame(rows); R.to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E9_llama_sectors.csv',index=False)
for sub,g in R.groupby('sub'):
    b=g.set_index('model').R2; print(sub,'gain cross %.4f  gain skip %.4f'%(b['+phi(cross,stride1)']-b['base'],b['+phi_skip(same,stride2)']-b['base']))
