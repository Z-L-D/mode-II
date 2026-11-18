import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# E7 / P7: context-invariant LLM observables. PR/n, g/g0 with g0=n/(n+D); Qwen size-pair depth derivative; 24-bin beta ladder.
import sys, glob; sys.path.insert(0,'.'); import numpy as np, pandas as pd
from scipy.stats import spearmanr
from beta24 import beta24, crossings, signs
rows=[]; Q={}
for f in sorted(glob.glob('pq/LLM_*.parquet')):
    name=f.split('LLM_')[1].split('_2026')[0]
    df=pd.read_parquet(f); df=df[df.token_type=='generated']
    if 'submodule' in df: df=df[df.submodule=='attn']
    D=df.hidden_dim.iloc[0]; n=df.seq_len.values
    df=df.assign(PRn=df.g_micro*D/n, gtil=df.g_micro/(n/(n+D)))
    m=df.groupby(['gen_idx','token_idx']).agg(n=('seq_len','first'),g=('g_micro','mean'),PRn=('PRn','mean'),gtil=('gtil','mean')).reset_index()
    rg=[];rp=[];rt=[]
    for _,d in m.groupby('gen_idx'):
        if len(d)>=20: rg.append(spearmanr(d.n,d.g)[0]); rp.append(spearmanr(d.n,d.PRn)[0]); rt.append(spearmanr(d.n,d.gtil)[0])
    rows.append(dict(model=name,D=D,gens=len(rg),rho_g=np.median(rg),rho_PRn=np.median(rp),abs_rho_PRn=np.median(np.abs(rp)),rho_gtilde=np.median(rt),frac_gtilde_decreasing=np.mean(np.array(rt)<0),
                     PRn_median=m.PRn.median(),n_median=m.n.median()))
    if 'qwen' in name: Q[name]=df
R=pd.DataFrame(rows); print(R.round(3).to_string()); R.to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E7_llm_context.csv',index=False)
# H7c: Qwen depth-derivative correlation at matched n
grid=np.linspace(0,1,24)
def profiles(df,var,bins):
    out=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        d=df[(df.seq_len>=lo)&(df.seq_len<hi)]
        p=d.groupby('layer')[var].mean(); y=np.log(p.values); u=np.linspace(0,1,len(y)); out.append(np.diff(np.interp(grid,u,y)))
    return np.array(out)
lo=max(d.seq_len.quantile(.05) for d in Q.values()); hi=min(d.seq_len.quantile(.95) for d in Q.values()); bins=np.linspace(lo,hi,9)
names=sorted(Q); pr=[]
for var in ['g_micro','gtil']:
    P={k:profiles(Q[k],var,bins) for k in names}
    for a,b in [('qwen25_1_5b','qwen25_7b'),('qwen25_3b','qwen25_7b'),('qwen25_1_5b','qwen25_3b')]:
        rs=[np.corrcoef(P[a][i],P[b][i])[0,1] for i in range(len(bins)-1)]
        pr.append(dict(obs=var,pair=f'{a}~{b}',median_r=np.median(rs),min_r=np.min(rs),max_r=np.max(rs)))
PR=pd.DataFrame(pr); print(PR.round(3).to_string()); PR.to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E7_qwen_pairs.csv',index=False)
# H7d: 24-bin beta ladder, raw g vs g/g0. crossing positions expressed as quantile of observable distribution
lad=[]
for k in names:
    d=Q[k]
    for var in ['g_micro','gtil']:
        piv=d.pivot_table(index=['gen_idx','token_idx'],columns='layer',values=var).values
        c,bm=beta24(piv); cr=crossings(c,bm); x=piv[:,:-1].ravel(); x=x[~np.isnan(x)]
        q=[(x<v).mean() for v,_ in cr]
        lad.append(dict(model=k,obs=var,signs=signs(bm),n_cross=len(cr),cross_values=';'.join(f'{v:.4g}' for v,_ in cr),cross_quantiles=';'.join(f'{v:.3f}' for v in q)))
L=pd.DataFrame(lad); print(L.to_string()); L.to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E7_qwen_beta_ladder.csv',index=False)
