import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import pandas as pd, numpy as np
P=f'{_COSPAFS_CACHE}/pq/'
CAMP={'ZIT':('ZIT-CFG-1.0-attn-1024-120',8),'ZIB':('ZIB-CFG-4.0-attn-1024-120',25),'MOCHI':('MOCHI-CFG-4.5-attn-480-120',30),
      'MOCHI1':('MOCHI-CFG-1.0-attn-480-120',100),'SD35M':('SD3.5MED-CFG-4.5-attn-1024-3TE-120',40),'SD35L':('SD3.5LAR-CFG-3.5-attn-1024-3TE-120',28),
      'SDRES':('SD3.5MED-CFG1.0-attn-512.768.1024-30',100)}
def load(c, sub=None):
    name,ns=CAMP[c]; df=pd.read_parquet(P+name+'.parquet')
    if 'submodule' not in df: df['submodule']='attn'
    if sub: df=df[df.submodule==sub]
    df=df.copy(); df['gid']=df.file+':'+df.gen_idx.astype(str)
    L=df.layer.nunique()
    cnt=df.groupby('gid').agg(n=('token_idx','nunique'),r=('layer','size'))
    nsub=df.submodule.nunique()
    ok=cnt[(cnt.n==ns)&(cnt.r==ns*L*nsub)].index
    df=df[df.gid.isin(ok)]
    meta=df.groupby('gid').agg(pt=('prompt_text','first'),dom=('domain','first'),ph=('prompt_hash','first'),seq=('seq_len','first'),pl=('prompt_len','first'),**({'res':('res','first')} if 'res' in df else {}))
    return df, meta, ns, L
def cube(df, var, ns, L, sub=None):
    """return gids, array [G, steps, layers]"""
    d=df if sub is None else df[df.submodule==sub]
    piv=d.pivot_table(index=['gid','token_idx'],columns='layer',values=var)
    gids=piv.index.get_level_values(0).unique()
    return list(gids), piv.values.reshape(len(gids),ns,L)
def fields(c, sub='attn'):
    df,meta,ns,L=load(c, sub)
    out={}
    for v in ['g_micro','spectral_gap','sv_entropy']:
        gids,X=cube(df,v,ns,L); out[v]=X
    sig=df[df.gid==gids[0]].groupby('token_idx').sigma.first().values
    X=np.stack([out['g_micro'],out['spectral_gap'],out['sv_entropy']],-1)  # G,T,L,3
    return gids, meta.loc[gids], X, sig
def syncR(X):  # X: T,L,3 ; z per layer over time
    Z=(X-X.mean(0,keepdims=True))/(X.std(0,keepdims=True)+1e-12)
    Z[...,1]*=-1
    N=X.shape[1]
    W2=((Z-Z.mean(1,keepdims=True))**2).sum(-1).mean(1)
    return 1-W2/(3*(1-1/N))
def depthnorm(M):  # M: T,L,3 -> z across layers per t,var ; uses log g
    Y=M.copy(); Y[...,0]=np.log(Y[...,0]); Y[...,1]=np.log(Y[...,1])
    return (Y-Y.mean(1,keepdims=True))/(Y.std(1,keepdims=True)+1e-12)
def shape_r(A,B):  # A,B: L,3 profiles (already depthnorm) -> pearson
    return np.corrcoef(A.ravel(),B.ravel())[0,1]
