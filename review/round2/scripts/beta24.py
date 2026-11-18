import numpy as np, pandas as pd
def pairs(df, steps=None, sub=None):
    d=df if sub is None else df[df.submodule==sub]
    if steps is not None: d=d[d.token_idx.isin(steps)]
    piv=d.pivot_table(index=['gid','token_idx'],columns='layer',values=['g_micro','spectral_gap','sv_entropy'])
    G=piv['g_micro'].values; GAP=piv['spectral_gap'].values; H=piv['sv_entropy'].values
    return G,GAP,H,piv.index.get_level_values(0).values
def beta24(G):
    g=G[:,:-1].ravel(); b=np.diff(G,axis=1).ravel(); m=~np.isnan(g)&~np.isnan(b); g,b=g[m],b[m]
    e=np.linspace(np.percentile(g,2),np.percentile(g,98),25); c=0.5*(e[:-1]+e[1:]); bm=np.full(24,np.nan)
    for i in range(24):
        k=(g>=e[i])&(g<e[i+1])
        if k.sum()>=5: bm[i]=b[k].mean()
    return c,bm
def crossings(c,bm):
    v=~np.isnan(bm); c,bm=c[v],bm[v]; out=[]
    for i in range(len(bm)-1):
        if bm[i]*bm[i+1]<0: out.append((c[i]-bm[i]*(c[i+1]-c[i])/(bm[i+1]-bm[i]), '-+' if bm[i]<0 else '+-'))
    return out
def TRI(c,bm):
    v=~np.isnan(bm); bm=bm[v]
    if bm[-1]>=0: return np.nan
    i=len(bm)-1
    while i>0 and bm[i-1]<0: i-=1
    br=bm[i:]; mn=br.min(); return (bm[-3:].mean()-mn)/abs(mn)
def signs(bm):
    s=np.sign(bm[~np.isnan(bm)]); return ''.join('+' if x>0 else '-' for x in s[np.r_[True,s[1:]!=s[:-1]]])
