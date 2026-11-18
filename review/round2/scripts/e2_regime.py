import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# E2 / P2: capture-regime effects. SD35L: pooled (seq 8192) vs branch-mean (4096) at matched domain x step.
import pickle, sys; sys.path.insert(0,'.'); from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold
rng=np.random.default_rng(3); out=[]
for c in ['SD35L','SD35M']:
    df,meta,ns,L=load(c,'attn'); gids,_,X,sig=pickle.load(open(f'F_{c}_attn.pkl','rb'))
    seq=df.groupby(['gid','token_idx']).seq_len.first().unstack().loc[gids].values  # G,T
    pooled=seq==seq.max()
    Y=X.copy(); Y[...,0]=np.log(Y[...,0]); Y[...,1]=np.log(Y[...,1])
    dom=np.asarray(meta.dom.astype(str)); pt=np.asarray(meta.pt.astype(str))
    # H2a: within (domain, step) cells containing both regimes, diff of layer-mean features pooled - branchmean
    d=[]
    for t in range(1,ns):
        for D_ in np.unique(dom):
            a=(dom==D_)&pooled[:,t]; b=(dom==D_)&~pooled[:,t]
            if a.sum()>=3 and b.sum()>=3:
                d.append(np.r_[Y[a,t].mean((0,1))-Y[b,t].mean((0,1)), a.sum(), b.sum(), t])
    d=np.array(d)
    if len(d):
        w=np.minimum(d[:,3],d[:,4])
        dl=np.average(d[:,:3],axis=0,weights=w)
        # shape r between regimes (depth-normalized profile) at matched cells, pooled
        print(c,'mixed cells',len(d),'weighted mean diff [dlog g, dlog gap, dH]',np.round(dl,4),'frac cells dlogg>0',np.mean(d[:,0]>0).round(2))
    # within-prompt comparison: SD35M step 1 has both regimes; also same prompt across seeds?
    # H2c: shape r between regimes: mean depthnorm profile pooled vs branch-mean at matched (domain, step)
    rs=[]
    for t in range(1,ns):
        for D_ in np.unique(dom):
            a=(dom==D_)&pooled[:,t]; b=(dom==D_)&~pooled[:,t]
            if a.sum()>=3 and b.sum()>=3:
                A=depthnorm(Y[a,t].mean(0)[None].copy()*0+X[a,t].mean(0)[None])[0]; B=depthnorm(X[b,t].mean(0)[None])[0]
                rs.append(shape_r(A,B))
    # noise-floor shape r: split one regime randomly within the same cells
    rn=[]
    for t in range(1,ns):
        for D_ in np.unique(dom):
            a=np.where((dom==D_)&pooled[:,t])[0]
            if len(a)>=6:
                p=rng.permutation(a); h=len(p)//2; rn.append(shape_r(depthnorm(X[p[:h],t].mean(0)[None])[0],depthnorm(X[p[h:],t].mean(0)[None])[0]))
    # H2b: domain decoding within regime-homogeneous subsets at matched steps (prompt-grouped CV)
    acc=[]
    for t in [ns//4, ns//2, 3*ns//4, ns-1]:
        for reg,mask in [('pooled',pooled[:,t]),('branchmean',~pooled[:,t])]:
            if mask.sum()<60 or len(np.unique(dom[mask]))<3: continue
            Z=Y[mask,t].reshape(mask.sum(),-1); y=dom[mask]; grp=pt[mask]
            k=min(5,len(np.unique(grp)))
            sc=cross_val_score(LogisticRegression(max_iter=3000,C=1.0),(Z-Z.mean(0))/(Z.std(0)+1e-9),y,cv=GroupKFold(k),groups=grp)
            acc.append(dict(ch=c,step=t,regime=reg,n=int(mask.sum()),n_dom=len(np.unique(y)),acc=sc.mean(),chance=1/len(np.unique(y)),ratio=sc.mean()*len(np.unique(y))))
    out.append(dict(ch=c,mixed_cells=len(d),dlogg=dl[0] if len(d) else np.nan,dloggap=dl[1] if len(d) else np.nan,dH=dl[2] if len(d) else np.nan,
                    shape_r_between_median=np.median(rs) if rs else np.nan,shape_r_between_min=np.min(rs) if rs else np.nan,shape_r_within_noise_median=np.median(rn) if rn else np.nan))
    A=pd.DataFrame(acc); print(A.round(3).to_string()); A.to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E2_decoding_{c}.csv',index=False)
O=pd.DataFrame(out); print(O.round(4).to_string()); O.to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E2_regime.csv',index=False)
