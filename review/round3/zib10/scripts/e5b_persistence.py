import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# E5 follow-up (post hoc, unscored): persistence index Var(res_T - res_0) / sum_t Var(increment_t); <1 = anti-persistent (mean-reverting) seed deviations.
import pickle, sys; sys.path.insert(0,'.'); from common import *
rows=[]
for c,sub in [('ZIT','attn'),('ZIT','mlp'),('ZIB','attn'),('ZIB','mlp'),('MOCHI','attn'),('SD35M','attn'),('SD35L','attn')]:
    g,m,X,s=pickle.load(open(f'F_{c}_{sub}.pkl','rb')); pt=np.asarray(m.pt.astype(str)); grp=pd.factorize(pt)[0]
    Y=np.log(np.abs(X)+1e-12); Y[...,2]=X[...,2]; G,T,L,_=Y.shape; F=Y.reshape(G,T,-1); Fz=(F-F.mean(0))/(F.std(0)+1e-12); res=np.empty_like(Fz)
    for t in range(T): res[:,t]=Fz[:,t]-pd.DataFrame(Fz[:,t]).groupby(grp).transform('mean').to_numpy()
    inc=np.diff(res,axis=1); tot=((res[:,-1]-res[:,0])**2).sum(); summ=(inc**2).sum()
    # lag-1 increment correlation (pooled over features)
    a=inc[:,:-1].ravel(); b=inc[:,1:].ravel(); r1=np.corrcoef(a,b)[0,1]
    # same excluding first increment
    tot2=((res[:,-1]-res[:,1])**2).sum(); summ2=(inc[:,1:]**2).sum()
    rows.append(dict(ch=f'{c}-{sub}',persistence=tot/summ,persistence_excl_first=tot2/summ2,lag1_inc_corr=r1,var_res0=(res[:,0]**2).mean(),var_resT=(res[:,-1]**2).mean()))
D=pd.DataFrame(rows); print(D.round(3).to_string()); D.to_csv(f'{_COSPAFS_ROOT}/review/round3/zib10/out/E5b_persistence.csv',index=False)
# variance-ratio test VR(k)=Var(res_{t+k}-res_t)/(k Var(res_{t+1}-res_t)), steps >=1 (excludes sigma=1 transient).
# Pure jitter around a fixed level: VR(k)=1/k. Random walk: 1. Jitter + walk: rises toward 1 with k. Mean-reverting dynamics: falls with k.
print('variance ratio VR(k), k=1,2,4,8')
vr=[]
for c,sub in [('ZIB','attn'),('ZIB','mlp'),('MOCHI','attn'),('SD35M','attn'),('SD35L','attn'),('ZIT','attn'),('ZIT','mlp')]:
    g,m,X,s=pickle.load(open(f'F_{c}_{sub}.pkl','rb')); pt=np.asarray(m.pt.astype(str)); grp=pd.factorize(pt)[0]
    Y=np.log(np.abs(X)+1e-12); Y[...,2]=X[...,2]; G,T,L,_=Y.shape; F=Y.reshape(G,T,-1); Fz=(F-F.mean(0))/(F.std(0)+1e-12); res=np.empty_like(Fz)
    for t in range(T): res[:,t]=Fz[:,t]-pd.DataFrame(Fz[:,t]).groupby(grp).transform('mean').to_numpy()
    r=res[:,1:]; v1=((r[:,1:]-r[:,:-1])**2).mean()
    out=[((r[:,k:]-r[:,:-k])**2).mean()/(k*v1) for k in [1,2,4,8] if k<r.shape[1]]
    vr.append(dict(ch=f'{c}-{sub}',**{f'VR{k}':v for k,v in zip([1,2,4,8],out)})); print(vr[-1])
pd.DataFrame(vr).to_csv(f'{_COSPAFS_ROOT}/review/round3/zib10/out/E5c_variance_ratio.csv',index=False)
