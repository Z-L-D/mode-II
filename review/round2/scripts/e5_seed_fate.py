import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# E5 / P5: seed-fate R2(t) vs diffusive (increment-shuffle) surrogate; jump size; prompt-cluster bootstrap CI of tau(R2=0.5).
import pickle, sys; sys.path.insert(0,'.'); from common import *
from sklearn.model_selection import GroupKFold
rng=np.random.default_rng(5)
def ridge_cv_r2(Xs,tgt,grp,alpha=100.0):
    pred=np.zeros_like(tgt)
    for tr,te in GroupKFold(5).split(tgt,groups=grp):
        A=Xs[tr]; mu=A.mean(0); B=tgt[tr]; mb=B.mean(0)
        W=np.linalg.solve((A-mu).T@(A-mu)+alpha*np.eye(A.shape[1]),(A-mu).T@(B-mb)); pred[te]=(Xs[te]-mu)@W+mb
    return 1-((tgt-pred)**2).sum()/((tgt-tgt.mean(0))**2).sum()
def curve(res,grp,alpha):
    tgt=res[:,-1]; return np.array([ridge_cv_r2(res[:,t],tgt,grp,alpha) for t in range(res.shape[1])])
def cross(c,tau):
    i=np.argmax(c>0.5) if (c>0.5).any() else None
    if i is None: return np.nan
    if i==0: return tau[0]
    return tau[i-1]+(0.5-c[i-1])*(tau[i]-tau[i-1])/(c[i]-c[i-1])
rows=[]; curves={}
for c,sub in [('ZIT','attn'),('ZIT','mlp'),('ZIB','attn'),('ZIB','mlp'),('MOCHI','attn'),('SD35M','attn'),('SD35L','attn')]:
    g,m,X,s=pickle.load(open(f'F_{c}_{sub}.pkl','rb'))
    pt=np.asarray(m.pt.astype(str)); grp=pd.factorize(pt)[0]
    Y=np.log(np.abs(X)+1e-12); Y[...,2]=X[...,2]; G,T,L,_=Y.shape; F=Y.reshape(G,T,-1)
    Fz=(F-F.mean(0))/(F.std(0)+1e-12); res=np.empty_like(Fz)
    for t in range(T): res[:,t]=Fz[:,t]-pd.DataFrame(Fz[:,t]).groupby(grp).transform('mean').to_numpy()
    alpha=100.0; tau=np.arange(T)/(T-1)
    obs=curve(res,grp,alpha)
    inc=np.diff(res,axis=1); sur=[]
    for _ in range(20):
        P=np.stack([inc[rng.permutation(G),t] for t in range(T-1)],1)
        rs=np.concatenate([res[:,:1],res[:,:1]+np.cumsum(P,1)],1); sur.append(curve(rs,grp,alpha))
    sur=np.array(sur); mid=int(round(0.5*(T-1)))
    jumps=np.diff(obs); jmax_excl=jumps[1:].max() if len(jumps)>1 else np.nan
    ug=np.unique(grp); bt=[]
    for _ in range(100):
        sel=rng.choice(ug,len(ug)); idx=np.concatenate([np.where(grp==u)[0] for u in sel])
        gg=np.concatenate([np.full((grp==u).sum(),k) for k,u in enumerate(sel)])
        bt.append(cross(curve(res[idx],gg,alpha),tau))
    bt=np.array(bt); lo,hi=np.nanpercentile(bt,[2.5,97.5])
    t50=cross(obs,tau); s50=np.interp(t50,tau,s) if not np.isnan(t50) else np.nan
    rows.append(dict(ch=f'{c}-{sub}',T=T,R2_mid_obs=obs[mid],R2_mid_sur=sur[:,mid].mean(),R2_mid_sur_p95=np.percentile(sur[:,mid],95),beats=obs[mid]>np.percentile(sur[:,mid],95),
                     first_jump=jumps[0],max_jump_excl_first=jmax_excl,tau50=t50,sigma50=s50,lambda50=np.log((1-s50)**2/s50**2) if s50==s50 else np.nan,
                     tau50_lo=lo,tau50_hi=hi,ci_width=hi-lo,boot_nan_frac=np.isnan(bt).mean()))
    curves[f'{c}-{sub}']=dict(obs=obs,sur=sur.mean(0),sigma=s); print(rows[-1],flush=True)
D=pd.DataFrame(rows); print(D.round(3).to_string()); D.to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E5_seed_fate.csv',index=False)
pickle.dump(curves,open('E5_curves.pkl','wb'))
