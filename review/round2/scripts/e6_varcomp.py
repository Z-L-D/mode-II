import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# E6 / P6: prompt vs seed share of within-(model, step) variance of A=[log g, log gap, H] across layers.
import pickle, sys; sys.path.insert(0,'.'); from common import *
CH=[('ZIT','attn'),('ZIT','mlp'),('ZIB','attn'),('ZIB','mlp'),('MOCHI','attn'),('MOCHI1','attn'),('SD35M','attn'),('SD35L','attn')]
rows=[]; summ=[]
for c,sub in CH:
    gids,meta,X,sig=pickle.load(open(f'F_{c}_{sub}.pkl','rb'))
    Y=X.copy(); Y[...,0]=np.log(Y[...,0]); Y[...,1]=np.log(Y[...,1])
    pt=np.asarray(meta.pt.astype(str)); up,inv=np.unique(pt,return_inverse=True)
    G,T,L,_=Y.shape; Yf=Y.reshape(G,T,-1)
    # standardize each feature within step so trace isn't dominated by one variable
    Yf=(Yf-Yf.mean(0))/(Yf.std(0)+1e-12)
    cnt=np.bincount(inv); k=len(up); n0=(G-(cnt**2).sum()/G)/(k-1)
    naive=[];anova=[]
    for t in range(T):
        Z=Yf[:,t]; mu=np.stack([Z[inv==i].mean(0) for i in range(k)])
        ssw=((Z-mu[inv])**2).sum(0); ssb=(cnt[:,None]*(mu-Z.mean(0))**2).sum(0)
        msw=ssw/(G-k); msb=ssb/(k-1); vp=np.maximum((msb-msw)/n0,0)
        naive.append(ssb.sum()/(ssb.sum()+ssw.sum())); anova.append(vp.sum()/(vp.sum()+msw.sum()))
    naive=np.array(naive); anova=np.array(anova); seed=1-anova
    for t in range(T): rows.append(dict(ch=f'{c}-{sub}',step=t,sigma=sig[t],prompt_share_naive=naive[t],prompt_share_anova=anova[t]))
    lo=max(1,int(0.1*T)); i_mx=seed.argmax(); i_mn=seed.argmin()
    def interior(i): return lo<=i<T-lo and min(abs(seed[i]-seed[0]),abs(seed[i]-seed[-1]))>=0.05
    nonmono=interior(i_mx) or interior(i_mn)
    summ.append(dict(ch=f'{c}-{sub}',G=G,prompts=k,seeds_per_prompt=round(n0,2),min_prompt_anova=anova.min(),min_excl_sigma1=anova[1:].min(),
                     max_prompt_anova=anova.max(),min_prompt_naive=naive.min(),seed_share_start=seed[0],seed_share_end=seed[-1],
                     seed_argmax=i_mx,seed_argmin=i_mn,T=T,seed_nonmonotone=nonmono))
S=pd.DataFrame(summ); print(S.round(3).to_string())
pd.DataFrame(rows).to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E6_varcomp_by_step.csv',index=False); S.to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E6_varcomp_summary.csv',index=False)
