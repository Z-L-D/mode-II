import pickle
from common import *
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
print('Passive commitment proxy: R2 of predicting final-state within-prompt residual (seed deviation) from residual at step t; prompt-grouped CV. Also prompt-ICC(t) and same-prompt retrieval(t).')
for c,sub in [('ZIT','attn'),('ZIT','mlp'),('ZIB','attn'),('ZIB','mlp'),('MOCHI','attn'),('SD35M','attn'),('SD35L','attn')]:
    g,m,X,s=pickle.load(open(f'F_{c}_{sub}.pkl','rb'))
    pt=np.asarray(m.pt.astype(str)); grp=pd.factorize(pt)[0]
    Y=np.log(np.abs(X)+1e-12); Y[...,2]=X[...,2]; G,T,L,_=Y.shape; F=Y.reshape(G,T,-1)
    Fz=(F-F.mean(0))/(F.std(0)+1e-12)
    res=np.empty_like(Fz)
    for t in range(T): res[:,t]=Fz[:,t]-pd.DataFrame(Fz[:,t]).groupby(grp).transform('mean').to_numpy()
    tgt=res[:,-1]; out=[]; icc=[]; ret=[]
    for t in range(T):
        pred=np.zeros_like(tgt)
        for tr,te in GroupKFold(5).split(tgt,groups=grp):
            mdl=RidgeCV(alphas=np.logspace(-1,4,12)).fit(res[tr,t],tgt[tr]); pred[te]=mdl.predict(res[te,t])
        out.append(1-((tgt-pred)**2).sum()/((tgt)**2).sum())
        tot=Fz[:,t].var(0).sum(); within=res[:,t].var(0).sum(); icc.append(1-within/tot)
        D=((Fz[:,t,None]-Fz[None,:,t])**2).sum(-1); np.fill_diagonal(D,np.inf); ret.append((grp[D.argmin(1)]==grp).mean())
    out=np.array(out); sn=(np.arange(T))/(T-1)
    t50=sn[np.argmax(out>0.5)] if (out>0.5).any() else np.nan
    k=max(1,T//8)
    print(f'{c:6s}{sub:5s} seed-fate R2(t): '+' '.join(f'{v:.2f}' for v in out[::k])+f' | first t/T with R2>0.5: {t50:.2f}')
    print(f'             prompt share of variance: '+' '.join(f'{v:.2f}' for v in np.array(icc)[::k])+'   retrieval: '+' '.join(f'{v:.2f}' for v in np.array(ret)[::k]),flush=True)
