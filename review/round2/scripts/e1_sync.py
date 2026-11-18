import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# E1 / P1: sync R vs per-layer monotone time-warp null; centering-free K(t), C(t); K vs circular-shift null.
import pickle, sys; sys.path.insert(0,'.'); from common import *
rng=np.random.default_rng(2026)
CH=[('ZIT','attn'),('ZIT','mlp'),('ZIB','attn'),('ZIB','mlp'),('MOCHI','attn'),('SD35M','attn'),('SD35L','attn')]
def warp(M,alpha=1.0):
    T,L,_=M.shape; t=np.linspace(0,1,T); W=np.empty_like(M)
    for l in range(L):
        w=np.r_[0,np.cumsum(rng.dirichlet(alpha*np.ones(T-1)))]; w/=w[-1]
        for k in range(3): W[:,l,k]=np.interp(w,t,M[:,l,k])
    return W
def Kt(Y):  # Y: G,T,L,3 transformed; returns per-gen K(t) array G,T-1
    dY=np.diff(Y,axis=1); U=dY/np.sqrt((dY**2).mean(1,keepdims=True)+1e-15)
    return (np.abs(U.mean(2))/np.sqrt((U**2).mean(2)+1e-15)).mean(-1), U
def Ct(Y):  # mean off-diagonal cross-layer corr of increments across gens, per step, averaged over vars
    dY=np.diff(Y,axis=1); G,T1,L,_=dY.shape; out=np.zeros(T1)
    for t in range(T1):
        cs=[]
        for k in range(3):
            Z=dY[:,t,:,k]; Z=(Z-Z.mean(0))/(Z.std(0)+1e-15); C=Z.T@Z/G; cs.append((C.sum()-np.trace(C))/(L*(L-1)))
        out[t]=np.mean(cs)
    return out
def interior(curve,se):
    T=len(curve); i=curve.argmax(); tau=i/(T-1)
    return (0.2<=tau<=0.8) and curve[i]>curve[0]+se[i] and curve[i]>curve[-1]+se[i], tau
rows=[]; curves={}
for c,sub in CH:
    g,m,X,s=pickle.load(open(f'F_{c}_{sub}.pkl','rb')); G,T,L,_=X.shape
    M=X.mean(0); R=syncR(M); Robs=R.max()
    null1=np.array([syncR(warp(M,1.0)).max() for _ in range(999)]); null10=np.array([syncR(warp(M,10.0)).max() for _ in range(999)])
    Y=np.log(np.abs(X)+1e-12); Y[...,2]=X[...,2]
    K,U=Kt(Y); Km=K.mean(0); C=Ct(Y)
    # bootstrap over gens
    Kb=[];Cb=[]
    for _ in range(200):
        i=rng.integers(0,G,G); Kb.append(K[i].mean(0)); Cb.append(Ct(Y[i]))
    Kse=np.std(Kb,0); Cse=np.std(Cb,0)
    kint,ktau=interior(Km,Kse); cint,ctau=interior(C,Cse)
    # circular shift null for integrated K
    Kn=[]
    for _ in range(200):
        sh=rng.integers(0,U.shape[1],size=(G,L)); idx=(np.arange(U.shape[1])[None,:,None]-sh[:,None,:])%U.shape[1]
        Us=np.take_along_axis(U,idx[...,None].repeat(3,-1),axis=1); Kn.append((np.abs(Us.mean(2))/np.sqrt((Us**2).mean(2))).mean())
    rows.append(dict(ch=f'{c}-{sub}',T=T,R_obs=Robs,R_argmax_tau=R.argmax()/(T-1),null_a1_p95=np.percentile(null1,95),p_a1=(1+(null1>=Robs).sum())/1000,
                     null_a10_p95=np.percentile(null10,95),p_a10=(1+(null10>=Robs).sum())/1000,
                     K_argmax_tau=ktau,K_interior=kint,C_argmax_tau=ctau,C_interior=cint,K_int=Km.mean(),K_shift_null=np.mean(Kn),K_shift_p95=np.percentile(Kn,95)))
    curves[f'{c}-{sub}']=dict(R=R,K=Km,Kse=Kse,C=C,Cse=Cse,sigma=s)
    print(rows[-1])
D=pd.DataFrame(rows); print(D.round(3).to_string())
D.to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E1_sync.csv',index=False)
pickle.dump(curves,open('E1_curves.pkl','wb'))
