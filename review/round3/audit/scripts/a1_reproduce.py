import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Audit step 1: recompute the reviewer's v0.13/v0.14 headline numbers with Claude's own Round 2 estimators.
# No new analyses here; every quantity below was reported by the reviewer.
import pickle, sys, os, json, numpy as np, pandas as pd
sys.path.insert(0,f'{_COSPAFS_ROOT}/review/round2/scripts'); sys.path.insert(0,'.')
from common import depthnorm, shape_r
from beta24 import beta24, crossings, TRI, signs
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
OUT=f'{_COSPAFS_ROOT}/review/round3/audit/out'; res={}
def F(name): return pickle.load(open(name,'rb'))
def arm(a,sub='attn'):
    if '+' in a:
        parts=[F(f'F3_{x}_{sub}.pkl') for x in a.split('+')]
        g=sum([p[0] for p in parts],[]); m=pd.concat([p[1] for p in parts]); X=np.concatenate([p[2] for p in parts]); s=parts[0][3]
        return g,m,X,s
    return F(f'F3_{a}_{sub}.pkl')
S10='S10_1024_s1001+S10_1024_s1002'; S45='S45_1024_s1001+S45_1024_s1002'
# ---- B8a tangent (E10 estimator) ----
gr=np.linspace(0,1,24)
def prof(X,s):
    D=depthnorm(X.mean(0)); u=np.linspace(0,1,D.shape[1])
    return np.stack([[np.interp(gr,u,D[t,:,k]) for k in range(3)] for t in range(len(s))]).transpose(0,2,1)
def cos(a,b): a=a.ravel(); b=b.ravel(); return a@b/np.linalg.norm(a)/np.linalg.norm(b)
def tangent(XA,sa,XB,sb):
    A=prof(XA,sa); B=prof(XB,sb); tc=[]; sr=[]
    for t,x in enumerate(sa):
        j=int(np.argmin(np.abs(sb-x))); tc.append(cos(np.diff(A[t],axis=0),np.diff(B[j],axis=0))); sr.append(shape_r(A[t],B[j]))
    return np.mean(tc), np.mean(sr)
for sub in ['attn','mlp']:
    _,_,Xt,st=F(f'F_ZIT_{sub}.pkl'); _,_,Xz,sz=arm('Z10_1024_s1001',sub)
    tc,sr=tangent(Xt,st,Xz,sz); res[f'B8a_tangent_{sub}']=tc; res[f'B8a_shape_r_{sub}']=sr
    # B8c post hoc (reviewer): median |dlog g| new ZIB1<->ZIT over old ZIB4(pooled archive)<->ZIT
    _,_,Xo,so=F(f'F_ZIB_{sub}.pkl')
    def dlog(XA,sa,XB,sb):
        ga=np.log(XA[...,0].mean(0)); gb=np.log(XB[...,0].mean(0)); return np.median([np.abs(ga[t]-gb[int(np.argmin(np.abs(sb-x)))]) for t,x in enumerate(sa)])
    res[f'B8c_posthoc_ratio_{sub}']=dlog(Xt,st,Xz,sz)/dlog(Xt,st,Xo,so)
# ---- B2a onset (E3 estimator) ----
def onset(X,s):
    Y=X.mean(0).copy(); Y[...,0]=np.log(Y[...,0]); Y[...,1]=np.log(Y[...,1]); A=Y.reshape(len(s),-1)
    sp=np.linalg.norm(np.diff(A,axis=0),axis=1)/np.abs(np.diff(s)); cm=(s[1:]+s[:-1])/2
    plat=np.median(sp[cm<0.6]); i=int(np.argmax(sp<2*plat)); return cm[i]
for a,sub in [('Z10_1024_s1001','attn'),('Z10_1024_s1001','mlp'),(S10,'attn')]:
    _,_,X,s=arm(a,sub); res[f'B2a_onset_{a}_{sub}']=onset(X,s)
# ---- E6 seed share (B3) ----
def seedshare(X,m):
    Y=X.copy(); Y[...,0]=np.log(Y[...,0]); Y[...,1]=np.log(Y[...,1])
    pt=np.asarray(m.ph.astype(str)); up,inv=np.unique(pt,return_inverse=True)
    G,T,L,_=Y.shape; Yf=Y.reshape(G,T,-1); Yf=(Yf-Yf.mean(0))/(Yf.std(0)+1e-12)
    cnt=np.bincount(inv); k=len(up); n0=(G-(cnt**2).sum()/G)/(k-1); out=[]
    for t in range(T):
        Z=Yf[:,t]; mu=np.stack([Z[inv==i].mean(0) for i in range(k)])
        ssw=((Z-mu[inv])**2).sum(0); ssb=(cnt[:,None]*(mu-Z.mean(0))**2).sum(0)
        msw=ssw/(G-k); msb=ssb/(k-1); vp=np.maximum((msb-msw)/n0,0); out.append(1-vp.sum()/(vp.sum()+msw.sum()))
    return np.array(out)
for a in [S10,S45]:
    g,m,X,s=arm(a); ss=seedshare(X,m); key='CFG1' if a==S10 else 'CFG45'
    res[f'B3_seedshare_{key}_step0']=ss[0]; res[f'B3_seedshare_{key}_step1']=ss[1]; res[f'B3b_range_steps>=2_{key}']=ss[2:].max()-ss[2:].min()
    res[f'B3b_range_steps>=1_{key}']=ss[1:].max()-ss[1:].min()
# ---- residuals for VR, seed fate, seed identity ----
def resid(X,m):
    pt=np.asarray(m.ph.astype(str)); grp=pd.factorize(pt)[0]
    Y=np.log(np.abs(X)+1e-12); Y[...,2]=X[...,2]; G,T,L,_=Y.shape; Fz=Y.reshape(G,T,-1); Fz=(Fz-Fz.mean(0))/(Fz.std(0)+1e-12); r=np.empty_like(Fz)
    for t in range(T): r[:,t]=Fz[:,t]-pd.DataFrame(Fz[:,t]).groupby(grp).transform('mean').to_numpy()
    return r,grp,Fz
g,m,X,s=arm(S10); r,grp,Fz=resid(X,m)
rr=r[:,1:]; v1=((rr[:,1:]-rr[:,:-1])**2).mean()
for k in [2,4,8]: res[f'VR{k}_CFG1']=((rr[:,k:]-rr[:,:-k])**2).mean()/(k*v1)
def ridge_cv_r2(Xs,tgt,grp,alpha=100.0):
    pred=np.zeros_like(tgt)
    for tr,te in GroupKFold(5).split(tgt,groups=grp):
        A=Xs[tr]; mu=A.mean(0); B=tgt[tr]; mb=B.mean(0)
        W=np.linalg.solve((A-mu).T@(A-mu)+alpha*np.eye(A.shape[1]),(A-mu).T@(B-mb)); pred[te]=(Xs[te]-mu)@W+mb
    return 1-((tgt-pred)**2).sum()/((tgt-tgt.mean(0))**2).sum()
def s50(r,grp,s):
    T=r.shape[1]; c=np.array([ridge_cv_r2(r[:,t],r[:,-1],grp) for t in range(T)]); tau=np.arange(T)/(T-1)
    i=int(np.argmax(c>0.5)) if (c>0.5).any() else None
    if i is None: return np.nan,c
    t50=tau[0] if i==0 else tau[i-1]+(0.5-c[i-1])*(tau[i]-tau[i-1])/(c[i]-c[i-1]); return float(np.interp(t50,tau,s)),c
for a,key in [(S10,'CFG1'),(S45,'CFG45')]:
    g,m,X,s=arm(a); r,grp,Fz=resid(X,m); v,c=s50(r,grp,s); res[f'sigma50_{key}']=v; res[f'R2_step0_{key}']=c[0]
    # seed identity at the first state from held-out prompts (raw standardized features, step 0)
    y=(m.seed.values==1001).astype(int); acc=[]
    for tr,te in GroupKFold(5).split(Fz[:,0],groups=grp):
        clf=LogisticRegression(max_iter=2000,C=1.0).fit(Fz[tr,0],y[tr]); acc.append((clf.predict(Fz[te,0])==y[te]).mean())
    res[f'seed_identity_acc_step0_{key}']=float(np.mean(acc))
# ---- B7 topology / TRI (frozen 24-bin estimator) ----
topo=[]
for a,lab in [(S45,'A1_CFG45_1024'),(S10,'A2_CFG1_1024'),('S45_512_s1001','A3_CFG45_512'),('S45_768_s1001','A4_CFG45_768')]:
    g,m,X,s=arm(a); G=X[...,0].reshape(-1,X.shape[2]); c,bm=beta24(G); cr=crossings(c,bm)
    topo.append(dict(arm=lab,gens=X.shape[0],signs=signs(bm),n_cross=len(cr),crossings=';'.join(f'{x:.5f}{d}' for x,d in cr),TRI=TRI(c,bm)))
T=pd.DataFrame(topo); print(T.to_string()); T.to_csv(f'{OUT}/A1_topology_repro.csv',index=False)
# ---- B6 spread ratio by quarter (pooled CFG4.5 vs cond-only CFG1; post hoc) ----
def lm(a):
    g,m,X,s=arm(a); lg=np.log(X[...,0]).mean(2); d=pd.DataFrame(lg); d['ph']=m.ph.values; d['seed']=m.seed.values
    return d.sort_values(['ph','seed']),s
d1,s=lm(S10); d4,_=lm(S45); ph=sorted(set(d1.ph)&set(d4.ph)); T_=len(s)
sd1=d1.groupby('ph')[list(range(T_))].std().loc[ph].values; sd4=d4.groupby('ph')[list(range(T_))].std().loc[ph].values
ratio=sd4/sd1; q=np.array_split(np.arange(T_),4)
for i,qq in enumerate(q): res[f'B6_ratio_quarter{i+1}']=float(np.median(ratio[:,qq]))
res['B6_median_last3q']=float(np.median(ratio[:,np.concatenate(q[1:])]))
for k,v in res.items(): print(f'{k:40s} {v:.6f}' if isinstance(v,float) or isinstance(v,np.floating) else f'{k:40s} {v}')
json.dump({k:float(v) for k,v in res.items()},open(f'{OUT}/A1_reproduce.json','w'),indent=1)
