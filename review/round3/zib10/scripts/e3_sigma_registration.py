import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# E3 / P3: per-prompt equal-sigma vs optimized monotone warp (ZIT 8 -> ZIB 25); plateau-onset clustering in log-SNR vs step fraction.
import pickle, sys; sys.path.insert(0,'.'); from common import *
def mono_dp(C):  # max sum over non-decreasing j(i)
    n,m=C.shape; V=np.full((n,m),-np.inf); B=np.zeros((n,m),int); V[0]=C[0]
    for i in range(1,n):
        cm=np.maximum.accumulate(V[i-1]); arg=np.array([np.argmax(V[i-1][:j+1]) for j in range(m)])
        V[i]=cm+C[i]; B[i]=arg
    j=int(V[-1].argmax()); path=[j]
    for i in range(n-1,0,-1): j=B[i,j]; path.append(j)
    return path[::-1], V[-1].max()/n
rows=[]
for sub in ['attn','mlp']:
    gt,mt,Xt,st=pickle.load(open(f'F_ZIT_{sub}.pkl','rb')); gb,mb,Xb,sb=pickle.load(open(f'F_ZIB_{sub}.pkl','rb'))
    ptt=np.asarray(mt.pt.astype(str)); ptb=np.asarray(mb.pt.astype(str)); sh=sorted(set(ptt)&set(ptb))
    eq=[int(np.argmin(np.abs(sb-x))) for x in st]
    for p in sh:
        Dt=depthnorm(Xt[ptt==p].mean(0)); Db=depthnorm(Xb[ptb==p].mean(0))
        C=np.array([[shape_r(Dt[i],Db[j]) for j in range(len(sb))] for i in range(len(st))])
        path,ropt=mono_dp(C); req=np.mean([C[i,j] for i,j in enumerate(eq)])
        rows.append(dict(sub=sub,prompt=p[:60],r_eq=req,r_opt=ropt,loss=ropt-req,path=' '.join(map(str,path)),max_dev_steps=max(abs(a-b) for a,b in zip(path,eq))))
R=pd.DataFrame(rows); R.to_csv(f'{_COSPAFS_ROOT}/review/round3/zib10/out/E3_registration_per_prompt.csv',index=False)
for sub,g in R.groupby('sub'):
    print(sub,'n',len(g),'frac loss<=0.01 %.3f'%(g.loss<=0.01).mean(),'median loss %.4f p90 %.4f max %.4f'%(g.loss.median(),g.loss.quantile(.9),g.loss.max()),'median r_eq %.3f'%g.r_eq.median(),
          'equal-sigma idx',eq,'modal opt path',g.path.mode().iloc[0])
# H3b plateau onset
lam=lambda s: np.log((1-s)**2/s**2)
l0,l1=lam(0.99),lam(0.01)
on=[]
for c,sub in [('ZIT','attn'),('ZIT','mlp'),('ZIB','attn'),('ZIB','mlp'),('MOCHI','attn'),('SD35M','attn'),('SD35L','attn'),('MOCHI1','attn')]:
    g,m,X,s=pickle.load(open(f'F_{c}_{sub}.pkl','rb'))
    Y=X.mean(0).copy(); Y[...,0]=np.log(Y[...,0]); Y[...,1]=np.log(Y[...,1]); A=Y.reshape(len(s),-1)
    sp=np.linalg.norm(np.diff(A,axis=0),axis=1)/np.abs(np.diff(s)); cm=(s[1:]+s[:-1])/2
    plat=np.median(sp[cm<0.6]); i=int(np.argmax(sp<2*plat)); so=cm[i]
    on.append(dict(ch=f'{c}-{sub}',onset_sigma=so,onset_stepfrac=(i+0.5)/(len(s)-1),onset_lambda=lam(so),onset_lambda01=(lam(so)-l0)/(l1-l0),plateau_speed=plat))
O=pd.DataFrame(on); print(O.round(3).to_string())
main=O[O.ch!='MOCHI1-attn']
print('SD step-frac %.4f  SD lambda01 %.4f  ratio %.3f'%(main.onset_stepfrac.std(),main.onset_lambda01.std(),main.onset_lambda01.std()/main.onset_stepfrac.std()))
O.to_csv(f'{_COSPAFS_ROOT}/review/round3/zib10/out/E3_plateau_onset.csv',index=False)
