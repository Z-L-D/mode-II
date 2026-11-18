import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# E4 / P4b: Mochi 30-step CFG4.5 (branch-mean) vs 100-step CFG1.0 (cond-only), per-sigma speed by band.
import pickle, sys; sys.path.insert(0,'.'); from common import *
bins=[(0.75,0.9),(0.6,0.75),(0.45,0.6),(0.3,0.45),(0.15,0.3),(0.0,0.15),(0.9,1.01)]
res={}
for c in ['MOCHI','MOCHI1']:
    g,m,X,s=pickle.load(open(f'F_{c}_attn.pkl','rb'))
    Y=X.mean(0).copy(); Y[...,0]=np.log(Y[...,0]); Y[...,1]=np.log(Y[...,1]); A=Y.reshape(len(s),-1)
    st=np.linalg.norm(np.diff(A,axis=0),axis=1); ds=np.abs(np.diff(s)); cm=(s[1:]+s[:-1])/2
    # path length per unit sigma within band (sum of step lengths / sigma covered) -- step-count dependent through sqrt(n) noise is ignored at pop mean
    res[c]=[st[(cm>lo)&(cm<=hi)].sum()/ds[(cm>lo)&(cm<=hi)].sum() if ((cm>lo)&(cm<=hi)).any() else np.nan for lo,hi in bins]
    print(c,'sigma range',s.min().round(3),s.max().round(3),'G',len(g))
r=np.array(res['MOCHI'])/np.array(res['MOCHI1'])
out=pd.DataFrame(dict(band=[f'{lo}-{hi}' for lo,hi in bins],speed_30=res['MOCHI'],speed_100=res['MOCHI1'],ratio=r)); print(out.round(3).to_string())
out.to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E4b_mochi_speed.csv',index=False)
# jitter-robust check: chord speed ||A(sig_hi)-A(sig_lo)||/dsigma on sigma-interpolated pop-mean trajectory
print('chord speeds (jitter-robust)')
ch={}
for c in ['MOCHI','MOCHI1']:
    g,m,X,s=pickle.load(open(f'F_{c}_attn.pkl','rb'))
    Y=X.mean(0).copy(); Y[...,0]=np.log(Y[...,0]); Y[...,1]=np.log(Y[...,1]); A=Y.reshape(len(s),-1)
    f=lambda q: np.array([np.interp(q,s[::-1],A[::-1,j]) for j in range(A.shape[1])])
    ch[c]=[np.linalg.norm(f(hi)-f(lo))/(hi-lo) if lo>=s.min() else np.nan for lo,hi in [(0.75,0.9),(0.6,0.75),(0.45,0.6),(0.3,0.45),(0.18,0.3),(0.06,0.18)]]
print(pd.DataFrame(ch,index=['0.75-0.9','0.6-0.75','0.45-0.6','0.3-0.45','0.18-0.3','0.06-0.18']).assign(ratio=lambda d:d.MOCHI/d.MOCHI1).round(3))
