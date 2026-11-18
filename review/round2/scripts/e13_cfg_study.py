import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# E13 / P13: SD3.5-M CFG 1.0 (cond-only) vs 5.5 (branch-mean), one prompt, 30 gens x 100 steps x 24 layers.
import sys; sys.path.insert(0,'.'); from beta24 import *
import numpy as np, pandas as pd
D=f'{_COSPAFS_ROOT}/data/sd35m_cfg1.0_vs_5.5/'
rng=np.random.default_rng(0); out={}
def load(f):
    d=pd.read_csv(D+f); d['gid']=d.gen_idx.astype(str); return d
def eig_fit(G,GAP,H,gid,gcut):
    X=np.stack([G,-GAP,H],-1)  # rows,L,3
    x0=X[:,:-1].reshape(-1,3); x1=X[:,1:].reshape(-1,3); gg=np.repeat(gid,X.shape[1]-1)
    m=x0[:,0]>gcut; x0,x1,gg=x0[m],x1[m],gg[m]
    def fit(a,b):
        mu=a.mean(0); sd=a.std(0)+1e-12  # standardize for conditioning; eigenvalues invariant under affine rescale
        A_=np.c_[(a-mu)/sd,np.ones(len(a))]; Bt=(b-mu)/sd
        W=np.linalg.lstsq(A_,Bt,rcond=None)[0][:3].T; ev=np.linalg.eigvals(W)
        cx=ev[np.abs(ev.imag)>1e-9]
        return (np.abs(cx[0]),abs(np.angle(cx[0]))) if len(cx) else (np.nan,np.nan)
    est=fit(x0,x1); ug=np.unique(gg); bs=[]
    for _ in range(500):
        s=rng.choice(ug,len(ug)); idx=np.concatenate([np.where(gg==u)[0] for u in s]); bs.append(fit(x0[idx],x1[idx]))
    bs=np.array(bs); ok=~np.isnan(bs[:,0])
    return est, ok.mean(), np.nanpercentile(bs,[2.5,97.5],axis=0)
rows=[]
for cfg,f in [('1.0','sd-3.5-medium-CFG1.0-30.1.csv'),('5.5','sd-3.5-medium-CFG5.5-30.1.csv')]:
    d=load(f)
    for win,steps in [('0-9',range(10)),('all',None)]:
        G,GAP,H,gid=pairs(d,steps=list(steps) if steps else None)
        c,bm=beta24(G); cr=crossings(c,bm)
        gcut=cr[-1][0] if cr and cr[-1][1]=='+-' else np.nan
        est,frac,ci=eig_fit(G,GAP,H,gid,gcut) if not np.isnan(gcut) else ((np.nan,np.nan),np.nan,np.full((2,2),np.nan))
        rows.append(dict(cfg=cfg,steps=win,signs=signs(bm),crossings=';'.join(f'{x:.5f}' for x,_ in cr),TRI=TRI(c,bm),
                         mod=est[0],ang=est[1],complex_frac=frac,mod_lo=ci[0,0],mod_hi=ci[1,0],ang_lo=ci[0,1],ang_hi=ci[1,1]))
    # spread: SD across gens of layer-mean log g, per step, median over steps
    piv=d.pivot_table(index=['gid','token_idx'],columns='layer',values='g_micro')
    lg=np.log(piv.values).mean(1).reshape(-1,100); out[cfg]=lg
R=pd.DataFrame(rows); print(R.round(4).to_string())
sp={k:v.std(0,ddof=1) for k,v in out.items()}
ratio=sp['5.5']/sp['1.0']
print('median spread 1.0 %.4f 5.5 %.4f ratio %.3f  (ratio by step quartiles: %s)'%(np.median(sp['1.0']),np.median(sp['5.5']),np.median(ratio),np.round([np.median(r) for r in np.array_split(ratio,4)],3)))
print('mean-over-steps spread ratio %.3f'%(np.mean(sp['5.5'])/np.mean(sp['1.0'])))
R.to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E13_cfg_study.csv',index=False)
pd.DataFrame(dict(step=range(100),spread_cfg1=sp['1.0'],spread_cfg55=sp['5.5'],ratio=ratio)).to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E13_spread_by_step.csv',index=False)
