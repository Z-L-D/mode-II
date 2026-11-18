import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Round 4 section P: calculations frozen in R4_PREREGISTRATION.md (SHA 82ef210a..., 2026-10-06T23:01:29Z).
# Run from the scratchpad (needs F_*.pkl archive fields, F3_*.pkl from round3/audit/scripts/r3_load.py, pq/*.parquet).
import pickle, sys, json, numpy as np, pandas as pd
sys.path.insert(0,f'{_COSPAFS_ROOT}/review/round2/scripts')
from common import depthnorm, shape_r, load as load_archive
from beta24 import beta24, crossings, TRI, signs
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
OUT=f'{_COSPAFS_ROOT}/review/round4/out'; import os; os.makedirs(OUT,exist_ok=True)
R={}; rng=np.random.default_rng(4)
def F(n): return pickle.load(open(n,'rb'))
def arm(a,sub='attn'):
    parts=[F(f'F3_{x}_{sub}.pkl') for x in a.split('+')]
    return sum([p[0] for p in parts],[]), pd.concat([p[1] for p in parts]), np.concatenate([p[2] for p in parts]), parts[0][3]
S10='S10_1024_s1001+S10_1024_s1002'; S45='S45_1024_s1001+S45_1024_s1002'
def per_prompt(X,m,prompts,k=0):  # mean over gens of log(var k) -> [P,T,L]
    pt=np.asarray(m.pt.astype(str)); return np.stack([np.log(X[pt==p][...,k]).mean(0) for p in prompts])
# ---------------- P1 pooling ----------------
_,ma,Xa,sa=F('F_SD35M_attn.pkl'); _,mr,Xr,sr=arm(S45)
sh=sorted(set(ma.pt.astype(str))&set(mr.pt.astype(str))); R['P1_SD35M_prompts']=len(sh)
assert np.allclose(sa,sr,atol=1e-4), 'sigma schedules differ'
for k,nm in [(0,'g'),(1,'gap')]:
    A=per_prompt(Xa,ma,sh,k); B=per_prompt(Xr,mr,sh,k); d=B-A   # pooled(R3) - BM(archive)
    R[f'P1a_SD35M_median_log_{nm}_pooled_over_BM_steps>=2']=float(np.median(d[:,2:]))
    R[f'P1d_SD35M_median_log_{nm}_R3_over_archive_step0']=float(np.median(d[:,0]))
    R[f'P1_SD35M_frac_positive_{nm}_steps>=2']=float((d[:,2:]>0).mean())
    R[f'P1_SD35M_median_log_{nm}_by_step']=[float(np.median(d[:,t])) for t in range(d.shape[1])]
for sub in ['attn','mlp']:
    _,mo,Xo,so=F(f'F_ZIB_{sub}.pkl'); _,mz,Xz,sz=arm('Z40_1024_s1001',sub)
    assert np.allclose(so,sz,atol=1e-4)
    sh2=sorted(set(mo.pt.astype(str))&set(mz.pt.astype(str))); R['P1_ZIB_prompts']=len(sh2)
    for k,nm in [(0,'g'),(1,'gap')]:
        d=per_prompt(Xo,mo,sh2,k)-per_prompt(Xz,mz,sh2,k)    # pooled(archive) - BM(R3)
        R[f'P1b_ZIB_{sub}_median_log_{nm}_pooled_over_BM']=float(np.median(d)); R[f'P1_ZIB_{sub}_frac_positive_{nm}']=float((d>0).mean())
# ---------------- P2 lineage decomposition ----------------
gr=np.linspace(0,1,24)
def prof(X,s):
    D=depthnorm(X.mean(0)); u=np.linspace(0,1,D.shape[1])
    return np.stack([[np.interp(gr,u,D[t,:,k]) for k in range(3)] for t in range(len(s))]).transpose(0,2,1)
def cos(a,b): a=a.ravel(); b=b.ravel(); return a@b/np.linalg.norm(a)/np.linalg.norm(b)
def tangent(XA,sa,XB,sb):
    A=prof(XA,sa); B=prof(XB,sb); tc=[]; srr=[]
    for t,x in enumerate(sa):
        j=int(np.argmin(np.abs(sb-x))); tc.append(cos(np.diff(A[t],axis=0),np.diff(B[j],axis=0))); srr.append(shape_r(A[t],B[j]))
    return float(np.mean(tc)), float(np.mean(srr))
for sub in ['attn','mlp']:
    _,mt,Xt,st=F(f'F_ZIT_{sub}.pkl'); _,mz4,Xz4,sz4=arm('Z40_1024_s1001',sub); _,mz1,Xz1,sz1=arm('Z10_1024_s1001',sub); _,mo,Xo,so=F(f'F_ZIB_{sub}.pkl')
    R[f'P2a_tangent_ZIB4BM_ZIT_{sub}'],R[f'P2a_shape_ZIB4BM_ZIT_{sub}']=tangent(Xt,st,Xz4,sz4)
    R[f'ref_tangent_ZIB4pooled_ZIT_{sub}'],_=tangent(Xt,st,Xo,so)
    R[f'ref_tangent_ZIB1_ZIT_{sub}'],_=tangent(Xt,st,Xz1,sz1)
    R[f'P2b_tangent_ZIB1_ZIB4BM_{sub}'],R[f'P2b_shape_ZIB1_ZIB4BM_{sub}']=tangent(Xz1,sz1,Xz4,sz4)
    R[f'post_tangent_ZIB4BM_ZIB4pooled_{sub}'],R[f'post_shape_ZIB4BM_ZIB4pooled_{sub}']=tangent(Xo,so,Xz4,sz4)
    sh3=sorted(set(mt.pt.astype(str))&set(mz4.pt.astype(str))&set(mo.pt.astype(str))&set(mz1.pt.astype(str)))
    def mlg(X,m): return per_prompt(X,m,sh3).mean(0)      # [T,L], balanced over prompts
    T_=mlg(Xt,mt); j4=[int(np.argmin(np.abs(sz4-x))) for x in st]; jo=[int(np.argmin(np.abs(so-x))) for x in st]; j1=[int(np.argmin(np.abs(sz1-x))) for x in st]
    off=lambda M,j: float(np.median(np.abs(M[j]-T_)))
    o4,oo,o1=off(mlg(Xz4,mz4),j4),off(mlg(Xo,mo),jo),off(mlg(Xz1,mz1),j1)
    R[f'P2c_offset_ZIB4BM_ZIT_{sub}']=o4; R[f'P2c_offset_ZIB4pooled_ZIT_{sub}']=oo; R[f'P2c_offset_ZIB1_ZIT_{sub}']=o1
    R[f'P2c_ratio_BM_over_pooled_{sub}']=o4/oo; R[f'post_ratio_ZIB1_over_pooled_{sub}']=o1/oo; R['P2c_prompts']=len(sh3)
# ---------------- P3 seed-centred fate and persistence ----------------
def resid(X,m):
    grp=pd.factorize(np.asarray(m.ph.astype(str)))[0]
    Y=np.log(np.abs(X)+1e-12); Y[...,2]=X[...,2]; G,T,L,_=Y.shape; Fz=Y.reshape(G,T,-1); Fz=(Fz-Fz.mean(0))/(Fz.std(0)+1e-12); r=np.empty_like(Fz)
    for t in range(T): r[:,t]=Fz[:,t]-pd.DataFrame(Fz[:,t]).groupby(grp).transform('mean').to_numpy()
    return r,grp
def ridge(A,B,alpha=100.0):
    mu=A.mean(0); mb=B.mean(0); W=np.linalg.solve((A-mu).T@(A-mu)+alpha*np.eye(A.shape[1]),(A-mu).T@(B-mb)); return lambda Z:(Z-mu)@W+mb
def curve(r,grp,seed,center):
    T=r.shape[1]; num=np.zeros(T); tgt_all=[]
    preds=np.zeros((T,)+r[:,-1].shape); tg=np.zeros_like(r[:,-1])
    for tr,te in GroupKFold(5).split(r[:,0],groups=grp):
        rc=r.copy()
        if center:
            for sd in np.unique(seed):
                msk_tr=tr[seed[tr]==sd]; mu=r[msk_tr].mean(0)          # [T,K] from training prompts only
                rc[seed==sd]=r[seed==sd]-mu
        tg[te]=rc[te,-1]
        for t in range(T): preds[t,te]=ridge(rc[tr,t],rc[tr,-1])(rc[te,t])
    return np.array([1-((tg-preds[t])**2).sum()/((tg-tg.mean(0))**2).sum() for t in range(T)])
def s50(c,s):
    T=len(c); tau=np.arange(T)/(T-1)
    if not (c>0.5).any(): return np.nan
    i=int(np.argmax(c>0.5)); t50=tau[0] if i==0 else tau[i-1]+(0.5-c[i-1])*(tau[i]-tau[i-1])/(c[i]-c[i-1]); return float(np.interp(t50,tau,s))
def VR(r,k_list=(2,4,8)):
    rr=r[:,1:]; v1=((rr[:,1:]-rr[:,:-1])**2).mean(); return {k:float(((rr[:,k:]-rr[:,:-k])**2).mean()/(k*v1)) for k in k_list}
for a,key in [(S10,'CFG1'),(S45,'CFG45')]:
    _,m,X,s=arm(a); r,grp=resid(X,m); seed=m.seed.values
    cu=curve(r,grp,seed,False); cc=curve(r,grp,seed,True)
    R[f'P3_sigma50_uncentred_{key}']=s50(cu,s); R[f'P3a_sigma50_centred_{key}']=s50(cc,s)
    R[f'P3_R2step0_uncentred_{key}']=float(cu[0]); R[f'P3b_R2step0_centred_{key}']=float(cc[0])
    R[f'P3_R2curve_centred_{key}']=cc.round(4).tolist(); R[f'P3_R2curve_uncentred_{key}']=cu.round(4).tolist()
    rc=r.copy()
    for sd in np.unique(seed): rc[seed==sd]-=r[seed==sd].mean(0)
    R[f'P3_VR_uncentred_{key}']=VR(r); R[f'P3_VR_centred_{key}']=VR(rc)
    R[f'P3_seed_component_var_share_{key}']=float(((r-rc)**2).sum()/(r**2).sum())
    # bootstrap over prompts for the centred sigma50 (CFG1 only, 100 draws)
    if key=='CFG1':
        ug=np.unique(grp); bt=[]
        for _ in range(100):
            sel=rng.choice(ug,len(ug),replace=True); idx=np.concatenate([np.where(grp==u)[0] for u in sel])
            gg=np.concatenate([np.full((grp==u).sum(),i) for i,u in enumerate(sel)])
            bt.append(s50(curve(r[idx],gg,seed[idx],True),s))
        R['P3a_sigma50_centred_CFG1_CI95']=[float(np.nanpercentile(bt,2.5)),float(np.nanpercentile(bt,97.5))]; R['P3a_boot_nan_frac']=float(np.isnan(bt).mean())
# ---------------- P4 four-crossing confound ----------------
dr,mrs,_,_=load_archive('SDRES')
def legacy_cube(res):
    d=dr[dr.gid.isin(mrs.index[mrs.res==res])]
    piv=d.pivot_table(index=['gid','token_idx'],columns='layer',values='g_micro'); g=piv.index.get_level_values(0).unique()
    sig=d[d.gid==g[0]].groupby('token_idx').sigma.first().values
    return piv.values.reshape(len(g),100,-1), sig
GL,sL=legacy_cube(1024); _,m10,X10,s10=arm(S10); GP=X10[...,0]
jL=[int(np.argmin(np.abs(sL-x))) for x in s10]
lr=np.log(GL.mean(0))[jL]-np.log(GP.mean(0)); R['P4a_median_log_g_legacy_over_passive']=float(np.median(lr))
R['P4a_rho_step_layer_mean_g_legacy']=float(spearmanr(np.arange(100),GL.mean(0).mean(1)).statistic)
R['P4a_rho_step_layer_mean_g_passive']=float(spearmanr(np.arange(len(s10)),GP.mean(0).mean(1)).statistic)
c,bm=beta24(GL.reshape(-1,GL.shape[2])); R['P4_legacy1024_base']=dict(signs=signs(bm),n=len(crossings(c,bm)))
from numpy.polynomial import legendre as Lg
us=2*(sL-sL.min())/(sL.max()-sL.min())-1; ul=np.linspace(-1,1,GL.shape[2])
Bs=np.stack([Lg.legval(us,np.eye(3)[i]) for i in range(3)]); Bl=np.stack([Lg.legval(ul,np.eye(3)[i]) for i in range(3)])
basis=np.einsum('it,jl->ijtl',Bs,Bl).reshape(9,len(sL),len(ul))
P4b=[]
for rms in [0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20]:
    keep=0; seqs={}
    for _ in range(200):
        e=np.tensordot(rng.normal(size=9),basis,1); e*=rms/np.sqrt((e**2).mean())
        G2=GL*(1+e)[None]; c2,b2=beta24(G2.reshape(-1,G2.shape[2])); sg=signs(b2); seqs[sg]=seqs.get(sg,0)+1; keep+=(sg=='-+-+-')
    P4b.append(dict(rms=rms,frac_keep=keep/200,top=sorted(seqs.items(),key=lambda x:-x[1])[:3]))
R['P4b']=P4b; lost=[x['rms'] for x in P4b if x['frac_keep']<=0.5]; R['P4b_threshold_rms']=lost[0] if lost else '>0.20'
# P4c single prompt (passive) / single generation (legacy, thinned to 40 sigma)
pt=np.asarray(m10.pt.astype(str)); first_p=[]; seq_p=[]
for p in np.unique(pt):
    c,bm=beta24(GP[pt==p].reshape(-1,GP.shape[2])); sg=signs(bm); first_p.append(sg[0]); seq_p.append(sg)
first_l=[]; seq_l=[]
for g in range(GL.shape[0]):
    c,bm=beta24(GL[g][jL]); sg=signs(bm); first_l.append(sg[0]); seq_l.append(sg)
R['P4c_passive_prompt_frac_first_plus']=float(np.mean(np.array(first_p)=='+')); R['P4c_legacy_gen_frac_first_minus']=float(np.mean(np.array(first_l)=='-'))
R['P4c_passive_prompt_seq_freq']=pd.Series(seq_p).value_counts().head(6).to_dict(); R['P4c_legacy_gen_seq_freq']=pd.Series(seq_l).value_counts().head(6).to_dict()
R['P4c_passive_prompt_frac_-+-+-']=float(np.mean(np.array(seq_p)=='-+-+-')); R['P4c_legacy_gen_frac_-+-+-']=float(np.mean(np.array(seq_l)=='-+-+-'))
# P4d legacy CFG study 256 px
D=f'{_COSPAFS_ROOT}/data/sd35m_cfg1.0_vs_5.5/'
for f,key in [('sd-3.5-medium-CFG1.0-30.1.csv','CFG1.0'),('sd-3.5-medium-CFG5.5-30.1.csv','CFG5.5')]:
    d=pd.read_csv(D+f); cnt=d.groupby('gen_idx').size(); d=d[d.gen_idx.isin(cnt[cnt==cnt.max()].index)]
    G=d.pivot_table(index=['gen_idx','token_idx'],columns='layer',values='g_micro').values
    c,bm=beta24(G); cr=crossings(c,bm); R[f'P4d_legacy256_{key}']=dict(signs=signs(bm),n_cross=len(cr),crossings=[round(x,5) for x,_ in cr],TRI=float(TRI(c,bm)) if TRI(c,bm)==TRI(c,bm) else None,gens=int(d.gen_idx.nunique()))
# ---------------- P5 onsets ----------------
def onset(X,s):
    Y=X.mean(0).copy(); Y[...,0]=np.log(Y[...,0]); Y[...,1]=np.log(Y[...,1]); A=Y.reshape(len(s),-1)
    sp=np.linalg.norm(np.diff(A,axis=0),axis=1)/np.abs(np.diff(s)); cm=(s[1:]+s[:-1])/2
    plat=np.median(sp[cm<0.6]); i=int(np.argmax(sp<2*plat)); return float(cm[i])
for a,sub in [(S45,'attn'),('S45_768_s1001','attn'),('S45_512_s1001','attn'),('Z40_1024_s1001','attn'),('Z40_1024_s1001','mlp')]:
    _,_,X,s=arm(a,sub); R[f'P5_onset_{a.split("+")[0]}_{sub}']=onset(X,s)
_,_,_,sz=arm('Z40_1024_s1001'); cmz=(sz[1:]+sz[:-1])/2; R['P5_ZIB_midpoint_grid_0.6-0.95']=[round(float(x),4) for x in cmz if 0.6<x<0.95]
json.dump(R,open(f'{OUT}/P_results.json','w'),indent=1,default=str)
for k,v in R.items():
    if 'curve' in k or 'by_step' in k: continue
    print(k,':',v)
