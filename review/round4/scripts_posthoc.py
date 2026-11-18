import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Round 4 post hoc checks (labelled post hoc; run after the P section was scored).
import pickle, sys, json, numpy as np, pandas as pd
sys.path.insert(0,f'{_COSPAFS_ROOT}/review/round2/scripts'); sys.path.insert(0,f'{_COSPAFS_ROOT}/review/round4')
from common import load as load_archive
from beta24 import beta24, crossings, TRI, signs
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
import importlib.util
spec=importlib.util.spec_from_file_location('pc',f'{_COSPAFS_ROOT}/review/round4/scripts_p_calcs.py')
OUT=f'{_COSPAFS_ROOT}/review/round4/out'; H={}; rng=np.random.default_rng(44)
def F(n): return pickle.load(open(n,'rb'))
def arm(a,sub='attn'):
    parts=[F(f'F3_{x}_{sub}.pkl') for x in a.split('+')]
    return sum([p[0] for p in parts],[]), pd.concat([p[1] for p in parts]), np.concatenate([p[2] for p in parts]), parts[0][3]
# H1 legacy level and trend by resolution (incl. 256 px CFG study) vs passive
dr,mrs,_,_=load_archive('SDRES'); rows=[]
for res in [512,768,1024]:
    d=dr[dr.gid.isin(mrs.index[mrs.res==res])]; pm=d.groupby('token_idx').g_micro.mean()
    c,bm=beta24(d.pivot_table(index=['gid','token_idx'],columns='layer',values='g_micro').values)
    rows.append(dict(set=f'legacy {res}',median_g=float(d.g_micro.median()),g_step0=float(pm.iloc[0]),g_last=float(pm.iloc[-1]),rho=float(spearmanr(pm.index,pm.values).statistic),signs=signs(bm),seq=int(d.seq_len.iloc[0])))
d=pd.read_csv(f'{_COSPAFS_ROOT}/data/sd35m_cfg1.0_vs_5.5/sd-3.5-medium-CFG1.0-30.1.csv'); pm=d.groupby('token_idx').g_micro.mean()
c,bm=beta24(d.pivot_table(index=['gen_idx','token_idx'],columns='layer',values='g_micro').values)
rows.append(dict(set='legacy 256 (CFG study, CFG1)',median_g=float(d.g_micro.median()),g_step0=float(pm.iloc[0]),g_last=float(pm.iloc[-1]),rho=float(spearmanr(pm.index,pm.values).statistic),signs=signs(bm),seq=int(d.seq_len.iloc[0])))
for a,lab in [('S10_1024_s1001+S10_1024_s1002','passive 1024 CFG1 (R3)'),('S45_1024_s1001+S45_1024_s1002','passive 1024 CFG4.5 pooled (R3)'),('S45_768_s1001','passive 768 CFG4.5 pooled'),('S45_512_s1001','passive 512 CFG4.5 pooled')]:
    _,m,X,s=arm(a); G=X[...,0]; pm=G.mean(0).mean(1); c,bm=beta24(G.reshape(-1,G.shape[2]))
    rows.append(dict(set=lab,median_g=float(np.median(G)),g_step0=float(pm[0]),g_last=float(pm[-1]),rho=float(spearmanr(np.arange(len(pm)),pm).statistic),signs=signs(bm),seq=int(m.seq.iloc[0])))
H['H1']=pd.DataFrame(rows); print(H['H1'].to_string())
# H2 the reviewer's 0.278 / 0.250: which arm reproduces it?
from common import depthnorm
for sub in ['attn','mlp']:
    _,mt,Xt,st=F(f'F_ZIT_{sub}.pkl'); _,mo,Xo,so=F(f'F_ZIB_{sub}.pkl')
    T_=np.log(Xt[...,0]).mean(0)
    def off(X,s): j=[int(np.argmin(np.abs(s-x))) for x in st]; return float(np.median(np.abs(np.log(X[...,0]).mean(0)[j]-T_)))
    oo=off(Xo,so)
    for a in ['Z10_1024_s1001','Z40_1024_s1001']:
        _,_,X,s=arm(a,sub); H[f'H2_fullpop_ratio_{a}_{sub}']=off(X,s)/oo
# H3 does the seeds-per-prompt design move sigma50? archive SD35M (4 runs/prompt) subsampled to 2 runs/prompt
exec(open(f'{_COSPAFS_ROOT}/review/round4/scripts_p_calcs.py').read().split('# ---------------- P3')[1].split('for a,key in')[0].split('\n',1)[1])
g,m,X,s=F('F_SD35M_attn.pkl'); pt=np.asarray(m.pt.astype(str)); m=m.copy(); m['ph']=pt
r,grp=resid(X,m); seedz=np.zeros(len(pt))
H['H3_archive_SD35M_4runs_sigma50']=s50(curve(r,grp,seedz,False),s)
vals=[]
for _ in range(20):
    idx=np.concatenate([rng.choice(np.where(pt==p)[0],2,replace=False) for p in np.unique(pt)])
    r2,g2=resid(X[idx],m.iloc[idx]); vals.append(s50(curve(r2,g2,np.zeros(len(idx)),False),s))
H['H3_archive_SD35M_2runs_sigma50_median']=float(np.nanmedian(vals)); H['H3_archive_SD35M_2runs_sigma50_range']=[float(np.nanmin(vals)),float(np.nanmax(vals))]
# archive VR on 2-run subsample (design check for VR)
vr=[]
for _ in range(20):
    idx=np.concatenate([rng.choice(np.where(pt==p)[0],2,replace=False) for p in np.unique(pt)])
    r2,_=resid(X[idx],m.iloc[idx]); vr.append(VR(r2)[4])
H['H3_archive_SD35M_2runs_VR4_median']=float(np.median(vr))
for k,v in H.items():
    if k!='H1': print(k,':',v)
H['H1']=H['H1'].to_dict('records'); json.dump(H,open(f'{OUT}/posthoc_results.json','w'),indent=1,default=str)
