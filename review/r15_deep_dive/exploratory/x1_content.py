# EXPLORATORY (not preregistered), 2026-11-05: content geometry on the real Base captures.
# X1a first-state domain decoding (Round 1 decoder) for Z1 cond, Z1 uncond, Z3, plus archive models for reference.
# X1b prompt RSA (Round 6 S2 features: first/middle/last state, 24-pt depth interp of log g, log gap, H) among Z1, Z3, ZIT, ZIB, SD35M, SD35L, MOCHI on shared prompts.
# X1c cross-model domain transfer: train decoder on model A (prompt means, 24-pt depth), test on model B.
import pickle, warnings, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
warnings.filterwarnings('ignore'); C='/tmp/claude-0/work/mode-I/cache'; OUT='/tmp/claude-0/survey/explore'
rng=np.random.default_rng(5)
def load(n):
    g,m,X,s=pickle.load(open(f'{C}/F_{n}.pkl','rb')); m=m.copy(); m['k']=np.asarray(m.pt.astype(str).str.strip().tolist()); return m,X,np.asarray(s,float)
def clf(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
gr=np.linspace(0,1,24)
def prof(X,t):
    Y=X[:,t].astype(float).copy(); Y[...,0]=np.log(np.abs(Y[...,0])+1e-12); Y[...,1]=np.log(np.abs(Y[...,1])+1e-12)
    return np.stack([np.concatenate([np.interp(gr,np.linspace(0,1,Y.shape[1]),Y[i,:,v]) for v in range(3)]) for i in range(len(Y))])
names=['Z1_attn','Z1_mlp','Z1U_attn','Z1U_mlp','Z3_attn','Z3_mlp','ZIT_attn','ZIT_mlp','ZIB_attn','ZIB_mlp','SD35M_attn','SD35L_attn','MOCHI_attn']
D={n:load(n) for n in names}
rows=[]
for n in names:
    m,X,s=D[n]; y=np.asarray(m.dom.astype(str).tolist()); grp=pd.factorize(m.k)[0]
    F=np.log(np.abs(X[:,0].astype(float))+1e-9).reshape(len(X),-1)
    a=float(cross_val_score(clf(),F,y,groups=grp,cv=GroupKFold(5)).mean())
    rows.append(dict(model=n,gens=len(X),prompts=m.k.nunique(),domains=len(set(y)),acc_first_state=a)); print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/x1a_decoding.csv',index=False)
# shared prompts
P=sorted(set.intersection(*[set(D[n][0].k) for n in names])); NP=len(P); print('shared prompts',NP)
dom=np.asarray(D['ZIT_attn'][0].groupby('k').dom.first().loc[P].astype(str).tolist())
def pm(n,tau):
    m,X,s=D[n]; T=X.shape[1]; t={'first':0,'middle':T//2,'last':T-1}[tau]; F=prof(X,t)
    M=np.stack([F[m.k.values==p].mean(0) for p in P]); return (M-M.mean(0))/(M.std(0)+1e-12)
iu=np.triu_indices(NP,1)
def mantel(A,B,nperm=999):
    a=rankdata(A[iu]); b=rankdata(B[iu]); r=np.corrcoef(a,b)[0,1]; nul=[]
    for _ in range(nperm):
        p=rng.permutation(NP); nul.append(np.corrcoef(a,rankdata(B[np.ix_(p,p)][iu]))[0,1])
    nul=np.array(nul); return r,(1+(nul>=r).sum())/(nperm+1),np.percentile(nul,95)
FE={(n,t):pm(n,t) for n in names for t in ['first','middle','last']}
RD={k:1-np.corrcoef(v) for k,v in FE.items()}
pairs=[('Z1_attn','Z3_attn'),('Z1_mlp','Z3_mlp'),('Z1_attn','ZIT_attn'),('Z1_mlp','ZIT_mlp'),('Z1_attn','ZIB_attn'),('Z1_mlp','ZIB_mlp'),('Z1_attn','Z1U_attn'),
       ('Z3_attn','ZIB_attn'),('ZIT_attn','ZIB_attn'),('Z1_attn','SD35M_attn'),('Z1_attn','SD35L_attn'),('Z1_attn','MOCHI_attn'),('ZIT_attn','SD35M_attn'),('SD35M_attn','SD35L_attn')]
rr=[]
for a,b in pairs:
    for t in ['first','middle','last']:
        r,p,n95=mantel(RD[(a,t)],RD[(b,t)]); rr.append(dict(a=a,b=b,tau=t,rsa=r,p=p,null_p95=n95))
RS=pd.DataFrame(rr); RS.to_csv(f'{OUT}/x1b_rsa.csv',index=False); print(RS.pivot_table(index=['a','b'],columns='tau',values='rsa').round(3))
# cross-model transfer of domain decoder (prompt-mean first-state profiles, z-scored within model)
tr=[]
mods=['Z1_attn','Z3_attn','ZIT_attn','ZIB_attn','SD35M_attn','SD35L_attn','MOCHI_attn','Z1_mlp','ZIT_mlp']
for a in mods:
    for b in mods:
        if a==b: continue
        FA,FB=FE[(a,'first')],FE[(b,'first')]
        c=clf().fit(FA,dom); acc=float((c.predict(FB)==dom).mean())
        nul=[float((clf().fit(FA,dom).predict(FB)==dom[rng.permutation(NP)]).mean()) for _ in range(0)]
        tr.append(dict(train=a,test=b,acc=acc))
T=pd.DataFrame(tr); T.to_csv(f'{OUT}/x1c_transfer.csv',index=False); print(T.pivot(index='train',columns='test',values='acc').round(2))
