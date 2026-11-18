# EXPLORATORY 2026-11-05: when does prompt-specific geometry appear in the unconditional branch?
# Per sampling step: Mantel r (no null) between cond and uncond prompt RDMs (Z1 Z-Image Base; ACE Base), and
# domain decoding accuracy (prompt-grouped 5-fold) on cond and on uncond, per step.
import pickle, warnings, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
warnings.filterwarnings('ignore'); C='/tmp/claude-0/work/mode-I/cache'; OUT='/tmp/claude-0/survey/explore'
def load(n):
    g,m,X,s=pickle.load(open(f'{C}/F_{n}.pkl','rb')); m=m.copy(); m['k']=np.asarray(m.pt.astype(str).str.strip().tolist()); return m,X.astype(float),np.asarray(s,float)
def clf(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
rows=[]
for fam,cn,un in [('Z-Image Base','Z1','Z1U'),('ACE-Step Base','ACEB','ACEBU')]:
  for sub in ['attn','mlp']:
    mc,Xc,s=load(f'{cn}_{sub}'); mu,Xu,_=load(f'{un}_{sub}')
    P=sorted(set(mc.k)&set(mu.k)); iu=np.triu_indices(len(P),1)
    dc=np.asarray(mc.dom.astype(str).tolist()) if 'dom' in mc else None
    for t in range(Xc.shape[1]):
        def pm(m,X):
            F=np.concatenate([np.log(np.abs(X[:,t,:,0])+1e-12),np.log(np.abs(X[:,t,:,1])+1e-12),X[:,t,:,2]],1)
            M=np.stack([F[m.k==p].mean(0) for p in P]); return (M-M.mean(0))/(M.std(0)+1e-12)
        Rc=1-np.corrcoef(pm(mc,Xc)); Ru=1-np.corrcoef(pm(mu,Xu))
        r=np.corrcoef(rankdata(Rc[iu]),rankdata(Ru[iu]))[0,1]
        row=dict(family=fam,sub=sub,step=t,sigma=s[t],rsa_cond_uncond=r)
        for lab,m,X in ([('cond',mc,Xc),('uncond',mu,Xu)] if fam.startswith('Z') else []):
            F=np.log(np.abs(X[:,t])+1e-9).reshape(len(X),-1); y=np.asarray(m.dom.astype(str).tolist()); grp=pd.factorize(m.k)[0]
            row[f'dec_{lab}']=float(cross_val_score(clf(),F,y,groups=grp,cv=GroupKFold(5)).mean())
        rows.append(row)
    d=pd.DataFrame([r for r in rows if r['family']==fam and r['sub']==sub]); print(fam,sub,'domains',len(set(mc.dom)),'prompts',len(P)); print(d[[c for c in ['step','sigma','rsa_cond_uncond','dec_cond','dec_uncond'] if c in d]].round(3).to_string(index=False),flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/x2_branch_timeline.csv',index=False)
