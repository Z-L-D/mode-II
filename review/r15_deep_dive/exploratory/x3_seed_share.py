# EXPLORATORY 2026-11-05: does distillation shrink seed-to-seed variation (a telemetry view of reduced sample diversity)?
# Seed share = mean within-prompt variance / total variance of the standardised depth profile (log g, log gap, H) at a state.
# Compared at matched sigma for Base vs Turbo (Z-Image: Z1 cond vs ZIT; ACE: ACEB cond vs ACET), with prompt bootstrap CIs.
import pickle, numpy as np, pandas as pd
C='/tmp/claude-0/work/mode-I/cache'; rng=np.random.default_rng(7)
def load(n):
    g,m,X,s=pickle.load(open(f'{C}/F_{n}.pkl','rb')); k=np.asarray(m.pt.astype(str).str.strip().tolist()); return k,X.astype(float),np.asarray(s,float)
def share(k,F,P):
    F=(F-F.mean(0))/(F.std(0)+1e-12); w=[]; 
    for p in P:
        f=F[k==p]
        if len(f)>=2: w.append(f.var(0,ddof=1).mean())
    return np.mean(w)/F.var(0,ddof=1).mean()
def feat(X,t): return np.concatenate([np.log(np.abs(X[:,t,:,0])+1e-12),np.log(np.abs(X[:,t,:,1])+1e-12),X[:,t,:,2]],1)
rows=[]
for fam,b,tb in [('Z-Image','Z1','ZIT'),('ACE-Step','ACEB','ACET')]:
  for sub in ['attn','mlp']:
    kb,Xb,sb=load(f'{b}_{sub}'); kt,Xt,st=load(f'{tb}_{sub}')
    for t_t in range(len(st)):
        t_b=int(np.argmin(np.abs(sb-st[t_t])))
        P=sorted(set(kb)&set(kt)); Fb=feat(Xb,t_b); Ft=feat(Xt,t_t)
        sh_b=share(kb,Fb,P); sh_t=share(kt,Ft,P)
        bs=[]
        for _ in range(200):
            Q=list(rng.choice(P,len(P))); bs.append(share(kb,Fb,Q)-share(kt,Ft,Q))
        rows.append(dict(family=fam,sub=sub,sigma_turbo=st[t_t],sigma_base=sb[t_b],seed_share_base=sh_b,seed_share_turbo=sh_t,diff=sh_b-sh_t,ci_lo=np.percentile(bs,2.5),ci_hi=np.percentile(bs,97.5),n_prompts=len(P)))
R=pd.DataFrame(rows); R.to_csv('/tmp/claude-0/survey/explore/x3_seed_share.csv',index=False); print(R.round(3).to_string(index=False))
