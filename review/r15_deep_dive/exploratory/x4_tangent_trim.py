# EXPLORATORY 2026-11-05: are the tangent tiers carried by the end layers? (Mode-III found the Q35~ACE depth-shape match was the last bin only.)
# Tangent (Round 5 E10 definition) recomputed with the first and/or last depth steps removed from the 23 depth differences; layer-permutation null p95 (200 perms).
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0,'/tmp/claude-0/work/mode-I/review/round5/scripts'); import os; os.environ.setdefault('SCR','/tmp/claude-0/work/mode-I/cache')
from r5_common import prof, cos
C='/tmp/claude-0/work/mode-I/cache'; rng=np.random.default_rng(3)
def cube(n): g,m,X,s=pickle.load(open(f'{C}/F_{n}.pkl','rb')); return X.astype(float),np.asarray(s,float)
def tan(XA,sa,XB,sb,lo,hi):
    A=prof(XA,sa); B=prof(XB,sb); tc=[]
    for t,x in enumerate(sa):
        j=int(np.argmin(np.abs(sb-x))); da=np.diff(A[t],axis=0)[lo:23-hi]; db=np.diff(B[j],axis=0)[lo:23-hi]; tc.append(cos(da,db))
    return float(np.mean(tc))
pairs=[('ZIT_attn','ZIB_attn','same weights'),('ZIT_mlp','ZIB_mlp','same weights'),('Z1_attn','ZIT_attn','lineage'),('Z1_mlp','ZIT_mlp','lineage'),
       ('ACEB_attn','ACET_attn','lineage'),('ACEB_mlp','ACET_mlp','lineage'),('ZIT_attn','SD35M_attn','unrelated'),('Z1_attn','SD35M_attn','unrelated'),
       ('ZIT_attn','MOCHI_attn','unrelated'),('SD35M_attn','SD35L_attn','scaling'),('ACET_attn','ZIT_attn','unrelated'),('ACET_mlp','ZIT_mlp','unrelated')]
rows=[]
for a,b,k in pairs:
    XA,sa=cube(a); XB,sb=cube(b)
    for lab,lo,hi in [('all',0,0),('drop last 1',0,1),('drop last 2',0,2),('drop first+last 2',2,2)]:
        t=tan(XA,sa,XB,sb,lo,hi)
        nul=[tan(XA,sa,XB[:,:,rng.permutation(XB.shape[2])],sb,lo,hi) for _ in range(200)]
        rows.append(dict(a=a,b=b,kind=k,trim=lab,tangent=t,null_p95=np.percentile(nul,95)))
    print(pd.DataFrame(rows[-4:]).round(3).to_string(index=False),flush=True)
pd.DataFrame(rows).to_csv('/tmp/claude-0/survey/explore/x4_tangent_trim.csv',index=False)
