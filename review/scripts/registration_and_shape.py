# ZIT->ZIB optimized warp vs equal-sigma map; cross-model shape-r calibration matrix.
import pickle; from common import *
for sub in ['attn','mlp']:
    gt,mt,Xt,st=pickle.load(open(f'F_ZIT_{sub}.pkl','rb')); gb,mb,Xb,sb=pickle.load(open(f'F_ZIB_{sub}.pkl','rb'))
    sh=sorted(set(mt.pt)&set(mb.pt)); Dt=depthnorm(Xt[np.isin(mt.pt,sh)].mean(0)); Db=depthnorm(Xb[np.isin(mb.pt,sh)].mean(0))
    C=np.array([[shape_r(Dt[i],Db[j]) for j in range(len(sb))] for i in range(len(st))]); eq=[int(np.argmin(np.abs(sb-x))) for x in st]
    print(sub,'warp',C.argmax(1),'r %.3f'%C.max(1).mean(),'| equal-sigma',eq,'r %.3f'%np.mean([C[i,j] for i,j in enumerate(eq)]))
def prof(c,sub):
    g,m,X,s=pickle.load(open(f'F_{c}_{sub}.pkl','rb')); D=depthnorm(X.mean(0)); u=np.linspace(0,1,D.shape[1]); gr=np.linspace(0,1,24)
    return np.stack([[np.interp(gr,u,D[t,:,k]) for k in range(3)] for t in range(len(s))]).transpose(0,2,1),s
P={k:prof(*k) for k in [('ZIT','attn'),('ZIB','attn'),('ZIT','mlp'),('ZIB','mlp'),('MOCHI','attn'),('SD35M','attn'),('SD35L','attn'),('MOCHI1','attn')]}
for a in P:
    print(a,[round(np.mean([shape_r(P[a][0][t],P[b][0][int(np.argmin(np.abs(P[b][1]-x)))]) for t,x in enumerate(P[a][1])]),2) for b in P])
