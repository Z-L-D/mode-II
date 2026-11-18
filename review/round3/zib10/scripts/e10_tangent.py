import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# E10 / P10: lineage calibration. Tangent cosine (depth derivative of depth-normalized profile, matched sigma) vs shape r.
import pickle, sys; sys.path.insert(0,'.'); from common import *
rng=np.random.default_rng(10); gr=np.linspace(0,1,24)
def prof(c,sub):
    g,m,X,s=pickle.load(open(f'F_{c}_{sub}.pkl','rb')); D=depthnorm(X.mean(0)); u=np.linspace(0,1,D.shape[1])
    return np.stack([[np.interp(gr,u,D[t,:,k]) for k in range(3)] for t in range(len(s))]).transpose(0,2,1),s,D
def cos(a,b): a=a.ravel(); b=b.ravel(); return a@b/np.linalg.norm(a)/np.linalg.norm(b)
P={k:prof(*k) for k in [('ZIT','attn'),('ZIB','attn'),('ZIT','mlp'),('ZIB','mlp'),('MOCHI','attn'),('SD35M','attn'),('SD35L','attn'),('MOCHI1','attn')]}
def pair(a,b):
    A,sa,Da=P[a]; B,sb,Db=P[b]; tc=[];sr=[];nul=[]
    for t,x in enumerate(sa):
        j=int(np.argmin(np.abs(sb-x))); dA=np.diff(A[t],axis=0); dB=np.diff(B[j],axis=0)
        tc.append(cos(dA,dB)); sr.append(shape_r(A[t],B[j]))
    # layer-order permutation null: permute native layers of b, renormalize, interpolate, derivative
    for _ in range(200):
        perm=rng.permutation(Db.shape[1]); vals=[]
        for t,x in enumerate(sa):
            j=int(np.argmin(np.abs(sb-x))); Bp=Db[j][perm]; u=np.linspace(0,1,len(perm))
            Bi=np.stack([np.interp(gr,u,Bp[:,k]) for k in range(3)],1); vals.append(cos(np.diff(A[t],axis=0),np.diff(Bi,axis=0)))
        nul.append(np.mean(vals))
    m=np.mean(tc); return dict(a='-'.join(a),b='-'.join(b),tangent_cos=m,shape_r=np.mean(sr),perm_null_p95=np.percentile(nul,95),perm_pct=(np.array(nul)<m).mean())
rows=[pair(('ZIT','attn'),('ZIB','attn')),pair(('ZIT','mlp'),('ZIB','mlp')),pair(('MOCHI','attn'),('MOCHI1','attn')),pair(('SD35M','attn'),('SD35L','attn')),pair(('ZIT','attn'),('ZIT','mlp'))]
rel=len(rows)
unrel=[('ZIT','attn'),('ZIB','attn'),('MOCHI','attn'),('SD35M','attn'),('SD35L','attn')]
fam={'ZIT':'Z','ZIB':'Z','MOCHI':'M','SD35M':'S','SD35L':'S'}
for i,a in enumerate(unrel):
    for b in unrel[i+1:]:
        if fam[a[0]]!=fam[b[0]]: rows.append(pair(a,b))
R=pd.DataFrame(rows); R['kind']=['lineage','lineage','same-model CFG','family size','attn-vs-mlp']+['unrelated']*(len(R)-rel)
print(R.round(3).to_string()); R.to_csv(f'{_COSPAFS_ROOT}/review/round3/zib10/out/E10_tangent.csv',index=False)
u=R[R.kind=='unrelated']
print('max |tangent| unrelated %.3f  max shape_r unrelated %.3f'%(u.tangent_cos.abs().max(),u.shape_r.max()))
for k in [0,1]: print(R.a[k],'tangent margin %.3f  shape margin %.3f'%(R.tangent_cos[k]-u.tangent_cos.max(),R.shape_r[k]-u.shape_r.max()))
