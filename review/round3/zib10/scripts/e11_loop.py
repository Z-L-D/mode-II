import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# E11 / P11: step-flow loop criterion per layer: >=2 sign changes of beta_step AND |winding about centroid| >= 0.5 turns, (g, beta_step) z-scored per layer.
import pickle, sys; sys.path.insert(0,'.'); from common import *
def crit(g):
    b=np.diff(g); x=g[:-1]; sc=(np.sign(b[1:])!=np.sign(b[:-1])).sum()
    P=np.stack([(x-x.mean())/(x.std()+1e-15),(b-b.mean())/(b.std()+1e-15)],1); c=P-P.mean(0)
    ang=np.unwrap(np.arctan2(c[:,1],c[:,0])); w=(ang[-1]-ang[0])/(2*np.pi)
    return sc, w, (sc>=2) and abs(w)>=0.5
rows=[]; summ=[]
for c in ['ZIT','ZIB','MOCHI','MOCHI1','SD35M','SD35L']:
    subs=['attn','mlp'] if c in ('ZIT','ZIB') else ['attn']
    res={}
    for sub in subs:
        g,m,X,s=pickle.load(open(f'F_{c}_{sub}.pkl','rb')); M=X.mean(0)[...,0]
        for l in range(M.shape[1]):
            sc,w,ok=crit(M[:,l]); rows.append(dict(ch=c,sub=sub,layer=l,sign_changes=sc,winding=w,loop=ok)); res.setdefault(l,[]).append(ok)
        # per-generation fraction (prior note: 87/87 gens): a gen has a loop if any layer meets it
        pg=[np.mean([crit(X[i,:,l,0])[2] for l in range(X.shape[2])]) for i in range(min(len(X),120))]
        summ.append(dict(ch=c,sub=sub,frac_layers_popmean=np.mean([r['loop'] for r in rows if r['ch']==c and r['sub']==sub]),
                         mean_frac_layers_per_gen=np.mean(pg),frac_gens_any_layer=np.mean(np.array(pg)>0)))
    summ.append(dict(ch=c,sub='either',frac_layers_popmean=np.mean([any(v) for v in res.values()])))
S=pd.DataFrame(summ); print(S.round(3).to_string())
pd.DataFrame(rows).to_csv(f'{_COSPAFS_ROOT}/review/round3/zib10/out/E11_loop_by_layer.csv',index=False); S.to_csv(f'{_COSPAFS_ROOT}/review/round3/zib10/out/E11_loop_summary.csv',index=False)
