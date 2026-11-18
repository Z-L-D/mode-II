import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# E12 / P12: frozen 24-bin beta_layer(g) on the 120-prompt SD35M CFG4.5 campaign vs 1-prompt res suite.
import sys; sys.path.insert(0,'.'); from common import *; from beta24 import *
rng=np.random.default_rng(1); rows=[]
df,meta,ns,L=load('SD35M','attn')
ref={}
dr,mr,_,_=load('SDRES')
for r in [512,768,1024]:
    G,_,_,_=pairs(dr[dr.gid.isin(mr.index[mr.res==r])]); c,bm=beta24(G); ref[r]=[x for x,_ in crossings(c,bm)]
def run(d,label):
    G,_,_,gid=pairs(d); c,bm=beta24(G); cr=crossings(c,bm)
    # prompt-cluster bootstrap of number of crossings
    ug=np.unique(meta.loc[np.unique(gid)].pt.values); pt=meta.loc[gid].pt.values; nb=[]
    for _ in range(200):
        s=rng.choice(ug,len(ug)); idx=np.concatenate([np.where(pt==u)[0] for u in s]); nb.append(len(crossings(*beta24(G[idx]))))
    rows.append(dict(set=label,rows=G.shape[0],signs=signs(bm),n_cross=len(cr),crossings=';'.join(f'{x:.5f}' for x,_ in cr),
                     TRI=TRI(c,bm),boot_ncross_mode=np.bincount(nb).argmax(),boot_frac_ge3=np.mean(np.array(nb)>=3)))
    return c,bm
run(df,'SD35M 120p all steps')
run(df[df.token_idx>=2],'SD35M 120p steps>=2 (branch-mean only)')
for dom in sorted(meta.dom.unique()):
    run(df[(df.token_idx>=2)&df.gid.isin(meta.index[meta.dom==dom])],f'  domain {dom} steps>=2')
R=pd.DataFrame(rows); print(R.round(3).to_string())
print('res-suite crossings',{k:np.round(v,5).tolist() for k,v in ref.items()})
R.to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E12_beta_topology.csv',index=False)
