import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# E9 calibration: same estimator on Z-Image (where the ledger reported cross-sector phi dominance).
import sys; sys.path.insert(0,'.'); from common import *
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
rows=[]
for c in ['ZIT','ZIB']:
    df,meta,ns,L=load(c)
    for sub in ['attn','mlp']:
        d=df[df.submodule==sub].sort_values(['gid','token_idx','layer']).copy()
        d['beta']=d.groupby(['gid','token_idx']).g_micro.shift(-1)-d.g_micro; d=d.dropna(subset=['beta','phi_micro','phi_micro_skip'])
        d['depth']=d.layer; d['sig']=d.sigma; base=['g_micro','spectral_gap','sv_entropy','depth','sig']
        y=d.beta.values; grp=pd.factorize(d.gid)[0]; r={}
        for name,feats in [('base',base),('cross',base+['phi_micro']),('skip',base+['phi_micro_skip'])]:
            X=d[feats].values; pred=np.zeros_like(y)
            for tr,te in GroupKFold(5).split(X,groups=grp): pred[te]=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.08,random_state=0).fit(X[tr],y[tr]).predict(X[te])
            r[name]=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
        rows.append(dict(ch=f'{c}-{sub}',R2_base=r['base'],gain_cross=r['cross']-r['base'],gain_skip=r['skip']-r['base'])); print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(f'{_COSPAFS_ROOT}/review/round2/out/E9b_zimage_calibration.csv',index=False)
