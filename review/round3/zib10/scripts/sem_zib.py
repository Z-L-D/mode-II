import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import pickle
import sys; sys.path.insert(0,f'{_COSPAFS_ROOT}/review/scripts')
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings; warnings.filterwarnings('ignore')
def clf(): return make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=3000))
for c,sub in [('ZIB','attn'),('ZIB','mlp')]:
    gids,meta,X,s=pickle.load(open(f'F_{c}_{sub}.pkl','rb'))
    y=np.asarray(meta.dom.astype(str)); grp=pd.factorize(np.asarray(meta.pt.astype(str)))[0]; cv=GroupKFold(5)
    T=X.shape[1]; ts=sorted(set([0,1,T//4,T//2,3*T//4,T-1]))
    acc={}
    for t in ts:
        F=np.log(np.abs(X[:,t]).reshape(len(X),-1)+1e-9)
        acc[t]=cross_val_score(clf(),F,y,groups=grp,cv=cv).mean()
    # scramble null at t0: permute layer order independently per generation
    rng=np.random.default_rng(0); F0=np.log(np.abs(X[:,0])+1e-9)
    Fs=np.stack([f[rng.permutation(f.shape[0])] for f in F0]).reshape(len(X),-1)
    scr=cross_val_score(clf(),Fs,y,groups=grp,cv=cv).mean()
    # layer-mean only
    Fm=F0.mean(1); lm=cross_val_score(clf(),Fm,y,groups=grp,cv=cv).mean()
    # text-only baseline: tfidf of prompt text, grouped CV
    txt=TfidfVectorizer(min_df=1).fit_transform(list(meta.pt.astype(str)))
    tx=cross_val_score(LogisticRegression(C=10,max_iter=3000),txt,y,groups=grp,cv=cv).mean()
    # same-prompt retrieval (1-NN excluding self) at t0 and last
    def retr(t):
        F=np.log(np.abs(X[:,t]).reshape(len(X),-1)+1e-9); F=(F-F.mean(0))/F.std(0)
        D=((F[:,None]-F[None])**2).sum(-1); np.fill_diagonal(D,np.inf); nn=D.argmin(1)
        return (grp[nn]==grp).mean()
    print(f'{c:6s}{sub:5s} chance {1/len(set(y)):.2f} | domain acc by t '+' '.join(f'{t}:{a:.3f}' for t,a in acc.items())+f' | t0 scramble {scr:.3f} | t0 layer-mean {lm:.3f} | TEXT tfidf {tx:.3f} | same-prompt 1NN retrieval t0 {retr(0):.2f} last {retr(T-1):.2f} (chance ~{3/len(X):.3f})',flush=True)
