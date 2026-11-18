from ph_common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
# ---- RC39 / RC46 first-state decoding (u3_content.py P5 code, unchanged) ----
def clf(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
def acc(F, y, grp): return float(cross_val_score(clf(), F, y, groups=grp, cv=GroupKFold(5)).mean())
def scr_one(a):
    F0, y, grp, seed = a; rng = np.random.default_rng(seed); Fs = np.stack([f[rng.permutation(f.shape[0])] for f in F0]).reshape(len(F0), -1); return acc(Fs, y, grp)
def decode(n, sub, X, pt, dom, pool, t=0):
    y = np.asarray([dom[p] for p in pt]); grp = pd.factorize(pt)[0]; pl = np.array([len(p) for p in pt], float)
    F0 = np.log(np.abs(X[:, t]) + 1e-9); real = acc(F0.reshape(len(X), -1), y, grp)
    sc = np.array(pool.map(scr_one, [(F0, y, grp, k) for k in range(99)]))
    txt = TfidfVectorizer(min_df=1).fit_transform(list(pt))
    T = X.shape[1]
    return dict(model=n, sub=sub, gens=len(X), prompts=len(set(pt)), ordered=real, layer_mean=acc(F0.mean(1), y, grp), prompt_length=acc(pl[:, None], y, grp),
                tfidf=float(cross_val_score(LogisticRegression(C=10, max_iter=3000), txt, y, groups=grp, cv=GroupKFold(5)).mean()),
                scramble_max=float(sc.max()), p=float((1 + (sc >= real).sum()) / 100), beats_all=bool(real > sc.max()),
                ordered_mid=acc(np.log(np.abs(X[:, T // 2]) + 1e-9).reshape(len(X), -1), y, grp), ordered_last=acc(np.log(np.abs(X[:, T - 1]) + 1e-9).reshape(len(X), -1), y, grp))
