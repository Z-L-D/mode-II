# Peer review 2026-10-17 (post hoc): first-state domain decoding on Round 3 matched captures,
# same decoder and features as papers/mode-I/scripts/fig_I_F05_semantic.py (Round 1 decoder).
# Arms (same 60 prompts): ZIB CFG1 cond-only; SD3.5M CFG1 cond-only (seeds 1001+1002); SD3.5M CFG4.5 pooled (seeds 1001+1002).
import pickle, sys, warnings, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings('ignore'); C = sys.argv[1]; NSCR = 99
def clf(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
def acc(F, y, grp): return float(cross_val_score(clf(), F, y, groups=grp, cv=GroupKFold(5)).mean())
ARMS = {'ZIB CFG1 cond-only attn': ['F3_Z10_1024_s1001_attn'], 'ZIB CFG1 cond-only mlp': ['F3_Z10_1024_s1001_mlp'],
        'ZIB CFG4 branch-mean attn': ['F3_Z40_1024_s1001_attn'], 'ZIB CFG4 branch-mean mlp': ['F3_Z40_1024_s1001_mlp'],
        'SD3.5M CFG1 cond-only attn': ['F3_S10_1024_s1001_attn', 'F3_S10_1024_s1002_attn'],
        'SD3.5M CFG4.5 pooled attn': ['F3_S45_1024_s1001_attn', 'F3_S45_1024_s1002_attn']}
rows = []
for lab, files in ARMS.items():
    Ms, Xs = [], []
    for f in files:
        g, m, X, s = pickle.load(open(f'{C}/{f}.pkl', 'rb')); Ms.append(m); Xs.append(X)
    m = pd.concat(Ms, ignore_index=True); X = np.concatenate(Xs)
    y = np.asarray(m.theme.astype(str)); grp = pd.factorize(np.asarray(m.pt.astype(str)))[0]
    F0 = np.log(np.abs(X[:, 0]) + 1e-9)
    real = acc(F0.reshape(len(X), -1), y, grp); lm = acc(F0.mean(1), y, grp)
    txt = TfidfVectorizer(min_df=1).fit_transform(list(m.pt.astype(str)))
    tx = float(cross_val_score(LogisticRegression(C=10, max_iter=3000), txt, y, groups=grp, cv=GroupKFold(5)).mean())
    rng = np.random.default_rng(0); null = []
    for k in range(NSCR):
        Fs = np.stack([f[rng.permutation(f.shape[0])] for f in F0]).reshape(len(F0), -1); null.append(acc(Fs, y, grp))
    null = np.array(null)
    rows.append(dict(arm=lab, gens=len(X), prompts=len(np.unique(grp)), ordered=real, layer_mean=lm, tfidf=tx,
                     scr_max=null.max(), scr_p95=np.quantile(null, .95), p=(1 + (null >= real).sum()) / (1 + NSCR)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
print(pd.DataFrame(rows).round(3).to_string())
