# RC12 (post hoc, GLM Q8): per-domain recall of the TF-IDF prompt-text decoder next to the telemetry decoder, ZIT and ZIT-25 attention, first state.
# Same decoder as the archive decoding (logistic regression C=10, prompt-grouped 5-fold); TF-IDF as in r3_ace/a2_posthoc.py.
import os, pickle, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
SCR = os.environ.get('SCR', os.path.join(ROOT, 'cache')); OUT = os.path.join(HERE, 'out')
rows = []
for n in ('ZIT_attn', 'ZIB_attn', 'SD35M_attn', 'MOCHI_attn'):
    g, m, X, s = pickle.load(open(f'{SCR}/F_{n}.pkl', 'rb'))
    pt = np.asarray(m.pt.astype(str).str.strip().tolist()); y = np.asarray(m.dom.astype(str).tolist()); grp = pd.factorize(pt)[0]
    F0 = np.log(np.abs(X[:, 0].astype(float)) + 1e-9).reshape(len(X), -1)
    pt_ = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=3000)), F0, y, groups=grp, cv=GroupKFold(5))
    tx = cross_val_predict(LogisticRegression(C=10, max_iter=3000), TfidfVectorizer(min_df=1).fit_transform(list(pt)), y, groups=grp, cv=GroupKFold(5))
    for d in np.unique(y):
        k = y == d; rows.append(dict(channel=n, domain=d, n=int(k.sum()), telemetry_recall=float((pt_[k] == d).mean()), tfidf_recall=float((tx[k] == d).mean())))
    rows.append(dict(channel=n, domain='ALL', n=len(y), telemetry_recall=float((pt_ == y).mean()), tfidf_recall=float((tx == y).mean())))
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/rc12_tfidf_per_domain.csv', index=False)
print(R.pivot_table(index='domain', columns='channel', values=['telemetry_recall', 'tfidf_recall']).round(2).to_string())
