# S2: content geometry (theme residuals and prompt-level RSA) on the 117 prompts shared by all five archive models (attn).
# Output: out/S2_theme_residual.csv, out/S2_rsa.csv, out/S2_text_rsa.csv, out/S2_results.json
from scipy.stats import spearmanr, rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from r6c_common import *
rng = np.random.default_rng(62); NPERM = 2000
D = {m: arch(m) for m in MODELS}
keys = [set(D[m][1].pt.astype(str).str.strip()) for m in MODELS]
P = sorted(set.intersection(*keys)); NP = len(P)
dom = D['ZIT'][1].assign(k=D['ZIT'][1].pt.astype(str).str.strip()).groupby('k').dom.first().loc[P].astype(str).values
TAUS = ['first', 'middle', 'last']
def feats(m, tau):
    g, meta, X, s = D[m]; T = X.shape[1]; t = {'first': 0, 'middle': T // 2, 'last': T - 1}[tau]
    k = meta.pt.astype(str).str.strip().values
    Y = X[:, t].copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1])     # [G, L, 3]
    F = np.stack([np.concatenate([np.interp(gr24, np.linspace(0, 1, Y.shape[1]), Y[i, :, v]) for v in range(3)]) for i in range(len(Y))])
    M = np.stack([F[k == p].mean(0) for p in P])
    return (M - M.mean(0)) / (M.std(0) + 1e-12)
FE = {(m, tau): feats(m, tau) for m in MODELS for tau in TAUS}
iu = np.triu_indices(NP, 1)
def rdm(F): return 1 - np.corrcoef(F)
RD = {k: rdm(v) for k, v in FE.items()}
def mantel(A, B, nperm=NPERM):
    a = rankdata(A[iu]); b = rankdata(B[iu]); r = np.corrcoef(a, b)[0, 1]; null = np.empty(nperm)
    for i in range(nperm):
        p = rng.permutation(NP); null[i] = np.corrcoef(a, rankdata(B[np.ix_(p, p)][iu]))[0, 1]
    return float(r), float((1 + (null >= r).sum()) / (nperm + 1)), float(np.percentile(null, 95))
udom = np.unique(dom)
def theme_r(FA, FB, labels):
    TA = np.stack([FA[labels == d].mean(0) for d in udom]); TB = np.stack([FB[labels == d].mean(0) for d in udom])
    return float(np.mean([np.corrcoef(TA[i], TB[i])[0, 1] for i in range(len(udom))]))
pairs = [('ZIT', 'ZIB')] + E10_UNREL + [('SD35M', 'SD35L')]
rows_t = []; rows_r = []
for a, b in pairs:
    for tau in TAUS:
        FA, FB = FE[(a, tau)], FE[(b, tau)]
        tr = theme_r(FA, FB, dom)
        nul = [theme_r(FA, FB[rng.permutation(NP)], dom) for _ in range(500)]   # null: B prompts shuffled before theme averaging
        rows_t.append(dict(a=a, b=b, kind=kind(a, b), tau=tau, theme_r=tr, null_p95=float(np.percentile(nul, 95)), p=float((1 + (np.array(nul) >= tr).sum()) / 501)))
        r, p, n95 = mantel(RD[(a, tau)], RD[(b, tau)])
        rows_r.append(dict(a=a, b=b, kind=kind(a, b), tau=tau, rsa=r, p=p, null_p95=n95))
TH = pd.DataFrame(rows_t); RS = pd.DataFrame(rows_r)
TH.to_csv(f'{OUT6}/S2_theme_residual.csv', index=False); RS.to_csv(f'{OUT6}/S2_rsa.csv', index=False)
# text RSA (TF-IDF cosine distance of the prompt text)
V = TfidfVectorizer().fit_transform(P); C = (V @ V.T).toarray(); TXT = 1 - C
rows_x = []
for m in MODELS:
    for tau in TAUS:
        r, p, n95 = mantel(RD[(m, tau)], TXT); rows_x.append(dict(model=m, tau=tau, rsa_text=r, p=p, null_p95=n95))
TX = pd.DataFrame(rows_x); TX.to_csv(f'{OUT6}/S2_text_rsa.csv', index=False)
# within-model domain structure (sanity): theme separation = mean between-theme distance / within-theme distance
res = dict(n_prompts=NP)
# ---- scoring ----
sc = {}
lt = TH[(TH.kind == 'lineage')]; sc['S2a'] = ('CONFIRMED' if (lt.theme_r > 0.45).sum() >= 2 else 'REFUTED', 'lineage theme r ' + ', '.join(f'{r.tau} {r.theme_r:.3f}' for r in lt.itertuples()))
ut = TH[TH.kind == 'unrelated']; per = ut.groupby(['a', 'b']).theme_r.apply(lambda v: (v <= 0.20).sum() >= 2)
sc['S2b'] = ('CONFIRMED' if per.all() else 'REFUTED', f'{int(per.sum())}/8 unrelated pairs with theme r <= 0.20 at >= 2 tau; max ' + f"{ut.theme_r.max():.3f}")
lr = RS[RS.kind == 'lineage']; sc['S2c'] = ('CONFIRMED' if ((lr.rsa >= 0.30) & (lr.p < 0.01)).all() else 'REFUTED', 'lineage RSA ' + ', '.join(f'{r.tau} {r.rsa:.3f} (p {r.p:.4f})' for r in lr.itertuples()))
f = RS[RS.tau == 'first'].set_index(['a', 'b']).rsa
sc['S2d'] = ('CONFIRMED' if f[('SD35M', 'SD35L')] >= 0.20 else 'REFUTED', f"SD35M-SD35L first-step RSA {f[('SD35M', 'SD35L')]:.3f}")
t1 = f[('ZIT', 'ZIB')]; t2 = f[('SD35M', 'SD35L')]; t3 = max(f[('MOCHI', 'SD35M')], f[('MOCHI', 'SD35L')])
t4 = max(f[(a, b)] for a, b in E10_UNREL if a in ('ZIT', 'ZIB'))
sc['S2e'] = ('CONFIRMED' if t1 > t2 > t3 > t4 else 'REFUTED', f'tiers {t1:.3f} > {t2:.3f} > {t3:.3f} > {t4:.3f}')
xf = TX[TX.tau == 'first']; nx = int(((xf.rsa_text > 0) & (xf.p < 0.05)).sum())
sc['S2f'] = ('CONFIRMED' if nx >= 4 else 'REFUTED', f'{nx}/5 models; ' + ', '.join(f'{r.model} {r.rsa_text:.3f} (p {r.p:.4f})' for r in xf.itertuples()))
S1 = pd.read_csv(f'{OUT6}/S1_hierarchy.csv').set_index(['a', 'b'])
rho = spearmanr([f[k] for k in S1.index], S1.L3_tangent.values).correlation
sc['S2g'] = ('CONFIRMED' if rho < 0.5 else 'REFUTED', f'Spearman(first-step RSA, tangent) over 10 pairs = {rho:.3f}')
res['score'] = sc; jdump(res, 'S2_results.json')
pd.set_option('display.width', 250)
print(TH.pivot_table(index=['a', 'b'], columns='tau', values='theme_r').round(3)); print(RS.pivot_table(index=['a', 'b'], columns='tau', values='rsa').round(3))
print(RS.pivot_table(index=['a', 'b'], columns='tau', values='p').round(4)); print(TX.round(4).to_string())
for k, v in sc.items(): print(k, v)
