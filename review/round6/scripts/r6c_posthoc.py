# Round 6 post hoc checks (labelled post hoc; run after S1, S2, S4 and S5 were scored). Output: out/posthoc_R6C.json
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from r6c_common import *
rng = np.random.default_rng(66); PH = {}
# ---- P1: the attention topology shared by all five architectures (S4e was refuted on '+-+') ----

def fit3(y, s):
    u = -s; T = len(y); best = None
    for i in range(1, T - 2):
        for j in range(i + 1, T - 1):
            A = np.stack([np.ones(T), u, np.maximum(0, u - u[i]), np.maximum(0, u - u[j])], 1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None); sse = ((A @ coef - y) ** 2).sum()
            if best is None or sse < best[0]: best = (sse, i, j, coef)
    sse, i, j, c = best; sl = np.array([c[1], c[1] + c[2], c[1] + c[2] + c[3]])
    return ''.join('+' if v > 0 else '-' for v in sl), i, j, sl
P1 = {}
for m in MODELS:
    g, meta, X, s = arch(m); k = meta.pt.astype(str).str.strip().values; up = np.unique(k)
    PP = np.stack([X[k == p][..., 0].mean((0, 2)) for p in up]); tops = []
    for _ in range(500):
        tops.append(fit3(PP[rng.integers(0, len(PP), len(PP))].mean(0), s)[0])
    y = PP.mean(0); t, i, j, sl = fit3(y, s); vals, cnt = np.unique(tops, return_counts=True)
    P1[m] = dict(topology=t, modal=str(vals[cnt.argmax()]), modal_frac=float(cnt.max() / len(tops)), break1=float(s[i]), break2=float(s[j]),
                 g_first=float(y[0]), g_max=float(y.max()), sigma_max=float(s[y.argmax()]), g_last=float(y[-1]), late_drop=float((y[-1] - y.max()) / y.max()))
PH['P1_attention_topology'] = P1
# ---- P2: is cross-model content agreement only theme blocks or text overlap? ----
D = {m: arch(m) for m in MODELS}
P = sorted(set.intersection(*[set(D[m][1].pt.astype(str).str.strip()) for m in MODELS])); NP = len(P)
dom = np.asarray(D['ZIT'][1].assign(k=D['ZIT'][1].pt.astype(str).str.strip()).groupby('k').dom.first().loc[P].astype(str).tolist())
def feats(m, tau):
    g, meta, X, s = D[m]; T = X.shape[1]; t = {'first': 0, 'middle': T // 2, 'last': T - 1}[tau]; k = meta.pt.astype(str).str.strip().values
    Y = X[:, t].copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1])
    F = np.stack([np.concatenate([np.interp(gr24, np.linspace(0, 1, Y.shape[1]), Y[i, :, v]) for v in range(3)]) for i in range(len(Y))])
    M = np.stack([F[k == p].mean(0) for p in P]); return (M - M.mean(0)) / (M.std(0) + 1e-12)
iu = np.triu_indices(NP, 1); same = (dom[:, None] == dom[None, :])[iu]
V = TfidfVectorizer().fit_transform(P); TXT = 1 - (V @ V.T).toarray(); tx = rankdata(TXT[iu])
def within_perm(perm_dom=True):
    p = np.arange(NP)
    for d in np.unique(dom):
        idx = np.where(dom == d)[0]; p[idx] = rng.permutation(idx)
    return p
def resid_on(a, b): A = np.stack([np.ones_like(b), b], 1); c, *_ = np.linalg.lstsq(A, a, rcond=None); return a - A @ c
rows = []
for a, b in [('ZIT', 'ZIB')] + E10_UNREL + [('SD35M', 'SD35L')]:
    for tau in ['first', 'middle', 'last']:
        FA, FB = feats(a, tau), feats(b, tau)
        # (i) within-theme RSA: features centred within theme (removes theme means), pairs restricted to same-theme prompts
        def centre(F): F = F.copy(); [F.__setitem__(dom == d, F[dom == d] - F[dom == d].mean(0)) for d in np.unique(dom)]; return F
        RA = 1 - np.corrcoef(centre(FA)); RB = 1 - np.corrcoef(centre(FB))
        ra = rankdata(RA[iu][same]); rb = RB[iu]
        r_w = np.corrcoef(ra, rankdata(rb[same]))[0, 1]; nul = []
        for _ in range(500):
            p = within_perm(); nul.append(np.corrcoef(ra, rankdata(RB[np.ix_(p, p)][iu][same]))[0, 1])
        # (ii) partial Mantel controlling text RSA (rank residuals), full matrices, prompt-permutation null
        RA0 = 1 - np.corrcoef(FA); RB0 = 1 - np.corrcoef(FB); xa = resid_on(rankdata(RA0[iu]), tx); xb = resid_on(rankdata(RB0[iu]), tx)
        r_p = np.corrcoef(xa, xb)[0, 1]; nulp = []
        for _ in range(500):
            p = rng.permutation(NP); nulp.append(np.corrcoef(xa, resid_on(rankdata(RB0[np.ix_(p, p)][iu]), tx))[0, 1])
        rows.append(dict(a=a, b=b, kind=kind(a, b), tau=tau, within_theme_rsa=r_w, within_p=float((1 + (np.array(nul) >= r_w).sum()) / 501),
                         partial_text_rsa=r_p, partial_p=float((1 + (np.array(nulp) >= r_p).sum()) / 501)))
P2 = pd.DataFrame(rows); P2.to_csv(f'{OUT6}/P2_rsa_robustness.csv', index=False); PH['P2'] = P2.to_dict('records')
jdump(PH, 'posthoc_R6C.json')
pd.set_option('display.width', 250); print(pd.DataFrame(P1).T.round(3).to_string())
print(P2.pivot_table(index=['a', 'b'], columns='tau', values='within_theme_rsa').round(3)); print(P2.pivot_table(index=['a', 'b'], columns='tau', values='within_p').round(3))
print(P2.pivot_table(index=['a', 'b'], columns='tau', values='partial_text_rsa').round(3))
