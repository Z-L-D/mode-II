"""R8-12 prompt-length control for the shared RSA, as frozen in R8_PREREGISTRATION.md (SHA acc578ac, 2026-10-09T08:30:59Z).
Prompt features, RDMs and the Mantel statistic are the Round 6 S2 code (round6/scripts/r6c_S2_content.py), copied unchanged.
Partial Mantel (Smouse et al. 1986) on ranks, controlling C = |len_i - len_j| (characters of the stripped prompt text).
Permutations: 10,000, rng 2026100908 (global R8 rule). Output: out/R8_12_partial_mantel.csv, out/R8_12_results.json"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import sys; sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round6/scripts')
from scipy.stats import rankdata
from r6c_common import *
OUT = f'{_COSPAFS_ROOT}/review/r8_correlations/out'
rng = np.random.default_rng(2026100908); NPERM = 10000
D = {m: arch(m) for m in MODELS}
keys = [set(D[m][1].pt.astype(str).str.strip()) for m in MODELS]
P = sorted(set.intersection(*keys)); NP = len(P)
TAUS = ['first', 'middle', 'last']
def feats(m, tau):
    g, meta, X, s = D[m]; T = X.shape[1]; t = {'first': 0, 'middle': T // 2, 'last': T - 1}[tau]
    k = meta.pt.astype(str).str.strip().values
    Y = X[:, t].copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1])
    F = np.stack([np.concatenate([np.interp(gr24, np.linspace(0, 1, Y.shape[1]), Y[i, :, v]) for v in range(3)]) for i in range(len(Y))])
    M = np.stack([F[k == p].mean(0) for p in P])
    return (M - M.mean(0)) / (M.std(0) + 1e-12)
iu = np.triu_indices(NP, 1)
def rdm(F): return 1 - np.corrcoef(F)
RD = {(m, tau): rdm(feats(m, tau)) for m in MODELS for tau in TAUS}
ln = np.array([len(p) for p in P], float); C = np.abs(ln[:, None] - ln[None, :]); c = rankdata(C[iu])
def pc(a, b):
    rab = np.corrcoef(a, b)[0, 1]; rac = np.corrcoef(a, c)[0, 1]; rbc = np.corrcoef(b, c)[0, 1]
    return rab, (rab - rac * rbc) / np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))
rows = []
for a_, b_ in E10_UNREL + [('ZIT', 'ZIB'), ('SD35M', 'SD35L')]:
    for tau in TAUS:
        A = RD[(a_, tau)]; B = RD[(b_, tau)]; a = rankdata(A[iu]); b = rankdata(B[iu])
        full, part = pc(a, b); nf = np.empty(NPERM); npart = np.empty(NPERM)
        for i in range(NPERM):
            p = rng.permutation(NP); nf[i], npart[i] = pc(a, rankdata(B[np.ix_(p, p)][iu]))
        rows.append(dict(a=a_, b=b_, kind=kind(a_, b_), tau=tau, rsa_full=full, p_full=(1 + (nf >= full).sum()) / (NPERM + 1),
                         rsa_partial=part, p_partial=(1 + (npart >= part).sum()) / (NPERM + 1), ratio=part / full,
                         r_A_len=np.corrcoef(a, c)[0, 1], r_B_len=np.corrcoef(b, c)[0, 1]))
        print(rows[-1], flush=True)
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/R8_12_partial_mantel.csv', index=False)
u = T[T.kind == 'unrelated']; ok = (u.p_partial <= 0.01) & (u.ratio >= 0.8)
res = {'n_prompts': NP, 'len_chars_median': float(np.median(ln)), 'len_chars_range': [float(ln.min()), float(ln.max())],
       'R8-12': ('survives the length control' if ok.all() else 'does not survive the length control',
                 f'{int(ok.sum())}/{len(u)} unrelated (pair, step) cells keep p <= 0.01 and partial >= 0.8 x full; min ratio {u.ratio.min():.3f}; max |r(RDM, length)| {np.abs(T[["r_A_len", "r_B_len"]].values).max():.3f}')}
json.dump(res, open(f'{OUT}/R8_12_results.json', 'w'), indent=1); print(res)
