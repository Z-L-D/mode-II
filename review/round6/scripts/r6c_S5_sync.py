import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# S5: synchronization order parameter R(sigma) (09-15 summary section 19) against a per-layer monotone time-warp null and the
# Round 1 linear-trend surrogate. Population-mean field. Output: out/S5_sync.csv, out/S5_results.json
from r6c_common import *
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round2/scripts'); from common import syncR
rng = np.random.default_rng(65); NN = 500
def warp_null(M):
    T, L, _ = M.shape; u = np.linspace(0, 1, T); out = np.empty_like(M)
    for l in range(L):
        c = np.concatenate([[0], np.cumsum(rng.dirichlet(np.ones(T - 1)))]); c[-1] = 1.0
        for v in range(3): out[:, l, v] = np.interp(c, u, M[:, l, v])
    return out
def trend(M):
    T = M.shape[0]; t = np.arange(T); A = np.stack([np.ones(T), t], 1); out = np.empty_like(M)
    for l in range(M.shape[1]):
        for v in range(3): c, *_ = np.linalg.lstsq(A, M[:, l, v], rcond=None); out[:, l, v] = A @ c
    return out
rows = []
for m, sub in [('ZIT', 'attn'), ('ZIT', 'mlp'), ('ZIB', 'attn'), ('ZIB', 'mlp'), ('MOCHI', 'attn'), ('SD35M', 'attn'), ('SD35L', 'attn')]:
    g, meta, X, s = arch(m, sub); M = X.mean(0); R = syncR(M); pk = float(R.max()); ip = int(R.argmax())
    nul = np.array([syncR(warp_null(M)).max() for _ in range(NN)]); tr = float(syncR(trend(M)).max())
    rows.append(dict(channel=f'{m}-{sub}', R_peak=pk, peak_index=ip, steps=len(s), peak_sigma=float(s[ip]), warp_null_median=float(np.median(nul)),
                     warp_null_p95=float(np.percentile(nul, 95)), warp_p=float((1 + (nul >= pk).sum()) / (NN + 1)), trend_R_peak=tr))
S5 = pd.DataFrame(rows); S5.to_csv(f'{OUT6}/S5_sync.csv', index=False)
na = int((S5.R_peak > S5.warp_null_p95).sum()); nb = int((S5.R_peak <= S5.trend_R_peak).sum())
sc = dict(S5a=('CONFIRMED' if na >= 5 else 'REFUTED', f'{na}/7 channels with R_peak above the warp-null p95'),
          S5b=('CONFIRMED' if nb == 7 else 'REFUTED', f'{nb}/7 channels with R_peak <= linear-trend surrogate'))
jdump(dict(score=sc), 'S5_results.json'); pd.set_option('display.width', 250); print(S5.round(3).to_string())
for k, v in sc.items(): print(k, v)
