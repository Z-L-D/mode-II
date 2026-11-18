# RC54-RC55 (post hoc, round-8 reviews 2026-11-11: GLM M3(b), M5; ChatGPT 1.3; DeepSeek 3).
# RC54  Depth-frequency decomposition of the tangent. On the native grid (n = hooked layers of the shallower model) the
#       depth differences of the standardised profiles of each state are expanded in an orthonormal DCT-II basis along
#       depth. The tangent cosine is recomputed on three bands of the n-1 coefficients (low: k < (n-1)/4; mid:
#       (n-1)/4 <= k < (n-1)/2; high: k >= (n-1)/2) and on the coefficients a 12-point grid can represent (k < 11), with a
#       300-draw layer-order permutation null (rng 54) per band. Also reported: the share of difference energy per band.
#       Because the basis is orthonormal the full-band cosine equals the native tangent (check column).
# RC55  MLP variants on the two pairs with row-split cubes: all rows, image rows, centred, top-8 channels removed,
#       each with and without the first and last hooked layer (native grid, perm95 300 draws, surrogate p95 300 draws).
# Usage: python rc54_rc55_round8.py <cache> <r16cubes> <r17cubes> <r18cubes> <outdir>
import os, sys, pickle, numpy as np, pandas as pd
from multiprocessing import Pool
sys.argv, ARGS = sys.argv[:4], sys.argv
HERE = sys.path[0]
RCD = os.environ.get('RCDIR'); sys.path.insert(0, RCD)
import rc22_round5_checks as R
from scipy.fft import dct
R18, OUT = ARGS[4], ARGS[5]; os.makedirs(OUT, exist_ok=True)
R.SRC.update({'SD35L18': (R18, 'SD35L18'), 'SD35LT': (R18, 'SD35LT'), 'ZB6': (R18, 'ZB6'), 'ZIT2': (R18, 'ZIT2'), 'SD35L18U': (R18, 'SD35L18U')})
def prof(M, ns, n):
    D = R.depthnorm(M); u = np.linspace(0, 1, D.shape[1]); G = np.linspace(0, 1, n)
    return np.stack([[np.interp(G, u, D[t, :, k]) for k in range(3)] for t in range(ns)]).transpose(0, 2, 1)
def coefs(A):  # T, n, 3 -> T, n-1, 3 DCT coefficients of depth differences
    return dct(np.diff(A, axis=1), type=2, norm='ortho', axis=1)
def bands(m):
    return {'full': np.arange(m), 'low': np.arange(m) < m / 4, 'mid': (np.arange(m) >= m / 4) & (np.arange(m) < m / 2),
            'high': np.arange(m) >= m / 2, 'grid12': np.arange(m) < 11}
def bcos(CA, sa, CB, sb, sel):
    v = []
    for t in range(len(sa)):
        j = int(np.argmin(np.abs(sb - sa[t]))); a = CA[t][sel]; b = CB[j][sel]
        v.append(float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300)))
    return float(np.mean(v))
def tanv(A, sa, B, sb):
    return float(np.mean([R.cos(np.diff(A[t], axis=0), np.diff(B[int(np.argmin(np.abs(sb - sa[t])))], axis=0)) for t in range(len(sa))]))
P54 = [('same weights, sessions', 'Z3', 'ZIT', s) for s in ('attn', 'mlp')] + [('same weights, sessions', 'SD35M17', 'SD3SV', 'attn'),
       ('same weights, sessions', 'T8', 'ACET', 'attn'), ('same weights, sessions', 'SD35L18', 'SD35L', 'attn'),
       ('same weights, run', 'PIXART', 'PIXARTU', 'attn'), ('successor', 'SD3M', 'SD35M17', 'attn'),
       ('distillation, documented', 'SD35L18', 'SD35LT', 'attn'), ('distillation', 'ACEB', 'T8', 'attn'), ('distillation', 'ACEB', 'T8', 'mlp'),
       ('sibling', 'Z1', 'ZIT', 'attn'), ('sibling', 'Z1', 'ZIT', 'mlp'), ('unrelated', 'PIXART', 'COSMOS2B', 'attn'), ('unrelated', 'PIXART', 'COSMOS2B', 'mlp'),
       ('unrelated, cross-modal', 'ACEB', 'SD35M17', 'attn'), ('unrelated, cross-modal', 'T8', 'SD3M', 'attn'), ('unrelated, cross-modal', 'ACET', 'SD3SV', 'attn')]
def job54(p):
    kind, a, b, sub = p; XA, sa = R.load(a, sub); XB, sb = R.load(b, sub); MA, MB = XA.mean(0), XB.mean(0)
    n = min(MA.shape[1], MB.shape[1]); A = prof(MA, len(sa), n); B = prof(MB, len(sb), n); CA, CB = coefs(A), coefs(B)
    bd = bands(n - 1); rng = np.random.default_rng(54); r = dict(kind=kind, a=a, b=b, sub=sub, n=n, native=tanv(A, sa, B, sb))
    perms = [coefs(prof(MB[:, rng.permutation(MB.shape[1])], len(sb), n)) for _ in range(300)]
    for k, sel in bd.items():
        r[k] = bcos(CA, sa, CB, sb, sel); r[k + '_p95'] = float(np.percentile([bcos(CA, sa, C, sb, sel) for C in perms], 95))
        if k != 'full': r[k + '_energyA'] = float((CA[:, sel] ** 2).sum() / (CA ** 2).sum())
    print({q: (round(v, 3) if isinstance(v, float) else v) for q, v in r.items()}, flush=True); return r
VARS = {'all': None, 'image': ('g_img', 'gap_img', 'H_img'), 'centred': ('g_ctr', 'gap_ctr', 'H_ctr'), 'top8 removed': ('g_topk', 'gap_topk', 'H_topk')}
def loadv(n, sub, var):
    if VARS[var] is None: return R.load(n, sub)
    d, f = R.SRC.get(n, (R.CA, n)); o = pickle.load(open(f'{d}/V_{f}_{sub}.pkl', 'rb'))
    X = o['V'][..., [o['cols'].index(c) for c in VARS[var]]].astype(float); _, s = R.load(n, sub)
    return X, s
P55 = [(kind, a, b, var, trim) for kind, a, b in (('sibling', 'Z2', 'ZITSV'), ('unrelated', 'PIXART', 'COSMOS2B'), ('same weights, run', 'PIXART', 'PIXARTU'))
       for var in VARS for trim in (False, True)]
def job55(p):
    kind, a, b, var, trim = p; XA, sa = loadv(a, 'mlp', var); XB, sb = loadv(b, 'mlp', var); MA, MB = XA.mean(0), XB.mean(0)
    if trim: MA, MB = MA[:, 1:-1], MB[:, 1:-1]
    n = min(MA.shape[1], MB.shape[1]); A = prof(MA, len(sa), n); B = prof(MB, len(sb), n); rng = np.random.default_rng(55)
    r = dict(kind=kind, a=a, b=b, variant=var, end_trim=trim, n=n, tangent=tanv(A, sa, B, sb))
    r['perm95'] = float(np.percentile([tanv(A, sa, prof(MB[:, rng.permutation(MB.shape[1])], len(sb), n), sb) for _ in range(300)], 95))
    r['surr95'] = float(np.percentile([R.tan_diff(A, sa, R.surrogate(B, rng), sb) for _ in range(300)], 95))
    print({q: (round(v, 3) if isinstance(v, float) else v) for q, v in r.items()}, flush=True); return r
if __name__ == '__main__':
    with Pool(4) as pool: D54 = pd.DataFrame(pool.map(job54, P54)); D55 = pd.DataFrame(pool.map(job55, P55))
    D54.to_csv(f'{OUT}/rc54_tangent_bands.csv', index=False); D55.to_csv(f'{OUT}/rc55_mlp_variants.csv', index=False)
    pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40)
    print(D54.round(3).to_string(index=False)); print(D55.round(3).to_string(index=False))
