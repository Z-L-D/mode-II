# Round 18, descriptive (not scored): native-grid tangent with the RC33 statistics (perm95 1000 draws rng 33, surrogate p95
# 300 draws, generation bootstrap 200 x 30, p10k) for the new pairs, so they enter Mode-I Table 4 on the same footing.
import os, sys, pickle, numpy as np, pandas as pd
from multiprocessing import Pool
SCR = os.environ['SCR']; R18 = os.environ['R18CUBES']; OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
NEW = {'SD35L18', 'SD35L18U', 'SD35LT', 'ZIT2', 'ZB6', 'ZB6U', 'S_SD35M', 'S_SD35M1'}
def load(n, sub, img):
    d = R18 if n in NEW else SCR
    if img:
        o = pickle.load(open(f'{d}/V_{n}_{sub}.pkl', 'rb')); return np.asarray(o['V'][..., [o['cols'].index(c) for c in ('g_img', 'gap_img', 'H_img')]], float), np.asarray(o['s'], float)
    g, m, X, s = pickle.load(open(f'{d}/F_{n}_{sub}.pkl', 'rb')); return np.asarray(X, float), np.asarray(s, float)
def dn(M):
    Y = M.copy(); Y[..., 0] = np.log(np.abs(Y[..., 0]) + 1e-30); Y[..., 1] = np.log(np.abs(Y[..., 1]) + 1e-30)
    return (Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-12)
def prof(M, ns, n):
    D = dn(M); u = np.linspace(0, 1, D.shape[1]); G = np.linspace(0, 1, n)
    return np.stack([[np.interp(G, u, D[t, :, k]) for k in range(3)] for t in range(ns)]).transpose(0, 2, 1)
def c_(a, b): a = a.ravel(); b = b.ravel(); return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
def tan(A, sa, B, sb): return float(np.mean([c_(np.diff(A[t], axis=0), np.diff(B[int(np.argmin(np.abs(sb - sa[t])))], axis=0)) for t in range(len(sa))]))
def tan_diff(A, sa, dB, sb): return float(np.mean([c_(np.diff(A[t], axis=0), dB[int(np.argmin(np.abs(sb - sa[t])))]) for t in range(len(sa))]))
def surrogate(B, rng):
    dB = np.diff(B, axis=1); F = np.fft.rfft(dB, axis=1); ph = rng.uniform(0, 2 * np.pi, F.shape[1]); ph[0] = 0
    return np.fft.irfft(F * np.exp(1j * ph)[None, :, None], n=dB.shape[1], axis=1)
def job(p):
    kind, a, b, sub, img = p; XA, sa = load(a, sub, img); XB, sb = load(b, sub, img); MA, MB = XA.mean(0), XB.mean(0)
    n = min(MA.shape[1], MB.shape[1]); rng = np.random.default_rng(33); A = prof(MA, len(sa), n); B = prof(MB, len(sb), n)
    r = dict(kind=kind, a=a, b=b, sub=sub, rows='image' if img else 'all', La=MA.shape[1], Lb=MB.shape[1], n=n, native=tan(A, sa, B, sb))
    r['perm95'] = float(np.percentile([tan(A, sa, prof(MB[:, rng.permutation(MB.shape[1])], len(sb), n), sb) for _ in range(1000)], 95))
    k = sum(tan(A, sa, prof(MB[:, rng.permutation(MB.shape[1])], len(sb), n), sb) >= r['native'] for _ in range(10000)); r['p10k'] = (k + 1) / 10001
    r['surr95'] = float(np.percentile([tan_diff(A, sa, surrogate(B, rng), sb) for _ in range(300)], 95))
    bs = []
    for _ in range(200):
        ia = rng.choice(len(XA), min(30, len(XA)), replace=False); ib = rng.choice(len(XB), min(30, len(XB)), replace=False)
        bs.append(tan(prof(XA[ia].mean(0), len(sa), n), sa, prof(XB[ib].mean(0), len(sb), n), sb))
    r['ci05'], r['ci95'] = (float(x) for x in np.percentile(bs, [5, 95])); r['rule_margin'] = r['native'] - r['perm95']
    print({q: (round(v, 4) if isinstance(v, float) else v) for q, v in r.items()}, flush=True); return r
P = [('distillation, documented', 'SD35L18', 'SD35LT', 'attn', False), ('same weights, sessions', 'SD35L18', 'SD35L', 'attn', False),
     ('same weights, run', 'SD35L18', 'SD35L18U', 'attn', False), ('same weights, CFG 4.5 vs 1', 'S_SD35M', 'S_SD35M1', 'attn', False),
     ('sibling, shift 6', 'ZB6', 'ZIT', 'attn', False), ('sibling, shift 6', 'ZB6', 'ZIT', 'mlp', False),
     ('same weights, shift 6 vs 3', 'ZB6', 'Z1', 'attn', False), ('same weights, shift 6 vs 3', 'ZB6', 'Z1', 'mlp', False),
     ('sibling, image rows', 'Z2', 'ZIT2', 'attn', True), ('sibling, image rows', 'Z2', 'ZIT2', 'mlp', True)]
if __name__ == '__main__':
    with Pool(4) as pool: D = pd.DataFrame(pool.map(job, P))
    D.to_csv(f'{OUT}/r18_native_grid.csv', index=False); pd.set_option('display.width', 250); print(D.round(4).to_string(index=False))
