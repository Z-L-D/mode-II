# RC22 (post hoc, round-5 reviews 2026-11-09: Kimi M1/M3/M4/Q1/Q2, GLM M3/Q3/Q6/minor 10, DeepSeek Q5).
# On every depth-matched pair of the tier table (Rounds 12-17 cubes) this computes, with the Round 6 estimator unchanged
# (24-point depth grid, nearest matched clock, cosine of first differences):
#   full      canonical tangent (must reproduce the paper's values)
#   trimmed   drop the last depth interval; the first and last one; the first and last two (24-grid intervals);
#             and a native-layer trim (first and last hooked layer removed before the depth grid)
#   no_s1     the first sampling state (sigma = 1, or sigma max) of the first-named model removed
#   perm95    layer-order permutation null of the second model (300 draws) for full and each trim
#   surr95    smoothness-preserving surrogate null: the second model's depth-difference profile is phase-randomised along
#             depth with one random phase vector shared by every state and coordinate (power spectrum, hence depth
#             autocorrelation and smoothness, kept; cross-state registration and coordinate coupling kept; DC kept),
#             300 draws; FPR = share of surrogate draws at or above the lineage rule (perm95 + 0.30) and above 0.47
#   ci90      5th-95th percentile of 200 draws of 30 generations per model (as R15-5), rng 22
# Image-row variants use the Round 11/17 row-split cubes (V_*). Usage: python rc22_round5_checks.py <cache> <r16cubes> <r17cubes>
import os, sys, pickle, numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
CA, R16, R17 = sys.argv[1:4]
GR = np.linspace(0, 1, 24)
def depthnorm(M):
    Y = M.copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1]); return (Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-12)
def cos(a, b): a = a.ravel(); b = b.ravel(); return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
TCLK = {'PIXART', 'COSMOS2B'}
SRC = {'SD35M17': (R17, 'SD35M'), 'SD35MA': (CA, 'SD35M'), 'SD3M': (R17, 'SD3M'), 'SD3MU': (R17, 'SD3MU'), 'SD35M17U': (R17, 'SD35MU'),
       'PIXART': (R17, 'PIXART'), 'COSMOS2B': (R17, 'COSMOS2B'), 'PIXARTU': (R17, 'PIXARTU'), 'COSMOS2BU': (R17, 'COSMOS2BU'),
       'SD3SV': (CA, 'SD3SV'), 'T8': (R16, 'T8'), 'T3': (R16, 'T3'), 'T10': (R16, 'T10')}
_C = {}
def load(n, sub, img=False):
    key = (n, sub, img)
    if key in _C: return _C[key]
    d, f = SRC.get(n, (CA, n))
    if img:
        o = pickle.load(open(f'{d}/V_{f}_{sub}.pkl', 'rb')); idx = [o['cols'].index(c) for c in ('g_img', 'gap_img', 'H_img')]
        X = o['V'][..., idx].astype(float); s = np.asarray(o['s'], float)
    else:
        _, _, X, s = pickle.load(open(f'{d}/F_{f}_{sub}.pkl', 'rb')); X = np.asarray(X, float); s = np.asarray(s, float)
    if n.rstrip('U') in TCLK: s = s / (1 + s)
    _C[key] = (X, s); return X, s
def prof(M, ns):  # M: T,L,3 gen-mean; returns T,24,3
    D = depthnorm(M); u = np.linspace(0, 1, D.shape[1])
    return np.stack([[np.interp(GR, u, D[t, :, k]) for k in range(3)] for t in range(ns)]).transpose(0, 2, 1)
def tan_from(A, sa, B, sb, seg=None, t0=0):
    v = []
    for t in range(t0, len(sa)):
        j = int(np.argmin(np.abs(sb - sa[t]))); a = np.diff(A[t], axis=0); b = np.diff(B[j], axis=0)
        if seg is not None: a, b = a[seg], b[seg]
        v.append(cos(a, b))
    return float(np.mean(v))
SEGS = {'full': None, 'drop_last1': slice(0, 22), 'drop_1+1': slice(1, 22), 'drop_2+2': slice(2, 21)}
def variants(MA, sa, MB, sb):
    A = prof(MA, len(sa)); B = prof(MB, len(sb)); r = {k: tan_from(A, sa, B, sb, seg) for k, seg in SEGS.items()}
    r['layer_trim'] = tan_from(prof(MA[:, 1:-1], len(sa)), sa, prof(MB[:, 1:-1], len(sb)), sb)
    r['no_s1'] = tan_from(A, sa, B, sb, None, 1); return r
def surrogate(B, rng):
    dB = np.diff(B, axis=1); F = np.fft.rfft(dB, axis=1); ph = rng.uniform(0, 2 * np.pi, F.shape[1]); ph[0] = 0
    return np.fft.irfft(F * np.exp(1j * ph)[None, :, None], n=dB.shape[1], axis=1)
def tan_diff(A, sa, dB, sb, seg=None):
    v = []
    for t in range(len(sa)):
        j = int(np.argmin(np.abs(sb - sa[t]))); a = np.diff(A[t], axis=0); b = dB[j]
        if seg is not None: a, b = a[seg], b[seg]
        v.append(cos(a, b))
    return float(np.mean(v))
def job(p):
    kind, a, b, sub, img = p; XA, sa = load(a, sub, img); XB, sb = load(b, sub, img)
    MA, MB = XA.mean(0), XB.mean(0); r = dict(kind=kind, a=a, b=b, sub=sub, rows='image' if img else 'all', L=MA.shape[1], LB=MB.shape[1])
    r.update(variants(MA, sa, MB, sb))
    rng = np.random.default_rng(22); A = prof(MA, len(sa)); perm = {k: [] for k in ('full', 'drop_2+2', 'layer_trim')}
    Al = prof(MA[:, 1:-1], len(sa))
    for _ in range(300):
        P = MB[:, rng.permutation(MB.shape[1])]; Bp = prof(P, len(sb))
        perm['full'].append(tan_from(A, sa, Bp, sb)); perm['drop_2+2'].append(tan_from(A, sa, Bp, sb, SEGS['drop_2+2']))
        perm['layer_trim'].append(tan_from(Al, sa, prof(P[:, 1:-1], len(sb)), sb))
    for k, v in perm.items(): r[f'perm95_{k}'] = float(np.percentile(v, 95))
    B = prof(MB, len(sb)); sur = np.array([tan_diff(A, sa, surrogate(B, rng), sb) for _ in range(300)])
    r['surr95'] = float(np.percentile(sur, 95)); r['surr_mean'] = float(sur.mean())
    r['FPR_lineage_rule'] = float((sur >= r['perm95_full'] + 0.30).mean()); r['FPR_ge_0.47'] = float((sur >= 0.47).mean())
    r['obs_pct_in_surr'] = float((sur < r['full']).mean())
    sur2 = np.array([tan_diff(A, sa, surrogate(B, rng), sb, SEGS['drop_2+2']) for _ in range(300)]); r['surr95_drop_2+2'] = float(np.percentile(sur2, 95))
    bs = []; bs22 = []
    for _ in range(200):
        ia = rng.choice(len(XA), min(30, len(XA)), replace=False); ib = rng.choice(len(XB), min(30, len(XB)), replace=False)
        Ai = prof(XA[ia].mean(0), len(sa)); Bi = prof(XB[ib].mean(0), len(sb))
        bs.append(tan_from(Ai, sa, Bi, sb)); bs22.append(tan_from(Ai, sa, Bi, sb, SEGS['drop_2+2']))
    r['ci05'], r['ci95'] = np.percentile(bs, [5, 95]); r['ci05_drop_2+2'], r['ci95_drop_2+2'] = np.percentile(bs22, [5, 95])
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True); return r
P = []
def add(kind, a, b, subs=('attn', 'mlp'), img=False):
    for s in subs: P.append((kind, a, b, s, img))
# same weights within one session/run
add('same: within run', 'Z1', 'Z1U'); add('same: within run', 'ACEB', 'ACEBU'); add('same: within run', 'ACEB0', 'ACEB1')
add('same: within session', 'T3', 'T8'); add('same: within session', 'T10', 'T8')
add('same: within run', 'SD3M', 'SD3MU', ('attn',)); add('same: within run', 'SD35M17', 'SD35M17U', ('attn',))
add('same: within run', 'PIXART', 'PIXARTU'); add('same: within run', 'COSMOS2B', 'COSMOS2BU')
# same weights across sessions
add('same: across sessions', 'Z3', 'ZIT'); add('same: across sessions', 'ZITSV', 'ZIT'); add('same: across sessions', 'ZITSV', 'Z3')
add('same: across sessions', 'ZIB', 'ZIT'); add('same: across sessions', 'T8', 'ACET')
add('same: across sessions', 'SD35M17', 'SD3SV', ('attn',)); add('same: across sessions', 'SD35M17', 'SD35MA', ('attn',)); add('same: across sessions', 'SD3SV', 'SD35MA', ('attn',))
# lineage / successor
add('lineage: distillation', 'Z1', 'ZIT'); add('lineage: distillation', 'Z1', 'Z3'); add('lineage: distillation', 'Z2', 'ZIT'); add('lineage: distillation', 'Z2', 'ZITSV')
add('lineage: distillation', 'ACEB', 'ACET'); add('lineage: distillation', 'ACEB', 'T8')
add('lineage: successor', 'SD3M', 'SD35M17', ('attn',)); add('lineage: successor', 'SD3M', 'SD3SV', ('attn',))
# unrelated, depth-matched
add('unrelated: depth-matched', 'PIXART', 'COSMOS2B')
for a in ('ACET', 'ACEB', 'T8'):
    for b in ('SD35M17', 'SD3M', 'SD3SV'): add('unrelated: depth-matched, cross-modal', a, b, ('attn',))
# image rows (row-split cubes)
add('same: within run (split capture)', 'SD35M17', 'SD35M17U', ('attn',), True); add('same: across sessions', 'SD35M17', 'SD3SV', ('attn',), True)
add('lineage: successor', 'SD3M', 'SD35M17', ('attn',), True); add('lineage: distillation', 'Z2', 'ZITSV', ('attn', 'mlp'), True)
add('unrelated: depth-matched', 'PIXART', 'COSMOS2B', ('attn', 'mlp'), True)
if __name__ == '__main__':
    with Pool(min(16, os.cpu_count())) as pool: R = pool.map(job, P)
    D = pd.DataFrame(R); D.to_csv(f'{OUT}/rc22_round5_tangent_checks.csv', index=False)
    pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40)
    cols = ['kind', 'a', 'b', 'sub', 'rows', 'full', 'ci05', 'ci95', 'drop_last1', 'drop_1+1', 'drop_2+2', 'layer_trim', 'no_s1', 'perm95_full', 'perm95_drop_2+2', 'perm95_layer_trim', 'surr95', 'surr_mean', 'FPR_lineage_rule', 'FPR_ge_0.47']
    print(D[cols].round(3).to_string(index=False))
