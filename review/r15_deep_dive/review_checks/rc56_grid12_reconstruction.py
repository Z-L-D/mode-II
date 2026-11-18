# RC56 (post hoc, round-9 review 2026-11-11: ChatGPT on "interpolation is a low-pass filter").
# Numerical reconstruction of the 12-point tangent from the native-grid depth-frequency coefficients.
# For each RC54 pair and state, the native-grid standardised profile (n points) is split by an orthonormal DCT-II of
# the profile itself (not of its differences) into a part the 12-point grid can resolve (k < 11 of n) and the rest.
# Each part is linearly interpolated onto the 12-point grid. Reported per pair:
#   t12        the actual 12-point tangent (interpolate the native profile to 12 points, re-standardise, difference)
#   t12_lp     the same after an ideal low-pass (keep k < 11 before interpolating) = what an ideal low-pass would give
#   band12     the RC54 grid12 band cosine (native coefficients k < 11 of the differences)
#   alias_share  share of the 12-point difference energy that comes from the unresolvable part (k >= 11)
# If interpolation were an ideal low-pass, t12 = t12_lp and alias_share = 0.
# Usage: RCDIR=<review_checks> python rc56_grid12_reconstruction.py <cache> <r16> <r17> <r18> <outdir>
import os, sys, numpy as np, pandas as pd
sys.argv, ARGS = sys.argv[:4], sys.argv
sys.path.insert(0, os.environ['RCDIR'])
import rc22_round5_checks as R
from scipy.fft import dct, idct
R18, OUT = ARGS[4], ARGS[5]; os.makedirs(OUT, exist_ok=True)
R.SRC.update({k: (R18, k) for k in ('SD35L18', 'SD35LT', 'ZB6', 'ZIT2', 'SD35L18U')})
B54 = pd.read_csv(os.path.join(os.environ['RCDIR'], 'out', 'rc54_tangent_bands.csv'))
def zs(x): return (x - x.mean(0)) / (x.std(0) + 1e-300)
def prof_native(M, ns, n):
    D = R.depthnorm(M); u = np.linspace(0, 1, D.shape[1]); G = np.linspace(0, 1, n)
    return np.stack([[np.interp(G, u, D[t, :, k]) for k in range(3)] for t in range(ns)]).transpose(0, 2, 1)  # T,n,3
def to12(P):  # T,n,3 -> T,12,3, re-standardised along depth
    n = P.shape[1]; u = np.linspace(0, 1, n); G = np.linspace(0, 1, 12)
    return np.stack([zs(np.stack([np.interp(G, u, P[t, :, k]) for k in range(3)], 1)) for t in range(P.shape[0])])
def split(P, kc=11):
    C = dct(P, type=2, norm='ortho', axis=1); L = C.copy(); L[:, kc:] = 0; H = C - L
    return idct(L, type=2, norm='ortho', axis=1), idct(H, type=2, norm='ortho', axis=1)
def tan(A, sa, B, sb):
    return float(np.mean([R.cos(np.diff(A[t], axis=0), np.diff(B[int(np.argmin(np.abs(sb - sa[t])))], axis=0)) for t in range(len(sa))]))
rows = []
for _, r in B54.iterrows():
    XA, sa = R.load(r.a, r['sub']); XB, sb = R.load(r.b, r['sub']); MA, MB = XA.mean(0), XB.mean(0)
    n = min(MA.shape[1], MB.shape[1]); A = prof_native(MA, len(sa), n); B = prof_native(MB, len(sb), n)
    AL, AH = split(A); BL, BH = split(B)
    u = np.linspace(0, 1, n); G = np.linspace(0, 1, 12)
    def i12(P): return np.stack([np.stack([np.interp(G, u, P[t, :, k]) for k in range(3)], 1) for t in range(P.shape[0])])
    eL = np.sum(np.diff(i12(AL), axis=1) ** 2); eH = np.sum(np.diff(i12(AH), axis=1) ** 2)
    eHn = np.sum(np.diff(AH, axis=1) ** 2); eAn = np.sum(np.diff(A, axis=1) ** 2)
    rows.append(dict(kind=r.kind, a=r.a, b=r.b, sub=r['sub'], n=n, native=r.native, t12=tan(to12(A), sa, to12(B), sb),
                     t12_lp=tan(to12(AL), sa, to12(BL), sb), band12=r.grid12, alias_share=eH / (eL + eH),
                     hi_share_native=eHn / eAn))
    print(rows[-1], flush=True)
D = pd.DataFrame(rows); D.to_csv(f'{OUT}/rc56_grid12_reconstruction.csv', index=False)
pd.set_option('display.width', 250); print(D.round(3).to_string(index=False))
rel = ~D.kind.str.startswith('unrelated')
print('corr t12 vs t12_lp', np.corrcoef(D.t12, D.t12_lp)[0, 1], 'corr t12 vs band12', np.corrcoef(D.t12, D.band12)[0, 1])
print('max |t12-t12_lp|', (D.t12 - D.t12_lp).abs().max(), 'alias share range', D.alias_share.min(), D.alias_share.max())
for lab, m in (('related', rel), ('unrelated', ~rel)):
    print(lab, 't12', D.t12[m].min(), D.t12[m].max(), 't12_lp', D.t12_lp[m].min(), D.t12_lp[m].max())
