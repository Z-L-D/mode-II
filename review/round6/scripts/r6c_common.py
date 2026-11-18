import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Round 6 (Claude) shared helpers. Reuses the Round 2-5 estimators unchanged (round5/scripts/r5_common.py).
import sys, os, json
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round5/scripts')
from r5_common import *          # ARCH, R3, R5, BM, prof, tangent, cos, resid, ridge, curve, s50, VR, depthnorm, shape_r ...
OUT6 = f'{_COSPAFS_ROOT}/review/round6/out'; FIG6 = f'{_COSPAFS_ROOT}/review/round6/fig'
os.makedirs(OUT6, exist_ok=True); os.makedirs(FIG6, exist_ok=True)
MODELS = ['ZIT', 'ZIB', 'MOCHI', 'SD35M', 'SD35L']
FAM = {'ZIT': 'Z', 'ZIB': 'Z', 'MOCHI': 'M', 'SD35M': 'S', 'SD35L': 'S'}
# Round 2 E10 pair order (a, b) so L2/L3 reproduce out/E10_tangent.csv exactly
E10_UNREL = [('ZIT', 'MOCHI'), ('ZIT', 'SD35M'), ('ZIT', 'SD35L'), ('ZIB', 'MOCHI'), ('ZIB', 'SD35M'), ('ZIB', 'SD35L'), ('MOCHI', 'SD35M'), ('MOCHI', 'SD35L')]
def kind(a, b):
    if {a, b} == {'ZIT', 'ZIB'}: return 'lineage'
    if FAM[a] == FAM[b]: return 'same family'
    return 'unrelated'
def arch(m, sub='attn'):
    g, meta, X, s = ARCH(f'{m}_{sub}'); return g, meta.copy(), X, np.asarray(s, float)
def interp_desc(s_from, y_from, s_to):
    """Interpolate y(s) given on a descending sigma grid onto s_to (linear in sigma)."""
    o = np.argsort(s_from); return np.interp(s_to, s_from[o], y_from[o])
def overlap(sa, sb):
    lo, hi = max(sa.min(), sb.min()), min(sa.max(), sb.max()); return lo, hi
def env_r(Ea, sa, Eb, sb):
    """Pearson r of two scalar envelopes; the denser one is interpolated onto the sparser grid inside the overlap."""
    if len(sa) > len(sb): Ea, sa, Eb, sb = Eb, sb, Ea, sa
    lo, hi = overlap(sa, sb); m = (sa >= lo - 1e-9) & (sa <= hi + 1e-9)
    return float(np.corrcoef(Ea[m], interp_desc(sb, Eb, sa[m]))[0, 1])
gr24 = np.linspace(0, 1, 24)
def depth24(M):
    """M: [T, L] -> [T, 24] by linear interpolation over normalized depth."""
    u = np.linspace(0, 1, M.shape[1]); return np.stack([np.interp(gr24, u, row) for row in M])
def field_r(XA, sa, XB, sb):
    """L1: mean log g over gens, depth->24 bins, matched sigma (nearest, iterating the sparser model), Pearson over cells."""
    A = depth24(np.log(XA[..., 0]).mean(0)); B = depth24(np.log(XB[..., 0]).mean(0))
    if len(sa) > len(sb): A, sa, B, sb = B, sb, A, sa
    lo, hi = overlap(sa, sb); rows_a = []; rows_b = []
    for t, x in enumerate(sa):
        if x < lo - 1e-9 or x > hi + 1e-9: continue
        j = int(np.argmin(np.abs(sb - x))); rows_a.append(A[t]); rows_b.append(B[j])
    return float(np.corrcoef(np.ravel(rows_a), np.ravel(rows_b))[0, 1])
def jdump(obj, name):
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    json.dump(obj, open(f'{OUT6}/{name}', 'w'), indent=1, default=conv)
