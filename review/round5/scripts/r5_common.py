import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Shared helpers for Round 5. Estimators are copied unchanged from the Round 2-4 scripts
# (round2/scripts/beta24.py, common.py; round4/scripts_p_calcs.py; round3/audit/scripts/a1_reproduce.py).
import os, sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round2/scripts')
from common import depthnorm, shape_r
from beta24 import beta24, crossings, TRI, signs
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
SCR = os.environ.get('SCR', f'{_COSPAFS_CACHE}')
OUT = f'{_COSPAFS_ROOT}/review/round5/out'; os.makedirs(OUT, exist_ok=True)

def R5(arm, sub='attn'):
    """New captures. arm may join several with '+'. Returns dict(gids, meta, X{branch}, PI, sigma)."""
    parts = [pickle.load(open(f'{SCR}/R5_{a}_{sub}.pkl', 'rb')) for a in arm.split('+')]
    if len(parts) == 1: return parts[0]
    o = dict(gids=sum([p['gids'] for p in parts], []), meta=pd.concat([p['meta'] for p in parts]),
             X={b: np.concatenate([p['X'][b] for p in parts]) for b in parts[0]['X']},
             PI=np.concatenate([p['PI'] for p in parts]), sigma=parts[0]['sigma'])
    return o

def BM(d):
    """Synthetic branch-mean (what the R3 instrument wrote when the split failed): mean of the two forwards."""
    return (d['X']['cond'] + d['X']['uncond']) / 2

def R3(arm, sub='attn'):
    parts = [pickle.load(open(f'{SCR}/F3_{a}_{sub}.pkl', 'rb')) for a in arm.split('+')]
    return sum([p[0] for p in parts], []), pd.concat([p[1] for p in parts]), np.concatenate([p[2] for p in parts]), parts[0][3]

def ARCH(name):
    return pickle.load(open(f'{SCR}/F_{name}.pkl', 'rb'))

# ---- E10 tangent ----
gr = np.linspace(0, 1, 24)
def prof(X, s):
    D = depthnorm(X.mean(0)); u = np.linspace(0, 1, D.shape[1])
    return np.stack([[np.interp(gr, u, D[t, :, k]) for k in range(3)] for t in range(len(s))]).transpose(0, 2, 1)
def cos(a, b): a = a.ravel(); b = b.ravel(); return a @ b / np.linalg.norm(a) / np.linalg.norm(b)
def tangent(XA, sa, XB, sb):
    A = prof(XA, sa); B = prof(XB, sb); tc = []; srr = []
    for t, x in enumerate(sa):
        j = int(np.argmin(np.abs(sb - x))); tc.append(cos(np.diff(A[t], axis=0), np.diff(B[j], axis=0))); srr.append(shape_r(A[t], B[j]))
    return float(np.mean(tc)), float(np.mean(srr))

# ---- E5 residuals, seed fate, VR ----
def resid(X, ph):
    grp = pd.factorize(np.asarray(ph).astype(str))[0]
    Y = np.log(np.abs(X) + 1e-12); Y[..., 2] = X[..., 2]; G, T, L, _ = Y.shape
    Fz = Y.reshape(G, T, -1); Fz = (Fz - Fz.mean(0)) / (Fz.std(0) + 1e-12); r = np.empty_like(Fz)
    for t in range(T): r[:, t] = Fz[:, t] - pd.DataFrame(Fz[:, t]).groupby(grp).transform('mean').to_numpy()
    return r, grp, Fz
def ridge(A, B, alpha=100.0):
    mu = A.mean(0); mb = B.mean(0)
    W = np.linalg.solve((A - mu).T @ (A - mu) + alpha * np.eye(A.shape[1]), (A - mu).T @ (B - mb)); return lambda Z: (Z - mu) @ W + mb
def curve(r, grp, seed, center, target=None):
    """R^2(state at step t -> final state) with prompt-grouped 5-fold CV; optional in-fold seed centring (Round 4 P3).
    target: optional separate residual array whose final step is the target (cross-branch prediction)."""
    tgtsrc = r if target is None else target
    T = r.shape[1]; preds = np.zeros((T,) + tgtsrc[:, -1].shape); tg = np.zeros_like(tgtsrc[:, -1])
    for tr, te in GroupKFold(5).split(r[:, 0], groups=grp):
        rc = r.copy(); tc = tgtsrc.copy()
        if center:
            for sd in np.unique(seed):
                msk = tr[seed[tr] == sd]
                rc[seed == sd] = r[seed == sd] - r[msk].mean(0)
                tc[seed == sd] = tgtsrc[seed == sd] - tgtsrc[msk].mean(0)
        tg[te] = tc[te, -1]
        for t in range(T): preds[t, te] = ridge(rc[tr, t], tc[tr, -1])(rc[te, t])
    return np.array([1 - ((tg - preds[t]) ** 2).sum() / ((tg - tg.mean(0)) ** 2).sum() for t in range(T)])
def s50(c, s):
    T = len(c); tau = np.arange(T) / (T - 1)
    if not (c > 0.5).any(): return np.nan
    i = int(np.argmax(c > 0.5)); t50 = tau[0] if i == 0 else tau[i - 1] + (0.5 - c[i - 1]) * (tau[i] - tau[i - 1]) / (c[i] - c[i - 1])
    return float(np.interp(t50, tau, s))
def VR(r, k_list=(2, 4, 8)):
    rr = r[:, 1:]; v1 = ((rr[:, 1:] - rr[:, :-1]) ** 2).mean()
    return {k: float(((rr[:, k:] - rr[:, :-k]) ** 2).mean() / (k * v1)) for k in k_list}
def seed_center(r, seed):
    rc = r.copy()
    for sd in np.unique(seed): rc[seed == sd] -= r[seed == sd].mean(0)
    return rc
def seed_identity(Fz, seed, grp):
    y = (seed == np.unique(seed)[0]).astype(int); acc = []
    for tr, te in GroupKFold(5).split(Fz[:, 0], groups=grp):
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Fz[tr, 0], y[tr]); acc.append((clf.predict(Fz[te, 0]) == y[te]).mean())
    return float(np.mean(acc))
def seedshare(X, ph):
    Y = X.copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1])
    up, inv = np.unique(np.asarray(ph).astype(str), return_inverse=True)
    G, T, L, _ = Y.shape; Yf = Y.reshape(G, T, -1); Yf = (Yf - Yf.mean(0)) / (Yf.std(0) + 1e-12)
    cnt = np.bincount(inv); k = len(up); n0 = (G - (cnt ** 2).sum() / G) / (k - 1); out = []
    for t in range(T):
        Z = Yf[:, t]; mu = np.stack([Z[inv == i].mean(0) for i in range(k)])
        ssw = ((Z - mu[inv]) ** 2).sum(0); ssb = (cnt[:, None] * (mu - Z.mean(0)) ** 2).sum(0)
        msw = ssw / (G - k); msb = ssb / (k - 1); vp = np.maximum((msb - msw) / n0, 0); out.append(1 - vp.sum() / (vp.sum() + msw.sum()))
    return np.array(out)

# ---- E3 onset ----
def onset(X, s):
    Y = X.mean(0).copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1]); A = Y.reshape(len(s), -1)
    sp = np.linalg.norm(np.diff(A, axis=0), axis=1) / np.abs(np.diff(s)); cm = (s[1:] + s[:-1]) / 2
    plat = np.median(sp[cm < 0.6]); i = int(np.argmax(sp < 2 * plat)); return float(cm[i])

def topo(G):
    c, bm = beta24(G.reshape(-1, G.shape[-1])); cr = crossings(c, bm); t = TRI(c, bm)
    return dict(signs=signs(bm), n_cross=len(cr), crossings=';'.join(f'{x:.5f}{d}' for x, d in cr), TRI=None if t != t else float(t))
