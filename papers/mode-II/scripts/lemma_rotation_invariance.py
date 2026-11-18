"""Numerical check of the rotation-invariance lemma (Mode-II, Lemma `lem:rot` and Corollary `cor:rot`).

Triple on the uncentred second moment C = X^T X / n:  g = (sum s)^2 / (D sum s^2), gap = s1/s2, H = entropy of s/sum s.
Checks (each row of ../data/II-lemma_rotation_check.csv):
  A  invariance of the triple under X -> c P X Q (orthogonal Q, scale c, token permutation P);
  B  semi-orthogonal widening D -> D': log g shifts by log(D/D') at every layer, gap and H unchanged,
     so the depth-standardised profile and the tangent are unchanged;
  C  a general invertible A and whitening change the triple (whitening gives g = 1 when n >= D);
  D  equal triples do not imply equal spectra (two spectra with the same triple, not orthogonally related);
  E  toy residual blocks: which reparametrisations are function-preserving symmetries of the network
     (RMSNorm: any Q after folding the gain; LayerNorm: Q with Q1 = 1; adaptive RMSNorm modulation:
     signed channel permutations only; general invertible A: none), and that the hooked outputs then
     transform as X -> X Q so the triple is unchanged.
Pure numpy; no telemetry needed. Seed fixed. Run: python3 lemma_rotation_invariance.py
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data', 'II-lemma_rotation_check.csv')
rng = np.random.default_rng(20261009)
TOL = 1e-9
rows = []


def triple_from_s(s, D):
    s = np.sort(np.asarray(s, float))[::-1]
    q = s / s.sum()
    q = q[q > 0]
    return np.array([s.sum() ** 2 / (D * (s ** 2).sum()), s[0] / s[1], -(q * np.log(q)).sum()])


def triple(X):
    """(g, gap, H) of X (n x D) on C = X^T X / n, as the analyzer defines them."""
    n, D = X.shape
    return triple_from_s(np.linalg.svd(X, compute_uv=False) / np.sqrt(n), D)  # = sqrt(eig C); zeros beyond rank add nothing


def haar(D):
    Q, R = np.linalg.qr(rng.normal(size=(D, D)))
    return Q * np.sign(np.diag(R))


def depthnorm(T):
    """T: (L, 3) of (g, gap, H) per layer -> z-scored (log g, log gap, H) across depth."""
    Z = np.column_stack([np.log(T[:, 0]), np.log(T[:, 1]), T[:, 2]])
    return (Z - Z.mean(0)) / Z.std(0)


def tangent_cos(Za, Zb):
    da, db = np.diff(Za, axis=0).ravel(), np.diff(Zb, axis=0).ravel()
    return da @ db / np.linalg.norm(da) / np.linalg.norm(db)


def add(part, case, quantity, value, expected, ok):
    rows.append(dict(part=part, case=case, quantity=quantity, value=float(value), expected=expected, pass_=bool(ok)))


def anisotropic(n, D, alpha=0.9, mean=2.0):
    """Token matrix with a power-law spectrum and a non-zero token mean (like hooked activations)."""
    X = rng.normal(size=(n, D)) * np.arange(1, D + 1) ** (-alpha / 2)
    return (X + mean * rng.normal(size=D) / np.sqrt(D)) @ haar(D)


# ---------------------------------------------------------------- A: the invariance group
n, D = 96, 48
X = anisotropic(n, D)
t0 = triple(X)
Q, P, c = haar(D), np.eye(n)[rng.permutation(n)], 3.7
U = haar(n)
for case, Y in [('right orthogonal X Q', X @ Q), ('scale c X', c * X), ('token permutation P X', P @ X),
                ('left orthogonal U X (U in O(n))', U @ X), ('c P X Q', c * P @ X @ Q)]:
    d = np.abs(triple(Y) - t0).max()
    add('A', case, 'max |triple change|', d, '< 1e-9', d < TOL)

# ---------------------------------------------------------------- B: widening and the depth profile
L, D, Dp = 12, 48, 80
stack = [anisotropic(n, D, alpha=0.3 + 0.12 * l, mean=0.5 + 0.3 * l) for l in range(L)]
V = np.linalg.qr(rng.normal(size=(Dp, D)))[0].T          # D x D' with orthonormal rows, V V^T = I_D
Tl = np.array([triple(Xl) for Xl in stack])
Tw = np.array([triple(Xl @ V) for Xl in stack])
shift = np.log(Tw[:, 0]) - np.log(Tl[:, 0])
add('B', f'widening D={D} -> D\'={Dp}', 'log g shift minus log(D/D\'), max over layers',
    np.abs(shift - np.log(D / Dp)).max(), '< 1e-9', np.abs(shift - np.log(D / Dp)).max() < TOL)
add('B', 'widening', 'max |gap, H change|', np.abs(Tw[:, 1:] - Tl[:, 1:]).max(), '< 1e-9', np.abs(Tw[:, 1:] - Tl[:, 1:]).max() < TOL)
dz = np.abs(depthnorm(Tw) - depthnorm(Tl)).max()
add('B', 'widening', 'max |depth-standardised profile change|', dz, '< 1e-9', dz < TOL)
tc = tangent_cos(depthnorm(Tl), depthnorm(Tw))
add('B', 'widening', 'tangent cosine original vs widened', tc, '1', abs(tc - 1) < TOL)
Qs = [haar(D) for _ in range(L)]                           # a different rotation at every layer
tq = tangent_cos(depthnorm(Tl), depthnorm(np.array([triple(Xl @ Ql) for Xl, Ql in zip(stack, Qs)])))
add('B', 'independent rotation per layer', 'tangent cosine', tq, '1', abs(tq - 1) < TOL)

# ---------------------------------------------------------------- C: what is not invariant
dA = []
for _ in range(200):
    A = rng.normal(size=(D, D)) + 0.5 * np.eye(D)
    dA.append(np.abs(np.log(triple(X @ A)[0]) - np.log(t0[0])))
add('C', 'general invertible A (200 draws)', 'median |log g change|', np.median(dA), '> 0', np.median(dA) > 1e-3)
TA = np.array([triple(Xl @ (rng.normal(size=(D, D)) + 0.5 * np.eye(D))) for Xl in stack])
tA = tangent_cos(depthnorm(Tl), depthnorm(TA))
add('C', 'independent invertible A per layer', 'tangent cosine', tA, '< 1', tA < 1 - 1e-3)
W = np.linalg.inv(np.linalg.cholesky(X.T @ X / n)).T     # whitening: C(XW) = I
add('C', 'whitening (n >= D)', 'g after whitening', triple(X @ W)[0], '1', abs(triple(X @ W)[0] - 1) < 1e-9)
Xn = anisotropic(30, D)                                   # n < D: whitening inside the row space flattens the
Un, _, Vn = np.linalg.svd(Xn, full_matrices=False)        # non-zero spectrum, so g reaches its ceiling n/D
gc = triple(np.sqrt(30) * Un @ Vn)[0]
add('C', 'whitening in the row space (n=30 < D)', 'g / (n/D)', gc / (30 / D), '1', abs(gc / (30 / D) - 1) < 1e-9)

# ---------------------------------------------------------------- D: the triple does not determine the spectrum
Dd = 6
sa = np.array([5.0, 3.0, 2.2, 1.4, 0.8, 0.3])
ta = triple_from_s(sa, Dd)


def loss(z):
    s = np.sort(np.exp(z))[::-1]
    s = s / s.sum() * sa.sum()
    return ((triple_from_s(s, Dd) - ta) ** 2).sum()


best = None
for _ in range(400):
    z0 = np.log(sa) + rng.normal(scale=0.6, size=Dd)
    r = minimize(loss, z0, method='Nelder-Mead', options=dict(xatol=1e-13, fatol=1e-26, maxiter=40000))
    s = np.sort(np.exp(r.x))[::-1]
    s = s / s.sum() * sa.sum()
    dist = np.abs(s - sa).max()
    if r.fun < 1e-20 and dist > 0.3 and (best is None or dist > best[1]):
        best = (s, dist, r.fun)
sb = best[0]
add('D', 'spectrum a = ' + ' '.join(f'{v:.3f}' for v in sa), 'triple (g; gap; H)', ta[0], ' / '.join(f'{v:.6f}' for v in ta), True)
add('D', 'spectrum b = ' + ' '.join(f'{v:.3f}' for v in sb), 'max |triple(b) - triple(a)|',
    np.abs(triple_from_s(sb, Dd) - ta).max(), '< 1e-9', np.abs(triple_from_s(sb, Dd) - ta).max() < TOL)
add('D', 'a vs b', 'max |s_a - s_b| (equal Frobenius norm imposed)', best[1], '> 0, so not orthogonally related', best[1] > 0.1)


# ---------------------------------------------------------------- E: network symmetries in toy residual blocks
def rms(h, gain=None):
    y = h / np.sqrt((h ** 2).mean(1, keepdims=True))
    return y if gain is None else y * gain


def ln(h, gain=None):
    y = h - h.mean(1, keepdims=True)
    y = y / np.sqrt((y ** 2).mean(1, keepdims=True))
    return y if gain is None else y * gain


def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def softmax(a):
    a = a - a.max(1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(1, keepdims=True)


def run(h, Ws, norm, mod=None):
    """Two pre-norm residual blocks (attention, MLP) and an unembedding. Returns logits and hooked outputs."""
    hooks = []
    for blk in Ws['blocks']:
        x = norm(h)
        if mod is not None:
            x = x * (1 + mod @ blk['Ws'])                    # adaptive (per-token, conditioning-dependent) scale
        q, k, v = x @ blk['Wq'], x @ blk['Wk'], x @ blk['Wv']
        a = softmax(q @ k.T / np.sqrt(q.shape[1])) @ v @ blk['Wo']
        hooks.append(a); h = h + a
        x = norm(h)
        if mod is not None:
            x = x * (1 + mod @ blk['Ws2'])
        m = gelu(x @ blk['Win']) @ blk['Wout']
        hooks.append(m); h = h + m
    return norm(h) @ Ws['WU'], hooks


def make_net(D, F=96, dh=24, V=50, nb=2, cond=8):
    return dict(blocks=[dict(Wq=rng.normal(size=(D, dh)) / np.sqrt(D), Wk=rng.normal(size=(D, dh)) / np.sqrt(D),
                             Wv=rng.normal(size=(D, D)) / np.sqrt(D), Wo=rng.normal(size=(D, D)) / np.sqrt(D),
                             Win=rng.normal(size=(D, F)) / np.sqrt(D), Wout=rng.normal(size=(F, D)) / np.sqrt(F),
                             Ws=0.3 * rng.normal(size=(cond, D)), Ws2=0.3 * rng.normal(size=(cond, D)))
                        for _ in range(nb)], WU=rng.normal(size=(D, V)) / np.sqrt(D))


def reparam(Ws, R):
    """h -> h R on the residual stream: read weights R^{-1} W, write weights W R, unembedding R^{-1} W_U."""
    Ri = np.linalg.inv(R)
    # The adaptive scale multiplies channels elementwise, so it can only follow a relabelling of channels:
    # |R| for a signed permutation. For any other R no transform of the modulation weights restores it.
    M = np.abs(R) if np.allclose(np.abs(R).sum(0), 1) and np.allclose(np.abs(R).sum(1), 1) else R
    out = dict(blocks=[], WU=Ri @ Ws['WU'])
    for b in Ws['blocks']:
        out['blocks'].append(dict(Wq=Ri @ b['Wq'], Wk=Ri @ b['Wk'], Wv=Ri @ b['Wv'], Wo=b['Wo'] @ R,
                                  Win=Ri @ b['Win'], Wout=b['Wout'] @ R, Ws=b['Ws'] @ M, Ws2=b['Ws2'] @ M))
    return out


Dn, nt = 32, 40
net = make_net(Dn)
h0 = rng.normal(size=(nt, Dn)) + 1.5 * rng.normal(size=Dn)
cvec = rng.normal(size=(1, 8))
Qr = haar(Dn)
B1 = np.linalg.qr(  # rotation fixing the ones vector: Q1 = 1
    np.column_stack([np.ones(Dn), rng.normal(size=(Dn, Dn - 1))]))[0]
Qone = B1 @ np.block([[np.ones((1, 1)), np.zeros((1, Dn - 1))], [np.zeros((Dn - 1, 1)), haar(Dn - 1)]]) @ B1.T
Sp = np.eye(Dn)[rng.permutation(Dn)] * rng.choice([-1, 1], size=Dn)   # signed permutation
Pm = np.eye(Dn)[rng.permutation(Dn)]                                   # plain permutation
A = rng.normal(size=(Dn, Dn)) + 0.5 * np.eye(Dn)


def check(label, norm, R, mod=None, sym=True):
    lg0, hk0 = run(h0, net, norm, mod)
    lg1, hk1 = run(h0 @ R, reparam(net, R), norm, None if mod is None else mod)
    dlog = np.abs(lg1 - lg0).max()
    dhook = max(np.abs(b - a @ R).max() for a, b in zip(hk0, hk1))
    dtri = max(np.abs(triple(b) - triple(a)).max() for a, b in zip(hk0, hk1))
    for q, v in [('max |logit change|', dlog), ('max |hooked output - X R|', dhook),
                 ('max |triple change| on hooked outputs', dtri)]:
        add('E', label, q, v, '< 1e-9 (symmetry)' if sym else '> 0 (not a symmetry)', (v < TOL) if sym else (v > 1e-3))


check('RMSNorm, rotation Q', rms, Qr)
check('RMSNorm, invertible A', rms, A, sym=False)
check('LayerNorm, rotation Q', ln, Qr, sym=False)
check('LayerNorm, rotation with Q1 = 1', ln, Qone)
check('LayerNorm, permutation', ln, Pm)
check('adaptive RMSNorm, rotation Q', rms, Qr, mod=cvec, sym=False)
check('adaptive RMSNorm, signed permutation', rms, Sp, mod=cvec)

# The learned RMSNorm gain must be folded into the read weights first (computational invariance):
gain = rng.uniform(0.5, 1.5, size=Dn)
lg0, _ = run(h0, net, lambda h: rms(h, gain))
folded = dict(blocks=[{**b, 'Wq': gain[:, None] * b['Wq'], 'Wk': gain[:, None] * b['Wk'], 'Wv': gain[:, None] * b['Wv'],
                       'Win': gain[:, None] * b['Win']} for b in net['blocks']], WU=gain[:, None] * net['WU'])
lg1, _ = run(h0 @ Qr, reparam(folded, Qr), rms)
add('E', 'RMSNorm with learned gain, folded then rotated', 'max |logit change|', np.abs(lg1 - lg0).max(),
    '< 1e-9 (symmetry)', np.abs(lg1 - lg0).max() < TOL)

df = pd.DataFrame(rows).rename(columns={'pass_': 'pass'})
df['source'] = 'papers/mode-II/scripts/lemma_rotation_invariance.py (synthetic; seed 20261009)'
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)
pd.set_option('display.width', 200, 'display.max_colwidth', 60)
print(df[['part', 'case', 'quantity', 'value', 'expected', 'pass']].to_string(index=False))
print(f"{int(df['pass'].sum())} of {len(df)} checks as expected")
raise SystemExit(0 if df['pass'].all() else 1)
