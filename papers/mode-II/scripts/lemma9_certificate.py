"""Lemma 9(iv) numerical certificate (post hoc, round-5 review Kimi 6, GLM minor 4).
Polishes the printed s_b to the exact solution of triple(s) = triple(s_a) at sum(s) = sum(s_a) (Gauss-Newton on 3 equations in the
5-dimensional slice), prints s_b at full precision, the Jacobian's singular values at s_b (rank 3 means, by the implicit function
theorem, that the solutions near s_b form a smooth 2-dimensional family, so the counterexample is not isolated), and walks along that
family to show other exact solutions at distances 0.01-0.1."""
import numpy as np
def T(s, D=6):
    s = np.sort(np.asarray(s, float))[::-1]; q = s / s.sum(); return np.array([s.sum() ** 2 / (D * (s ** 2).sum()), s[0] / s[1], -(q * np.log(q)).sum()])
sa = np.array([5.0, 3.0, 2.2, 1.4, 0.8, 0.3]); ta = T(sa); S = sa.sum()
B = np.linalg.qr(np.c_[np.ones(6), np.eye(6)[:, :5]])[0][:, 1:]   # orthonormal basis of the sum-zero slice
def J(s, h=1e-7): return np.stack([(T(s + h * B[:, k]) - T(s - h * B[:, k])) / (2 * h) for k in range(5)], 1)
def polish(s):
    for _ in range(60):
        r = T(s) - ta; Jm = J(s); s = s - B @ np.linalg.lstsq(Jm, r, rcond=None)[0]
    return s
s0 = np.array([5.078, 3.047, 1.489, 1.461, 1.443, 0.183]); sb = polish(s0 / s0.sum() * S)
np.set_printoptions(precision=17)
print('s_b =', repr(sb)); print('sum', sb.sum(), 'vs', S); print('max |T(s_b) - T(s_a)| =', np.abs(T(sb) - ta).max())
sv = np.linalg.svd(J(sb), compute_uv=False); print('Jacobian singular values at s_b:', sv, '-> rank', int((sv > 1e-8 * sv[0]).sum()))
N = np.linalg.svd(J(sb))[2][3:].T                                  # null-space directions (in slice coordinates)
for d in (0.01, 0.03, 0.1):
    for k in range(N.shape[1]):
        s2 = polish(sb + d * (B @ N[:, k])); print(f'step {d} dir {k}: |s2-s_b|_inf {np.abs(s2 - sb).max():.4f}, residual {np.abs(T(s2) - ta).max():.1e}, |s2-s_a|_inf {np.abs(s2 - sa).max():.3f}, min s {s2.min():.3f}')
