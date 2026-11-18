"""Numerical check of the pooling theorem (Mode-II, Theorem 2).
g(X) = (tr sqrt M)^2 / (D tr M), M = X^T X / N.  Claim: g(pool) >= sum_k w_k g_k, w_k = pi_k E_k / sum_j pi_j E_j.
Also counts how often the UNWEIGHTED bound g(pool) >= mean g_k fails when branch energies differ."""
import numpy as np

def g_and_energy(X):
    s = np.linalg.svd(X, compute_uv=False) / np.sqrt(len(X))
    return (s.sum() ** 2 / (s ** 2).sum()) / X.shape[1], (s ** 2).sum()

def main(n_weighted=5000, n_unweighted=2000):
    rng = np.random.default_rng(1)
    viol, slack = 0, []
    for _ in range(n_weighted):
        D = rng.integers(2, 20); na, nb = rng.integers(5, 40, 2)
        A = rng.normal(size=(na, D)) @ rng.normal(size=(D, D)) * rng.uniform(0.1, 10)
        B = rng.normal(size=(nb, D)) @ np.diag(rng.gamma(0.3, size=D)) * rng.uniform(0.1, 10)
        pa = na / (na + nb); pb = 1 - pa
        gp, _ = g_and_energy(np.vstack([A, B])); ga, Ea = g_and_energy(A); gb, Eb = g_and_energy(B)
        lb = (pa * Ea * ga + pb * Eb * gb) / (pa * Ea + pb * Eb)
        viol += gp < lb - 1e-12; slack.append(gp - lb)
    rng = np.random.default_rng(0); unw = 0
    for _ in range(n_unweighted):
        D, n = 8, 30
        A = rng.normal(size=(n, D)) @ np.diag(rng.gamma(0.5, size=D)) * rng.uniform(0.2, 5)
        B = rng.normal(size=(n, D)) @ np.diag(rng.gamma(0.5, size=D)) * rng.uniform(0.2, 5)
        gp, _ = g_and_energy(np.vstack([A, B])); ga, _ = g_and_energy(A); gb, _ = g_and_energy(B)
        unw += gp < (ga + gb) / 2
    print(f'energy-weighted bound: {viol} violations in {n_weighted}; min slack {min(slack):.2e}')
    print(f'unweighted bound fails in {unw} of {n_unweighted} unequal-energy draws')

if __name__ == '__main__':
    main()
