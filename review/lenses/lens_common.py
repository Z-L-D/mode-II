"""Shared helpers for the lens tests (LENS_PREREGISTRATION.md).

The triple, from capture_code_r2/cos_pafs_llm_measure.py (l. 671-681): sigma_i = sqrt(eig(X^T X / N)), p_i = sigma_i / sum(sigma)
  H2 = ln(g * D)          (Renyi-2 entropy of p; g*D = 1/sum p^2)
  H1 = sv_entropy         (Shannon entropy of p)
  Lam = ln(spectral_gap)  (= ln(p1/p2))
Spectral families are parametrised by singular values sigma_i over i = 1..R (R = rank).
For a spectrum with spike s on top of a bulk w_i (i >= 2), with S1 = sum w, S2 = sum w^2, T = sum w ln w:
  Z = s + S1;  H2 = ln(Z^2 / (s^2 + S2));  H1 = ln Z - (s ln s + T) / Z;  Lam = ln(s / w_2)
"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import os, numpy as np
from scipy.special import xlogy
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
CACHE = f'{_COSPAFS_ROOT}/review/standing_wave/sw_cache'
CH = [('ZIT attn', 'ZIT_attn', 3840, 'Z-Image'), ('ZIT MLP', 'ZIT_mlp', 3840, 'Z-Image'), ('ZIB attn', 'ZIB_attn', 3840, 'Z-Image'),
      ('ZIB MLP', 'ZIB_mlp', 3840, 'Z-Image'), ('Mochi attn', 'Mochi_attn', 3072, 'Mochi'),
      ('SD3.5 M attn', 'SD35M_attn', 1536, 'SD3.5'), ('SD3.5 L attn', 'SD35L_attn', 2432, 'SD3.5')]


def load(key):
    d = np.load(f'{CACHE}/{key}.npz', allow_pickle=True)
    return {k: d[k] for k in d.files}


def triple(d, D):
    """-> H1, H2, Lam arrays [n, t, L]."""
    return d['ent_gen'].astype(float), np.log(d['g_gen'].astype(float) * D), np.log(d['gap_gen'].astype(float))


def bulk_sums(w):
    w = np.asarray(w, float); return w.sum(), (w ** 2).sum(), (w * np.log(w)).sum()


def spike_bulk(s, S1, S2, T):
    """vectorised over s (and broadcast S1, S2, T)."""
    Z = s + S1
    return np.log(Z) - (s * np.log(s) + T) / Z, np.log(Z ** 2 / (s ** 2 + S2))   # H1, H2


class Family:
    """One-parameter families on R singular values; grid over the parameter, H2 monotone along the grid."""
    def __init__(self, R, kind):
        i = np.arange(1, R + 1, dtype=float); self.kind = kind
        if kind == 'PL':
            self.par = np.linspace(0, 4, 1601); W = [i ** -a for a in self.par]
        elif kind == 'EXP':
            self.par = np.concatenate([np.linspace(0, 10, 501), np.geomspace(10.02, 3000, 800)]); W = [np.exp(-b * (i - 1) / R) for b in self.par]
        elif kind == 'SB':
            self.par = np.geomspace(1, 1e6, 1601); W = None
        if W is not None:
            st = np.array([[w.sum(), (w ** 2).sum(), xlogy(w, w).sum(), np.log(w[0] / max(w[1], 1e-300))] for w in W])
            Z = st[:, 0]; self.H1 = np.log(Z) - st[:, 2] / Z; self.H2 = np.log(Z ** 2 / st[:, 1]); self.Lam = st[:, 3]
        else:
            S1 = S2 = R - 1.0; self.H1, self.H2 = spike_bulk(self.par, S1, S2, 0.0); self.Lam = np.log(self.par)
        o = np.argsort(self.H2); self.H2s = self.H2[o]; self.H1s = self.H1[o]; self.Lams = self.Lam[o]; self.pars = self.par[o]

    def predict(self, H2):
        """-> predicted H1, Lam, parameter, in_range flag (H2 inside the family's range)."""
        H2 = np.asarray(H2, float); inr = (H2 >= self.H2s[0]) & (H2 <= self.H2s[-1])
        return np.interp(H2, self.H2s, self.H1s), np.interp(H2, self.H2s, self.Lams), np.interp(H2, self.H2s, self.pars), inr


class SPL:
    """Spike + power-law bulk: sigma_1 = s (s >= 1 relative to the bulk's first value 1), sigma_i = i^-a for i >= 2.
    Fixed by (H2, Lam): Lam = ln s + a ln 2. Predicts H1."""
    def __init__(self, R):
        i = np.arange(2, R + 1, dtype=float); self.a = np.linspace(0, 4, 801)
        st = np.array([bulk_sums(i ** -a) for a in self.a]); self.S1, self.S2, self.T = st.T

    def fit(self, H2, Lam, chunk=20000):
        H2 = np.asarray(H2, float).ravel(); Lam = np.asarray(Lam, float).ravel()
        H1p = np.full(H2.shape, np.nan); ap = np.full(H2.shape, np.nan); sp = np.full(H2.shape, np.nan)
        for k in range(0, len(H2), chunk):
            h, l = H2[k:k + chunk, None], Lam[k:k + chunk, None]
            s = np.exp(l - self.a[None] * np.log(2)); ok = s >= 1                       # spike at least the PL continuation
            H1g, H2g = spike_bulk(np.where(ok, s, 1.0), self.S1[None], self.S2[None], self.T[None])
            d = np.where(ok, H2g - h, np.nan)
            # first sign change along a (H2 is not strictly monotone in a at fixed Lam; take the root nearest a=0)
            sg = np.sign(d); ch = (sg[:, :-1] * sg[:, 1:] <= 0) & np.isfinite(d[:, :-1]) & np.isfinite(d[:, 1:])
            has = ch.any(1); j = np.argmax(ch, 1)
            r = np.arange(len(h))
            d0, d1 = d[r, j], d[r, j + 1]; w = np.where(d1 != d0, d0 / (d0 - d1), 0.0)
            H1p[k:k + chunk] = np.where(has, H1g[r, j] + w * (H1g[r, j + 1] - H1g[r, j]), np.nan)
            ap[k:k + chunk] = np.where(has, self.a[j] + w * (self.a[j + 1] - self.a[j]), np.nan)
            sp[k:k + chunk] = np.where(has, s[r, j] * (s[r, j + 1] / s[r, j]) ** w, np.nan)
        return H1p, ap, sp

    def spectrum(self, a, s, R):
        i = np.arange(1, R + 1, dtype=float); w = i ** -a; w[0] = s; return w


def entropies_from_sigma(sig):
    """sig [..., R] singular values -> H1, H2, Lam."""
    p = sig / sig.sum(-1, keepdims=True)
    H1 = -(p * np.log(p)).sum(-1); H2 = -np.log((p ** 2).sum(-1)); ps = -np.sort(-p, -1)
    return H1, H2, np.log(ps[..., 0] / ps[..., 1])


def prompt_boot_idx(prompts, rng):
    up = np.unique(prompts); idx = {p: np.where(prompts == p)[0] for p in up}
    return np.concatenate([idx[p] for p in rng.choice(up, len(up))])
