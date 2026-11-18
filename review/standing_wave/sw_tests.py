"""Standing-wave re-test, frozen in SW_PREREGISTRATION.md (SHA-256 e9c74d7f..., 2026-10-08T10:45:59Z).

Reads sw_cache/<channel>.npz written by sw_load.py (analyzer DataManager, unchanged) and runs P1-P6.
Outputs: out/SW_P1_reproduction.csv, SW_P2_wave.csv, SW_P3_nulls.csv, SW_P4_coords.csv, SW_P5_bootstrap.csv, SW_psi_profiles.csv
"""
import os, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, 'sw_cache'); OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)
CENTRAL = [9, 12, 15, 18, 21]
PINCH = 0.75
rng = np.random.default_rng(20261008)


def field(G, SEC, layers, scale='z'):
    """Analyzer 3D trajectory (g, secondary, L) per step -> fixed-PCA transverse complex field z[t, L]."""
    T, L = G.shape
    P = np.column_stack([G.ravel(), SEC.ravel(), np.tile(layers.astype(float), T)])
    if scale == 'z':
        P = (P - P.mean(0)) / P.std(0)
    else:
        P = P - P.mean(0)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    S = P @ Vt.T
    return (S[:, 1] + 1j * S[:, 2]).reshape(T, L)


def modes(Z, cols=None):
    M = Z.T if cols is None else Z[cols].T                     # [L, t]
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    psi = U[:, 0]; A = s[0] * Vh[0].conj()
    return psi, A, s[0] ** 2 / np.sum(s ** 2)


def interior_minima(a):
    return np.array([i for i in range(1, len(a) - 1) if a[i] < a[i - 1] and a[i] < a[i + 1]])


def run3(mins):
    if len(mins) < 2: return 0
    sp = np.diff(mins); best = cur = 0
    for d in sp:
        cur = cur + 1 if d == 3 else 0; best = max(best, cur)
    return best


def cv(mins):
    if len(mins) < 3: return np.nan
    sp = np.diff(mins); return sp.std() / sp.mean()


def reversals(psi):
    k = np.diff(np.unwrap(np.angle(psi))); s = np.sign(k[k != 0]); return int(np.sum(s[1:] != s[:-1]))


def summarise(G, SEC, layers, sigma, scale):
    Z = field(G, SEC, layers, scale)
    psi, A, e_full = modes(Z)
    post = np.where(sigma <= PINCH)[0]
    _, _, e_post = modes(Z, post) if len(post) >= 2 else (None, None, np.nan)
    a = np.abs(psi); mins = interior_minima(a)
    return dict(Z=Z, psi=psi, A=A, e_full=e_full, e_post=e_post, mins=mins, gmin=int(layers[np.argmin(a)]),
                rev=reversals(psi), r3=run3(mins), cv=cv(mins))


def load(name):
    d = np.load(os.path.join(CACHE, f'{name}.npz'), allow_pickle=True)
    return {k: d[k] for k in d.files}


def phase_rand_complex(psi):
    F = np.fft.fft(psi); return np.fft.ifft(np.abs(F) * np.exp(2j * np.pi * rng.random(len(F))))


def phase_rand_profiles(X):
    """Same random Fourier phases at every step, after removing each step's linear trend in L."""
    T, L = X.shape; x = np.arange(L)
    coef = np.polyfit(x, X.T, 1); trend = (np.outer(coef[0], x) + coef[1][:, None])
    R = X - trend; F = np.fft.rfft(R, axis=1)
    ph = np.exp(2j * np.pi * rng.random(F.shape[1])); ph[0] = 1
    if L % 2 == 0: ph[-1] = np.sign(np.cos(2 * np.pi * rng.random()))
    return np.fft.irfft(F * ph[None, :], n=L, axis=1) + trend


CHANNELS = ['ZIT_mlp', 'ZIB_mlp', 'ZIT_attn', 'ZIB_attn', 'Mochi_attn', 'SD35M_attn', 'SD35L_attn']
p1, p2, p3, p4, p5, prof = [], [], [], [], [], []
for ch in CHANNELS:
    d = load(ch); layers, sigma = d['layers'], d['sigma']
    var = {'G-z': (d['G'], d['GAP'], 'z'), 'G-raw': (d['G'], d['GAP'], 'raw'), 'G-ent': (d['G'], d['ENT'], 'z')}
    res = {k: summarise(G, S, layers, sigma, sc) for k, (G, S, sc) in var.items()}
    for k, r in res.items():
        p1.append(dict(channel=ch, geometry=k, mode1_energy_full=r['e_full'], mode1_energy_post=r['e_post'],
                       psi_minima=' '.join(map(str, layers[r['mins']])) if len(r['mins']) else '', global_min=r['gmin'],
                       phase_gradient_reversals=r['rev'], longest_spacing3_run=r['r3'], spacing_cv=r['cv']))
        for L_, v in zip(layers, r['psi']):
            prof.append(dict(channel=ch, geometry=k, layer=int(L_), abs_psi=abs(v), arg_psi=np.angle(v)))
        if ch.endswith('mlp') or True:
            p4.append(dict(channel=ch, geometry=k, central_nodes_recovered=sum(any(abs(m - c) <= 1 for m in layers[r['mins']]) for c in CENTRAL)))
    # P2 on primary geometry
    r = res['G-z']; A = r['A']
    ph = np.unwrap(np.angle(A * np.conj(A[0]) / abs(A[0])))
    W = np.array([A.real, A.imag]); ev, evec = np.linalg.eigh(W @ W.T); dvec = evec[:, -1]
    proj = dvec @ W; sc = int(np.sum(np.sign(proj[1:]) != np.sign(proj[:-1])))
    post = np.where(sigma <= PINCH)[0]; Z = r['Z']
    cn = [m for m in layers[r['mins']] if 8 <= m <= 22] or CENTRAL
    disp = []
    for t in post:
        mt = layers[interior_minima(np.abs(Z[t]))]
        for c in cn:
            disp.append(np.min(np.abs(mt - c)) if len(mt) else np.nan)
    p2.append(dict(channel=ch, A_phase_excursion_rad=ph.max() - ph.min(), A_realproj_sign_changes=sc,
                   oscillation_W=bool((ph.max() - ph.min()) >= np.pi or sc >= 1),
                   A_abs_min_over_max=np.abs(A).min() / np.abs(A).max(),
                   node_median_disp=np.nanmedian(disp) if disp else np.nan, node_stable=bool(np.nanmedian(disp) <= 1) if disp else False,
                   n_post_steps=len(post), central_nodes_used=' '.join(map(str, cn))))
    # P3 nulls on primary geometry
    n1 = [(run3(m), cv(m)) for m in (interior_minima(np.abs(phase_rand_complex(r['psi']))) for _ in range(10000))]
    n2 = []
    for _ in range(1000):
        Gs, Ss = phase_rand_profiles(d['G']), phase_rand_profiles(d['GAP'])
        z = summarise(Gs, Ss, layers, sigma, 'z'); n2.append((z['r3'], z['cv']))
    n1, n2 = np.array(n1, float), np.array(n2, float)
    def pv(n, i, obs, ge=True):
        x = n[:, i]; x = x[~np.isnan(x)]
        return np.nan if np.isnan(obs) else ((np.sum(x >= obs) if ge else np.sum(x <= obs)) + 1) / (len(x) + 1)
    p3.append(dict(channel=ch, r3_obs=r['r3'], p_r3_null1=pv(n1, 0, r['r3']), p_r3_null2=pv(n2, 0, r['r3']),
                   cv_obs=r['cv'], p_cv_null1=pv(n1, 1, r['cv'], False), p_cv_null2=pv(n2, 1, r['cv'], False),
                   null1_mean_r3=n1[:, 0].mean(), null2_mean_r3=n2[:, 0].mean()))
    # P5 prompt bootstrap (G-z and G-ent)
    up = np.unique(d['prompts']); idx = {p: np.where(d['prompts'] == p)[0] for p in up}
    for geo, sec in (('G-z', 'gap_gen'), ('G-ent', 'ent_gen')):
        locs = []
        for _ in range(500):
            ii = np.concatenate([idx[p] for p in rng.choice(up, len(up))])
            G = d['g_gen'][ii].mean(0); S = d[sec][ii].mean(0)
            psi, _, _ = modes(field(G, S, layers, 'z')); locs.append(int(layers[np.argmin(np.abs(psi))]))
        vc = pd.Series(locs).value_counts()
        p5.append(dict(channel=ch, geometry=geo, modal_global_min=int(vc.index[0]), modal_share=vc.iloc[0] / 500,
                       L15_share=(np.array(locs) == 15).mean(), distribution=' '.join(f'{k}:{v}' for k, v in vc.items())))
    print(ch, 'done', flush=True)

for nm, rows in (('SW_P1_reproduction', p1), ('SW_P2_wave', p2), ('SW_P3_nulls', p3), ('SW_P4_coords', p4),
                 ('SW_P5_bootstrap', p5), ('SW_psi_profiles', prof)):
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f'{nm}.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
for nm, rows in (('P1', p1), ('P2', p2), ('P3', p3), ('P4', p4), ('P5', p5)):
    print('\n==', nm); print(pd.DataFrame(rows).round(4).to_string(index=False))
