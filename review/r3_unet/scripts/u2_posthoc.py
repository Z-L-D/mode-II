"""U2 POST HOC (not preregistered; R3C_SCORING_PLAN.md P1-P4). SD1.5 cond rows; archive methods unchanged.
P1 backbone (calibrated on archive SD3.5 M / ZIB first), P2 plateau onset (e3 H3b code on t = sigma/(1+sigma)),
P3 prompt share by step (e6 ANOVA share), P4 triple PCA (lens_L1 pca_shares). Output: out/P*_*.csv"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round2/scripts'); sys.path.insert(0, f'{_COSPAFS_ROOT}/review/lenses')
# pca_shares copied verbatim from review/lenses/lens_L1_triple.py (importing that module runs its whole analysis)
def pca_shares(H1, H2, L):
    M = np.column_stack([H1, H2, L]); M = (M - M.mean(0)) / M.std(0)
    ev = np.sort(np.linalg.eigvalsh(np.cov(M.T)))[::-1]; return ev / ev.sum()
c = np.load(f'{BASE}/out/u0_cache.npz', allow_pickle=True); NL = c['NL']; pt = c['pt']
sig_ve = c['sigma'].astype(float); t_rf = sig_ve / (1 + sig_ve)

def sd15_X(sub):   # G, T, L, 3 (g, gap, H), cond
    return np.stack([c[f'{sub}_cond_{v}'].astype(float) for v in ('g_micro', 'spectral_gap', 'sv_entropy')], -1)

# ---- P1 backbone ----
def backbone(X, nnull=999, seed=0):
    Y = np.nanmean(X, 0).copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1])     # T, L, 3
    Z = (Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-12)
    def stats(Z):
        A = Z.reshape(Z.shape[0], -1); A = A - A.mean(0); ev = np.clip(np.linalg.svd(A, compute_uv=False) ** 2, 0, None); p = ev / ev.sum(); p = p[p > 0]
        return p[0], float(np.exp(-(p * np.log(p)).sum()))
    pc1, er = stats(Z); r = np.random.default_rng(seed); T, L, _ = Z.shape; npc, nrk = [], []
    for _ in range(nnull):
        Zs = np.stack([np.roll(Z[t], r.integers(L), axis=0) for t in range(T)]); a, b = stats(Zs); npc.append(a); nrk.append(b)
    npc, nrk = np.array(npc), np.array(nrk)
    return dict(pc1=pc1, eff_rank=er, null_pc1_mean=npc.mean(), null_rank_mean=nrk.mean(), p_rank=(1 + (nrk <= er).sum()) / (nnull + 1), p_pc1=(1 + (npc >= pc1).sum()) / (nnull + 1))

if __name__ == '__main__':
    pd.set_option('display.width', 250)
    from common import fields
    rows = []
    for ch, sub, ref in (('SD35M', 'attn', (0.897, 1.510)), ('ZIB', 'attn', (0.710, 2.531))):
        _, _, X, _ = fields(ch, sub); b = backbone(X); b.update(channel=f'archive {ch} {sub} (calibration)', ref_pc1=ref[0], ref_rank=ref[1]); rows.append(b)
    for sub in ('attn', 'mlp'):
        b = backbone(sd15_X(sub)); b.update(channel=f'SD1.5 {sub} (cond)'); rows.append(b)
    P1 = pd.DataFrame(rows); P1.to_csv(f'{BASE}/out/P1_backbone.csv', index=False); print(P1.round(3).to_string(index=False), flush=True)

    # ---- P2 plateau onset (e3 H3b, unchanged except the clock) ----
    lam = lambda s: np.log((1 - s) ** 2 / s ** 2); l0, l1 = lam(0.99), lam(0.01); rows = []
    for sub in ('attn', 'mlp'):
        for clock, s in (('t=sigma/(1+sigma)', t_rf),):
            X = sd15_X(sub); Y = np.nanmean(X, 0).copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1]); A = Y.reshape(len(s), -1)
            sp = np.linalg.norm(np.diff(A, axis=0), axis=1) / np.abs(np.diff(s)); cm = (s[1:] + s[:-1]) / 2
            plat = np.median(sp[cm < 0.6]); i = int(np.argmax(sp < 2 * plat)); so = cm[i]
            rows.append(dict(ch=f'SD1.5-{sub}', clock=clock, onset_t=so, onset_sigma_VE=so / (1 - so), onset_stepfrac=(i + 0.5) / (len(s) - 1), plateau_speed=plat,
                             n_steps_below_0p6=int((cm < 0.6).sum()), t_first=s[0], t_last=s[-1]))
    P2 = pd.DataFrame(rows); P2.to_csv(f'{BASE}/out/P2_plateau_onset.csv', index=False); print(P2.round(3).to_string(index=False), flush=True)

    # ---- P3 prompt share by step (e6, ANOVA) ----
    rows = []; summ = []
    up, inv = np.unique(pt, return_inverse=True); k = len(up); cnt = np.bincount(inv)
    for sub in ('attn', 'mlp'):
        Y = sd15_X(sub).copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1])
        G, T, L, _ = Y.shape; Yf = Y.reshape(G, T, -1)
        Yf = np.where(np.isfinite(Yf), Yf, np.nanmean(Yf, 0, keepdims=True))      # the single missing cell -> population mean
        Yf = (Yf - Yf.mean(0)) / (Yf.std(0) + 1e-12); n0 = (G - (cnt ** 2).sum() / G) / (k - 1); an = []
        for t in range(T):
            Z = Yf[:, t]; mu = np.stack([Z[inv == i].mean(0) for i in range(k)])
            ssw = ((Z - mu[inv]) ** 2).sum(0); ssb = (cnt[:, None] * (mu - Z.mean(0)) ** 2).sum(0)
            msw = ssw / (G - k); msb = ssb / (k - 1); vp = np.maximum((msb - msw) / n0, 0); an.append(vp.sum() / (vp.sum() + msw.sum()))
            rows.append(dict(ch=f'SD1.5-{sub}', step=t, t=t_rf[t], sigma_VE=sig_ve[t], prompt_share_anova=an[-1]))
        an = np.array(an); d = np.diff(an)
        summ.append(dict(ch=f'SD1.5-{sub}', share_step0=an[0], share_max=an.max(), argmax_step=int(an.argmax()), share_end=an[-1], steepest_rise_interval=int(d.argmax()), steepest_rise=d.max()))
    pd.DataFrame(rows).to_csv(f'{BASE}/out/P3_prompt_share_by_step.csv', index=False)
    P3 = pd.DataFrame(summ); P3.to_csv(f'{BASE}/out/P3_prompt_share_summary.csv', index=False); print(P3.round(3).to_string(index=False), flush=True)

    # ---- P4 triple PCA ----
    rows = []
    for sub in ('attn', 'mlp'):
        X = sd15_X(sub); D = NL[:, 1][None, None, :].astype(float)
        H1 = X[..., 2] / np.log(D); H2 = np.log(X[..., 0] * D) / np.log(D); Lm = np.log(X[..., 1]); ok = np.isfinite(H1) & np.isfinite(H2) & np.isfinite(Lm)
        sh = pca_shares(H1[ok], H2[ok], Lm[ok]); rows.append(dict(ch=f'SD1.5-{sub}', pc1=sh[0], pc2=sh[1], pc12=sh[0] + sh[1], rows=int(ok.sum())))
    P4 = pd.DataFrame(rows); P4.to_csv(f'{BASE}/out/P4_triple_pca.csv', index=False); print(P4.round(4).to_string(index=False))
