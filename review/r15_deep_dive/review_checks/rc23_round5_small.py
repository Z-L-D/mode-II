# RC23 (post hoc, round-5 reviews 2026-11-09). Usage: python rc23_round5_small.py [A|B|C|D ...]
# A  ACE-Step RSA power (Kimi minor): split-half reliability of each model's prompt RDM (seed halves within prompt, 100 splits,
#    Spearman-Brown), the implied ceiling sqrt(relA relB) on an observable Mantel rho, the observed value a true rho of 0.30 would
#    give, and the RSA between same-weights captures (ACE Base GPU 0 vs GPU 1; turbo 8 vs 6 steps) with the R15-3 estimator.
# B  relative onset vs ridge alpha (DeepSeek Q8): the RC20 code (Round 6 S3 curve) with alpha 1, 10, 100 (frozen), 1000, 10000.
# C  pooling theorem slack (DeepSeek Q9): distribution of log(g_pool / energy-weighted mean) in the Fig. II-2a Monte Carlo.
# D  local complex-mode census null (Kimi minor, Mode-II Sec. 11): one layer-order permutation per draw applied to every
#    generation of a domain cell, the affine layer map refitted; complex-pair rate and dominant modulus against 200 draws.
import os as _os, sys
_COSPAFS_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))
HERE = _os.path.dirname(_os.path.abspath(__file__)); OUTR = _os.path.join(HERE, 'out')
import numpy as np, pandas as pd, pickle
from scipy.stats import rankdata
WHAT = sys.argv[1:] or ['A', 'B', 'C', 'D']
CA = _os.environ.get('CACHE', f'{_COSPAFS_ROOT}/cache'); R16 = _os.environ.get('R16CUBES', CA)
def F(n, d=CA):
    g, m, X, s = pickle.load(open(f'{d}/F_{n}.pkl', 'rb')); return np.asarray(m.pt.astype(str).str.strip().tolist()), m, X.astype(float), np.asarray(s, float)
def feat(X, t): return np.concatenate([np.log(np.abs(X[:, t, :, 0]) + 1e-12), np.log(np.abs(X[:, t, :, 1]) + 1e-12), X[:, t, :, 2]], 1)
def pmeans(k, Fm, P):
    M = np.stack([Fm[k == p].mean(0) for p in P]); sd = M.std(0); return (M - M.mean(0)) / (sd + 1e-12)
def rsa(A, B, nperm=0, seed=15):
    iu = np.triu_indices(len(A), 1); RA = 1 - np.corrcoef(A); RB = 1 - np.corrcoef(B)
    a = rankdata(RA[iu]); b = rankdata(RB[iu]); r = float(np.corrcoef(a, b)[0, 1])
    if not nperm: return r, np.nan, np.nan
    rng = np.random.default_rng(seed); n = len(A); nul = np.array([np.corrcoef(a, rankdata(RB[np.ix_(p, p)][iu]))[0, 1] for p in (rng.permutation(n) for _ in range(nperm))])
    return r, float((1 + (nul >= r).sum()) / (nperm + 1)), float(np.percentile(nul, 95))
if 'A' in WHAT:
    rng = np.random.default_rng(23); rows = []
    def splithalf(k, Fm, P):
        v = []
        for _ in range(100):
            ha, hb = [], []
            for p in P:
                idx = np.where(k == p)[0]; rng.shuffle(idx); h = len(idx) // 2; ha.append(Fm[idx[:h]].mean(0)); hb.append(Fm[idx[h:2 * h]].mean(0))
            A = np.stack(ha); B = np.stack(hb); A = (A - A.mean(0)) / (A.std(0) + 1e-12); B = (B - B.mean(0)) / (B.std(0) + 1e-12); v.append(rsa(A, B)[0])
        r = float(np.mean(v)); return r, 2 * r / (1 + r) if r > -1 else np.nan
    for sub in ('attn', 'mlp'):
        cub = {n: F(f'{n}_{sub}') for n in ('ACEB', 'ACET', 'ACEB0', 'ACEB1', 'Z1', 'ZIT')}
        cub['T8'] = F(f'T8_{sub}', R16)
        for lab, sig in [('first', 1.0), ('sigma0.75', 0.75), ('last', 0.0)]:
            rel = {}
            for n in ('ACEB', 'ACET', 'T8', 'Z1', 'ZIT'):
                k, _, X, s = cub[n]; t = int(np.argmin(np.abs(s - sig))); cnt = pd.Series(k).value_counts(); P = sorted(cnt[cnt >= 2].index)
                r, sb = splithalf(k, feat(X, t), P); rel[n] = sb
                rows.append(dict(part='reliability', sub=sub, state=lab, model=n, prompts=len(P), gens_per_prompt=float(cnt[P].mean()), splithalf_r=r, spearman_brown=sb))
            for a, b in [('ACEB', 'ACET'), ('Z1', 'ZIT')]:
                ceil = float(np.sqrt(max(rel[a], 0) * max(rel[b], 0)))
                ka, _, XA, sa = cub[a]; kb, _, XB, sb_ = cub[b]; P = sorted(set(ka) & set(kb))
                ta = int(np.argmin(np.abs(sa - sig))); tb = int(np.argmin(np.abs(sb_ - sa[ta]))); r, p, n95 = rsa(pmeans(ka, feat(XA, ta), P), pmeans(kb, feat(XB, tb), P), 999)
                rows.append(dict(part='ceiling', sub=sub, state=lab, model=f'{a}-{b}', prompts=len(P), ceiling=ceil, expected_obs_true030=0.30 * ceil, observed=r, p=p, null95=n95))
            for a, b in [('ACEB0', 'ACEB1'), ('T8', 'ACET')]:
                ka, _, XA, sa = cub[a]; kb, _, XB, sb_ = cub[b]; P = sorted(set(ka) & set(kb))
                ta = int(np.argmin(np.abs(sa - sig))); tb = int(np.argmin(np.abs(sb_ - sa[ta]))); r, p, n95 = rsa(pmeans(ka, feat(XA, ta), P), pmeans(kb, feat(XB, tb), P), 999)
                rows.append(dict(part='same weights', sub=sub, state=lab, model=f'{a}-{b}', prompts=len(P), observed=r, p=p, null95=n95))
            print(pd.DataFrame(rows[-9:]).round(3).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(f'{OUTR}/rc23A_ace_rsa_power.csv', index=False)
if 'B' in WHAT:
    import sys as _s; _s.path.insert(0, f'{_COSPAFS_ROOT}/review/round6/scripts'); _s.path.insert(0, f'{_COSPAFS_ROOT}/review/round5/scripts')
    cwd = _os.getcwd(); _os.chdir(f'{_COSPAFS_ROOT}/review/round5/scripts')
    exec(open(f'{_COSPAFS_ROOT}/review/round5/scripts/r5_q_calcs.py').read().split('# ================= Q3 =================')[1].split('d1 = R5(S10)')[0])
    from r6c_common import arch, resid
    import r5_common
    _os.chdir(cwd)
    base_ridge = r5_common.ridge
    def cross(c, s, th):
        T = len(c); tau = np.arange(T) / (T - 1)
        if not (c > th).any(): return np.nan
        i = int(np.argmax(c > th)); t = tau[0] if i == 0 else tau[i - 1] + (th - c[i - 1]) * (tau[i] - tau[i - 1]) / (c[i] - c[i - 1])
        return float(np.interp(t, tau, s))
    CH = [('ZIT', 'attn'), ('ZIT', 'mlp'), ('ZIB', 'attn'), ('ZIB', 'mlp'), ('MOCHI', 'attn'), ('SD35M', 'attn'), ('SD35L', 'attn')]
    rows = []
    for m, sub in CH:
        g, meta, X, s = arch(m, sub); ph = meta.pt.astype(str).str.strip().values; seed0 = np.zeros(len(X)); r, grp, _ = resid(X, ph)
        row = dict(channel=f'{m}-{sub}')
        for al in (1.0, 10.0, 100.0, 1000.0, 10000.0):
            globals()['ridge'] = (lambda A, B, alpha=al: base_ridge(A, B, alpha)); v = []
            for f in partitions(len(np.unique(grp)), 20, 1):
                c = curve_part(r, grp, seed0, False, f); v.append(cross(c / c[-1], s, 0.5) if c[-1] > 0 else np.nan)
            row[f'alpha_{int(al)}'] = float(np.nanmedian(v))
        rows.append(row); print(row, flush=True)
    globals()['ridge'] = base_ridge
    RB = pd.DataFrame(rows); RB.to_csv(f'{OUTR}/rc23B_onset_ridge_alpha.csv', index=False)
    cols = [c for c in RB.columns if c.startswith('alpha')]
    print('max |change| vs alpha 100:', float((RB[cols].sub(RB['alpha_100'], axis=0)).abs().max().max()))
    print('Spearman with alpha 100:', {c: round(RB[c].corr(RB['alpha_100'], method='spearman'), 3) for c in cols})
if 'C' in WHAT:
    def amp_g(M):
        ev = np.clip(np.linalg.eigvalsh(M), 0, None); D = len(ev); return float(np.sqrt(ev).sum() ** 2 / (D * ev.sum()))
    rng = np.random.default_rng(2026); rec = []
    for k in range(20000):
        D = int(rng.integers(2, 33)); Ms = []
        for b in range(2):
            r = int(rng.integers(1, D + 1)); W = rng.normal(size=(r, D)) * rng.gamma(0.7, 1.0, size=D); mu = rng.normal(size=D) * rng.uniform(0, 2)
            M = (W.T @ W) / r + np.outer(mu, mu); Ms.append(M * np.exp(rng.normal(0, 1.0)))
        gi = np.array([amp_g(M) for M in Ms]); E = np.array([np.trace(M) for M in Ms]); gp = amp_g((Ms[0] + Ms[1]) / 2)
        Mn = [M / np.trace(M) for M in Ms]; dis = float(np.linalg.norm(Mn[0] - Mn[1]) / (np.linalg.norm(Mn[0]) + np.linalg.norm(Mn[1])))
        rec.append((D, np.log(gp / ((E * gi).sum() / E.sum())), dis, E.max() / E.min()))
    S = pd.DataFrame(rec, columns=['D', 'slack', 'dissimilarity', 'energy_ratio'])
    q = S.slack.quantile([0, .05, .25, .5, .75, .95, 1]).to_dict()
    from scipy.stats import spearmanr
    out = dict(n=len(S), violations=int((S.slack < -1e-12).sum()), **{f'q{int(k * 100)}': v for k, v in q.items()},
               spearman_slack_dissimilarity=float(spearmanr(S.slack, S.dissimilarity)[0]), share_below_001=float((S.slack < 0.01).mean()))
    print(out); pd.DataFrame([out]).to_csv(f'{OUTR}/rc23C_pooling_slack.csv', index=False)
if 'D' in WHAT:
    def fitA(Xs):
        A0 = Xs[:, :-1].reshape(-1, 3); A1 = Xs[:, 1:].reshape(-1, 3)
        coef, *_ = np.linalg.lstsq(np.c_[A0, np.ones(len(A0))], A1, rcond=None); ev = np.linalg.eigvals(coef[:3].T)
        return float(np.abs(ev).max()), bool((np.abs(ev.imag) > 1e-9).any())
    rng = np.random.default_rng(11); rows = []
    for m, sub in [('ZIT', 'attn'), ('ZIT', 'mlp'), ('ZIB', 'attn'), ('ZIB', 'mlp'), ('MOCHI', 'attn'), ('SD35M', 'attn'), ('SD35L', 'attn')]:
        _, meta, X, s = F(f'{m}_{sub}'); Y = X.copy(); Y[..., 1] = -Y[..., 1]; T, L = X.shape[1], X.shape[2]
        for st, t in [('first', 0), ('middle', T // 2), ('last', T - 1)]:
            for dom in sorted(meta.dom.unique()):
                idx = np.where(meta.dom.values == dom)[0]; mod, cx = fitA(Y[idx, t]); nm = []; nc = []
                for _ in range(200):
                    pm = rng.permutation(L); a, b = fitA(Y[idx, t][:, pm]); nm.append(a); nc.append(b)
                rows.append(dict(channel=f'{m}-{sub}', state=st, domain=dom, modulus=mod, complex=cx, null_mod_med=float(np.median(nm)),
                                 null_mod_p05=float(np.percentile(nm, 5)), null_mod_p95=float(np.percentile(nm, 95)), null_complex_rate=float(np.mean(nc))))
        print(m, sub, flush=True)
    DD = pd.DataFrame(rows); DD.to_csv(f'{OUTR}/rc23D_complex_mode_null.csv', index=False)
    S = DD.groupby(['channel', 'state'], sort=False).agg(complex_obs=('complex', 'mean'), complex_null=('null_complex_rate', 'mean'),
        mod_obs=('modulus', 'median'), mod_null=('null_mod_med', 'median'), above_null95=('modulus', lambda v: 0)).reset_index()
    S['mod_above_null_p95'] = DD.assign(a=DD.modulus > DD.null_mod_p95).groupby(['channel', 'state'], sort=False).a.mean().values
    S['mod_below_null_p05'] = DD.assign(a=DD.modulus < DD.null_mod_p05).groupby(['channel', 'state'], sort=False).a.mean().values
    S = S.drop(columns='above_null95'); S.to_csv(f'{OUTR}/rc23D_complex_mode_null_summary.csv', index=False); print(S.round(3).to_string(index=False))
    print('overall: complex obs', DD.complex.mean().round(3), 'null', DD.null_complex_rate.mean().round(3), '; modulus above null p95 in', (DD.modulus > DD.null_mod_p95).mean().round(3))
