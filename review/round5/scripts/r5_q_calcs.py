# Round 5 section Q: calculations exactly as frozen in R5_PREREGISTRATION.md
# (SHA-256 78b250a2b3e22047cb6fa3bf113b6900e6340648f0c3ac200f3050a7d7367018, frozen 2026-10-08T05:24:22Z).
# Inputs: R5_*.pkl (r5_load.py), F3_*.pkl (round3/audit/scripts/r3_load.py), F_*.pkl (Round 2 archive fields).
# Output: out/Q_results.json
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from r5_common import *
R = {}; rng = np.random.default_rng(55)
def pair(meta_a, meta_b):
    ka = list(zip(meta_a.pt.astype(str), meta_a.seed.astype('int64'))); kb = list(zip(meta_b.pt.astype(str), meta_b.seed.astype('int64')))
    common = [k for k in ka if k in kb]
    return [ka.index(k) for k in common], [kb.index(k) for k in common]
S10 = 'S10_1024_s1001+S10_1024_s1002'; S45 = 'S45_1024_s1001+S45_1024_s1002'
# ================= Q1 =================
_, m3, X3, s3 = R3(S10); d = R5(S10); ia, ib = pair(m3, d['meta'])
lr = np.log(d['X']['cond'][ib][..., 0] / X3[ia][..., 0])
R['Q1a_pairs'] = len(ia); R['Q1a_median_abs_log_g'] = float(np.median(np.abs(lr)))
R['Q1a_by_step_median_abs'] = np.median(np.abs(lr), (0, 2)).round(5).tolist(); R['Q1a_max_abs'] = float(np.abs(lr).max())
R['Q1a_exact_equal_frac'] = float(np.mean(np.abs(lr) < 1e-6))
for sub in ['attn', 'mlp']:
    _, mz, Xz, sz = R3('Z40_1024_s1001', sub); dz = R5('Z40_1024_s1001', sub); ia, ib = pair(mz, dz['meta'])
    l = np.log(BM(dz)[ib][..., 0] / Xz[ia][..., 0]); R[f'Q1b_pairs_{sub}'] = len(ia)
    R[f'Q1b_ZIB4_median_abs_log_g_BMnew_vs_R3_{sub}'] = float(np.median(np.abs(l)))
    R[f'Q1b_ZIB4_cond_vs_R3_{sub}'] = float(np.median(np.abs(np.log(dz['X']['cond'][ib][..., 0] / Xz[ia][..., 0]))))
    R[f'Q1b_ZIB4_gap_BMnew_vs_R3_{sub}'] = float(np.median(np.abs(np.log(BM(dz)[ib][..., 1] / Xz[ia][..., 1]))))
    _, m1z, X1z, _ = R3('Z10_1024_s1001', sub); d1z = R5('Z10_1024_s1001', sub); ia, ib = pair(m1z, d1z['meta'])
    R[f'Q1b_ZIB1_retest_median_abs_log_g_{sub}'] = float(np.median(np.abs(np.log(d1z['X']['cond'][ib][..., 0] / X1z[ia][..., 0]))))
def pooled_vs(arm3, arm5, steps_from=2):
    _, mp, Xp, sp = R3(arm3); dn = R5(arm5)
    keep = np.array([len(str(x).split(',')) == 1 for x in mp.seqs]) if 'seqs' in mp else np.ones(len(mp), bool)
    mp2 = mp[keep]; Xp2 = Xp[keep]; ia, ib = pair(mp2, dn['meta'])
    P = Xp2[ia]; B = BM(dn)[ib]; C = dn['X']['cond'][ib]; U = dn['X']['uncond'][ib]; PI = dn['PI'][ib]
    return P, B, C, U, PI, int(len(ia)), int((~keep).sum())
P, B, C, U, PI, n, nd = pooled_vs(S45, S45)
lg = np.log(P[..., 0] / B[..., 0]); lgap = np.log(P[..., 1] / B[..., 1])
R['Q1c_pairs'] = n; R['Q1c_median_log_g_pooled_over_BM_steps>=2'] = float(np.median(lg[:, 2:]))
R['Q1c_frac_positive_steps>=2'] = float(np.mean(lg[:, 2:] > 0)); R['Q1c_median_log_gap_steps>=2'] = float(np.median(lgap[:, 2:]))
R['Q1c_median_log_g_pooled_over_cond_steps>=2'] = float(np.median(np.log(P[..., 0] / C[..., 0])[:, 2:]))
R['Q1c_median_log_g_by_step'] = np.median(lg, (0, 2)).round(4).tolist(); R['Q1c_median_log_g_by_layer'] = np.median(lg[:, 2:], (0, 1)).round(4).tolist()
R['Q1c_median_log_H_diff_steps>=2'] = float(np.median((P[..., 2] - B[..., 2])[:, 2:]))
for res, a3 in [('768', 'S45_768_s1001'), ('512', 'S45_512_s1001')]:
    P_, B_, C_, U_, PI_, n_, nd_ = pooled_vs(a3, a3)
    R[f'Q1d_{res}_pairs'] = n_; R[f'Q1d_{res}_dropped_mixed_regime_gens'] = nd_
    R[f'Q1d_{res}_median_log_g_pooled_over_BM_steps>=2'] = float(np.median(np.log(P_[..., 0] / B_[..., 0])[:, 2:]))
    R[f'Q1d_{res}_median_log_gap_steps>=2'] = float(np.median(np.log(P_[..., 1] / B_[..., 1])[:, 2:]))
lhs = np.sqrt(P[..., 0]); rhs = (np.sqrt(C[..., 0]) + np.sqrt(U[..., 0])) / 2
R['Q1e_frac_cells_bound_holds'] = float(np.mean(lhs >= rhs)); R['Q1e_frac_by_step'] = np.mean(lhs >= rhs, (0, 2)).round(3).tolist()
R['Q1e_median_slack_log'] = float(np.median(np.log(lhs / rhs)))
ex = np.median(lg, 0); dis = np.median(1 - PI, 0)              # [T, L]
R['Q1f_spearman_excess_vs_1-PI'] = float(spearmanr(ex.ravel(), dis.ravel()).statistic)
R['Q1f_spearman_steps>=2'] = float(spearmanr(ex[2:].ravel(), dis[2:].ravel()).statistic)
# ================= Q2 =================
for sub in ['attn', 'mlp']:
    _, mt, Xt, st = ARCH(f'ZIT_{sub}'); dz = R5('Z40_1024_s1001', sub); d1z = R5('Z10_1024_s1001', sub); sz = dz['sigma']
    R[f'Q2a_tangent_ZIB4cond_ZIT_{sub}'], R[f'Q2a_shape_ZIB4cond_ZIT_{sub}'] = tangent(Xt, st, dz['X']['cond'], sz)
    R[f'Q2b_tangent_ZIB4uncond_ZIT_{sub}'], _ = tangent(Xt, st, dz['X']['uncond'], sz)
    R[f'Q2_ref_tangent_ZIB4BMsynth_ZIT_{sub}'], _ = tangent(Xt, st, BM(dz), sz)
    R[f'Q2_ref_tangent_ZIB1new_ZIT_{sub}'], _ = tangent(Xt, st, d1z['X']['cond'], d1z['sigma'])
    R[f'Q2c_tangent_ZIB1_ZIB4cond_{sub}'], R[f'Q2c_shape_ZIB1_ZIB4cond_{sub}'] = tangent(d1z['X']['cond'], d1z['sigma'], dz['X']['cond'], sz)
# ================= Q3 =================
def curve_part(r, grp, seed, center, fold_of_group):
    T = r.shape[1]; preds = np.zeros((T,) + r[:, -1].shape); tg = np.zeros_like(r[:, -1]); fo = fold_of_group[grp]
    for f in range(5):
        tr = np.where(fo != f)[0]; te = np.where(fo == f)[0]; rc = r.copy()
        if center:
            for sd in np.unique(seed): rc[seed == sd] = r[seed == sd] - r[tr[seed[tr] == sd]].mean(0)
        tg[te] = rc[te, -1]
        for t in range(T): preds[t, te] = ridge(rc[tr, t], rc[tr, -1])(rc[te, t])
    return np.array([1 - ((tg - preds[t]) ** 2).sum() / ((tg - tg.mean(0)) ** 2).sum() for t in range(T)])
def partitions(ngroups, n=200, seed=0):
    g = np.random.default_rng(seed); out = []
    for _ in range(n):
        f = np.empty(ngroups, int); f[g.permutation(ngroups)] = np.arange(ngroups) % 5; out.append(f)
    return out
def dist(X, ph, seed, s, center, parts):
    r, grp, _ = resid(X, ph); v = []; r0 = []
    for f in parts:
        c = curve_part(r, grp, seed, center, f); v.append(s50(c, s)); r0.append(c[0])
    v = np.array(v); return v, np.array(r0)
def summ(v): return dict(median=float(np.nanmedian(v)), p5=float(np.nanpercentile(v, 5)), p95=float(np.nanpercentile(v, 95)),
                         width=float(np.nanpercentile(v, 95) - np.nanpercentile(v, 5)), nan_frac=float(np.isnan(v).mean()))
d1 = R5(S10); d4 = R5(S45); ia, ib = pair(d1['meta'], d4['meta'])
ph = d1['meta'].ph.values[ia]; sd = d1['meta'].seed.values[ia]; s = d1['sigma']
reps = {'CFG1_cond': d1['X']['cond'][ia], 'CFG45_cond': d4['X']['cond'][ib], 'CFG45_uncond': d4['X']['uncond'][ib], 'CFG45_BM': BM(d4)[ib]}
parts30 = partitions(len(np.unique(ph)), 200, 30); Q3 = {}
for k, X in reps.items():
    for cen in [True, False]:
        v, r0 = dist(X, ph, sd, s, cen, parts30); Q3[(k, cen)] = v
        R[f'Q3_{k}_{"centred" if cen else "uncentred"}_sigma50'] = summ(v); R[f'Q3_{k}_{"centred" if cen else "uncentred"}_R2step0'] = summ(r0)
_, m60, X60, s60 = R3(S10); parts60 = partitions(m60.ph.nunique(), 200, 60)
v60, r060 = dist(X60, m60.ph.values, m60.seed.values, s60, True, parts60)
R['Q3a_R3_60prompt_CFG1_centred_sigma50'] = summ(v60); R['Q3a_R3_60prompt_CFG1_centred_R2step0'] = summ(r060)
v60u, _ = dist(X60, m60.ph.values, m60.seed.values, s60, False, parts60); R['Q3_R3_60prompt_CFG1_uncentred_sigma50'] = summ(v60u)
diff = Q3[('CFG45_cond', True)] - Q3[('CFG1_cond', True)]
R['Q3b_median_paired_diff'] = float(np.nanmedian(diff)); R['Q3b_frac_positive'] = float(np.nanmean(diff > 0))
_, ma, Xa, sa = ARCH('SD35M_attn'); core = set(d1['meta'].pt)
gid = np.array(_ if False else ARCH('SD35M_attn')[0]); pa = ma.pt.astype(str).values
sel = []
for p in sorted(core):
    idx = np.where(pa == p)[0]; idx = idx[np.argsort(gid[idx])][:2]; sel += list(idx)
sel = np.array(sel); R['Q3c_archive_gens'] = int(len(sel)); R['Q3c_archive_prompts'] = int(len(set(pa[sel])))
va, ra = dist(Xa[sel], ma.ph.values[sel], np.zeros(len(sel)), sa, False, partitions(len(set(pa[sel])), 200, 31))
R['Q3c_archive30_uncentred_sigma50'] = summ(va); R['Q3c_archive30_R2step0'] = summ(ra)
# ================= Q4 =================
def vr_one(X, k):
    r, _, _ = resid(X, np.zeros(len(X))); return VR(r, (k,))[k]
p10_40, p10_100, p45_40, p45_100 = R5('P10_40'), R5('P10_100'), R5('P45_40'), R5('P45_100')
R['Q4a_VR4_CFG1_40'] = vr_one(p10_40['X']['cond'], 4)
for lab, p, k in [('40', p45_40, 4), ('100', p45_100, 10)]:
    R[f'Q4_VR{k}_CFG45_{lab}_cond'] = vr_one(p['X']['cond'], k); R[f'Q4_VR{k}_CFG45_{lab}_uncond'] = vr_one(p['X']['uncond'], k)
    R[f'Q4_VR{k}_CFG45_{lab}_BM'] = vr_one(BM(p), k)
R['Q4_VR10_CFG1_100'] = vr_one(p10_100['X']['cond'], 10)
R['Q4c_ratio_CFG1'] = R['Q4_VR10_CFG1_100'] / R['Q4a_VR4_CFG1_40']; R['Q4c_ratio_CFG45_cond'] = R['Q4_VR10_CFG45_100_cond'] / R['Q4_VR4_CFG45_40_cond']
R['Q4_seeds'] = dict(P10_40=len(p10_40['gids']), P10_100=len(p10_100['gids']), P45_40=len(p45_40['gids']), P45_100=len(p45_100['gids']))
# ================= Q5 =================
def shift(A, sA, Bm, sB):
    j = [int(np.argmin(np.abs(sB - x))) for x in sA]; return float(np.median(np.abs(np.log(Bm.mean(0)[j]) - np.log(A.mean(0)))))
def null(G, n40, n100, draws=500):
    out = []; n = len(G)
    for _ in range(draws):
        p = rng.permutation(n); a, b = p[:n // 2], p[n // 2:]
        v = np.median(np.abs(np.log(G[a].mean(0)) - np.log(G[b].mean(0))))
        out.append(v * np.sqrt((1 / n40 + 1 / n100) / (1 / len(a) + 1 / len(b))))
    return np.array(out)
for lab, a40, a100 in [('CFG1', p10_40['X']['cond'][..., 0], p10_100['X']['cond'][..., 0]), ('CFG45_cond', p45_40['X']['cond'][..., 0], p45_100['X']['cond'][..., 0])]:
    sA = p10_40['sigma']; sB = p10_100['sigma']; obs = shift(a40, sA, a100, sB)
    nl = np.concatenate([null(a40, len(a40), len(a100)), null(a100, len(a40), len(a100))])
    R[f'Q5_{lab}_observed'] = obs; R[f'Q5_{lab}_null_p50'] = float(np.percentile(nl, 50)); R[f'Q5_{lab}_null_p95'] = float(np.percentile(nl, 95))
    R[f'Q5_{lab}_perm_p'] = float(np.mean(nl >= obs))
    j = [int(np.argmin(np.abs(sB - x))) for x in sA]; sg = np.log(a100.mean(0)[j]) - np.log(a40.mean(0))
    R[f'Q5_{lab}_median_signed'] = float(np.median(sg)); R[f'Q5_{lab}_frac_cells_positive'] = float(np.mean(sg > 0))
# ================= Q6 =================
def beta_edges(G, e):
    g = G[:, :-1].ravel(); b = np.diff(G, axis=1).ravel(); bm = np.full(24, np.nan)
    for i in range(24):
        k = (g >= e[i]) & (g < e[i + 1])
        if k.sum() >= 5: bm[i] = b[k].mean()
    return bm
def lobe_test(Gc, groups, B=500):
    """Gc [G,T,L]; groups: prompt id per generation (bootstrap unit)."""
    G = Gc.reshape(-1, Gc.shape[-1]); c, bm = beta24(G); cr = crossings(c, bm)
    if len(cr) < 3: return None
    x1, x2 = cr[0][0], cr[1][0]
    g = G[:, :-1].ravel(); e = np.linspace(np.percentile(g, 2), np.percentile(g, 98), 25)
    ug = np.unique(groups); bs = []
    for _ in range(B):
        pick = rng.choice(ug, len(ug), replace=True); idx = np.concatenate([np.where(groups == u)[0] for u in pick])
        bs.append(beta_edges(Gc[idx].reshape(-1, Gc.shape[-1]), e))
    se = np.nanstd(np.array(bs), 0)
    inside = np.where((c > x1) & (c < x2) & ~np.isnan(bm))[0]
    if len(inside) == 0:
        v = np.where(~np.isnan(bm))[0]; lo = v[c[v] < x1].max(); hi = v[c[v] > x2].min(); inside = np.array([lo, hi])
    i = inside[np.argmax(np.abs(bm[inside]))]
    return dict(x1=float(x1), x2=float(x2), bins=inside.tolist(), beta=float(bm[i]), se=float(se[i]), z=float(abs(bm[i]) / se[i]),
                main_peak_abs_beta=float(np.nanmax(np.abs(bm))), lobe_over_peak=float(abs(bm[i]) / np.nanmax(np.abs(bm))))
Q6 = {}
for lab, arm, brs in [('CFG1_1024', S10, ['cond']), ('CFG45_1024', S45, ['cond', 'uncond', 'BM']), ('CFG45_768', 'S45_768_s1001', ['cond', 'uncond', 'BM']), ('CFG45_512', 'S45_512_s1001', ['cond', 'uncond', 'BM'])]:
    dd = R5(arm); grp = pd.factorize(dd['meta'].pt)[0]
    for b in brs:
        G = (BM(dd) if b == 'BM' else dd['X'][b])[..., 0]; t = topo(G)
        res = lobe_test(G, grp) if t['signs'] == '+-+-' else None
        Q6[f'{lab}_{b}'] = dict(signs=t['signs'], crossings=t['crossings'], lobe=res)
R['Q6a'] = Q6; zs = [v['lobe']['z'] for v in Q6.values() if v['lobe']]
R['Q6a_max_z'] = float(max(zs)) if zs else None; R['Q6a_n_arms_tested'] = len(zs)
_, m512, X512, _ = R3('S45_512_s1001'); share = set(R5('S45_512_s1001')['meta'].pt)
k = np.array([p in share for p in m512.pt.astype(str)]); R['Q6b_R3_512_shared_prompts'] = int(len(set(m512.pt.astype(str)[k])))
R['Q6b_R3pooled512_shared'] = topo(X512[k][..., 0]); R['Q6b_R3pooled512_other'] = topo(X512[~k][..., 0]); R['Q6b_R3pooled512_all'] = topo(X512[..., 0])
# ================= Q7 =================
def sd_ratio(A, Bm):
    a = np.log(A[..., 0]).mean(2).std(0, ddof=1); b = np.log(Bm[..., 0]).mean(2).std(0, ddof=1); return b / a
for lab, a, b in [('40', p10_40['X']['cond'], p45_40['X']['cond']), ('100', p10_100['X']['cond'], p45_100['X']['cond'])]:
    rr = sd_ratio(a, b); q = np.array_split(np.arange(len(rr)), 4)
    R[f'Q7_{lab}_quarters'] = [float(np.median(rr[x])) for x in q]; R[f'Q7a_{lab}_last3q'] = float(np.median(rr[np.concatenate(q[1:])]))
    R[f'Q7b_{lab}_first_q'] = float(np.median(rr[q[0]]))
# ================= Q8 =================
on = {}
for lab, X, sg in [('S10_1024', d1['X']['cond'], s), ('S45_1024_cond', d4['X']['cond'], s), ('S45_1024_uncond', d4['X']['uncond'], s),
                   ('S45_768_cond', R5('S45_768_s1001')['X']['cond'], s), ('S45_512_cond', R5('S45_512_s1001')['X']['cond'], s),
                   ('P10_40', p10_40['X']['cond'], p10_40['sigma']), ('P10_100', p10_100['X']['cond'], p10_100['sigma']),
                   ('P45_40_cond', p45_40['X']['cond'], p45_40['sigma']), ('P45_100_cond', p45_100['X']['cond'], p45_100['sigma'])]:
    on[lab] = onset(X, sg)
for sub in ['attn', 'mlp']:
    on[f'Z10_{sub}'] = onset(R5('Z10_1024_s1001', sub)['X']['cond'], R5('Z10_1024_s1001', sub)['sigma'])
    on[f'Z40_cond_{sub}'] = onset(R5('Z40_1024_s1001', sub)['X']['cond'], R5('Z40_1024_s1001', sub)['sigma'])
R['Q8_onsets'] = on; sz = R5('Z40_1024_s1001')['sigma']; R['Q8_ZIB_midpoint_grid'] = [round(float(x), 4) for x in (sz[1:] + sz[:-1]) / 2 if 0.6 < x < 0.95]
json.dump(R, open(f'{OUT}/Q_results.json', 'w'), indent=1, default=str)
for k, v in R.items():
    if 'by_step' in k or 'by_layer' in k: continue
    print(k, ':', v)
