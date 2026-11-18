"""R8-1, 3, 4, 5, 6, 9, 10, 11, 13 as frozen in R8_PREREGISTRATION.md (SHA acc578ac, 2026-10-09T08:30:59Z).
Estimators are the Round 2/5/6 code, unchanged (r5_common: prof, tangent, cos, R3, R5, BM, ARCH; r5_q_calcs: pair, pooled_vs;
r5_scoreR: R1f spread-ratio definition; lens_posthoc_isotropy: g_iso). Output: out/R8_b_*.csv, out/R8_b_results.json"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import sys, json
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round6/scripts')
from r6c_common import *
from scipy.stats import spearmanr
from sklearn.preprocessing import SplineTransformer, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
RV = f'{_COSPAFS_ROOT}/review'; OUT = f'{RV}/r8_correlations/out'; PAP = f'{_COSPAFS_ROOT}/papers'
PSEED = 2026100908; NPERM = 10000
RES = {}
def perm_p(x, y, seed=PSEED):
    x = np.asarray(x, float); y = np.asarray(y, float); r = spearmanr(x, y).statistic; rg = np.random.default_rng(seed)
    nul = np.array([spearmanr(x, rg.permutation(y)).statistic for _ in range(NPERM)])
    return float(r), float((1 + (np.abs(nul) >= abs(r)).sum()) / (NPERM + 1))
def loo(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); r0 = spearmanr(x, y).statistic; out = []
    for i in range(len(x)):
        k = np.arange(len(x)) != i; out.append(spearmanr(x[k], y[k]).statistic)
    out = np.array(out); return dict(rho=float(r0), loo_min_abs=float(np.min(np.abs(out))), loo_sign_kept=bool(np.all(np.sign(out) == np.sign(r0))),
                                     loo_ok=bool(np.all(np.sign(out) == np.sign(r0)) and np.all(np.abs(out) >= 0.5)), loo=out.round(3).tolist())
def stab_sigma(s, y, tol):
    """largest sigma after which y stays within tol of its median over sigma < 0.6 (states ordered by descending sigma)"""
    o = np.argsort(-s); s = s[o]; y = y[o]; med = np.median(y[s < 0.6]); ok = np.abs(y - med) <= tol
    for i in range(len(s)):
        if ok[i:].all(): return float(s[i])
    return np.nan
CH7 = ['ZIT-attn', 'ZIT-mlp', 'ZIB-attn', 'ZIB-mlp', 'MOCHI-attn', 'SD35M-attn', 'SD35L-attn']
# ======================= R8-1 =======================
E3 = pd.read_csv(f'{RV}/round2/out/E3_plateau_onset.csv').set_index('ch')
S3 = pd.read_csv(f'{RV}/round6/out/S3_estimators.csv').set_index('channel')
E6 = pd.read_csv(f'{RV}/round2/out/E6_varcomp_by_step.csv')
F5 = pd.read_csv(f'{PAP}/mode-I/data/I-F5_semantic_decoding.csv'); F5 = F5[(F5.measure == 'ordered_profile') & F5.realization.isna()]
S4 = pd.read_csv(f'{RV}/round6/out/S4_topology.csv').set_index('set')
rows = []
for ch in CH7:
    e = E6[E6.ch == ch]; f = F5[F5.channel == ch]
    rows.append(dict(channel=ch, plateau_onset=float(E3.loc[ch, 'onset_sigma']), rel_onset=float(S3.loc[ch, 'matched60_rel_median']),
                     share_stab=stab_sigma(e.sigma.values, e.prompt_share_naive.values, 0.02),
                     decode_stab=stab_sigma(f.sigma.values, f.acc.values, 0.05), decode_states=len(f)))
A1 = pd.DataFrame(rows); A1.to_csv(f'{OUT}/R8_1_events.csv', index=False)
win = {}
for ch in ['ZIT-mlp', 'ZIB-mlp']:
    r = S4.loc[f'archive {ch}']; lo, hi = float(r.break2_sigma), float(r.break1_sigma); ro = float(A1.set_index('channel').loc[ch, 'rel_onset'])
    win[ch] = dict(window=[lo, hi], rel_onset=ro, inside=bool(lo <= ro <= hi))
RES['R8-1a'] = ('aligned' if all(v['inside'] for v in win.values()) else 'not aligned (bottleneck stays an envelope description)', win)
qs = ['plateau_onset', 'rel_onset', 'share_stab', 'decode_stab']; pr = {}
for i in range(4):
    for j in range(i + 1, 4):
        a, b = A1[qs[i]].values, A1[qs[j]].values; k = np.isfinite(a) & np.isfinite(b); L = loo(a[k], b[k])
        L['perm_p'] = perm_p(a[k], b[k])[1]; L['n'] = int(k.sum()); L['aligned'] = bool(L['rho'] >= 0.75 and L['loo_ok']); pr[f'{qs[i]}~{qs[j]}'] = L
RES['R8-1b'] = pr
print(A1.round(3).to_string()); print('R8-1a', RES['R8-1a']); [print(k, round(v['rho'], 3), v['loo_ok'], v['aligned']) for k, v in pr.items()]
# ======================= R8-3 =======================
PH = pd.read_csv(f'{RV}/lenses/out/PH_isotropy_baseline.csv'); L2 = pd.read_csv(f'{RV}/lenses/out/L2_R1_autonomy.csv')
nm = {'ZIT attn': 'ZIT-attn', 'ZIT MLP': 'ZIT-mlp', 'ZIB attn': 'ZIB-attn', 'ZIB MLP': 'ZIB-mlp', 'Mochi attn': 'MOCHI-attn', 'SD3.5 M attn': 'SD35M-attn', 'SD3.5 L attn': 'SD35L-attn'}
PH['ch'] = PH.channel.map(nm); L2['ch'] = L2.channel.map(nm)
F50 = F5[F5.step == 0].set_index('channel').acc
rows = [dict(channel=c, arch='transformer', ratio=float(PH.set_index('ch').loc[c, 'ratio_median_to_iso']), decode0=float(F50[c]),
             onset=float(E3.loc[c, 'onset_sigma']), dR2=float(L2.set_index('ch').loc[c, 'dR2'])) for c in CH7]
CU = pd.read_csv(f'{RV}/r3_unet/out/C_scores.csv'); CU = CU[CU.branch == 'cond'].set_index('channel')
P2 = pd.read_csv(f'{RV}/r3_unet/out/P2_plateau_onset.csv').set_index('ch'); P5 = pd.read_csv(f'{RV}/r3_unet/out/P5_decoding.csv')
for sub in ['attn', 'mlp']:
    rows.append(dict(channel=f'SD1.5-{sub}', arch='unet', ratio=float(CU.loc[sub, 'C1_median_cell_ratio']),
                     decode0=float(P5[(P5.channel == f'SD1.5-{sub}') & (P5.measure == 'ordered_profile') & (P5.step == 0)].acc.iloc[0]),
                     onset=float(P2.loc[f'SD1.5-{sub}', 'onset_t']), dR2=float(CU.loc[sub, 'C2_dR2'])))
BD = json.load(open(f'{RV}/r3_ace/out/B_details.json'))
for sub in ['attn', 'mlp']:
    rows.append(dict(channel=f'ACE-{sub}', arch='audio', ratio=BD[f'B2_{sub}']['median_ratio'], decode0=np.nan,
                     onset=BD['B8'][sub]['onset_sigma'], dR2=BD['B4'][sub]['dR2']))
A3 = pd.DataFrame(rows); A3['log_ratio'] = np.log(A3.ratio); A3.to_csv(f'{OUT}/R8_3_channel_table.csv', index=False)
print(A3.round(3).to_string())
k = A3.decode0.notna(); ra, pa = perm_p(A3.log_ratio[k], A3.decode0[k]); t7 = A3.arch == 'transformer'; r7, p7 = perm_p(A3.log_ratio[t7], A3.decode0[t7])
RES['R8-3a'] = ('supported' if (ra <= -0.6 and r7 <= -0.6) else 'tested, not supported', dict(rho_all=ra, p_all=pa, n_all=int(k.sum()), rho_transformers=r7, p_transformers=p7, loo_all=loo(A3.log_ratio[k], A3.decode0[k])))
L3b = loo(A3.log_ratio, A3.onset); L3b['perm_p'] = perm_p(A3.log_ratio, A3.onset)[1]
L3b_t = loo(A3.log_ratio[t7], A3.onset[t7]); L3b_t['perm_p'] = perm_p(A3.log_ratio[t7], A3.onset[t7])[1]
RES['R8-3b'] = ('associated' if (abs(L3b['rho']) >= 0.6 and L3b['loo_ok']) else 'tested, not supported', dict(all11=L3b, transformers7_descriptive=L3b_t))
o = A3.sort_values('ratio'); th = np.array_split(np.arange(len(o)), 3); most = o.iloc[th[0]].dR2; least = o.iloc[th[2]].dR2
ov = bool(max(most.min(), least.min()) <= min(most.max(), least.max())); rc, pc_ = perm_p(A3.log_ratio, A3.dR2)
RES['R8-3c'] = ('non-autonomy robust to anisotropy' if (abs(rc) < 0.5 and ov) else 'tested, not supported',
                dict(rho=rc, perm_p=pc_, most_aniso_third=dict(zip(o.iloc[th[0]].channel, most.round(3))), least_aniso_third=dict(zip(o.iloc[th[2]].channel, least.round(3))), ranges_overlap=ov))
for kk in ['R8-3a', 'R8-3b', 'R8-3c']: print(kk, RES[kk][0], {a: (round(b, 3) if isinstance(b, float) else b) for a, b in RES[kk][1].items() if not isinstance(b, dict)})
# ======================= R8-6 and R8-9 (tabulations) =======================
Q = json.load(open(f'{RV}/round5/out/Q_results.json')); E10 = pd.read_csv(f'{RV}/round2/out/E10_tangent.csv')
r6 = {}
for sub in ['attn', 'mlp']:
    arch_t = float(E10[(E10.a == f'ZIT-{sub}') & (E10.b == f'ZIB-{sub}')].tangent_cos.iloc[0])
    t1 = Q[f'Q2_ref_tangent_ZIB1new_ZIT_{sub}']; t4 = Q[f'Q2a_tangent_ZIB4cond_ZIT_{sub}']
    dg = t1 - t4; dc = t4 - arch_t
    r6[sub] = dict(tangent_CFG1_cond=t1, tangent_CFG4_cond=t4, tangent_archive_pooled=arch_t, d_guidance=dg, d_capture=dc, holds=bool(abs(dg) < abs(dc) / 3),
                   S4_contraction={k_: float(S4.loc[k_, 'contraction']) for k_ in [f'archive ZIB-{sub}', f'R4 ZIB CFG4 cond-{sub}', f'R4 ZIB CFG1 cond-{sub}', f'R3 ZIB CFG1 cond-only-{sub}']})
RES['R8-6'] = ('the drop is the capture functional, not guidance' if all(v['holds'] for v in r6.values()) else 'tested, not supported', r6)
print('R8-6', RES['R8-6'][0], {s_: (round(v['d_guidance'], 4), round(v['d_capture'], 4)) for s_, v in r6.items()})
nl = E10[~E10.kind.isin(['lineage', 'same-model CFG'])]
r9a = perm_p(E10.shape_r, E10.tangent_cos); r9b = perm_p(nl.shape_r, nl.tangent_cos)
RES['R8-9'] = ('tabulation', dict(rho_13=r9a[0], p_13=r9a[1], rho_10=r9b[0], p_10=r9b[1], n_10=len(nl)))
print('R8-9', RES['R8-9'])
# ======================= R8-5 =======================
THIRDS = [slice(0, 8), slice(8, 16), slice(16, 24)]
def tangent_thirds(XA, sa, XB, sb):
    A = prof(XA, sa); B = prof(XB, sb); out = []
    for sl in [slice(0, 24)] + THIRDS:
        tc = []
        for t, x in enumerate(sa):
            j = int(np.argmin(np.abs(sb - x))); tc.append(cos(np.diff(A[t][sl], axis=0), np.diff(B[j][sl], axis=0)))
        out.append(float(np.mean(tc)))
    return out
r5 = {}; rows = []
for sub in ['attn', 'mlp']:
    _, mt, Xt, st = ARCH(f'ZIT_{sub}'); _, mb, Xb, sb_ = ARCH(f'ZIB_{sub}'); st = np.asarray(st, float); sb_ = np.asarray(sb_, float)
    full, e, m, l = tangent_thirds(Xt, st, Xb, sb_); rows.append(dict(capture='archive pooled CFG4', sub=sub, full=full, early=e, middle=m, late=l))
    dz = R5('Z40_1024_s1001', sub); f2 = tangent_thirds(Xt, st, dz['X']['cond'], dz['sigma']); rows.append(dict(capture='R4 CFG4 cond', sub=sub, full=f2[0], early=f2[1], middle=f2[2], late=f2[3]))
    r5[sub] = dict(early_minus_late_archive=e - l, holds=bool(e - l >= 0.05))
A5 = pd.DataFrame(rows); A5.to_csv(f'{OUT}/R8_5_tangent_thirds.csv', index=False); print(A5.round(4).to_string())
E8 = pd.read_csv(f'{RV}/round2/out/E8_ablit_prefill_by_layer.csv'); sp = np.array_split(np.arange(len(E8)), 3)
mis = dict(abs_dlogg_thirds=[float(E8.d_g_micro.iloc[x].abs().mean()) for x in sp], layers=[[int(x[0]), int(x[-1])] for x in sp])
a5 = all(v['holds'] for v in r5.values())
RES['R8-5a'] = ('dynamical inheritance is depth-localised' if a5 else 'tested, not supported', r5)
RES['R8-5b'] = ('one depth law across interventions' if (a5 and mis['abs_dlogg_thirds'][2] > mis['abs_dlogg_thirds'][0]) else 'tested, not supported', mis)
print('R8-5', RES['R8-5a'][0], RES['R8-5b'])
# ======================= R8-13 =======================
rg = np.random.default_rng(64); r13 = {}; rows = []
for sub in ['attn', 'mlp']:
    _, mt, Xt, st = ARCH(f'ZIT_{sub}'); _, mb, Xb, sb_ = ARCH(f'ZIB_{sub}'); st = np.asarray(st, float); sb_ = np.asarray(sb_, float)
    pt_t = mt.pt.astype(str).str.strip().values; pt_b = mb.pt.astype(str).str.strip().values; shared = np.array(sorted(set(pt_t) & set(pt_b)))
    full = tangent(Xt, st, Xb, sb_)[0]; full_shared = tangent(Xt[np.isin(pt_t, shared)], st, Xb[np.isin(pt_b, shared)], sb_)[0]
    for n in [60, 120, 240]:
        v = []; ng = []
        for _ in range(50):
            pk = rg.choice(shared, n // 4, replace=False); it = np.isin(pt_t, pk); ib = np.isin(pt_b, pk)
            v.append(tangent(Xt[it], st, Xb[ib], sb_)[0]); ng.append((int(it.sum()), int(ib.sum())))
        v = np.array(v); rows.append(dict(sub=sub, n=n, prompts=n // 4, gens_ZIT_mean=np.mean([a for a, _ in ng]), gens_ZIB_mean=np.mean([b for _, b in ng]),
                                          median=float(np.median(v)), p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)), full=full, full_shared_prompts=full_shared))
    r60 = rows[-3]; r13[sub] = dict(full=full, median60=r60['median'], p5_60=r60['p5'], holds=bool(abs(r60['median'] - full) <= 0.02 and r60['p5'] > 0.24))
A13 = pd.DataFrame(rows); A13.to_csv(f'{OUT}/R8_13_tangent_n.csv', index=False); print(A13.round(4).to_string())
RES['R8-13'] = ('n-stable' if all(v['holds'] for v in r13.values()) else 'tested, not supported', r13)
# ======================= R8-10 and R8-11 =======================
S10 = 'S10_1024_s1001+S10_1024_s1002'; S45 = 'S45_1024_s1001+S45_1024_s1002'
d1 = R5(S10); d4 = R5(S45); m1 = d1['meta']; k1 = list(zip(m1.pt, m1.seed)); k4 = list(zip(d4['meta'].pt, d4['meta'].seed)); o4 = [k4.index(k_) for k_ in k1]
C4 = d4['X']['cond'][o4]; PI4 = d4['PI'][o4]; X1 = d1['X']['cond']; s = d1['sigma']; q = np.array_split(np.arange(len(s)), 4); late = np.concatenate(q[1:])
def sd_layer(X, m):
    G, T, L, _ = X.shape; Y = np.log(X[..., 0]).reshape(G, -1); d = pd.DataFrame(Y); d['ph'] = m.ph.values
    return d.groupby('ph').std().values.reshape(-1, T, L), d.groupby('ph').std().index
A_, ia = sd_layer(X1, m1); B_, ib = sd_layer(C4, m1); assert (ia == ib).all()
n_zero = int((A_ == 0).sum())   # CFG1 cells whose two seeds logged identical g (6-decimal precision): ratio undefined
with np.errstate(divide='ignore', invalid='ignore'): ratio = B_ / A_
ratio[~np.isfinite(ratio)] = np.nan; ratio_l = np.nanmedian(ratio[:, late, :], (0, 1)); div_l = np.nanmean((1 - PI4)[:, late, :], (0, 1))
r10, p10 = perm_p(ratio_l, div_l)
pd.DataFrame(dict(layer=np.arange(len(ratio_l)), spread_ratio=ratio_l, mean_1mPI=div_l)).to_csv(f'{OUT}/R8_10_by_layer.csv', index=False)
RES['R8-10'] = ('contraction strongest where branches diverge most' if (r10 <= -0.5 and p10 < 0.05) else 'tested, not supported',
                dict(rho=r10, perm_p=p10, n_layers=len(ratio_l), n_zero_spread_cells=n_zero, cells_total=int(A_.size), ratio_median_all_layers=float(np.nanmedian(ratio[:, late, :])), ratio_range=[float(ratio_l.min()), float(ratio_l.max())]))
print('R8-10', RES['R8-10'])
r11 = {}; rows = []
for lab, PIx, sg in [('SD35M CFG4.5 1024 attn', PI4, s)] + [(f'ZIB CFG4 {sub}', R5('Z40_1024_s1001', sub)['PI'], R5('Z40_1024_s1001', sub)['sigma']) for sub in ['attn', 'mlp']]:
    M = np.nanmedian(1 - PIx, 0); am = M.argmax(1)
    for t in range(M.shape[0]):
        for l in range(M.shape[1]): rows.append(dict(capture=lab, step=t, sigma=float(sg[t]), layer=l, median_1mPI=float(M[t, l])))
    rr = spearmanr(np.arange(len(am)), am).statistic
    r11[lab] = dict(argmax_layer_by_step=am.tolist(), spearman_argmax_vs_step=float(rr), argmax_steps_ge2=am[2:].tolist())
pd.DataFrame(rows).to_csv(f'{OUT}/R8_11_divergence_map.csv', index=False)
RES['R8-11'] = ('descriptive', r11); print('R8-11', {k_: (v['spearman_argmax_vs_step'], v['argmax_layer_by_step']) for k_, v in r11.items()})
# ======================= R8-4 =======================
def pair(meta_a, meta_b):   # copied unchanged from round5/scripts/r5_q_calcs.py
    ka = list(zip(meta_a.pt.astype(str), meta_a.seed.astype('int64'))); kb = list(zip(meta_b.pt.astype(str), meta_b.seed.astype('int64')))
    common = [k for k in ka if k in kb]
    return [ka.index(k) for k in common], [kb.index(k) for k in common]
def pooled_vs_ph(arm3, arm5):
    _, mp, Xp, sp_ = R3(arm3); dn = R5(arm5)
    keep = np.array([len(str(x).split(',')) == 1 for x in mp.seqs]) if 'seqs' in mp else np.ones(len(mp), bool)
    mp2 = mp[keep]; Xp2 = Xp[keep]; ia, ib = pair(mp2, dn['meta'])
    return Xp2[ia], BM(dn)[ib], dn['PI'][ib], mp2.pt.astype(str).values[ia], int(np.median([int(str(x)) for x in mp2.seqs]))
cells = []; seqs = {}
for res_, arm in [('1024', S45), ('768', 'S45_768_s1001'), ('512', 'S45_512_s1001')]:
    P, B, PI, ph, sq = pooled_vs_ph(arm, arm); seqs[res_] = sq
    e = np.log(P[..., 0] / B[..., 0])[:, 2:]; dd = (1 - PI)[:, 2:]; G, T, L = e.shape
    cells.append(pd.DataFrame(dict(res=res_, ph=np.repeat(ph, T * L), e=e.ravel(), d=dd.ravel())))
CE = pd.concat(cells, ignore_index=True); CE = CE[np.isfinite(CE.e) & np.isfinite(CE.d)].reset_index(drop=True)
from sklearn.model_selection import GroupKFold
def cv_r2(Xdf, y, groups, model_fn):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xdf, groups=groups):
        mdl = model_fn().fit(Xdf.iloc[tr], y[tr]); pred[te] = mdl.predict(Xdf.iloc[te])
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
spl = lambda col: ColumnTransformer([('s', SplineTransformer(n_knots=5, degree=3), [col])])
mA = lambda: make_pipeline(spl('d'), LinearRegression())
mB = lambda: make_pipeline(ColumnTransformer([('s', SplineTransformer(n_knots=5, degree=3), ['d']), ('r', OneHotEncoder(drop='first'), ['res'])]), LinearRegression())
grp = pd.factorize(CE.ph)[0]; yA = CE.e.values
r2A = cv_r2(CE[['d', 'res']], yA, grp, mA); r2B = cv_r2(CE[['d', 'res']], yA, grp, mB); dR = r2B - r2A
fe = mB().fit(CE[['d', 'res']], yA); coefs = fe[-1].coef_[-2:]
rgm = np.random.default_rng(PSEED)
def giso(N, D=1536):
    s_ = np.linalg.svd(rgm.standard_normal((N, D)) / np.sqrt(N), compute_uv=False); return s_.sum() ** 2 / (D * (s_ ** 2).sum())
mp_ref = {}
for res_, sq in seqs.items():
    N = sq // 2; v = [np.log(giso(2 * N)) - np.log(giso(N)) for _ in range(20)]
    mp_ref[res_] = dict(N=N, log_giso_2N_minus_N=float(np.mean(v)), sd=float(np.std(v)), observed_median_e=float(CE[CE.res == res_].e.median()))
r4c = {}
for res_ in seqs:
    c_ = CE[CE.res == res_].reset_index(drop=True); r4c[res_] = cv_r2(c_[['e']], c_.d.values, pd.factorize(c_.ph)[0], lambda: make_pipeline(spl('e'), LinearRegression()))
RES['R8-4'] = dict(n_cells=len(CE), cells_by_res=CE.res.value_counts().to_dict(), heldout_R2_spline=r2A, heldout_R2_spline_plus_resFE=r2B, dR2_resFE=dR, resFE_coefs_vs_1024=dict(zip(['512', '768'], coefs.round(4).tolist())),
                   verdict_a_b=('branch difference explains the resolution trend' if dR < 0.02 else ('N/D component present' if dR >= 0.05 else 'partly')), MP_reference=mp_ref)
RES['R8-4c'] = ('excess is a usable proxy for 1-PI' if all(v >= 0.5 for v in r4c.values()) else 'tested, not supported', r4c)
print('R8-4', RES['R8-4']); print('R8-4c', RES['R8-4c'])
def conv(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
json.dump(RES, open(f'{OUT}/R8_b_results.json', 'w'), indent=1, default=conv)
