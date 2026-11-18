"""R9 Part A (Codex memo), as frozen in R9_PREREGISTRATION.md (SHA 09c1f170, 2026-10-09T09:07:54Z).
Features: u3/S2 feats (unchanged); theme_r: Round 6 S2 (unchanged). Output: out/R9_A_*.csv, out/R9_A_results.json"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import sys, json, pickle, itertools
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round6/scripts')
from r6c_common import *
from scipy.stats import spearmanr
RV = f'{_COSPAFS_ROOT}/review'; OUT = f'{RV}/r9_salvage/out'; RES = {}
def exact_p(x, y):
    r = spearmanr(x, y).statistic; ps = [spearmanr(x, p).statistic for p in itertools.permutations(y)]
    return float(r), float(np.mean(np.abs(ps) >= abs(r) - 1e-12))
def loo(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); r0 = spearmanr(x, y).statistic
    o = np.array([spearmanr(np.delete(x, i), np.delete(y, i)).statistic for i in range(len(x))])
    return dict(rho=float(r0), loo=o.round(3).tolist(), loo_ok=bool(np.all(np.sign(o) == np.sign(r0)) and np.all(np.abs(o) >= 0.5)))
# ---------- A1 [tab] ----------
TH = pd.read_csv(f'{RV}/round6/out/S2_theme_residual.csv'); RS = pd.read_csv(f'{RV}/round6/out/S2_rsa.csv'); S1 = pd.read_csv(f'{RV}/round6/out/S1_hierarchy.csv')
a1 = {}
for tau in ['first', 'middle', 'last']:
    a = RS[(RS.kind == 'unrelated') & (RS.tau == tau)].set_index(['a', 'b']).rsa; b = TH[(TH.kind == 'unrelated') & (TH.tau == tau)].set_index(['a', 'b']).theme_r.loc[a.index]
    t = S1.set_index(['a', 'b']).L3_tangent.loc[a.index]
    a1[tau] = dict(rsa_theme=exact_p(a.values, b.values), rsa_tangent=exact_p(a.values, t.values))
RES['R9-A1'] = ('tabulation (memo reproduced)', a1); print('A1', a1)
# ---------- A2 U-Net held-out pairs ----------
c = np.load(f'{RV}/r3_unet/out/u0_cache.npz', allow_pickle=True); upt = np.char.strip(c['pt'].astype(str))
def X_sd(sub): return np.stack([c[f'{sub}_cond_{v}'].astype(float) for v in ('g_micro', 'spectral_gap', 'sv_entropy')], -1)
D = {m: arch(m) for m in MODELS}
keys = [set(D[m][1].pt.astype(str).str.strip()) for m in MODELS] + [set(upt)]
P = sorted(set.intersection(*keys)); NP = len(P)
zdom = D['ZIT'][1].assign(k=D['ZIT'][1].pt.astype(str).str.strip()).groupby('k').dom.first().loc[P].astype(str).values
udom = np.unique(zdom)
def feats_arr(X, k, t):   # u3_content.py, unchanged
    Y = X[:, t].copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1])
    F = np.stack([np.concatenate([np.interp(gr24, np.linspace(0, 1, Y.shape[1]), Y[i, :, v]) for v in range(3)]) for i in range(len(Y))])
    M = np.stack([np.nanmean(F[k == p], 0) for p in P]); return (M - M.mean(0)) / (M.std(0) + 1e-12)
def theme_r(FA, FB, labels):   # r6c_S2_content.py, unchanged
    TA = np.stack([FA[labels == d].mean(0) for d in udom]); TB = np.stack([FB[labels == d].mean(0) for d in udom])
    return float(np.mean([np.corrcoef(TA[i], TB[i])[0, 1] for i in range(len(udom))]))
FE = {}
for m in MODELS:
    g, meta, X, s = D[m]; k = meta.pt.astype(str).str.strip().values; T = X.shape[1]
    for tau, t in (('first', 0), ('middle', T // 2), ('last', T - 1)): FE[(m, tau)] = feats_arr(X, k, t)
for sub in ('attn', 'mlp'):
    X = X_sd(sub); T = X.shape[1]
    for tau, t in (('first', 0), ('middle', T // 2), ('last', T - 1)): FE[(f'SD15{sub}', tau)] = feats_arr(X, upt, t)
P6 = pd.read_csv(f'{RV}/r3_unet/out/P6_rsa.csv').set_index(['a', 'b', 'tau']).rsa
rows = []
for sub in ('attn', 'mlp'):
    for m in MODELS:
        for tau in ('first', 'middle', 'last'):
            rows.append(dict(a=f'SD15{sub}', b=m, tau=tau, rsa=float(P6[(f'SD15{sub}', m, tau)]), theme_r=theme_r(FE[(f'SD15{sub}', tau)], FE[(m, tau)], zdom)))
A2 = pd.DataFrame(rows); A2.to_csv(f'{OUT}/R9_A2_unet_pairs.csv', index=False); print(A2.round(3).to_string())
a2 = {}
for sub in ('attn', 'mlp'):
    for tau in ('first', 'middle', 'last'):
        d = A2[(A2.a == f'SD15{sub}') & (A2.tau == tau)]; a2[f'{sub}_{tau}'] = exact_p(d.rsa.values, d.theme_r.values)
ok = all(a2[f'{s}_last'][0] >= 0.7 for s in ('attn', 'mlp')) and all(a2[f'{s}_middle'][0] >= 0 for s in ('attn', 'mlp'))
RES['R9-A2'] = ('replicates' if ok else 'tested, not supported', a2); print('A2', RES['R9-A2'])
# ---------- A3 three-crossing incidence vs onset ----------
A4 = pd.read_csv(f'{RV}/round6/out/A4_beta_layer_per_step.csv'); E3 = pd.read_csv(f'{RV}/round2/out/E3_plateau_onset.csv').set_index('ch')
cmap = {'SD35M archive attn': 'SD35M-attn', 'SD35L archive attn': 'SD35L-attn', 'ZIT archive attn': 'ZIT-attn', 'ZIT archive mlp': 'ZIT-mlp',
        'ZIB archive (pooled CFG4) attn': 'ZIB-attn', 'MOCHI archive CFG4.5 attn': 'MOCHI-attn'}
rows = []
for ch, e in cmap.items():
    d = A4[A4.channel == ch]; on = float(E3.loc[e, 'onset_sigma']); three = (d.pattern.fillna('') == '+- -+ +-')
    bel = d.sigma < on
    rows.append(dict(channel=ch, onset=on, n_below=int(bel.sum()), n_above=int((~bel).sum()), share3_below=float(three[bel].mean()), share3_above=float(three[~bel].mean())))
A3 = pd.DataFrame(rows); A3['below_gt_above'] = A3.share3_below > A3.share3_above; A3.to_csv(f'{OUT}/R9_A3_three_crossing_vs_onset.csv', index=False); print(A3.round(3).to_string())
n3 = int(A3.below_gt_above.sum()); RES['R9-A3'] = ('plateau-concentrated' if n3 >= 5 else 'tested, not supported', f'{n3}/6 channels')
# ---------- A4 prompt share vs decoding / RSA ----------
E6 = pd.read_csv(f'{RV}/round2/out/E6_varcomp_by_step.csv'); F5 = pd.read_csv(f'{_COSPAFS_ROOT}/papers/mode-I/data/I-F5_semantic_decoding.csv')
F5 = F5[(F5.measure == 'ordered_profile') & F5.realization.isna() & (F5.step == 0)].set_index('channel').acc
CH7 = ['ZIT-attn', 'ZIT-mlp', 'ZIB-attn', 'ZIB-mlp', 'MOCHI-attn', 'SD35M-attn', 'SD35L-attn']
sh0 = [float(E6[(E6.ch == ch) & (E6.step == 0)].prompt_share_anova.iloc[0]) for ch in CH7]; dec = [float(F5[ch]) for ch in CH7]
a4a = loo(sh0, dec)
mid = RS[RS.tau == 'middle']; mr = {m: float(mid[(mid.a == m) | (mid.b == m)].rsa.mean()) for m in MODELS}
shm = {m: float(E6[E6.ch == f'{m}-attn'].prompt_share_anova.mean()) for m in MODELS}
a4b = spearmanr([shm[m] for m in MODELS], [mr[m] for m in MODELS]).statistic
RES['R9-A4'] = ('associated' if (abs(a4a['rho']) >= 0.6 and a4a['loo_ok']) else 'tested, not supported',
                dict(a=a4a, share_first=dict(zip(CH7, np.round(sh0, 3))), decode_first=dict(zip(CH7, np.round(dec, 3))), b_rho_descriptive=float(a4b), b_share=shm, b_mean_rsa_middle=mr))
print('A4', RES['R9-A4'][0], a4a, a4b)
# ---------- A5 anisotropy vs backbone ----------
BB = pd.read_csv(f'{_COSPAFS_ROOT}/papers/mode-I/data/I-F2_backbone.csv').set_index('model').pc1
T3 = pd.read_csv(f'{RV}/r8_correlations/out/R8_3_channel_table.csv').set_index('channel').ratio
mapping = {'ZIT': 'ZIT-attn', 'ZIB': 'ZIB-attn', 'MOCHI': 'MOCHI-attn', 'SD35M': 'SD35M-attn', 'SD35L': 'SD35L-attn', 'SD15attn': 'SD1.5-attn', 'SD15mlp': 'SD1.5-mlp', 'ACEattn': 'ACE-attn', 'ACEmlp': 'ACE-mlp'}
x = [np.log(T3[v]) for v in mapping.values()]; y = [BB[k] for k in mapping]
a5 = loo(x, y); RES['R9-A5'] = ('associated' if (abs(a5['rho']) >= 0.6 and a5['loo_ok']) else 'tested, not supported', dict(a5, pc1=dict(zip(mapping, y)), log_ratio=dict(zip(mapping, np.round(x, 3)))))
print('A5', RES['R9-A5'][0], a5)
# ---------- A6 capture shift vs tangent [descriptive] ----------
Q = json.load(open(f'{RV}/round5/out/Q_results.json')); E10 = pd.read_csv(f'{RV}/round2/out/E10_tangent.csv'); a6 = {}
for sub in ('attn', 'mlp'):
    dz = R5('Z40_1024_s1001', sub); _, _, Xa, _ = ARCH(f'ZIB_{sub}')
    med = lambda X: np.median(np.log(X[..., 0]), 0)
    base = med(dz['X']['cond'])
    sh = {'uncond': abs(float(np.median(med(dz['X']['uncond']) - base))), 'BM': abs(float(np.median(med(BM(dz)) - base))), 'pooled_archive': abs(float(np.median(med(Xa) - base)))}
    tg = {'uncond': Q[f'Q2b_tangent_ZIB4uncond_ZIT_{sub}'], 'BM': Q[f'Q2_ref_tangent_ZIB4BMsynth_ZIT_{sub}'],
          'pooled_archive': float(E10[(E10.a == f'ZIT-{sub}') & (E10.b == f'ZIB-{sub}')].tangent_cos.iloc[0])}
    k = list(sh); a6[sub] = dict(shift=sh, tangent=tg, spearman_shift_vs_1mtangent=float(spearmanr([sh[i] for i in k], [1 - tg[i] for i in k]).statistic))
RES['R9-A6'] = ('descriptive', a6); print('A6', a6)
def conv(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)
json.dump(RES, open(f'{OUT}/R9_A_results.json', 'w'), indent=1, default=conv)
