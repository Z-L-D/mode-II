import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
"""R14-B + R12-7: R3 Part B (B1-B10) replayed unchanged on ACE Base cond (PREREG_R14.md, sha dc997f6d).
Copy of review/r3_ace/scripts/a1_score_B.py. Changes, all forced by the data and fixed in PREREG_R14: input = R12-style cubes
(F_ACEB_{attn,mlp}.pkl, keyed by (file, gen_idx, seed), so the two GPU files do not collide); B1 on the two raw CSVs;
B6 data frame built from the cube g values (as r12_8_crossings.py); B9 edge 2 / tol 2 (50 steps, Amendment 1).
Usage: SCR=<cube dir> python r14_B_replay.py"""
import os, sys, glob, json, types, pickle
sys.modules['gradio'] = types.ModuleType('gradio')
import numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE); OUT = f'{BASE}/out'
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/lenses')
import lens_L2_rg as L2
from lens_common import prompt_boot_idx
SCR = os.environ.get('SCR', _COSPAFS_CACHE)
def load(sub):
    _, meta, X, s = pickle.load(open(f'{SCR}/F_ACEB_{sub}.pkl', 'rb'))
    return dict(g_gen=X[..., 0], gap_gen=X[..., 1], ent_gen=X[..., 2], sigma=np.asarray(s, float), prompts=meta.pt.astype(str).str.strip().values)
C = {s: load(s) for s in ('attn', 'mlp')}
D = 2048; N = 2250; lnD = np.log(D); res = {}; det = {}

# B1 gate
raw = pd.concat([pd.read_csv(f).assign(file=os.path.basename(f)) for f in sorted(glob.glob(os.environ.get('ACEB_DATA', f'{BASE}/data') + '/*.csv'))]); raw = raw[raw.branch == 'cond']
cnt = raw.groupby(['file', 'gen_idx', 'token_idx', 'submodule']).layer.nunique()
b1 = bool((cnt == 24).all() and set(raw.submodule) == {'attn', 'mlp'} and (raw.seq_len == N).all() and (raw.hidden_dim == D).all())
res['B1'] = dict(result=f'24 layers x attn/mlp on every step: {bool((cnt == 24).all())}; seq_len {sorted(raw.seq_len.unique())}; D {sorted(raw.hidden_dim.unique())}',
                 verdict='PASS' if b1 else 'FAIL')
assert b1

# B2 anisotropy vs g_MP
rng = np.random.default_rng(0); gs = []
for _ in range(20):
    s = np.linalg.svd(rng.standard_normal((N, D)) / np.sqrt(N), compute_uv=False); gs.append(s.sum() ** 2 / (D * (s ** 2).sum()))
gmp = float(np.mean(gs)); det['g_MP'] = gmp; fr = {}
for s in C:
    ratio = np.median(C[s]['g_gen'], 0) / gmp; fr[s] = float((ratio >= 0.10).mean())
    det[f'B2_{s}'] = dict(frac_cells_ge_0p10=fr[s], median_ratio=float(np.median(ratio)), max_ratio=float(ratio.max()), min_ratio=float(ratio.min()))
res['B2'] = dict(result=f"g_MP {gmp:.4f}; cells >= 0.10: attn {fr['attn']:.3f}, mlp {fr['mlp']:.3f}; median ratio attn {det['B2_attn']['median_ratio']:.4f}, mlp {det['B2_mlp']['median_ratio']:.4f}",
                 verdict='CONFIRMED' if max(fr.values()) <= 0.05 else 'REFUTED')

# B3 triple PCA (pca_shares copied verbatim from lens_L1_triple.py)
def pca_shares(H1, H2, L):
    M = np.column_stack([H1, H2, L]); M = (M - M.mean(0)) / M.std(0)
    ev = np.sort(np.linalg.eigvalsh(np.cov(M.T)))[::-1]; return ev / ev.sum()
p3 = {}
for s in C:
    sh = pca_shares(C[s]['ent_gen'].ravel() / lnD, np.log(C[s]['g_gen'].ravel() * D) / lnD, np.log(C[s]['gap_gen'].ravel()))
    p3[s] = sh; det[f'B3_{s}'] = sh.tolist()
res['B3'] = dict(result=f"PC1+PC2 attn {p3['attn'][:2].sum():.4f}, mlp {p3['mlp'][:2].sum():.4f}", verdict='CONFIRMED' if min(v[:2].sum() for v in p3.values()) >= 0.98 else 'REFUTED')

# B4 R1 autonomy (lens_L2_rg.r1 body, unchanged, on the ACE cache)
def r1(d):
    g = d['g_gen'].astype(float); n, T, L = g.shape; pr = d['prompts']
    x = g[:, :, :-1].ravel(); y = (g[:, :, 1:] - g[:, :, :-1]).ravel()
    lay = np.broadcast_to(np.arange(L - 1), (n, T, L - 1)).ravel(); grp = np.repeat(pr, T * (L - 1))
    ok = np.isfinite(x) & np.isfinite(y); x, y, lay, grp = x[ok], y[ok], lay[ok], grp[ok]
    up = np.unique(pr); rng = np.random.default_rng(1); fold_of = dict(zip(up, rng.permutation(len(up)) % 5))
    fold = np.array([fold_of[p] for p in grp]); pred_g, pred_gl = np.empty_like(y), np.empty_like(y)
    for f in range(5):
        tr, te = fold != f, fold == f
        lo, hi = x[tr].min(), x[tr].max(); kn = np.quantile(x[tr], np.linspace(0, 1, 14)[1:-1])
        Btr, Bte = L2.spline_basis(x[tr], lo, hi, kn), L2.spline_basis(x[te], lo, hi, kn)
        c = np.linalg.lstsq(Btr, y[tr], rcond=None)[0]; pred_g[te] = Bte @ c
        Etr = np.eye(L - 1)[lay[tr]][:, 1:]; Ete = np.eye(L - 1)[lay[te]][:, 1:]
        c2 = np.linalg.lstsq(np.hstack([Btr, Etr]), y[tr], rcond=None)[0]; pred_gl[te] = np.hstack([Bte, Ete]) @ c2
    ss = np.sum((y - y.mean()) ** 2); a = 1 - np.sum((y - pred_g) ** 2) / ss; b = 1 - np.sum((y - pred_gl) ** 2) / ss
    return dict(pairs=len(y), R2_g=a, R2_g_layer=b, dR2=b - a)
r = {s: r1(C[s]) for s in C}; det['B4'] = r
res['B4'] = dict(result=f"dR2 attn {r['attn']['dR2']:.3f} (R2 g {r['attn']['R2_g']:.3f} -> {r['attn']['R2_g_layer']:.3f}), mlp {r['mlp']['dR2']:.3f} (R2 g {r['mlp']['R2_g']:.3f} -> {r['mlp']['R2_g_layer']:.3f})",
                 verdict='CONFIRMED' if min(v['dR2'] for v in r.values()) >= 0.30 else 'REFUTED')

# B5 layer parity
p5 = {}
for s in C:
    lg = np.log(C[s]['g_gen']); d = lg[:, :, 0::2].mean(2) - lg[:, :, 1::2].mean(2)
    pos = float((d > 0).mean()); p5[s] = dict(share_modal=max(pos, 1 - pos), modal_sign='even>odd' if pos >= 0.5 else 'even<odd', median_diff=float(np.median(d)))
det['B5'] = p5; b5 = p5['attn']['share_modal'] >= 0.80
res['B5'] = dict(result=f"attn {p5['attn']['modal_sign']} in {p5['attn']['share_modal']:.3f} of (gen, step) cells (median diff {p5['attn']['median_diff']:.3f}); mlp {p5['mlp']['modal_sign']} {p5['mlp']['share_modal']:.3f} (descriptive)",
                 verdict='CONFIRMED' if b5 else 'REFUTED')

# B6 / B7 layer-axis crossings (analyzer plot_beta_layer unchanged; prompt bootstraps)
Xa = C['attn']['g_gen']; G_, T_, L_ = Xa.shape
gi, tt, ll = np.meshgrid(np.arange(G_) + 1, np.arange(T_), np.arange(L_), indexing='ij')
ATT = pd.DataFrame(dict(gen_idx=gi.ravel(), token_idx=tt.ravel(), layer=ll.ravel(), g_micro=Xa.ravel()))
pt = pd.Series(C['attn']['prompts'], index=np.arange(G_) + 1)
pat = lambda c: ''.join('+' if d == 'pos→neg' else '-' for _, d, _ in c)
def pboot(args):
    df, seed = args; rng = np.random.default_rng(seed); up = np.array(sorted(pt.unique())); pick = rng.choice(up, len(up), replace=True)
    parts, k = [], 0
    for p in pick:
        for gi in pt.index[pt == p]:
            k += 1; parts.append(df[df.gen_idx == gi].assign(gen_idx=k))
    return L2.curve(pd.concat(parts, ignore_index=True))
def run(df, tag):
    obs = L2.curve(df)
    with Pool(8) as pool: bs = pool.map(pboot, [(df, 3000 + k) for k in range(100)])
    counts = pd.Series([len(b) for b in bs]); mode = int(counts.mode()[0]); share = float((counts == mode).mean())
    return obs, bs, counts, mode, share
obs, bs, counts, mode, share = run(ATT, 'raw')
pd.DataFrame(dict(k=range(100), n_cross=counts, pattern=[pat(b) for b in bs])).to_csv(f'{OUT}/r14_B6_boot.csv', index=False)
det['B6'] = dict(point=[(round(g, 4), d, round(y, 4)) for g, d, y in obs], point_pattern=pat(obs), boot_mode=mode, boot_mode_share=share,
                 boot_count_dist=counts.value_counts().sort_index().to_dict(), share3=float((counts == 3).mean()))
b6 = len(obs) == 3 and pat(obs) == '+-+' and mode == 3 and share >= 0.5
# '+-+' = crossings pos->neg, neg->pos, pos->neg, i.e. the + - + - sign sequence
res['B6'] = dict(result=f"point estimate {len(obs)} crossings, pattern {pat(obs) or '(none)'}; bootstrap modal count {mode} in {share:.2f} (3 in {(counts == 3).mean():.2f})",
                 verdict='CONFIRMED' if b6 else 'REFUTED')
if b5:
    par = {}
    for nm, keep in (('even', 0), ('odd', 1)):
        sub = ATT[ATT.layer % 2 == keep].copy(); sub['layer'] = sub.layer // 2
        o = L2.curve(sub); par[nm] = dict(n=len(o), pattern=pat(o), crossings=[(round(g, 4), d, round(y, 4)) for g, d, y in o])
    det['B6_parity_descriptive'] = par
if b6:
    same = [b for b in bs if pat(b) == pat(obs)]; ys = np.array([b[2][2] for b in same]); y3 = obs[2][2]; se = ys.std(ddof=1)
    det['B7'] = dict(y3=y3, se=se, n_same=len(same))
    out = (y3 + 2 * se < -0.80) or (y3 - 2 * se > -0.15)
    res['B7'] = dict(result=f'y3 {y3:.3f} +- {se:.3f} (n same pattern {len(same)}); interval [-0.80, -0.15]', verdict='REFUTED' if out else 'CONFIRMED')
else:
    res['B7'] = dict(result='not run: B6 did not hold', verdict='NOT RUN')

# B8 plateau onset (E3 H3b, unchanged)
on = {}
for s in C:
    X = np.stack([C[s]['g_gen'], C[s]['gap_gen'], C[s]['ent_gen']], -1); sg = C[s]['sigma'].astype(float)
    Y = X.mean(0).copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1]); Am = Y.reshape(len(sg), -1)
    sp = np.linalg.norm(np.diff(Am, axis=0), axis=1) / np.abs(np.diff(sg)); cm = (sg[1:] + sg[:-1]) / 2
    plat = np.median(sp[cm < 0.6]); i = int(np.argmax(sp < 2 * plat)); on[s] = dict(onset_sigma=float(cm[i]), speeds=sp.round(4).tolist(), midpoints=cm.round(4).tolist(),
                                                                                    n_intervals_below_0p6=int((cm < 0.6).sum()), plateau_speed=float(plat))
det['B8'] = on
res['B8'] = dict(result=f"onset sigma attn {on['attn']['onset_sigma']:.3f}, mlp {on['mlp']['onset_sigma']:.3f} (plateau from {on['attn']['n_intervals_below_0p6']} interval)",
                 verdict='CONFIRMED' if all(0.73 <= v['onset_sigma'] <= 0.92 for v in on.values()) else 'REFUTED')

# B9 no susceptibility peak (chi_logit + interior_peak, Amendment 1; copied)
def chi_logit(d, ix):
    g = d['g_gen'][ix]; lg = np.nanmean(np.log(g / (1 - g)), 2); df = pd.DataFrame(lg); df['p'] = d['prompts'][ix]
    return df.groupby('p').var(ddof=1).dropna().mean(0).values
brng = np.random.default_rng(20261008)
def interior_peak(curve_fn, d, edge, tol, NB=100):
    c0 = curve_fn(np.arange(len(d['prompts']))); T = len(c0); j0 = int(np.argmax(c0))
    B = np.array([curve_fn(prompt_boot_idx(d['prompts'], brng)) for _ in range(NB)])
    stab = np.mean(np.abs(B.argmax(1) - j0) <= tol)
    se_first = np.std(B[:, j0] - B[:, 0], ddof=1); se_last = np.std(B[:, j0] - B[:, -1], ddof=1)
    interior = edge <= j0 <= T - 1 - edge
    margin_ok = (c0[j0] - c0[0] > 2 * se_first) and (c0[j0] - c0[-1] > 2 * se_last)
    return c0, j0, stab, interior, margin_ok, interior and stab >= 0.8 and margin_ok
pk = {}
for s in C:
    c0, j0, stab, inter, marg, peak = interior_peak(lambda ix: chi_logit(C[s], ix), C[s], 2, 2)
    pk[s] = dict(curve=c0.round(5).tolist(), argmax=j0, stab=float(stab), interior=bool(inter), margin=bool(marg), peak=bool(peak))
det['B9'] = pk
res['B9'] = dict(result=f"attn argmax step {pk['attn']['argmax']} (interior {pk['attn']['interior']}, stab {pk['attn']['stab']:.2f}, margin {pk['attn']['margin']}); mlp argmax {pk['mlp']['argmax']} (interior {pk['mlp']['interior']}, stab {pk['mlp']['stab']:.2f}, margin {pk['mlp']['margin']})",
                 verdict='REFUTED' if any(v['peak'] for v in pk.values()) else 'CONFIRMED')

# B10 prompt share rises first (E6 ANOVA share, unchanged); g-only sensitivity
def e6(Y, pr):
    up, inv = np.unique(pr, return_inverse=True); k = len(up); cnt = np.bincount(inv); G, T = Y.shape[:2]; Yf = Y.reshape(G, T, -1)
    Yf = (Yf - Yf.mean(0)) / (Yf.std(0) + 1e-12); n0 = (G - (cnt ** 2).sum() / G) / (k - 1); nv, an = [], []
    for t in range(T):
        Z = Yf[:, t]; mu = np.stack([Z[inv == i].mean(0) for i in range(k)])
        ssw = ((Z - mu[inv]) ** 2).sum(0); ssb = (cnt[:, None] * (mu - Z.mean(0)) ** 2).sum(0)
        msw = ssw / (G - k); msb = ssb / (k - 1); vp = np.maximum((msb - msw) / n0, 0)
        nv.append(ssb.sum() / (ssb.sum() + ssw.sum())); an.append(vp.sum() / (vp.sum() + msw.sum()))
    return np.array(nv), np.array(an)
b10 = {}; rows = []
for s in C:
    Y = np.stack([np.log(C[s]['g_gen']), np.log(C[s]['gap_gen']), C[s]['ent_gen']], -1)
    nv, an = e6(Y, C[s]['prompts']); _, ang = e6(np.log(C[s]['g_gen'])[..., None], C[s]['prompts'])
    b10[s] = dict(anova=an.round(4).tolist(), naive=nv.round(4).tolist(), steepest=int(np.diff(an).argmax()), g_only=ang.round(4).tolist(), g_only_steepest=int(np.diff(ang).argmax()))
    for t in range(len(an)): rows.append(dict(ch=f'ACEB-{s}', step=t, sigma=C[s]['sigma'][t], prompt_share_naive=nv[t], prompt_share_anova=an[t], prompt_share_anova_g_only=ang[t]))
pd.DataFrame(rows).to_csv(f'{OUT}/r14_prompt_share_by_step.csv', index=False)
det['B10'] = b10
res['B10'] = dict(result=f"steepest rise interval attn {b10['attn']['steepest']}, mlp {b10['mlp']['steepest']} (g-only {b10['attn']['g_only_steepest']}/{b10['mlp']['g_only_steepest']}); attn {b10['attn']['anova'][0]:.2f}->{max(b10['attn']['anova']):.2f}, mlp {b10['mlp']['anova'][0]:.2f}->{max(b10['mlp']['anova']):.2f}",
                  verdict='CONFIRMED' if all(v['steepest'] == 0 for v in b10.values()) else 'REFUTED')

S = pd.DataFrame([dict(id=k, **v) for k, v in res.items()]); S.to_csv(f'{OUT}/r14_B_scores.csv', index=False)
json.dump(det, open(f'{OUT}/r14_B_details.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 200); print(S.to_string(index=False))

m, sh = det['B6']['boot_mode'], det['B6']['boot_mode_share']
r127 = 'DISTILLATION (3 crossings modal)' if (m == 3 and sh >= 0.5) else 'MODALITY (1 crossing modal)' if (m == 1 and sh >= 0.5) else 'UNRESOLVED'
pd.DataFrame([dict(id='R12-7', test='ACE Base attn layer-axis crossings (frozen decision rule)', verdict=r127,
                   detail=f"point {len(obs)} ({pat(obs)}); modal {m} in {sh:.2f}; dist {det['B6']['boot_count_dist']}")]).to_csv(f'{OUT}/r12_7_verdict.csv', index=False)
print('R12-7:', r127, det['B6'])
