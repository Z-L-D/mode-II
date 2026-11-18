# Review checks requested by the Kimi and GLM reviews (post hoc, 2026-11-05).
# RC5  depth-mismatch control (GLM M1): same model compared with a depth-decimated or depth-resampled copy of itself;
#      unrelated pairs at matched native depth; tier table after low-pass smoothing across depth.
# RC6  tangent asymmetry (GLM m14, Kimi M5): tangent(a,b) against tangent(b,a) for every tier pair.
# RC7  phase analysis for SD3.5 M<->L (GLM Q2): tangent against a fractional depth shift of one profile, per depth third.
# RC8  tier table on image rows (GLM M2, Kimi M1) from the Round 11 row-split cubes, with a same-weights split-half value.
# RC9  energy diagnostics of the caption rows (Kimi minor 5): text-row energy share, top-1 channel share, massive-activation count.
# RC10 identity-gate operating characteristics (Kimi M2, GLM Q7): step-0 depth-profile r on known-same and known-different pairs.
# RC11 per-family separation and the ACE MLP clearance under resampling (GLM M4, Kimi M3).
import os, sys, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R15 = os.path.dirname(HERE); ROOT = os.path.abspath(os.path.join(R15, '..', '..'))
SCR = os.environ.get('SCR', os.path.join(ROOT, 'cache')); OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, 'review', 'round5', 'scripts')); os.environ.setdefault('SCR', SCR)
from r5_common import depthnorm, cos
GR = np.linspace(0, 1, 24)
C = {}
def F(n):
    if n not in C:
        g, m, X, s = pickle.load(open(f'{SCR}/F_{n}.pkl', 'rb')); C[n] = (np.asarray(m.pt.astype(str).str.strip().tolist()), X.astype(float), np.asarray(s, float))
    return C[n]
def V(n, cols):
    o = pickle.load(open(f'{SCR}/V_{n}.pkl', 'rb')); idx = [o['cols'].index(c) for c in cols]
    return np.asarray(o['meta'].pt.astype(str).str.strip().tolist()), o['V'][..., idx].astype(float), np.asarray(o['s'], float), o
def prof(X, s, smooth=0, shift=0.0):
    D = depthnorm(X.mean(0)); L = D.shape[1]; u = np.linspace(0, 1, L) + shift
    if smooth:
        k = np.ones(smooth) / smooth; D = np.stack([[np.convolve(np.pad(D[t, :, j], (smooth // 2, smooth - 1 - smooth // 2), mode='edge'), k, 'valid') for j in range(3)] for t in range(len(s))]).transpose(0, 2, 1)
    return np.stack([[np.interp(GR, u, D[t, :, j]) for j in range(3)] for t in range(len(s))]).transpose(0, 2, 1)
def tangent(XA, sa, XB, sb, smooth=0, shift=0.0, seg=None):
    A = prof(XA, sa, smooth); B = prof(XB, sb, smooth, shift); v = []
    for t, x in enumerate(sa):
        j = int(np.argmin(np.abs(sb - x))); a = np.diff(A[t], axis=0); b = np.diff(B[j], axis=0)
        if seg is not None: a, b = a[seg], b[seg]
        v.append(cos(a, b))
    return float(np.mean(v))
def native(XA, sa, XB, sb):  # matched-depth tangent with no interpolation
    A = depthnorm(XA.mean(0)); B = depthnorm(XB.mean(0))
    return float(np.mean([cos(np.diff(A[t], axis=0), np.diff(B[int(np.argmin(np.abs(sb - x)))], axis=0)) for t, x in enumerate(sa)]))
def halves(P, rng):
    u = np.unique(P); rng.shuffle(u); h = set(u[: len(u) // 2]); m = np.array([p in h for p in P]); return m, ~m
def decimate(X, Lp): return X[:, :, np.unique(np.round(np.linspace(0, X.shape[2] - 1, Lp)).astype(int)), :]
def resample(X, Lp):
    L = X.shape[2]; u = np.linspace(0, 1, L); w = np.linspace(0, 1, Lp); Y = np.log(np.abs(X[..., :2]) + 1e-30)
    Y = np.apply_along_axis(lambda y: np.interp(w, u, y), 2, Y); H = np.apply_along_axis(lambda y: np.interp(w, u, y), 2, X[..., 2:])
    return np.concatenate([np.exp(Y), H], -1)
rng = np.random.default_rng(55)
# ---------------- RC5 ----------------
rows = []
for name, targets in [('MOCHI_attn', (16, 24, 30, 38)), ('SD35L_attn', (16, 24, 30)), ('ZIT_attn', (16, 24)), ('ZIT_mlp', (16, 24)), ('Z1_attn', (16, 24)), ('Z1_mlp', (16, 24)),
                      ('ACET_attn', (16,)), ('ACET_mlp', (16,)), ('SD35M_attn', (16,))]:
    P, X, s = F(name); L = X.shape[2]
    for rep in range(5):
        a, b = halves(P, rng)
        base = tangent(X[a], s, X[b], s)
        for Lp in targets:
            rows.append(dict(model=name, L=L, Lp=Lp, op='decimate', rep=rep, split_half_same_depth=base, tangent=tangent(X[a], s, decimate(X[b], Lp), s)))
        for Lp in sorted(set(targets) | {int(1.6 * L)}):
            Y = resample(resample(X[b], Lp), L) if Lp > L else resample(X[b], Lp)
            rows.append(dict(model=name, L=L, Lp=Lp, op='resample' if Lp < L else f'up{Lp}->back', rep=rep, split_half_same_depth=base, tangent=tangent(X[a], s, Y, s)))
R5d = pd.DataFrame(rows).groupby(['model', 'L', 'Lp', 'op'], as_index=False)[['split_half_same_depth', 'tangent']].mean()
R5d.to_csv(f'{OUT}/rc5_depth_selfcontrol.csv', index=False); print(R5d.round(3).to_string(index=False), flush=True)
# matched-depth unrelated pairs (24 layers: ACE-Step and SD3.5 Medium), native grid and 24-point grid
rows = []
for a, b in [('ACET_attn', 'SD35M_attn'), ('ACEB_attn', 'SD35M_attn'), ('SD35M_attn', 'ACET_attn'), ('SD35M_attn', 'ACEB_attn')]:
    PA, XA, sa = F(a); PB, XB, sb = F(b)
    nb = [tangent(XA, sa, XB[:, :, rng.permutation(24)], sb) for _ in range(300)]
    rows.append(dict(a=a, b=b, depth='24|24', native=native(XA, sa, XB, sb), grid24=tangent(XA, sa, XB, sb), perm_p95=float(np.percentile(nb, 95))))
R5m = pd.DataFrame(rows); R5m.to_csv(f'{OUT}/rc5_matched_depth_unrelated.csv', index=False); print(R5m.round(3).to_string(index=False), flush=True)
# low-pass control: tier pairs with a moving average over 3 or 5 layers before interpolation
PAIRS = [('same', 'ZIT_attn', 'ZIB_attn'), ('same', 'ZIT_mlp', 'ZIB_mlp'), ('same', 'ACEB0_attn', 'ACEB1_attn'), ('same', 'ACEB0_mlp', 'ACEB1_mlp'),
         ('lineage', 'Z1_attn', 'ZIT_attn'), ('lineage', 'Z1_mlp', 'ZIT_mlp'), ('lineage', 'ACEB_attn', 'ACET_attn'), ('lineage', 'ACEB_mlp', 'ACET_mlp'),
         ('unrelated', 'ZIT_attn', 'MOCHI_attn'), ('unrelated', 'ZIT_attn', 'SD35M_attn'), ('unrelated', 'ZIT_attn', 'SD35L_attn'), ('unrelated', 'Z1_attn', 'MOCHI_attn'),
         ('unrelated', 'Z1_attn', 'SD35M_attn'), ('unrelated', 'MOCHI_attn', 'SD35M_attn'), ('unrelated', 'MOCHI_attn', 'SD35L_attn'), ('unrelated', 'SD35M_attn', 'SD35L_attn'),
         ('unrelated', 'ACET_attn', 'ZIT_attn'), ('unrelated', 'ACEB_mlp', 'Z1_mlp'), ('unrelated', 'ACET_attn', 'SD35M_attn')]
rows = []
for k, a, b in PAIRS:
    _, XA, sa = F(a); _, XB, sb = F(b)
    rows.append(dict(kind=k, a=a, b=b, LA=XA.shape[2], LB=XB.shape[2], tangent_ab=tangent(XA, sa, XB, sb), tangent_ba=tangent(XB, sb, XA, sa),
                     smooth3=tangent(XA, sa, XB, sb, 3), smooth5=tangent(XA, sa, XB, sb, 5)))
R6 = pd.DataFrame(rows); R6['asym'] = R6.tangent_ab - R6.tangent_ba
R6.to_csv(f'{OUT}/rc5_rc6_smoothing_asymmetry.csv', index=False); print(R6.round(3).to_string(index=False), flush=True)
# ---------------- RC7 ----------------
rows = []
_, XM, sM = F('SD35M_attn'); _, XL, sL = F('SD35L_attn')
for sh in np.round(np.arange(-0.2, 0.2001, 0.025), 3):
    rows.append(dict(shift=sh, tangent=tangent(XM, sM, XL, sL, shift=sh), first_third=tangent(XM, sM, XL, sL, shift=sh, seg=slice(0, 8)),
                     mid_third=tangent(XM, sM, XL, sL, shift=sh, seg=slice(8, 16)), last_third=tangent(XM, sM, XL, sL, shift=sh, seg=slice(16, 23))))
R7 = pd.DataFrame(rows)
for j, nm in enumerate(['g', 'gap', 'H']):  # per statistic at zero shift
    A = prof(XM, sM); B = prof(XL, sL)
    R7.loc[R7['shift'] == 0, f'stat_{nm}'] = np.mean([cos(np.diff(A[t], axis=0)[:, j], np.diff(B[int(np.argmin(np.abs(sL - x)))], axis=0)[:, j]) for t, x in enumerate(sM)])
R7.to_csv(f'{OUT}/rc7_sd35_phase.csv', index=False); print(R7.round(3).to_string(index=False), flush=True)
# ---------------- RC8 ----------------
rows = []
SETS = {'image rows': ['g_img', 'gap_img', 'H_img'], 'caption rows': ['g_txt', 'gap_txt', 'H_txt'], 'centred': ['g_ctr', 'gap_ctr', 'H_ctr'], 'top-8 removed': ['g_topk', 'gap_topk', 'H_topk']}
def allrows(n):  # all-rows statistics of the same generations, from the F cube of the same capture
    return F(n)
for sub in ('attn', 'mlp'):
    for rs, cols in [('all rows', None)] + list(SETS.items()):
        if cols is None:
            PB, XB, sB = allrows(f'Z2_{sub}'); PT, XT, sT = allrows(f'ZITSV_{sub}'); PS, XS, sS = allrows('SD3SV_attn')
        else:
            PB, XB, sB, _ = V(f'Z2_{sub}', cols); PT, XT, sT, _ = V(f'ZITSV_{sub}', cols)
            PS, XS, sS, _ = V('SD3SV_attn', cols)
        sw = []
        for rep in range(20):
            a, b = halves(PT, rng); sw.append(tangent(XT[a], sT, XT[b], sT))
        swB = []
        for rep in range(20):
            a, b = halves(PB, rng); swB.append(tangent(XB[a], sB, XB[b], sB))
        lin = tangent(XB, sB, XT, sT)
        nb = [tangent(XB, sB, XT[:, :, rng.permutation(30)], sT) for _ in range(300)]
        un = tangent(XT, sT, XS, sS) if sub == 'attn' else np.nan
        un2 = tangent(XB, sB, XS, sS) if sub == 'attn' else np.nan
        rows.append(dict(sub=sub, rows=rs, same_weights_turbo_splithalf=np.mean(sw), same_weights_base_splithalf=np.mean(swB), lineage_base_turbo=lin,
                         perm_p95=float(np.percentile(nb, 95)), unrelated_turbo_sd35m=un, unrelated_base_sd35m=un2))
R8 = pd.DataFrame(rows); R8.to_csv(f'{OUT}/rc8_rowset_tiers.csv', index=False); print(R8.round(3).to_string(index=False), flush=True)
# ---------------- RC9 ----------------
rows = []
for n in ('Z2_attn', 'Z2_mlp', 'ZITSV_attn', 'ZITSV_mlp', 'SD3SV_attn'):
    _, Xd, s, o = V(n, ['txt_energy_share', 'mean_energy_share', 'top1_chan_share', 'n_massive_100x', 'n_txt_rows', 'n_img_rows'])
    m = Xd.mean((0, 1))
    for l in range(m.shape[0]):
        rows.append(dict(capture=n, layer=l, txt_energy_share=m[l, 0], mean_energy_share=m[l, 1], top1_chan_share=m[l, 2], n_massive_100x=m[l, 3], n_txt_rows=m[l, 4], n_img_rows=m[l, 5]))
R9 = pd.DataFrame(rows); R9.to_csv(f'{OUT}/rc9_caption_energy.csv', index=False)
R9['band'] = np.where(R9.layer < 8, 'layers 0-7', 'layers 8+')
print(R9.groupby(['capture', 'band'])[['txt_energy_share', 'mean_energy_share', 'top1_chan_share', 'n_massive_100x', 'n_txt_rows']].mean().round(4).to_string(), flush=True)
# ---------------- RC10 ----------------
def step0(X): return np.log(np.abs(X[:, 0, :, 0]) + 1e-30).mean(0)
def gate(XA, XB):
    a = step0(XA); b = step0(XB)
    return float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1]) if len(a) == len(b) else np.nan
def F3(n):
    g, m, X, s = pickle.load(open(f'{SCR}/F3_{n}.pkl', 'rb')); return X.astype(float)
GATE = [('same', 'Z-Image Turbo: 25-step pooled archive vs 8-step CFG1', 'ZIB', 'ZIT'), ('same', 'Z-Image Turbo: 25-step CFG4 cond vs 8-step CFG1', 'Z3', 'ZIT'),
        ('same', 'Z-Image Turbo: Round 11 capture vs archive', 'ZITSV', 'ZIT'), ('same', 'Z-Image Base: CFG4 cond vs CFG1', 'Z1', 'Z2'),
        ('same', 'Z-Image Base: cond vs uncond branch', 'Z1', 'Z1U'), ('same', 'ACE Base: GPU split 0 vs 1', 'ACEB0', 'ACEB1'), ('same', 'ACE Base: cond vs uncond', 'ACEB', 'ACEBU'),
        ('same', 'Mochi: archive vs Round 5 rerun (CFG, steps, node)', 'MOCHI', 'MOCHI1'), ('same', 'SD3.5 M: archive vs Round 11 CFG1', 'SD35M', 'SD3SV'),
        ('different', 'Z-Image Base vs Turbo (lineage)', 'Z1', 'ZIT'), ('different', 'Z-Image Base CFG1 vs Turbo', 'Z2', 'ZIT'), ('different', 'ACE Base vs Turbo (lineage)', 'ACEB', 'ACET'),
        ('different', 'ACE-Step Turbo vs SD3.5 M (unrelated, both 24 layers)', 'ACET', 'SD35M'), ('different', 'ACE-Step Base vs SD3.5 M (unrelated, both 24 layers)', 'ACEB', 'SD35M')]
rows = []
for kind, label, a, b in GATE:
    for sub in ('attn', 'mlp'):
        try: XA = F(f'{a}_{sub}')[1]; XB = F(f'{b}_{sub}')[1]
        except FileNotFoundError: continue
        rows.append(dict(kind=kind, pair=label, a=a, b=b, sub=sub, LA=XA.shape[2], LB=XB.shape[2], r=gate(XA, XB)))
for label, a, b in [('SD3.5 M: 512 vs 1024 px, CFG4.5', 'S45_512_s1001_attn', 'S45_1024_s1001_attn'), ('SD3.5 M: 768 vs 1024 px', 'S45_768_s1001_attn', 'S45_1024_s1001_attn'),
                    ('SD3.5 M: CFG1 vs CFG4.5', 'S10_1024_s1001_attn', 'S45_1024_s1001_attn'), ('SD3.5 M: seed 1001 vs 1002', 'S45_1024_s1001_attn', 'S45_1024_s1002_attn'),
                    ('Z-Image Turbo: CFG1 vs CFG4 (Round 3)', 'Z10_1024_s1001_attn', 'Z40_1024_s1001_attn'), ('Z-Image Turbo: CFG1 vs CFG4 (Round 3)', 'Z10_1024_s1001_mlp', 'Z40_1024_s1001_mlp')]:
    rows.append(dict(kind='same', pair=label, a=a, b=b, sub=a.split('_')[-1], LA=24, LB=24, r=gate(F3(a), F3(b))))
for gs in (1, 3, 10):  # one-generation and few-generation gate against the full reference
    for a, b, kind in [('Z1', 'ZIT', 'different'), ('Z3', 'ZIT', 'same'), ('ACEB', 'ACET', 'different'), ('ACEB0', 'ACEB1', 'same')]:
        for sub in ('attn', 'mlp'):
            XA = F(f'{a}_{sub}')[1]; XB = F(f'{b}_{sub}')[1]
            v = [gate(XA[rng.choice(len(XA), gs, replace=False)], XB) for _ in range(200)]
            rows.append(dict(kind=kind, pair=f'{a} ({gs} gen) vs {b} reference', a=a, b=b, sub=sub, LA=XA.shape[2], LB=XB.shape[2], r=float(np.median(v)), r_min=float(np.min(v)), r_max=float(np.max(v)), n_gen=gs))
R10 = pd.DataFrame(rows); R10.to_csv(f'{OUT}/rc10_identity_gate.csv', index=False); print(R10.round(4).to_string(index=False), flush=True)
# ---------------- RC11 ----------------
D = pd.read_csv(os.path.join(R15, 'out', 'R15_5_roc_draws.csv'))
ace = D[(D.a == 'ACEB_mlp') & (D.b == 'ACET_mlp')].tangent
pp = pd.read_csv(os.path.join(R15, 'out', 'R15_5_per_pair.csv'))
def fam(n): return 'ACE' if n.startswith('ACE') else 'Z' if n.startswith('Z') else 'other'
pp['family'] = [fam(a) if fam(a) == fam(b) else 'cross' for a, b in zip(pp.a, pp.b)]
fams = []
for f in ('Z', 'ACE'):
    lin = pp[(pp.kind == 'lineage') & (pp.family == f)]['mean']; unr = pp[(pp.kind == 'unrelated') & (pp.a.map(fam).eq(f) | pp.b.map(fam).eq(f))]['mean']
    fams.append(dict(family=f, n_lineage=len(lin), lineage_min=lin.min(), n_unrelated=len(unr), unrelated_max=unr.max(), separated=bool(lin.min() > unr.max())))
R11 = pd.DataFrame(fams); R11['ace_mlp_draws_above_0.542'] = float((ace > 0.542).mean()); R11['ace_mlp_draw_p05'] = float(np.percentile(ace, 5))
R11.to_csv(f'{OUT}/rc11_family_separation.csv', index=False); print(R11.round(3).to_string(index=False))
