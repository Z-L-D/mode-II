# Round 15 scoring (PREREG_R15.md, sha 3acbb027). Usage: SCR=<cube dir> python score_r15.py
import os, sys, json, pickle, numpy as np, pandas as pd
from scipy.stats import rankdata, mannwhitneyu
HERE = os.path.dirname(os.path.abspath(__file__)); R15 = os.path.dirname(HERE); ROOT = os.path.abspath(os.path.join(R15, '..', '..'))
SCR = os.environ.get('SCR', os.path.join(ROOT, 'cache')); OUT = os.path.join(R15, 'out'); LLM = os.path.join(OUT, 'llm_cubes')
sys.path.insert(0, os.path.join(ROOT, 'review', 'round5', 'scripts')); os.environ.setdefault('SCR', SCR)
from r5_common import prof, cos
V, ROWS = [], {}
def verdict(i, ok, detail): V.append(dict(id=i, verdict='SUPPORTED' if ok else 'REFUTED', detail=detail)); print(i, V[-1]['verdict'], detail, flush=True)
def F(n): g, m, X, s = pickle.load(open(f'{SCR}/F_{n}.pkl', 'rb')); return np.asarray(m.pt.astype(str).str.strip().tolist()), X.astype(float), np.asarray(s, float)
def Lc(n): g, m, X, s = pickle.load(open(f'{LLM}/L_{n}.pkl', 'rb')); return np.asarray(m.pt.astype(str).str.strip().tolist()), X.astype(float), np.asarray(s, float)
def R5(arms, sub, br):
    ks, Xs = [], []
    for a in arms:
        d = pickle.load(open(f'{SCR}/R5_{a}_{sub}.pkl', 'rb')); ks.append(np.asarray(d['meta'].pt.astype(str).str.strip().tolist())); Xs.append(d['X'][br].astype(float)); s = np.asarray(d['sigma'], float)
    return np.concatenate(ks), np.concatenate(Xs), s
def feat(X, t): return np.concatenate([np.log(np.abs(X[:, t, :, 0]) + 1e-12), np.log(np.abs(X[:, t, :, 1]) + 1e-12), X[:, t, :, 2]], 1)
def pmeans(k, Fm, P):
    M = np.stack([Fm[k == p].mean(0) for p in P]); sd = M.std(0); return (M - M.mean(0)) / (sd + 1e-12), float(sd.max())
def rsa(A, B, nperm=0, seed=15):
    iu = np.triu_indices(len(A), 1); RA = 1 - np.corrcoef(A); RB = 1 - np.corrcoef(B)
    if not (np.isfinite(RA).all() and np.isfinite(RB).all()): return np.nan, np.nan
    a = rankdata(RA[iu]); b = rankdata(RB[iu]); r = float(np.corrcoef(a, b)[0, 1])
    if not nperm: return r, np.nan
    rng = np.random.default_rng(seed); n = len(A); nul = np.array([np.corrcoef(a, rankdata(RB[np.ix_(p, p)][iu]))[0, 1] for p in (rng.permutation(n) for _ in range(nperm))])
    return r, float((1 + (nul >= r).sum()) / (nperm + 1))
def tangent(XA, sa, XB, sb, hi=0):
    A = prof(XA, sa); B = prof(XB, sb); tc = []
    for t, x in enumerate(sa):
        j = int(np.argmin(np.abs(sb - x))); tc.append(cos(np.diff(A[t], axis=0)[:23 - hi], np.diff(B[j], axis=0)[:23 - hi]))
    return float(np.mean(tc))
def blockperm(L, rng):
    p = np.arange(L)
    for b in np.array_split(np.arange(L), 4): p[b] = rng.permutation(b)
    return p
# ---------------- R15-1 ----------------
uc = np.load(os.path.join(ROOT, 'review', 'r3_unet', 'out', 'u0_cache.npz'), allow_pickle=True)
def unet(sub, br): return np.asarray([str(x).strip() for x in uc['pt']]), np.stack([uc[f'{sub}_{br}_{v}'] for v in ('g_micro', 'spectral_gap', 'sv_entropy')], -1).astype(float), np.asarray(uc['sigma'], float)
CH = {'SD3.5M CFG4.5 attn': lambda br: R5(['S45_1024_s1001', 'S45_1024_s1002'], 'attn', br),
      'ZIT-25 CFG4 attn': lambda br: R5(['Z40_1024_s1001'], 'attn', br), 'ZIT-25 CFG4 mlp': lambda br: R5(['Z40_1024_s1001'], 'mlp', br),
      'SD1.5 U-Net attn': lambda br: unet('attn', br), 'SD1.5 U-Net mlp': lambda br: unet('mlp', br)}
rows = []; npass = 0
for ch, f in CH.items():
    kc, Xc, s = f('cond'); ku, Xu, _ = f('uncond')
    ok = np.isfinite(Xc).all(axis=(1, 2, 3)) & np.isfinite(Xu).all(axis=(1, 2, 3)); kc, Xc, ku, Xu = kc[ok], Xc[ok], ku[ok], Xu[ok]
    P = sorted(set(kc) & set(ku)); T = Xc.shape[1]; tb = int(np.ceil(0.10 * T)); r = {}
    for t in range(T):
        A, _ = pmeans(kc, feat(Xc, t), P); B, sdmax = pmeans(ku, feat(Xu, t), P); rr, _ = rsa(A, B)
        r[t] = (rr, sdmax); rows.append(dict(channel=ch, step=t, sigma=s[t], rsa=rr, uncond_sd_max=sdmax, prompts=len(P)))
    a = (r[0][1] < 1e-9) or (np.isfinite(r[0][0]) and r[0][0] <= 0.25); b = r[tb][0] >= 0.50; c = r[T - 1][0] >= 0.80
    npass += a and b and c
    ROWS.setdefault('R15-1', []).append(f"{ch}: step0 {r[0][0]:.3f} (uncond sd {r[0][1]:.2g}), step {tb} {r[tb][0]:.3f}, final {r[T-1][0]:.3f} -> {'pass' if a and b and c else 'fail'}")
pd.DataFrame(rows).to_csv(f'{OUT}/R15_1_handoff.csv', index=False)
verdict('R15-1', npass >= 4, f'{npass}/5 channels; ' + '; '.join(ROWS['R15-1']))
# ---------------- R15-2 ----------------
def sshare(k, Fm, P):
    Fz = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-12); w = [Fz[k == p].var(0, ddof=1).mean() for p in P if (k == p).sum() >= 2]
    return float(np.mean(w) / Fz.var(0, ddof=1).mean())
k45, X45, s = R5(['S45_1024_s1001', 'S45_1024_s1002'], 'attn', 'cond'); k10, X10, _ = R5(['S10_1024_s1001', 'S10_1024_s1002'], 'attn', 'cond')
P = sorted(p for p in set(k45) & set(k10) if (k45 == p).sum() >= 2 and (k10 == p).sum() >= 2)
rows = [dict(step=t, sigma=s[t], share_cfg45=sshare(k45, feat(X45, t), P), share_cfg1=sshare(k10, feat(X10, t), P)) for t in range(X45.shape[1])]
D2 = pd.DataFrame(rows); D2.to_csv(f'{OUT}/R15_2_cfg_seed_share.csv', index=False); frac = float((D2.share_cfg45 <= D2.share_cfg1 + 0.02).mean())
verdict('R15-2', frac >= 0.75, f'{len(P)} prompts; CFG4.5 <= CFG1 + 0.02 at {frac:.0%} of steps; mean share CFG4.5 {D2.share_cfg45.mean():.3f}, CFG1 {D2.share_cfg1.mean():.3f}')
# ---------------- R15-3 ----------------
rows = []; okall = True
for sub in ('attn', 'mlp'):
    kb, Xb, sb = F(f'ACEB_{sub}'); kt, Xt, st = F(f'ACET_{sub}'); P = sorted(set(kb) & set(kt)); hits = 0
    for lab, tb_ in [('first', 0), ('sigma0.75', int(np.argmin(np.abs(sb - 0.75)))), ('last', len(sb) - 1)]:
        tt = int(np.argmin(np.abs(st - sb[tb_]))); A, _ = pmeans(kb, feat(Xb, tb_), P); B, _ = pmeans(kt, feat(Xt, tt), P); r, p = rsa(A, B, 999)
        rows.append(dict(sub=sub, state=lab, sigma_base=sb[tb_], sigma_turbo=st[tt], rsa=r, p=p, prompts=len(P))); hits += (r >= 0.30 and p < 0.01)
    okall &= hits >= 2
D3 = pd.DataFrame(rows); D3.to_csv(f'{OUT}/R15_3_ace_rsa.csv', index=False)
verdict('R15-3', okall, '; '.join(f'{r.sub} {r.state} {r.rsa:.3f} (p {r.p:.3f})' for r in D3.itertuples()))
# ---------------- R15-4 ----------------
SAME = [('ZIT_attn', 'ZIB_attn'), ('ZIT_mlp', 'ZIB_mlp')]; LIN = [('Z1_attn', 'ZIT_attn'), ('Z1_mlp', 'ZIT_mlp'), ('ACEB_attn', 'ACET_attn'), ('ACEB_mlp', 'ACET_mlp')]
ARCHU = [('ZIT', 'MOCHI'), ('ZIT', 'SD35M'), ('ZIT', 'SD35L'), ('ZIB', 'MOCHI'), ('ZIB', 'SD35M'), ('ZIB', 'SD35L'), ('MOCHI', 'SD35M'), ('MOCHI', 'SD35L')]
UNREL = [(a + '_attn', b + '_attn') for a, b in ARCHU] + [('Z1_attn', 'SD35M_attn'), ('Z1_attn', 'SD35L_attn'), ('Z1_attn', 'MOCHI_attn'),
         ('ACET_attn', 'ZIT_attn'), ('ACET_mlp', 'ZIT_mlp'), ('ACEB_attn', 'Z1_attn'), ('ACEB_mlp', 'Z1_mlp')]
C = {}
def cube(n):
    if n not in C: C[n] = F(n)
    return C[n]
rows = []
for kind, pairs in [('same', SAME), ('lineage', LIN), ('unrelated', UNREL)]:
    for a, b in pairs:
        _, XA, sa = cube(a); _, XB, sb = cube(b); t = tangent(XA, sa, XB, sb); rng = np.random.default_rng(15); L = XB.shape[2]
        nb = [tangent(XA, sa, XB[:, :, blockperm(L, rng)], sb) for _ in range(1000)]
        rows.append(dict(kind=kind, a=a, b=b, tangent=t, block_p95=float(np.percentile(nb, 95)))); print(rows[-1], flush=True)
D4 = pd.DataFrame(rows); D4['margin'] = D4.tangent - D4.block_p95; D4.to_csv(f'{OUT}/R15_4_block_null.csv', index=False)
A_ = D4[(D4.kind == 'same') | ((D4.kind == 'lineage') & (D4.a != 'ACEB_mlp'))]; a_ok = bool((A_.margin >= 0.20).all())
b_ok = bool((D4[D4.a == 'ACEB_mlp'].margin < 0.20).all()); c_ok = bool((D4[D4.kind == 'unrelated'].margin <= 0.10).all())
verdict('R15-4', a_ok and b_ok and c_ok, f"(a) {a_ok}: min margin {A_.margin.min():.3f}; (b) {b_ok}: ACE mlp margin {D4[D4.a=='ACEB_mlp'].margin.iloc[0]:.3f}; (c) {c_ok}: max unrelated margin {D4[D4.kind=='unrelated'].margin.max():.3f}")
# ---------------- R15-5 ----------------
SAME5 = [('ZIT_attn', 'ZIB_attn'), ('Z3_attn', 'ZIB_attn'), ('ACEB0_attn', 'ACEB1_attn'), ('ZIT_mlp', 'ZIB_mlp'), ('Z3_mlp', 'ZIB_mlp'), ('ACEB0_mlp', 'ACEB1_mlp')]
LIN5 = [('Z1_attn', 'ZIT_attn'), ('Z1_attn', 'Z3_attn'), ('ACEB_attn', 'ACET_attn'), ('Z1_mlp', 'ZIT_mlp'), ('Z1_mlp', 'Z3_mlp'), ('ACEB_mlp', 'ACET_mlp')]
UNREL5 = [p for p in UNREL if p[0].endswith('attn')] + [('ACET_mlp', 'ZIT_mlp'), ('ACEB_mlp', 'Z1_mlp'), ('ACET_mlp', 'Z1_mlp'), ('ACEB_mlp', 'ZIT_mlp')]
rng = np.random.default_rng(15); rows = []
for kind, pairs in [('same', SAME5), ('lineage', LIN5), ('unrelated', UNREL5)]:
    for a, b in pairs:
        _, XA, sa = cube(a); _, XB, sb = cube(b)
        for d in range(200):
            ia = rng.choice(len(XA), min(30, len(XA)), replace=False); ib = rng.choice(len(XB), min(30, len(XB)), replace=False)
            rows.append(dict(kind=kind, sub=a.split('_')[1], a=a, b=b, draw=d, tangent=tangent(XA[ia], sa, XB[ib], sb)))
D5 = pd.DataFrame(rows); D5.to_csv(f'{OUT}/R15_5_roc_draws.csv', index=False)
def auc(pos, neg): return float(mannwhitneyu(pos, neg, alternative='greater').statistic / (len(pos) * len(neg)))
g = lambda k, s: D5[(D5.kind == k) & (D5['sub'] == s)].tangent.values
a5 = auc(g('lineage', 'attn'), g('unrelated', 'attn')); b5 = auc(g('same', 'attn'), g('lineage', 'attn')); c5 = auc(g('lineage', 'mlp'), g('unrelated', 'mlp'))
per = D5.groupby(['kind', 'a', 'b']).tangent.describe(percentiles=[.05, .95])[['mean', '5%', '95%']]; per.to_csv(f'{OUT}/R15_5_per_pair.csv')
verdict('R15-5', a5 >= 0.95 and b5 >= 0.95 and c5 >= 0.90, f'attn AUC lineage>unrelated {a5:.3f}; attn AUC same>lineage {b5:.3f}; mlp AUC lineage>unrelated {c5:.3f}')
# ---------------- R15-6 ----------------
rows = []
for sub in ('attn', 'mlp'):
    for a, b in [('Q35', 'Q25'), ('Q35', 'G3'), ('Q25', 'G3')]:
        _, XA, sa = Lc(f'{a}_{sub}'); _, XB, sb = Lc(f'{b}_{sub}'); rng = np.random.default_rng(15)
        for hi in (0, 1):
            t = tangent(XA, sa, XB, sb, hi); nul = [tangent(XA, sa, XB[:, :, rng.permutation(XB.shape[2])], sb, hi) for _ in range(1000)]
            rows.append(dict(sub=sub, a=a, b=b, drop_last=hi, tangent=t, null_p95=float(np.percentile(nul, 95))))
D6 = pd.DataFrame(rows); D6.to_csv(f'{OUT}/R15_6_llm_tangents.csv', index=False)
a6 = bool((D6[(D6['sub'] == 'attn') & (D6.drop_last == 0)].eval('tangent < null_p95 + 0.30')).all())
b6 = bool((D6[(D6['sub'] == 'mlp') & (D6.drop_last == 1)].eval('tangent < null_p95 + 0.30')).all())
verdict('R15-6', a6 and b6, '; '.join(f'{r.sub} {r.a}-{r.b}{" (drop last)" if r.drop_last else ""} {r.tangent:.3f} (p95 {r.null_p95:.3f})' for r in D6.itertuples()))
# ---------------- R15-7 ----------------
kz, Xz, _ = F('Z1_attn'); rows = []
for m in ('Q25', 'Q35'):
    for sub in ('attn', 'mlp'):
        kl, Xl, _ = Lc(f'{m}_{sub}'); P = sorted(set(kz) & set(kl)); A, _ = pmeans(kl, feat(Xl, 0), P); B, _ = pmeans(kz, feat(Xz, 0), P); r, p = rsa(A, B, 999)
        rows.append(dict(llm=m, sub=sub, prompts=len(P), rsa=r, p=p))
D7 = pd.DataFrame(rows); D7.to_csv(f'{OUT}/R15_7_llm_dit_rsa.csv', index=False)
verdict('R15-7', bool(((D7.rsa > 0) & (D7.p < 0.01)).any()), '; '.join(f'{r.llm} {r.sub} {r.rsa:.3f} (p {r.p:.3f}, {r.prompts} prompts)' for r in D7.itertuples()))
pd.DataFrame(V).to_csv(f'{OUT}/R15_verdicts.csv', index=False)
