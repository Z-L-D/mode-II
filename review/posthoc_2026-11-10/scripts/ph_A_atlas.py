"""Block A of PLAN.md (sha 27d056e4): atlas results replayed on the Round 17 models. RC34-RC37, RC39-RC41, RC46.
Every estimator is copied unchanged from the script named beside it. Usage: python ph_A_atlas.py [rc34 rc35 ...]"""
from ph_common import *
from multiprocessing import Pool
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
CH = [('PIXART', 'attn'), ('PIXART', 'mlp'), ('COSMOS2B', 'attn'), ('COSMOS2B', 'mlp'), ('SD3M', 'attn'), ('SD35M17', 'attn')]
N_TOK = 4096
want = set(sys.argv[1:]) or {'rc34', 'rc35', 'rc36', 'rc37', 'rc39', 'rc40', 'rc41', 'rc46'}
def save(rows, name):
    d = pd.DataFrame(rows); d.to_csv(f'{OUT}/{name}.csv', index=False); pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
    print(f'\n== {name}'); print(d.round(4).to_string(index=False), flush=True); return d

# ---- RC34 backbone (posthoc_r14.py backbone(), unchanged; null rank mean added) ----
def backbone(X, nnull=999, seed=0):
    Y = np.nanmean(X, 0).copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1])
    Z = (Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-12)
    def stats(Z):
        A = Z.reshape(Z.shape[0], -1); A = A - A.mean(0); ev = np.clip(np.linalg.svd(A, compute_uv=False) ** 2, 0, None); p = ev / ev.sum(); p = p[p > 0]
        return p[0], float(np.exp(-(p * np.log(p)).sum()))
    pc1, er = stats(Z); r = np.random.default_rng(seed); T, L, _ = Z.shape; npc, nrk = [], []
    for _ in range(nnull):
        Zs = np.stack([np.roll(Z[t], r.integers(L), axis=0) for t in range(T)]); a, b = stats(Zs); npc.append(a); nrk.append(b)
    npc, nrk = np.array(npc), np.array(nrk)
    return dict(pc1=pc1, eff_rank=er, null_pc1_mean=npc.mean(), null_pc1_max=npc.max(), null_rank_mean=nrk.mean(), null_rank_min=nrk.min(),
                p_pc1=(1 + (npc >= pc1).sum()) / (nnull + 1), p_rank=(1 + (nrk <= er).sum()) / (nnull + 1))
if 'rc34' in want:
    rows = []
    for n, sub in CH + [('PIXARTU', 'attn'), ('PIXARTU', 'mlp'), ('COSMOS2BU', 'attn'), ('COSMOS2BU', 'mlp'), ('SD3MU', 'attn'), ('SD35M17U', 'attn')]:
        _, X, s = cube(n, sub); b = backbone(X); rows.append(dict(model=n, sub=sub, steps=len(s), **b, rule_pass=bool(b['pc1'] >= 0.70 and b['p_pc1'] <= 0.001)))
    save(rows, 'rc34_backbone')

# ---- RC35 (r14_B_replay.py B2, B3, B4 unchanged) ----
if 'rc35' in want:
    L2 = analyzer(); gmp = {}
    def g_mp(D):
        if D not in gmp:
            rng = np.random.default_rng(0); gs = []
            for _ in range(20):
                s = np.linalg.svd(rng.standard_normal((N_TOK, D)) / np.sqrt(N_TOK), compute_uv=False); gs.append(s.sum() ** 2 / (D * (s ** 2).sum()))
            gmp[D] = float(np.mean(gs))
        return gmp[D]
    def pca_shares(H1, H2, L):
        M = np.column_stack([H1, H2, L]); M = (M - M.mean(0)) / M.std(0); ev = np.sort(np.linalg.eigvalsh(np.cov(M.T)))[::-1]; return ev / ev.sum()
    def r1(g, pr):
        n, T, L = g.shape; x = g[:, :, :-1].ravel(); y = (g[:, :, 1:] - g[:, :, :-1]).ravel()
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
        return a, b
    rows = []
    for n, sub in CH:
        pr, X, s = cube(n, sub); D = DIM[n]; lnD = np.log(D); g, gap, H = X[..., 0], X[..., 1], X[..., 2]
        sh = pca_shares(H.ravel() / lnD, np.log(g.ravel() * D) / lnD, np.log(gap.ravel()))
        ratio = np.median(g, 0) / g_mp(D); a, b = r1(g, pr)
        rows.append(dict(model=n, sub=sub, D=D, g_MP=g_mp(D), pc12=float(sh[:2].sum()), B3_pass=bool(sh[:2].sum() >= 0.98),
                         median_ratio=float(np.median(ratio)), frac_cells_ge_0p10=float((ratio >= 0.10).mean()), B2_pass=bool((ratio >= 0.10).mean() <= 0.05),
                         R2_g=a, R2_g_layer=b, dR2=b - a, B4_pass=bool(b - a >= 0.30)))
    save(rows, 'rc35_triple_anisotropy_autonomy')

# ---- RC36 plateau onset (r14_B_replay.py B8 unchanged; clock as loaded) ----
if 'rc36' in want:
    rows = []
    for n, sub in CH + [('Z1', 'attn'), ('Z1', 'mlp'), ('Z3', 'attn'), ('Z3', 'mlp')]:
        _, X, sg = cube(n, sub); Y = X.mean(0).copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1]); Am = Y.reshape(len(sg), -1)
        sp = np.linalg.norm(np.diff(Am, axis=0), axis=1) / np.abs(np.diff(sg)); cm = (sg[1:] + sg[:-1]) / 2
        r = dict(model=n, sub=sub, clock='t' if n in TCLK else 'sigma', n_intervals_below_0p6=int((cm < 0.6).sum()))
        for f in (1.5, 2.0, 3.0):
            plat = np.median(sp[cm < 0.6]); i = int(np.argmax(sp < f * plat)); r[f'onset_{f}x'] = float(cm[i])
        r['B8_pass'] = bool(0.73 <= r['onset_2.0x'] <= 0.92); rows.append(r)
    save(rows, 'rc36_plateau_onset')

# ---- RC37 clock-axis envelope (posthoc_r14.py fit3 / clock, unchanged; every pattern's share recorded) ----
def fit3(y, s):
    u = -s; T = len(y); best = None
    for i in range(1, T - 2):
        for j in range(i + 1, T - 1):
            A = np.stack([np.ones(T), u, np.maximum(0, u - u[i]), np.maximum(0, u - u[j])], 1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None); sse = ((A @ coef - y) ** 2).sum()
            if best is None or sse < best[0]: best = (sse, i, j, coef)
    sse, i, j, c = best; sl = np.array([c[1], c[1] + c[2], c[1] + c[2] + c[3]])
    return ''.join('+' if v > 0 else '-' for v in sl), i, j, sl
def clock_job(a):
    n, sub = a; k, X, s = cube(n, sub); up = np.unique(k); PP = np.stack([X[k == p][..., 0].mean((0, 2)) for p in up]); env = PP.mean(0)
    top, i, j, sl = fit3(env, s); r = np.random.default_rng(64)
    boot = np.array([fit3(PP[r.integers(0, len(PP), len(PP))].mean(0), s)[0] for _ in range(1000)])
    v, c = np.unique(boot, return_counts=True); o = np.argsort(-c)
    return dict(model=n, sub=sub, clock='t' if n in TCLK else 'sigma', topology=top, break1=float(s[i]), break2=float(s[j]), share_ppm=float((boot == '++-').mean()),
                share_pmp=float((boot == '+-+').mean()), share_ppp=float((boot == '+++').mean()), modal=v[o[0]], modal_share=float(c[o[0]] / 1000),
                env_first=float(env[0]), env_max=float(env.max()), argmax_clock=float(s[env.argmax()]), env_last=float(env[-1]))
if 'rc37' in want:
    with Pool(10) as pool: rows = pool.map(clock_job, CH + [('Z1', 'attn'), ('Z1', 'mlp'), ('Z3', 'attn'), ('Z3', 'mlp')])
    d = save(rows, 'rc37_clock_envelope')

# ---- RC39 / RC46 first-state decoding (u3_content.py P5 code, unchanged) ----
def clf(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
def acc(F, y, grp): return float(cross_val_score(clf(), F, y, groups=grp, cv=GroupKFold(5)).mean())
def scr_one(a):
    F0, y, grp, seed = a; rng = np.random.default_rng(seed); Fs = np.stack([f[rng.permutation(f.shape[0])] for f in F0]).reshape(len(F0), -1); return acc(Fs, y, grp)
def decode(n, sub, X, pt, dom, pool, t=0):
    y = np.asarray([dom[p] for p in pt]); grp = pd.factorize(pt)[0]; pl = np.array([len(p) for p in pt], float)
    F0 = np.log(np.abs(X[:, t]) + 1e-9); real = acc(F0.reshape(len(X), -1), y, grp)
    sc = np.array(pool.map(scr_one, [(F0, y, grp, k) for k in range(99)]))
    txt = TfidfVectorizer(min_df=1).fit_transform(list(pt))
    T = X.shape[1]
    return dict(model=n, sub=sub, gens=len(X), prompts=len(set(pt)), ordered=real, layer_mean=acc(F0.mean(1), y, grp), prompt_length=acc(pl[:, None], y, grp),
                tfidf=float(cross_val_score(LogisticRegression(C=10, max_iter=3000), txt, y, groups=grp, cv=GroupKFold(5)).mean()),
                scramble_max=float(sc.max()), p=float((1 + (sc >= real).sum()) / 100), beats_all=bool(real > sc.max()),
                ordered_mid=acc(np.log(np.abs(X[:, T // 2]) + 1e-9).reshape(len(X), -1), y, grp), ordered_last=acc(np.log(np.abs(X[:, T - 1]) + 1e-9).reshape(len(X), -1), y, grp))
if 'rc39' in want:
    dom = domains(); rows = []
    with Pool(12) as pool:
        for n, sub in CH:
            for br in ('', 'U'):
                pt, X, s = cube(n + br, sub); ok = np.array([p in dom for p in pt]); rows.append(dict(branch='uncond' if br else 'cond', **decode(n, sub, X[ok], pt[ok], dom, pool)))
    save(rows, 'rc39_decoding_r17')
if 'rc46' in want:
    rows = []
    with Pool(12) as pool:
        for n in ('Z1', 'Z1U', 'Z3'):
            for sub in ('attn', 'mlp'):
                g, meta, X, s = pickle.load(open(f'{CA}/F_{n}_{sub}.pkl', 'rb')); pt = strip(meta.pt); dm = dict(zip(pt, meta.dom.astype(str)))
                X = np.asarray(X, float); rows.append(decode(n, sub, X, pt, dm, pool))
    save(rows, 'rc46_decoding_realbase')

# ---- RC40 shared prompt geometry (S2 features and Mantel, R8-12 partial Mantel; 2000 permutations) ----
if 'rc40' in want:
    gr24 = np.linspace(0, 1, 24); NPERM = 2000
    MODS = [('PIXART', 'attn'), ('COSMOS2B', 'attn'), ('SD3M', 'attn'), ('SD35M17', 'attn'), ('SD3SV', 'attn'), ('Z2', 'attn'), ('ZITSV', 'attn'),
            ('PIXART', 'mlp'), ('COSMOS2B', 'mlp'), ('Z2', 'mlp'), ('ZITSV', 'mlp')]
    D = {m: cube(*m) for m in MODS}; P = sorted(set.intersection(*[set(D[m][0]) for m in MODS])); NP = len(P); print('RC40 shared prompts', NP)
    FAMILY = {'PIXART': 'PixArt', 'COSMOS2B': 'Cosmos', 'SD3M': 'SD3', 'SD35M17': 'SD3', 'SD3SV': 'SD3', 'Z2': 'Z-Image', 'ZITSV': 'Z-Image'}
    WEIGHTS = {'SD35M17': 'SD3.5M', 'SD3SV': 'SD3.5M'}
    def feats(m, tau):
        k, X, s = D[m]; T = X.shape[1]; t = {'first': 0, 'middle': T // 2, 'last': T - 1}[tau]
        Y = X[:, t].copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1])
        F = np.stack([np.concatenate([np.interp(gr24, np.linspace(0, 1, Y.shape[1]), Y[i, :, v]) for v in range(3)]) for i in range(len(Y))])
        M = np.stack([F[k == p].mean(0) for p in P]); return (M - M.mean(0)) / (M.std(0) + 1e-12)
    iu = np.triu_indices(NP, 1); TAUS = ['first', 'middle', 'last']
    RD = {(m, tau): 1 - np.corrcoef(feats(m, tau)) for m in MODS for tau in TAUS}
    V = TfidfVectorizer().fit_transform(P); TXT = 1 - (V @ V.T).toarray(); ct = rankdata(TXT[iu])
    ln = np.array([len(p) for p in P], float); cl = rankdata(np.abs(ln[:, None] - ln[None, :])[iu])
    def pcor(a, b, c):
        rab = np.corrcoef(a, b)[0, 1]; rac = np.corrcoef(a, c)[0, 1]; rbc = np.corrcoef(b, c)[0, 1]; return (rab - rac * rbc) / np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))
    def job(a):
        ma, mb, tau = a; rng = np.random.default_rng(40); A = RD[(ma, tau)]; B = RD[(mb, tau)]; x = rankdata(A[iu]); y = rankdata(B[iu])
        r = np.corrcoef(x, y)[0, 1]; pt_, pl_ = pcor(x, y, ct), pcor(x, y, cl); nul = np.empty((NPERM, 3))
        for i in range(NPERM):
            p = rng.permutation(NP); yb = rankdata(B[np.ix_(p, p)][iu]); nul[i] = np.corrcoef(x, yb)[0, 1], pcor(x, yb, ct), pcor(x, yb, cl)
        same_w = WEIGHTS.get(ma[0], ma[0]) == WEIGHTS.get(mb[0], mb[0]); fam = FAMILY[ma[0]] == FAMILY[mb[0]]
        return dict(a=ma[0], b=mb[0], sub_a=ma[1], sub_b=mb[1], kind='same weights' if same_w else ('same family' if fam else 'unrelated'), tau=tau, prompts=NP,
                    rsa=r, p=(1 + (nul[:, 0] >= r).sum()) / (NPERM + 1), null_p95=float(np.percentile(nul[:, 0], 95)),
                    partial_text=pt_, p_text=(1 + (nul[:, 1] >= pt_).sum()) / (NPERM + 1), partial_len=pl_, p_len=(1 + (nul[:, 2] >= pl_).sum()) / (NPERM + 1))
    todo = [(a, b, tau) for i, a in enumerate(MODS) for b in MODS[i + 1:] if a[1] == b[1] and a[0] != b[0] for tau in TAUS]
    with Pool(14) as pool: rows = pool.map(job, todo)
    d = save(rows, 'rc40_prompt_rsa')
    txt = [dict(model=m[0], sub=m[1], tau=tau, rsa_text=float(np.corrcoef(rankdata(RD[(m, tau)][iu]), ct)[0, 1])) for m in MODS for tau in TAUS]
    pd.DataFrame(txt).to_csv(f'{OUT}/rc40_text_rsa.csv', index=False)
    for sub in ('attn', 'mlp'):
        e = d[d.sub_a == sub]; print(f'\n{sub}: rsa by kind'); print(e.groupby('kind').agg(n=('rsa', 'size'), lo=('rsa', 'min'), hi=('rsa', 'max'), pmax=('p', 'max'), ptext_max=('p_text', 'max'), plen_max=('p_len', 'max')).round(4))
        print(e.pivot_table(index=['a', 'b'], columns='tau', values='rsa')[TAUS].round(3).to_string())

# ---- RC41 handoff (score_r15.py R15-1 code, unchanged) ----
if 'rc41' in want:
    def feat(X, t): return np.concatenate([np.log(np.abs(X[:, t, :, 0]) + 1e-12), np.log(np.abs(X[:, t, :, 1]) + 1e-12), X[:, t, :, 2]], 1)
    def pmeans(k, Fm, P):
        M = np.stack([Fm[k == p].mean(0) for p in P]); sd = M.std(0); return (M - M.mean(0)) / (sd + 1e-12), float(sd.max())
    def rsa(A, B):
        iu = np.triu_indices(len(A), 1); RA = 1 - np.corrcoef(A); RB = 1 - np.corrcoef(B)
        if not (np.isfinite(RA).all() and np.isfinite(RB).all()): return np.nan
        return float(np.corrcoef(rankdata(RA[iu]), rankdata(RB[iu]))[0, 1])
    rows, summ = [], []
    for n, sub in CH:
        kc, Xc, s = cube(n, sub); ku, Xu, _ = cube(n + 'U', sub); _, _, sraw = cube(n, sub, raw_clock=True)
        P = sorted(set(kc) & set(ku)); T = Xc.shape[1]; tb = int(np.ceil(0.10 * T)); r = {}
        for t in range(T):
            A, _ = pmeans(kc, feat(Xc, t), P); B, sdmax = pmeans(ku, feat(Xu, t), P); rr = rsa(A, B); r[t] = (rr, sdmax)
            rows.append(dict(model=n, sub=sub, step=t, clock=s[t], sigma_raw=sraw[t], rsa=rr, uncond_sd_max=sdmax, prompts=len(P)))
        a = (r[0][1] < 1e-9) or (np.isfinite(r[0][0]) and r[0][0] <= 0.25); b = r[tb][0] >= 0.50; c = r[T - 1][0] >= 0.80
        summ.append(dict(model=n, sub=sub, steps=T, step0=r[0][0], uncond_sd0=r[0][1], step1=r[1][0], step_10pct=tb, rsa_10pct=r[tb][0], clock_10pct=s[tb], final=r[T - 1][0],
                         max=max(v[0] for v in r.values() if np.isfinite(v[0])), a=bool(a), b=bool(b), c=bool(c), R15_1_pass=bool(a and b and c)))
    pd.DataFrame(rows).to_csv(f'{OUT}/rc41_handoff_by_step.csv', index=False); save(summ, 'rc41_handoff')
