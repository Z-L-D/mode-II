"""Blocks B and C of PLAN.md (sha 27d056e4): RC38, RC42-RC45, RC47-RC49. Tangent code copied from rc22/rc26/rc33
(depth z-scoring, linear interpolation to n points, nearest matched clock, cosine of first differences; permutation and
phase-randomised surrogate nulls). Usage: python ph_BC.py [rc42 ...]"""
from ph_common import *
from multiprocessing import Pool
from scipy.stats import rankdata
want = set(sys.argv[1:]) or {'rc38', 'rc42', 'rc43', 'rc44', 'rc45', 'rc47', 'rc48', 'rc49'}
def save(rows, name):
    d = pd.DataFrame(rows); d.to_csv(f'{OUT}/{name}.csv', index=False); pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
    print(f'\n== {name}'); print(d.round(4).to_string(index=False), flush=True); return d
def depthnorm(M):
    Y = M.copy(); Y[..., 0] = np.log(np.abs(Y[..., 0]) + 1e-30); Y[..., 1] = np.log(np.abs(Y[..., 1]) + 1e-30)
    return (Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-12)
def cos(a, b): a = a.ravel(); b = b.ravel(); return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
def prof(M, ns, n):
    D = depthnorm(M); u = np.linspace(0, 1, D.shape[1]); G = np.linspace(0, 1, n)
    return np.stack([[np.interp(G, u, D[t, :, k]) for k in range(3)] for t in range(ns)]).transpose(0, 2, 1)
def tan(A, sa, B, sb, seg=None, per=False):
    v = []
    for t in range(len(sa)):
        j = int(np.argmin(np.abs(sb - sa[t]))); a = np.diff(A[t], axis=0); b = np.diff(B[j], axis=0)
        if seg is not None: a, b = a[seg], b[seg]
        v.append(cos(a, b))
    return np.array(v) if per else float(np.mean(v))
def surrogate(B, rng):
    dB = np.diff(B, axis=1); F = np.fft.rfft(dB, axis=1); ph = rng.uniform(0, 2 * np.pi, F.shape[1]); ph[0] = 0
    return np.fft.irfft(F * np.exp(1j * ph)[None, :, None], n=dB.shape[1], axis=1)
def tan_diff(A, sa, dB, sb):
    return float(np.mean([cos(np.diff(A[t], axis=0), dB[int(np.argmin(np.abs(sb - sa[t])))]) for t in range(len(sa))]))
def load(n, sub, rs):
    if rs == 'all': _, X, s = cube(n, sub)
    else: _, X, s = vcube(n, sub, rs)
    return X, s

# ---- RC42 tangent on all rows / centred / top-8 removed, native grid ----
P42 = [('successor', 'SD3M', 'SD35M17', 'attn'), ('same weights, 2 sessions', 'SD35M17', 'SD3SV', 'attn'),
       ('unrelated', 'PIXART', 'COSMOS2B', 'attn'), ('unrelated', 'PIXART', 'COSMOS2B', 'mlp'),
       ('same run', 'PIXART', 'PIXARTU', 'attn'), ('same run', 'PIXART', 'PIXARTU', 'mlp'), ('same run', 'COSMOS2B', 'COSMOS2BU', 'attn'),
       ('same run', 'COSMOS2B', 'COSMOS2BU', 'mlp'), ('same run', 'SD3M', 'SD3MU', 'attn'), ('same run', 'SD35M17', 'SD35M17U', 'attn')]
def j42(p):
    kind, a, b, sub, rs = p; XA, sa = load(a, sub, rs); XB, sb = load(b, sub, rs); MA, MB = np.nanmean(XA, 0), np.nanmean(XB, 0)
    n = min(MA.shape[1], MB.shape[1]); rng = np.random.default_rng(42); A = prof(MA, len(sa), n); B = prof(MB, len(sb), n)
    r = dict(kind=kind, a=a, b=b, sub=sub, rows=rs, n=n, tangent=tan(A, sa, B, sb))
    r['perm95'] = float(np.percentile([tan(A, sa, prof(MB[:, rng.permutation(MB.shape[1])], len(sb), n), sb) for _ in range(1000)], 95))
    r['surr95'] = float(np.percentile([tan_diff(A, sa, surrogate(B, rng), sb) for _ in range(300)], 95))
    r['margin_perm'] = r['tangent'] - r['perm95']; r['margin_surr'] = r['tangent'] - r['surr95']; return r
if 'rc42' in want:
    with Pool(4) as pool: d = save(pool.map(j42, [p + (rs,) for p in P42 for rs in ('all', 'ctr', 'topk')]), 'rc42_tangent_variants')

# ---- RC43 energy diagnostics by layer ----
if 'rc43' in want:
    rows = []
    for n, sub in [('PIXART', 'attn'), ('PIXART', 'mlp'), ('COSMOS2B', 'attn'), ('COSMOS2B', 'mlp'), ('SD3M', 'attn'), ('SD35M17', 'attn'), ('SD3SV', 'attn'), ('Z2', 'attn'), ('Z2', 'mlp'), ('ZITSV', 'attn'), ('ZITSV', 'mlp')]:
        vals = {c: vcube(n, sub, c)[1][..., 0] for c in ('mean_energy_share', 'top1_chan_share', 'n_massive_100x', 'txt_energy_share')}
        L = vals['mean_energy_share'].shape[2]
        for l in range(L):
            rows.append(dict(model=n, sub=sub, layer=l, **{c: float(np.nanmedian(v[:, :, l])) for c, v in vals.items()}))
    d = pd.DataFrame(rows); d.to_csv(f'{OUT}/rc43_energy_by_layer.csv', index=False)
    s = d.groupby(['model', 'sub']).agg(mean_share_med=('mean_energy_share', 'median'), mean_share_min=('mean_energy_share', 'min'), mean_share_max=('mean_energy_share', 'max'),
                                        top1_med=('top1_chan_share', 'median'), top1_max=('top1_chan_share', 'max'), massive_max=('n_massive_100x', 'max')).reset_index()
    last = d.loc[d.groupby(['model', 'sub']).layer.idxmax()][['model', 'sub', 'mean_energy_share', 'top1_chan_share']].rename(columns={'mean_energy_share': 'last_mean_share', 'top1_chan_share': 'last_top1'})
    save(s.merge(last, on=['model', 'sub']).to_dict('records'), 'rc43_energy_summary')

# ---- RC44 identity gate on row sets / variants ----
def step0(X): return np.log(np.abs(X[:, 0, :, 0]) + 1e-30).mean(0)
def gate(x, y): return float(np.corrcoef(x - x.mean(), y - y.mean())[0, 1])
if 'rc44' in want:
    rows = []
    for sub in ('attn', 'mlp'):
        for rs in ('all', 'img', 'txt', 'ctr', 'topk'):
            ka, XA, _ = (cube('Z2', sub) if rs == 'all' else vcube('Z2', sub, rs)); kb, XB, _ = (cube('ZITSV', sub) if rs == 'all' else vcube('ZITSV', sub, rs))
            rng = np.random.default_rng(44); halves = []
            for X, k in ((XA, ka), (XB, kb)):
                up = np.unique(k); hs = []
                for _ in range(50):
                    pick = set(rng.choice(up, len(up) // 2, replace=False)); m = np.array([p in pick for p in k]); hs.append(gate(step0(X[m]), step0(X[~m])))
                halves.append(float(np.min(hs)))
            rows.append(dict(pair='Z2 (Base) vs ZITSV (Turbo)', sub=sub, rows=rs, r=gate(step0(XA), step0(XB)), split_half_min_a=halves[0], split_half_min_b=halves[1]))
    for kind, a, b, sub in [('successor', 'SD3M', 'SD35M17', 'attn'), ('same weights', 'SD35M17', 'SD3SV', 'attn'), ('unrelated', 'PIXART', 'COSMOS2B', 'attn'), ('unrelated', 'PIXART', 'COSMOS2B', 'mlp')]:
        for rs in ('all', 'ctr', 'topk'):
            XA, _ = load(a, sub, rs); XB, _ = load(b, sub, rs); rows.append(dict(pair=f'{a} vs {b} ({kind})', sub=sub, rows=rs, r=gate(step0(XA), step0(XB))))
    save(rows, 'rc44_gate_variants')

# ---- RC45 decoding by row set (Z2, ZITSV) ----
if 'rc45' in want:
    from ph_A_atlas_dec import decode
    dom = domains(); rows = []
    with Pool(4) as pool:
        for n in ('Z2', 'ZITSV'):
            for sub in ('attn', 'mlp'):
                for rs in ('all', 'img', 'txt', 'ctr', 'topk'):
                    pt, X, s = (cube(n, sub) if rs == 'all' else vcube(n, sub, rs)); ok = np.array([p in dom for p in pt])
                    rows.append(dict(rows=rs, **decode(n, sub, X[ok], pt[ok], dom, pool)))
    save(rows, 'rc45_decoding_rowsets')

# ---- RC47 successor by depth half ----
if 'rc47' in want:
    rows = []
    for kind, a, b in [('successor', 'SD3M', 'SD35M17'), ('same weights, 2 sessions', 'SD35M17', 'SD3SV'), ('same run', 'SD3M', 'SD3MU')]:
        XA, sa = load(a, 'attn', 'all'); XB, sb = load(b, 'attn', 'all'); MA, MB = XA.mean(0), XB.mean(0); A = prof(MA, len(sa), 24); B = prof(MB, len(sb), 24)
        rng = np.random.default_rng(47); perms = [rng.permutation(24) for _ in range(1000)]
        for lab, seg in [('all 23 intervals', None), ('intervals 0-11 (blocks 0-12, MMDiT-X self-attention in SD3.5 M)', slice(0, 12)), ('intervals 12-22 (blocks 12-23)', slice(12, 23))]:
            v = tan(A, sa, B, sb, seg); nul = [tan(A, sa, prof(MB[:, q], len(sb), 24), sb, seg) for q in perms]
            rows.append(dict(kind=kind, a=a, b=b, segment=lab, tangent=v, perm95=float(np.percentile(nul, 95))))
    save(rows, 'rc47_successor_depth_halves')

# ---- RC49 per-state tangent, depth-matched pairs ----
P49 = [('same weights', 'Z3', 'ZIT', 'attn'), ('same weights', 'Z3', 'ZIT', 'mlp'), ('sibling', 'Z1', 'ZIT', 'attn'), ('sibling', 'Z1', 'ZIT', 'mlp'),
       ('distillation', 'ACEB', 'ACET', 'attn'), ('distillation', 'ACEB', 'ACET', 'mlp'), ('successor', 'SD3M', 'SD35M17', 'attn'),
       ('same weights', 'SD35M17', 'SD3SV', 'attn'), ('unrelated', 'PIXART', 'COSMOS2B', 'attn'), ('unrelated', 'PIXART', 'COSMOS2B', 'mlp'),
       ('unrelated', 'ACEB', 'SD35M17', 'attn'), ('unrelated', 'ACET', 'SD3M', 'attn')]
def j49(p):
    kind, a, b, sub = p; XA, sa = load(a, sub, 'all'); XB, sb = load(b, sub, 'all'); MA, MB = XA.mean(0), XB.mean(0); n = min(MA.shape[1], MB.shape[1])
    A = prof(MA, len(sa), n); B = prof(MB, len(sb), n); per = tan(A, sa, B, sb, per=True); rng = np.random.default_rng(49)
    nul = np.array([tan(A, sa, prof(MB[:, rng.permutation(MB.shape[1])], len(sb), n), sb, per=True) for _ in range(300)]); p95 = np.percentile(nul, 95, axis=0)
    return dict(kind=kind, a=a, b=b, sub=sub, n=n, states=len(sa), mean=float(per.mean()), min=float(per.min()), max=float(per.max()),
                share_above_p95=float((per > p95).mean()), first=float(per[0]), last=float(per[-1]), per_state=';'.join(f'{x:.3f}:{y:.3f}' for x, y in zip(sa, per)))
if 'rc49' in want:
    with Pool(4) as pool: d = pd.DataFrame(pool.map(j49, P49))
    d.to_csv(f'{OUT}/rc49_tangent_by_state.csv', index=False); pd.set_option('display.width', 250); print(d.drop(columns='per_state').round(3).to_string(index=False))
    rel = d[d.kind != 'unrelated']
    for sub in ('attn', 'mlp'):
        u = d[(d.kind == 'unrelated') & (d['sub'] == sub)]
        umax = max(float(x.split(':')[1]) for s in u.per_state for x in s.split(';'))
        for r in rel[rel['sub'] == sub].itertuples():
            v = np.array([float(x.split(':')[1]) for x in r.per_state.split(';')]); print(f'{sub} {r.a}-{r.b}: share of states above max unrelated per-state value {umax:.3f}: {(v > umax).mean():.2f}')

# ---- RC38 layer-axis crossings on the Round 17 cubes ----
def r38(a):
    n, sub, rs = a; L2 = analyzer(); pts, X, _ = (cube(n, sub) if rs == 'all' else vcube(n, sub, rs)); G, T, L = X.shape[:3]
    gi, t, l = np.meshgrid(np.arange(G) + 1, np.arange(T), np.arange(L), indexing='ij')
    df = pd.DataFrame(dict(gen_idx=gi.ravel(), token_idx=t.ravel(), layer=l.ravel(), g_micro=X[..., 0].ravel()))
    pat = lambda c: ''.join('+' if d == 'pos→neg' else '-' for _, d, _ in c); obs = L2.curve(df); g = {gid: d for gid, d in df.groupby('gen_idx')}; up = np.array(sorted(set(pts))); cnt = []
    for k in range(100):
        rng = np.random.default_rng(3000 + k); parts, m = [], 0
        for p in rng.choice(up, len(up), replace=True):
            for gid in np.where(pts == p)[0] + 1: m += 1; parts.append(g[gid].assign(gen_idx=m))
        cnt.append(len(L2.curve(pd.concat(parts, ignore_index=True))))
    c = pd.Series(cnt)
    return dict(model=n, sub=sub, rows=rs, point_n=len(obs), point_pattern=pat(obs), boot_mode=int(c.mode()[0]), boot_mode_share=float((c == c.mode()[0]).mean()), share3=float((c == 3).mean()), dist=str(c.value_counts().sort_index().to_dict()))
if 'rc38' in want:
    todo = [(n, s, rs) for n, s in [('PIXART', 'attn'), ('PIXART', 'mlp'), ('COSMOS2B', 'attn'), ('COSMOS2B', 'mlp'), ('SD3M', 'attn'), ('SD35M17', 'attn')] for rs in ('all', 'ctr', 'topk')]
    with Pool(4) as pool: save(pool.map(r38, todo), 'rc38_layer_crossings')

# ---- RC48 multiple regression on distance matrices (archive RSA) ----
if 'rc48' in want:
    sys.path.insert(0, f'{ROOT}/review/round6/scripts'); from r6c_common import arch, MODELS, E10_UNREL, kind as kd
    from sklearn.feature_extraction.text import TfidfVectorizer
    gr24 = np.linspace(0, 1, 24); D = {m: arch(m) for m in MODELS}; P = sorted(set.intersection(*[set(strip(D[m][1].pt)) for m in MODELS])); NP = len(P)
    def feats(m, tau):
        g, meta, X, s = D[m]; T = X.shape[1]; t = {'first': 0, 'middle': T // 2, 'last': T - 1}[tau]; k = strip(meta.pt)
        Y = X[:, t].copy(); Y[..., 0] = np.log(Y[..., 0]); Y[..., 1] = np.log(Y[..., 1])
        F = np.stack([np.concatenate([np.interp(gr24, np.linspace(0, 1, Y.shape[1]), Y[i, :, v]) for v in range(3)]) for i in range(len(Y))])
        M = np.stack([F[k == p].mean(0) for p in P]); return (M - M.mean(0)) / (M.std(0) + 1e-12)
    iu = np.triu_indices(NP, 1); V = TfidfVectorizer().fit_transform(P); txt = rankdata((1 - (V @ V.T).toarray())[iu])
    ln = np.array([len(p) for p in P], float); lng = rankdata(np.abs(ln[:, None] - ln[None, :])[iu])
    def coef(y, xb):
        Xm = np.column_stack([np.ones_like(y), xb, txt, lng]); Xm = (Xm - np.r_[0, Xm[:, 1:].mean(0)]) / np.r_[1, Xm[:, 1:].std(0)]
        return np.linalg.lstsq(Xm, (y - y.mean()) / y.std(), rcond=None)[0][1]
    def job(a):
        ma, mb, tau = a; rng = np.random.default_rng(48); A = 1 - np.corrcoef(feats(ma, tau)); B = 1 - np.corrcoef(feats(mb, tau)); y = rankdata(A[iu])
        b0 = coef(y, rankdata(B[iu])); nul = np.array([coef(y, rankdata(B[np.ix_(p, p)][iu])) for p in (rng.permutation(NP) for _ in range(2000))])
        return dict(a=ma, b=mb, kind=kd(ma, mb), tau=tau, beta_B=b0, p=float((1 + (nul >= b0).sum()) / 2001), null_p95=float(np.percentile(nul, 95)))
    with Pool(4) as pool: d = save(pool.map(job, [(a, b, t) for a, b in [('ZIT', 'ZIB')] + E10_UNREL + [('SD35M', 'SD35L')] for t in ('first', 'middle', 'last')]), 'rc48_mrm')
    u = d[d.kind == 'unrelated']; print('unrelated cells with beta>0 and p<=0.01:', int(((u.beta_B > 0) & (u.p <= 0.01)).sum()), 'of', len(u))
