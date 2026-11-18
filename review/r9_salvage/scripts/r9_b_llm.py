"""R9 Part B (LLM salvage), as frozen in R9_PREREGISTRATION.md (SHA 09c1f170, 2026-10-09T09:07:54Z).
Statistics reused unchanged: E7 derivative r (round2/scripts/e7_llm_context.py), E10 tangent cos with depthnorm (round2 common.py),
E8 prefill pairing, S2/R8-12 RDM + partial Mantel. No PSTF/pstf_llama data. Output: out/R9_B_*.csv, out/R9_B_results.json"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import sys, os, glob, json, numpy as np, pandas as pd
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round2/scripts')
from common import depthnorm
from scipy.stats import spearmanr, rankdata
SCR = os.environ.get('SCR', f'{_COSPAFS_CACHE}'); PQ = f'{SCR}/pq'
OUT = f'{_COSPAFS_ROOT}/review/r9_salvage/out'; PSEED = 2026100908; NPERM = 10000; RES = {}
grid = np.linspace(0, 1, 24)
def load(name):
    d = pd.read_parquet(glob.glob(f'{PQ}/LLM_{name}_*.parquet')[0])
    if 'submodule' in d: d = d[d.submodule == 'attn']
    d['prompt_text'] = d.groupby('gen_idx').prompt_text.transform('first'); return d
QN = ['qwen25_1_5b', 'qwen25_3b', 'qwen25_7b']; Q = {k: load(k) for k in QN}
MI, MA, LI = load('mistral-03-7B-instruct'), load('mistral-03-7B-instruct-ablit'), load('llama-31-8B-instruct')
def conv(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
def dump(): json.dump(RES, open(f'{OUT}/R9_B_results.json', 'w'), indent=1, default=conv)
# ======================= B1 nested i.i.d. capture =======================
rng = np.random.default_rng(PSEED)
n0s = Q['qwen25_1_5b'][Q['qwen25_1_5b'].token_type == 'prefill'].groupby('gen_idx').seq_len.first().values
def stats_from_gram(Xn):
    n = Xn.shape[0]; w = np.clip(np.linalg.eigvalsh(Xn @ Xn.T / n), 0, None); s = np.sqrt(w); D = Xn.shape[1]
    g = s.sum() ** 2 / (D * (s ** 2).sum()); p = s / s.sum(); p = p[p > 0]; H = float(-(p * np.log(p)).sum())
    return g, H
rows = []
for D in (1536, 4096):
    lam_pl = 1.0 / np.arange(1, D + 1)
    for pop in ('isotropic', 'powerlaw', 'powerlaw+spike'):
        lam = np.ones(D) if pop == 'isotropic' else lam_pl / lam_pl.mean()
        mu = np.zeros(D)
        if pop == 'powerlaw+spike':
            u = rng.standard_normal(D); u /= np.linalg.norm(u); mu = u * np.sqrt(0.25 * lam.sum())   # spike = 20% of uncentred energy
        for gsim in range(30):
            n0 = int(rng.choice(n0s)); N = n0 + 255
            X = rng.standard_normal((N, D)) * np.sqrt(lam) + mu
            ns = np.arange(n0, N + 1); gs = []; Hs = []
            for n in ns:
                g, H = stats_from_gram(X[:n]); gs.append(g); Hs.append(H)
            gs = np.array(gs); rows.append(dict(D=D, pop=pop, sim=gsim, n0=n0, rho_g=spearmanr(ns, gs).statistic, rho_PRn=spearmanr(ns, gs * D / ns).statistic,
                                                rho_H=spearmanr(ns, Hs).statistic, g_over_ceiling_median=float(np.median(gs * D / ns))))
        print(D, pop, pd.DataFrame(rows).query('D==@D and pop==@pop')[['rho_g', 'rho_PRn', 'rho_H', 'g_over_ceiling_median']].median().round(4).to_dict(), flush=True)
B1 = pd.DataFrame(rows); B1.to_csv(f'{OUT}/R9_B1_nested_iid.csv', index=False)
summ = B1.groupby(['D', 'pop'])[['rho_g', 'rho_PRn', 'rho_H', 'g_over_ceiling_median']].median()
ok = bool(((summ.rho_g.abs() >= 0.99) & (summ.rho_PRn.abs() >= 0.99)).all())
RES['R9-B1'] = ('the retirement statistic is model-free' if ok else 'tested, not supported', summ.round(4).reset_index().to_dict('records')); dump()
# ======================= B2 Qwen tangent triple with uncertainty =======================
G = {k: v[v.token_type == 'generated'] for k, v in Q.items()}
lo = max(d.seq_len.quantile(.05) for d in G.values()); hi = min(d.seq_len.quantile(.95) for d in G.values()); bins = np.linspace(lo, hi, 9)
prompts = sorted(set(G['qwen25_1_5b'].prompt_text))
def sums(d):
    d = d[(d.seq_len >= bins[0]) & (d.seq_len < bins[-1])].copy(); d['bin'] = np.digitize(d.seq_len, bins) - 1
    L = d.layer.nunique(); S = np.zeros((len(prompts), 8, L)); Nn = np.zeros_like(S); pi = {p: i for i, p in enumerate(prompts)}
    a = d.groupby(['prompt_text', 'bin', 'layer']).g_micro.agg(['sum', 'count']).reset_index()
    S[a.prompt_text.map(pi), a.bin, a.layer] = a['sum']; Nn[a.prompt_text.map(pi), a.bin, a.layer] = a['count']; return S, Nn
SN = {k: sums(G[k]) for k in QN}
def deriv(S, Nn, w, perm=None):
    P = (w[:, None, None] * S).sum(0) / (w[:, None, None] * Nn).sum(0)    # [bin, L] mean g
    if perm is not None: P = P[:, perm]
    y = np.log(P); u = np.linspace(0, 1, y.shape[1]); return np.array([np.diff(np.interp(grid, u, yy)) for yy in y])
def stat(a, b, w, perm=None):
    A = deriv(*SN[a], w); B = deriv(*SN[b], w, perm); return float(np.median([np.corrcoef(A[i], B[i])[0, 1] for i in range(8)])), [float(np.corrcoef(A[i], B[i])[0, 1]) for i in range(8)]
pairs = [('qwen25_1_5b', 'qwen25_7b'), ('qwen25_3b', 'qwen25_7b'), ('qwen25_1_5b', 'qwen25_3b')]
w1 = np.ones(len(prompts)); rb = np.random.default_rng(64); rp = np.random.default_rng(PSEED); b2 = {}; rows = []
for a, b in pairs:
    est, perbin = stat(a, b, w1)
    boots = []
    for _ in range(1000):
        w = np.bincount(rb.integers(0, len(prompts), len(prompts)), minlength=len(prompts)).astype(float); boots.append(stat(a, b, w)[0])
    Lb = SN[b][0].shape[2]; nul = [stat(a, b, w1, rp.permutation(Lb))[0] for _ in range(1000)]
    b2[f'{a}~{b}'] = dict(r=est, per_bin=perbin, boot_p5=float(np.percentile(boots, 5)), boot_p95=float(np.percentile(boots, 95)),
                          null_p95=float(np.percentile(nul, 95)), above_null=bool(np.percentile(boots, 5) > np.percentile(nul, 95)),
                          r_vs_bin_spearman=float(spearmanr(np.arange(8), perbin).statistic))
    print('B2', a, b, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in b2[f'{a}~{b}'].items() if k != 'per_bin'}, flush=True)
k13 = 'qwen25_1_5b~qwen25_3b'; sep = b2[k13]['boot_p5'] > max(b2['qwen25_1_5b~qwen25_7b']['boot_p95'], b2['qwen25_3b~qwen25_7b']['boot_p95'])
RES['R9-B2'] = dict(pairs=b2, verdict_1_5b_3b=('above null' if b2[k13]['above_null'] else 'tested, not supported'), separated=bool(sep), bins=bins.tolist())
RES['R9-B7'] = ('descriptive', {k: v['r_vs_bin_spearman'] for k, v in b2.items()}); dump()
# ======================= B3 Mistral ablation tangent at prefill =======================
pa = MI[MI.token_type == 'prefill']; pb = MA[MA.token_type == 'prefill']
ta = pa.groupby('gen_idx').prompt_text.first(); tb = pb.groupby('gen_idx').prompt_text.first()
same = [g for g in ta.index if g in tb.index and ta[g] == tb[g]]
def cube(p, gens):
    return np.stack([p[p.gen_idx == g].sort_values('layer')[['g_micro', 'spectral_gap', 'sv_entropy']].values for g in gens])   # [G, L, 3]
CA = cube(pa, same); CB = cube(pb, same)
def tang(A, B):
    DA = depthnorm(A.mean(0)[None])[0]; DB = depthnorm(B.mean(0)[None])[0]; u = np.linspace(0, 1, DA.shape[0])
    ia = np.stack([np.interp(grid, u, DA[:, k]) for k in range(3)], 1); ib = np.stack([np.interp(grid, u, DB[:, k]) for k in range(3)], 1)
    x = np.diff(ia, axis=0).ravel(); y = np.diff(ib, axis=0).ravel(); cosv = x @ y / np.linalg.norm(x) / np.linalg.norm(y)
    la = np.diff(np.interp(grid, u, np.log(A[..., 0].mean(0)))); lb = np.diff(np.interp(grid, u, np.log(B[..., 0].mean(0))))
    return float(cosv), float(np.corrcoef(la, lb)[0, 1])
c0, r0 = tang(CA, CB); bt = []
for _ in range(1000):
    i = rb.integers(0, len(same), len(same)); bt.append(tang(CA[i], CB[i]))
bt = np.array(bt)
RES['R9-B3'] = ('the edit preserves the tangent' if (c0 >= 0.9 and np.percentile(bt[:, 0], 5) >= 0.9) else 'tested, not supported',
                dict(n_prompts=len(same), tangent_cos=c0, cos_p5=float(np.percentile(bt[:, 0], 5)), cos_p95=float(np.percentile(bt[:, 0], 95)), deriv_r_logg=r0, r_p5=float(np.percentile(bt[:, 1], 5))))
print('B3', RES['R9-B3'], flush=True); dump()
# ======================= B4/B5 divergence and localisation along the shared prefix =======================
ga = MI[MI.token_type == 'generated']; gb = MA[MA.token_type == 'generated']
E8p = pd.read_csv(f'{_COSPAFS_ROOT}/review/round2/out/E8_ablit_prefill_by_prompt.csv').set_index('gen_idx').mean_abs_dlogg
div = []; loc = []
A0 = ga.pivot_table(index=['gen_idx', 'token_idx'], columns='layer', values='g_micro'); B0 = gb.pivot_table(index=['gen_idx', 'token_idx'], columns='layer', values='g_micro')
for g in same:
    a = A0.loc[g]; b = B0.loc[g]; idx = a.index.intersection(b.index); a = a.loc[idx]; b = b.loc[idx]
    diff0 = (a[0].values != b[0].values); d_tok = int(idx[np.argmax(diff0)]) if diff0.any() else int(idx.max()) + 1
    div.append(dict(gen_idx=g, divergence_token=d_tok, censored=not diff0.any(), prefill_mean_abs_dlogg=float(E8p.get(g, np.nan)), gen_len_instruct=int(ga[ga.gen_idx == g].gen_len.iloc[0])))
    sh = a[a.index < d_tok]; shb = b[b.index < d_tok]
    for t in sh.index:
        dl = np.abs(np.log(shb.loc[t].values / sh.loc[t].values))
        loc.append(dict(gen_idx=g, token_idx=int(t), first6_identical=bool((dl[:6] == 0).all()), argmax_layer=int(np.argmax(dl)), final_is_argmax=bool(np.argmax(dl) == len(dl) - 1),
                        final_over_next=float(dl[-1] / (np.sort(dl)[-2] + 1e-15))))
DV = pd.DataFrame(div); DV.to_csv(f'{OUT}/R9_B4_divergence.csv', index=False); LC = pd.DataFrame(loc); LC.to_csv(f'{OUT}/R9_B5_shared_prefix.csv', index=False)
x = DV.prefill_mean_abs_dlogg.values; y = DV.divergence_token.values; r4 = spearmanr(x, y).statistic; rp2 = np.random.default_rng(PSEED)
nul = np.array([spearmanr(x, rp2.permutation(y)).statistic for _ in range(NPERM)]); p4 = float((1 + (np.abs(nul) >= abs(r4)).sum()) / (NPERM + 1))
RES['R9-B4'] = ('prefill deformation predicts divergence' if (abs(r4) >= 0.3 and p4 < 0.01) else 'tested, not supported',
                dict(n=len(DV), divergence_median=float(DV.divergence_token.median()), q10_q90=[float(DV.divergence_token.quantile(.1)), float(DV.divergence_token.quantile(.9))],
                     censored=int(DV.censored.sum()), rho=float(r4), perm_p=p4))
f6 = float(LC.first6_identical.mean()); fa = float(LC.final_is_argmax.mean())
RES['R9-B5'] = ('localisation stable along the shared prefix' if (f6 == 1.0 and fa >= 0.95) else 'tested, not supported',
                dict(shared_tokens=len(LC), frac_layers0_5_identical=f6, frac_final_argmax=fa, argmax_layer_counts=LC.argmax_layer.value_counts().head(5).to_dict(), final_over_next_median=float(LC.final_over_next.median())))
print('B4', RES['R9-B4']); print('B5', RES['R9-B5'], flush=True); dump()
# ======================= B6 prompt RSA across LLMs at matched context =======================
def prof_at(d, k):
    r = d[d.token_idx == k]; out = {}; nn = {}
    for p, dd in r.groupby('prompt_text'):
        F = []
        for g, gg in dd.groupby('gen_idx'):
            Y = gg.sort_values('layer')[['g_micro', 'spectral_gap', 'sv_entropy']].values.astype(float); Y[:, 0] = np.log(Y[:, 0]); Y[:, 1] = np.log(Y[:, 1])
            u = np.linspace(0, 1, len(Y)); F.append(np.concatenate([np.interp(grid, u, Y[:, v]) for v in range(3)]))
        out[p] = np.mean(F, 0); nn[p] = float(dd.seq_len.iloc[0])
    return out, nn
def resid_feats(F, n, P):
    M = np.stack([F[p] for p in P]); X = np.c_[np.ones(len(P)), np.log([n[p] for p in P])]
    R = M - X @ np.linalg.lstsq(X, M, rcond=None)[0]; return (R - R.mean(0)) / (R.std(0) + 1e-12)
def zfe(F, P):
    M = np.stack([F[p] for p in P]); return (M - M.mean(0)) / (M.std(0) + 1e-12)
def mantel_partial(RA, RB, C, rng_):
    iu = np.triu_indices(RA.shape[0], 1); a = rankdata(RA[iu]); b = rankdata(RB[iu]); c = rankdata(C[iu])
    def pc(a, b):
        rab = np.corrcoef(a, b)[0, 1]; rac = np.corrcoef(a, c)[0, 1]; rbc = np.corrcoef(b, c)[0, 1]
        return rab, (rab - rac * rbc) / np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))
    full, part = pc(a, b); nf = np.empty(NPERM); npt = np.empty(NPERM); NP = RA.shape[0]
    for i in range(NPERM):
        pm = rng_.permutation(NP); nf[i], npt[i] = pc(a, rankdata(RB[np.ix_(pm, pm)][iu]))
    return float(full), float((1 + (nf >= full).sum()) / (NPERM + 1)), float(part), float((1 + (npt >= part).sum()) / (NPERM + 1))
rdm = lambda F: 1 - np.corrcoef(F)
models = {**Q, 'mistral_instruct': MI, 'mistral_ablit': MA, 'llama31_instruct': LI}
PRO = {(m, k): prof_at(d, k) for m, d in models.items() for k in (0, 128, 255)}
rr = np.random.default_rng(PSEED); rows = []
for a, b in [('qwen25_1_5b', 'qwen25_3b'), ('qwen25_1_5b', 'qwen25_7b'), ('qwen25_3b', 'qwen25_7b'), ('mistral_instruct', 'llama31_instruct'), ('mistral_instruct', 'mistral_ablit')]:
    for k in (0, 128, 255):
        (Fa, na), (Fb, nb) = PRO[(a, k)], PRO[(b, k)]; P = sorted(set(Fa) & set(Fb))
        RA = rdm(resid_feats(Fa, na, P)); RB = rdm(resid_feats(Fb, nb, P))
        ln = np.array([(np.log(na[p]) + np.log(nb[p])) / 2 for p in P]); C = np.abs(ln[:, None] - ln[None, :])
        full, pf, part, pp = mantel_partial(RA, RB, C, rr)
        raw_full = mantel_partial(rdm(zfe(Fa, P)), rdm(zfe(Fb, P)), C, rr)
        rows.append(dict(a=a, b=b, k=k, prompts=len(P), resid_full=full, resid_full_p=pf, resid_partial=part, resid_partial_p=pp, raw_full=raw_full[0], raw_full_p=raw_full[1], raw_partial=raw_full[2], raw_partial_p=raw_full[3]))
        print(rows[-1], flush=True)
B6 = pd.DataFrame(rows); B6.to_csv(f'{OUT}/R9_B6_llm_rsa.csv', index=False)
b6 = {f'{a}~{b}': dict(shared=bool((d.resid_partial_p <= 0.01).sum() >= 2), partial=d.resid_partial.round(3).tolist(), p=d.resid_partial_p.round(4).tolist())
      for (a, b), d in B6.groupby(['a', 'b'], sort=False)}
RES['R9-B6'] = b6; dump()
# ======================= B8 Qwen terminal collapse =======================
b8 = {}
for k, d in Q.items():
    L = d.layer.max(); pv = d.pivot_table(index=['gen_idx', 'token_idx'], columns='layer', values='g_micro')
    ratio = pv[L] / pv[L - 1] - 1; gl = d.groupby('gen_idx').gen_len.first(); last_tok = d.groupby('gen_idx').token_idx.max()
    idx = ratio.index.to_frame(index=False); eos = idx.gen_idx.map(gl < 256).values; final = (idx.token_idx.values == idx.gen_idx.map(last_tok).values)
    coll = (ratio.values <= -0.8)
    fe = coll[eos & final].mean() if (eos & final).any() else np.nan; other = coll[~(eos & final)].mean()
    b8[k] = dict(frac_rows_collapse=float(coll.mean()), ratio_median=float(np.median(ratio.values)), ratio_p05=float(np.percentile(ratio.values, 5)), n_eos_gens=int((gl < 256).sum()),
                 collapse_rate_final_eos_token=float(fe) if fe == fe else None, collapse_rate_other=float(other),
                 verdict='EOS signature' if (fe == fe and other > 0 and fe > 3 * other) or (fe == fe and other == 0 and fe > 0) else ('readout' if coll.mean() >= 0.8 else 'neither'))
RES['R9-B8'] = b8; print('B8', b8); dump()
