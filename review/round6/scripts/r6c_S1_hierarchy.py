# S1: observable hierarchy on one pipeline from raw archive cubes (attn), plus historical-scalar reproduction and
# the capture-regime substitution (S1d). Output: out/S1_hierarchy.csv, out/S1_results.json
import itertools
from r6c_common import *
D = {m: arch(m) for m in MODELS}
E1 = {m: D[m][2][..., 0].mean((0, 2)) for m in MODELS}            # D1: mean g over gens and layers
E2 = {m: np.log(D[m][2][..., 0]).mean((0, 2)) for m in MODELS}    # D2: mean log g
rows = []
pairs = [('ZIT', 'ZIB')] + E10_UNREL + [('SD35M', 'SD35L')]
for a, b in pairs:
    _, _, XA, sa = D[a]; _, _, XB, sb = D[b]
    tc, sr = tangent(XA, sa, XB, sb)
    rows.append(dict(a=a, b=b, kind=kind(a, b), L0_D1=env_r(E1[a], sa, E1[b], sb), L0_D2=env_r(E2[a], sa, E2[b], sb),
                     L1_field=field_r(XA, sa, XB, sb), L2_shape=sr, L3_tangent=tc))
H = pd.DataFrame(rows); H.to_csv(f'{OUT6}/S1_hierarchy.csv', index=False)
lin = H[H.kind == 'lineage'].iloc[0]; un = H[H.kind == 'unrelated']
levels = ['L0_D1', 'L0_D2', 'L1_field', 'L2_shape', 'L3_tangent']
margin = {k: float(lin[k] - un[k].max()) for k in levels}
res = dict(margin=margin, unrelated_max={k: float(un[k].max()) for k in levels}, lineage={k: float(lin[k]) for k in levels})
# ---- reproduction of the historical scalar correlations (09-15 summary section 16) ----
hist = {('ZIB', 'MOCHI'): 0.9405, ('ZIT', 'ZIB'): 0.9325, ('ZIT', 'MOCHI'): 0.822}
def prompt_median(a, b, logv):
    ga, ma, XA, sa = D[a]; gb, mb, XB, sb = D[b]
    pa = ma.pt.astype(str).str.strip().values; pb = mb.pt.astype(str).str.strip().values; rs = []
    for p in sorted(set(pa) & set(pb)):
        A = XA[pa == p][..., 0]; B = XB[pb == p][..., 0]
        if logv: A = np.log(A); B = np.log(B)
        rs.append(env_r(A.mean((0, 2)), sa, B.mean((0, 2)), sb))
    return float(np.median(rs))
def cell_r(a, b):
    """sigma-aligned layer x sigma g field correlation on normalized depth (24 bins), raw g (not log)."""
    _, _, XA, sa = D[a]; _, _, XB, sb = D[b]
    A = depth24(XA[..., 0].mean(0)); B = depth24(XB[..., 0].mean(0))
    if len(sa) > len(sb): A, sa, B, sb = B, sb, A, sa
    lo, hi = overlap(sa, sb); m = (sa >= lo - 1e-9) & (sa <= hi + 1e-9)
    Bi = np.stack([interp_desc(sb, B[:, k], sa[m]) for k in range(24)], 1)
    return float(np.corrcoef(A[m].ravel(), Bi.ravel())[0, 1])
rep = []
for (a, b), h in hist.items():
    rep.append(dict(pair=f'{a}-{b}', historical=h, D1_env_g=env_r(E1[a], D[a][3], E1[b], D[b][3]), D2_env_logg=env_r(E2[a], D[a][3], E2[b], D[b][3]),
                    prompt_median_g=prompt_median(a, b, False), prompt_median_logg=prompt_median(a, b, True), cell_field_g=cell_r(a, b)))
REP = pd.DataFrame(rep); REP.to_csv(f'{OUT6}/S1_historical_reproduction.csv', index=False)
res['reproduction'] = REP.to_dict('records')
# ---- S1d: capture-regime substitution for ZIB (attn) ----
sub = {}
z4 = R5('Z40_1024_s1001', 'attn'); s4 = np.asarray(z4['sigma'], float)
z1 = R5('Z10_1024_s1001', 'attn'); s1 = np.asarray(z1['sigma'], float)
_, m3, X3, s3 = R3('Z10_1024_s1001', 'attn')
cands = {'archive ZIB (pooled, CFG4, 117 prompts)': (D['ZIB'][2], D['ZIB'][3]),
         'R4 ZIB CFG4 cond (explicit, 19 prompts)': (z4['X']['cond'], s4),
         'R4 ZIB CFG4 uncond (explicit, 19 prompts)': (z4['X']['uncond'], s4),
         'R4 ZIB CFG4 branch-mean (synthetic, 19 prompts)': (BM(z4), s4),
         'R4 ZIB CFG1 cond (20 prompts)': (z1['X']['cond'], s1),
         'R3 ZIB CFG1 cond-only (60 prompts)': (X3, np.asarray(s3, float))}
# prompt-matched archive subset (same 19 prompts as R4 CFG4) separates prompt subset from capture regime
pz = set(z4['meta'].pt.astype(str).str.strip()); pa = D['ZIB'][1].pt.astype(str).str.strip().values
cands['archive ZIB restricted to the R4 CFG4 prompts (pooled)'] = (D['ZIB'][2][np.isin(pa, list(pz))], D['ZIB'][3])
for lab, (X, s) in cands.items():
    for dname, f in [('D1', lambda Z: Z[..., 0].mean((0, 2))), ('D2', lambda Z: np.log(Z[..., 0]).mean((0, 2)))]:
        Ez = f(X); r_zit = env_r(Ez, s, (E1 if dname == 'D1' else E2)['ZIT'], D['ZIT'][3]); r_mo = env_r(Ez, s, (E1 if dname == 'D1' else E2)['MOCHI'], D['MOCHI'][3])
        sub[f'{lab} | {dname}'] = dict(r_ZIT=r_zit, r_MOCHI=r_mo, delta=r_zit - r_mo, n=len(X))
res['S1d'] = sub
base = sub['archive ZIB (pooled, CFG4, 117 prompts) | D1']['delta']; clean = sub['R4 ZIB CFG4 cond (explicit, 19 prompts) | D1']['delta']
res['S1d_change'] = clean - base
# ---- scoring ----
sc = {}
sc['S1a'] = ('CONFIRMED' if min(margin['L0_D1'], margin['L0_D2']) <= 0.05 else 'REFUTED', f"margin L0 D1 {margin['L0_D1']:.3f}, D2 {margin['L0_D2']:.3f}")
n6 = int((un['L0_D1'] >= 0.6).sum()); sc['S1b'] = ('CONFIRMED' if n6 >= 6 else 'REFUTED', f'{n6}/8 unrelated pairs with L0 D1 r >= 0.6; values ' + ', '.join(f"{r.a}-{r.b} {r.L0_D1:.3f}" for r in un.itertuples()))
ok = margin['L3_tangent'] >= 0.5 and margin['L3_tangent'] > margin['L2_shape'] > margin['L0_D1']
sc['S1c'] = ('CONFIRMED' if ok else 'REFUTED', 'margins ' + ', '.join(f'{k} {v:+.3f}' for k, v in margin.items()))
sc['S1d'] = ('CONFIRMED' if abs(res['S1d_change']) > 0.008 else 'REFUTED', f"delta archive {base:+.4f} -> R4 cond {clean:+.4f} (change {res['S1d_change']:+.4f})")
res['score'] = sc
jdump(res, 'S1_results.json')
pd.set_option('display.width', 250); print(H.round(3).to_string()); print(REP.round(4).to_string())
for k, v in sub.items(): print(f'{k:70s} r_ZIT {v["r_ZIT"]:.4f} r_MOCHI {v["r_MOCHI"]:.4f} delta {v["delta"]:+.4f}')
for k, v in sc.items(): print(k, v)
