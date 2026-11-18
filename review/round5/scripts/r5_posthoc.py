import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Round 5 post hoc analyses (labelled post hoc; run after section Q was scored). Output: out/posthoc_R5.json
import json, numpy as np, pandas as pd
from r5_common import *
H = {}; rng = np.random.default_rng(56)
S10 = 'S10_1024_s1001+S10_1024_s1002'; S45 = 'S45_1024_s1001+S45_1024_s1002'
exec(open(f'{_COSPAFS_ROOT}/review/round5/scripts/r5_q_calcs.py').read().split('# ================= Q3 =================')[1].split('d1 = R5(S10)')[0])
def pair(meta_a, meta_b):
    ka = list(zip(meta_a.pt.astype(str), meta_a.seed.astype('int64'))); kb = list(zip(meta_b.pt.astype(str), meta_b.seed.astype('int64')))
    common = [k for k in ka if k in kb]; return [ka.index(k) for k in common], [kb.index(k) for k in common]
def rel_onset(c, s):
    """sigma where R^2(t)/R^2(final) first exceeds 0.5 (removes the ridge-shrinkage ceiling)."""
    return s50(c / c[-1], s) if c[-1] > 0 else np.nan
def dist2(X, ph, seed, s, center, parts):
    r, grp, _ = resid(X, ph); a = []; b = []; fin = []
    for f in parts:
        c = curve_part(r, grp, seed, center, f); a.append(s50(c, s)); b.append(rel_onset(c, s)); fin.append(c[-1])
    return np.array(a), np.array(b), np.array(fin)
def sm(v): v = np.asarray(v, float); return dict(median=float(np.nanmedian(v)), p5=float(np.nanpercentile(v, 5)) if np.isfinite(v).any() else None,
                                                  p95=float(np.nanpercentile(v, 95)) if np.isfinite(v).any() else None, nan_frac=float(np.isnan(v).mean()))
# ---- H1: seed-fate estimator vs sample size; ceiling-free relative onset ----
g, ma, Xa, sa = ARCH('SD35M_attn'); gid = np.array(g); pa = ma.pt.astype(str).values; up = np.unique(pa)
core = sorted(set(R5(S10)['meta'].pt))
rows = []
def add(label, n_prompts, runs, X, ph, seed, s, center, nparts=60, pseed=0):
    a, b, fin = dist2(X, ph, seed, s, center, partitions(len(np.unique(ph)), nparts, pseed))
    rows.append(dict(set=label, prompts=n_prompts, runs_per_prompt=runs, n=len(X), centred=center, sigma50=sm(a)['median'], sigma50_p5=sm(a)['p5'], sigma50_p95=sm(a)['p95'],
                     sigma50_undefined_frac=sm(a)['nan_frac'], rel_onset=sm(b)['median'], rel_p5=sm(b)['p5'], rel_p95=sm(b)['p95'], final_R2=float(np.median(fin))))
for k in [30, 60, 120]:
    for runs in [2, 4]:
        pick = np.array(core) if k == 30 else rng.choice(up, k, replace=False) if k < 120 else up
        sel = np.concatenate([np.where(pa == p)[0][np.argsort(gid[pa == p])][:runs] for p in pick])
        add(f'archive SD35M CFG4.5 BM ({"core 30" if k == 30 else f"{k} prompts"})', k, runs, Xa[sel], ma.ph.values[sel], np.zeros(len(sel)), sa, False)
_, m60, X60, s60 = R3(S10)
add('R3 CFG1 cond-only, 60 prompts', 60, 2, X60, m60.ph.values, m60.seed.values, s60, False); add('R3 CFG1 cond-only, 60 prompts', 60, 2, X60, m60.ph.values, m60.seed.values, s60, True)
d1 = R5(S10); d4 = R5(S45); ia, ib = pair(d1['meta'], d4['meta']); ph = d1['meta'].ph.values[ia]; sd = d1['meta'].seed.values[ia]; s = d1['sigma']
for lab, X in [('new CFG1 cond (30 prompts)', d1['X']['cond'][ia]), ('new CFG4.5 cond', d4['X']['cond'][ib]), ('new CFG4.5 uncond', d4['X']['uncond'][ib]), ('new CFG4.5 branch-mean', BM(d4)[ib])]:
    for cen in [False, True]: add(lab, 30, 2, X, ph, sd, s, cen)
H1 = pd.DataFrame(rows); H1.to_csv(f'{OUT}/H1_seedfate_vs_sample_size.csv', index=False); H['H1'] = H1.to_dict('records')
pd.set_option('display.width', 250); print(H1.round(3).to_string())
# ---- H2: VR granularity: thin 100-step arms to the 40-step sigma grid ----
def vr_one(X, k):
    r, _, _ = resid(X, np.zeros(len(X))); return VR(r, (k,))[k]
p10_40, p10_100, p45_40, p45_100 = R5('P10_40'), R5('P10_100'), R5('P45_40'), R5('P45_100')
j = sorted(set(int(np.argmin(np.abs(p10_100['sigma'] - x))) for x in p10_40['sigma']))
H['H2_thinned_steps'] = len(j)
for lab, A, Bt in [('CFG1', p10_40['X']['cond'], p10_100['X']['cond']), ('CFG45_cond', p45_40['X']['cond'], p45_100['X']['cond']),
                   ('CFG45_uncond', p45_40['X']['uncond'], p45_100['X']['uncond'])]:
    H[f'H2_{lab}_VR4_40'] = vr_one(A, 4); H[f'H2_{lab}_VR4_100_thinned'] = vr_one(Bt[:, j], 4); H[f'H2_{lab}_VR10_100'] = vr_one(Bt, 10)
    H[f'H2_{lab}_VR1step_var_ratio_100_over_40'] = None
# ---- H3: SD3 resolution shift applied to onset sigma (Esser et al. 2024, t_m = a t / (1 + (a-1) t), a = sqrt(m/n)) ----
Q = json.load(open(f'{OUT}/Q_results.json')); P4 = json.load(open(f'{_COSPAFS_ROOT}/review/round4/out/P_results.json'))
def shift(t, res): a = 1024 / res; return a * t / (1 + (a - 1) * t)
H['H3'] = {'cond': {r: dict(raw=Q['Q8_onsets'][k], shifted_to_1024=shift(Q['Q8_onsets'][k], r)) for r, k in [(1024, 'S45_1024_cond'), (768, 'S45_768_cond'), (512, 'S45_512_cond')]},
           'pooled_R3': {r: dict(raw=P4[k], shifted_to_1024=shift(P4[k], r)) for r, k in [(1024, 'P5_onset_S45_1024_s1001_attn'), (768, 'P5_onset_S45_768_s1001_attn'), (512, 'P5_onset_S45_512_s1001_attn')]}}
print(json.dumps(H['H3'], indent=1))
# ---- H4: contraction timing: pyramid seed-bootstrap CIs; core per-prompt early ratios ----
def quarters(A, B):
    a = np.log(A[..., 0]).mean(2).std(0, ddof=1); b = np.log(B[..., 0]).mean(2).std(0, ddof=1); r = b / a; q = np.array_split(np.arange(len(r)), 4)
    return [float(np.median(r[x])) for x in q]
for lab, A, B in [('40', p10_40['X']['cond'], p45_40['X']['cond']), ('100', p10_100['X']['cond'], p45_100['X']['cond'])]:
    bs = np.array([quarters(A[rng.integers(0, len(A), len(A))], B[rng.integers(0, len(B), len(B))]) for _ in range(1000)])
    H[f'H4_pyramid_{lab}_quarters'] = quarters(A, B); H[f'H4_pyramid_{lab}_quarters_CI95'] = [[float(np.percentile(bs[:, i], 2.5)), float(np.percentile(bs[:, i], 97.5))] for i in range(4)]
m1 = d1['meta']; k1 = list(zip(m1.pt, m1.seed)); k4 = list(zip(d4['meta'].pt, d4['meta'].seed)); o4 = [k4.index(k) for k in k1]
def sdp(X):
    d = pd.DataFrame(np.log(X[..., 0]).mean(2)); d['ph'] = m1.ph.values; return d.groupby('ph')[list(range(X.shape[1]))].std()
a = sdp(d1['X']['cond']); b = sdp(d4['X']['cond'][o4]).loc[a.index]; rr = b.values / a.values
H['H4_core_per_step_median_ratio'] = np.median(rr, 0).round(3).tolist()
H['H4_core_frac_prompts_Q1_ratio_below_0.8'] = float(np.mean(np.median(rr[:, :10], 1) < 0.8))
H['H4_core_steps5to9_median'] = float(np.median(rr[:, 5:10]))
# ---- H5: decompose Round 4 P1a (+0.071) into pooling on one trajectory and run/build difference ----
ia2, ib2 = [], []
mA = ma.copy(); pa_ = pa
dn = d4; ib_ = [i for i, p in enumerate(dn['meta'].pt) if p in set(pa_)]
newBM = np.stack([np.log(BM(dn)[[i for i, p in enumerate(dn['meta'].pt) if p == q]][..., 0]).mean(0) for q in core])
arch = np.stack([np.log(Xa[pa_ == q][..., 0]).mean(0) for q in core])
_, m3p, X3p, _ = R3(S45); pooled = np.stack([np.log(X3p[np.asarray(m3p.pt.astype(str)) == q][..., 0]).mean(0) for q in core])
H['H5_pooledR3_over_archiveBM_steps>=2_core30'] = float(np.median((pooled - arch)[:, 2:]))
H['H5_pooledR3_over_newBM_steps>=2_core30'] = float(np.median((pooled - newBM)[:, 2:]))
H['H5_newBM_over_archiveBM_steps>=2_core30'] = float(np.median((newBM - arch)[:, 2:]))
json.dump(H, open(f'{OUT}/posthoc_R5.json', 'w'), indent=1, default=str)
for k, v in H.items():
    if k not in ('H1', 'H3'): print(k, ':', v)
