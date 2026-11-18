import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# S3: estimator re-statement for the seven archive channels (the "estimator calibration" gate).
# Seed-fate sigma50 and relative onset at matched n (60 prompts x their gens, 50 random prompt subsets, each with its own
# random fold partition) and at native n (all prompts, 50 random fold partitions); VR4 against walk / AR(0.8) / iid references
# simulated at each channel's own shape with the same estimator. Output: out/S3_estimators.csv, out/S3_results.json
exec(open(f'{_COSPAFS_ROOT}/review/round5/scripts/r5_q_calcs.py').read().split('# ================= Q3 =================')[1].split('d1 = R5(S10)')[0])
from r6c_common import *
rng = np.random.default_rng(63)
def rel_onset(c, s): return s50(c / c[-1], s) if c[-1] > 0 else np.nan
CH = [('ZIT', 'attn'), ('ZIT', 'mlp'), ('ZIB', 'attn'), ('ZIB', 'mlp'), ('MOCHI', 'attn'), ('SD35M', 'attn'), ('SD35L', 'attn')]
def sim_vr(kind_, G, ph, T, K, reps=20):
    v = []
    for _ in range(reps):
        if kind_ == 'walk': X = np.cumsum(rng.normal(size=(G, T, K)), 1)
        elif kind_ == 'iid': X = rng.normal(size=(G, T, K))
        else:
            X = np.zeros((G, T, K)); X[:, 0] = rng.normal(size=(G, K))
            for t in range(1, T): X[:, t] = 0.8 * X[:, t - 1] + rng.normal(size=(G, K))
        Y = X.reshape(G, T, K // 3, 3); Y = np.concatenate([np.exp(0.05 * Y[..., :2]), Y[..., 2:]], -1)
        r, _, _ = resid(Y, ph); v.append(VR(r, (4,))[4])
    return float(np.mean(v))
rows = []; NS = 50
for m, sub in CH:
    g, meta, X, s = arch(m, sub); ph = meta.pt.astype(str).str.strip().values; up = np.unique(ph); seed0 = np.zeros(len(X))
    # native n
    r, grp, _ = resid(X, ph); a = []; b = []; fin = []
    for f in partitions(len(np.unique(grp)), NS, 1):
        c = curve_part(r, grp, seed0, False, f); a.append(s50(c, s)); b.append(rel_onset(c, s)); fin.append(c[-1])
    nat = dict(s50=np.array(a), rel=np.array(b), fin=np.array(fin))
    # matched n: 60 prompts
    a = []; b = []; fin = []
    for i in range(NS):
        pick = rng.choice(up, 60, replace=False); sel = np.isin(ph, pick)
        r2, grp2, _ = resid(X[sel], ph[sel]); f = partitions(60, 1, 1000 + i)[0]
        c = curve_part(r2, grp2, seed0[sel], False, f); a.append(s50(c, s)); b.append(rel_onset(c, s)); fin.append(c[-1])
    mat = dict(s50=np.array(a), rel=np.array(b), fin=np.array(fin))
    vr = VR(r, (4,))[4]; G, T, L, _ = X.shape
    refs = {k: sim_vr(k, G, ph, T, L * 3) for k in ['walk', 'ar1_0.8', 'iid']}
    def q(v, p): return float(np.nanpercentile(v, p)) if np.isfinite(v).any() else np.nan
    row = dict(channel=f'{m}-{sub}', steps=T, gens=G, prompts=len(up))
    for lab, d in [('native', nat), ('matched60', mat)]:
        for k in ['s50', 'rel']:
            row[f'{lab}_{k}_median'] = float(np.nanmedian(d[k])) if np.isfinite(d[k]).any() else np.nan
            row[f'{lab}_{k}_p5'] = q(d[k], 5); row[f'{lab}_{k}_p95'] = q(d[k], 95); row[f'{lab}_{k}_nanfrac'] = float(np.isnan(d[k]).mean())
        row[f'{lab}_finalR2_median'] = float(np.median(d['fin']))
    row.update(VR4=vr, VR4_walk=refs['walk'], VR4_ar08=refs['ar1_0.8'], VR4_iid=refs['iid'], VR4_over_walk=vr / refs['walk'])
    rows.append(row); print(row, flush=True)
S3 = pd.DataFrame(rows); S3.to_csv(f'{OUT6}/S3_estimators.csv', index=False)
sc = {}
rel = S3.matched60_rel_median; s5 = S3.matched60_s50_median
n_in = int(((rel >= 0.80) & (rel <= 0.97)).sum()); sc['S3a'] = ('CONFIRMED' if n_in >= 6 else 'REFUTED', f'{n_in}/7 channels in [0.80, 0.97]: ' + ', '.join(f'{c} {v:.3f}' for c, v in zip(S3.channel, rel)))
rr = rel.max() - rel.min(); rs = np.nanmax(s5) - np.nanmin(s5)
sc['S3b'] = ('CONFIRMED' if rr < rs else 'REFUTED', f'range relative onset {rr:.3f} vs sigma50 {rs:.3f}')
wr = S3.matched60_rel_p95 - S3.matched60_rel_p5; ws = S3.matched60_s50_p95 - S3.matched60_s50_p5; nw = int((wr < ws).sum())
sc['S3c'] = ('CONFIRMED' if nw >= 6 else 'REFUTED', f'{nw}/7 channels with narrower relative-onset spread')
nv = int((S3.VR4_over_walk >= 1).sum()); sc['S3d'] = ('CONFIRMED' if nv >= 5 else 'REFUTED', f'{nv}/7 channels VR4/walk >= 1: ' + ', '.join(f'{c} {v:.2f}' for c, v in zip(S3.channel, S3.VR4_over_walk)))
jdump(dict(score=sc), 'S3_results.json')
for k, v in sc.items(): print(k, v)
