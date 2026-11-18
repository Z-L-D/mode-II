"""R8-2 (step grid vs relative onset and VR4) and R8-8 (VR4 along sigma), as frozen in R8_PREREGISTRATION.md
(SHA acc578ac, 2026-10-09T08:30:59Z). Estimators are the Round 6 S3 code, unchanged (exec'd from r5_q_calcs Q3 section)."""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import sys
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round6/scripts')
exec(open(f'{_COSPAFS_ROOT}/review/round5/scripts/r5_q_calcs.py').read().split('# ================= Q3 =================')[1].split('d1 = R5(S10)')[0])
from r6c_common import *
OUT = f'{_COSPAFS_ROOT}/review/r8_correlations/out'
rng = np.random.default_rng(63)
def rel_onset(c, s): return s50(c / c[-1], s) if c[-1] > 0 else np.nan
def sim_vr(kind_, G, ph, T, K, reps=20):
    v = []
    for _ in range(reps):
        if kind_ == 'walk': X = np.cumsum(rng.normal(size=(G, T, K)), 1)
        Y = X.reshape(G, T, K // 3, 3); Y = np.concatenate([np.exp(0.05 * Y[..., :2]), Y[..., 2:]], -1)
        r, _, _ = resid(Y, ph); v.append(VR(r, (4,))[4])
    return float(np.mean(v))
def matched_rel(X, ph, s, NS=50, seed_base=1000):
    up = np.unique(ph); seed0 = np.zeros(len(X)); b = []
    for i in range(NS):
        pick = rng.choice(up, 60, replace=False); sel = np.isin(ph, pick)
        r2, grp2, _ = resid(X[sel], ph[sel]); f = partitions(60, 1, seed_base + i)[0]
        c = curve_part(r2, grp2, seed0[sel], False, f); b.append(rel_onset(c, s))
    return np.array(b)
_, _, _, sZ = arch('ZIT', 'attn')
def nearest_idx(s, targets):
    idx = []; avail = list(range(len(s)))
    for t in targets:
        j = min(avail, key=lambda k: abs(s[k] - t)); idx.append(j); avail.remove(j)
    return np.array(sorted(idx))
rows = []; vrows = []
E3 = pd.read_csv(f'{_COSPAFS_ROOT}/review/round2/out/E3_plateau_onset.csv').set_index('ch')
for m, sub in [('ZIB', 'attn'), ('ZIB', 'mlp'), ('MOCHI', 'attn'), ('SD35M', 'attn'), ('SD35L', 'attn')]:
    g, meta, X, s = arch(m, sub); ph = meta.pt.astype(str).str.strip().values; G, T, L, _ = X.shape
    idx = nearest_idx(s, sZ); Xs = X[:, idx]; ss = s[idx]
    full = matched_rel(X, ph, s); sub8 = matched_rel(Xs, ph, ss)
    r, _, _ = resid(X, ph); vr_full = VR(r, (4,))[4] / sim_vr('walk', G, ph, T, L * 3)
    r8, _, _ = resid(Xs, ph); vr8 = VR(r8, (4,))[4] / sim_vr('walk', G, ph, 8, L * 3)
    row = dict(channel=f'{m}-{sub}', steps=T, sub_sigmas=';'.join(f'{x:.3f}' for x in ss),
               rel_full_median=float(np.nanmedian(full)), rel_sub8_median=float(np.nanmedian(sub8)),
               rel_sub8_p5=float(np.nanpercentile(sub8, 5)), rel_sub8_p95=float(np.nanpercentile(sub8, 95)),
               rel_drop=float(np.nanmedian(full) - np.nanmedian(sub8)), VR4w_full=vr_full, VR4w_sub8=vr8)
    rows.append(row); print(row, flush=True)
    # R8-8: VR4/walk in sliding windows of 9 states, step 4
    walk9 = sim_vr('walk', G, ph, 9, L * 3); ons = float(E3.loc[f'{m}-{sub}', 'onset_sigma'])
    for a in range(0, T - 9 + 1, 4):
        w = r[:, a:a + 9]; v = VR(w, (4,))[4] / walk9
        pos = 'above' if s[a + 8] > ons else ('below' if s[a] < ons else 'straddle')
        vrows.append(dict(channel=f'{m}-{sub}', start=a, sigma_hi=float(s[a]), sigma_lo=float(s[a + 8]), VR4w=float(v), onset=ons, position=pos))
A = pd.DataFrame(rows); A.to_csv(f'{OUT}/R8_2_grid_subsample.csv', index=False)
V = pd.DataFrame(vrows); V.to_csv(f'{OUT}/R8_8_vr_windows.csv', index=False)
n_on = int((A.rel_drop >= 0.10).sum()); n_vr = int((A.VR4w_sub8 >= 1.0).sum())
res = {'R8-2a': ('grid artifact' if n_on >= 3 else 'not a grid artifact', f'{n_on}/5 channels drop >= 0.10'),
       'R8-2b': ('grid artifact' if n_vr >= 3 else 'not a grid artifact', f'{n_vr}/5 channels VR4/walk >= 1 on the 8-state grid')}
dirs = []
for ch, d in V.groupby('channel'):
    ab = d[d.position == 'above'].VR4w.mean(); be = d[d.position == 'below'].VR4w.mean(); dirs.append((ch, ab, be, be - ab))
dd = [x[3] for x in dirs if np.isfinite(x[3])]
npos = sum(x >= 0.2 for x in dd); nneg = sum(x <= -0.2 for x in dd)
res['R8-8'] = ('changes at onset' if max(npos, nneg) >= 4 else 'no consistent change', '; '.join(f'{c}: above {a:.2f} below {b:.2f}' for c, a, b, _ in dirs))
json.dump(res, open(f'{OUT}/R8_2_8_results.json', 'w'), indent=1); print(res)
