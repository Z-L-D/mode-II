# Review round 3 checks (post hoc, existing data, 2026-11-06).
# RC20 relative-onset threshold sensitivity (DeepSeek Q8): the ceiling-normalised curve's crossing at 0.3/0.4/0.5/0.6/0.7, native n,
#      20 fold partitions, 7 archive channels (the Round 6 S3 code with the threshold as a parameter).
# RC21 pooling increment by resolution (DeepSeek Q5): generation-bootstrap 90% intervals of the median log(g_pool/g_BM), steps >= 2,
#      and of the 512-minus-1024 difference.
import os as _os
_COSPAFS_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))
import sys; sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round6/scripts'); sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round5/scripts')
_os.chdir(f'{_COSPAFS_ROOT}/review/round5/scripts')
exec(open(f'{_COSPAFS_ROOT}/review/round5/scripts/r5_q_calcs.py').read().split('# ================= Q3 =================')[1].split('d1 = R5(S10)')[0])
from r6c_common import *
OUTR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'out')
rng = np.random.default_rng(20)
def cross(c, s, th):
    T = len(c); tau = np.arange(T) / (T - 1)
    if not (c > th).any(): return np.nan
    i = int(np.argmax(c > th)); t = tau[0] if i == 0 else tau[i - 1] + (th - c[i - 1]) * (tau[i] - tau[i - 1]) / (c[i] - c[i - 1])
    return float(np.interp(t, tau, s))
CH = [('ZIT', 'attn'), ('ZIT', 'mlp'), ('ZIB', 'attn'), ('ZIB', 'mlp'), ('MOCHI', 'attn'), ('SD35M', 'attn'), ('SD35L', 'attn')]
TH = (0.3, 0.4, 0.5, 0.6, 0.7); rows = []
for m, sub in CH:
    g, meta, X, s = arch(m, sub); ph = meta.pt.astype(str).str.strip().values; seed0 = np.zeros(len(X))
    r, grp, _ = resid(X, ph); vals = {t: [] for t in TH}
    for f in partitions(len(np.unique(grp)), 20, 1):
        c = curve_part(r, grp, seed0, False, f)
        for t in TH: vals[t].append(cross(c / c[-1], s, t) if c[-1] > 0 else np.nan)
    row = dict(channel=f'{m}-{sub}'); row.update({f'rel_{t}': float(np.nanmedian(vals[t])) for t in TH}); rows.append(row); print(row, flush=True)
R20 = pd.DataFrame(rows); R20.to_csv(f'{OUTR}/rc20_onset_threshold.csv', index=False)
print('RC20 range across channels:', {t: round(R20[f'rel_{t}'].max() - R20[f'rel_{t}'].min(), 3) for t in TH})
print('RC20 rank order vs 0.5 (Spearman):', {t: round(R20[f'rel_{t}'].corr(R20['rel_0.5'], method='spearman'), 3) for t in TH})
# ---------------- RC21 ----------------
def pair(meta_a, meta_b):
    ka = list(zip(meta_a.pt.astype(str), meta_a.seed.astype('int64'))); kb = list(zip(meta_b.pt.astype(str), meta_b.seed.astype('int64')))
    common = [k for k in ka if k in kb]; return [ka.index(k) for k in common], [kb.index(k) for k in common]
def pooled_lg(arm):
    _, mp, Xp, sp = R3(arm); dn = R5(arm)
    keep = np.array([len(str(x).split(',')) == 1 for x in mp.seqs]) if 'seqs' in mp else np.ones(len(mp), bool)
    ia, ib = pair(mp[keep], dn['meta']); return np.log(Xp[keep][ia][..., 0] / BM(dn)[ib][..., 0])[:, 2:]
rows = []; boot = {}
for res, arm in [('1024', 'S45_1024_s1001+S45_1024_s1002'), ('768', 'S45_768_s1001'), ('512', 'S45_512_s1001')]:
    lg = pooled_lg(arm); n = len(lg); b = [np.median(lg[rng.integers(0, n, n)]) for _ in range(2000)]; boot[res] = np.array(b)
    rows.append(dict(res=res, pairs=n, median=float(np.median(lg)), p05=float(np.percentile(b, 5)), p95=float(np.percentile(b, 95))))
d = boot['512'] - boot['1024']; rows.append(dict(res='512 minus 1024', pairs=np.nan, median=float(np.median(d)), p05=float(np.percentile(d, 5)), p95=float(np.percentile(d, 95))))
d = boot['768'] - boot['1024']; rows.append(dict(res='768 minus 1024', pairs=np.nan, median=float(np.median(d)), p05=float(np.percentile(d, 5)), p95=float(np.percentile(d, 95))))
R21 = pd.DataFrame(rows); R21.to_csv(f'{OUTR}/rc21_pooling_by_resolution.csv', index=False); print(R21.round(4).to_string(index=False))
