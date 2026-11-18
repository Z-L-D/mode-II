# S4: label-free three-segment topology of the population-mean layer-mean g(sigma), archive and clean ZIB captures.
# Output: out/S4_topology.csv, out/S4_results.json
from r6c_common import *
rng = np.random.default_rng(64); NB = 1000
def fit3(y, s):
    """continuous 3-segment piecewise-linear fit in time u=-sigma; breakpoints on the native grid; >=2 points per segment."""
    u = -s; T = len(y); best = None
    for i in range(1, T - 2):
        for j in range(i + 1, T - 1):
            A = np.stack([np.ones(T), u, np.maximum(0, u - u[i]), np.maximum(0, u - u[j])], 1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None); sse = ((A @ coef - y) ** 2).sum()
            if best is None or sse < best[0]: best = (sse, i, j, coef)
    sse, i, j, c = best; sl = np.array([c[1], c[1] + c[2], c[1] + c[2] + c[3]])
    return ''.join('+' if v > 0 else '-' for v in sl), i, j, sl
def extrema(y, s):
    tr = int(np.argmin(y)); pk = int(np.argmax(y[:tr + 1])) if tr > 0 else 0
    return s[pk], s[tr], (y[tr] - y[pk]) / y[pk], (y[-1] - y[tr]) / y[tr]
def per_prompt(X, meta):
    k = meta.pt.astype(str).str.strip().values if 'pt' in meta else meta
    up = np.unique(k); return np.stack([X[k == p][..., 0].mean((0, 2)) for p in up])
def analyse(lab, X, meta, s):
    PP = per_prompt(X, meta); y = PP.mean(0); top, i, j, sl = fit3(y, s); pk, tr, amp, reb = extrema(y, s)
    boot = []; bi = []; bj = []
    for _ in range(NB):
        yb = PP[rng.integers(0, len(PP), len(PP))].mean(0); t, ii, jj, _ = fit3(yb, s); boot.append(t); bi.append(s[ii]); bj.append(s[jj])
    boot = np.array(boot)
    return dict(set=lab, prompts=len(PP), steps=len(s), topology=top, frac_pmp=float((boot == '+-+').mean()), break1_sigma=float(s[i]), break2_sigma=float(s[j]),
                break1_ci=(float(np.percentile(bi, 2.5)), float(np.percentile(bi, 97.5))), break2_ci=(float(np.percentile(bj, 2.5)), float(np.percentile(bj, 97.5))),
                slopes=sl.round(5).tolist(), peak_sigma=float(pk), trough_sigma=float(tr), contraction=float(amp), rebound=float(reb))
rows = []
for m, sub in [('ZIT', 'mlp'), ('ZIB', 'mlp'), ('ZIT', 'attn'), ('ZIB', 'attn'), ('MOCHI', 'attn'), ('SD35M', 'attn'), ('SD35L', 'attn')]:
    g, meta, X, s = arch(m, sub); rows.append(analyse(f'archive {m}-{sub}', X, meta, s))
for sub in ['mlp', 'attn']:
    z4 = R5('Z40_1024_s1001', sub); rows.append(analyse(f'R4 ZIB CFG4 cond-{sub}', z4['X']['cond'], z4['meta'], np.asarray(z4['sigma'], float)))
    rows.append(analyse(f'R4 ZIB CFG4 uncond-{sub}', z4['X']['uncond'], z4['meta'], np.asarray(z4['sigma'], float)))
    z1 = R5('Z10_1024_s1001', sub); rows.append(analyse(f'R4 ZIB CFG1 cond-{sub}', z1['X']['cond'], z1['meta'], np.asarray(z1['sigma'], float)))
    _, m3, X3, s3 = R3('Z10_1024_s1001', sub); rows.append(analyse(f'R3 ZIB CFG1 cond-only-{sub}', X3, m3, np.asarray(s3, float)))
TP = pd.DataFrame(rows); TP.to_csv(f'{OUT6}/S4_topology.csv', index=False)
R = TP.set_index('set'); sc = {}
a = R.loc['archive ZIT-mlp', 'frac_pmp']; b = R.loc['archive ZIB-mlp', 'frac_pmp']
sc['S4a'] = ('CONFIRMED' if min(a, b) >= 0.95 else 'REFUTED', f'+-+ in {a:.3f} (ZIT mlp) and {b:.3f} (ZIB mlp) of bootstraps')
c1 = R.loc['R4 ZIB CFG4 cond-mlp', 'frac_pmp']; c2 = R.loc['R3 ZIB CFG1 cond-only-mlp', 'frac_pmp']
sc['S4b'] = ('CONFIRMED' if min(c1, c2) >= 0.95 else 'REFUTED', f'+-+ in {c1:.3f} (R4 CFG4 cond mlp) and {c2:.3f} (R3 CFG1 cond-only mlp)')
d1 = abs(R.loc['R4 ZIB CFG4 cond-mlp', 'break1_sigma'] - R.loc['archive ZIB-mlp', 'break1_sigma']); d2 = abs(R.loc['R4 ZIB CFG4 cond-mlp', 'break2_sigma'] - R.loc['archive ZIB-mlp', 'break2_sigma'])
sc['S4c'] = ('CONFIRMED' if d1 <= 0.03 and d2 <= 0.05 else 'REFUTED', f'breakpoints archive {R.loc["archive ZIB-mlp","break1_sigma"]:.3f}/{R.loc["archive ZIB-mlp","break2_sigma"]:.3f} vs clean {R.loc["R4 ZIB CFG4 cond-mlp","break1_sigma"]:.3f}/{R.loc["R4 ZIB CFG4 cond-mlp","break2_sigma"]:.3f}')
dd = abs(R.loc['R4 ZIB CFG4 cond-mlp', 'contraction'] - R.loc['archive ZIB-mlp', 'contraction']) * 100
sc['S4d'] = ('CONFIRMED' if dd <= 10 else 'REFUTED', f'contraction archive {100*R.loc["archive ZIB-mlp","contraction"]:.1f}% vs clean {100*R.loc["R4 ZIB CFG4 cond-mlp","contraction"]:.1f}% (diff {dd:.1f} points)')
at = [R.loc[f'archive {m}-attn', 'frac_pmp'] for m in MODELS]; na = int(sum(v >= 0.8 for v in at))
sc['S4e'] = ('CONFIRMED' if na >= 4 else 'REFUTED', f'{na}/5 attn channels: ' + ', '.join(f'{m} {R.loc[f"archive {m}-attn","topology"]} {v:.3f}' for m, v in zip(MODELS, at)))
jdump(dict(score=sc), 'S4_results.json')
pd.set_option('display.width', 300); print(TP.drop(columns=['slopes']).round(3).to_string())
for k, v in sc.items(): print(k, v)
