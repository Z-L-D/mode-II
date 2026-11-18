# Round 18 descriptive extras (not scored). (1) replay check: are R18 arms bit-identical to earlier same-seed captures?
# (2) post hoc split-half image-row tangent inside ZIT2 (prompt-set band), (3) addendum A: tr M_cond / tr M_uncond.
import os, sys, pickle
sys.path.insert(0, '/mnt/project-files/review/round6/scripts')
from r6c_common import *
SCR = os.environ['SCR']; R18 = os.environ['R18CUBES']; RAW = os.environ['R18RAW']; R17 = os.environ['R17CUBES']
OUTD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out'); out = []
def F(d, n, sub): g, m, X, s = pickle.load(open(f'{d}/F_{n}_{sub}.pkl', 'rb')); return m, np.asarray(X, float), np.asarray(s, float)
def replay(da, a, db, b, sub):
    ma, XA, _ = F(da, a, sub); mb, XB, _ = F(db, b, sub)
    if XA.shape[1:] != XB.shape[1:]: return dict(a=a, b=b, sub=sub, shared=0, note=f'shapes differ {XA.shape[1:]} vs {XB.shape[1:]}')
    A = dict(zip(ma.pt.str.strip(), XA)); B = dict(zip(mb.pt.str.strip(), XB)); k = [p for p in A if p in B]
    rel = np.array([np.abs(A[p] / B[p] - 1).max() for p in k])
    return dict(a=a, b=b, sub=sub, shared=len(k), identical=int((rel == 0).sum()), median_max_rel=float(np.median(rel)), max_rel=float(rel.max()))
for sub in ('attn', 'mlp'): out.append(replay(R18, 'ZIT2', SCR, 'ZITSV', sub))
for n, subs in [('PIXART', ('attn', 'mlp')), ('COSMOS2B', ('attn', 'mlp')), ('SD3M', ('attn',)), ('SD35M', ('attn',))]:
    for sub in subs:
        if os.path.exists(f'{R18}/F_S_{n}_{sub}.pkl'): out.append(replay(R18, 'S_' + n, R17, n, sub))
d = pd.DataFrame(out); d.to_csv(f'{OUTD}/r18_replay_check.csv', index=False); print(d.to_string(index=False), flush=True)
# split-half band on image rows
VI = ['g_img', 'gap_img', 'H_img']; rows = []
def Vc(dir_, n, sub): o = pickle.load(open(f'{dir_}/V_{n}_{sub}.pkl', 'rb')); return np.asarray(o['V'][..., [o['cols'].index(c) for c in VI]], float), np.asarray(o['s'], float)
for sub in ('attn', 'mlp'):
    X, s = Vc(R18, 'ZIT2', sub); Xa, _ = F(R18, 'ZIT2', sub)[1:]; Z, sz = Vc(SCR, 'Z2', sub); rng = np.random.default_rng(183); v = []; va = []; lin = []
    for _ in range(200):
        p = rng.permutation(len(X)); h1, h2 = p[:30], p[30:]; v.append(tangent(X[h1], s, X[h2], s)[0]); va.append(tangent(Xa[h1], s, Xa[h2], s)[0])
        q = rng.permutation(len(Z))[:30]; lin.append(tangent(Z[q], sz, X[h1], s)[0])
    rows.append(dict(sub=sub, split_half_image_p5=np.percentile(v, 5), split_half_image_median=np.median(v), split_half_all_p5=np.percentile(va, 5),
                     lineage_30v30_image_p95=np.percentile(lin, 95), lineage_30v30_image_median=np.median(lin), gap_p5_minus_p95=np.percentile(v, 5) - np.percentile(lin, 95)))
d = pd.DataFrame(rows); d.to_csv(f'{OUTD}/r18_splithalf_image_band.csv', index=False); print(d.round(4).to_string(index=False), flush=True)
# addendum A: trace ratio
rows = []
for n in ('R18_SD35L', 'R18_ZB6'):
    sp = pd.read_csv(f'{RAW}/{n}_spectrum.csv', usecols=['gen_idx', 'token_idx', 'layer', 'submodule', 'branch', 'rowset', 'energy'], low_memory=False)
    sp = sp[sp.rowset == 'all']; w = sp.pivot_table(index=['gen_idx', 'token_idx', 'layer', 'submodule'], columns='branch', values='energy').dropna()
    w['lr'] = np.log(w.cond / w.uncond)
    for sub, g in w.groupby('submodule'):
        cell = g.groupby(['token_idx', 'layer']).lr.mean(); rows.append(dict(arm=n, sub=sub, cells=len(cell), median_ratio=float(np.exp(cell.median())), p5=float(np.exp(cell.quantile(.05))), p95=float(np.exp(cell.quantile(.95))),
            share_within_10pct=float((cell.abs() <= np.log(1.1)).mean()), share_within_25pct=float((cell.abs() <= np.log(1.25)).mean()), min=float(np.exp(cell.min())), max=float(np.exp(cell.max()))))
d = pd.DataFrame(rows); d.to_csv(f'{OUTD}/r18_trace_ratio.csv', index=False); print(d.round(4).to_string(index=False))
