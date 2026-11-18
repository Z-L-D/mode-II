# Round 16 scoring (PREREG_R16.md, sha a3b59c92): gate R16-0, predictions R16-1..R16-8, exploratory panel rows.
# Estimators copied unchanged from ace_base_r14/scripts/score_r14.py (tangent, null, end collapse, parity, E6 share) and
# r14_B_replay.py (B6 crossings, 100 prompt bootstraps, rng 3000+k). Null: rng seed 16 (prereg).
# Usage: SCR=<cube dir with F_ACEB*, F_ACET*, F_T<k>*> python score_r16.py
import os, sys, json, pickle, types
sys.modules['gradio'] = types.ModuleType('gradio')
sys.path.insert(0, '/mnt/project-files/review/round6/scripts'); sys.path.insert(0, '/mnt/project-files/review/lenses')
from r6c_common import *
import lens_L2_rg as L2
from multiprocessing import Pool
OUTD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out'); SCR = os.environ['SCR']
K = [1, 2, 3, 4, 5, 7, 8, 9, 10]
def cube(n, sub):
    g, meta, X, s = pickle.load(open(f'{SCR}/F_{n}_{sub}.pkl', 'rb')); return meta, X, np.asarray(s, float)
rows, V = [], []
def T(a, b, sub, null=False, seed=16, tag=''):
    _, XA, sa = cube(a, sub); _, XB, sb = cube(b, sub); tc, sr = tangent(XA, sa, XB, sb); p95 = float('nan')
    if null:
        rng = np.random.default_rng(seed)
        p95 = float(np.percentile([tangent(XA, sa, XB[:, :, rng.permutation(XB.shape[2])], sb)[0] for _ in range(1000)], 95))
    r = dict(a=a, b=b, sub=sub, tangent=tc, shape=sr, null_p95=p95, tag=tag); rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True); return r
def verdict(i, test, ok, detail): V.append(dict(id=i, test=test, verdict='SUPPORTED' if ok else 'REFUTED', detail=detail)); print(V[-1], flush=True)
def step0(X): return np.log(np.abs(X[:, 0, :, 0]) + 1e-30).mean(0)
def gate(XA, XB):
    a = step0(XA); b = step0(XB); return float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1])
# R16-0 gate
G = []; okg = True
for k in K:
    for sub in ('attn', 'mlp'):
        _, X, _ = cube(f'T{k}', sub); rt = gate(X, cube('ACET', sub)[1]); rb = gate(X, cube('ACEB', sub)[1])
        G.append(dict(arm=f'T{k}', sub=sub, r_T6=rt, r_Base=rb, ok=rt >= 0.85 and rb < 0.85)); okg &= G[-1]['ok']
pd.DataFrame(G).to_csv(f'{OUTD}/r16_gate.csv', index=False)
gd = pd.DataFrame(G)
V.append(dict(id='R16-0', test='every arm: step-0 r with T6 >= 0.85 and with Base < 0.85 (attn, mlp)', verdict='PASS' if okg else 'FAIL',
              detail=f"r with T6 {gd.r_T6.min():.3f}-{gd.r_T6.max():.3f}; r with Base {gd.r_Base.min():.3f}-{gd.r_Base.max():.3f}; failing: {list(gd[~gd.ok].arm + '/' + gd[~gd.ok]['sub'])}"))
print(V[-1])
# R16-1
r1 = {s: T('T8', 'ACET', s, tag='R16-1') for s in ('attn', 'mlp')}
for s in ('attn', 'mlp'): T('ACET', 'T8', s, tag='R16-1 reverse')
verdict('R16-1', 'T8 <-> T6 tangent >= 0.94, attn and mlp', all(v['tangent'] >= 0.94 for v in r1.values()), '; '.join(f"{s} {v['tangent']:.3f}" for s, v in r1.items()))
# R16-2, R16-3
r2 = {s: T('ACEB', 'T8', s, null=True, tag='R16-2') for s in ('attn', 'mlp')}
for s in ('attn', 'mlp'): T('T8', 'ACEB', s, tag='R16-2 reverse')
verdict('R16-2', 'Base <-> T8 tangent >= p95 + 0.30 and <= 0.95, attn and mlp', all(v['tangent'] >= v['null_p95'] + 0.30 and v['tangent'] <= 0.95 for v in r2.values()),
        '; '.join(f"{s} {v['tangent']:.3f} (p95 {v['null_p95']:.3f} + 0.30 = {v['null_p95'] + 0.30:.3f})" for s, v in r2.items()))
ref = dict(attn=0.850, mlp=0.572)
verdict('R16-3', 'Base <-> T8 within 0.10 of Base <-> T6 (0.850 attn, 0.572 mlp)', all(abs(r2[s]['tangent'] - ref[s]) <= 0.10 for s in ref),
        '; '.join(f"{s} {r2[s]['tangent']:.3f} vs {ref[s]:.3f} (diff {r2[s]['tangent'] - ref[s]:+.3f})" for s in ref))
# R16-4 + exploratory k=1,2 and Base <-> T_k
r4 = {}
for k in K:
    if k == 8: continue
    for s in ('attn', 'mlp'):
        t = T(f'T{k}', 'T8', s, tag='R16-4' if k >= 3 else 'exploratory k<3'); T('T8', f'T{k}', s, tag='reverse')
        if k >= 3: r4[(k, s)] = t['tangent']
fails = [f'T{k} {s} {v:.3f}' for (k, s), v in r4.items() if v < 0.94]
verdict('R16-4', 'T_k <-> T8 >= 0.94 for k in {3,4,5,7,9,10}, attn and mlp (12 tests)', not fails,
        f"min {min(r4.values()):.3f}; below 0.94: {fails if fails else 'none'}")
for k in K:
    for s in ('attn', 'mlp'): T('ACEB', f'T{k}', s, tag='exploratory Base<->T_k')
# R16-5 crossings on T8 attn
meta, Xa8, _ = cube('T8', 'attn'); Xa = Xa8[..., 0]; G_, T_, L_ = Xa.shape
gi, tt, ll = np.meshgrid(np.arange(G_) + 1, np.arange(T_), np.arange(L_), indexing='ij')
ATT = pd.DataFrame(dict(gen_idx=gi.ravel(), token_idx=tt.ravel(), layer=ll.ravel(), g_micro=Xa.ravel()))
pt = pd.Series(meta.pt.astype(str).str.strip().values, index=np.arange(G_) + 1)
pat = lambda c: ''.join('+' if d == 'pos→neg' else '-' for _, d, _ in c)
def pboot(args):
    df, seed = args; rng = np.random.default_rng(seed); up = np.array(sorted(pt.unique())); pick = rng.choice(up, len(up), replace=True)
    parts, k = [], 0
    for p in pick:
        for g_ in pt.index[pt == p]:
            k += 1; parts.append(df[df.gen_idx == g_].assign(gen_idx=k))
    return L2.curve(pd.concat(parts, ignore_index=True))
obs = L2.curve(ATT)
with Pool(8) as pool: bs = pool.map(pboot, [(ATT, 3000 + k) for k in range(100)])
counts = pd.Series([len(b) for b in bs]); share1 = float((counts == 1).mean())
verdict('R16-5', 'T8 attn: 1 crossing modal in >= 50% of prompt bootstraps', share1 >= 0.50 and int(counts.mode()[0]) == 1,
        f"point {len(obs)} ({pat(obs)}); 1 crossing in {share1:.2f}; dist {counts.value_counts().sort_index().to_dict()}")
# R16-6 end collapse, R16-7 parity
ec, par = {}, {}
for k in K:
    _, X, _ = cube(f'T{k}', 'mlp'); p_ = np.log(X[..., 0]).mean((0, 1)); ec[k] = float(np.median(p_[12:23]) - p_[23])
    _, X, _ = cube(f'T{k}', 'attn'); lg = np.log(X[..., 0]); d = lg[:, :, 0::2].mean(2) - lg[:, :, 1::2].mean(2); par[k] = float((d > 0).mean())
verdict('R16-6', 'MLP L23 below median(L12-22) by > 0.20, T8 and every k >= 2', all(ec[k] > 0.20 for k in K if k >= 2),
        '; '.join(f'k{k} {v:.2f}' for k, v in ec.items()))
verdict('R16-7', 'attn even > odd in >= 80% of (gen, step) cells, T8 and every T_k', all(v >= 0.80 for v in par.values()),
        '; '.join(f'k{k} {v:.2f}' for k, v in par.items()))
# R16-8 E6 prompt share on T8
def e6(Y, pr):
    up, inv = np.unique(pr, return_inverse=True); k = len(up); cnt_ = np.bincount(inv); G, T_ = Y.shape[:2]; Yf = Y.reshape(G, T_, -1)
    Yf = (Yf - Yf.mean(0)) / (Yf.std(0) + 1e-12); n0 = (G - (cnt_ ** 2).sum() / G) / (k - 1); an = []
    for t in range(T_):
        Z = Yf[:, t]; mu = np.stack([Z[inv == i].mean(0) for i in range(k)])
        ssw = ((Z - mu[inv]) ** 2).sum(0); ssb = (cnt_[:, None] * (mu - Z.mean(0)) ** 2).sum(0)
        msw = ssw / (G - k); msb = ssb / (k - 1); vp = np.maximum((msb - msw) / n0, 0); an.append(vp.sum() / (vp.sum() + msw.sum()))
    return np.array(an)
ps = {}
for s in ('attn', 'mlp'):
    meta, X, _ = cube('T8', s); Y = np.stack([np.log(X[..., 0]), np.log(X[..., 1]), X[..., 2]], -1); ps[s] = e6(Y, meta.pt.values)
verdict('R16-8', 'T8 final-step E6 prompt share >= 0.70, attn and mlp', all(v[-1] >= 0.70 for v in ps.values()), f"attn {ps['attn'][-1]:.3f}, mlp {ps['mlp'][-1]:.3f}")
pd.DataFrame(rows).to_csv(f'{OUTD}/r16_tangents.csv', index=False); pd.DataFrame(V).to_csv(f'{OUTD}/r16_verdicts.csv', index=False)
json.dump(dict(end_collapse=ec, parity=par, prompt_share={k: v.tolist() for k, v in ps.items()}, B6_counts=counts.tolist(), B6_point=pat(obs)), open(f'{OUTD}/r16_details.json', 'w'), indent=1)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 200); print(pd.DataFrame(V).to_string(index=False))
