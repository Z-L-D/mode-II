# Round 17 scoring (PREREG_R17.md, sha 16ccbe54). Estimators unchanged: Round 6 tangent() (24-point depth grid, nearest matched clock),
# layer-order permutation null (1000 draws, rng 17, p95), R12-0 step-0 gate. Clock: t = sigma/(1+sigma) for PixArt (VP-eps) and
# Cosmos (EDM); raw flow sigma for SD3/SD3.5. Tests needing R17_SD35M are skipped (and say so) until that cube exists.
# Usage: SCR=<cube dir> python score_r17.py
import os, sys, pickle
sys.path.insert(0, '/mnt/project-files/review/round6/scripts')
from r6c_common import *
SCR = os.environ['SCR']; OUTD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
CLOCK = {'PIXART': 't', 'PIXARTU': 't', 'COSMOS2B': 't', 'COSMOS2BU': 't'}
def have(n, sub): return os.path.exists(f'{SCR}/F_{n}_{sub}.pkl')
def cube(n, sub):
    g, meta, X, s = pickle.load(open(f'{SCR}/F_{n}_{sub}.pkl', 'rb')); s = np.asarray(s, float)
    if CLOCK.get(n) == 't': s = s / (1 + s)
    return meta, X, s
rows, V = [], []
def T(a, b, sub, null=False, tag=''):
    _, XA, sa = cube(a, sub); _, XB, sb = cube(b, sub); tc, sr = tangent(XA, sa, XB, sb); p95 = float('nan')
    if null:
        rng = np.random.default_rng(17)
        p95 = float(np.percentile([tangent(XA, sa, XB[:, :, rng.permutation(XB.shape[2])], sb)[0] for _ in range(1000)], 95))
    r = dict(a=a, b=b, sub=sub, tangent=tc, shape=sr, null_p95=p95, tag=tag); rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True); return r
def step0(X): return np.log(np.abs(X[:, 0, :, 0]) + 1e-30).mean(0)
def gate(a, b, sub):
    x = step0(cube(a, sub)[1]); y = step0(cube(b, sub)[1]); return float(np.corrcoef(x - x.mean(), y - y.mean())[0, 1])
def add(i, test, verdict, detail): V.append(dict(id=i, test=test, verdict=verdict, detail=detail)); print(V[-1], flush=True)
S35 = have('SD35M', 'attn')
# R17-0a/b: completeness and hooks (cube builder keeps complete gens only; block counts from the cube depth)
det = []
for n, L, subs in [('PIXART', 28, ('attn', 'mlp')), ('COSMOS2B', 28, ('attn', 'mlp')), ('SD3M', 24, ('attn',)), ('SD35M', 24, ('attn',))]:
    for s in subs:
        if not have(n, s): det.append(f'{n} {s}: not received'); continue
        _, X, _ = cube(n, s); det.append(f'{n} {s}: {X.shape[0]} gens, {X.shape[2]}/{L} blocks')
ok0 = all(('not received' in d) or (int(d.split(': ')[1].split()[0]) >= 58 and f'/{d.split("/")[-1]}' and d.split(', ')[1].split('/')[0] == d.split('/')[-1].split()[0]) for d in det)
add('R17-0a/b', '>= 58 complete gens per session; every block hooked', 'PASS' if ok0 else 'FAIL', '; '.join(det))
# R17-0c gate
g = {s: gate('PIXART', 'COSMOS2B', s) for s in ('attn', 'mlp')}
txt = f"PixArt vs Cosmos attn {g['attn']:.3f}, mlp {g['mlp']:.3f}"; ok = g['attn'] < 0.85
if S35:
    g2 = gate('SD3M', 'SD35M', 'attn'); g3 = gate('SD35M', 'SD3SV', 'attn'); ok &= g2 < 0.85 and g3 >= 0.85
    txt += f"; SD3 M vs SD3.5 M {g2:.3f}; SD3.5 M recapture vs R11 {g3:.3f}"
else: txt += '; SD3.5 M parts not scored (no capture yet)'
add('R17-0c', 'right checkpoints (step-0 r)', 'PASS' if ok else 'FAIL', txt)
# R17-1
r1 = {s: T('PIXART', 'COSMOS2B', s, null=True, tag='R17-1') for s in ('attn', 'mlp')}
for s in ('attn', 'mlp'): T('COSMOS2B', 'PIXART', s, tag='R17-1 reverse')
below = all(v['tangent'] < v['null_p95'] for v in r1.values()); high = any(v['tangent'] >= 0.47 for v in r1.values())
v1 = 'SUPPORTED' if below else ('REFUTED (bottom tier did not replicate)' if high else 'REFUTED (above null, below 0.47)')
add('R17-1', 'PixArt <-> Cosmos tangent below its null p95, attn and mlp', v1, '; '.join(f"{s} {v['tangent']:.3f} (p95 {v['null_p95']:.3f})" for s, v in r1.items()))
# R17-2 and R17-4
if S35:
    r2 = T('SD3M', 'SD35M', 'attn', null=True, tag='R17-2'); T('SD35M', 'SD3M', 'attn', tag='R17-2 reverse'); t, p = r2['tangent'], r2['null_p95']
    out = 'MIDDLE TIER' if (t >= p + 0.30 and t <= 0.95) else ('UNRELATED TIER' if t < p else 'NO TIER CLAIM')
    add('R17-2', 'SD3 M <-> SD3.5 M (attn): decision rule', out, f'{t:.3f} (p95 {p:.3f}; middle needs >= {p + 0.30:.3f} and <= 0.95)')
    rs = T('SD35M', 'SD3SV', 'attn', tag='R17-3 recapture')
else:
    add('R17-2', 'SD3 M <-> SD3.5 M (attn): decision rule', 'NOT SCORED', 'R17_SD35M.csv not received')
# R17-3 cond <-> uncond
c = {}
for n, subs in [('PIXART', ('attn', 'mlp')), ('COSMOS2B', ('attn', 'mlp')), ('SD3M', ('attn',))] + ([('SD35M', ('attn',))] if S35 else []):
    for s in subs: c[f'{n} {s}'] = T(n, n + 'U', s, tag='R17-3 cond-uncond')['tangent']
ok3 = all(v >= 0.90 for v in c.values()); det3 = '; '.join(f'{k} {v:.3f}' for k, v in c.items())
if S35: ok3 &= rs['tangent'] >= 0.94; det3 += f"; recapture vs R11 {rs['tangent']:.3f}"
add('R17-3', 'same-weights references: cond<->uncond >= 0.90; SD3.5 M recapture vs R11 >= 0.94', ('SUPPORTED' if ok3 else 'REFUTED') + ('' if S35 else ' (partial: no SD3.5 M recapture)'), det3)
if S35:
    add('R17-4', 'recapture <-> R11 > SD3 M <-> SD3.5 M', 'SUPPORTED' if rs['tangent'] > r2['tangent'] else 'REFUTED', f"{rs['tangent']:.3f} vs {r2['tangent']:.3f}")
# descriptive
d = T('SD3M', 'SD3SV', 'attn', null=True, tag='descriptive SD3 M <-> R11 SD3.5 M (CFG 1)')
pd.DataFrame(rows).to_csv(f'{OUTD}/r17_tangents.csv', index=False); pd.DataFrame(V).to_csv(f'{OUTD}/r17_verdicts.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 220); print(pd.DataFrame(V).to_string(index=False))
