# Round 18 scoring (PREREG_R18.md, sha 6db32be7). Estimators unchanged: Round 6 tangent() (24-point depth grid, nearest matched
# sigma, first-named model supplies the states), layer-order permutation null (1000 draws, rng 17, p95), R12-0 step-0 gate.
# Secondary (not scored): native-grid tangent, RC22 phase-randomised surrogate p95 (300 draws, rng 22), 90% interval from
# 200 draws of 30 generations (rng 18). Usage: SCR=<archive/R11/R12 cube dir> R18CUBES=<dir> R18RAW=<csv dir> python score_r18.py
import os, sys, pickle, json
sys.path.insert(0, '/mnt/project-files/review/round6/scripts')
from r6c_common import *
SCR = os.environ['SCR']; R18 = os.environ['R18CUBES']; RAW = os.environ.get('R18RAW', '')
OUTD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
NEW = {'SD35L18', 'SD35L18U', 'SD35LT', 'ZIT2', 'ZB6', 'ZB6U'}
VI = ['g_img', 'gap_img', 'H_img']
def cube(n, sub, img=False):
    d = R18 if n in NEW else SCR
    if img:
        o = pickle.load(open(f'{d}/V_{n}_{sub}.pkl', 'rb')); return o['meta'], np.asarray(o['V'][..., [o['cols'].index(c) for c in VI]], float), np.asarray(o['s'], float)
    g, meta, X, s = pickle.load(open(f'{d}/F_{n}_{sub}.pkl', 'rb')); return meta, np.asarray(X, float), np.asarray(s, float)
rows, V = [], []
def dn(M):
    Y = M.copy(); Y[..., 0] = np.log(np.abs(Y[..., 0]) + 1e-30); Y[..., 1] = np.log(np.abs(Y[..., 1]) + 1e-30)
    return (Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-12)
def c_(a, b): a = a.ravel(); b = b.ravel(); return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
def pr(M, n):
    D = dn(M); u = np.linspace(0, 1, D.shape[1]); G = np.linspace(0, 1, n)
    return np.stack([[np.interp(G, u, D[t, :, k]) for k in range(3)] for t in range(D.shape[0])]).transpose(0, 2, 1)
def native(XA, sa, XB, sb):
    MA, MB = XA.mean(0), XB.mean(0); n = min(MA.shape[1], MB.shape[1]); A, B = pr(MA, n), pr(MB, n); rng = np.random.default_rng(22)
    f = lambda dB: float(np.mean([c_(np.diff(A[t], axis=0), dB[int(np.argmin(np.abs(sb - sa[t])))]) for t in range(len(sa))]))
    dB = np.diff(B, axis=1); F = np.fft.rfft(dB, axis=1); sur = []
    for _ in range(300):
        ph = rng.uniform(0, 2 * np.pi, F.shape[1]); ph[0] = 0; sur.append(f(np.fft.irfft(F * np.exp(1j * ph)[None, :, None], n=dB.shape[1], axis=1)))
    return f(dB), float(np.percentile(sur, 95)), n
def T(a, b, sub, null=False, tag='', img=False, boot=False):
    _, XA, sa = cube(a, sub, img); _, XB, sb = cube(b, sub, img); tc, sr = tangent(XA, sa, XB, sb); p95 = lo = hi = float('nan')
    if null:
        rng = np.random.default_rng(17)
        p95 = float(np.percentile([tangent(XA, sa, XB[:, :, rng.permutation(XB.shape[2])], sb)[0] for _ in range(1000)], 95))
    if boot:
        rng = np.random.default_rng(18)
        bs = [tangent(XA[rng.choice(len(XA), 30, replace=False)], sa, XB[rng.choice(len(XB), 30, replace=False)], sb)[0] for _ in range(200)]
        lo, hi = float(np.percentile(bs, 5)), float(np.percentile(bs, 95))
    nt, s95, n = native(XA, sa, XB, sb)
    r = dict(a=a, b=b, sub=sub, rows='image' if img else 'all', tangent=tc, shape=sr, null_p95=p95, ci5=lo, ci95=hi, native=nt, native_n=n, surr95_native=s95, tag=tag)
    rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True); return r
def step0(X): return np.log(np.abs(X[:, 0, :, 0]) + 1e-30).mean(0)
def gate(a, b, sub, img=False):
    x = step0(cube(a, sub, img)[1]); y = step0(cube(b, sub, img)[1]); return float(np.corrcoef(x - x.mean(), y - y.mean())[0, 1])
def add(i, test, verdict, detail): V.append(dict(id=i, test=test, verdict=verdict, detail=detail)); print(V[-1], flush=True)

# ---- gates ----
need = {'SD35L18': (58, 38, ('attn',)), 'SD35LT': (58, 38, ('attn',)), 'ZIT2': (58, 30, ('attn', 'mlp')), 'ZB6': (116, 30, ('attn', 'mlp'))}
okarm = {}; det = []; hook = True
for n, (g, L, subs) in need.items():
    for s in subs:
        _, X, _ = cube(n, s); okarm[n] = okarm.get(n, True) and X.shape[0] >= g; hook &= X.shape[2] == L
        det.append(f'{n} {s}: {X.shape[0]} gens (needs {g}), {X.shape[2]}/{L} blocks, {X.shape[1]} steps')
add('R18-0a', 'complete arms', 'PASS' if all(okarm.values()) else 'FAIL: ' + ', '.join(k for k, v in okarm.items() if not v), '; '.join(det))
o = pickle.load(open(f'{R18}/V_ZIT2_attn.pkl', 'rb')); ni = o['V'][..., o['cols'].index('n_img_rows')]; nt = o['V'][..., o['cols'].index('n_txt_rows')]
rowok = bool((ni == 4096).all() and np.isin(nt, (32, 64)).all())
add('R18-0b', 'every block hooked; ZIT2 row counts', 'PASS' if hook and rowok else 'FAIL', f'blocks as R18-0a; ZIT2 n_img_rows {sorted(set(ni.ravel().astype(int)))}, n_txt_rows {sorted(set(nt.ravel().astype(int)))}')
g = dict(zit2_a=gate('ZIT2', 'ZITSV', 'attn'), zit2_m=gate('ZIT2', 'ZITSV', 'mlp'), zb6_z1=gate('ZB6', 'Z1', 'attn'), zb6_zit=gate('ZB6', 'ZIT', 'attn'),
         l_arch=gate('SD35L18', 'SD35L', 'attn'), l_lt=gate('SD35L18', 'SD35LT', 'attn'), zb6_z1_m=gate('ZB6', 'Z1', 'mlp'), zb6_zit2=gate('ZB6', 'ZIT2', 'attn'))
hfile = os.path.join(RAW, 'r18_weight_hashes.json'); hs = 'r18_weight_hashes.json not in the upload, hash part not checked'
okc = g['zit2_a'] >= 0.98 and g['zit2_m'] >= 0.98 and g['zb6_z1'] >= 0.85 and g['zb6_zit'] < 0.85 and g['l_arch'] >= 0.85
if os.path.exists(hfile): hs = 'hashes: ' + json.dumps(json.load(open(hfile)))[:400]
add('R18-0c', 'right checkpoints (step-0 r; hashes)', ('PASS' if okc else 'FAIL') + (' (step-0 part only)' if not os.path.exists(hfile) else ''),
    f"ZIT2 vs ZITSV attn {g['zit2_a']:.4f}, mlp {g['zit2_m']:.4f}; ZB6 vs Z1 attn {g['zb6_z1']:.4f} (mlp {g['zb6_z1_m']:.4f}); ZB6 vs ZIT attn {g['zb6_zit']:.4f}; "
    f"R18 SD35L vs archive {g['l_arch']:.4f}; descriptive SD35L vs SD35LT {g['l_lt']:.4f}, ZB6 vs ZIT2 {g['zb6_zit2']:.4f}; {hs}")
s3 = cube('Z1', 'attn')[2]; s6 = cube('ZB6', 'attn')[2]; u = s3 / (3 - 2 * s3); e6 = float(np.abs(s6 - 6 * u / (1 + 5 * u)).max())
st = cube('ZIT2', 'attn')[2]; sv = cube('ZITSV', 'attn')[2]; et = float(np.abs(st - sv).max()) if len(st) == len(sv) else float('inf')
add('R18-0d', 'schedules', 'PASS' if e6 <= 1e-3 and et <= 1e-3 else 'FAIL', f'ZB6 max |sigma - shift6(Z1)| {e6:.2e}; ZIT2 vs ZITSV max diff {et:.2e}; SD35L18 sigma {np.round(cube("SD35L18","attn")[2][:4],4)}..., SD35LT sigma {np.round(cube("SD35LT","attn")[2],4)}')

# ---- R18-1 ----
r1 = T('SD35L18', 'SD35LT', 'attn', null=True, tag='R18-1', boot=True); t, p = r1['tangent'], r1['null_p95']
out = 'SUPPORTED (middle tier)' if (t >= p + 0.30 and t <= 0.95) else ('REFUTED (same-weights level, > 0.95)' if t > 0.95 else 'REFUTED (lineage rule not met)')
if not okarm['SD35LT']: out = 'NOT SCORED (R18-0a: SD35LT session incomplete); value by the frozen rule would read: ' + out
add('R18-1', 'SD3.5 L <-> L Turbo (attn): lineage rule and <= 0.95', out, f'{t:.3f} (p95 {p:.3f}; needs >= {p + 0.30:.3f} and <= 0.95); 90% interval {r1["ci5"]:.3f}-{r1["ci95"]:.3f}; native {r1["native"]:.3f} (surrogate p95 {r1["surr95_native"]:.3f})')
T('SD35LT', 'SD35L18', 'attn', null=True, tag='descriptive: Turbo first')
T('SD35LT', 'SD35L', 'attn', null=True, tag='descriptive: L Turbo <-> archive SD3.5 L'); T('SD35L', 'SD35LT', 'attn', null=True, tag='descriptive: archive SD3.5 L <-> L Turbo')
# ---- R18-2 ----
a = T('SD35L18', 'SD35L', 'attn', null=True, tag='R18-2 sessions', boot=True); b = T('SD35L18', 'SD35L18U', 'attn', tag='R18-2 cond-uncond')
add('R18-2', 'same weights: R18 SD35L <-> archive >= 0.95; cond <-> uncond >= 0.90', 'SUPPORTED' if a['tangent'] >= 0.95 and b['tangent'] >= 0.90 else 'REFUTED', f"{a['tangent']:.3f}; {b['tangent']:.3f}")
T('SD35L', 'SD35L18', 'attn', tag='descriptive: archive first')
# ---- R18-3 ----
i = {s: T('ZIT2', 'ZITSV', s, null=True, tag='R18-3a image rows', img=True, boot=True) for s in ('attn', 'mlp')}
al = {s: T('ZIT2', 'ZITSV', s, null=True, tag='R18-3a all rows') for s in ('attn', 'mlp')}
ok = all(i[s]['tangent'] >= 0.97 and al[s]['tangent'] >= 0.98 for s in ('attn', 'mlp'))
add('R18-3a', 'ZIT2 <-> ZITSV: image rows >= 0.97, all rows >= 0.98 (attn and MLP)', 'SUPPORTED' if ok else 'REFUTED',
    '; '.join(f"{s}: image {i[s]['tangent']:.3f}, all {al[s]['tangent']:.3f}" for s in ('attn', 'mlp')))
l = {s: T('Z2', 'ZIT2', s, null=True, tag='R18-3b image rows', img=True, boot=True) for s in ('attn', 'mlp')}
ok = 0.90 <= l['attn']['tangent'] <= 0.95 and 0.70 <= l['mlp']['tangent'] <= 0.80
add('R18-3b', 'Z2 <-> ZIT2 image rows: attn in [0.90, 0.95], MLP in [0.70, 0.80]', 'SUPPORTED' if ok else 'REFUTED', f"attn {l['attn']['tangent']:.3f}; MLP {l['mlp']['tangent']:.3f} (on file vs ZITSV: 0.928, 0.754)")
gap = i['attn']['tangent'] - l['attn']['tangent']
add('R18-3c', 'session band minus lineage on image rows (attn) >= 0.03', 'SUPPORTED' if gap >= 0.03 else 'REFUTED', f"{i['attn']['tangent']:.3f} - {l['attn']['tangent']:.3f} = {gap:.3f}; MLP {i['mlp']['tangent'] - l['mlp']['tangent']:.3f}")
for s in ('attn', 'mlp'):
    T('Z2', 'ZITSV', s, tag='reference on file: Z2 <-> ZITSV image rows', img=True); T('Z2', 'ZIT2', s, null=True, tag='descriptive: Z2 <-> ZIT2 all rows'); T('ZIT2', 'Z2', s, tag='descriptive: Turbo first, image rows', img=True)
# ---- R18-4 ----
ref = {'attn': 0.707, 'mlp': 0.719}; z = {s: T('ZB6', 'ZIT', s, null=True, tag='R18-4 lineage at shift 6', boot=True) for s in ('attn', 'mlp')}
sw = {s: T('ZB6', 'Z1', s, null=True, tag='R18-4 same weights, shift 6 vs 3') for s in ('attn', 'mlp')}
z1 = {s: T('Z1', 'ZIT', s, null=True, tag='reference on file: Z1 <-> ZIT') for s in ('attn', 'mlp')}
ok = all(abs(z[s]['tangent'] - ref[s]) <= 0.10 and z[s]['tangent'] >= z[s]['null_p95'] + 0.30 and sw[s]['tangent'] >= 0.90 for s in ('attn', 'mlp'))
add('R18-4', 'ZB6 <-> ZIT within 0.10 of Z1 <-> ZIT and lineage rule; ZB6 <-> Z1 >= 0.90', 'SUPPORTED' if ok else 'REFUTED',
    '; '.join(f"{s}: ZB6-ZIT {z[s]['tangent']:.3f} (ref {ref[s]}, p95 {z[s]['null_p95']:.3f}), ZB6-Z1 {sw[s]['tangent']:.3f}" for s in ('attn', 'mlp')))
for s in ('attn', 'mlp'):
    T('ZIT', 'ZB6', s, tag='descriptive: Turbo first'); T('ZB6', 'Z3', s, null=True, tag='descriptive: ZB6 <-> Z3'); T('ZB6', 'ZIT2', s, null=True, tag='descriptive: ZB6 <-> ZIT2'); T('ZB6', 'ZB6U', s, tag='descriptive: cond <-> uncond')
pd.DataFrame(rows).to_csv(f'{OUTD}/r18_tangents.csv', index=False); pd.DataFrame(V).to_csv(f'{OUTD}/r18_verdicts.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 400); print(pd.DataFrame(V).to_string(index=False))
