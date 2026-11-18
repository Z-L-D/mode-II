# RC27 (post hoc, round-6 reviews: Qwen M7/Q6/Q7, DeepSeek M4/Q2, GLM 5.5, Kimi M6): the first-state identity gate on
# EVERY pair of captured checkpoints with equal hooked-layer counts (cond branch, gen-mean step-0 depth profile of log g,
# Pearson r, as R12-0). Each run carries its weights label (which file it ran, as established in the record; the archive
# "ZIB" campaign is labelled Turbo per the 2026-10-18 audit). Within-run splits (U = uncond branch, H, GPU halves) are left out
# so every pair is two separate runs. Reports, per band edge (0.85, 0.92, 0.95, 0.98): share of different-weights pairs at or
# above it (false-positive share) and of same-weights pairs below it (false-negative share), and a leave-one-run-out check of
# the 0.98 / 0.85 edges. Usage: python rc27_gate_allpairs.py <cache> <r16cubes> <r17cubes>
import os, sys, pickle, itertools, numpy as np, pandas as pd
CA, R16, R17 = sys.argv[1:4]; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
RUNS = {  # name: (dir, file stem, weights label)
    'Z1': (CA, 'Z1', 'Z-Image Base'), 'Z2': (CA, 'Z2', 'Z-Image Base'),
    'ZIT': (CA, 'ZIT', 'Z-Image Turbo'), 'ZITSV': (CA, 'ZITSV', 'Z-Image Turbo'), 'Z3': (CA, 'Z3', 'Z-Image Turbo'), 'ZIT25': (CA, 'ZIB', 'Z-Image Turbo'),
    'SD35M_arch': (CA, 'SD35M', 'SD3.5 Medium'), 'SD35M_R11': (CA, 'SD3SV', 'SD3.5 Medium'), 'SD35M_R17': (R17, 'SD35M', 'SD3.5 Medium'),
    'SD3M_R17': (R17, 'SD3M', 'SD3 Medium'), 'SD35L_arch': (CA, 'SD35L', 'SD3.5 Large'),
    'PIXART': (R17, 'PIXART', 'PixArt-Sigma'), 'COSMOS2B': (R17, 'COSMOS2B', 'Cosmos-Predict2 2B'),
    'MOCHI': (CA, 'MOCHI', 'Mochi'), 'MOCHI1': (CA, 'MOCHI1', 'Mochi'),
    'ACEB': (CA, 'ACEB', 'ACE-Step Base'), 'ACET_R14': (CA, 'ACET', 'ACE-Step turbo'),
}
for k in (1, 2, 3, 4, 5, 7, 8, 9, 10): RUNS[f'ACET_T{k}'] = (R16, f'T{k}', 'ACE-Step turbo')
def step0(d, f, sub):
    p = f'{d}/F_{f}_{sub}.pkl'
    if not os.path.exists(p): return None
    X = np.asarray(pickle.load(open(p, 'rb'))[2], float); return np.log(np.abs(X[:, 0, :, 0]) + 1e-30).mean(0)
rows = []
for sub in ('attn', 'mlp'):
    P = {n: step0(d, f, sub) for n, (d, f, _) in RUNS.items()}; P = {n: v for n, v in P.items() if v is not None}
    for a, b in itertools.combinations(P, 2):
        if len(P[a]) != len(P[b]): continue
        r = float(np.corrcoef(P[a], P[b])[0, 1]); same = RUNS[a][2] == RUNS[b][2]
        rows.append(dict(sub=sub, a=a, b=b, L=len(P[a]), wa=RUNS[a][2], wb=RUNS[b][2], same=same, r=r))
D = pd.DataFrame(rows); D.to_csv(f'{OUT}/rc27_gate_allpairs.csv', index=False)
pd.set_option('display.width', 250)
for sub in ('attn', 'mlp'):
    d = D[D['sub'] == sub]; S, X = d[d.same], d[~d.same]
    print(f'\n== {sub}: {len(S)} same-weights pairs, {len(X)} different-weights pairs (equal depth)')
    print(f'same-weights r: min {S.r.min():.3f}, median {S.r.median():.3f}; different-weights r: max {X.r.max():.3f}, median {X.r.median():.3f}')
    for e in (0.85, 0.92, 0.95, 0.98):
        print(f'  edge {e}: different-weights >= edge {int((X.r >= e).sum())}/{len(X)}; same-weights < edge {int((S.r < e).sum())}/{len(S)}')
    print('  different-weights pairs >= 0.85:'); print(X[X.r >= 0.85].sort_values('r', ascending=False)[['a', 'b', 'wa', 'wb', 'r']].round(3).to_string(index=False))
    print('  same-weights pairs < 0.98:'); print(S[S.r < 0.98].sort_values('r')[['a', 'b', 'wa', 'r']].round(3).to_string(index=False))
    # leave one run out: are the edges still clean (no different-weights pair >= 0.98, no same-weights pair < 0.85) without each run?
    runs = sorted(set(d.a) | set(d.b)); bad = []
    for r0 in runs:
        e = d[(d.a != r0) & (d.b != r0)]
        bad.append((r0, int((e[~e.same].r >= 0.98).sum()), int((e[e.same].r < 0.85).sum())))
    print('  leave-one-run-out (run, diff>=0.98, same<0.85):', [b for b in bad if b[1] or b[2]] or 'no violations whichever run is left out')
