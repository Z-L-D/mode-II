# Post hoc (not preregistered): where does the PixArt <-> Cosmos tangent come from? Clock mapping, per step, per statistic,
# and with the first or last depth interval removed (end effects). Same tangent() and prof() as score_r17.py.
import os, sys, pickle
sys.path.insert(0, '/mnt/project-files/review/round6/scripts')
from r6c_common import *
SCR = os.environ['SCR']; OUTD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
def cube(n, sub, clock='t'):
    g, meta, X, s = pickle.load(open(f'{SCR}/F_{n}_{sub}.pkl', 'rb')); s = np.asarray(s, float)
    return X, (s / (1 + s) if clock == 't' else s)
rows = []
for sub in ('attn', 'mlp'):
    XA, sa = cube('PIXART', sub); XB, sb = cube('COSMOS2B', sub)
    A = prof(XA, sa); B = prof(XB, sb); J = [int(np.argmin(np.abs(sb - x))) for x in sa]
    dA = np.diff(A, axis=1); dB = np.diff(B, axis=1)
    per = [cos(dA[t], dB[j]) for t, j in enumerate(J)]
    rows.append(dict(sub=sub, test='primary (t clock)', value=float(np.mean(per))))
    XA2, sa2 = cube('PIXART', sub, 'raw'); XB2, sb2 = cube('COSMOS2B', sub, 'raw'); rows.append(dict(sub=sub, test='raw sigma clock', value=tangent(XA2, sa2, XB2, sb2)[0]))
    rows.append(dict(sub=sub, test='per step min..max', value=float(np.min(per)), value2=float(np.max(per))))
    rows.append(dict(sub=sub, test='first half of steps / second half', value=float(np.mean(per[:len(per)//2])), value2=float(np.mean(per[len(per)//2:]))))
    for k, nm in enumerate(['log g', 'log gap', 'H']):
        rows.append(dict(sub=sub, test=f'statistic {nm} only', value=float(np.mean([cos(dA[t][:, k], dB[j][:, k]) for t, j in enumerate(J)]))))
    rows.append(dict(sub=sub, test='drop first depth interval', value=float(np.mean([cos(dA[t][1:], dB[j][1:]) for t, j in enumerate(J)]))))
    rows.append(dict(sub=sub, test='drop last depth interval', value=float(np.mean([cos(dA[t][:-1], dB[j][:-1]) for t, j in enumerate(J)]))))
    rows.append(dict(sub=sub, test='drop first 2 and last 2 intervals', value=float(np.mean([cos(dA[t][2:-2], dB[j][2:-2]) for t, j in enumerate(J)]))))
    # share of squared difference energy in the first/last two intervals
    eA = (dA ** 2).sum(2).mean(0); eB = (dB ** 2).sum(2).mean(0)
    rows.append(dict(sub=sub, test='energy share of end intervals (2+2) PixArt / Cosmos', value=float((eA[:2].sum() + eA[-2:].sum()) / eA.sum()), value2=float((eB[:2].sum() + eB[-2:].sum()) / eB.sum())))
d = pd.DataFrame(rows); d.to_csv(f'{OUTD}/posthoc_r17_pixart_cosmos.csv', index=False); print(d.round(3).to_string(index=False))
