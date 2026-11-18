import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Round 14 scoring (PREREG_R14.md, sha dc997f6d): R12-6 tangent, R14-1..R14-6. R12-7 and R14-B are in r14_B_replay.py.
# Usage: SCR=<cube dir with F_ACEB*, F_ACET*> python score_r14.py
import os, sys, json, pickle
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round6/scripts')
from r6c_common import *
OUTD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
SCR = os.environ.get('SCR', _COSPAFS_CACHE)
def cube(n, sub):
    g, meta, X, s = pickle.load(open(f'{SCR}/F_{n}_{sub}.pkl', 'rb')); return meta, X, np.asarray(s, float)
rows, V = [], []
def T(a, b, sub, null=False, seed=14):
    _, XA, sa = cube(a, sub); _, XB, sb = cube(b, sub); tc, sr = tangent(XA, sa, XB, sb); p95 = float('nan')
    if null:
        rng = np.random.default_rng(seed)
        p95 = float(np.percentile([tangent(XA, sa, XB[:, :, rng.permutation(XB.shape[2])], sb)[0] for _ in range(1000)], 95))
    r = dict(a=a, b=b, sub=sub, tangent=tc, shape=sr, null_p95=p95); rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True); return r
def verdict(i, test, ok, detail): V.append(dict(id=i, test=test, verdict='SUPPORTED' if ok else 'REFUTED', detail=detail))
# R12-6
r6 = {sub: T('ACEB', 'ACET', sub, null=True) for sub in ('attn', 'mlp')}
for sub in ('attn', 'mlp'): T('ACET', 'ACEB', sub)                       # sensitivity: reverse order
ok = all(r6[s]['tangent'] >= r6[s]['null_p95'] + 0.30 and r6[s]['tangent'] <= 0.95 for s in r6)
verdict('R12-6', 'ACE Base <-> ACE Turbo tangent >= null p95 + 0.30 and <= 0.95 (attn and mlp)', ok,
        '; '.join(f"{s} {r6[s]['tangent']:.3f} (p95 {r6[s]['null_p95']:.3f} + 0.30 = {r6[s]['null_p95'] + 0.30:.3f})" for s in r6))
# R14-1 GPU gate
det = {}
for sub in ('attn', 'mlp'):
    t = T('ACEB0', 'ACEB1', sub); _, X0, _ = cube('ACEB0', sub); _, X1, _ = cube('ACEB1', sub)
    det[sub] = (t['tangent'], float(np.corrcoef(np.log(X0[..., 0]).mean(0).ravel(), np.log(X1[..., 0]).mean(0).ravel())[0, 1]))
verdict('R14-1', 'gpu0 vs gpu1: tangent >= 0.95 and field r >= 0.98', all(a >= 0.95 and b >= 0.98 for a, b in det.values()),
        '; '.join(f'{s} tangent {a:.3f}, field r {b:.4f}' for s, (a, b) in det.items()))
# R14-2 Base lower g than Turbo, per layer, at Turbo's sigmas
cnt = {}; per = {}
for sub in ('attn', 'mlp'):
    _, XB, sB = cube('ACEB', sub); _, XT, sT = cube('ACET', sub)
    idx = [int(np.argmin(np.abs(sB - x))) for x in sT]
    lb = np.log(XB[..., 0]).mean(0)[idx].mean(0); lt = np.log(XT[..., 0]).mean(0).mean(0)
    cnt[sub] = int((lb < lt).sum()); per[sub] = dict(base=lb.tolist(), turbo=lt.tolist(), matched_base_steps=idx, base_sigma=sB[idx].tolist(), turbo_sigma=sT.tolist())
verdict('R14-2', 'Base log g < Turbo in >= 18/24 layers (sigma-matched), attn and mlp', all(v >= 18 for v in cnt.values()),
        f"attn {cnt['attn']}/24, mlp {cnt['mlp']}/24")
# R14-3 cond vs uncond
r3 = {sub: T('ACEB', 'ACEBU', sub) for sub in ('attn', 'mlp')}
verdict('R14-3', 'Base cond <-> uncond tangent >= 0.90', all(v['tangent'] >= 0.90 for v in r3.values()),
        f"attn {r3['attn']['tangent']:.3f}, mlp {r3['mlp']['tangent']:.3f}")
# R14-4 end collapse (MLP)
ec = {}
for n in ('ACEB', 'ACET'):
    _, X, _ = cube(n, 'mlp'); prof_ = np.log(X[..., 0]).mean((0, 1)); ec[n] = dict(L23=float(prof_[23]), med_L12_22=float(np.median(prof_[12:23])), drop=float(np.median(prof_[12:23]) - prof_[23]), profile=prof_.tolist())
verdict('R14-4', 'Base MLP: L23 below median(L12-22) by > 0.20 (log g)', ec['ACEB']['drop'] > 0.20,
        f"Base drop {ec['ACEB']['drop']:.3f} (Turbo, descriptive: {ec['ACET']['drop']:.3f})")
# R14-5 parity sign
_, X, _ = cube('ACEB', 'attn'); lg = np.log(X[..., 0]); d = lg[:, :, 0::2].mean(2) - lg[:, :, 1::2].mean(2); share = float((d > 0).mean())
_, Xm, _ = cube('ACEB', 'mlp'); lgm = np.log(Xm[..., 0]); dm_ = lgm[:, :, 0::2].mean(2) - lgm[:, :, 1::2].mean(2)
verdict('R14-5', 'attn even > odd (Turbo sign) in >= 80% of (gen, step) cells', share >= 0.80,
        f'attn even>odd share {share:.3f}, median diff {np.median(d):.3f}; mlp (descriptive) even>odd {float((dm_ > 0).mean()):.3f}, median {np.median(dm_):.3f}')
# R14-6 final-step prompt share (E6 ANOVA share, B10 code)
def e6(Y, pr):
    up, inv = np.unique(pr, return_inverse=True); k = len(up); cnt_ = np.bincount(inv); G, T_ = Y.shape[:2]; Yf = Y.reshape(G, T_, -1)
    Yf = (Yf - Yf.mean(0)) / (Yf.std(0) + 1e-12); n0 = (G - (cnt_ ** 2).sum() / G) / (k - 1); an = []
    for t in range(T_):
        Z = Yf[:, t]; mu = np.stack([Z[inv == i].mean(0) for i in range(k)])
        ssw = ((Z - mu[inv]) ** 2).sum(0); ssb = (cnt_[:, None] * (mu - Z.mean(0)) ** 2).sum(0)
        msw = ssw / (G - k); msb = ssb / (k - 1); vp = np.maximum((msb - msw) / n0, 0); an.append(vp.sum() / (vp.sum() + msw.sum()))
    return np.array(an)
ps = {}
for sub in ('attn', 'mlp'):
    meta, X, s = cube('ACEB', sub); Y = np.stack([np.log(X[..., 0]), np.log(X[..., 1]), X[..., 2]], -1); ps[sub] = e6(Y, meta.pt.values)
verdict('R14-6', 'E6 ANOVA prompt share at last step >= 0.70', all(v[-1] >= 0.70 for v in ps.values()),
        f"attn {ps['attn'][-1]:.3f}, mlp {ps['mlp'][-1]:.3f}")
pd.DataFrame(rows).to_csv(f'{OUTD}/r14_tangents.csv', index=False)
pd.DataFrame(V).to_csv(f'{OUTD}/r14_verdicts_A.csv', index=False)
jdump_ = dict(R14_2=per, R14_4=ec, R14_6={k: v.tolist() for k, v in ps.items()})
json.dump(jdump_, open(f'{OUTD}/r14_details_A.json', 'w'), indent=1)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 200); print(pd.DataFrame(V).to_string(index=False))
