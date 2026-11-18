import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# POST HOC (asked by Zac 2026-10-27): Base<->Turbo tangent within caption rows only (V2) vs image rows only (V1) vs all rows (V0).
# Same tangent() and layer-permutation null (1000 draws) as score_r11.py; 60 prereg prompts; cond branch.
# Also SD3.5 M vs each (attn, image rows only exist there, so V2 is n/a for SD3.5 M).
import os, sys, pickle
sys.path.insert(0, f'{_COSPAFS_ROOT}/review/round6/scripts')
from r6c_common import *
SCR = os.environ.get('SCR', _COSPAFS_CACHE); OUTD = f'{_COSPAFS_ROOT}/review/peer_review_2026-10-17/r11/out'
P60 = [l.strip() for l in open(f'{_COSPAFS_ROOT}/review/peer_review_2026-10-17/capture/prompts_r3_cond60.txt') if l.strip()]
VI = {'V1_image': (0, 1, 2), 'V2_caption': (3, 4, 5)}
def load(arm, sub, var):
    g, meta, X, s = pickle.load(open(f'{SCR}/F_{arm}_{sub}.pkl', 'rb')); keep = meta.pt.astype(str).str.strip().isin(P60).values
    if var != 'V0_all': X = pickle.load(open(f'{SCR}/V_{arm}_{sub}.pkl', 'rb'))['V'][..., list(VI[var])]
    return X[keep].astype(float), np.asarray(s, float)
rng = np.random.default_rng(2); rows = []
for var in ('V0_all', 'V1_image', 'V2_caption'):
    for sub in ('attn', 'mlp'):
        XA, sa = load('Z2', sub, var); XB, sb = load('ZITSV', sub, var)
        ok = np.isfinite(XA).all() and np.isfinite(XB).all() and (XA[..., 0] > 0).all() and (XB[..., 0] > 0).all()
        tc, sr = tangent(XA, sa, XB, sb)
        p95 = np.percentile([tangent(XA, sa, XB[:, :, rng.permutation(XB.shape[2])], sb)[0] for _ in range(1000)], 95)
        # step-0 depth-profile r of mean log g (identity-gate statistic) in this variant
        a0 = np.log(XA[:, 0, :, 0]).mean(0); b0 = np.log(XB[:, 0, :, 0]).mean(0); r0 = np.corrcoef(a0, b0)[0, 1]
        rows.append(dict(var=var, sub=sub, tangent=tc, shape=sr, null_p95=p95, step0_profile_r=r0, finite=ok)); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{OUTD}/posthoc_text_vs_image_rows.csv', index=False)
