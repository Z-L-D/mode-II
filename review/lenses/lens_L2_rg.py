"""L2 RG lens (LENS_PREREGISTRATION.md).
R1 autonomy: out-of-sample R2 of beta_layer = f(g) vs f(g) + layer fixed effect (5-fold CV grouped by prompt), from the
   analyzer-cached per-generation g fields (sw_cache). beta_L = g_{L+1} - g_L, the analyzer's definition.
R2 crossing slopes and R3 cross-ratios: from the analyzer's plot_beta_layer (unchanged) on 100 generation bootstraps
   (seeds 3000+k, drawn copies relabelled, same crossing pattern only), as in salem_avalanche/sa_observables.py.
Output: out/L2_R1_autonomy.csv, out/L2_R2_slopes.csv, out/L2_R3_cross_ratios.csv"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import os, sys, glob, re, types, importlib.util
sys.modules['gradio'] = types.ModuleType('gradio')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.interpolate import BSpline
from lens_common import CH, OUT, load
REVIEW = f'{_COSPAFS_ROOT}/review'
SCR = os.environ.get('SCR', f'{_COSPAFS_CACHE}')
TEL = f'{SCR}/tel/DM'; NB = int(os.environ.get('NB', 100))
spec = importlib.util.spec_from_file_location('cpla', f'{REVIEW}/round6/analyzer/cos_pafs_live_analyzer.py'); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
FOLDER = {'ZIT attn': ('ZIT-CFG-1.0-attn-1024-120', 'attn'), 'ZIT MLP': ('ZIT-CFG-1.0-attn-1024-120', 'mlp'),
          'ZIB attn': ('ZIB-CFG-4.0-attn-1024-120', 'attn'), 'ZIB MLP': ('ZIB-CFG-4.0-attn-1024-120', 'mlp'),
          'Mochi attn': ('MOCHI-CFG-4.5-attn-480-120', None), 'SD3.5 M attn': ('SD3.5MED-CFG-4.5-attn-1024-3TE-120', None),
          'SD3.5 L attn': ('SD3.5LAR-CFG-3.5-attn-1024-3TE-120', None)}


def spline_basis(x, lo, hi, knots_inner):
    t = np.r_[[lo] * 4, knots_inner, [hi] * 4]
    return BSpline.design_matrix(np.clip(x, lo, hi - 1e-12), t, 3).toarray()


def r1(lab, key):
    d = load(key); g = d['g_gen'].astype(float); n, T, L = g.shape; pr = d['prompts']
    x = g[:, :, :-1].ravel(); y = (g[:, :, 1:] - g[:, :, :-1]).ravel()
    lay = np.broadcast_to(np.arange(L - 1), (n, T, L - 1)).ravel(); grp = np.repeat(pr, T * (L - 1))
    ok = np.isfinite(x) & np.isfinite(y); x, y, lay, grp = x[ok], y[ok], lay[ok], grp[ok]
    up = np.unique(pr); rng = np.random.default_rng(1); fold_of = dict(zip(up, rng.permutation(len(up)) % 5))
    fold = np.array([fold_of[p] for p in grp])
    pred_g, pred_gl = np.empty_like(y), np.empty_like(y)
    for f in range(5):
        tr, te = fold != f, fold == f
        lo, hi = x[tr].min(), x[tr].max(); kn = np.quantile(x[tr], np.linspace(0, 1, 14)[1:-1])
        Btr, Bte = spline_basis(x[tr], lo, hi, kn), spline_basis(x[te], lo, hi, kn)
        c = np.linalg.lstsq(Btr, y[tr], rcond=None)[0]; pred_g[te] = Bte @ c
        Etr = np.eye(L - 1)[lay[tr]][:, 1:]; Ete = np.eye(L - 1)[lay[te]][:, 1:]
        c2 = np.linalg.lstsq(np.hstack([Btr, Etr]), y[tr], rcond=None)[0]; pred_gl[te] = np.hstack([Bte, Ete]) @ c2
    ss = np.sum((y - y.mean()) ** 2); R2g = 1 - np.sum((y - pred_g) ** 2) / ss; R2gl = 1 - np.sum((y - pred_gl) ** 2) / ss
    return dict(channel=lab, pairs=len(y), R2_g=R2g, R2_g_layer=R2gl, dR2=R2gl - R2g, autonomous=(R2gl - R2g) < 0.05)


def load_df(lab):
    folder, sub = FOLDER[lab]
    dm = A.DataManager(); dm.load_paths(','.join(sorted(glob.glob(f'{TEL}/{folder}/*.csv'))))
    df = dm.get_df(); df = df[df.g_micro.notna()]
    return df[df.submodule == sub] if sub else df


def curve(df):
    fig = A.plot_beta_layer(df); ax = fig.axes[0]
    line = [l for l in ax.get_lines() if l.get_linewidth() == 2][0]; x, m = (np.asarray(v, float) for v in line.get_data())
    cr = [(float(gs), dd) for gs, dd in re.findall(r'g\*=([0-9.]+) \(([a-z]+→[a-z]+)\)', ax.get_title())]; plt.close(fig)
    out = []
    for gs, dd in cr:
        j = np.searchsorted(x, gs) - 1; j = int(np.clip(j, 0, len(x) - 2))
        out.append((gs, dd, (m[j + 1] - m[j]) / (x[j + 1] - x[j])))
    return out


def boot(args):
    df, seed = args; rng = np.random.default_rng(seed); g = df.gen_idx.unique(); pick = rng.choice(g, len(g), replace=True)
    return curve(pd.concat([df[df.gen_idx == x].assign(gen_idx=i + 1) for i, x in enumerate(pick)], ignore_index=True))


def cross_ratio(a, b, c, d): return (c - a) * (d - b) / ((c - b) * (d - a))


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    R1 = pd.DataFrame([r1(lab, key) for lab, key, *_ in CH]); R1.to_csv(os.path.join(OUT, 'L2_R1_autonomy.csv'), index=False)
    print(R1.round(4).to_string(index=False), flush=True)
    sl, crr = [], []
    for lab, key, D, fam in CH:
        df = load_df(lab); obs = curve(df); pat = ''.join('+' if d == 'pos→neg' else '-' for _, d, _ in obs)
        with Pool(3) as p: res = p.map(boot, [(df, 3000 + k) for k in range(NB)])
        same = [r for r in res if ''.join('+' if d == 'pos→neg' else '-' for _, d, _ in r) == pat]
        for k, (gs, dd, y) in enumerate(obs):
            ys = np.array([r[k][2] for r in same]); gss = np.array([r[k][0] for r in same])
            sl.append(dict(channel=lab, family=fam, k=k + 1, g_star=gs, g_star_se=gss.std(ddof=1) if len(same) > 2 else np.nan, direction=dd,
                           slope_y=y, slope_se=ys.std(ddof=1) if len(same) > 2 else np.nan, n_same=len(same)))
        if len(obs) == 3:
            def crs(c):
                g1, g2, g3 = (z[0] for z in c); return cross_ratio(0, g1, g2, g3), cross_ratio(g1, g2, g3, 1), g2 / g1, g3 / g1
            c0 = crs(obs); B = np.array([crs(r) for r in same])
            for i, nm in enumerate(('CR1(0,g1;g2,g3)', 'CR2(g1,g2;g3,1)', 'r21', 'r31')):
                crr.append(dict(channel=lab, family=fam, quantity=nm, value=c0[i], se=B[:, i].std(ddof=1), n_same=len(same)))
        print(lab, 'R2 done', pat, len(same), flush=True)
    S = pd.DataFrame(sl); S.to_csv(os.path.join(OUT, 'L2_R2_slopes.csv'), index=False); print(S.round(4).to_string(index=False))
    C = pd.DataFrame(crr); C.to_csv(os.path.join(OUT, 'L2_R3_cross_ratios.csv'), index=False); print(C.round(4).to_string(index=False))
