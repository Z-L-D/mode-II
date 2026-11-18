# RC17 (post hoc, round-2 review GLM m12): recompute the binned beta_layer(g) curves of the layer-beta figure, panel (c),
# directly from the telemetry with numpy, without the analyzer's plotting function, and compare them with the curves
# extracted from the analyzer's figure object (papers/mode-I/data/I-F8c_beta_curves.csv). The extraction reads the
# plotted line's data arrays, not pixels; this check confirms that they equal an independent computation.
# Rule reimplemented: per generation, pivot g by (layer, step), beta = g(L+1) - g(L) against g(L); pool; 29 equal-width
# bins over [min g, max g]; bin mean where the bin holds >= 10 points; crossings by linear interpolation between bins.
import os, sys, glob, types, importlib.util
sys.modules['gradio'] = types.ModuleType('gradio')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
spec = importlib.util.spec_from_file_location('cpla', f'{ROOT}/review/round6/analyzer/cos_pafs_live_analyzer.py'); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
import numpy as np, pandas as pd
TEL = sys.argv[1]; REF = pd.read_csv(f'{ROOT}/papers/mode-I/data/I-F8c_beta_curves.csv')
CH = [('SD3.5 M attn', 'SD3.5MED-CFG-4.5-attn-1024-3TE-120', None), ('SD3.5 L attn', 'SD3.5LAR-CFG-3.5-attn-1024-3TE-120', None),
      ('ZIB attn', 'ZIB-CFG-4.0-attn-1024-120', 'attn'), ('ZIB MLP', 'ZIB-CFG-4.0-attn-1024-120', 'mlp'), ('ZIT MLP', 'ZIT-CFG-1.0-attn-1024-120', 'mlp'),
      ('Mochi attn', 'MOCHI-CFG-4.5-attn-480-120', None), ('ZIT attn', 'ZIT-CFG-1.0-attn-1024-120', 'attn')]
rows = []
for lab, folder, sub in CH:
    dm = A.DataManager(); dm.load_paths(','.join(sorted(glob.glob(f'{TEL}/{folder}/*.csv')))); df = dm.get_df(); df = df[df.g_micro.notna()]
    if sub: df = df[df.submodule == sub]
    G, B = [], []
    for _, d in df.groupby('gen_idx'):
        M = d.pivot_table(index='layer', columns='token_idx', values='g_micro', aggfunc='mean').to_numpy()
        b = np.diff(M, axis=0); g = M[:-1]; ok = np.isfinite(b) & np.isfinite(g); G.append(g[ok]); B.append(b[ok])
    g = np.concatenate(G); b = np.concatenate(B); e = np.linspace(g.min(), g.max(), 30); c = 0.5 * (e[:-1] + e[1:])
    idx = np.clip(np.searchsorted(e, g, side='right') - 1, 0, 28); inside = g < e[-1]  # bins are [e_i, e_{i+1}); the maximum is excluded, as in the analyzer
    n = np.bincount(idx[inside], minlength=29); s = np.bincount(idx[inside], b[inside], minlength=29)
    m = np.where(n >= 10, s / np.maximum(n, 1), np.nan); v = np.isfinite(m)
    x = []
    for i in range(28):
        if v[i] and v[i + 1] and m[i] * m[i + 1] < 0: x.append((round(float(c[i] + (c[i + 1] - c[i]) * (-m[i]) / (m[i + 1] - m[i])), 4), '-+' if m[i] < 0 else '+-'))
    R = REF[REF.channel == lab]
    rows.append(dict(channel=lab, bins=int(v.sum()), ref_bins=len(R), max_abs_diff_center=float(np.max(np.abs(c[v] - R.g_center.to_numpy()))) if v.sum() == len(R) else np.nan,
                     max_abs_diff_mean=float(np.max(np.abs(m[v] - R.beta_mean.to_numpy()))) if v.sum() == len(R) else np.nan, pattern=' '.join(d for _, d in x), crossings=';'.join(f'{a}' for a, _ in x)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{HERE}/out/rc17_beta_curves_recompute.csv', index=False)
