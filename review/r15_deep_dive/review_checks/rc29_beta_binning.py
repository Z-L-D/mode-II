# RC29 (post hoc, round-6 review GLM 7): sensitivity of the analyzer's layer-direction beta(g) crossing count to the number of
# bins. Same computation as the analyzer's plot_beta_layer (per generation: g pivoted to layer x step, beta = diff along layer,
# all points pooled; equal-width bins over [min g, max g]; a bin needs >= 10 points; a crossing is a sign change between
# adjacent valid bins), with 15, 29 (the analyzer's own) and 50 bins. The 29-bin result is checked against the analyzer
# function itself. Archive channels as in Round 6 A2. Usage: python rc29_beta_binning.py <telemetry DM dir> <analyzer.py>
import sys, glob, re, types, importlib.util, numpy as np, pandas as pd, os
sys.modules['gradio'] = types.ModuleType('gradio'); TEL, AN = sys.argv[1:3]
spec = importlib.util.spec_from_file_location('cpla', AN); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
SETS = [('ZIT (8 steps)', 'ZIT-CFG-1.0-attn-1024-120', ['attn', 'mlp']), ('ZIT-25 (pooled CFG 4)', 'ZIB-CFG-4.0-attn-1024-120', ['attn', 'mlp']),
        ('Mochi CFG 4.5', 'MOCHI-CFG-4.5-attn-480-120', [None]), ('SD3.5 M', 'SD3.5MED-CFG-4.5-attn-1024-3TE-120', [None]),
        ('SD3.5 L', 'SD3.5LAR-CFG-3.5-attn-1024-3TE-120', [None])]
def points(df):
    G, B = [], []
    for _, gdf in df.groupby('gen_idx'):
        m = gdf.pivot_table(index='layer', columns='token_idx', values='g_micro', aggfunc='mean').values
        b = np.diff(m, axis=0); g = m[:-1]; k = ~np.isnan(b) & ~np.isnan(g); G.append(g[k]); B.append(b[k])
    return np.concatenate(G), np.concatenate(B)
def pattern(g, b, nb):
    e = np.linspace(g.min(), g.max(), nb + 1); c = .5 * (e[:-1] + e[1:]); bm = np.full(nb, np.nan)
    for i in range(nb):
        m = (g >= e[i]) & (g < e[i + 1])
        if m.sum() >= 10: bm[i] = b[m].mean()
    out = []
    for i in range(nb - 1):
        if np.isnan(bm[i]) or np.isnan(bm[i + 1]): continue
        if bm[i] * bm[i + 1] < 0: out.append('-+' if bm[i] < 0 else '+-')
    return ' '.join(out)
rows = []
for lab, d, subs in SETS:
    dm = A.DataManager(); dm.load_paths(','.join(sorted(glob.glob(f'{TEL}/{d}/*.csv')))); base = dm.get_df(); base = base[base.g_micro.notna()]
    for s in subs:
        df = base if s is None else base[base.submodule == s]; g, b = points(df)
        fig = A.plot_beta_layer(df); t = fig.axes[0].get_title(); plt.close(fig)
        an = ' '.join('+-' if x == 'pos→neg' else '-+' for x in re.findall(r'\(([a-z]+→[a-z]+)\)', t))
        r = dict(channel=lab, sub=s or 'pooled', analyzer_29=an, **{f'bins_{nb}': pattern(g, b, nb) for nb in (15, 20, 29, 40, 50)})
        r['n_' + 'cross'] = {k: len(v.split()) for k, v in r.items() if k.startswith('bins_')}; rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/rc29_beta_binning.csv', index=False)
