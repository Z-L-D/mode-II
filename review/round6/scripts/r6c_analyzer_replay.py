import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# A1 (Round 6, 2026-10-08): replay of the CoS-PAFS Live Analyzer v3 on the telemetry, using the analyzer's own code.
# Nothing is re-implemented. The analyzer file is imported unchanged (analyzer/cos_pafs_live_analyzer.py,
# SHA-256 4089062551a3c900132a205559d8932b87d7d207f2f44cf49496fade570b41f9). gradio is only used by make_app(),
# so it is replaced by an empty module to allow the import. Data are loaded with the analyzer's DataManager.load_paths
# (pd.read_csv per file + _deduplicate_gen_idx), exactly as the dashboard's "Load Paths" button does.
# Numbers are read from the analyzer's return values or from the figures it draws (titles, marker lines).
# Output: out/A1_analyzer_replay.csv, out/A1_analyzer_replay.json, analyzer_exports/<dataset>/<variant>_*.png
import sys, os, glob, json, types, re, importlib.util
sys.modules['gradio'] = types.ModuleType('gradio')
R6 = f'{_COSPAFS_ROOT}/review/round6'
spec = importlib.util.spec_from_file_location('cpla', f'{R6}/analyzer/cos_pafs_live_analyzer.py'); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
SCR = os.environ.get('SCR', f'{_COSPAFS_CACHE}')
TEL = f'{SCR}/tel/DM'; R4 = f'{_COSPAFS_ROOT}/data/r4_captures'
EXPORT = '--export' in sys.argv
DATASETS = {
    'ZIT archive (CFG1, 8 steps)': sorted(glob.glob(f'{TEL}/ZIT-CFG-1.0-attn-1024-120/*.csv')),
    'ZIB archive (CFG4 pooled, 25 steps)': sorted(glob.glob(f'{TEL}/ZIB-CFG-4.0-attn-1024-120/*.csv')),
    'MOCHI archive (CFG4.5, 30 steps)': sorted(glob.glob(f'{TEL}/MOCHI-CFG-4.5-attn-480-120/*.csv')),
    'MOCHI archive (CFG1, 100 steps)': sorted(glob.glob(f'{TEL}/MOCHI-CFG-1.0-attn-480-120/*.csv')),
    'SD35M archive (CFG4.5, 40 steps)': sorted(glob.glob(f'{TEL}/SD3.5MED-CFG-4.5-attn-1024-3TE-120/*.csv')),
    'SD35L archive (CFG3.5, 28 steps)': sorted(glob.glob(f'{TEL}/SD3.5LAR-CFG-3.5-attn-1024-3TE-120/*.csv')),
    'ZIB R4 (CFG4, explicit branches)': sorted(glob.glob(f'{R4}/ZIB-attn-TE1-CFG4.0-*/*.csv')),
    'ZIB R4 (CFG1, cond)': sorted(glob.glob(f'{R4}/ZIB-attn-TE1-CFG1.0-*/*.csv')),
}

def spiral_crossing_layers(df):
    """Panel A of plot_spiral_analysis: dotted red vlines = layers where beta_layer of the mean depth profile changes sign."""
    fig = A.plot_spiral_analysis(df); ax = fig.axes[0]
    xs = [float(l.get_xdata()[0]) for l in ax.lines if l.get_linestyle() == ':']
    plt.close(fig); return xs

def beta_layer_cloud(df):
    fig = A.plot_beta_layer(df); t = fig.axes[0].get_title(); plt.close(fig)
    return [(float(a), b) for a, b in re.findall(r'g\*=([0-9.eE+-]+) \(([^)]+)\)', t)]

def bootstrap_title(df):
    fig = A.plot_bootstrap_ratio(df)
    if fig is None: return None
    t = ' || '.join(ax.get_title().replace('\n', ' ') for ax in fig.axes if ax.get_title()); plt.close(fig); return t

def summarize(label, variant, df, export_dir=None):
    L = df.layer.nunique(); steps = df.token_idx.nunique()
    prof = df.groupby('layer')['g_micro'].mean()   # what plot_depth_profile draws
    cl = spiral_crossing_layers(df)
    cloud = beta_layer_cloud(df)
    subs = sorted(df.submodule.dropna().unique()) if 'submodule' in df else ['all']
    mp = {}
    for sm in (subs if 'submodule' in df else ['all']):
        _, _, fps, mg = A._get_mean_profile_crossings(df, sm)
        mp[sm] = fps
    # same function on the frame without token_type (DM rows are token_type 'step', so the analyzer's 'generated' filter empties them)
    d2 = df.drop(columns=['token_type']) if 'token_type' in df else df
    mp2 = {}
    for sm in (subs if 'submodule' in df else ['all']):
        _, _, fps, mg = A._get_mean_profile_crossings(d2, sm)
        mp2[sm] = [(round(g, 6), d) for g, d in fps]
    tb = A.detect_phase_boundaries(df)
    sig = df.groupby('token_idx')['sigma'].mean() if 'sigma' in df else None
    tb_s = [(round(x, 2), d, round(float(np.interp(x, sig.index.values, sig.values)), 3) if sig is not None else None) for x, d in tb]
    boot = bootstrap_title(d2)
    row = dict(dataset=label, variant=variant, rows=len(df), gens=df.gen_idx.nunique(), layers=L, steps=steps,
               depth_turning_layers=[round(x, 2) for x in cl], n_depth_turns=len(cl),
               depth_argmax_layer=int(prof.idxmax()), depth_argmin_layer=int(prof.idxmin()),
               beta_layer_cloud_crossings=[(round(g, 6), d) for g, d in cloud], n_cloud_crossings=len(cloud),
               mean_profile_crossings_as_written={k: len(v) for k, v in mp.items()},
               mean_profile_crossings_no_token_type=mp2,
               bootstrap_ratio_no_token_type=boot,
               temporal_boundaries=tb_s)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True); A.export_all_plots(df, export_dir, variant)
    return row

rows = []
for label, paths in DATASETS.items():
    dm = A.DataManager(); dm.load_paths(','.join(paths)); df = dm.get_df()
    if df is None or 'g_micro' not in df: print('cannot load', label); continue
    bad = df['g_micro'].isna().sum()
    print(f'{label}: {len(paths)} files, {len(df)} rows ({bad} rows with no g_micro after read_csv)', flush=True)
    variants = [('as loaded', df)]
    if 'submodule' in df and df.submodule.nunique() > 1:
        variants += [(f'{s} only', df[df.submodule == s]) for s in sorted(df.submodule.unique())]
    if 'branch' in df and df.branch.nunique() > 1:
        for b in sorted(df.branch.dropna().unique()):
            for s in sorted(df.submodule.unique()):
                variants.append((f'{b} branch, {s} only', df[(df.branch == b) & (df.submodule == s)]))
    for v, d in variants:
        ed = f"{R6}/analyzer_exports/{re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_')}" if EXPORT else None
        r = summarize(label, v, d.copy(), ed); r['rows_without_g'] = int(bad); rows.append(r)
        print('  ', v, '| depth turns', r['n_depth_turns'], r['depth_turning_layers'], '| cloud', r['n_cloud_crossings'],
              '| mean-profile (as written)', r['mean_profile_crossings_as_written'], '| no token_type', {k: len(x) for k, x in r['mean_profile_crossings_no_token_type'].items()},
              '| temporal', r['temporal_boundaries'], flush=True)
out = pd.DataFrame(rows)
out.to_csv(f'{R6}/out/A1_analyzer_replay.csv', index=False)
json.dump(rows, open(f'{R6}/out/A1_analyzer_replay.json', 'w'), indent=1, default=str)
