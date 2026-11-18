import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# A3 (Round 6): the analyzer's own layer-axis figures for each telemetry set, drawn by the analyzer's functions unchanged
# (plot_depth_profile, plot_beta_layer, plot_spiral_analysis, plot_beta_g_comparison). Output: analyzer_exports/<set>__<variant>__<plot>.png
import sys, os, glob, types, re, importlib.util
sys.modules['gradio'] = types.ModuleType('gradio')
R6 = f'{_COSPAFS_ROOT}/review/round6'
spec = importlib.util.spec_from_file_location('cpla', f'{R6}/analyzer/cos_pafs_live_analyzer.py'); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
import matplotlib.pyplot as plt
SCR = os.environ.get('SCR', f'{_COSPAFS_CACHE}')
TEL = f'{SCR}/tel/DM'; R4 = f'{_COSPAFS_ROOT}/data/r4_captures'; OUT = f'{R6}/analyzer_exports'; os.makedirs(OUT, exist_ok=True)
SETS = [('ZIT_archive', f'{TEL}/ZIT-CFG-1.0-attn-1024-120/*.csv', [None, 'attn', 'mlp'], None),
        ('ZIB_archive', f'{TEL}/ZIB-CFG-4.0-attn-1024-120/*.csv', [None, 'attn', 'mlp'], None),
        ('MOCHI_CFG4.5', f'{TEL}/MOCHI-CFG-4.5-attn-480-120/*.csv', [None], None),
        ('MOCHI_CFG1', f'{TEL}/MOCHI-CFG-1.0-attn-480-120/*.csv', [None], None),
        ('SD35M', f'{TEL}/SD3.5MED-CFG-4.5-attn-1024-3TE-120/*.csv', [None], None),
        ('SD35L', f'{TEL}/SD3.5LAR-CFG-3.5-attn-1024-3TE-120/*.csv', [None], None),
        ('ZIB_R4_CFG4_cond', f'{R4}/ZIB-attn-TE1-CFG4.0-*/*.csv', ['attn', 'mlp'], 'cond'),
        ('ZIB_R4_CFG1', f'{R4}/ZIB-attn-TE1-CFG1.0-*/*.csv', ['attn', 'mlp'], None)]
PLOTS = {'depth_profile': A.plot_depth_profile, 'beta_layer': A.plot_beta_layer, 'spiral_analysis': A.plot_spiral_analysis}
for lab, pat, subs, br in SETS:
    dm = A.DataManager(); dm.load_paths(','.join(sorted(glob.glob(pat)))); base = dm.get_df()
    if br: base = base[base.branch == br]
    for s in subs:
        df = base if s is None else base[base.submodule == s]; v = 'pooled' if s is None else s
        for name, fn in PLOTS.items():
            fig = fn(df)
            if fig is not None: fig.savefig(f'{OUT}/{lab}__{v}__{name}.png', dpi=110, bbox_inches='tight'); plt.close(fig)
        print(lab, v, flush=True)
