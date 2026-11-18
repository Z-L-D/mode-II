import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# A2 (Round 6): how stable is the analyzer's layer-direction beta(g) crossing count?
# Calls the analyzer's own plot_beta_layer (dashboard "beta_layer" tab) on generation bootstraps.
# A bootstrap draws generations with replacement; each drawn copy gets a new gen_idx so the analyzer treats it as its own generation.
# Output: out/A2_beta_layer_bootstrap.csv
import sys, os, glob, types, re, importlib.util
sys.modules['gradio'] = types.ModuleType('gradio')
R6 = f'{_COSPAFS_ROOT}/review/round6'
spec = importlib.util.spec_from_file_location('cpla', f'{R6}/analyzer/cos_pafs_live_analyzer.py'); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from multiprocessing import Pool
SCR = os.environ.get('SCR', f'{_COSPAFS_CACHE}')
TEL = f'{SCR}/tel/DM'; R4 = f'{_COSPAFS_ROOT}/data/r4_captures'; NB = int(os.environ.get('NB', 100))
SETS = [('ZIT archive', f'{TEL}/ZIT-CFG-1.0-attn-1024-120/*.csv', [None, 'attn', 'mlp'], None),
        ('ZIB archive (pooled CFG4)', f'{TEL}/ZIB-CFG-4.0-attn-1024-120/*.csv', [None, 'attn', 'mlp'], None),
        ('MOCHI archive CFG4.5', f'{TEL}/MOCHI-CFG-4.5-attn-480-120/*.csv', [None], None),
        ('MOCHI archive CFG1', f'{TEL}/MOCHI-CFG-1.0-attn-480-120/*.csv', [None], None),
        ('SD35M archive', f'{TEL}/SD3.5MED-CFG-4.5-attn-1024-3TE-120/*.csv', [None], None),
        ('SD35L archive', f'{TEL}/SD3.5LAR-CFG-3.5-attn-1024-3TE-120/*.csv', [None], None),
        ('ZIB R4 CFG4 cond branch', f'{R4}/ZIB-attn-TE1-CFG4.0-*/*.csv', ['attn', 'mlp'], 'cond'),
        ('ZIB R4 CFG1 cond', f'{R4}/ZIB-attn-TE1-CFG1.0-*/*.csv', ['attn', 'mlp'], None)]
def crossings(df):
    fig = A.plot_beta_layer(df); t = fig.axes[0].get_title(); plt.close(fig)
    return re.findall(r'\(([a-z]+→[a-z]+)\)', t)
def one(args):
    df, seed = args; rng = np.random.default_rng(seed); g = df.gen_idx.unique()
    pick = rng.choice(g, len(g), replace=True)
    parts = [df[df.gen_idx == x].assign(gen_idx=i + 1) for i, x in enumerate(pick)]
    c = crossings(pd.concat(parts, ignore_index=True)); return len(c), ' '.join('+-' if d == 'pos→neg' else '-+' for d in c)
if __name__ == '__main__':
    rows = []
    for lab, pat, subs, br in SETS:
        dm = A.DataManager(); dm.load_paths(','.join(sorted(glob.glob(pat)))); base = dm.get_df()
        base = base[base.g_micro.notna()]
        if br: base = base[base.branch == br]
        for s in subs:
            df = base if s is None else base[base.submodule == s]
            obs = crossings(df)
            with Pool(4) as p: res = p.map(one, [(df, 1000 + k) for k in range(NB)])
            n = np.array([r[0] for r in res]); pat3 = np.mean([r[1] == '+- -+ +-' for r in res])
            rows.append(dict(set=lab, submodule=s or 'as loaded (pooled)', gens=df.gen_idx.nunique(), observed_n=len(obs),
                             observed_pattern=' '.join('+-' if d == 'pos→neg' else '-+' for d in obs),
                             boot=NB, frac_1=float(np.mean(n == 1)), frac_3=float(np.mean(n == 3)), frac_5plus=float(np.mean(n >= 5)),
                             frac_3_plus_minus_plus=float(pat3), modal_n=int(np.bincount(n).argmax())))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f'{R6}/out/A2_beta_layer_bootstrap.csv', index=False)
