"""R8-7 (clock axis x depth axis), as frozen in R8_PREREGISTRATION.md (SHA acc578ac, 2026-10-09T08:30:59Z).
ZIB MLP archive, per-step layer-axis beta from the analyzer's own plot_beta_layer (analyzer imported unchanged, as in
round6/scripts/r6c_analyzer_beta_nulls.py). Records crossing count and first pos->neg g* per step.
Output: out/R8_7_per_step.csv, out/R8_7_results.json"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import sys, os, glob, types, re, json, importlib.util
sys.modules['gradio'] = types.ModuleType('gradio')
R6 = f'{_COSPAFS_ROOT}/review/round6'; OUT = f'{_COSPAFS_ROOT}/review/r8_correlations/out'
spec = importlib.util.spec_from_file_location('cpla', f'{R6}/analyzer/cos_pafs_live_analyzer.py'); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.stats import mannwhitneyu, fisher_exact
SCR = os.environ.get('SCR', f'{_COSPAFS_CACHE}'); TEL = f'{SCR}/tel/DM'
def cross(df):
    fig = A.plot_beta_layer(df); t = fig.axes[0].get_title(); plt.close(fig)
    return [(float(a), b) for a, b in re.findall(r'g\*=([0-9.eE+-]+) \(([^)]+)\)', t)]
def step_cross(args):
    df, t = args; return t, cross(df[df.token_idx == t])
if __name__ == '__main__':
    dm = A.DataManager(); dm.load_paths(','.join(sorted(glob.glob(f'{TEL}/ZIB-CFG-4.0-attn-1024-120/*.csv')))); df = dm.get_df(); df = df[df.g_micro.notna()]
    df = df[df.submodule == 'mlp']
    cnt = df.groupby('token_idx').gen_idx.nunique(); steps = cnt[cnt >= cnt.max() / 2].index.tolist()
    sig = df.groupby('token_idx').sigma.mean()
    with Pool(4) as p: res = p.map(step_cross, [(df, t) for t in steps])
    S4 = pd.read_csv(f'{R6}/out/S4_topology.csv').set_index('set').loc['archive ZIB-mlp']
    lo, hi = float(S4.break2_sigma), float(S4.break1_sigma)
    rows = []
    for t, cr in res:
        s = float(sig[t]); first = next((x for x, d in cr if d == 'pos→neg'), np.nan)
        rows.append(dict(step=int(t), sigma=round(s, 4), n_cross=len(cr), crossings=';'.join(f'{x:.5f} {d}' for x, d in cr),
                         g_star_first_pos_neg=first, inside_window=bool(lo - 1e-9 <= s <= hi + 1e-9)))
    D = pd.DataFrame(rows).sort_values('step'); D.to_csv(f'{OUT}/R8_7_per_step.csv', index=False)
    a = D[D.inside_window].g_star_first_pos_neg.dropna(); b = D[~D.inside_window].g_star_first_pos_neg.dropna()
    mw = mannwhitneyu(a, b, alternative='two-sided') if len(a) and len(b) else None
    tab = [[int((D[D.inside_window].n_cross == 3).sum()), int((D[D.inside_window].n_cross != 3).sum())],
           [int((D[~D.inside_window].n_cross == 3).sum()), int((D[~D.inside_window].n_cross != 3).sum())]]
    fe = fisher_exact(tab)
    p_mw = float(mw.pvalue) if mw else np.nan
    res = dict(window=[lo, hi], n_inside=int(D.inside_window.sum()), n_outside=int((~D.inside_window).sum()),
               gstar_inside=a.tolist(), gstar_outside=b.tolist(), mw_p=p_mw, fisher_table=tab, fisher_p=float(fe.pvalue),
               verdict='coupled' if (p_mw < 0.05 or fe.pvalue < 0.05) else 'not coupled')
    json.dump(res, open(f'{OUT}/R8_7_results.json', 'w'), indent=1); print(D.to_string()); print(res)
