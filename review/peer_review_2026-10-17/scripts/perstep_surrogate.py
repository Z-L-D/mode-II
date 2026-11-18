import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# Per-step surrogate for the layer-axis beta sign pattern (peer review 2026-10-17, post hoc).
# Same analyzer call and channel inputs as review/round6/scripts/r6c_analyzer_beta_nulls.py (A4).
# At a single step, Surrogates A and B coincide: each layer's g is permuted across generations,
# keeping every layer's marginal (= the mean field at that sigma) and destroying within-generation coupling.
import sys, os, glob, types, re, importlib.util
sys.modules['gradio'] = types.ModuleType('gradio')
R6 = f'{_COSPAFS_ROOT}/review/round6'
spec = importlib.util.spec_from_file_location('cpla', f'{R6}/analyzer/cos_pafs_live_analyzer.py'); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from multiprocessing import Pool
TEL = sys.argv[1]; OUT = sys.argv[2]; ND = int(sys.argv[3]) if len(sys.argv) > 3 else 40
SETS = [('SD35M archive', 'SD3.5MED-CFG-4.5-attn-1024-3TE-120', None),
        ('SD35L archive', 'SD3.5LAR-CFG-3.5-attn-1024-3TE-120', None),
        ('ZIT archive', 'ZIT-CFG-1.0-attn-1024-120', 'mlp'),
        ('ZIB archive (pooled CFG4)', 'ZIB-CFG-4.0-attn-1024-120', 'mlp'),
        ('ZIB archive (pooled CFG4)', 'ZIB-CFG-4.0-attn-1024-120', 'attn'),
        ('ZIT archive', 'ZIT-CFG-1.0-attn-1024-120', 'attn'),
        ('MOCHI archive CFG4.5', 'MOCHI-CFG-4.5-attn-480-120', None)]
P3 = '+- -+ +-'
def pat(df):
    fig = A.plot_beta_layer(df); t = fig.axes[0].get_title(); plt.close(fig)
    return ' '.join('+-' if d == 'pos→neg' else '-+' for d in re.findall(r'\(([a-z]+→[a-z]+)\)', t))
def job(args):
    d0, seed = args
    if seed < 0: return pat(d0)
    rng = np.random.default_rng(seed); d = d0.copy()
    d['g_micro'] = d.groupby('layer')['g_micro'].transform(lambda x: rng.permutation(x.values))
    return pat(d)
if __name__ == '__main__':
    rows = []
    ONLY = sys.argv[4].split(';') if len(sys.argv) > 4 else None
    for lab, folder, sub in SETS:
        if ONLY and f"{lab} {sub or 'attn'}" not in ONLY: continue
        dm = A.DataManager(); dm.load_paths(','.join(sorted(glob.glob(f'{TEL}/{folder}/*.csv')))); df = dm.get_df(); df = df[df.g_micro.notna()]
        if sub: df = df[df.submodule == sub]
        ch = f"{lab} {sub or 'attn'}"
        cnt = df.groupby('token_idx').gen_idx.nunique(); steps = cnt[cnt >= cnt.max() / 2].index.tolist()
        sig = df.groupby('token_idx').sigma.mean()
        for t in steps:
            d0 = df[df.token_idx == t].copy()  # keep every column: the analyzer keys generations by more than gen_idx
            with Pool(os.cpu_count()) as p: res = p.map(job, [(d0, -1)] + [(d0, int(1000 * t + k)) for k in range(ND)])
            obs, sur = res[0], res[1:]
            rows.append(dict(channel=ch, step=t, sigma=round(float(sig[t]), 4), observed=obs, obs_3cross=obs == P3,
                             surr_frac_3cross=float(np.mean([s == P3 for s in sur])), surr_frac_obs=float(np.mean([s == obs for s in sur])), n_draws=ND))
        R = pd.DataFrame([r for r in rows if r['channel'] == ch])
        print(ch, 'steps', len(R), '3cross', int(R.obs_3cross.sum()), 'of which surrogate 3cross<=5%:', int((R.obs_3cross & (R.surr_frac_3cross <= 0.05)).sum()), flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
