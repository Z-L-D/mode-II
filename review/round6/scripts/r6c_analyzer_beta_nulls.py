import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
# A4 (Round 6): where do the analyzer's three beta_layer crossings come from? All crossing counts are from the analyzer's own plot_beta_layer.
#  (1) per step: the cloud restricted to one denoising step (one sigma), i.e. the layer axis at fixed time
#  (2) Surrogate A (09-15 summary section 8): each layer's g permuted across all (generation, step) rows; keeps each layer's marginal, destroys trajectories
#  (3) Surrogate B (09-15 summary section 9): g permuted across generations within each (layer, step) cell; keeps the mean field g(L, sigma), destroys generation coupling
# Output: out/A4_beta_layer_per_step.csv, out/A4_beta_layer_surrogates.csv
import sys, os, glob, types, re, importlib.util
sys.modules['gradio'] = types.ModuleType('gradio')
R6 = f'{_COSPAFS_ROOT}/review/round6'
spec = importlib.util.spec_from_file_location('cpla', f'{R6}/analyzer/cos_pafs_live_analyzer.py'); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from multiprocessing import Pool
SCR = os.environ.get('SCR', f'{_COSPAFS_CACHE}'); TEL = f'{SCR}/tel/DM'; ND = 30
SETS = [('SD35M archive', f'{TEL}/SD3.5MED-CFG-4.5-attn-1024-3TE-120/*.csv', None),
        ('SD35L archive', f'{TEL}/SD3.5LAR-CFG-3.5-attn-1024-3TE-120/*.csv', None),
        ('ZIT archive', f'{TEL}/ZIT-CFG-1.0-attn-1024-120/*.csv', 'mlp'),
        ('ZIB archive (pooled CFG4)', f'{TEL}/ZIB-CFG-4.0-attn-1024-120/*.csv', 'mlp'),
        ('ZIB archive (pooled CFG4)', f'{TEL}/ZIB-CFG-4.0-attn-1024-120/*.csv', 'attn'),
        ('ZIT archive', f'{TEL}/ZIT-CFG-1.0-attn-1024-120/*.csv', 'attn'),
        ('MOCHI archive CFG4.5', f'{TEL}/MOCHI-CFG-4.5-attn-480-120/*.csv', None)]
def pat(df):
    fig = A.plot_beta_layer(df); t = fig.axes[0].get_title(); plt.close(fig)
    return ' '.join('+-' if d == 'pos→neg' else '-+' for d in re.findall(r'\(([a-z]+→[a-z]+)\)', t))
def surrogate(args):
    df, kind, seed = args; rng = np.random.default_rng(seed); d = df.copy()
    keys = ['layer'] if kind == 'A' else ['layer', 'token_idx']
    d['g_micro'] = d.groupby(keys)['g_micro'].transform(lambda x: rng.permutation(x.values))
    return pat(d)
def step_pat(args):
    df, t = args; return t, pat(df[df.token_idx == t])
if __name__ == '__main__':
    ps, sr = [], []
    for lab, glb, sub in SETS:
        dm = A.DataManager(); dm.load_paths(','.join(sorted(glob.glob(glb)))); df = dm.get_df(); df = df[df.g_micro.notna()]
        if sub: df = df[df.submodule == sub]
        ch = f"{lab} {sub or 'attn'}"
        # main schedule only (steps reached by >= half the generations), so auxiliary long runs do not add sparse steps
        cnt = df.groupby('token_idx').gen_idx.nunique(); steps = cnt[cnt >= cnt.max() / 2].index.tolist()
        sig = df.groupby('token_idx').sigma.mean()
        with Pool(4) as p: res = p.map(step_pat, [(df, t) for t in steps])
        for t, pt in res: ps.append(dict(channel=ch, step=t, sigma=round(float(sig[t]), 4), pattern=pt, n=len(pt.split()) if pt else 0))
        obs = pat(df)
        with Pool(4) as p:
            ra = p.map(surrogate, [(df, 'A', 10 + k) for k in range(ND)]); rb = p.map(surrogate, [(df, 'B', 500 + k) for k in range(ND)])
        P3 = '+- -+ +-'
        stp = [r for r in ps if r['channel'] == ch]
        sr.append(dict(channel=ch, observed=obs, steps=len(stp), per_step_frac_3cross=np.mean([r['pattern'] == P3 for r in stp]),
                       per_step_modal=pd.Series([r['pattern'] for r in stp]).mode()[0],
                       surrA_frac_obs=np.mean([x == obs for x in ra]), surrA_frac_3=np.mean([x == P3 for x in ra]), surrA_modal=pd.Series(ra).mode()[0],
                       surrB_frac_obs=np.mean([x == obs for x in rb]), surrB_frac_3=np.mean([x == P3 for x in rb]), surrB_modal=pd.Series(rb).mode()[0]))
        print(sr[-1], flush=True)
    pd.DataFrame(ps).to_csv(f'{R6}/out/A4_beta_layer_per_step.csv', index=False)
    pd.DataFrame(sr).to_csv(f'{R6}/out/A4_beta_layer_surrogates.csv', index=False)
