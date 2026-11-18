"""A3 POST HOC: Round 6 analyzer replay (A2 generation bootstrap, A4 per-step and Surrogates A/B) on ACE attn and MLP, so that
ACE can sit in Mode-I Table tab:beta with the same columns. Functions copied from round6/scripts/r6c_analyzer_beta_{bootstrap,nulls}.py.
Also saves the analyzer's binned beta(g) curves (attn, attn even-layer subsequence, MLP) for the figure.
Output: out/A2_bootstrap.csv, out/A4_per_step.csv, out/A4_surrogates.csv, out/beta_curves.csv"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import sys, os, glob, types, re, importlib.util
sys.modules['gradio'] = types.ModuleType('gradio')
spec = importlib.util.spec_from_file_location('cpla', f'{_COSPAFS_ROOT}/review/round6/analyzer/cos_pafs_live_analyzer.py'); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from multiprocessing import Pool
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f'{BASE}/out'; NB = 100; ND = 30; P3 = '+- -+ +-'
def pat(df):
    fig = A.plot_beta_layer(df); t = fig.axes[0].get_title(); plt.close(fig)
    return ' '.join('+-' if d == 'pos→neg' else '-+' for d in re.findall(r'\(([a-z]+→[a-z]+)\)', t))
def one(args):
    df, seed = args; rng = np.random.default_rng(seed); g = df.gen_idx.unique(); pick = rng.choice(g, len(g), replace=True)
    p = pat(pd.concat([df[df.gen_idx == x].assign(gen_idx=i + 1) for i, x in enumerate(pick)], ignore_index=True)); return len(p.split()) if p else 0, p
def surrogate(args):
    df, kind, seed = args; rng = np.random.default_rng(seed); d = df.copy(); keys = ['layer'] if kind == 'A' else ['layer', 'token_idx']
    d['g_micro'] = d.groupby(keys)['g_micro'].transform(lambda x: rng.permutation(x.values)); return pat(d)
def step_pat(args):
    df, t = args; return t, pat(df[df.token_idx == t])
def curve_xy(df):
    fig = A.plot_beta_layer(df); ax = fig.axes[0]; line = [l for l in ax.get_lines() if l.get_linewidth() == 2][0]
    x, m = (np.asarray(v, float) for v in line.get_data()); plt.close(fig); return x, m
if __name__ == '__main__':
    dm = A.DataManager(); dm.load_paths(','.join(sorted(glob.glob(f'{BASE}/data/*.csv')))); base = dm.get_df(); base = base[base.g_micro.notna()]
    sig = base.groupby('token_idx').sigma.mean(); bo, ps, sr, cv = [], [], [], []
    for sub in ('attn', 'mlp'):
        df = base[base.submodule == sub]; obs = pat(df); ch = f'ACE {sub}'
        with Pool(4) as p: res = p.map(one, [(df, 1000 + k) for k in range(NB)])
        n = np.array([r[0] for r in res])
        bo.append(dict(channel=ch, gens=df.gen_idx.nunique(), observed_n=len(obs.split()) if obs else 0, observed_pattern=obs, boot=NB, frac_1=float(np.mean(n == 1)),
                       frac_3=float(np.mean(n == 3)), frac_5plus=float(np.mean(n >= 5)), frac_3_plus_minus_plus=float(np.mean([r[1] == P3 for r in res])), modal_n=int(np.bincount(n).argmax())))
        with Pool(4) as p: st = p.map(step_pat, [(df, t) for t in sorted(df.token_idx.unique())])
        for t, q in st: ps.append(dict(channel=ch, step=t, sigma=round(float(sig[t]), 4), pattern=q, n=len(q.split()) if q else 0))
        with Pool(4) as p:
            ra = p.map(surrogate, [(df, 'A', 10 + k) for k in range(ND)]); rb = p.map(surrogate, [(df, 'B', 500 + k) for k in range(ND)])
        stp = [r for r in ps if r['channel'] == ch]
        sr.append(dict(channel=ch, observed=obs, steps=len(stp), per_step_frac_3cross=np.mean([r['pattern'] == P3 for r in stp]), per_step_modal=pd.Series([r['pattern'] for r in stp]).mode()[0],
                       surrA_frac_obs=np.mean([x == obs for x in ra]), surrA_frac_3=np.mean([x == P3 for x in ra]), surrA_modal=pd.Series(ra).mode()[0],
                       surrB_frac_obs=np.mean([x == obs for x in rb]), surrB_frac_3=np.mean([x == P3 for x in rb]), surrB_modal=pd.Series(rb).mode()[0]))
        x, m = curve_xy(df); cv += [dict(series=f'ACE {sub}', g=a, beta=b) for a, b in zip(x, m)]
        print(bo[-1], sr[-1], flush=True)
    ev = base[(base.submodule == 'attn') & (base.layer % 2 == 0)].copy(); ev['layer'] = ev.layer // 2
    x, m = curve_xy(ev); cv += [dict(series='ACE attn, even layers only', g=a, beta=b) for a, b in zip(x, m)]
    pd.DataFrame(bo).to_csv(f'{OUT}/A2_bootstrap.csv', index=False); pd.DataFrame(ps).to_csv(f'{OUT}/A4_per_step.csv', index=False)
    pd.DataFrame(sr).to_csv(f'{OUT}/A4_surrogates.csv', index=False); pd.DataFrame(cv).to_csv(f'{OUT}/beta_curves.csv', index=False)
    print(pd.DataFrame(ps).to_string())
