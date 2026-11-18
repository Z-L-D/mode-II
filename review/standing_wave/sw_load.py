"""Load each channel with the CoS-PAFS Live Analyzer's DataManager (unchanged) and build per-step mean depth profiles.

Writes sw_cache/<channel>.npz with: G[t,L] (mean g_micro), GAP[t,L], ENT[t,L], sigma[t], and per-generation
arrays g_gen[n,t,L], gap_gen, ent_gen plus prompt labels (for prompt bootstraps).
"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import os, sys, glob, types, importlib.util
sys.modules['gradio'] = types.ModuleType('gradio')
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW = os.environ.get('REVIEW', f'{_COSPAFS_ROOT}/review')
ANALYZER = os.environ.get('ANALYZER', f'{REVIEW}/round6/analyzer/cos_pafs_live_analyzer.py')
SCR = os.environ.get('SCR', f'{_COSPAFS_CACHE}')
TEL = os.environ.get('TEL', f'{SCR}/tel/DM')
OUT = os.path.join(HERE, 'sw_cache'); os.makedirs(OUT, exist_ok=True)

spec = importlib.util.spec_from_file_location('cpla', ANALYZER); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)

CH = [('ZIT_mlp', 'ZIT-CFG-1.0-attn-1024-120', 'mlp'),
      ('ZIB_mlp', 'ZIB-CFG-4.0-attn-1024-120', 'mlp'),
      ('ZIT_attn', 'ZIT-CFG-1.0-attn-1024-120', 'attn'),
      ('ZIB_attn', 'ZIB-CFG-4.0-attn-1024-120', 'attn'),
      ('Mochi_attn', 'MOCHI-CFG-4.5-attn-480-120', None),
      ('SD35M_attn', 'SD3.5MED-CFG-4.5-attn-1024-3TE-120', None),
      ('SD35L_attn', 'SD3.5LAR-CFG-3.5-attn-1024-3TE-120', None)]

for name, folder, sub in CH:
    dm = A.DataManager(); dm.load_paths(','.join(sorted(glob.glob(f'{TEL}/{folder}/*.csv'))))
    df = dm.get_df(); df = df[df.g_micro.notna()]
    ptext = df.dropna(subset=['prompt_text']).groupby('gen_idx').prompt_text.first()   # MLP rows carry no prompt text
    if sub: df = df[df.submodule == sub]
    # complete generations only: modal (steps x layers) count
    nl = df.layer.nunique()
    cnt = df.groupby('gen_idx').size(); steps_per = df.groupby('gen_idx').token_idx.nunique()
    T = int(steps_per.mode()[0])
    keep = cnt.index[(steps_per == T) & (cnt == T * nl)]
    df = df[df.gen_idx.isin(keep)]
    gens = np.array(sorted(keep)); steps = np.sort(df.token_idx.unique()); layers = np.sort(df.layer.unique())
    piv = lambda col: df.pivot_table(index=['gen_idx', 'token_idx'], columns='layer', values=col, aggfunc='mean') \
                        .reindex(pd.MultiIndex.from_product([gens, steps])).values.reshape(len(gens), len(steps), len(layers))
    g, gap, ent = piv('g_micro'), piv('spectral_gap'), piv('sv_entropy')
    sig = df.groupby('token_idx').sigma.mean().reindex(steps).values
    prompts = ptext.reindex(gens).astype(str).values
    np.savez(os.path.join(OUT, f'{name}.npz'), g_gen=g, gap_gen=gap, ent_gen=ent, sigma=sig, layers=layers,
             prompts=prompts, G=np.nanmean(g, 0), GAP=np.nanmean(gap, 0), ENT=np.nanmean(ent, 0))
    print(f'{name}: {len(gens)} complete gens, {len(steps)} steps, {len(layers)} layers, '
          f'{len(set(prompts))} prompts, nan={np.isnan(g).sum()}, sigma {sig[0]:.3f}->{sig[-1]:.3f}', flush=True)
